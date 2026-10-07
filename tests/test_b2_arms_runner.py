"""Software tests of the arm trainer and runner (2-6 optimizer steps on 2x12 networks inside
pytest temp dirs - not experiments, nothing kept), checkpoint/resume equivalence, determinism."""
import subprocess
import sys

import pytest
import torch

from physref.arms import ArmTrainer
from physref.frozen import to_experiment_config
from physref.paths import PINNRP, RESULTS_B2


def tiny():
    cfg, _ = to_experiment_config("batch1_hard_tanh2")
    cfg.model.depth, cfg.model.width, cfg.model.m_fourier = 2, 12, 8
    cfg.sampler.n_per_term, cfg.sampler.mini_batch = 8, 4
    cfg.train.log_every = 1
    return cfg


@pytest.mark.parametrize("arm", ["mixed", "modal"])
def test_arm_runs_and_accounts(tmp_path, arm):
    tr = ArmTrainer(tiny(), arm, tmp_path, f"T-{arm}", strict=False)
    assert tr.run(3) == "completed"
    assert tr.acc["optimizer_steps"] == 3 and tr.acc["pde_evaluations"] == 12
    assert (tmp_path / f"T-{arm}" / "checkpoints" / "latest.pt").exists()


@pytest.mark.parametrize("arm", ["mixed", "modal"])
def test_arm_resume_is_exact_and_deterministic(tmp_path, arm):
    a = ArmTrainer(tiny(), arm, tmp_path / "a", "T", strict=False); a.run(6)
    b = ArmTrainer(tiny(), arm, tmp_path / "b", "T", strict=False)
    assert b.run(6, stop_at_step=3) == "interrupted" and b.step == 3
    b2 = ArmTrainer(tiny(), arm, tmp_path / "b", "T", strict=False)
    assert b2.run(6) == "completed" and b2.step == 6
    for p, q in zip(a.model.state_dict().values(), b2.model.state_dict().values()):
        assert torch.allclose(p, q, rtol=0, atol=1e-6)
    c = ArmTrainer(tiny(), arm, tmp_path / "c", "T", strict=False); c.run(6)
    for p, q in zip(a.model.state_dict().values(), c.model.state_dict().values()):
        assert torch.equal(p, q)                                  # same seed -> identical


def test_arm_requires_b1_base(tmp_path):
    cfg, _ = to_experiment_config("baseline_fourier_ntk")
    with pytest.raises(ValueError):
        ArmTrainer(cfg, "mixed", tmp_path, "T", strict=False)


def test_arm_strict_mode_rejects_temp_dirs(tmp_path):
    from physref.safety import RootWriteError
    with pytest.raises(RootWriteError):
        ArmTrainer(tiny(), "modal", tmp_path, "T", strict=True)


@pytest.mark.parametrize("arm", ["B1", "B1_fp64", "mixed", "modal"])
def test_runner_dry_run_writes_nothing(arm):
    before = sorted(p for p in RESULTS_B2.rglob("*"))
    spec = PINNRP / "configs" / "batch2" / "B2-E01_mechanism_attribution.yaml"
    r = subprocess.run([sys.executable, str(PINNRP / "experiments" / "run_batch2.py"), "--spec", str(spec), "--arm", arm],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "DRY RUN" in r.stdout
    if arm == "B1":
        assert "f78aa7f1da" in r.stdout                         # = Batch-1 Z4 (5k) run key
    assert sorted(p for p in RESULTS_B2.rglob("*")) == before
    if arm == "B1_fp64":
        assert "float64" in r.stdout


def test_modal_net_init_identical_to_b1_temporal_branch():
    """Modal arm control: same temporal Fourier draws and identical trunk/head initialisation as B1."""
    from beampinn.models.networks import SpatioTemporalFourierPINN
    from physref.formulations.modal import TemporalFourierNet
    cfg, _ = to_experiment_config("batch1_hard_tanh2")
    m = cfg.model
    torch.set_default_dtype(torch.float32)
    b1 = SpatioTemporalFourierPINN(m, 2.75, 1.0, cfg.seed)
    q = TemporalFourierNet(1.0, m.m_fourier, m.sigma_t, m.depth, m.width, cfg.seed, m.two_pi, burn_spatial_sigmas=m.sigma_x)
    for a, b in zip(b1.enc_t, q.enc):
        assert torch.equal(a.B, b.B)
    for (ka, a), (kb, b) in zip(list(b1.trunk.state_dict().items()) + list(b1.head.state_dict().items()),
                                list(q.trunk.state_dict().items()) + list(q.head.state_dict().items())):
        assert torch.equal(a, b), ka


def test_fp64_control_starts_from_identical_fp32_weights(tmp_path):
    """Precision control: FP64 model = FP32 initial state cast to float64 (only precision differs)."""
    from beampinn.training.trainer import Trainer
    from physref.arms import fp32_initial_state, load_cast_state
    c = tiny(); c.loss.hard_constraints = "ff_tanh2"
    c32 = c; t32 = Trainer(c32, root=tmp_path / "a")
    ref = {k: v.clone() for k, v in t32.model.state_dict().items()}
    import copy
    c64 = copy.deepcopy(c); c64.precision = "float64"
    sd = fp32_initial_state(c64)
    t64 = Trainer(c64, root=tmp_path / "b")
    load_cast_state(t64.model, sd)
    for k, v in t64.model.state_dict().items():
        assert v.dtype == torch.float64 or not v.is_floating_point()
        assert torch.equal(v, ref[k].to(v.dtype)), k
    torch.set_default_dtype(torch.float32)


def test_execute_refused_for_unapproved_ids():
    from physref.gate import TrainingNotApproved, require_approval
    with pytest.raises(TrainingNotApproved):
        require_approval("B2-E01-mixed-s1237")            # only seeds 1234-1236 are approved
