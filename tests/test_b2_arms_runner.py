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


@pytest.mark.parametrize("arm", ["B1", "mixed", "modal"])
def test_runner_dry_run_writes_nothing_and_execute_is_refused(arm):
    before = sorted(p for p in RESULTS_B2.rglob("*"))
    spec = PINNRP / "configs" / "batch2" / "B2-E01_mechanism_attribution.yaml"
    r = subprocess.run([sys.executable, str(PINNRP / "experiments" / "run_batch2.py"), "--spec", str(spec), "--arm", arm],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "DRY RUN" in r.stdout
    if arm == "B1":
        assert "f78aa7f1da" in r.stdout                         # = Batch-1 Z4 (5k) run key
    r = subprocess.run([sys.executable, str(PINNRP / "experiments" / "run_batch2.py"), "--spec", str(spec), "--arm", arm, "--execute"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "TrainingNotApproved" in r.stderr
    assert sorted(p for p in RESULTS_B2.rglob("*")) == before
