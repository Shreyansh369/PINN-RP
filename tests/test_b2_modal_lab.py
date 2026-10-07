"""B2-E02 modal-lab controllers: frozen-control identity, same initialisation, compute accounting,
causal weights. Software tests: <= 12 optimizer steps on 2x12 networks in temp dirs."""
import math

import pytest
import torch

from physref.arms import ArmTrainer
from physref.frozen import to_experiment_config
from physref.modal_lab import (AdamLBFGSModalTrainer, CausalModalTrainer, build_trainer, causal_epsilon)


def tiny(mb=4, n=8):
    cfg, _ = to_experiment_config("batch1_hard_tanh2")
    cfg.model.depth, cfg.model.width, cfg.model.m_fourier = 2, 12, 8
    cfg.sampler.n_per_term, cfg.sampler.mini_batch = n, mb
    cfg.train.log_every = 1
    return cfg


def test_all_controllers_start_from_identical_weights(tmp_path):
    ref = ArmTrainer(tiny(), "modal", tmp_path / "a", "T", strict=False).model.state_dict()
    for c in ("adam_lbfgs", "adam_causal"):
        m = build_trainer(c, tiny(), tmp_path / c, "T", strict=False, **({"switch_evals": 8, "total_evals": 40,
                          "n_lbfgs": 8, "max_ls": 2, "snapshot_evals": 8} if c == "adam_lbfgs" else {})).model.state_dict()
        for k in ref:
            assert torch.equal(ref[k], m[k]), (c, k)


def test_lbfgs_stage1_is_bitwise_the_control_and_budget_is_respected(tmp_path):
    ctrl = ArmTrainer(tiny(), "modal", tmp_path / "c", "T", strict=False)
    ctrl.run(3)
    two = AdamLBFGSModalTrainer(tiny(), tmp_path / "l", "T", strict=False, switch_evals=12, total_evals=12 + 8 * 20,
                                n_lbfgs=8, max_ls=3, snapshot_evals=8 * 5)
    # stage 1 alone (3 Adam steps x 4 points = 12 evaluations) must equal the control after 3 steps
    stage1 = AdamLBFGSModalTrainer(tiny(), tmp_path / "s", "T", strict=False, switch_evals=12, total_evals=12,
                                   n_lbfgs=8, max_ls=3, snapshot_evals=8)
    ArmTrainer.run(stage1, 3)
    for k, v in ctrl.model.state_dict().items():
        assert torch.equal(v, stage1.model.state_dict()[k])
    assert two.run() == "completed"
    a = two.acc
    assert a["switch_step"] == 3 and a["switch_pde_evaluations"] == 12
    assert a["pde_evaluations"] <= 12 + 8 * 20 and a["pde_evaluations"] == 12 + 8 * a["lbfgs_closure_evals"]
    assert a["forward_passes"] == a["backward_passes"] == 3 + a["lbfgs_closure_evals"]
    assert a["lbfgs_stop_reason"] in ("budget", "lbfgs_tolerance")
    assert math.isfinite(two.history[-1]["loss"])


def test_causal_epsilon_and_weights(tmp_path):
    eps = causal_epsilon(129.373, 0.08, 16)
    L_tol = 0.5 * (0.01 * 129.373 ** 2 * 0.08) ** 2
    assert abs(eps * 16 * L_tol - 1) < 1e-12
    tr = CausalModalTrainer(tiny(mb=16, n=16), tmp_path, "T", strict=False, n_bins=4)
    batch = tr.sampler.next_batch()
    total, parts = tr.loss(batch)
    w = tr.last_w
    assert torch.all(w[1:] <= w[:-1] + 1e-12) and abs(float(w[0]) - 1) < 1e-12      # causal: monotone, first bin 1
    tr.eps = 0.0                                                                     # eps -> 0: plain bin-mean loss
    total0, _ = tr.loss(batch)
    assert torch.all(tr.last_w == 1)
    assert tr.run(3) == "completed" and tr.acc["pde_evaluations"] == 48


def test_fullfield_lbfgs_stage1_is_b1_and_budget_respected(tmp_path):
    """Transfer arm: stage 1 = the frozen Batch-1 Trainer (bitwise); L-BFGS never exceeds the budget."""
    from beampinn.training.trainer import Trainer
    from physref.fullfield_lab import AdamLBFGSFullFieldTrainer
    cfg = tiny()
    cfg.train.max_steps, cfg.train.eval_every, cfg.train.log_every = 6, 1000, 1000
    ref = Trainer(cfg, root=tmp_path / "b1")
    ref.run(final_eval=False, stop_at_step=3)
    tr = AdamLBFGSFullFieldTrainer(cfg, tmp_path / "l", switch_evals=12, total_evals=12 + 8 * 15, n_lbfgs=8,
                                   max_ls=3, snapshot_evals=40)
    st0 = tr.run(resume=False, final_eval=False, stop_at_step=3)
    for k, v in ref.model.state_dict().items():
        assert torch.equal(v, tr.model.state_dict()[k]), k
    tr2 = AdamLBFGSFullFieldTrainer(cfg, tmp_path / "l2", switch_evals=12, total_evals=12 + 8 * 15, n_lbfgs=8,
                                    max_ls=3, snapshot_evals=40)
    assert tr2.run_two_stage() == "completed"
    a = tr2.acc
    assert a["switch_pde_evaluations"] == 12 and a["pde_evaluations"] <= 12 + 8 * 15
    assert a["pde_evaluations"] == 12 + 8 * a["lbfgs_closure_evals"]
    assert a["forward_passes"] == 3 + a["lbfgs_closure_evals"] == a["backward_passes"]
    assert (tmp_path / "l2" / "logs").exists()
