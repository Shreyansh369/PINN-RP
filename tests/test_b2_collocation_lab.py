"""B2-E03 collocation lab: identical initialisation (checksum), stage-1 identity with B1, shared set0,
resampling / reservoir semantics, exact budget accounting. Software tests on 2x12 nets in temp dirs."""
import torch

from beampinn.training.trainer import Trainer
from physref.collocation_lab import ARMS, CollocationLabTrainer, state_checksum
from physref.frozen import to_experiment_config


def tiny():
    cfg, _ = to_experiment_config("batch1_hard_tanh2")
    cfg.model.depth, cfg.model.width, cfg.model.m_fourier = 2, 12, 8
    cfg.sampler.n_per_term, cfg.sampler.mini_batch = 8, 4
    cfg.train.max_steps, cfg.train.eval_every, cfg.train.log_every = 6, 1000, 1000
    return cfg


KW = dict(switch_evals=12, total_evals=12 + 8 * 40, n_active=8, switch_every=5, max_ls=3, snapshot_evals=80,
          adam_full_steps=36)


def test_identical_init_checksum_and_stage1_equals_b1(tmp_path):
    b1 = Trainer(tiny(), root=tmp_path / "b1")
    ck = state_checksum(b1.model)
    b1.run(final_eval=False, stop_at_step=3)
    for arm in ARMS:
        tr = CollocationLabTrainer(tiny(), tmp_path / arm, arm, **KW)
        assert tr.init_checksum == ck, arm
        tr.run(resume=False, final_eval=False, stop_at_step=3)
        for k, v in b1.model.state_dict().items():
            assert torch.equal(v, tr.model.state_dict()[k]), (arm, k)


def test_set0_shared_and_semantics(tmp_path):
    out = {}
    for arm in ARMS:
        tr = CollocationLabTrainer(tiny(), tmp_path / arm, arm, **KW)
        assert tr.run_two_stage() == "completed"
        out[arm] = tr
        a = tr.acc
        assert a["pde_evaluations"] <= 12 + 8 * 40
        assert a["pde_evaluations"] == 12 + 8 * a["stage2_closure_evals"]
        assert a["forward_passes"] == a["backward_passes"] == 3 + a["stage2_closure_evals"]
    s0 = out["LBFGS-F"].coll["sets"][0]
    for arm in ARMS:
        x, t = out[arm].coll["sets"][0]
        assert torch.equal(x, s0[0]) and torch.equal(t, s0[1]), arm          # same first set in every arm
    assert len(out["LBFGS-F"].coll["sets"]) == 1 and len(out["ADAM-FULL"].coll["sets"]) == 1
    assert len(out["LBFGS-4X"].coll["sets"]) == 4
    r = out["LBFGS-R"]
    assert len(r.coll["sets"]) == 1 + len(r.coll["events"]) - 1 and len(r.coll["sets"]) > 1
    for e in r.coll["events"][1:]:
        assert e["closure_index"] % 5 >= 0 and e["closure_index"] >= 5          # switches only at >= every 5 closures
    ev4 = [e["set"] for e in out["LBFGS-4X"].coll["events"]]
    assert ev4[:5] == [0, 1, 2, 3, 0][:len(ev4[:5])]                           # cyclic reservoir order
    assert out["ADAM-FULL"].acc["adam_full_steps"] == 36


def test_e03_evaluator_training_sets_and_independence(tmp_path):
    """Evaluator semantics (docs/hypotheses/B2-E03.md section 7) on tiny lab runs."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import analyze_b2_e03 as E
    for arm in ("LBFGS-R", "LBFGS-4X", "LBFGS-F"):
        tr = CollocationLabTrainer(tiny(), tmp_path / arm, arm, **KW)
        assert tr.run_two_stage() == "completed"
        ck = next((tmp_path / arm / "checkpoints").iterdir())
        fin = torch.load(ck / "final.pt", weights_only=False)
        fcoll = fin["collocation"]
        (xs, ts), sec = E.training_sets(fin, fcoll, is_switch=False)
        if arm == "LBFGS-4X":
            assert len(xs) == 4 * 8 and sec is not None
        elif arm == "LBFGS-R":
            assert torch.equal(xs, fcoll["sets"][fcoll["active_set"]][0])
        else:
            assert torch.equal(xs, fcoll["sets"][0][0])
        sw = torch.load(ck / "switch.pt", weights_only=False)
        (x0, t0), _ = E.training_sets(sw, fcoll, is_switch=True)
        assert torch.equal(x0[:8], fcoll["sets"][0][0])                      # switch state uses the stage-2 set
        E.check_independent(xs, ts)                                           # random points never hit grid nodes
        row, _, _ = E.eval_point(ck / "final.pt", fcoll)
        assert row["R_train"] > 0 and row["R_dense"] > 0 and abs(row["G"] - row["R_dense"] / row["R_train"]) < 1e-12
    import pytest
    with pytest.raises(AssertionError):                                       # contamination is caught
        E.check_independent(torch.tensor([[0.0]], dtype=torch.float64), torch.tensor([[0.0]], dtype=torch.float64))
