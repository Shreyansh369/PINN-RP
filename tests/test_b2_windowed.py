"""Software tests of the B2-E05 windowed trainer (docs/hypotheses/B2-E05.md). Tiny networks and a few optimizer
steps inside pytest temp dirs - not experiments, nothing kept - except the W1000 == B1 check, which runs 30 steps
of the full frozen B1 configuration."""
import math
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from beampinn.models.networks import SpatioTemporalFourierPINN
from beampinn.physics.beam import mode_shape_raw
from beampinn.training.trainer import Trainer
from physref.frozen import to_experiment_config
from physref.paths import PINNRP, RESULTS_B2
from physref.windowed import (FFModeBasis, HandoffFF, N_MODES, RESULT_COLUMNS, WindowedTrainer, gauss_legendre,
                              project, results_row)

SPEC = PINNRP / "configs" / "batch2" / "B2-E05_window_law.yaml"
L = 2.75


def tiny(max_steps=8):
    cfg, _ = to_experiment_config("batch1_hard_tanh2")
    cfg.model.depth, cfg.model.width, cfg.model.m_fourier = 2, 12, 8
    cfg.sampler.n_per_term, cfg.sampler.mini_batch = 8, 4          # 2 steps per epoch
    cfg.train.log_every = 1
    cfg.train.max_steps = max_steps
    return cfg


def _grad(y, v):
    return torch.autograd.grad(y, v, torch.ones_like(y), create_graph=True)[0]


# ------------------------------------------------------------------ mode basis
def test_mode_basis_matches_beampinn_and_is_orthogonal():
    B = FFModeBasis(L).double()
    x = np.linspace(0.0, L, 1001)
    P = B(torch.tensor(x).reshape(-1, 1)).numpy()
    for n in range(3):                                   # low modes: the textbook form is still accurate
        U = mode_shape_raw("fixed-fixed", B.betas[n], L, x)
        assert np.allclose(P[:, n], U / U[np.argmax(np.abs(U))], atol=1e-9)
    assert np.allclose(np.abs(P).max(0), 1.0, atol=1e-4)
    xg, wg = gauss_legendre(64, L)
    Pg = B.numpy(xg)
    G = (Pg * wg) @ Pg.T
    assert np.abs(G - np.diag(np.diag(G))).max() < 1e-10 * np.diag(G).min()


def test_mode_basis_fourth_derivative_and_bcs():
    B = FFModeBasis(L).double()
    x = torch.linspace(0.0, L, 101, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    for n in range(N_MODES):
        p = B(x)[:, n:n + 1]
        d1 = _grad(p, x); d4 = _grad(_grad(_grad(d1, x), x), x)
        assert torch.allclose(d4, B.betas[n] ** 4 * p, atol=1e-8 * B.betas[n] ** 4)
        ends = torch.tensor([[0.0], [L]], dtype=torch.float64, requires_grad=True)
        pe = B(ends)[:, n:n + 1]
        assert pe.abs().max() < 1e-12 and (_grad(pe, ends) * L).abs().max() < 1e-10


# ------------------------------------------------------------------ ansatz: IC / BC exact
@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-12), (torch.float32, 2e-6)])
def test_handoff_ansatz_ic_bc_exact(dtype, tol):
    cfg = tiny()
    net = SpatioTemporalFourierPINN(cfg.model, L, 0.1, 1234).to(dtype)
    with torch.no_grad():
        net.head.weight.mul_(50.0)                        # make N clearly non-zero
    rng = np.random.default_rng(0)
    a, b = 0.01 * rng.standard_normal(N_MODES), 1.0 * rng.standard_normal(N_MODES)
    m = HandoffFF(net, FFModeBasis(L).to(dtype), a, b, L, 0.08, 129.37)
    B = FFModeBasis(L).double()
    x = torch.linspace(0.0, L, 57, dtype=dtype).reshape(-1, 1)
    t0 = torch.zeros_like(x).requires_grad_(True)
    u = m(x, t0)
    ut = torch.autograd.grad(u, t0, torch.ones_like(u))[0]
    Pa = (B(x.double()) @ torch.tensor(a)).reshape(-1, 1)
    Pb = (B(x.double()) @ torch.tensor(b)).reshape(-1, 1)
    assert (u.double() - Pa).abs().max() < tol * 0.08 * 10           # IC displacement = a(x)
    assert (ut.double() - Pb).abs().max() < tol * 10                 # IC velocity     = b(x)
    t = torch.linspace(0.0, 0.1, 41, dtype=dtype).reshape(-1, 1)
    for xe in (0.0, L):                                              # BCs u = u_x = 0 for all t
        xx = torch.full_like(t, xe).requires_grad_(True)
        ue = m(xx, t)
        assert ue.abs().max() < tol * 0.08 * 10
        assert (_grad(ue, xx) * L).abs().max() < tol * 0.08 * 50


# ------------------------------------------------------------------ hand-off continuity
def _check_continuity(tr, raw_exact):
    """u_k(x,0) and d/dt u_k(x,0) equal the 8-mode projection of the previous window's end state (exact), and the
    jump to the raw end state equals the logged projection remainder."""
    xg, wg = gauss_legendre(64, L)
    Pg = FFModeBasis(L).numpy(xg)
    for k in range(1, tr.N):
        prev, cur = tr.windows[k - 1], tr.windows[k]
        F, G = tr._end_state(prev)
        x = torch.tensor(xg, dtype=torch.float32).reshape(-1, 1)
        t = torch.zeros_like(x).requires_grad_(True)
        u = cur(x, t)
        ut = torch.autograd.grad(u, t, torch.ones_like(u))[0]
        u, ut = u.detach().double().numpy().ravel(), ut.detach().double().numpy().ravel()
        a, eu = project(F, Pg, wg)
        b, ev = project(G, Pg, wg)
        assert np.abs(u - Pg.T @ a).max() <= 1e-6 * np.abs(F).max()
        assert np.abs(ut - Pg.T @ b).max() <= 1e-6 * np.abs(G).max()
        log = tr.window_log[k - 1]
        assert math.isclose(log["proj_err_u"], eu, rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(log["proj_err_ut"], ev, rel_tol=1e-9, abs_tol=1e-12)
        if raw_exact:                                                # end state in the span: jump vanishes
            assert np.abs(u - F).max() <= 1e-6 * np.abs(F).max()
            assert np.abs(ut - G).max() <= 1e-6 * np.abs(G).max()
            assert eu < 1e-6 and ev < 1e-6


def test_handoff_continuity_every_window(tmp_path):
    tr = WindowedTrainer(tiny(8), tmp_path, 0.25)                    # 4 windows x 2 steps
    assert tr.run(final_eval=False) == "completed" and len(tr.windows) == 4
    _check_continuity(tr, raw_exact=False)
    assert all("proj_err_u" in w for w in tr.window_log[:-1]) and "proj_err_u" not in tr.window_log[-1]


def test_handoff_continuity_is_exact_when_end_state_is_in_mode_span(tmp_path):
    """A window with N == 0 has the field a(x) + tau b(x), which lies in the 8-mode span. Handing it off through
    WindowedTrainer.handoff makes u and u_t continuous to round-off (zero projection remainder)."""
    tr = WindowedTrainer(tiny(8), tmp_path, 0.25)
    net = SpatioTemporalFourierPINN(tr.cfg.model, L, 0.25, 7)
    with torch.no_grad():
        net.head.weight.zero_(); net.head.bias.zero_()
    rng = np.random.default_rng(1)
    prev = HandoffFF(net, tr.basis, 0.02 * rng.standard_normal(N_MODES), 2.0 * rng.standard_normal(N_MODES),
                     L, 0.08, tr.omega_1)
    a, b, err = tr.handoff(prev)
    assert err["proj_err_u"] < 1e-6 and err["proj_err_ut"] < 1e-6
    assert err["jump_u_max_over_A0"] < 1e-5 and err["jump_ut_max_over_wA0"] < 1e-5
    cur = HandoffFF(SpatioTemporalFourierPINN(tr.cfg.model, L, 0.25, 8), tr.basis, a, b, L, 0.08, tr.omega_1)
    x = torch.linspace(0.0, L, 31).reshape(-1, 1)
    te = torch.full_like(x, 0.25).requires_grad_(True)
    t0 = torch.zeros_like(x).requires_grad_(True)
    ue, us = prev(x, te), cur(x, t0)
    ve = torch.autograd.grad(ue, te, torch.ones_like(ue))[0]
    vs = torch.autograd.grad(us, t0, torch.ones_like(us))[0]
    assert (ue - us).abs().max() <= 1e-5 * ue.abs().max()
    assert (ve - vs).abs().max() <= 1e-5 * ve.abs().max()


def test_stitched_field_evaluates_each_window_in_local_time(tmp_path):
    tr = WindowedTrainer(tiny(8), tmp_path, 0.25)
    tr.run(final_eval=False)
    f = tr.stitched()
    x = torch.rand(40, 1) * L
    t = torch.rand(40, 1)
    t[:4] = torch.tensor([[0.0], [0.25], [0.5], [1.0]])
    out = f(x, t)
    for i in range(40):
        k = min(int(math.floor(float(t[i]) / 0.25)), 3)
        ref = tr.windows[k](x[i:i + 1], t[i:i + 1] - k * 0.25)
        assert torch.allclose(out[i:i + 1], ref, rtol=1e-5, atol=1e-9)   # batched vs single: fp32 round-off


# ------------------------------------------------------------------ W1000 == B1
def test_w1000_is_bit_identical_to_b1_trainer(tmp_path):
    """Full frozen B1 configuration, 30 steps: same weights, same loss history (bit-exact)."""
    cfg, _ = to_experiment_config("batch1_hard_tanh2")
    cfg.train.max_steps = 30
    b1 = Trainer(cfg, tmp_path / "b1")
    assert b1.run(final_eval=False) == "completed"
    cfg2, _ = to_experiment_config("batch1_hard_tanh2")
    cfg2.train.max_steps = 30
    w = WindowedTrainer(cfg2, tmp_path / "w", 1.0)
    assert w.run(final_eval=False) == "completed"
    sa, sb = b1.model.state_dict(), w.model.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    la = [(r["step"], r["loss"]) for r in b1.history if "loss" in r]
    lb = {r["step"]: r["loss"] for r in w.history if "loss" in r}
    assert la and all(lb[s] == v for s, v in la if s in lb)
    assert w.acc["pde_evaluations"] == b1.acc["pde_evaluations"] == 30 * 128


# ------------------------------------------------------------------ budget / config
def test_spec_arms_split_640k_equally(tmp_path):
    spec = yaml.safe_load(open(SPEC))
    expected = {"W1000": (1, 5000), "W250": (4, 1250), "W100": (10, 500), "W050": (20, 250)}
    assert set(spec["arms"]) == set(expected)
    for arm, (n, spw) in expected.items():
        cfg, _ = to_experiment_config(spec["base_frozen"])
        cfg.train.max_steps = spec["max_steps"]
        assert (cfg.precision, cfg.device, cfg.threads) == ("float32", "cpu", 1)
        tr = WindowedTrainer(cfg, tmp_path / arm, spec["arms"][arm]["T_w"])
        assert (tr.N, tr.steps_per_window) == (n, spw)
        assert tr.N * tr.steps_per_window * cfg.sampler.mini_batch == 640_000
        assert tr.sampler.T == pytest.approx(spec["arms"][arm]["T_w"])
        assert float(tr.net.tt.scale) == pytest.approx(spec["arms"][arm]["T_w"] / math.sqrt(12.0))


def test_bad_window_lengths_are_refused(tmp_path):
    with pytest.raises(ValueError):
        WindowedTrainer(tiny(8), tmp_path / "a", 0.3)                # does not divide T
    with pytest.raises(ValueError):
        WindowedTrainer(tiny(8), tmp_path / "b", 0.125)              # 8 windows x 1 step: not whole epochs


def test_budget_accounting_tiny(tmp_path):
    tr = WindowedTrainer(tiny(8), tmp_path, 0.25)
    tr.run(final_eval=False)
    assert tr.acc["optimizer_steps"] == 8 and tr.acc["pde_evaluations"] == 32
    assert [w["step_end"] for w in tr.window_log] == [2, 4, 6, 8]


def test_resume_after_window_is_bit_exact(tmp_path):
    a = WindowedTrainer(tiny(8), tmp_path / "a", 0.25); a.run(final_eval=False)
    b = WindowedTrainer(tiny(8), tmp_path / "b", 0.25)
    assert b.run(final_eval=False, stop_after_window=2) == "interrupted"
    b2 = WindowedTrainer(tiny(8), tmp_path / "b", 0.25)
    assert b2.run(final_eval=False) == "completed" and b2.window == 4
    for wa, wb in zip(a.windows, b2.windows):
        for (ka, va), (kb, vb) in zip(wa.state_dict().items(), wb.state_dict().items()):
            assert ka == kb and torch.equal(va, vb), ka


def test_finalise_metrics_and_results_row(tmp_path):
    tr = WindowedTrainer(tiny(8), tmp_path, 0.25)
    assert tr.run() == "completed"
    import json
    m = json.load(open(tr.paths["logs"] / "metrics.json"))
    for k in ("amp_ratio_t0.1", "amp_ratio_t0.25", "amp_ratio_t0.5", "amp_ratio_t0.75", "amp_ratio_t1",
              "persistence_cycles", "fit_lam", "decay_error", "L2_exact", "R_k_mean", "R_k_max"):
        assert np.isfinite(m[k]), k
    assert len(m["R_k"]) == 4 and all(r > 0 for r in m["R_k"])
    assert m["decay_error"] == pytest.approx(m["fit_lam"] - 3.54)
    row = results_row("W250", m)
    assert list(row) == RESULT_COLUMNS and row["N_windows"] == 4 and row["T_w"] == 0.25
    assert (tmp_path / "logs" / tr.run_id / "windows.csv").exists()


# ------------------------------------------------------------------ runner
@pytest.mark.parametrize("arm", ["W1000", "W250", "W100", "W050"])
def test_runner_dry_run_writes_nothing(arm):
    before = sorted(p for p in RESULTS_B2.rglob("*"))
    r = subprocess.run([sys.executable, str(PINNRP / "experiments" / "run_batch2.py"), "--spec", str(SPEC),
                        "--arm", arm], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert f"B2-E05-{arm}-s1234" in r.stdout and "640,000 PDE evaluations" in r.stdout
    assert "float32, 1 thread(s)" in r.stdout and "DRY RUN" in r.stdout
    assert sorted(p for p in RESULTS_B2.rglob("*")) == before
