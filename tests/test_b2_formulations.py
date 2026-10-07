"""PDE equivalence, derivative correctness and IC/BC correctness of the Batch-2 formulations."""
import math

import numpy as np
import pytest
import torch

from beampinn.losses.residuals import pde_residual
from beampinn.physics.benchmarks import get_benchmark
from physref.derivatives import derivatives, strong_residual
from physref.formulations.mixed import MixedHardFF, TwoHeadFourierPINN, mixed_residuals, _ModeShapeXX
from physref.formulations.modal import (ModalField, ModalHardQ, TemporalFourierNet, galerkin_modal_system,
                                        modal_residual)
from physref.frozen import to_experiment_config

BM = get_benchmark("FE-D-M1")
EX = BM.reference("exact")
C2, G = BM.pde_coeffs("paper_eq49")
W1 = BM.fundamental_omega()


def tiny_cfg():
    cfg, _ = to_experiment_config("batch1_hard_tanh2")
    cfg.model.depth, cfg.model.width, cfg.model.m_fourier = 2, 12, 8
    return cfg


def pts(n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 1, generator=g, dtype=torch.float64) * BM.L,
            torch.rand(n, 1, generator=g, dtype=torch.float64) * BM.t_end)


# ------------------------------------------------------------------ derivatives
def test_forward_equals_reverse_values_and_param_grads():
    torch.set_default_dtype(torch.float64)
    from beampinn.models.networks import SpatioTemporalFourierPINN
    from beampinn.models.constraints import HardConstrainedFF
    net = SpatioTemporalFourierPINN(tiny_cfg().model, BM.L, BM.t_end, 3).double()
    model = HardConstrainedFF(net, EX, BM.L, BM.t_end, "tanh2", W1).double()
    x, t = pts()
    Dr, Df = derivatives(model, x, t, "reverse"), derivatives(model, x, t, "forward")
    for k in Dr:
        assert torch.allclose(Dr[k], Df[k], rtol=1e-9, atol=1e-9 * float(Dr[k].abs().max())), k
    ps = list(model.parameters())
    gr = torch.autograd.grad((strong_residual(model, x, t, C2, G, "reverse") ** 2).mean(), ps, allow_unused=True, materialize_grads=True)
    gf = torch.autograd.grad((strong_residual(model, x, t, C2, G, "forward") ** 2).mean(), ps, allow_unused=True, materialize_grads=True)
    for a, b in zip(gr, gf):
        assert torch.allclose(a, b, rtol=1e-8, atol=1e-10 * (1 + float(a.abs().max())))
    # and the strong residual equals the Batch-1 implementation
    assert torch.allclose(strong_residual(model, x, t, C2, G), pde_residual(model, x.clone().requires_grad_(), t.clone().requires_grad_(), C2, G))


@pytest.mark.parametrize("strategy", ["reverse", "forward"])
def test_exact_solution_has_zero_strong_residual(probe_factory, strategy):
    probe = probe_factory(EX)
    x, t = pts()
    r = strong_residual(probe, x, t, C2, G, strategy)
    assert float(r.abs().max()) < 1e-9 * C2 * EX.beta ** 4 * EX.A0


# ------------------------------------------------------------------ mixed
def test_mode_shape_second_derivative_closed_form():
    x = torch.linspace(0, BM.L, 101, dtype=torch.float64).reshape(-1, 1)
    np.testing.assert_allclose(_ModeShapeXX(EX)(x).numpy().ravel(), EX.u0(x.numpy().ravel()) * 0 + EX.A0 * EX.mode_shape(x.numpy().ravel(), 2), rtol=1e-10, atol=1e-12)


class _ExactMixed(torch.nn.Module):
    def __init__(self, probe):
        super().__init__(); self.p = probe
    def forward(self, x, t):
        x2 = x if x.requires_grad else x.clone().requires_grad_(True)
        u = self.p(x2, t)
        ux = torch.autograd.grad(u, x2, torch.ones_like(u), create_graph=True)[0]
        return u, torch.autograd.grad(ux, x2, torch.ones_like(ux), create_graph=True)[0]


def test_mixed_residuals_vanish_on_exact_solution(probe_factory):
    r_link, r_pde = mixed_residuals(_ExactMixed(probe_factory(EX)), *pts(), C2, G, EX.beta)
    scale = C2 * EX.beta ** 4 * EX.A0
    assert float(r_link.abs().max()) < 1e-9 * scale and float(r_pde.abs().max()) < 1e-8 * scale


def test_mixed_equals_strong_when_link_is_satisfied(probe_factory):
    """Equivalence: with v := u_xx the mixed PDE residual IS the strong residual, for ANY u."""
    torch.set_default_dtype(torch.float64)
    from beampinn.models.networks import SpatioTemporalFourierPINN
    net = SpatioTemporalFourierPINN(tiny_cfg().model, BM.L, BM.t_end, 5).double()
    m = _ExactMixed(net)
    x, t = pts()
    r_link, r_pde = mixed_residuals(m, x, t, C2, G, EX.beta)
    assert float(r_link.abs().max()) < 1e-10 * float(r_pde.abs().max() + 1)
    torch.testing.assert_close(r_pde, strong_residual(net, x, t, C2, G), rtol=1e-9, atol=1e-9)


def test_mixed_hard_ansatz_ic_bc_exact():
    torch.set_default_dtype(torch.float64)
    cfg = tiny_cfg()
    model = MixedHardFF(TwoHeadFourierPINN(cfg.model, BM.L, BM.t_end, 7), EX, BM.L, BM.t_end, "tanh2", W1).double()
    x = torch.linspace(0, BM.L, 33, dtype=torch.float64).reshape(-1, 1)
    t0 = torch.zeros_like(x).requires_grad_(True)
    u, v = model(x, t0)
    ut = torch.autograd.grad(u, t0, torch.ones_like(u))[0]
    np.testing.assert_allclose(u.detach().numpy().ravel(), EX.u0(x.numpy().ravel()), atol=1e-14)
    np.testing.assert_allclose(v.detach().numpy().ravel(), EX.A0 * EX.mode_shape(x.numpy().ravel(), 2), rtol=1e-9, atol=1e-12)
    assert float(ut.abs().max()) < 1e-14
    t = torch.linspace(0, 1, 21, dtype=torch.float64).reshape(-1, 1)
    for xe in (0.0, BM.L):
        xx = torch.full_like(t, xe).requires_grad_(True)
        ue, _ = model(xx, t)
        uxe = torch.autograd.grad(ue, xx, torch.ones_like(ue))[0]
        assert float(ue.abs().max()) < 1e-12 and float(uxe.abs().max()) < 1e-10


def test_two_head_net_u_head_equals_batch1_head_at_init():
    from beampinn.models.networks import SpatioTemporalFourierPINN
    torch.set_default_dtype(torch.float64)
    cfg = tiny_cfg()
    a = SpatioTemporalFourierPINN(cfg.model, BM.L, BM.t_end, 11).double()
    b = TwoHeadFourierPINN(cfg.model, BM.L, BM.t_end, 11).double()
    x, t = pts()
    torch.testing.assert_close(a(x, t), b(x, t)[0])


# ------------------------------------------------------------------ modal
def test_galerkin_projection_matches_closed_form_and_ic():
    sysm = galerkin_modal_system(EX, n_modes=3)
    for m in sysm:
        assert abs(m["omega2_galerkin"] / m["omega2_closed"] - 1) < 1e-6
    assert abs(sysm[0]["omega2_closed"] - EX.omega ** 2) < 1e-9 * EX.omega ** 2
    assert abs(sysm[0]["q0"] - 1) < 1e-6 and abs(sysm[1]["q0"]) < 1e-6 and abs(sysm[2]["q0"]) < 1e-6


class _ExactQ(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.d = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    def forward(self, t):
        w, g = EX.omega, EX.gamma
        wd = math.sqrt(w * w - g * g / 4)
        return torch.exp(-g * t / 2) * (torch.cos(wd * t) + g / (2 * wd) * torch.sin(wd * t)) + 0 * self.d


def test_modal_exact_q_gives_zero_residual_and_exact_field():
    t = torch.linspace(0, 1, 201, dtype=torch.float64).reshape(-1, 1)
    r = modal_residual(_ExactQ(), t, C2 * EX.beta ** 4, G)
    assert float(r.abs().max()) < 1e-8 * EX.omega ** 2
    field = ModalField(_ExactQ(), EX)
    from beampinn.evaluation.metrics import grid, predict, rel_l2
    _, _, X, T = grid(BM.L, BM.t_end, 41, 401)
    assert rel_l2(predict(field, X, T), EX.u(X, T)) < 1e-12


def test_modal_field_pde_residual_is_shape_times_ode_residual():
    torch.set_default_dtype(torch.float64)
    q = ModalHardQ(TemporalFourierNet(BM.t_end, 8, (10.0, 1.0), 2, 12, 3), 1.0, BM.t_end, "tanh2", W1).double()
    field = ModalField(q, EX)
    x, t = pts()
    r_full = strong_residual(field, x, t, C2, G)
    r_ode = modal_residual(q, t, C2 * EX.beta ** 4, G)
    torch.testing.assert_close(r_full, field.shape(x) * r_ode, rtol=1e-8, atol=1e-8)


def test_modal_hard_ic():
    q = ModalHardQ(TemporalFourierNet(BM.t_end, 8, (10.0, 1.0), 2, 12, 3), 1.0, BM.t_end, "tanh2", W1).double()
    t0 = torch.zeros(5, 1, dtype=torch.float64, requires_grad=True)
    v = q(t0)
    assert torch.allclose(v, torch.ones_like(v))
    assert float(torch.autograd.grad(v.sum(), t0)[0].abs().max()) == 0.0
