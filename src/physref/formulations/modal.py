"""Modal / reduced-order formulation (axis 15.3). Modal reduction itself is ESTABLISHED
(classical modal superposition; modal/reduced-order PINNs, e.g. RO-PINN, Zhang-Vlachas-Chatzi
2026). It is used here as a CONTROLLED DIAGNOSTIC, not a claimed contribution.

Derivation (Galerkin projection). Insert u(x,t) = sum_n A0 phi_n(x) q_n(t) into
    c2 u_xxxx + u_tt + gamma u_t = 0
with fixed-fixed eigenfunctions phi_n'''' = beta_n^4 phi_n (exact roots) and project onto phi_m.
Orthogonality of the phi_n gives decoupled modal ODEs
    q_n'' + gamma q_n' + w_n^2 q_n = 0,     w_n^2 = c2 beta_n^4,
    q_n(0) = <u0, phi_n> / (A0 <phi_n, phi_n>),   q_n'(0) = 0   (u_t(x,0) = 0).
For FE-D-M1 the IC is exactly A0 phi_1, so q_1(0) = 1 and q_n(0) = 0 (n > 1): the ONE-mode
ansatz contains the exact solution. Consequence (stated up front, never hidden): on this
benchmark a modal PINN solves a 1-D damped-oscillator ODE over the same 20.6 cycles; it isolates
TEMPORAL propagation from the 4th-order SPATIAL operator. It is not a fair full-field competitor
and its accuracy must never be presented as a full-field PINN result.

Hard IC ansatz for the modal ODE (same time-factor family as Batch-1 B1):
    q(t) = q0 + g(t) N(t),  g(0) = g'(0) = 0  ->  q(0) = q0, q'(0) = 0 exactly.
"""
import numpy as np
import torch
import torch.nn as nn

from beampinn.physics.beam import eigen_root, mode_shape_raw

from ..derivatives import _d_rev


def galerkin_modal_system(case, n_modes=3, nq=20001):
    """Numerical Galerkin projection (trapezoid on nq points). Returns per-mode beta_n, w_n^2
    (from <phi, c2 phi''''>/<phi, phi>), the closed-form c2 beta_n^4, and IC coefficients."""
    x = np.linspace(0.0, case.L, nq)
    u0 = case.u0(x)
    out = []
    for n in range(1, n_modes + 1):
        beta = eigen_root(case.bc_type, n) / case.L
        phi = mode_shape_raw(case.bc_type, beta, case.L, x)
        phi4 = mode_shape_raw(case.bc_type, beta, case.L, x, 4)
        m = np.trapezoid(phi * phi, x)
        k = np.trapezoid(phi * case.c2 * phi4, x)
        phi_n = phi / np.max(np.abs(phi))
        a = np.trapezoid(u0 * phi_n, x) / (case.A0 * np.trapezoid(phi_n * phi_n, x))
        out.append({"n": n, "beta": beta, "omega2_galerkin": k / m,
                    "omega2_closed": case.c2 * beta ** 4, "q0": a})
    return out


class TemporalFourierNet(nn.Module):
    """t -> N(t): the Batch-1 temporal branch (standardised t, Fourier sigma_t, tanh MLP),
    without the spatial branch. Declared representation for the modal arm."""

    def __init__(self, T, m=100, sigma_t=(10.0, 1.0), depth=6, width=200, seed=1234, two_pi=False):
        super().__init__()
        from beampinn.models.networks import FourierEncoding, InputTransform, _init, _mlp
        g = torch.Generator().manual_seed(seed)
        self.tt = InputTransform("standardize", 0.0, T)
        self.enc = nn.ModuleList(FourierEncoding(m, s, two_pi, g) for s in sigma_t)
        self.trunk = _mlp(2 * m, depth, width, nn.Tanh)
        self.head = nn.Linear(width * len(sigma_t), 1)
        _init(self, "xavier_normal", "zeros", torch.Generator().manual_seed(seed + 1))

    def forward(self, t):
        ts = self.tt(t)
        return self.head(torch.cat([self.trunk(e(ts)) for e in self.enc], dim=1))


class ModalHardQ(nn.Module):
    """q(t) = q0 + g(t) N(t), g = tanh^2(w1 t) (or (t/T)^2)."""

    def __init__(self, net, q0=1.0, T=1.0, time_factor="tanh2", omega_1=None):
        super().__init__()
        self.net, self.q0, self.T = net, float(q0), float(T)
        self.time_factor, self.omega_1 = time_factor, float(omega_1 or 0.0)
        if time_factor == "tanh2" and not omega_1:
            raise ValueError("tanh2 needs omega_1")

    def g(self, t):
        return (t / self.T) ** 2 if self.time_factor == "t2" else torch.tanh(self.omega_1 * t) ** 2

    def forward(self, t):
        return self.q0 + self.g(t) * self.net(t)


class ModalField(nn.Module):
    """u(x,t) = A0 phi_1(x)/phi_1(x_norm) * q(t). Same (x, t) interface as the Batch-1 models,
    so the IDENTICAL full-field evaluation (201 x 2001 grid, both references) applies."""

    def __init__(self, q_model, case):
        super().__init__()
        from beampinn.models.constraints import FixedFixedModeShape
        self.q, self.shape = q_model, FixedFixedModeShape(case)

    def forward(self, x, t):
        return self.shape(x) * self.q(t)


def modal_residual(q_model, t, omega2, gamma):
    """r = q'' + gamma q' + w^2 q  (scaled by nothing; units 1/s^2, same as r_pde / A0)."""
    t = t.detach().clone().requires_grad_(True)
    q = q_model(t)
    q_t = _d_rev(q, t)
    q_tt = _d_rev(q_t, t)
    return q_tt + gamma * q_t + omega2 * q
