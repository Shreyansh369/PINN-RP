"""Mixed / auxiliary-variable formulation of the damped Euler-Bernoulli beam (axis 15.2).

    v := u_xx                         (auxiliary field = curvature; bending moment / EI)
    r_link = c2 * beta1^2 * (v - u_xx)        -> scaled into the units of r_pde (see below)
    r_pde  = c2 * v_xx + u_tt + gamma * u_t

Equivalence: if r_link = 0 identically, r_pde equals the strong residual c2 u_xxxx + u_tt +
gamma u_t, so the solution set is unchanged (both fields smooth). Highest derivative order is
2 in x and 2 in t, instead of 4 in x. This is ESTABLISHED technique (deep mixed residual method,
Lyu et al. 2022; auxiliary PINNs, Yuan et al. 2022; A-PINN for EB beams 2026) - not our novelty.

Scaling of r_link (declared, dimensional, not tuned): for a modal field u ~ A0 phi(x) q(t),
|u_xx| ~ beta1^2 A0 and |r_pde terms| ~ w1^2 A0 = c2 beta1^4 A0, so c2*beta1^2*(v-u_xx) has the
same characteristic magnitude as r_pde. No further weight is introduced.

Hard constraints (fixed-fixed, same ansatz family as Batch-1 B1):
    u = u0(x) + g(t) Phi(x) A0 N_u(x,t)
    v = u0''(x) + g(t) beta1^2 A0 N_v(x,t)   -> v(x,0) = u0''(x) exactly (consistent with u)
v carries no BC: at a clamped end the moment is not zero.
"""
import math

import torch
import torch.nn as nn

from beampinn.models.constraints import FixedFixedModeShape, HardConstrainedFF
from beampinn.models.networks import SpatioTemporalFourierPINN

from ..derivatives import _d_rev


class TwoHeadFourierPINN(SpatioTemporalFourierPINN):
    """The Batch-1 spatio-temporal Fourier network with a 2-output linear head (N_u, N_v).
    Trunk, encodings and initialisation are identical; only the head width changes."""

    def __init__(self, cfg, L, t_end, seed):
        super().__init__(cfg, L, t_end, seed)
        old = self.head
        self.head = nn.Linear(old.in_features, 2)
        g = torch.Generator().manual_seed(seed + 2)
        std = math.sqrt(2.0 / (old.in_features + 2))
        with torch.no_grad():
            self.head.weight.copy_(torch.randn(self.head.weight.shape, generator=g) * std)
            self.head.weight[0].copy_(old.weight[0])          # N_u head = Batch-1 head at init
            self.head.bias.zero_()
            self.head.bias[0] = old.bias[0]

    def forward(self, x, t):
        xs, ts = self.tx(x), self.tt(t)
        hx = [self.trunk(e(xs)) for e in self.enc_x]
        ht = [self.trunk(e(ts)) for e in self.enc_t]
        out = self.output_scale * self.head(torch.cat([a * b for a in hx for b in ht], dim=1))
        return out[:, :1], out[:, 1:2]


class _ModeShapeXX(nn.Module):
    """Second x-derivative of A0*U(x)/U(x_norm) (fixed-fixed), closed form in torch."""

    def __init__(self, case):
        super().__init__()
        self.ms = FixedFixedModeShape(case)

    def forward(self, x):
        b, s, sc = self.ms.b, self.ms.s, self.ms.scale
        z = b * x
        return sc * b * b * (torch.cosh(z) + torch.cos(z) - s * (torch.sinh(z) + torch.sin(z)))


class MixedHardFF(nn.Module):
    """(x, t) -> (u, v) with all u IC/BCs and v(x,0) = u0'' exact. Reuses the Batch-1 ansatz."""

    def __init__(self, net2, case, L, T, time_factor="tanh2", omega_1=None):
        super().__init__()
        self.net2 = net2
        self.hard = HardConstrainedFF(_First(net2), case, L, T, time_factor, omega_1)
        self.u0xx = _ModeShapeXX(case)
        self.vscale = float(case.A0) * case.beta ** 2

    def forward(self, x, t):
        n_u, n_v = self.net2(x, t)
        u = self.hard.u0(x) + self.hard.g(t) * self.hard.phi(x) * self.hard.A0 * n_u
        v = self.u0xx(x) + self.hard.g(t) * self.vscale * n_v
        return u, v

    def displacement(self, x, t):
        return self.forward(x, t)[0]


class _First(nn.Module):
    def __init__(self, net2):
        super().__init__()
        self.net2 = [net2]                      # not registered twice

    def forward(self, x, t):
        return self.net2[0](x, t)[0]


class DisplacementView(nn.Module):
    """Expose u only, so the Batch-1 evaluation (beampinn.evaluation.metrics) runs unchanged."""

    def __init__(self, mixed):
        super().__init__()
        self.mixed = mixed

    def forward(self, x, t):
        return self.mixed(x, t)[0]


def mixed_residuals(model, x, t, c2, gamma, beta1):
    """Return (r_link, r_pde); reverse mode, max derivative order 2."""
    x = x.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)
    u, v = model(x, t)
    u_xx = _d_rev(_d_rev(u, x), x)
    v_xx = _d_rev(_d_rev(v, x), x)
    u_t = _d_rev(u, t)
    u_tt = _d_rev(u_t, t)
    r_link = c2 * beta1 ** 2 * (v - u_xx)
    r_pde = c2 * v_xx + u_tt + gamma * u_t
    return r_link, r_pde
