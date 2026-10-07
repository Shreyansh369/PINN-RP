"""Temporal domain decomposition PLANNING (axis 15.4). Dry-run scaffolding only: it builds and
validates window plans and the interface-continuity residuals; no training loop is included.

Prior art (must be cited by any result): time-marching / seq2seq (Krishnapriyan et al. 2021),
bc-PINN (Mattey & Ghosh 2022), XPINN temporal decomposition and stacked decomposition with
transfer learning (Penwarden et al. 2023), exact temporal continuity in sequential PINNs (Roy
et al. 2024), AT-PINN / AT-PINN-HC for structural vibration (Chen et al. 2024/2025).
For a 2nd-order-in-time PDE the interface needs BOTH u and u_t continuity."""
from dataclasses import dataclass, field
from typing import List

import torch

from .derivatives import _d_rev


@dataclass
class Window:
    index: int
    t0: float
    t1: float

    @property
    def length(self):
        return self.t1 - self.t0


@dataclass
class TemporalPlan:
    T: float
    windows: List[Window] = field(default_factory=list)
    overlap: float = 0.0
    init: str = "transfer"          # 'transfer' (previous weights) | 'fresh'
    interface: str = "hard_ic"      # 'hard_ic' (window IC := previous terminal state) | 'soft'

    def cycles_per_window(self, omega_d):
        import math
        return [w.length * omega_d / (2 * math.pi) for w in self.windows]


def make_plan(T, n_windows, overlap=0.0, init="transfer", interface="hard_ic"):
    if n_windows < 1 or not (0.0 <= overlap < T / n_windows):
        raise ValueError("need n_windows >= 1 and 0 <= overlap < T/n_windows")
    h = T / n_windows
    ws = [Window(i, max(0.0, i * h - overlap), (i + 1) * h) for i in range(n_windows)]
    plan = TemporalPlan(T, ws, overlap, init, interface)
    validate_plan(plan)
    return plan


def validate_plan(plan):
    ws = plan.windows
    assert abs(ws[0].t0) < 1e-15 and abs(ws[-1].t1 - plan.T) < 1e-12, "windows must cover [0, T]"
    for a, b in zip(ws, ws[1:]):
        assert b.t0 <= a.t1 + 1e-15, "gap between windows"
        assert b.t1 > a.t1, "windows must advance"
    return True


def interface_residuals(model_a, model_b, x, t_iface):
    """Continuity of u and u_t at t = t_iface between consecutive window models (vectors)."""
    t = torch.full_like(x, float(t_iface)).requires_grad_(True)
    ua, ub = model_a(x, t), model_b(x, t)
    return ua - ub, _d_rev(ua, t) - _d_rev(ub, t)
