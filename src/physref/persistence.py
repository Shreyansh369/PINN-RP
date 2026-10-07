"""Persistence / collapse metric - PORTED VERBATIM from Batch 1 (ROOT
experiments/budget_diagnostic.py @ d31864a, functions local_amp and persistence). The definition
is FROZEN for Batch 2 (master prompt section 19: never change the success metric after seeing
results). Only change: the velocity trace is computed here instead of being imported from the
Batch-1 plotting module (same autograd definition: du/dt at x_norm).

  A(t)  = half peak-to-peak of the mid-span displacement in a sliding window of one damped period
          P = 2 pi / omega_d centred at t;  R(t) = A_pred(t) / A_exact(t)
  collapse time = first t >= P/2 with R(t) < 0.5  (= t_end if never)
  persistence   = collapse time * f_d [cycles]; full window = 20.6 cycles for FE-D-M1.
"""
import numpy as np
import torch

from beampinn.evaluation.metrics import predict

COLLAPSE_RATIO = 0.5          # FROZEN (Batch 1)
N_TRACE = 4001                # FROZEN (Batch 1)


def local_amp(y, t, P):
    dt = t[1] - t[0]; h = max(1, int(round(P / 2 / dt)))
    out = np.empty_like(y)
    for i in range(len(y)):
        w = y[max(0, i - h):i + h + 1]
        out[i] = 0.5 * (w.max() - w.min())
    return out


def velocity_trace(model, xn, t):
    dt = next(model.parameters()).dtype
    tt = torch.tensor(t, dtype=dt).reshape(-1, 1).requires_grad_(True)
    xx = torch.full_like(tt, float(xn))
    u = model(xx, tt)
    v = torch.autograd.grad(u, tt, torch.ones_like(u))[0]
    return v.detach().double().numpy().ravel()


def persistence(model, ref, T):
    t = np.linspace(0, T, N_TRACE)
    xn = ref.x_norm
    up = predict(model, np.full_like(t, xn)[:, None], t[:, None]).ravel()
    vp = velocity_trace(model, xn, t)
    ue, ve = ref.u(xn, t), ref.u(xn, t, 0, 1)
    P = 2 * np.pi / ref.omega_d
    R = local_amp(up, t, P) / local_amp(ue, t, P)
    Rv = local_amp(vp, t, P) / local_amp(ve, t, P)
    ok = t >= P / 2

    def first_below(r):
        idx = np.where(ok & (r < COLLAPSE_RATIO))[0]
        return float(t[idx[0]]) if len(idx) else float(T)
    tc, tcv = first_below(R), first_below(Rv)
    fd = ref.omega_d / (2 * np.pi)
    return {"collapse_time_s": tc, "persistence_cycles": tc * fd, "collapse_time_vel_s": tcv,
            "persistence_cycles_vel": tcv * fd, "full_window_cycles": T * fd}, (t, up, vp, ue, ve, R, Rv)


def passes_persistence_gate(p):
    """Batch-1 gate: no collapse inside the window (displacement AND velocity)."""
    full = p["full_window_cycles"]
    return p["persistence_cycles"] >= full * (1 - 1e-12) and p["persistence_cycles_vel"] >= full * (1 - 1e-12)
