"""Derivative-computation strategies for the beam residual (research axis 15.9).

  'reverse' : nested reverse-mode autograd (torch.autograd.grad, create_graph=True) - the
              Batch-1 implementation (beampinn.losses.residuals).
  'forward' : nested forward-mode JVPs (torch.func.jvp) for the pure x- and t-derivatives.
              Valid because the model is POINTWISE in (x, t) (no coupling across the batch), so
              a tangent of ones gives every per-point derivative at once. Parameter gradients
              are then taken by reverse mode through the JVP graph (forward-over-reverse).

Both give identical values and parameter gradients up to round-off (tests/test_b2_derivatives.py).
Which is FASTER is an empirical question per architecture/hardware; it is measured, never assumed
(scripts/profile_derivatives.py)."""
import torch
from torch.func import jvp

STRATEGIES = ("reverse", "forward")


def _d_rev(y, v):
    return torch.autograd.grad(y, v, torch.ones_like(y), create_graph=True)[0]


def derivs_reverse(model, x, t, kx=4, kt=2):
    """Return dict with u, u_t, u_tt (kt<=2) and u_x..u_x^kx via nested reverse mode."""
    x = x.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)
    u = model(x, t)
    out = {"u": u}
    g = u
    for k in range(1, kx + 1):
        g = _d_rev(g, x)
        out["u_" + "x" * k] = g
    g = u
    for k in range(1, kt + 1):
        g = _d_rev(g, t)
        out["u_" + "t" * k] = g
    return out


def _nested_jvp(fun, v, k):
    """k-th derivative of the pointwise function `fun` w.r.t. its argument (forward mode)."""
    one = torch.ones_like(v)
    g = fun
    for _ in range(k):
        g = (lambda h: (lambda z: jvp(h, (z,), (one,))[1]))(g)
    return g(v)


def derivs_forward(model, x, t, kx=4, kt=2):
    x, t = x.detach(), t.detach()
    out = {"u": model(x, t)}
    for k in range(1, kx + 1):
        out["u_" + "x" * k] = _nested_jvp(lambda z: model(z, t), x, k)
    for k in range(1, kt + 1):
        out["u_" + "t" * k] = _nested_jvp(lambda z: model(x, z), t, k)
    return out


def derivatives(model, x, t, strategy="reverse", kx=4, kt=2):
    if strategy == "reverse":
        return derivs_reverse(model, x, t, kx, kt)
    if strategy == "forward":
        return derivs_forward(model, x, t, kx, kt)
    raise ValueError(strategy)


def strong_residual(model, x, t, c2, gamma, strategy="reverse"):
    """Strong form r = c2 u_xxxx + u_tt + gamma u_t (Batch-1 Eq. 49 form), any strategy.
    NB: the 'forward' strategy evaluates intermediate orders it does not need; a production
    implementation would compute only orders 4 (x) and 1, 2 (t) - see COMPUTE_COST_MODEL.md."""
    D = derivatives(model, x, t, strategy)
    return c2 * D["u_xxxx"] + D["u_tt"] + gamma * D["u_t"]
