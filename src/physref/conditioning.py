"""Physics / conditioning ANALYZER (research axis 15.1). Diagnostic only - it reports measurable
properties and CANDIDATE transformations; it does not choose a method. Selection rules may only be
written after controlled experiments establish them (docs/BATCH2_RESEARCH_HYPOTHESES.md H1).

Implemented for linear, constant-coefficient PDEs of the form
    sum_{(i,j)} a_ij d^i_x d^j_t u = 0 on [0,L] x [0,T]
with a known fundamental spatial wavenumber beta1 (from the BC eigenproblem). The damped
Euler-Bernoulli beam is the first instance: a_40 = c2, a_02 = 1, a_01 = gamma.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
from scipy.special import erfc


@dataclass
class LinearPDESpec:
    name: str
    terms: Dict[Tuple[int, int], float]     # {(x_order, t_order): coefficient}
    L: float
    T: float
    beta1: float                             # fundamental wavenumber [1/m]
    A0: float                                # amplitude scale of u
    bc_type: str = ""
    notes: str = field(default="")

    @property
    def max_x_order(self):
        return max(i for i, _ in self.terms)

    @property
    def max_t_order(self):
        return max(j for _, j in self.terms)


def beam_spec(benchmark_id="FE-D-M1", pde_coeffs="paper_eq49"):
    from beampinn.physics.benchmarks import get_benchmark
    bm = get_benchmark(benchmark_id)
    ref = bm.reference("exact", pde_coeffs)
    c2, g = bm.pde_coeffs(pde_coeffs)
    return LinearPDESpec(benchmark_id, {(4, 0): c2, (0, 2): 1.0, (0, 1): g}, bm.L, bm.t_end,
                         ref.beta, ref.A0, bm.bc_type), ref


def modal_frequency(spec):
    """Undamped modal frequency of the fundamental mode for a 2nd-order-in-time operator:
    sum over x-only terms (even order 2k) of a_(2k,0) beta^(2k) (-1)^k ... restricted here to
    the beam/wave families: w^2 = sum_{i>0} |a_i0| beta1^i / a_02."""
    a02 = spec.terms.get((0, 2))
    if not a02:
        raise NotImplementedError("first-order-in-time PDEs: use decay rate instead of frequency")
    return math.sqrt(sum(abs(a) * spec.beta1 ** i for (i, j), a in spec.terms.items() if j == 0 and i > 0) / a02)


def characteristic_scales(spec):
    w = modal_frequency(spec)
    gamma = spec.terms.get((0, 1), 0.0) / spec.terms[(0, 2)]
    wd = math.sqrt(max(w * w - 0.25 * gamma * gamma, 0.0))
    zeta = gamma / (2 * w)
    out = {"omega1_rad_s": w, "omega_d_rad_s": wd, "f_d_Hz": wd / (2 * math.pi),
           "period_s": 2 * math.pi / wd if wd else float("inf"),
           "cycles_in_window": spec.T * wd / (2 * math.pi), "damping_ratio": zeta,
           "decay_rate_1_s": 0.5 * gamma, "envelope_at_T": math.exp(-0.5 * gamma * spec.T),
           "timescale_ratio_T_over_1_by_w": spec.T * w,
           "stiffness_ratio_w_over_decay": w / (0.5 * gamma) if gamma else float("inf"),
           "wavelength_ratio_L_beta1": spec.L * spec.beta1,
           "max_x_order": spec.max_x_order, "max_t_order": spec.max_t_order}
    return out


def term_magnitudes(spec):
    """Characteristic magnitude of each PDE term for a modal field u ~ A0 phi(beta1 x) q(w t):
    |a_ij d^i_x d^j_t u| ~ |a_ij| beta1^i w^j A0. Ratios expose built-in scale disparities."""
    w = modal_frequency(spec)
    mags = {f"a{i}{j}": abs(a) * spec.beta1 ** i * w ** j * spec.A0 for (i, j), a in spec.terms.items()}
    ref = max(mags.values())
    return {"magnitudes": mags, "ratios_to_max": {k: v / ref for k, v in mags.items()},
            "residual_scale_w2A0": w * w * spec.A0}


def nondimensionalization_candidates(spec):
    """Coefficients of the PDE after x = Lx * xi, t = Tc * tau, u = A0 * U, for candidate scales.
    Coefficient of term (i,j) becomes a_ij / (Lx^i Tc^j); all are divided by the u_tt coefficient."""
    w = modal_frequency(spec)
    cands = {"physical (Lx=1 m, Tc=1 s)": (1.0, 1.0),
             "domain (Lx=L, Tc=T)": (spec.L, spec.T),
             "modal (Lx=1/beta1, Tc=1/w1)": (1.0 / spec.beta1, 1.0 / w)}
    out = {}
    for name, (Lx, Tc) in cands.items():
        c = {(i, j): a / (Lx ** i * Tc ** j) for (i, j), a in spec.terms.items()}
        norm = c[(0, 2)]
        c = {f"a{i}{j}": v / norm for (i, j), v in c.items()}
        out[name] = {"coeffs": c, "xi_range": spec.L / Lx, "tau_range": spec.T / Tc,
                     "coeff_spread": max(abs(v) for v in c.values()) / min(abs(v) for v in c.values())}
    return out


def hard_ansatz_conditioning(spec, time_factors=("t2", "tanh2")):
    """Required network output near t = 0 for u = u0 + g(t) Phi A0 N: matching u_tt(x,0) =
    -w^2 u0 gives N* = -w^2 u0 / (g''(0) Phi A0). Reported as -w^2/g''(0) (times the O(1) shape
    ratio u0/(Phi A0), measured in Batch 1 as [0.32, 3.8] for FE-D-M1 -> N* in [-1.9, -0.16] for
    tanh2). g''(0): t2 -> 2/T^2; tanh2 -> 2 w1^2."""
    w = modal_frequency(spec)
    g2 = {"t2": 2.0 / spec.T ** 2, "tanh2": 2.0 * w * w}
    return {tf: {"g2_at_0": g2[tf], "N_star_scale": -w * w / g2[tf]} for tf in time_factors}


def fourier_coverage(sigma, target_rad_s, input_norm="standardize", two_pi=False, span=1.0, m=100):
    """Expected fraction (and count of m) of Gaussian Fourier features whose PHYSICAL angular
    frequency exceeds `target_rad_s`. Physical frequency = factor*|B|/scale, B ~ N(0, sigma^2);
    scale = span/sqrt(12) (standardize), span (unit), 1 (physical)."""
    scale = {"standardize": span / math.sqrt(12.0), "unit": span, "physical": 1.0}[input_norm]
    factor = 2 * math.pi if two_pi else 1.0
    thr = target_rad_s * scale / factor              # threshold on |B|
    frac = float(erfc(thr / (sigma * math.sqrt(2.0))))
    return {"sigma": sigma, "threshold_on_B": thr, "expected_fraction": frac, "expected_count": frac * m}


def spectral_content(y, dt, energy=0.99, pad=16):
    """Dominant frequency and the bandwidth holding `energy` of the spectral energy of a trace
    (Hann window, zero-padded x`pad`; raw resolution is 1/(N dt))."""
    y = np.asarray(y, float) - np.mean(y)
    n = pad * len(y)
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y)), n)) ** 2
    f = np.fft.rfftfreq(n, dt)
    c = np.cumsum(Y) / Y.sum()
    return {"dominant_Hz": float(f[np.argmax(Y)]), f"band_{int(energy*100)}pct_Hz": float(f[np.searchsorted(c, energy)])}


def analyze(spec, ref=None, fourier=None):
    rep = {"problem": spec.name, "bc_type": spec.bc_type, "terms": {f"a{i}{j}": a for (i, j), a in spec.terms.items()},
           "scales": characteristic_scales(spec), "term_magnitudes": term_magnitudes(spec),
           "nondimensionalization": nondimensionalization_candidates(spec),
           "hard_ansatz": hard_ansatz_conditioning(spec)}
    if ref is not None:
        t = np.linspace(0, spec.T, 20001)
        rep["spectral_content_midspan"] = spectral_content(ref.u(ref.x_norm, t), t[1] - t[0])
    if fourier:
        wd = rep["scales"]["omega_d_rad_s"]
        rep["fourier_coverage_t"] = {name: [fourier_coverage(s, wd, cfg["input_norm"], cfg["two_pi"], spec.T)
                                            for s in cfg["sigma_t"]] for name, cfg in fourier.items()}
    return rep


def init_term_statistics(model, residual_terms, params):
    """Loss magnitude and gradient norm of each loss term AT INITIALISATION (no optimizer step).
    residual_terms: {name: residual vector}. Returns {name: {"loss", "grad_norm"}} and ratios."""
    import torch
    out = {}
    for k, r in residual_terms.items():
        loss = 0.5 * (r ** 2).mean()
        g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        out[k] = {"loss": float(loss), "grad_norm": math.sqrt(sum(float((x.double() ** 2).sum()) for x in g if x is not None))}
    gmax = max(v["grad_norm"] for v in out.values()) or 1.0
    for v in out.values():
        v["grad_ratio_to_max"] = v["grad_norm"] / gmax
    return out
