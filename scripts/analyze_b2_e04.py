"""B2-E04 - POST-HOC MECHANISTIC ANALYSIS of EXISTING checkpoints. NO TRAINING.

Implements docs/hypotheses/B2-E04.md EXACTLY (every constant and threshold below is copied from it).
Nothing here creates, modifies or resumes a training checkpoint: checkpoints are only `torch.load`-ed,
their state is copied into freshly built (frozen) modules, and diagnostics are evaluated on a float64 copy.
No optimizer is ever constructed and no parameter is ever updated.

    python scripts/analyze_b2_e04.py manifest            # sha256 of every source checkpoint + integrity
    python scripts/analyze_b2_e04.py dryrun              # pre-registered dry-run (one checkpoint) + checks
    python scripts/analyze_b2_e04.py unit KEY            # diagnostics of one state
    python scripts/analyze_b2_e04.py run --workers 3     # every state not yet done (subprocesses)
    python scripts/analyze_b2_e04.py aggregate           # CSVs, classification, figures

Outputs (never overwritten): results_batch2/B2-E04/ (unit files, manifests, dry-run),
results_batch2/B2-E04_{DIAGNOSTICS,MODAL_SPECTRUM,RESIDUAL_SPECTRUM,GRADIENT_CONFLICT,CONDITIONING}.csv,
results_batch2/B2-E04_CLASSIFICATION.json, results_batch2/figures/B2-E04/*.png
"""
import copy
import csv
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from beampinn.config import ExperimentConfig  # noqa: E402
from beampinn.evaluation.metrics import EVAL_NT, EVAL_NX, grid, l2_pair, predict  # noqa: E402
from beampinn.losses.residuals import d, pde_residual  # noqa: E402
from beampinn.models.networks import build_model  # noqa: E402
from beampinn.physics.beam import eigen_root, mode_shape_raw, modal_time  # noqa: E402
from beampinn.training.trainer import build_hard, resolve_problem, setup_torch  # noqa: E402
from physref.collocation_lab import state_checksum  # noqa: E402
from physref.formulations.mixed import DisplacementView, MixedHardFF, TwoHeadFourierPINN  # noqa: E402
from physref.formulations.modal import ModalField, ModalHardQ, TemporalFourierNet  # noqa: E402
from physref.persistence import N_TRACE, local_amp, persistence  # noqa: E402
from physref.safety import assert_safe_output  # noqa: E402

RUNS = REPO / "results_batch2" / "runs"
OUT = REPO / "results_batch2" / "B2-E04"
UNITS = OUT / "units"
FIG = REPO / "results_batch2" / "figures" / "B2-E04"
SEEDS = (1234, 1235, 1236)
F64 = torch.float64

# ------------------------------------------------------------------ pre-registered constants (B2-E04.md)
N_MODES = 8                       # section 4.1
NX_GL_A, NT_A = 64, N_TRACE       # grid A: 64 Gauss-Legendre x-nodes x 4001 uniform t (frozen trace grid)
NX_GL_B, NT_B = 64, 501           # grid B: 64 GL x 501 uniform t (frozen PDE_NT)
NX_GL_C, NT_C, CT = 16, 256, 8    # set C: 16 GL x 256 t-midpoints; chunks of 8 t-nodes (32 chunks)
N_J, J_SEED, J_CHUNK = 640, 20261008, 32
TAU_RES = 1e-10
TAUS = (1e-2, 1e-4, 1e-6, 1e-8, 1e-10)
KS = (2, 5, 10, 20, 50, 100, 200, 400, 640)
WINDOWS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))   # W1 closed, others (a, b]
H_ONSET = 0.05                    # non-fundamental energy onset threshold (fraction of exact energy)
# pathology criteria (section 7) and paired-difference margins (section 8)
THR = dict(A_nonfund_pre=0.05, B_hr_pre=0.5, B_minC=-0.1, B_cos_front=-0.1, B_starv=0.01,
           C_cst=-0.5, C_ratio=10.0, D_dlog_kappa=1.0, D_rank_frac=0.75)
MARGIN = dict(frac=0.1, nonfund=0.02, cos=0.1, log10=0.3, rank_rel=0.10)
LEVELS = {"0k": 0, "128k": 128_000, "256k": 256_000, "320k": 320_000, "384k": 384_000, "512k": 512_000,
          "640k": 640_000, "623k": 623_360}


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# =============================================================================================== states
def _rid_dir(run):
    d = RUNS / run / "checkpoints"
    subs = [p for p in d.iterdir() if p.is_dir()]
    return subs[0] if subs else d


def states(seed):
    """Pre-registered state set (section 3). key -> {problem, path (None = reconstructed init), nominal}."""
    b1, sw = _rid_dir(f"B2-E01-B1-s{seed}"), _rid_dir(f"B2-E03-LBFGS-F-s{seed}")
    tr, m0 = _rid_dir(f"B2-E02T-LBFGS-s{seed}"), _rid_dir(f"B2-E02-M0-s{seed}")
    ml, mx = _rid_dir(f"B2-E02-LBFGS-s{seed}"), _rid_dir(f"B2-E01-mixed-s{seed}")
    S = {}

    def add(prob, label, path, lev, cfg_from):
        S[f"{prob}_{label}_s{seed}"] = {"problem": prob, "label": label, "seed": seed,
                                         "path": None if path is None else str(path), "level": lev,
                                         "cfg_from": str(cfg_from)}
    add("FF", "init", None, "0k", b1 / "step_1000.pt")
    for k, lev in ((1000, "128k"), (2000, "256k"), (3000, "384k"), (4000, "512k"), (5000, "640k")):
        add("FF", f"b1-{lev}", b1 / f"step_{k}.pt", lev, b1 / f"step_{k}.pt")
    add("FF", "sw-320k", sw / "switch.pt", "320k", sw / "switch.pt")
    for name, lev in (("step_3000.pt", "384k"), ("step_4000.pt", "512k"), ("final.pt", "623k")):
        add("FF", f"tr-{lev}", tr / name, lev, tr / name)
    add("MO", "init", None, "0k", m0 / "step_1000.pt")
    for k, lev in ((1000, "128k"), (2000, "256k"), (3000, "384k"), (4000, "512k"), (5000, "640k")):
        add("MO", f"m0-{lev}", m0 / f"step_{k}.pt", lev, m0 / f"step_{k}.pt")
    for name, lev in (("step_3000.pt", "384k"), ("step_4000.pt", "512k"), ("final.pt", "623k")):
        add("MO", f"ml-{lev}", ml / name, lev, ml / name)
    add("MX", "init", None, "0k", mx / "step_1000.pt")
    for k, lev in ((1000, "128k"), (2000, "256k"), (3000, "384k"), (4000, "512k"), (5000, "640k")):
        add("MX", f"mx-{lev}", mx / f"step_{k}.pt", lev, mx / f"step_{k}.pt")
    return S


def all_states():
    out = {}
    for s in SEEDS:
        out.update(states(s))
    return out


def trajectories(seed):
    """trajectory -> [(level, state key)] (section 3.2). Shared pre-switch states appear in both arms."""
    k = lambda p, l: f"{p}_{l}_s{seed}"  # noqa: E731
    ff_pre = [("0k", k("FF", "init")), ("128k", k("FF", "b1-128k")), ("256k", k("FF", "b1-256k")),
              ("320k", k("FF", "sw-320k"))]
    mo_pre = [("0k", k("MO", "init")), ("128k", k("MO", "m0-128k")), ("256k", k("MO", "m0-256k"))]
    return {
        "FF-B1": ff_pre + [(l, k("FF", f"b1-{l}")) for l in ("384k", "512k", "640k")],
        "FF-LBFGS": ff_pre + [(l, k("FF", f"tr-{l}")) for l in ("384k", "512k", "623k")],
        "MO-ADAM": mo_pre + [(l, k("MO", f"m0-{l}")) for l in ("384k", "512k", "640k")],
        "MO-LBFGS": mo_pre + [(l, k("MO", f"ml-{l}")) for l in ("384k", "512k", "623k")],
        "MX": [("0k", k("MX", "init"))] + [(l, k("MX", f"mx-{l}")) for l in ("128k", "256k", "384k", "512k", "640k")],
    }


# =============================================================================================== models
def load_cfg(path):
    blob = torch.load(path, weights_only=False, map_location="cpu")
    return ExperimentConfig.from_dict(blob["config"]), blob


def build(prob, cfg):
    """Fresh module exactly as the trainers build it (float32). Returns (trainable model, field view, problem)."""
    setup_torch(cfg)
    bm, refs, c2, gamma, _ = resolve_problem(cfg)
    ex = refs["exact"]
    om1 = bm.fundamental_omega(cfg.benchmark.pde_coeffs)
    m = cfg.model
    if prob == "FF":
        model = build_hard(build_model(cfg, bm), cfg, bm, refs).to(torch.float32)
        view = model
    elif prob == "MO":
        qnet = TemporalFourierNet(bm.t_end, m.m_fourier, m.sigma_t, m.depth, m.width, cfg.seed, m.two_pi,
                                  burn_spatial_sigmas=m.sigma_x)
        model = ModalHardQ(qnet, 1.0, bm.t_end, "tanh2", om1).to(torch.float32)
        view = ModalField(model, ex).to(torch.float32)
    elif prob == "MX":
        net2 = TwoHeadFourierPINN(m, bm.L, bm.t_end, cfg.seed)
        model = MixedHardFF(net2, ex, bm.L, bm.t_end, "tanh2", om1).to(torch.float32)
        view = DisplacementView(model)
    else:
        raise ValueError(prob)
    P = {"bm": bm, "refs": refs, "ex": ex, "c2": c2, "gamma": gamma, "omega2": c2 * ex.beta ** 4,
         "A0": ex.A0, "L": bm.L, "T": bm.t_end}
    return model, view, P


def load_state(st):
    """(model32, view32, model64, view64, P, blob_meta). Read-only: the checkpoint file is only loaded."""
    cfg, blob = load_cfg(st["cfg_from"])
    model, view, P = build(st["problem"], cfg)
    meta = {"pde_evaluations": 0, "status": "init", "step": 0}
    if st["path"] is not None:
        b = blob if st["path"] == st["cfg_from"] else torch.load(st["path"], weights_only=False, map_location="cpu")
        model.load_state_dict(b["model"])
        meta = {"pde_evaluations": int(b["acc"]["pde_evaluations"]), "status": b["status"], "step": int(b["step"])}
    model.eval()
    m64 = copy.deepcopy(model).to(F64)
    if st["problem"] == "FF":
        v64 = m64
    elif st["problem"] == "MO":
        v64 = ModalField(m64, P["ex"]).to(F64)
    else:
        v64 = DisplacementView(m64)
    return model, view, m64, v64, P, meta


def params_of(model):
    return [p for p in model.parameters() if p.requires_grad]


def flat(gs, params):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, params)])


# ============================================================================================== physics
def ff_terms(view, x, t, c2, gamma):
    """Physical operator components of the strong residual (paper Eq. 49, pde_scale 1):
    r_time = u_tt + gamma u_t,  r_space = c2 u_xxxx,  r = r_time + r_space."""
    x = x.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)
    u = view(x, t)
    ux = d(u, x); uxx = d(ux, x); uxxx = d(uxx, x); uxxxx = d(uxxx, x)
    ut = d(u, t); utt = d(ut, t)
    return utt + gamma * ut, c2 * uxxxx


def mo_terms(model, t, P):
    """Modal training residual (physref.arms.ArmTrainer.loss): A0 (q'' + gamma q' + w1^2 q), split into
    r_time = A0 (q'' + gamma q') and r_space = A0 w1^2 q (Galerkin projection of c2 u_xxxx onto phi_1)."""
    t = t.detach().clone().requires_grad_(True)
    q = model(t)
    qt = d(q, t); qtt = d(qt, t)
    return P["A0"] * (qtt + P["gamma"] * qt), P["A0"] * P["omega2"] * q


def basis(P, x):
    """phi_n(x), n = 1..N_MODES: exact-root fixed-fixed eigenfunctions, max|phi_n| = 1 (sign: +1 at argmax
    |U_n| on a 20001-point grid). phi_1 is identical to the benchmark's mode shape (asserted)."""
    bm = P["bm"]
    xs = np.linspace(0.0, bm.L, 20001)
    out = []
    for n in range(1, N_MODES + 1):
        beta = eigen_root(bm.bc_type, n) / bm.L
        U = mode_shape_raw(bm.bc_type, beta, bm.L, xs)
        out.append(mode_shape_raw(bm.bc_type, beta, bm.L, x) / U[np.argmax(np.abs(U))])
    Phi = np.array(out)
    assert np.allclose(Phi[0], P["ex"].mode_shape(x), rtol=0, atol=1e-12), "phi_1 != benchmark mode shape"
    return Phi


def gl(n, L):
    xi, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * L * (xi + 1.0), 0.5 * L * w


def project(F, Phi, w):
    """F: (nx, nt) field on GL nodes. Returns coefficients (N_MODES, nt), mode energies m_n a_n^2 (N_MODES, nt),
    total energy sum_i w_i F^2 (nt), remainder energy (nt)."""
    m = (Phi ** 2) @ w
    a = (Phi * w) @ F / m[:, None]
    e = a ** 2 * m[:, None]
    tot = w @ F ** 2
    rem = w @ (F - Phi.T @ a) ** 2
    return a, e, tot, rem, m


def periodP(P):
    return 2 * np.pi / P["ex"].omega_d


def regions(t, tc, P):
    """Section 4.3: W1..W4, FULL, PRE = [0, t_c), POST = [t_c, T] (POST only if t_c < T - P_d)."""
    R = {"FULL": np.ones_like(t, bool)}
    for i, (a, b) in enumerate(WINDOWS):
        R[f"W{i+1}"] = (t >= a) & (t <= b) if i == 0 else (t > a) & (t <= b)
    R["PRE"] = t < tc
    if tc < P["T"] - periodP(P):
        R["POST"] = t >= tc
    return R


# ======================================================================================= diagnostic A
def diag_A(view64, P, tc):
    L, T, A0 = P["L"], P["T"], P["A0"]
    x, w = gl(NX_GL_A, L)
    t = np.linspace(0.0, T, NT_A)
    X, Tm = np.meshgrid(x, t, indexing="ij")
    U = predict(view64, X, Tm)
    Phi = basis(P, x)
    a, e, tot, rem, m = project(U, Phi, w)
    a1ex = A0 * modal_time(P["ex"].omega, P["gamma"], t)
    eex = a1ex ** 2 * m[0]
    Pd = periodP(P)
    R1 = local_amp(a[0], t, Pd) / local_amp(a1ex, t, Pd)
    idx = np.where((t >= Pd / 2) & (R1 < 0.5))[0]
    tc1 = float(t[idx[0]]) if len(idx) else float(T)
    nonfund = e[1:].sum(0) + rem
    h = max(1, int(round(Pd / 2 / (t[1] - t[0]))))
    kern = np.ones(2 * h + 1)
    Hloc = np.convolve(nonfund, kern, "same") / np.convolve(eex, kern, "same")
    idx = np.where((t >= Pd / 2) & (Hloc >= H_ONSET))[0]
    th = float(t[idx[0]]) if len(idx) else float("nan")
    out = {"t_c1": tc1, "t_h": th}
    for rn, msk in regions(t, tc, P).items():
        if msk.sum() < 2:
            continue
        den = eex[msk].mean()
        r = {f"rho_{n+1}": float(e[n, msk].mean() / den) for n in range(N_MODES)}
        r["rho_hi"] = float(e[1:, msk].mean(1).sum() / den)
        r["rho_2_4"] = float(e[1:4, msk].mean(1).sum() / den)
        r["rho_5_8"] = float(e[4:, msk].mean(1).sum() / den)
        r["rho_even"] = float(e[1::2, msk].mean(1).sum() / den)
        r["rho_rem"] = float(rem[msk].mean() / den)
        r["rho_nonfund"] = r["rho_hi"] + r["rho_rem"]
        r["rho_tot"] = float(tot[msk].mean() / den)
        es = e[:, msk].mean(1)
        for n in range(N_MODES):
            r[f"f_{n+1}"] = float(es[n] / es.sum())
        r["f_hi"] = float(es[1:].sum() / es.sum())
        r["eps1"] = float(np.linalg.norm(a[0, msk] - a1ex[msk]) / np.linalg.norm(a1ex[msk]))
        out[rn] = r
    if "POST" in out:
        miss = max(0.0, 1.0 - out["POST"]["rho_1"])
        nf = out["POST"]["rho_nonfund"]
        cls = "REDISTRIBUTION" if nf >= 0.5 * miss else ("AMPLITUDE_LOSS" if nf <= 0.1 * miss else "MIXED")
        out["post_class"], out["post_missing"] = cls, miss
    else:
        out["post_class"], out["post_missing"] = "NOT_COLLAPSED", float("nan")
    arrays = {"A_t": t, "A_a": a, "A_e": e, "A_tot": tot, "A_rem": rem, "A_a1ex": a1ex, "A_R1": R1, "A_H": Hloc}
    return out, arrays


# ======================================================================================= diagnostic B
def field_residual(view64, P, X, Tm, chunk=1024):
    xs = torch.from_numpy(X.reshape(-1, 1)); ts = torch.from_numpy(Tm.reshape(-1, 1))
    RT, RS = [], []
    for i in range(0, len(xs), chunk):
        rt, rs = ff_terms(view64, xs[i:i + chunk], ts[i:i + chunk], P["c2"], P["gamma"])
        RT.append(rt.detach()); RS.append(rs.detach())
    rt = torch.cat(RT).numpy().reshape(X.shape); rs = torch.cat(RS).numpy().reshape(X.shape)
    return rt, rs


def diag_B(view64, P, tc):
    L, T = P["L"], P["T"]
    x, w = gl(NX_GL_B, L)
    t = np.linspace(0.0, T, NT_B)
    X, Tm = np.meshgrid(x, t, indexing="ij")
    rt, rs = field_residual(view64, P, X, Tm)
    r = rt + rs
    Phi = basis(P, x)
    rn, s, sr, srem, m = project(r, Phi, w)
    _, st_, _, _, _ = project(rt, Phi, w)
    _, ss_, _, _, _ = project(rs, Phi, w)
    utt = P["ex"].u(X, Tm, 0, 2)
    den = math.sqrt(np.mean(w @ utt ** 2) / L)
    out = {}
    for rname, msk in regions(t, tc, P).items():
        if msk.sum() < 2:
            continue
        tot = sr[msk].mean()
        sm = s[:, msk].mean(1)
        o = {f"sig_{n+1}": float(sm[n] / tot) for n in range(N_MODES)}
        o["sig_rem"] = float(srem[msk].mean() / tot)
        o["sig_even"] = float(sm[1::2].sum() / tot)
        o["h_r"] = float(1.0 - sm[0] / tot)
        o["centroid"] = float((np.arange(1, N_MODES + 1) * sm).sum() / sm.sum())
        cand = list(sm) + [srem[msk].mean()]
        o["dominant"] = int(np.argmax(cand)) + 1 if np.argmax(cand) < N_MODES else "rem"
        o["R_norm"] = float(math.sqrt(tot / L) / den)
        o["time_mode1_frac"] = float(st_[0, msk].mean() / (w @ rt[:, msk] ** 2).mean())
        o["space_mode1_frac"] = float(ss_[0, msk].mean() / (w @ rs[:, msk] ** 2).mean())
        out[rname] = o
    arrays = {"B_t": t, "B_rn": rn, "B_s": s, "B_sr": sr, "B_srem": srem}
    return out, arrays


# ================================================================================== diagnostics C + D
def chunk_grads(prob, model64, view64, P):
    """Per t-chunk (8 consecutive t-midpoints x 16 GL x-nodes for FF; 8 t-midpoints for MO):
    G_T = grad sum w sg(r) r_time, G_S = grad sum w sg(r) r_space, H_T = grad sum w sg(r_time) r_time,
    H_S = grad sum w sg(r_space) r_space, Omega = sum w, and the weighted sums of r^2, r_time^2, r_space^2."""
    params = params_of(model64)
    T = P["T"]
    tm = (np.arange(NT_C) + 0.5) * T / NT_C
    xg, wg = gl(NX_GL_C, P["L"])
    nchunk = NT_C // CT
    G = {k: np.zeros((nchunk, sum(p.numel() for p in params))) for k in ("GT", "GS", "HT", "HS")}
    S = {k: np.zeros(nchunk) for k in ("Om", "r2", "rt2", "rs2")}
    for c in range(nchunk):
        tc_ = tm[c * CT:(c + 1) * CT]
        if prob == "FF":
            X, Tm = np.meshgrid(xg, tc_, indexing="ij")
            W = np.repeat(wg[:, None], CT, 1).reshape(-1, 1)
            xt = torch.from_numpy(X.reshape(-1, 1)); tt = torch.from_numpy(Tm.reshape(-1, 1))
            rt, rs = ff_terms(view64, xt, tt, P["c2"], P["gamma"])
        else:
            W = np.ones((CT, 1))
            rt, rs = mo_terms(model64, torch.from_numpy(tc_.reshape(-1, 1)), P)
        Wt = torch.from_numpy(W)
        r = rt + rs
        rd, rtd, rsd = r.detach(), rt.detach(), rs.detach()
        objs = {"GT": (Wt * rd * rt).sum(), "GS": (Wt * rd * rs).sum(), "HT": (Wt * rtd * rt).sum(),
                "HS": (Wt * rsd * rs).sum()}
        keys = list(objs)
        for i, k in enumerate(keys):
            G[k][c] = flat(torch.autograd.grad(objs[k], params, retain_graph=i < len(keys) - 1, allow_unused=True),
                           params).numpy()
        S["Om"][c] = float(Wt.sum()); S["r2"][c] = float((Wt * rd ** 2).sum())
        S["rt2"][c] = float((Wt * rtd ** 2).sum()); S["rs2"][c] = float((Wt * rsd ** 2).sum())
    return G, S, tm


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def window_vectors(G, S, chunks):
    Om = S["Om"][chunks].sum()
    gT, gS = G["GT"][chunks].sum(0) / Om, G["GS"][chunks].sum(0) / Om
    return {"g": gT + gS, "gT": gT, "gS": gS, "g_time": 2 * G["HT"][chunks].sum(0) / Om,
            "g_space": 2 * G["HS"][chunks].sum(0) / Om, "L": 0.5 * S["r2"][chunks].sum() / Om,
            "L_time": S["rt2"][chunks].sum() / Om, "L_space": S["rs2"][chunks].sum() / Om}


def st_metrics(v):
    nt, ns = np.linalg.norm(v["g_time"]), np.linalg.norm(v["g_space"])
    nT, nS = np.linalg.norm(v["gT"]), np.linalg.norm(v["gS"])
    return {"C_st": cos(v["g_time"], v["g_space"]), "n_gtime": float(nt), "n_gspace": float(ns),
            "ratio_space_time": float(ns / nt) if nt > 0 else float("nan"),
            "C_TS": cos(v["gT"], v["gS"]), "chi": float(np.linalg.norm(v["g"]) / (nT + nS)) if nT + nS > 0 else float("nan"),
            "n_gT": float(nT), "n_gS": float(nS), "n_g": float(np.linalg.norm(v["g"])),
            "L": float(v["L"]), "L_time": float(v["L_time"]), "L_space": float(v["L_space"])}


def front_chunks(tc, P):
    """Section 5.3 front-relative windows on the chunk grid (chunk c covers [c, c+1] * T/32)."""
    Pd, T = periodP(P), P["T"]
    if tc >= T - Pd:
        return None
    n = NT_C // CT
    lo = np.arange(n) * T / n; hi = lo + T / n
    behind = np.where(hi <= tc - Pd)[0]
    ahead = np.where(lo >= tc + Pd)[0]
    front = np.array([c for c in range(n) if c not in set(behind) | set(ahead)])
    return {"BEHIND": behind, "FRONT": front, "AHEAD": ahead}


def diag_CD(prob, model64, view64, P, tc):
    G, S, tm = chunk_grads(prob, model64, view64, P)
    n = NT_C // CT
    wins = [np.arange(i * n // 4, (i + 1) * n // 4) for i in range(4)]
    V = [window_vectors(G, S, c) for c in wins]
    full = window_vectors(G, S, np.arange(n))
    out = {}
    for i in range(4):
        out[f"n_g{i+1}"] = float(np.linalg.norm(V[i]["g"]))
        out[f"L_W{i+1}"] = float(V[i]["L"])
        out[f"cos_g{i+1}_full"] = cos(V[i]["g"], full["g"])
        for j in range(i + 1, 4):
            out[f"C_{i+1}{j+1}"] = cos(V[i]["g"], V[j]["g"])
    Cs = [out[f"C_{i+1}{j+1}"] for i in range(4) for j in range(i + 1, 4)]
    out["minC"] = float(np.nanmin(Cs))
    out["frac_neg"] = float(np.mean([c < 0 for c in Cs]))
    out["starv"] = out["n_g4"] / out["n_g1"] if out["n_g1"] > 0 else float("nan")
    out["max_late_over_early"] = max(out[f"n_g{i}"] for i in (2, 3, 4)) / out["n_g1"] if out["n_g1"] > 0 else float("nan")
    fr = front_chunks(tc, P)
    fv = {}
    if fr is not None:
        for k, c in fr.items():
            fv[k] = window_vectors(G, S, c) if len(c) else None
    def fcos(a, b):
        return cos(fv[a]["g"], fv[b]["g"]) if fv.get(a) is not None and fv.get(b) is not None else float("nan")
    out["cos_behind_front"] = fcos("BEHIND", "FRONT")
    out["cos_front_ahead"] = fcos("FRONT", "AHEAD")
    out["cos_behind_ahead"] = fcos("BEHIND", "AHEAD")
    for k in ("BEHIND", "FRONT", "AHEAD"):
        out[f"n_g_{k}"] = float(np.linalg.norm(fv[k]["g"])) if fv.get(k) is not None else float("nan")
    out["front_over_behind"] = out["n_g_FRONT"] / out["n_g_BEHIND"] if out["n_g_BEHIND"] > 0 else float("nan")
    out["ahead_over_front"] = out["n_g_AHEAD"] / out["n_g_FRONT"] if out["n_g_FRONT"] > 0 else float("nan")
    D = {"FULL": st_metrics(full)}
    for i in range(4):
        D[f"W{i+1}"] = st_metrics(V[i])
    if fv.get("BEHIND") is not None:
        D["BEHIND"] = st_metrics(fv["BEHIND"])
    Cmat = np.eye(4)
    for i in range(4):
        for j in range(i + 1, 4):
            Cmat[i, j] = Cmat[j, i] = out[f"C_{i+1}{j+1}"]
    return out, D, {"C_matrix": Cmat, "C_tm": tm}


# ======================================================================================= diagnostic E
def jac_points(P):
    g = torch.Generator().manual_seed(J_SEED)
    x = torch.rand(N_J, 1, generator=g, dtype=F64) * P["L"]
    t = torch.rand(N_J, 1, generator=g, dtype=F64) * P["T"]
    return x, t


def jacobian(prob, model64, view64, P, chunk=J_CHUNK):
    params = params_of(model64)
    npar = sum(p.numel() for p in params)
    x, t = jac_points(P)
    J = np.empty((N_J, npar), order="C")
    for i0 in range(0, N_J, chunk):
        if prob == "FF":
            rt, rs = ff_terms(view64, x[i0:i0 + chunk], t[i0:i0 + chunk], P["c2"], P["gamma"])
        else:
            rt, rs = mo_terms(model64, t[i0:i0 + chunk], P)
        r = (rt + rs).reshape(-1)
        for i in range(len(r)):
            J[i0 + i] = flat(torch.autograd.grad(r[i], params, retain_graph=i < len(r) - 1, allow_unused=True),
                             params).numpy()
    return J, x.numpy().ravel(), t.numpy().ravel()


def svals_inplace(J):
    """Singular values of the N x P (N << P) matrix J via Householder QR of J^T (LAPACK dgeqrf, in place;
    J is destroyed) and the SVD of the N x N triangular factor. Backward stable: |error| ~ eps ||J||."""
    from scipy.linalg import lapack, svdvals
    A = J.T                                         # F-contiguous view, no copy
    qr, tau, work, info = lapack.dgeqrf(A, overwrite_a=1)
    assert info == 0
    n = J.shape[0]
    return np.sort(svdvals(np.triu(qr[:n, :n])))[::-1]


def spec_metrics(s):
    s1 = s[0]
    o = {"sigma_1": float(s1), "sigma_N": float(s[-1]), "N": int(len(s))}
    for tau in TAUS:
        o[f"rank_{tau:.0e}"] = int((s >= tau * s1).sum())
    res = s[s >= TAU_RES * s1]
    o["sigma_res"] = float(res[-1])
    o["kappa_hat"] = float(s1 / res[-1])
    o["log10_kappa_hat"] = float(np.log10(s1 / res[-1]))
    o["saturated"] = bool(len(res) < len(s))
    for k in KS:
        if k <= len(s):
            o[f"s{k}_over_s1"] = float(s[k - 1] / s1)
    p = s / s.sum()
    p = p[p > 0]
    o["rank_entropy"] = float(np.exp(-(p * np.log(p)).sum()))
    return o


def diag_E(prob, model64, view64, P):
    J, xj, tj = jacobian(prob, model64, view64, P)
    out = {}
    for i, (a, b) in enumerate(WINDOWS):
        msk = (tj >= a) & (tj <= b) if i == 0 else (tj > a) & (tj <= b)
        sw = np.linalg.svd(J[msk], compute_uv=False)
        o = spec_metrics(sw)
        out[f"W{i+1}"] = {k: o[k] for k in ("N", "sigma_1", "rank_1e-06", "log10_kappa_hat", "saturated")}
    s = svals_inplace(J)
    del J
    out["FULL"] = spec_metrics(s)
    return out, {"E_s": s}


# ================================================================================ frozen metrics + unit
def frozen_metrics(view32, P):
    _, _, X, Tm = grid(P["L"], P["T"], EVAL_NX, EVAL_NT)
    out = l2_pair(predict(view32, X, Tm), P["refs"], X, Tm)
    pers, _ = persistence(view32, P["ex"], P["T"])
    out.update(pers)
    return {k: out[k] for k in ("L2_exact", "L2_paper", "L2_late_exact", "collapse_time_s", "persistence_cycles",
                                "collapse_time_vel_s")}


def unit_paths(key):
    return UNITS / f"{key}.json", UNITS / f"{key}.npz"


def compute_unit(key, st, do_grad=True, write=True):
    t0 = time.perf_counter()
    torch.set_num_threads(1)
    model32, view32, m64, v64, P, meta = load_state(st)
    rec = {"key": key, **{k: st[k] for k in ("problem", "label", "seed", "path", "level")}, **meta,
           "nominal_evaluations": LEVELS[st["level"]], "timing_s": {}}
    rec["frozen"] = frozen_metrics(view32, P)
    tc = rec["frozen"]["collapse_time_s"]
    arrays = {}
    t1 = time.perf_counter()
    rec["A"], a = diag_A(v64, P, tc); arrays.update(a)
    rec["timing_s"]["A"] = time.perf_counter() - t1; t1 = time.perf_counter()
    rec["B"], a = diag_B(v64, P, tc); arrays.update(a)
    rec["timing_s"]["B"] = time.perf_counter() - t1; t1 = time.perf_counter()
    if do_grad and st["problem"] in ("FF", "MO"):
        rec["C"], rec["D"], a = diag_CD(st["problem"], m64, v64, P, tc); arrays.update(a)
        rec["timing_s"]["CD"] = time.perf_counter() - t1; t1 = time.perf_counter()
        rec["E"], a = diag_E(st["problem"], m64, v64, P); arrays.update(a)
        rec["timing_s"]["E"] = time.perf_counter() - t1
    rec["timing_s"]["total"] = time.perf_counter() - t0
    rec["peak_rss_mb"] = rss_mb()
    if write:
        jp, npz = unit_paths(key)
        assert not jp.exists() and not npz.exists(), f"refusing to overwrite {jp}"
        np.savez_compressed(npz, **{k: np.asarray(v) for k, v in arrays.items()})
        with open(jp, "w") as f:
            json.dump(rec, f, indent=1, default=float)
    return rec, arrays


# ============================================================================= manifest + integrity
def manifest(tag):
    files = sorted({st["path"] for st in all_states().values() if st["path"]} |
                   {st["cfg_from"] for st in all_states().values()})
    return {"tag": tag, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": {str(Path(p).relative_to(REPO)): {"sha256": sha256_file(p), "mode": oct(os.stat(p).st_mode & 0o777)}
                      for p in files}}


def sd_equal(p1, p2):
    a = torch.load(p1, weights_only=False, map_location="cpu")["model"]
    b = torch.load(p2, weights_only=False, map_location="cpu")["model"]
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def integrity():
    """Section 3.3 assertions (STOP on failure)."""
    res = {}
    for s in SEEDS:
        S = states(s)
        r = {}
        b1, sw = _rid_dir(f"B2-E01-B1-s{s}"), _rid_dir(f"B2-E03-LBFGS-F-s{s}")
        tr, m0, ml = _rid_dir(f"B2-E02T-LBFGS-s{s}"), _rid_dir(f"B2-E02-M0-s{s}"), _rid_dir(f"B2-E02-LBFGS-s{s}")
        for k in (1000, 2000):
            assert sd_equal(b1 / f"step_{k}.pt", tr / f"step_{k}.pt"), f"STOP: transfer != B1 at step {k} (s{s})"
            assert sd_equal(b1 / f"step_{k}.pt", sw / f"step_{k}.pt"), f"STOP: E03 != B1 at step {k} (s{s})"
            assert sd_equal(m0 / f"step_{k}.pt", ml / f"step_{k}.pt"), f"STOP: modal LBFGS != M0 at step {k} (s{s})"
        for arm in ("LBFGS-R", "LBFGS-4X", "ADAM-FULL"):
            assert sd_equal(sw / "switch.pt", _rid_dir(f"B2-E03-{arm}-s{s}") / "switch.pt"), f"STOP: switch differs {arm} s{s}"
        r["shared_pre_switch_states_bitwise_equal"] = True
        cks = json.load(open(RUNS / f"B2-E03-B1-s{s}" / "init_checksum.json"))["init_checksum"]
        st = S[f"FF_init_s{s}"]
        cfg, _ = load_cfg(st["cfg_from"])
        mff, _, _ = build("FF", cfg)
        assert state_checksum(mff) == cks, f"STOP: reconstructed FF init differs from the recorded checksum (s{s})"
        r["ff_init_checksum"] = cks
        cfgm, _ = load_cfg(S[f"MO_init_s{s}"]["cfg_from"])
        mmo, _, _ = build("MO", cfgm)
        a, b = mff.state_dict(), mmo.state_dict()
        pairs = [(k, k) for k in a if k.startswith("net.trunk") or k.startswith("net.head")]
        pairs += [(f"net.enc_t.{i}.B", f"net.enc.{i}.B") for i in range(2)]
        assert all(torch.equal(a[x], b[y]) for x, y in pairs), f"STOP: modal init != B1 temporal branch (s{s})"
        r["mo_init_equals_b1_temporal_branch"] = True
        cfgx, _ = load_cfg(S[f"MX_init_s{s}"]["cfg_from"])
        _, vmx, P = build("MX", cfgx)
        _, _, X, Tm = grid(P["L"], P["T"], 51, 101)
        u1, u2 = predict(mff, X, Tm), predict(vmx, X, Tm)
        assert np.max(np.abs(u1 - u2)) <= 1e-6 * np.max(np.abs(u1)), f"STOP: mixed init u != B1 init u (s{s})"
        r["mx_init_u_equals_b1_init_u"] = True
        for key, stt in S.items():
            if stt["path"]:
                b = torch.load(stt["path"], weights_only=False, map_location="cpu")
                assert b["status"] in ("running", "interrupted", "completed"), (key, b["status"])
                e = int(b["acc"]["pde_evaluations"])
                assert e == LEVELS[stt["level"]], f"STOP: {key} has {e} evaluations, expected {LEVELS[stt['level']]}"
        r["evaluation_counts_match_levels"] = True
        res[s] = r
    return res


CROSS = [  # (table, filter, our state label template) for frozen-metric cross-checks
    ("results_batch2/tables/B2-E01S_SEED_SNAPSHOTS.csv", "arm", "step", {"B1": ("FF", "b1"), "mixed": ("MX", "mx")}),
    ("results_batch2/B2-E02_SNAPSHOTS.csv", "arm", "checkpoint", {"M0": ("MO", "m0"), "LBFGS": ("MO", "ml")}),
    ("results_batch2/B2-E02T_SNAPSHOTS.csv", "arm", "checkpoint", {"B1": ("FF", "b1"), "LBFGS": ("FF", "tr")}),
    ("results_batch2/B2-E03_SNAPSHOTS.csv", "arm", "checkpoint", {"LBFGS-F": ("FF", "sw")}),
]
T_STEP = 1.0 / (N_TRACE - 1)      # one frozen persistence-trace step (tolerance of the t_c cross-check)
STEP2LEV = {"1000": "128k", "2000": "256k", "3000": "384k", "4000": "512k", "5000": "640k"}


def cross_check(U):
    """Frozen L2/collapse time recomputed here vs the previously reported snapshot tables."""
    n = 0
    for f, ak, ck, amap in CROSS:
        for row in csv.DictReader(open(REPO / f)):
            if row[ak] not in amap:
                continue
            prob, lab = amap[row[ak]]
            c = row[ck].replace("step_", "").replace(".pt", "")
            if c == "switch":
                lev = "320k"
            elif c == "final":
                lev = "623k"
            else:
                lev = STEP2LEV.get(c)
            if lev is None:
                continue
            key = f"{prob}_{lab}-{lev}_s{row['seed']}"
            if key not in U:
                continue
            fz = U[key]["frozen"]
            assert abs(fz["L2_exact"] / float(row["L2_exact"]) - 1) < 1e-6, ("STOP: L2 mismatch", key, f)
            assert abs(fz["collapse_time_s"] - float(row["collapse_time_s"])) <= T_STEP + 1e-12, ("STOP: t_c mismatch", key, f)
            n += 1
    return n


# ============================================================================================ dry-run
DRY_KEY = "FF_b1-128k_s1234"


def dryrun():
    out = assert_safe_output(OUT / "dryrun", strict=True, purpose="B2-E04 dry-run")
    out.mkdir(parents=True, exist_ok=False)
    rep = {"checkpoint": DRY_KEY}
    st = all_states()[DRY_KEY]
    before = sha256_file(st["path"])
    model32, view32, m64, v64, P, meta = load_state(st)
    # (1) basis orthogonality + exact-solution projection on grid A
    x, w = gl(NX_GL_A, P["L"])
    Phi = basis(P, x)
    M = (Phi * w) @ Phi.T
    off = np.max(np.abs(M - np.diag(np.diag(M))) / np.sqrt(np.outer(np.diag(M), np.diag(M))))
    t = np.linspace(0, P["T"], NT_A)
    X, Tm = np.meshgrid(x, t, indexing="ij")
    a, e, tot, rem, m = project(P["ex"].u(X, Tm), Phi, w)
    a1ex = P["A0"] * modal_time(P["ex"].omega, P["gamma"], t)
    rep["basis_offdiag_max"] = float(off)
    rep["exact_a1_relerr"] = float(np.linalg.norm(a[0] - a1ex) / np.linalg.norm(a1ex))
    rep["exact_nonfund_energy_frac"] = float((e[1:].sum() + rem.sum()) / e[0].sum())
    assert off < 1e-10 and rep["exact_a1_relerr"] < 1e-10 and rep["exact_nonfund_energy_frac"] < 1e-18, rep
    # (2) residual decomposition == frozen pde_residual
    g = torch.Generator().manual_seed(1)
    xs = torch.rand(64, 1, generator=g, dtype=F64) * P["L"]; ts = torch.rand(64, 1, generator=g, dtype=F64) * P["T"]
    rt, rs = ff_terms(v64, xs, ts, P["c2"], P["gamma"])
    xr, tr_ = xs.clone().requires_grad_(True), ts.clone().requires_grad_(True)
    rf = pde_residual(v64, xr, tr_, P["c2"], P["gamma"], 1.0)
    rep["residual_vs_frozen_maxrel"] = float(((rt + rs) - rf).abs().max() / rf.abs().max())
    assert rep["residual_vs_frozen_maxrel"] < 1e-12, rep
    # exact reference residual on grid B is ~0 (exact root, same coefficients)
    xb, wb = gl(NX_GL_B, P["L"]); tb = np.linspace(0, P["T"], NT_B)
    Xb, Tb = np.meshgrid(xb, tb, indexing="ij")
    rex = P["ex"].pde_residual(Xb, Tb)
    rep["exact_residual_rel"] = float(np.sqrt(np.mean(rex ** 2)) / np.sqrt(np.mean(P["ex"].u(Xb, Tb, 0, 2) ** 2)))
    assert rep["exact_residual_rel"] < 1e-9, rep
    # (3) finite-difference check of the window-loss gradient and of the decomposition identity
    params = params_of(m64)
    G, S, tm = chunk_grads("FF", m64, v64, P)
    W1 = window_vectors(G, S, np.arange(0, 8))
    theta0 = [p.detach().clone() for p in params]
    gv = torch.Generator().manual_seed(2)
    v = [torch.randn(p.shape, generator=gv, dtype=F64) for p in params]
    vflat = torch.cat([q.reshape(-1) for q in v]).numpy()
    vn = np.linalg.norm(vflat)
    th = np.linalg.norm(torch.cat([q.reshape(-1) for q in theta0]).numpy())
    eps = 1e-6 * th / vn

    def Lw1(sign):
        with torch.no_grad():
            for p, p0, vv in zip(params, theta0, v):
                p.copy_(p0 + sign * eps * vv)
        _, S2, _ = chunk_grads_loss_only(m64, v64, P, range(0, 8))
        return S2
    fd = (Lw1(+1) - Lw1(-1)) / (2 * eps)
    with torch.no_grad():
        for p, p0 in zip(params, theta0):
            p.copy_(p0)
    an = float(W1["g"] @ vflat)
    rep["fd_window_grad_relerr"] = abs(fd - an) / abs(an)
    assert rep["fd_window_grad_relerr"] < 1e-5, rep
    # g = g_T + g_S (by construction) and g equals the autograd gradient of the window loss directly
    xg, wg = gl(NX_GL_C, P["L"])
    Xc, Tc = np.meshgrid(xg, tm[:64], indexing="ij")
    Wc = torch.from_numpy(np.repeat(wg[:, None], 64, 1).reshape(-1, 1))
    xr = torch.from_numpy(Xc.reshape(-1, 1)).requires_grad_(True); tr2 = torch.from_numpy(Tc.reshape(-1, 1)).requires_grad_(True)
    rr = pde_residual(v64, xr, tr2, P["c2"], P["gamma"], 1.0)
    Ld = 0.5 * (Wc * rr ** 2).sum() / Wc.sum()
    gd = flat(torch.autograd.grad(Ld, params), params).numpy()
    rep["window_grad_vs_direct_relerr"] = float(np.linalg.norm(gd - W1["g"]) / np.linalg.norm(gd))
    assert rep["window_grad_vs_direct_relerr"] < 1e-10, rep
    # (4) Jacobian: chunk-size reproducibility, J^T r / N == gradient of 1/2 mean r^2 on set J
    tj0 = time.perf_counter()
    J32, xj, tj = jacobian("FF", m64, v64, P, chunk=32)
    rep["jacobian_seconds"] = time.perf_counter() - tj0
    J16, _, _ = jacobian("FF", m64, v64, P, chunk=16)
    rep["jacobian_chunk_maxrel"] = float(np.abs(J32 - J16).max() / np.abs(J32).max())
    del J16
    assert rep["jacobian_chunk_maxrel"] < 1e-12, rep
    xjt = torch.from_numpy(xj.reshape(-1, 1)); tjt = torch.from_numpy(tj.reshape(-1, 1))
    rt, rs = ff_terms(v64, xjt, tjt, P["c2"], P["gamma"])
    r = (rt + rs)
    gJ = flat(torch.autograd.grad(0.5 * (r ** 2).mean(), params), params).numpy()
    rep["JTr_vs_grad_relerr"] = float(np.linalg.norm(J32.T @ r.detach().numpy().ravel() / N_J - gJ) / np.linalg.norm(gJ))
    assert rep["JTr_vs_grad_relerr"] < 1e-10, rep
    s_svd = np.linalg.svd(J32[:128], compute_uv=False)
    s_qr = svals_inplace(np.ascontiguousarray(J32[:128]))
    tq = time.perf_counter()
    svals_inplace(J32)
    rep["qr_svd_seconds"] = time.perf_counter() - tq
    del J32
    keep = s_svd >= TAU_RES * s_svd[0]
    rep["qr_vs_svd_maxrel_resolvable"] = float(np.max(np.abs(s_qr[keep] - s_svd[keep]) / s_svd[keep]))
    assert rep["qr_vs_svd_maxrel_resolvable"] < 1e-6, rep
    rep["eps_floor_over_sigma1"] = float(np.finfo(float).eps * math.sqrt(N_J))
    assert rep["eps_floor_over_sigma1"] < TAU_RES, rep
    # (5) full unit timing (not written to units/), checkpoint untouched
    t0 = time.perf_counter()
    rec, _ = compute_unit(DRY_KEY, st, do_grad=True, write=False)
    rep["unit_seconds"] = time.perf_counter() - t0
    rep["unit_timing"] = rec["timing_s"]
    rep["peak_rss_mb"] = rss_mb()
    rep["checkpoint_sha256_unchanged"] = sha256_file(st["path"]) == before
    assert rep["checkpoint_sha256_unchanged"]
    with open(out / "dryrun_checks.json", "w") as f:
        json.dump(rep, f, indent=1, default=float)
    with open(out / "dryrun_unit.json", "w") as f:
        json.dump(rec, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in rep.items() if k != "unit_timing"}, indent=1, default=float))
    return rep


def chunk_grads_loss_only(model64, view64, P, chunks):
    """Window loss value (W1 in the dry-run) by the same quadrature, no gradients (finite-difference check)."""
    tm = (np.arange(NT_C) + 0.5) * P["T"] / NT_C
    xg, wg = gl(NX_GL_C, P["L"])
    num = den = 0.0
    for c in chunks:
        X, Tm = np.meshgrid(xg, tm[c * CT:(c + 1) * CT], indexing="ij")
        W = np.repeat(wg[:, None], CT, 1).reshape(-1)
        rt, rs = ff_terms(view64, torch.from_numpy(X.reshape(-1, 1)), torch.from_numpy(Tm.reshape(-1, 1)), P["c2"], P["gamma"])
        r = (rt + rs).detach().numpy().ravel()
        num += float((W * r ** 2).sum()); den += float(W.sum())
    return None, 0.5 * num / den, None


# ================================================================================= classification
def load_units():
    U = {}
    for p in sorted(UNITS.glob("*.json")):
        U[p.stem] = json.load(open(p))
    return U


def g(rec, *path, default=float("nan")):
    cur = rec
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def nanbool(x):
    return bool(x) if x is not None and not (isinstance(x, float) and math.isnan(x)) else False


def crit(mech, rec, ref=None):
    """Section 7: pathology criterion of each mechanism for one state. Returns (bool, detail dict)."""
    if mech == "A":
        nf = g(rec, "A", "PRE", "rho_nonfund"); hr = g(rec, "B", "PRE", "h_r"); cl = g(rec, "A", "post_class", default="")
        c1 = nf >= THR["A_nonfund_pre"] if not math.isnan(nf) else False
        c2 = hr >= THR["B_hr_pre"] if not math.isnan(hr) else False
        c3 = cl in ("REDISTRIBUTION", "MIXED")
        return c1 or c2 or c3, {"behind": c1 or c2, "post": c3}
    if mech == "B":
        mc, cbf, cfa, sv = (g(rec, "C", k) for k in ("minC", "cos_behind_front", "cos_front_ahead", "starv"))
        c1 = nanbool(mc <= THR["B_minC"]); c2 = nanbool(cbf <= THR["B_cos_front"])
        c3 = nanbool(cfa <= THR["B_cos_front"]); c4 = nanbool(sv <= THR["B_starv"])
        return c1 or c2 or c3 or c4, {"behind": c2, "at": c3, "window": c1, "starv": c4}
    if mech == "C":
        cst, ra = g(rec, "D", "FULL", "C_st"), g(rec, "D", "FULL", "ratio_space_time")
        c = nanbool(cst <= THR["C_cst"]) or nanbool(ra >= THR["C_ratio"]) or nanbool(ra <= 1 / THR["C_ratio"])
        cb, rb = g(rec, "D", "BEHIND", "C_st"), g(rec, "D", "BEHIND", "ratio_space_time")
        beh = nanbool(cb <= THR["C_cst"]) or nanbool(rb >= THR["C_ratio"]) or nanbool(rb <= 1 / THR["C_ratio"])
        return c, {"behind": beh}
    if mech == "D":
        if ref is None:
            return False, {}
        dk = g(rec, "E", "FULL", "log10_kappa_hat") - g(ref, "E", "FULL", "log10_kappa_hat")
        r6, r6r = g(rec, "E", "FULL", "rank_1e-06"), g(ref, "E", "FULL", "rank_1e-06")
        c = nanbool(dk >= THR["D_dlog_kappa"]) or nanbool(r6 <= THR["D_rank_frac"] * r6r)
        return c, {"dlog10_kappa": dk, "rank6_ratio": r6 / r6r if r6r else float("nan")}
    raise ValueError(mech)


# primary scalars for paired differences (section 8): (getter, pathology direction +1 = increase, margin kind)
PRIMARY = {
    "A": [(lambda r: g(r, "B", "PRE", "h_r"), +1, "frac"), (lambda r: g(r, "A", "PRE", "rho_nonfund"), +1, "nonfund")],
    "B": [(lambda r: g(r, "C", "minC"), -1, "cos"), (lambda r: g(r, "C", "cos_behind_front"), -1, "cos")],
    "C": [(lambda r: g(r, "D", "FULL", "C_st"), -1, "cos"),
          (lambda r: abs(math.log10(g(r, "D", "FULL", "ratio_space_time"))), +1, "log10")],
    "D": [(lambda r: g(r, "E", "FULL", "log10_kappa_hat"), +1, "log10"),
          (lambda r: g(r, "E", "FULL", "rank_1e-06"), -1, "rank_rel")],
}


def pathological_delta(mech, a, b):
    """True if state a is worse than paired state b beyond the margin on ANY primary scalar of mech."""
    for get, sgn, mk in PRIMARY[mech]:
        try:
            va, vb = get(a), get(b)
        except (ValueError, TypeError):
            continue
        if any(isinstance(v, float) and math.isnan(v) for v in (va, vb)):
            continue
        dv = sgn * (va - vb)
        thr = MARGIN[mk] * (abs(vb) if mk == "rank_rel" else 1.0)
        if dv > thr:
            return True
    return False


def end(traj):
    return "623k" if traj.endswith("LBFGS") else "640k"


KPOST = ("384k", "512k", "end")


def classify(U):
    Pd = 2 * np.pi / _omega_d()
    out = {"constants": {"THR": THR, "MARGIN": MARGIN, "P_d": Pd}, "per_seed": {}, "mechanisms": {}}
    per = {}
    for s in SEEDS:
        T = {k: dict(v) for k, v in trajectories(s).items()}
        rec = lambda tr, lev: U[T[tr][end(tr) if lev == "end" else lev]]  # noqa: E731
        ps = {}
        # degradation events (section 6.1)
        ev = None
        for lev in KPOST:
            if rec("FF-LBFGS", lev)["frozen"]["collapse_time_s"] < rec("FF-B1", lev)["frozen"]["collapse_time_s"] - Pd / 2:
                ev = lev; break
        ps["transfer_fall_behind_event"] = ev
        best, ev_m0 = -1.0, None
        for lev, key in trajectories(s)["MO-ADAM"]:
            tcv = U[key]["frozen"]["collapse_time_s"]
            if tcv < best - Pd / 2 and ev_m0 is None:
                ev_m0 = lev
            best = max(best, tcv)
        ps["m0_retreat_event"] = ev_m0
        order = [l for l, _ in trajectories(s)["FF-LBFGS"]]
        for mech in "ABCD":
            m = {}
            reff = U[T["FF-LBFGS"]["320k"]]; refm = U[T["MO-LBFGS"]["256k"]]
            cf = [crit(mech, rec("FF-LBFGS", l), reff)[0] for l in KPOST]
            cm = [crit(mech, rec("MO-LBFGS", l), refm)[0] for l in KPOST]
            m["crit_FF_LBFGS_post"] = cf
            m["crit_MO_LBFGS_post"] = cm
            m["crit_FF_B1_post"] = [crit(mech, rec("FF-B1", l), reff)[0] for l in KPOST]
            m["crit_MO_ADAM_post"] = [crit(mech, rec("MO-ADAM", l), refm)[0] for l in KPOST]
            m["R"] = sum(cf) >= 2
            m["S1"] = (sum(cf) >= 2) and (sum(not c for c in cm) >= 2) and mech != "A"
            dfull = [pathological_delta(mech, rec("FF-LBFGS", l), rec("FF-B1", l)) for l in KPOST]
            dmod = [pathological_delta(mech, rec("MO-LBFGS", l), rec("MO-ADAM", l)) for l in KPOST]
            m["delta_full_pathological"] = dfull
            m["delta_modal_pathological"] = dmod
            m["S2"] = (sum(dfull) >= 2) and (sum(not x for x in dmod) >= 2 or mech == "A")
            # ordering (section 6)
            first = None
            for l in order:
                key = T["FF-LBFGS"][l]
                if crit(mech, U[key], reff if LEVELS[l] >= 320_000 else None)[0] if mech == "D" else crit(mech, U[key])[0]:
                    first = l; break
            if ev is None:
                otr = "no_event"
            elif first is None:
                otr = "absent"
            else:
                ie, ifi = order.index("623k" if ev == "end" else ev), order.index(first)
                otr = "precedes" if ifi < ie else ("coincides" if ifi == ie else "follows")
            m["first_pathological_level"] = first
            m["O_train"] = otr
            evl = "623k" if ev == "end" else ev
            ophys = "none"
            if evl is not None:
                _, det = crit(mech, U[T["FF-LBFGS"][evl]], reff)
                if det.get("behind"):
                    ophys = "before"
                elif det.get("at"):
                    ophys = "at"
            m["O_phys_at_event"] = ophys
            m["O"] = otr in ("precedes", "coincides") or ophys in ("before", "at")
            ps[mech] = m
        per[s] = ps
    out["per_seed"] = per
    for mech in "ABCD":
        n = lambda k: sum(per[s][mech][k] for s in SEEDS)  # noqa: E731
        R, S1, S2, O = n("R") >= 2, n("S1") >= 2, n("S2") >= 2, n("O") >= 2
        S = S2 if mech == "A" else (S1 or S2)
        verdict = "SUPPORTED" if (R and S and O) else ("PARTIAL" if R and (S or O) else "NOT_SUPPORTED")
        out["mechanisms"][mech] = {"R_seeds": n("R"), "S1_seeds": n("S1"), "S2_seeds": n("S2"), "O_seeds": n("O"),
                                   "R": R, "S": S, "O": O, "verdict": verdict}
    sup = [m for m in "ABCD" if out["mechanisms"][m]["verdict"] == "SUPPORTED"]
    out["CASE"] = "F" if not sup else (sup[0] if len(sup) == 1 else "E")
    out["supported"] = sup
    return out


def _omega_d():
    from beampinn.physics.benchmarks import get_benchmark
    return get_benchmark("FE-D-M1").reference("exact").omega_d


# ============================================================================================ tables
def flat_dict(d, prefix=""):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict):
            o.update(flat_dict(v, f"{prefix}{k}."))
        elif not isinstance(v, (list, tuple)):
            o[f"{prefix}{k}"] = v
    return o


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    assert not path.exists(), f"refusing to overwrite {path}"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def tables(U, base):
    traj_of = {}
    for s in SEEDS:
        for tr, lst in trajectories(s).items():
            for lev, key in lst:
                traj_of.setdefault(key, []).append(tr)
    diag, modal, resid, grad, cond = [], [], [], [], []
    for s in SEEDS:
        for tr, lst in trajectories(s).items():
            for lev, key in lst:
                r = U[key]
                row = {"trajectory": tr, "seed": s, "level": lev, "state": key,
                       "pde_evaluations": r["pde_evaluations"], **r["frozen"],
                       "t_c1": r["A"]["t_c1"], "t_h": r["A"]["t_h"], "post_class": r["A"]["post_class"]}
                for reg in ("FULL", "PRE", "POST"):
                    row[f"rho_1_{reg}"] = g(r, "A", reg, "rho_1")
                    row[f"rho_nonfund_{reg}"] = g(r, "A", reg, "rho_nonfund")
                    row[f"h_r_{reg}"] = g(r, "B", reg, "h_r")
                    row[f"centroid_{reg}"] = g(r, "B", reg, "centroid")
                row["eps1_FULL"] = g(r, "A", "FULL", "eps1")
                row["R_norm_FULL"] = g(r, "B", "FULL", "R_norm")
                for k in ("n_g1", "n_g2", "n_g3", "n_g4", "minC", "frac_neg", "starv", "cos_behind_front",
                          "cos_front_ahead", "front_over_behind"):
                    row[k] = g(r, "C", k)
                for k in ("C_st", "ratio_space_time", "C_TS", "chi"):
                    row[k] = g(r, "D", "FULL", k)
                for k in ("sigma_1", "log10_kappa_hat", "rank_1e-06", "rank_1e-02", "rank_entropy",
                          "saturated"):
                    row[k] = g(r, "E", "FULL", k)
                diag.append(row)
    for key, r in sorted(U.items()):
        common = {"state": key, "problem": r["problem"], "seed": r["seed"], "level": r["level"],
                  "trajectories": "+".join(traj_of.get(key, [])), "pde_evaluations": r["pde_evaluations"],
                  "collapse_time_s": r["frozen"]["collapse_time_s"]}
        for reg, v in r["A"].items():
            if isinstance(v, dict):
                modal.append({**common, "region": reg, **v, "t_c1": r["A"]["t_c1"], "t_h": r["A"]["t_h"],
                              "post_class": r["A"]["post_class"]})
        for reg, v in r["B"].items():
            resid.append({**common, "region": reg, **v})
        if "C" in r:
            row = {**common, **r["C"]}
            for reg, v in r["D"].items():
                row.update({f"D_{reg}.{k}": x for k, x in v.items()})
            grad.append(row)
            row = {**common, **{f"FULL.{k}": v for k, v in r["E"]["FULL"].items()}}
            for i in range(1, 5):
                row.update({f"W{i}.{k}": v for k, v in r["E"][f"W{i}"].items()})
            cond.append(row)
    write_csv(base / "B2-E04_DIAGNOSTICS.csv", diag)
    write_csv(base / "B2-E04_MODAL_SPECTRUM.csv", modal)
    write_csv(base / "B2-E04_RESIDUAL_SPECTRUM.csv", resid)
    write_csv(base / "B2-E04_GRADIENT_CONFLICT.csv", grad)
    write_csv(base / "B2-E04_CONDITIONING.csv", cond)
    return diag


def summaries(U):
    """Section 9: per (trajectory, level) mean / std / min / max over seeds of the unified record."""
    out = {}
    keys = {"L2_exact": ("frozen", "L2_exact"), "collapse_time_s": ("frozen", "collapse_time_s"),
            "rho_1_PRE": ("A", "PRE", "rho_1"), "rho_nonfund_PRE": ("A", "PRE", "rho_nonfund"),
            "rho_nonfund_POST": ("A", "POST", "rho_nonfund"), "h_r_PRE": ("B", "PRE", "h_r"),
            "h_r_FULL": ("B", "FULL", "h_r"), "centroid_FULL": ("B", "FULL", "centroid"), "minC": ("C", "minC"),
            "cos_behind_front": ("C", "cos_behind_front"), "starv": ("C", "starv"), "C_st": ("D", "FULL", "C_st"),
            "ratio_space_time": ("D", "FULL", "ratio_space_time"), "chi": ("D", "FULL", "chi"),
            "log10_kappa_hat": ("E", "FULL", "log10_kappa_hat"), "rank_1e-06": ("E", "FULL", "rank_1e-06")}
    for tr in ("FF-B1", "FF-LBFGS", "MO-ADAM", "MO-LBFGS", "MX"):
        for i, (lev, _) in enumerate(trajectories(SEEDS[0])[tr]):
            d = {}
            for name, path in keys.items():
                vals = [g(U[dict(trajectories(s)[tr])[lev]], *path) for s in SEEDS]
                vals = [float(v) for v in vals if isinstance(v, (int, float)) and not math.isnan(float(v))]
                if vals:
                    d[name] = {"per_seed": vals, "mean": float(np.mean(vals)),
                               "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
                               "min": float(np.min(vals)), "max": float(np.max(vals))}
            out[f"{tr}@{lev}"] = d
    return out


# ============================================================================================ figures
def figures(U, C):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = assert_safe_output(FIG, strict=True, purpose="B2-E04 figures")
    fig_dir.mkdir(parents=True, exist_ok=False)
    COL = {"FF-B1": "#2a78d6", "FF-LBFGS": "#eb6834", "MO-ADAM": "#1baf7a", "MO-LBFGS": "#7a4fd0", "MX": "#888888"}
    LS = {1234: "-", 1235: "--", 1236: ":"}
    MODE_C = plt.cm.viridis(np.linspace(0, 0.9, N_MODES))

    def save(fig, name):
        fig.tight_layout()
        p = fig_dir / name
        assert not p.exists()
        fig.savefig(p, dpi=130)
        plt.close(fig)

    def arr(key):
        return np.load(UNITS / f"{key}.npz")

    def ev(tr, s):
        return [(LEVELS[l] / 1e3 if l != "623k" else 623.36, U[k]) for l, k in trajectories(s)[tr]]

    def prog(getter, ylab, name, trs=("FF-B1", "FF-LBFGS", "MO-ADAM", "MO-LBFGS"), logy=False, hline=None):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for tr in trs:
            for s in SEEDS:
                pts = [(e, getter(r)) for e, r in ev(tr, s)]
                pts = [(e, v) for e, v in pts if isinstance(v, (int, float)) and not math.isnan(float(v))]
                if pts:
                    ax.plot(*zip(*pts), LS[s], marker="o", ms=3, color=COL[tr], label=f"{tr} s{s}")
        if hline is not None:
            ax.axhline(hline, color="k", lw=0.7, ls="-.")
        ax.axvline(320, color="0.6", lw=0.7)
        ax.set_xlabel("PDE evaluations [thousands] (switch at 320k)"); ax.set_ylabel(ylab)
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=6, ncol=2); save(fig, name)

    # 1 / Plot A: modal coefficients a_1..a_8 (s1234; B1 and transfer at 623/640k, modal LBFGS and M0)
    for s in SEEDS:
        sel = [("FF-B1", f"FF_b1-640k_s{s}"), ("FF-LBFGS", f"FF_tr-623k_s{s}"), ("MO-ADAM", f"MO_m0-640k_s{s}"),
               ("MO-LBFGS", f"MO_ml-623k_s{s}")]
        fig, axs = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
        for ax, (tr, key) in zip(axs, sel):
            a = arr(key)
            ax.plot(a["A_t"], a["A_a1ex"], color="k", lw=0.6, label="A0 q_exact")
            for n in range(N_MODES):
                ax.plot(a["A_t"], a["A_a"][n], color=MODE_C[n], lw=0.8 if n else 1.0, label=f"a_{n+1}")
            ax.axvline(U[key]["frozen"]["collapse_time_s"], color="r", lw=0.8, ls="--")
            ax.set_title(f"{tr} {key}  (red: frozen collapse time)", fontsize=8); ax.set_ylabel("a_n(t) [m]")
        axs[0].legend(fontsize=6, ncol=5); axs[-1].set_xlabel("t [s]")
        save(fig, f"B2-E04_01_modal_coefficients_s{s}.png")
    # 2 modal energy distribution (POST and PRE nonfundamental) vs progress
    prog(lambda r: g(r, "A", "PRE", "rho_nonfund"), "non-fundamental energy / exact energy, behind front (PRE)",
         "B2-E04_02a_modal_energy_nonfund_pre.png", trs=("FF-B1", "FF-LBFGS", "MX"), logy=True, hline=THR["A_nonfund_pre"])
    prog(lambda r: g(r, "A", "POST", "rho_1"), "first-mode energy retention, ahead of front (POST)",
         "B2-E04_02b_modal_energy_rho1_post.png", trs=("FF-B1", "FF-LBFGS", "MO-ADAM", "MX"))
    prog(lambda r: g(r, "A", "POST", "rho_nonfund"), "non-fundamental energy / exact energy, ahead of front (POST)",
         "B2-E04_02c_modal_energy_nonfund_post.png", trs=("FF-B1", "FF-LBFGS", "MX"), logy=True)
    # 3 residual spectrum by mode (bar chart, FULL and PRE) at end states
    for reg in ("FULL", "PRE"):
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
        for ax, s in zip(axs, SEEDS):
            keys = [(f"FF_sw-320k_s{s}", "switch 320k"), (f"FF_b1-640k_s{s}", "B1 640k"), (f"FF_tr-623k_s{s}", "L-BFGS 623k"),
                    (f"MX_mx-640k_s{s}", "mixed 640k")]
            wdt = 0.2
            for j, (k, lab) in enumerate(keys):
                v = [g(U[k], "B", reg, f"sig_{n}") for n in range(1, N_MODES + 1)] + [g(U[k], "B", reg, "sig_rem")]
                ax.bar(np.arange(N_MODES + 1) + (j - 1.5) * wdt, v, wdt, label=lab)
            ax.set_xticks(range(N_MODES + 1)); ax.set_xticklabels([str(n) for n in range(1, N_MODES + 1)] + ["rem"])
            ax.set_yscale("log"); ax.set_title(f"s{s} residual energy fraction by mode ({reg})", fontsize=8)
        axs[0].legend(fontsize=7); save(fig, f"B2-E04_03_residual_spectrum_{reg}.png")
    prog(lambda r: g(r, "B", "PRE", "h_r"), "non-fundamental residual share behind front (h_r PRE)",
         "B2-E04_03b_residual_hr_pre.png", trs=("FF-B1", "FF-LBFGS", "MX"), hline=THR["B_hr_pre"])
    # 4 / Plot B: temporal-window cosine matrices
    for s in SEEDS:
        sel = [f"FF_sw-320k_s{s}", f"FF_b1-512k_s{s}", f"FF_tr-512k_s{s}", f"MO_m0-512k_s{s}", f"MO_ml-512k_s{s}"]
        fig, axs = plt.subplots(1, len(sel), figsize=(15, 3.4))
        for ax, k in zip(axs, sel):
            im = ax.imshow(arr(k)["C_matrix"], vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set_xticks(range(4)); ax.set_yticks(range(4))
            ax.set_xticklabels(["W1", "W2", "W3", "W4"]); ax.set_yticklabels(["W1", "W2", "W3", "W4"])
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, f"{arr(k)['C_matrix'][i, j]:.2f}", ha="center", va="center", fontsize=7)
            ax.set_title(k, fontsize=7)
        fig.colorbar(im, ax=axs, shrink=0.8); save(fig, f"B2-E04_04_window_cosine_s{s}.png")
    prog(lambda r: g(r, "C", "minC"), "min pairwise window-gradient cosine", "B2-E04_04b_minC.png", hline=THR["B_minC"])
    prog(lambda r: g(r, "C", "cos_behind_front"), "cos(g_behind, g_front)", "B2-E04_04c_cos_behind_front.png",
         hline=THR["B_cos_front"])
    # 5 temporal gradient norms
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, tr in zip(axs, ("FF-B1", "FF-LBFGS", "MO-ADAM", "MO-LBFGS")):
        for i, c in zip(range(1, 5), MODE_C[::2]):
            for s in SEEDS:
                pts = [(e, g(r, "C", f"n_g{i}")) for e, r in ev(tr, s)]
                ax.plot(*zip(*pts), LS[s], color=c, marker="o", ms=2, label=f"W{i}" if s == 1234 else None)
        ax.set_yscale("log"); ax.set_title(tr, fontsize=8); ax.axvline(320, color="0.6", lw=0.7)
        ax.set_xlabel("PDE evaluations [k]")
    axs[0].set_ylabel("||grad L_W||"); axs[0].legend(fontsize=7); save(fig, "B2-E04_05_window_gradient_norms.png")
    prog(lambda r: g(r, "C", "starv"), "||g_W4|| / ||g_W1||", "B2-E04_05b_starvation.png", logy=True, hline=THR["B_starv"])
    # 6 / Plot C: space/time gradient cosine; 7 norm ratio
    prog(lambda r: g(r, "D", "FULL", "C_st"), "C_st = cos(grad L_time, grad L_space)", "B2-E04_06_Cst.png", hline=THR["C_cst"])
    prog(lambda r: g(r, "D", "FULL", "C_TS"), "cos(g_T, g_S) (decomposition of the training gradient)", "B2-E04_06b_CTS.png")
    prog(lambda r: g(r, "D", "FULL", "ratio_space_time"), "||grad L_space|| / ||grad L_time||", "B2-E04_07_space_time_ratio.png",
         logy=True, hline=THR["C_ratio"])
    prog(lambda r: g(r, "D", "FULL", "chi"), "cancellation index chi", "B2-E04_07b_chi.png", logy=True)
    # 8 Jacobian singular spectrum; 9 / Plot D condition trajectory; 10 condition vs L2
    for s in SEEDS:
        fig, axs = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
        for ax, tr in zip(axs, ("FF-B1", "FF-LBFGS", "MO-ADAM", "MO-LBFGS")):
            lst = trajectories(s)[tr]
            cm = plt.cm.plasma(np.linspace(0, 0.85, len(lst)))
            for (lev, k), c in zip(lst, cm):
                sv = arr(k)["E_s"]
                ax.semilogy(np.arange(1, len(sv) + 1), sv / sv[0], color=c, lw=0.9, label=lev)
            ax.axhline(TAU_RES, color="k", lw=0.6, ls=":")
            ax.set_title(f"{tr} s{s}", fontsize=8); ax.set_xlabel("index i")
        axs[0].set_ylabel("sigma_i / sigma_1"); axs[0].legend(fontsize=6); save(fig, f"B2-E04_08_jacobian_spectrum_s{s}.png")
    prog(lambda r: g(r, "E", "FULL", "log10_kappa_hat"), "log10 estimated residual-Jacobian spectral condition",
         "B2-E04_09_condition_trajectory.png")
    prog(lambda r: g(r, "E", "FULL", "rank_1e-06"), "numerical rank r(1e-6) of J (640 rows)", "B2-E04_09b_rank.png")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for tr in ("FF-B1", "FF-LBFGS", "MO-ADAM", "MO-LBFGS"):
        for s in SEEDS:
            pts = [(g(r, "E", "FULL", "log10_kappa_hat"), r["frozen"]["L2_exact"]) for _, r in ev(tr, s)]
            ax.plot(*zip(*pts), LS[s], marker="o", ms=3, color=COL[tr], label=f"{tr} s{s}")
    ax.set_xlabel("log10 kappa_hat"); ax.set_ylabel("L2_exact"); ax.legend(fontsize=6); save(fig, "B2-E04_10_condition_vs_L2.png")
    # 11 diagnostics vs PDE evaluations (panel)
    panel = [("L2_exact", lambda r: r["frozen"]["L2_exact"]), ("t_c [s]", lambda r: r["frozen"]["collapse_time_s"]),
             ("h_r PRE", lambda r: g(r, "B", "PRE", "h_r")), ("minC", lambda r: g(r, "C", "minC")),
             ("C_st", lambda r: g(r, "D", "FULL", "C_st")), ("log10 kappa", lambda r: g(r, "E", "FULL", "log10_kappa_hat"))]
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (lab, get) in zip(axs.ravel(), panel):
        for tr in ("FF-B1", "FF-LBFGS", "MO-ADAM", "MO-LBFGS"):
            for s in SEEDS:
                pts = [(e, get(r)) for e, r in ev(tr, s)]
                pts = [(e, v) for e, v in pts if not (isinstance(v, float) and math.isnan(v))]
                if pts:
                    ax.plot(*zip(*pts), LS[s], marker="o", ms=2, color=COL[tr], label=f"{tr}" if s == 1234 else None)
        ax.set_title(lab, fontsize=9); ax.axvline(320, color="0.6", lw=0.7)
    axs[0, 0].legend(fontsize=7); save(fig, "B2-E04_11_diagnostics_vs_evaluations.png")
    # 12 collapse front vs diagnostics
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (lab, get) in zip(axs, panel[2:]):
        for tr in ("FF-B1", "FF-LBFGS", "MO-ADAM", "MO-LBFGS"):
            for s in SEEDS:
                pts = [(get(r), r["frozen"]["collapse_time_s"]) for _, r in ev(tr, s)]
                pts = [p for p in pts if not (isinstance(p[0], float) and math.isnan(p[0]))]
                if pts:
                    ax.plot(*zip(*pts), LS[s], marker="o", ms=3, color=COL[tr], label=tr if s == 1234 else None)
        ax.set_xlabel(lab); ax.set_ylabel("collapse time t_c [s]")
    axs[0].legend(fontsize=7); save(fig, "B2-E04_12_collapse_front_vs_diagnostics.png")
    # 13 modal vs full-field comparison (paired post-switch differences)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (mech, (get, sgn, mk)) in zip(axs, [(m, PRIMARY[m][0]) for m in "ABCD"]):
        for s in SEEDS:
            T = {k: dict(v) for k, v in trajectories(s).items()}
            xs = [384, 512, 640]
            df = [get(U[T["FF-LBFGS"][l if l != "640k" else "623k"]]) - get(U[T["FF-B1"][l]]) for l in ("384k", "512k", "640k")]
            dm = [get(U[T["MO-LBFGS"][l if l != "640k" else "623k"]]) - get(U[T["MO-ADAM"][l]]) for l in ("384k", "512k", "640k")]
            ax.plot(xs, df, LS[s], marker="o", color=COL["FF-LBFGS"], label="full: LBFGS - B1" if s == 1234 else None)
            ax.plot(xs, dm, LS[s], marker="s", color=COL["MO-LBFGS"], label="modal: LBFGS - Adam" if s == 1234 else None)
        ax.axhline(0, color="k", lw=0.6); ax.set_title(f"mechanism {mech}: paired difference (pathology sign {sgn:+d})", fontsize=8)
        ax.set_xlabel("PDE evaluations [k]")
    axs[0].legend(fontsize=7); save(fig, "B2-E04_13_modal_vs_fullfield_paired.png")
    # 14 seed-to-seed variability (mean +- range at the end states)
    names = ["L2_exact", "collapse_time_s", "h_r_PRE", "minC", "C_st", "log10_kappa_hat"]
    S = summaries(U)
    fig, axs = plt.subplots(1, len(names), figsize=(17, 3.6))
    trs = [("FF-B1", "640k"), ("FF-LBFGS", "623k"), ("MO-ADAM", "640k"), ("MO-LBFGS", "623k")]
    for ax, nm in zip(axs, names):
        for i, (tr, lev) in enumerate(trs):
            d = S[f"{tr}@{lev}"].get(nm)
            if d:
                ax.errorbar(i, d["mean"], yerr=[[d["mean"] - d["min"]], [d["max"] - d["mean"]]], fmt="o", color=COL[tr])
                ax.scatter([i] * len(d["per_seed"]), d["per_seed"], s=8, color=COL[tr], alpha=0.6)
        ax.set_xticks(range(len(trs))); ax.set_xticklabels([t for t, _ in trs], rotation=30, fontsize=7); ax.set_title(nm, fontsize=9)
    save(fig, "B2-E04_14_seed_variability.png")


# ============================================================================================== main
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    assert_safe_output(OUT, strict=True, purpose="B2-E04")
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "manifest":
        p = OUT / f"checkpoint_manifest_{sys.argv[2]}.json"
        assert not p.exists(), p
        m = manifest(sys.argv[2])
        if sys.argv[2] == "before":
            m["integrity"] = integrity()
        json.dump(m, open(p, "w"), indent=1)
        print(f"{len(m['files'])} files -> {p}")
    elif cmd == "dryrun":
        dryrun()
    elif cmd == "unit":
        UNITS.mkdir(parents=True, exist_ok=True)
        key = sys.argv[2]
        rec, _ = compute_unit(key, all_states()[key])
        print(key, json.dumps(rec["timing_s"]), f"rss {rec['peak_rss_mb']:.0f} MB", flush=True)
    elif cmd == "run":
        UNITS.mkdir(parents=True, exist_ok=True)
        nw = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 3
        todo = [k for k in all_states() if not unit_paths(k)[0].exists()]
        todo.sort(key=lambda k: (k.split("_")[0] != "FF", k))          # expensive full-field states first
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        procs = []
        log = open(OUT / "run.log", "a")
        while todo or procs:
            while todo and len(procs) < nw:
                k = todo.pop(0)
                procs.append((k, subprocess.Popen([sys.executable, __file__, "unit", k], stdout=log, stderr=log, env=env)))
            time.sleep(5)
            for k, p in list(procs):
                if p.poll() is not None:
                    procs.remove((k, p))
                    if p.returncode != 0:
                        print(f"FAILED {k} rc={p.returncode}", file=log, flush=True)
        log.close()
    elif cmd == "aggregate":
        U = load_units()
        assert set(U) == set(all_states()), f"missing units: {sorted(set(all_states()) - set(U))}"
        n = cross_check(U)
        base = REPO / "results_batch2"
        tables(U, base)
        C = classify(U)
        C["frozen_metric_crosschecks_passed"] = n
        C["summaries"] = summaries(U)
        p = base / "B2-E04_CLASSIFICATION.json"
        assert not p.exists()
        json.dump(C, open(p, "w"), indent=1, default=str)
        figures(U, C)
        print(json.dumps({k: C[k] for k in ("CASE", "supported", "mechanisms")}, indent=1, default=str))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
