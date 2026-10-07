"""B2-E01 analysis: evaluation of every arm's snapshots and final model with ONE procedure, the
pre-registered classification (docs/hypotheses/B2-E01.md section 7) computed in code, the results
table and the figures. Evaluation only - no training.

    python scripts/analyze_b2_e01.py

Writes (never overwrites):
  results_batch2/tables/B2-E01_RESULTS.csv      one row per arm (final model)
  results_batch2/tables/B2-E01_SNAPSHOTS.csv    one row per arm x snapshot (1k..5k steps)
  results_batch2/tables/B2-E01_CLASSIFICATION.json
  results_batch2/figures/B2-E01_*.png
"""
import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from beampinn.config import ExperimentConfig  # noqa: E402
from beampinn.evaluation.metrics import grid, l2_pair, predict, spectral_metrics  # noqa: E402
from beampinn.models.networks import build_model  # noqa: E402
from beampinn.training.trainer import DTYPES, build_hard, resolve_problem  # noqa: E402
from physref.formulations.mixed import DisplacementView, MixedHardFF, TwoHeadFourierPINN, mixed_residuals  # noqa: E402
from physref.formulations.modal import ModalField, ModalHardQ, TemporalFourierNet  # noqa: E402
from physref.persistence import passes_persistence_gate, persistence  # noqa: E402
from physref.safety import assert_safe_output  # noqa: E402

RUNS = REPO / "results_batch2" / "runs"
ARMS = [("B1", "B2-E01-B1-s1234"), ("B1_fp64", "B2-E01-B1_fp64-s1234"),
        ("mixed", "B2-E01-mixed-s1234"), ("modal", "B2-E01-modal-s1234")]
LABEL = {"B1": "B1 (FP32, reference)", "B1_fp64": "B1 FP64 (precision control)",
         "mixed": "Mixed v = u_xx", "modal": "Modal q(t) (diagnostic)"}
COLOR = {"B1": "#2a78d6", "B1_fp64": "#eb6834", "mixed": "#1baf7a", "modal": "#eda100"}   # slots 1-4, fixed order
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
BATCH1_Z4_5K = {"L2_exact": 0.5263868153173645, "persistence_cycles": 3.1542914950274312,
                "persistence_cycles_vel": 3.360118019988438, "PDE_residual_rel": 0.11022354745212025,
                "fit_w": 128.07802380678396, "fit_decay": 8.981398750079364}
STEPS = (1000, 2000, 3000, 4000, 5000)


# ------------------------------------------------------------------ loading
def paths(arm, eid):
    d = RUNS / eid
    if arm in ("B1", "B1_fp64"):
        rid = next((d / "checkpoints").iterdir()).name
        return d / "checkpoints" / rid, d / "logs" / rid
    return d / "checkpoints", d / "logs"


def build(arm, cfg):
    dt = DTYPES[cfg.precision]
    torch.set_default_dtype(dt)
    bm, refs, c2, g, _ = resolve_problem(cfg)
    ex, w1 = refs["exact"], bm.fundamental_omega(cfg.benchmark.pde_coeffs)
    m = cfg.model
    if arm in ("B1", "B1_fp64"):
        model = build_hard(build_model(cfg, bm), cfg, bm, refs).to(dt)
        return model, model, (bm, refs, c2, g)
    if arm == "mixed":
        model = MixedHardFF(TwoHeadFourierPINN(m, bm.L, bm.t_end, cfg.seed), ex, bm.L, bm.t_end, "tanh2", w1).to(dt)
        return model, DisplacementView(model), (bm, refs, c2, g)
    q = ModalHardQ(TemporalFourierNet(bm.t_end, m.m_fourier, m.sigma_t, m.depth, m.width, cfg.seed, m.two_pi,
                                      burn_spatial_sigmas=m.sigma_x), 1.0, bm.t_end, "tanh2", w1).to(dt)
    return q, ModalField(q, ex).to(dt), (bm, refs, c2, g)


def load(arm, ckpt_file):
    blob = torch.load(ckpt_file, weights_only=False)
    cfg = ExperimentConfig.from_dict(blob["config"])
    model, view, prob = build(arm, cfg)
    model.load_state_dict(blob["model"])
    model.eval()
    return cfg, model, view, prob, blob


# --------------------------------------------------------------- evaluation
def snapshot_metrics(view, prob):
    bm, refs, c2, g = prob
    _, _, X, T = grid(bm.L, bm.t_end, 201, 2001)
    out = l2_pair(predict(view, X, T), refs, X, T)
    pers, (t, up, vp, ue, ve, R, Rv) = persistence(view, refs["exact"], bm.t_end)
    out.update(pers)
    out["amplitude_ratio"] = float(up.std() / ue.std())
    out["max_ut_ratio"] = float(np.abs(vp).max() / np.abs(ve).max())
    sp = spectral_metrics(view, bm, refs)
    out.update(fit_w=sp["fit_w"], fit_decay=sp["fit_lam"], frequency_error_exact=sp["frequency_error_exact"],
               phase_error_exact=sp["phase_error_exact"], amplitude_error_exact=sp["amplitude_error_exact"])
    return out, (t, up, vp, ue, ve)


def link_residual_rel(model, prob):
    bm, refs, c2, g = prob
    ex = refs["exact"]
    _, _, X, T = grid(bm.L, bm.t_end, 51, 501)
    dt = next(model.parameters()).dtype
    xs = torch.from_numpy(X.reshape(-1, 1)).to(dt); ts = torch.from_numpy(T.reshape(-1, 1)).to(dt)
    rl, rp = [], []
    for i in range(0, len(xs), 1024):
        a, b = mixed_residuals(model, xs[i:i + 1024], ts[i:i + 1024], c2, g, ex.beta)
        rl.append(a.detach().double()); rp.append(b.detach().double())
    rl = torch.cat(rl).numpy(); rp = torch.cat(rp).numpy()
    uxx = ex.u(X, T, 2, 0).ravel()
    return {"link_residual_rel": float(np.sqrt(np.mean((rl / (c2 * ex.beta ** 2)) ** 2)) / np.sqrt(np.mean(uxx ** 2))),
            "mixed_pde_residual_rel": float(np.sqrt(np.mean(rp ** 2)) / np.sqrt(np.mean(ex.u(X, T, 0, 2) ** 2)))}


def read_history(logdir):
    with open(logdir / "history.csv") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------ classification
def classify(x, b):
    """docs/hypotheses/B2-E01.md section 7.1 (dynamics first)."""
    full = x["persistence_cycles"] >= x["full_window_cycles"] * (1 - 1e-12) and \
        x["persistence_cycles_vel"] >= x["full_window_cycles"] * (1 - 1e-12)
    dP = x["persistence_cycles"] - b["persistence_cycles"]
    rL = x["L2_exact"] / b["L2_exact"]
    if full:
        cls = "FULL-WINDOW"
    elif dP >= 1.0 and rL <= 1.0:
        cls = "BETTER"
    elif abs(dP) < 1.0 and rL <= 1.25:
        cls = "COMPARABLE"
    elif dP <= -1.0 and rL >= 1.0:
        cls = "WORSE"
    elif abs(dP) < 1.0 and rL > 1.25:
        cls = "WORSE"
    else:
        cls = "TRADE-OFF"
    dyn_ok = (abs(x["fit_w"] / b["omega_d"] - 1) <= 0.02 and 0.5 <= x["amplitude_ratio"] <= 2
              and 0.5 <= x["max_ut_ratio"] <= 2)
    if cls in ("BETTER", "FULL-WINDOW") and not dyn_ok:
        cls += " (blocked: dynamics diagnostics violated)"
    return cls, dP, rL, dyn_ok


def decide(R):
    b = R["B1"]
    out = {}
    g = b
    gate = (abs(g["L2_exact"] / BATCH1_Z4_5K["L2_exact"] - 1) <= 0.05
            and abs(g["persistence_cycles"] - BATCH1_Z4_5K["persistence_cycles"]) <= 0.5
            and g["run_key_ok"])
    out["reproducibility_gate"] = {"pass": bool(gate),
                                   "dL2e_rel": g["L2_exact"] / BATCH1_Z4_5K["L2_exact"] - 1,
                                   "dP_cycles": g["persistence_cycles"] - BATCH1_Z4_5K["persistence_cycles"],
                                   "run_key_ok": g["run_key_ok"]}
    for arm in ("B1_fp64", "mixed", "modal"):
        cls, dP, rL, dyn = classify(R[arm], b)
        out[arm] = {"class": cls, "dP_cycles": dP, "L2e_ratio": rL, "dynamics_ok": dyn,
                    "W_ratio": R[arm]["train_seconds"] / b["train_seconds"]}
    m = out["mixed"]
    cheap = m["W_ratio"] <= 0.75
    if cheap and m["class"].split(" ")[0] in ("FULL-WINDOW", "BETTER", "COMPARABLE"):
        m["case"] = "A"
    elif cheap:
        m["case"] = "B"
    else:
        m["case"] = "no material compute gain"
    md, Pb = R["modal"], b["persistence_cycles"]
    if (md["persistence_cycles"] >= 2 * Pb or out["modal"]["class"].startswith("FULL")) and md["L2_exact"] <= 0.5 * b["L2_exact"]:
        out["modal"]["case"] = "C"
    elif md["persistence_cycles"] <= 1.5 * Pb:
        out["modal"]["case"] = "H2-support"
    else:
        out["modal"]["case"] = "partial"
    f = R["B1_fp64"]
    qual = (((abs(f["frequency_error_exact"]) > 0.10) != (abs(b["frequency_error_exact"]) > 0.10))
            or ((f["amplitude_ratio"] < 0.2) != (b["amplitude_ratio"] < 0.2)))     # oscillatory <-> static
    material = abs(f["persistence_cycles"] - Pb) >= 1.0 or abs(f["L2_exact"] / b["L2_exact"] - 1) >= 0.25 or qual
    dramatic = f["persistence_cycles"] >= 2 * Pb or out["B1_fp64"]["class"].startswith("FULL")
    out["B1_fp64"]["case"] = ("D (dramatic: separate mechanism)" if dramatic else "D") if material else "not material"
    similar = all(abs(R[a]["persistence_cycles"] - Pb) < 1.0 and abs(R[a]["L2_exact"] / b["L2_exact"] - 1) < 0.25
                  and not out[a]["class"].startswith("FULL") for a in ("B1_fp64", "mixed", "modal"))
    out["case_E_all_similar"] = bool(similar)
    return out


# ------------------------------------------------------------------ figures
def style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10); ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11, loc="left")


def figures(R, S, H, traces, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9})
    arms = [a for a, _ in ARMS]

    def save(fig, name):
        p = fig_dir / f"B2-E01_{name}.png"
        if p.exists():
            raise FileExistsError(p)
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)

    # 1 L2 vs PDE evaluations
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for a in arms:
        xs = [r["pde_evaluations"] for r in S[a]]; ys = [r["L2_exact"] for r in S[a]]
        ax.plot(xs, ys, color=COLOR[a], lw=2, marker="o", ms=5, mec=SURF, mew=1.5, label=LABEL[a])
        ax.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", color=INK2, fontsize=8, va="center")
    ax.set_yscale("log")
    style(ax, "PDE (collocation) evaluations", "relative L2 vs exact reference", "L2_exact at snapshots (201x2001 grid)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    save(fig, "L2_vs_pde_evals")

    # 2 runtime vs PDE evaluations
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for a in arms:
        h = [r for r in H[a] if r.get("train_seconds") not in (None, "") and r.get("pde_evaluations") not in (None, "")]
        xs = [float(r["pde_evaluations"]) for r in h]; ys = [float(r["train_seconds"]) for r in h]
        ax.plot(xs, ys, color=COLOR[a], lw=2, label=LABEL[a])
        ax.annotate(f"{ys[-1]:.0f} s", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", color=INK2, fontsize=8, va="center")
    style(ax, "PDE (collocation) evaluations", "cumulative training seconds", "Training wall-clock vs PDE evaluations (4 concurrent 1-thread runs)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    save(fig, "runtime_vs_pde_evals")

    # 3 convergence curves (small multiples: loss definitions differ per formulation)
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for ax, a in zip(axs.ravel(), arms):
        h = [r for r in H[a] if r.get("loss") not in (None, "")]
        xs = [float(r["pde_evaluations"]) for r in h]
        for key, ls in (("loss", "-"),):
            ax.plot(xs, [float(r[key]) for r in h], color=COLOR[a], lw=2, ls=ls)
        ax.set_yscale("log")
        style(ax, "PDE evaluations", "training loss", LABEL[a])
    fig.suptitle("Training-loss convergence (each panel its own loss definition; not comparable across panels)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "convergence")

    # 4/5 displacement and velocity traces (small multiples)
    for kind, idx_p, idx_e, unit, name in (("displacement", 1, 3, "u(L/2, t) [m]", "displacement_trace"),
                                           ("velocity", 2, 4, "u_t(L/2, t) [m/s]", "velocity_trace")):
        fig, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
        for ax, a in zip(axs, arms):
            t, *tr = traces[a]
            arr = [t] + tr
            ax.plot(t, arr[idx_e], color=INK, lw=1, label="exact")
            ax.plot(t, arr[idx_p], color=COLOR[a], lw=1.6, label=LABEL[a])
            key = "collapse_time_s" if kind == "displacement" else "collapse_time_vel_s"
            tc = R[a][key]
            if tc < t[-1]:
                ax.axvline(tc, color=MUTED, lw=1)
                ax.annotate(f"collapse {tc:.3f} s", (tc, ax.get_ylim()[1]), xytext=(4, -10), textcoords="offset points", color=INK2, fontsize=8)
            style(ax, "t [s]" if a == arms[-1] else "", unit, f"{LABEL[a]} - {kind} at mid-span (final model)")
            ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
        fig.tight_layout()
        save(fig, name)

    # 6 Pareto: accuracy vs compute (two panels, one axis each)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for a in arms:
        hollow = a == "modal"
        for ax, key in ((axs[0], "L2_exact"), (axs[1], "persistence_cycles")):
            ax.scatter(R[a]["train_seconds"], R[a][key], s=70, color=SURF if hollow else COLOR[a],
                       edgecolors=COLOR[a], linewidths=2, zorder=3, label=LABEL[a])
            ax.annotate(a + (" (diag.)" if hollow else ""), (R[a]["train_seconds"], R[a][key]),
                        xytext=(7, 4), textcoords="offset points", color=INK2, fontsize=8)
    axs[0].set_yscale("log")
    style(axs[0], "training seconds (matched 640,000 PDE evaluations)", "L2_exact (lower is better)", "Accuracy vs compute")
    style(axs[1], "training seconds (matched 640,000 PDE evaluations)", "persistence [cycles] (higher is better)", "Persistence vs compute")
    axs[1].axhline(R["B1"]["full_window_cycles"], color=MUTED, lw=1)
    axs[1].annotate("full window 20.58", (axs[1].get_xlim()[0], R["B1"]["full_window_cycles"]), xytext=(4, -12), textcoords="offset points", color=INK2, fontsize=8)
    axs[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    save(fig, "pareto")

    # 7 collapse front vs PDE evaluations
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for a in arms:
        xs = [r["pde_evaluations"] for r in S[a]]; ys = [r["collapse_time_s"] for r in S[a]]
        ax.plot(xs, ys, color=COLOR[a], lw=2, marker="o", ms=5, mec=SURF, mew=1.5, label=LABEL[a])
    ax.axhline(1.0, color=MUTED, lw=1)
    style(ax, "PDE (collocation) evaluations", "collapse time [s] (1.0 = full window)", "Collapse front vs PDE evaluations")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    save(fig, "collapse_front")


# --------------------------------------------------------------------- main
def main():
    tab_dir = assert_safe_output(REPO / "results_batch2" / "tables", strict=True, purpose="B2-E01 analysis")
    fig_dir = assert_safe_output(REPO / "results_batch2" / "figures", strict=True, purpose="B2-E01 analysis")
    torch.set_num_threads(4)
    R, S, H, traces = {}, {}, {}, {}
    for arm, eid in ARMS:
        ck, lg = paths(arm, eid)
        met = json.load(open(lg / "metrics.json"))
        H[arm] = read_history(lg)
        S[arm] = []
        for s in STEPS:
            cfg, model, view, prob, blob = load(arm, ck / f"step_{s}.pt")
            row, _ = snapshot_metrics(view, prob)
            row.update(arm=arm, step=s, pde_evaluations=s * cfg.sampler.mini_batch)
            S[arm].append(row)
            print(f"{arm:8s} {s:5d}: L2e {row['L2_exact']:.4f} P {row['persistence_cycles']:.2f} tc {row['collapse_time_s']:.3f}")
        cfg, model, view, prob, blob = load(arm, ck / "final.pt")
        row, tr = snapshot_metrics(view, prob)
        traces[arm] = tr
        bm, refs, _, _ = prob
        r = {"arm": arm, "experiment_id": eid, "role": LABEL[arm], "precision": cfg.precision,
             "run_key": cfg.run_id() if arm in ("B1", "B1_fp64") else "n/a (arm trainer)",
             "omega_d": refs["exact"].omega_d}
        r["run_key_ok"] = (cfg.run_id() == "Z4_20K__s1234__f78aa7f1da") if arm == "B1" else True
        for k in ("L2_exact", "L2_paper", "L2_late_exact", "RMSE_exact", "PDE_residual_rel", "IC_error_max", "BC_error_max",
                  "parameters", "model_size_bytes", "optimizer_steps", "pde_evaluations", "train_seconds", "peak_rss_mb",
                  "peak_vram_mb", "inference_us_per_point", "inference_single_point_us"):
            r[k] = met.get(k)
        r.update({k: row[k] for k in ("persistence_cycles", "persistence_cycles_vel", "collapse_time_s", "collapse_time_vel_s",
                                      "full_window_cycles", "amplitude_ratio", "max_ut_ratio", "fit_w", "fit_decay",
                                      "frequency_error_exact", "phase_error_exact", "amplitude_error_exact")})
        assert abs(row["L2_exact"] / met["L2_exact"] - 1) < 1e-6, "final re-evaluation does not match the run record"
        r["passes_persistence_gate"] = passes_persistence_gate(row)
        if arm in ("B1", "B1_fp64"):
            r["residual_evaluations"] = met["pde_evaluations"]
            r["forward_passes"] = met["optimizer_steps"]              # one batched residual-graph forward per step
            r["backward_passes"] = met["grad_evaluations"]
            r["diag_backward_passes"] = met["diag_gradients"]         # B1 grad-norm logging (excluded from W)
            r["highest_derivative_order"] = 4
        else:
            r["residual_evaluations"] = met["residual_evaluations"]
            r["forward_passes"] = met["forward_passes"]
            r["backward_passes"] = met["backward_passes"]
            r["diag_backward_passes"] = 0
            r["highest_derivative_order"] = 2
        r["seconds_per_step"] = r["train_seconds"] / r["optimizer_steps"]
        if arm == "mixed":
            r.update(link_residual_rel(model, prob))
        R[arm] = r
    cls = decide(R)
    cls["single_seed_provisional"] = True
    cols = list(R["B1"].keys()) + [k for k in R["mixed"] if k not in R["B1"]]
    p = tab_dir / "B2-E01_RESULTS.csv"
    with open(p, "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["class_vs_B1", "case"]); w.writeheader()
        for arm, _ in ARMS:
            row = dict(R[arm])
            row["class_vs_B1"] = cls.get(arm, {}).get("class", "reference")
            row["case"] = cls.get(arm, {}).get("case", "reproducibility gate " + ("PASS" if cls["reproducibility_gate"]["pass"] else "FAIL"))
            w.writerow(row)
    with open(tab_dir / "B2-E01_SNAPSHOTS.csv", "x", newline="") as f:
        keys = list(S["B1"][0].keys())
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for arm, _ in ARMS:
            w.writerows(S[arm])
    with open(tab_dir / "B2-E01_CLASSIFICATION.json", "x") as f:
        json.dump(cls, f, indent=2, default=float)
    figures(R, S, H, traces, fig_dir)
    print(json.dumps(cls, indent=2, default=float))


if __name__ == "__main__":
    main()
