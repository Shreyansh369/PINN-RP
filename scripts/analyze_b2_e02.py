"""B2-E02 analysis (modal temporal-optimisation screen). Evaluation only - no training.

Implements EXACTLY the metrics, collapse-front definitions, q(t) error decomposition and decision
rules pre-registered in docs/hypotheses/B2-E02.md (sections 6-8). Snapshot evaluation reuses the
frozen B2-E01 evaluator (scripts/analyze_b2_e01.py) and adds, per snapshot, the strong PDE residual
of the physical displacement u (Batch-1 `physics_metrics`, 51x501 grid).

    python scripts/analyze_b2_e02.py

Writes (never overwrites): results_batch2/B2-E02_{RESULTS,SNAPSHOTS}.csv,
results_batch2/B2-E02_CLASSIFICATION.json, results_batch2/figures/B2-E02_*.png
"""
import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import analyze_b2_e01 as A  # noqa: E402  (frozen evaluator)
from beampinn.evaluation.metrics import physics_metrics, predict  # noqa: E402
from physref.persistence import local_amp, passes_persistence_gate, velocity_trace  # noqa: E402
from physref.safety import assert_safe_output  # noqa: E402

RUNS = REPO / "results_batch2" / "runs"
SEEDS = (1234, 1235, 1236)
ARMS = ("M0", "LBFGS", "CAUSAL")
LABEL = {"M0": "M0 control: Adam + decay", "LBFGS": "Adam -> L-BFGS", "CAUSAL": "Adam + causal weighting"}
COLOR = {"M0": "#2a78d6", "LBFGS": "#eb6834", "CAUSAL": "#1baf7a"}     # validated slots 1-3, fixed order
MARK = {1234: "o", 1235: "s", 1236: "^"}
LS = {1234: "-", 1235: "--", 1236: ":"}
DECAY_EXACT = 3.54
BUDGET = 640_000


def eid(arm, seed):
    return f"B2-E02-{arm}-s{seed}"


def ckpts(arm, seed):
    d = RUNS / eid(arm, seed) / "checkpoints"
    snaps = sorted(d.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return snaps, d / "final.pt", RUNS / eid(arm, seed) / "logs"


# ------------------------------------------------------------- q(t) decomposition (section 7)
def damped_cos(t, a, lam, w, phi):
    return a * np.exp(-lam * t) * np.cos(w * t + phi)


def zero_crossings(t, y):
    s = np.sign(y)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    return t[idx] - y[idx] * (t[idx + 1] - t[idx]) / (y[idx + 1] - y[idx])


def decompose(view, ref, T, fit):
    """Pre-registered decomposition of the error of q_theta(t) = u(L/2,t)/A0 vs the exact q(t)."""
    t = np.linspace(0, T, 4001)
    xn = ref.x_norm
    q = predict(view, np.full_like(t, xn)[:, None], t[:, None]).ravel() / ref.A0
    qe = ref.u(xn, t) / ref.A0
    nrm = np.linalg.norm(qe)
    a, lam, w, phi = ref.damped_cosine_params()
    fa, flam, fw, fphi = fit["fit_a"] / ref.A0, fit["fit_lam"], fit["fit_w"], fit["fit_phi"]
    base = damped_cos(t, a, lam, w, phi)
    out = {"L2_q": float(np.linalg.norm(q - qe) / nrm)}
    # one-parameter sensitivity: exact parameters with ONE replaced by its fitted value
    out["err_frequency"] = float(np.linalg.norm(damped_cos(t, a, lam, fw, phi) - base) / nrm)
    out["err_decay"] = float(np.linalg.norm(damped_cos(t, a, flam, w, phi) - base) / nrm)
    out["err_phase"] = float(np.linalg.norm(damped_cos(t, a, lam, w, fphi) - base) / nrm)
    out["err_amplitude"] = float(np.linalg.norm(damped_cos(t, fa, lam, w, phi) - base) / nrm)
    q_fit = damped_cos(t, fa, flam, fw, fphi) + fit.get("fit_offset", 0.0) / ref.A0
    out["err_non_damped_cosine"] = float(np.linalg.norm(q - q_fit) / nrm)      # collapse distortion
    # local (pre-collapse) diagnostics
    P = 2 * np.pi / ref.omega_d
    R = local_amp(q, t, P) / local_amp(qe, t, P)
    ok = t >= P / 2
    below = np.where(ok & (R < 0.5))[0]
    tc = float(t[below[0]]) if len(below) else float(T)
    out["envelope_error_precollapse"] = float(np.mean(np.abs(R[(t >= P / 2) & (t < tc)] - 1))) if tc > P else float("nan")
    zq, ze = zero_crossings(t, q), zero_crossings(t, qe)
    zq, ze = zq[zq < tc], ze[ze < tc]
    n = min(len(zq), len(ze))
    if n >= 3:
        hp_q, hp_e = np.diff(zq[:n]), np.diff(ze[:n])
        out["local_frequency_error_precollapse"] = float(np.median(np.abs(hp_e / hp_q - 1)))
        out["phase_drift_at_collapse_rad"] = float(ref.omega_d * (zq[n - 1] - ze[n - 1]))
    else:
        out["local_frequency_error_precollapse"] = float("nan"); out["phase_drift_at_collapse_rad"] = float("nan")
    m = (t >= P) & (t < tc)
    if m.sum() > 50:
        env = local_amp(q, t, P)
        out["local_decay_precollapse"] = float(-np.polyfit(t[m], np.log(np.maximum(env[m], 1e-12)), 1)[0])
    else:
        out["local_decay_precollapse"] = float("nan")
    return out, (t, q, qe, R)


# ------------------------------------------------------------------ evaluation
def eval_ckpt(path):
    cfg, model, view, prob, blob = A.load("modal", path)
    row, traces = A.snapshot_metrics(view, prob)
    bm, refs, c2, g = prob
    row["PDE_residual_rel"] = physics_metrics(view, bm, refs["exact"], c2, g)["PDE_residual_rel"]
    sp = A.spectral_metrics(view, bm, refs)
    dec, qtr = decompose(view, refs["exact"], bm.t_end, sp)
    row.update(dec)
    row["pde_evaluations"] = int(blob["acc"]["pde_evaluations"])
    row["train_seconds"] = float(blob["acc"]["train_seconds"])
    row["optimizer_steps_or_evals"] = int(blob["acc"]["forward_passes"])
    return row, qtr, blob


def evaluate_run(arm, seed):
    snaps, final, lg = ckpts(arm, seed)
    met = json.load(open(lg / "metrics.json"))
    S = []
    for p in snaps:
        r, _, _ = eval_ckpt(p)
        r.update(arm=arm, seed=seed, checkpoint=p.name)
        S.append(r)
    r, qtr, blob = eval_ckpt(final)
    assert abs(r["L2_exact"] / met["L2_exact"] - 1) < 1e-6, "final re-evaluation does not match the run record"
    acc = blob["acc"]
    F = {"arm": arm, "seed": seed, "experiment_id": eid(arm, seed), **{k: r[k] for k in r}}
    F.update(parameters=met["parameters"], peak_rss_mb=met["peak_rss_mb"], IC_error_max=met["IC_error_max"],
             BC_error_max=met["BC_error_max"], optimizer_steps=acc.get("optimizer_steps"),
             lbfgs_closure_evals=acc.get("lbfgs_closure_evals", 0), switch_pde_evaluations=acc.get("switch_pde_evaluations", ""),
             lbfgs_stop_reason=acc.get("lbfgs_stop_reason", ""), forward_passes=acc["forward_passes"],
             backward_passes=acc["backward_passes"], decay_error=abs(r["fit_decay"] - DECAY_EXACT),
             passes_persistence_gate=passes_persistence_gate(r))
    # snapshots + final model, deduplicated by PDE-evaluation count (section 6.2)
    if all(x["pde_evaluations"] != r["pde_evaluations"] for x in S):
        S.append({**{k: r[k] for k in r}, "arm": arm, "seed": seed, "checkpoint": "final.pt"})
    S.sort(key=lambda x: x["pde_evaluations"])
    # collapse-front metrics (section 6.2)
    tcs = [s["collapse_time_s"] for s in S]
    F["CF_AUC"] = float(np.mean(tcs) / 1.0)
    period = 2 * np.pi / 129.3246197462831
    F["CF_regressions"] = int(sum(1 for a, b in zip(tcs, tcs[1:]) if b < a - period))
    return F, S, qtr


# ------------------------------------------------------------------ decisions (section 8)
def decide(F):
    """Per-seed and all-seed classification of each candidate vs the M0 control."""
    out = {}
    ctrl = {s: F[("M0", s)] for s in SEEDS}
    ctrl_P_range = max(c["persistence_cycles"] for c in ctrl.values()) - min(c["persistence_cycles"] for c in ctrl.values())
    dP_thr = max(1.0, 2 * ctrl_P_range)
    out["thresholds"] = {"dP_cycles": dP_thr, "L2_ratio": 0.9, "W_ratio_eff": 0.75, "freq_err_max": 0.02}
    for arm in ("LBFGS", "CAUSAL"):
        per = {}
        for s in SEEDS:
            c, x = ctrl[s], F[(arm, s)]
            acc_ok = (x["persistence_cycles"] >= c["persistence_cycles"] + dP_thr
                      and x["L2_exact"] <= 0.9 * c["L2_exact"]
                      and abs(x["frequency_error_exact"]) <= 0.02
                      and x["decay_error"] <= c["decay_error"]
                      and x["PDE_residual_rel"] <= 1.1 * c["PDE_residual_rel"])
            worse = (x["persistence_cycles"] <= c["persistence_cycles"] - dP_thr or x["L2_exact"] > 1.1 * c["L2_exact"])
            # efficiency: first snapshot reaching the control's FINAL accuracy and persistence
            reach = [r for r in sorted(SN[(arm, s)], key=lambda r: r["pde_evaluations"])
                     if r["L2_exact"] <= c["L2_exact"] and r["persistence_cycles"] >= c["persistence_cycles"]]
            reach_E = reach[0]["pde_evaluations"] if reach else None
            reach_W = reach[0]["train_seconds"] if reach else None
            eff_ok = bool(reach) and (reach_E <= 0.75 * BUDGET or reach_W <= 0.75 * c["train_seconds"])
            per[s] = {"accuracy_improvement": bool(acc_ok), "worse": bool(worse), "efficiency_improvement": bool(eff_ok),
                      "W_ratio": x["train_seconds"] / c["train_seconds"], "dP": x["persistence_cycles"] - c["persistence_cycles"],
                      "L2_ratio": x["L2_exact"] / c["L2_exact"], "reach_control_E": reach_E, "reach_control_W": reach_W,
                      "dominance": bool(acc_ok and x["train_seconds"] <= c["train_seconds"])}
        n_acc = sum(p["accuracy_improvement"] for p in per.values())
        n_eff = sum(p["efficiency_improvement"] for p in per.values())
        n_dom = sum(p["dominance"] for p in per.values())
        n_worse = sum(p["worse"] for p in per.values())
        if n_dom == 3:
            label = "DOMINANCE"
        elif n_acc == 3 and n_eff < 3:
            label = "ACCURACY-ONLY"
        elif n_eff == 3 and n_acc < 3:
            label = "EFFICIENCY-ONLY"
        elif n_acc == 3 and n_eff == 3:
            label = "ACCURACY+EFFICIENCY"
        elif (n_acc >= 1 or n_eff >= 1) and n_worse >= 1:
            label = "TRADE-OFF / UNSTABLE"
        elif n_acc == 2 or n_eff == 2:
            label = "PARTIAL (2/3 seeds)"
        else:
            label = "NO IMPROVEMENT"
        out[arm] = {"per_seed": per, "label": label, "n_accuracy": n_acc, "n_efficiency": n_eff, "n_dominance": n_dom,
                    "transfer_gate": label in ("DOMINANCE", "ACCURACY-ONLY", "ACCURACY+EFFICIENCY", "EFFICIENCY-ONLY")}
    return out


def mechanism(F, arm):
    """Section 7.3: what does the candidate fix? (seed means of the decomposition terms)."""
    def m(a, k):
        v = [F[(a, s)][k] for s in SEEDS]
        return float(np.nanmean(v))
    fp_c = m("M0", "err_frequency") + m("M0", "err_phase")
    fp_x = m(arm, "err_frequency") + m(arm, "err_phase")
    da_c = m("M0", "err_decay") + m("M0", "err_amplitude")
    da_x = m(arm, "err_decay") + m(arm, "err_amplitude")
    red_fp = 1 - fp_x / fp_c if fp_c else 0.0
    red_da = 1 - da_x / da_c if da_c else 0.0
    sd = lambda a: float(np.std([F[(a, s)]["persistence_cycles"] for s in SEEDS], ddof=1))
    regr = lambda a: sum(F[(a, s)]["CF_regressions"] for s in SEEDS)
    stab = (sd(arm) <= 0.5 * sd("M0")) or (regr(arm) < regr("M0"))
    if max(red_fp, red_da) >= 0.25:
        cat = "A (phase/frequency)" if red_fp >= red_da else "B (amplitude decay)"
    elif stab:
        cat = "C (optimisation stability)"
    else:
        cat = "D (none)"
    return {"reduction_freq_phase": red_fp, "reduction_decay_amplitude": red_da, "P_std_ctrl": sd("M0"),
            "P_std_arm": sd(arm), "regressions_ctrl": regr("M0"), "regressions_arm": regr(arm), "category": cat}


# ------------------------------------------------------------------ figures
def figures(F, SNX, QTR, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": A.SURF, "savefig.facecolor": A.SURF, "font.size": 9})

    def save(fig, name):
        p = fig_dir / f"B2-E02_{name}.png"
        if p.exists():
            raise FileExistsError(p)
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)

    def series(key, ylab, name, logy=False, title=""):
        fig, ax = plt.subplots(figsize=(7.5, 4.3))
        for arm in ARMS:
            for s in SEEDS:
                pts = sorted(SNX[(arm, s)], key=lambda r: r["pde_evaluations"])
                ax.plot([r["pde_evaluations"] for r in pts], [r[key] for r in pts], color=COLOR[arm], lw=2, ls=LS[s],
                        marker=MARK[s], ms=5, mec=A.SURF, mew=1.2, label=f"{LABEL[arm]}, seed {s}")
        if logy:
            ax.set_yscale("log")
        A.style(ax, "PDE (collocation) evaluations", ylab, title)
        ax.legend(frameon=False, fontsize=6.5, labelcolor=A.INK2, ncol=1, loc="best")
        save(fig, name)

    series("L2_exact", "L2_exact", "L2_vs_progress", True, "L2_exact vs training progress (all snapshots, 3 seeds)")
    series("collapse_time_s", "collapse time [s] (1.0 = full window)", "collapse_front", False, "Collapse front vs PDE evaluations")
    series("frequency_error_exact", "fitted frequency error (relative)", "frequency_error", False, "Frequency error vs PDE evaluations")
    series("PDE_residual_rel", "PDE residual of u (relative)", "pde_residual", True, "Strong PDE residual of u vs PDE evaluations")
    series("fit_decay", "fitted decay rate [1/s] (exact 3.54)", "decay_rate", False, "Fitted decay rate vs PDE evaluations")

    # q(t) and envelope: seed 1234, all arms
    fig, axs = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    for ax, arm in zip(axs, ARMS):
        t, q, qe, R = QTR[(arm, 1234)]
        ax.plot(t, qe, color=A.INK, lw=1, label="exact q(t)")
        ax.plot(t, q, color=COLOR[arm], lw=1.6, label=f"{LABEL[arm]} (seed 1234)")
        A.style(ax, "t [s]" if arm == ARMS[-1] else "", "q(t)", f"{LABEL[arm]}: learned temporal coefficient")
        ax.legend(frameon=False, fontsize=8, labelcolor=A.INK2, loc="upper right")
    fig.tight_layout(); save(fig, "q_of_t")

    fig, ax = plt.subplots(figsize=(8, 4.3))
    for arm in ARMS:
        for s in SEEDS:
            t, q, qe, R = QTR[(arm, s)]
            ax.plot(t, R, color=COLOR[arm], lw=1.5, ls=LS[s], label=f"{LABEL[arm]}, seed {s}")
    ax.axhline(1.0, color=A.MUTED, lw=1); ax.axhline(0.5, color=A.MUTED, lw=1)
    ax.annotate("collapse threshold 0.5", (0.0, 0.5), xytext=(4, -12), textcoords="offset points", color=A.INK2, fontsize=8)
    A.style(ax, "t [s]", "local amplitude ratio A_pred/A_exact", "Amplitude envelope (final models)")
    ax.set_ylim(0, 1.4)
    ax.legend(frameon=False, fontsize=6.5, labelcolor=A.INK2)
    save(fig, "envelope")

    # wall-clock vs accuracy (snapshots) and final Pareto
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for arm in ARMS:
        for s in SEEDS:
            pts = sorted(SNX[(arm, s)], key=lambda r: r["train_seconds"])
            ax.plot([r["train_seconds"] for r in pts], [r["L2_exact"] for r in pts], color=COLOR[arm], lw=2, ls=LS[s],
                    marker=MARK[s], ms=5, mec=A.SURF, mew=1.2, label=f"{LABEL[arm]}, seed {s}")
    ax.set_yscale("log")
    A.style(ax, "measured training seconds", "L2_exact", "Accuracy vs measured wall-clock (snapshots)")
    ax.legend(frameon=False, fontsize=6.5, labelcolor=A.INK2)
    save(fig, "wallclock_vs_accuracy")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    for arm in ARMS:
        for s in SEEDS:
            x = F[(arm, s)]
            for ax, key in ((axs[0], "L2_exact"), (axs[1], "persistence_cycles")):
                ax.scatter(x["train_seconds"], x[key], s=60, marker=MARK[s], color=COLOR[arm], edgecolors=A.SURF,
                           linewidths=1.2, zorder=3, label=f"{arm} s{s}" if ax is axs[0] else None)
    axs[0].set_yscale("log")
    A.style(axs[0], "training seconds (<= 640,000 PDE evaluations)", "L2_exact (lower better)", "Compute Pareto: accuracy")
    A.style(axs[1], "training seconds (<= 640,000 PDE evaluations)", "persistence [cycles] (higher better)", "Compute Pareto: persistence")
    axs[0].legend(frameon=False, fontsize=7, labelcolor=A.INK2, ncol=3)
    fig.tight_layout(); save(fig, "pareto")


SN = {}


def main():
    tab = assert_safe_output(REPO / "results_batch2", strict=True, purpose="B2-E02")      # PI-specified paths
    fig_dir = assert_safe_output(REPO / "results_batch2" / "figures", strict=True, purpose="B2-E02")
    torch.set_num_threads(4)
    F, QTR, rows, snaps = {}, {}, [], []
    for arm in ARMS:
        for s in SEEDS:
            f, S, qtr = evaluate_run(arm, s)
            F[(arm, s)], SN[(arm, s)], QTR[(arm, s)] = f, S, qtr
            rows.append(f); snaps.extend(S)
            print(f"{arm:6s} s{s}: L2e {f['L2_exact']:.4f} P {f['persistence_cycles']:.2f} tc {f['collapse_time_s']:.3f} "
                  f"freq {f['frequency_error_exact']:+.4f} decay {f['fit_decay']:.2f} res {f['PDE_residual_rel']:.4f} "
                  f"W {f['train_seconds']:.0f}s E {f['pde_evaluations']}")
    # frozen-control check: M0 must equal the Phase-A modal runs (same code, same seed)
    ref = {(r["arm"], int(r["seed"])): r for r in csv.DictReader(open(tab / "tables" / "B2-E01S_SEED_RESULTS.csv"))}
    for s in SEEDS:
        for k in ("L2_exact", "persistence_cycles", "fit_decay"):
            a, b = float(ref[("modal", s)][k]), F[("M0", s)][k]
            assert abs(a - b) <= 1e-9 * max(1.0, abs(b)), ("modal control changed", s, k, a, b)
    cls = decide(F)
    for arm in ("LBFGS", "CAUSAL"):
        cls[arm]["mechanism"] = mechanism(F, arm)
    with open(tab / "B2-E02_RESULTS.csv", "x", newline="") as f:
        cols = list(rows[0].keys()) + [k for r in rows for k in r if k not in rows[0]]
        cols = list(dict.fromkeys(cols))
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    with open(tab / "B2-E02_SNAPSHOTS.csv", "x", newline="") as f:
        cols = list(dict.fromkeys(k for r in snaps for k in r))
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(snaps)
    with open(tab / "B2-E02_CLASSIFICATION.json", "x") as f:
        json.dump(cls, f, indent=2, default=float)
    figures(F, SN, QTR, fig_dir)
    print(json.dumps(cls, indent=2, default=float))


if __name__ == "__main__":
    main()
