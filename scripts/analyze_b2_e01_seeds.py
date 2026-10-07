"""B2-E01 Phase A: seed replication analysis (seeds 1234, 1235, 1236). Evaluation only.

Uses the IDENTICAL evaluator as scripts/analyze_b2_e01.py (imported, not copied): the same snapshot
metrics, final-model metrics and pre-registered classification (docs/hypotheses/B2-E01.md section 7)
applied PER SEED against the same-seed B1. The Batch-1 reproducibility gate exists only for seed
1234 (Batch 1 ran Z4 only with seed 1234). Seeds are NOT pooled into a single number: every per-seed
row is kept, and mean / sample std / min / max are reported next to them.

    python scripts/analyze_b2_e01_seeds.py

Writes (never overwrites):
  results_batch2/tables/B2-E01S_SEED_RESULTS.csv       arm x seed, final models
  results_batch2/tables/B2-E01S_SEED_SNAPSHOTS.csv     arm x seed x snapshot
  results_batch2/tables/B2-E01S_SEED_SUMMARY.csv       arm x metric: mean, std, min, max, range, CV
  results_batch2/tables/B2-E01S_SEED_CLASSIFICATION.json
  results_batch2/figures/B2-E01S_*.png
"""
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import analyze_b2_e01 as A  # noqa: E402  (frozen B2-E01 evaluator)
from physref.persistence import passes_persistence_gate  # noqa: E402
from physref.safety import assert_safe_output  # noqa: E402

SEEDS = (1234, 1235, 1236)
ARMS = ("B1", "B1_fp64", "mixed", "modal")
PRINCIPAL = ["L2_exact", "persistence_cycles", "persistence_cycles_vel", "collapse_time_s", "frequency_error_exact",
             "fit_decay", "PDE_residual_rel", "IC_error_max", "BC_error_max", "amplitude_ratio", "max_ut_ratio",
             "train_seconds", "seconds_per_step", "peak_rss_mb"]
MARK = {1234: "o", 1235: "s", 1236: "^"}
LS = {1234: "-", 1235: "--", 1236: ":"}


def eid(arm, seed):
    return f"B2-E01-{arm}-s{seed}"


def evaluate_run(arm, seed):
    ck, lg = A.paths(arm, eid(arm, seed))
    met = json.load(open(lg / "metrics.json"))
    snaps = []
    for s in A.STEPS:
        cfg, model, view, prob, blob = A.load(arm, ck / f"step_{s}.pt")
        row, _ = A.snapshot_metrics(view, prob)
        row.update(arm=arm, seed=seed, step=s, pde_evaluations=s * cfg.sampler.mini_batch)
        snaps.append(row)
    cfg, model, view, prob, blob = A.load(arm, ck / "final.pt")
    row, tr = A.snapshot_metrics(view, prob)
    assert abs(row["L2_exact"] / met["L2_exact"] - 1) < 1e-6, "final re-evaluation does not match the run record"
    bm, refs, _, _ = prob
    r = {"arm": arm, "seed": seed, "experiment_id": eid(arm, seed), "precision": cfg.precision,
         "run_key": cfg.run_id() if arm in ("B1", "B1_fp64") else "n/a (arm trainer)",
         "omega_d": refs["exact"].omega_d, "run_key_ok": True}
    if arm == "B1" and seed == 1234:
        r["run_key_ok"] = cfg.run_id() == "Z4_20K__s1234__f78aa7f1da"
    for k in ("L2_exact", "L2_paper", "L2_late_exact", "PDE_residual_rel", "IC_error_max", "BC_error_max",
              "parameters", "model_size_bytes", "optimizer_steps", "pde_evaluations", "train_seconds", "peak_rss_mb",
              "peak_vram_mb", "inference_us_per_point"):
        r[k] = met.get(k)
    r.update({k: row[k] for k in ("persistence_cycles", "persistence_cycles_vel", "collapse_time_s", "collapse_time_vel_s",
                                  "full_window_cycles", "amplitude_ratio", "max_ut_ratio", "fit_w", "fit_decay",
                                  "frequency_error_exact", "phase_error_exact", "amplitude_error_exact")})
    r["passes_persistence_gate"] = passes_persistence_gate(row)
    r["seconds_per_step"] = r["train_seconds"] / r["optimizer_steps"]
    if arm in ("B1", "B1_fp64"):
        r.update(residual_evaluations=met["pde_evaluations"], forward_passes=met["optimizer_steps"],
                 backward_passes=met["grad_evaluations"], diag_backward_passes=met["diag_gradients"])
    else:
        r.update(residual_evaluations=met["residual_evaluations"], forward_passes=met["forward_passes"],
                 backward_passes=met["backward_passes"], diag_backward_passes=0)
    if arm == "mixed":
        r.update(A.link_residual_rel(model, prob))
    return r, snaps, tr, blob


def classify_seed(R, seed):
    """B2-E01 section 7 per seed vs same-seed B1 (A.decide), gate only meaningful for seed 1234."""
    out = A.decide(R)
    if seed != 1234:
        out["reproducibility_gate"] = {"pass": None, "note": "not applicable: Batch 1 ran Z4 only with seed 1234"}
    return out


def summary(rows):
    out = []
    for arm in ARMS:
        for k in PRINCIPAL:
            v = np.array([float(r[k]) for r in rows if r["arm"] == arm], dtype=float)
            sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
            out.append({"arm": arm, "metric": k, "n_seeds": len(v), "seed1234": v[0], "seed1235": v[1], "seed1236": v[2],
                        "mean": float(v.mean()), "std": sd, "min": float(v.min()), "max": float(v.max()),
                        "range": float(v.max() - v.min()), "cv": sd / abs(v.mean()) if v.mean() else float("nan")})
    return out


def phase_b(R_by_seed, C_by_seed):
    """The five Phase-B questions, computed (no judgement in code beyond the stated rules)."""
    q = {}
    q["modal_case_per_seed"] = {s: C_by_seed[s]["modal"]["case"] for s in SEEDS}
    q["modal_P_ratio_per_seed"] = {s: R_by_seed[s]["modal"]["persistence_cycles"] / R_by_seed[s]["B1"]["persistence_cycles"] for s in SEEDS}
    q["modal_L2_ratio_per_seed"] = {s: R_by_seed[s]["modal"]["L2_exact"] / R_by_seed[s]["B1"]["L2_exact"] for s in SEEDS}
    q["modal_collapse_time_per_seed"] = {s: R_by_seed[s]["modal"]["collapse_time_s"] for s in SEEDS}
    q["B1_collapse_time_per_seed"] = {s: R_by_seed[s]["B1"]["collapse_time_s"] for s in SEEDS}
    q["fp64_case_per_seed"] = {s: C_by_seed[s]["B1_fp64"]["case"] for s in SEEDS}
    q["fp64_max_rel_dL2"] = max(abs(R_by_seed[s]["B1_fp64"]["L2_exact"] / R_by_seed[s]["B1"]["L2_exact"] - 1) for s in SEEDS)
    q["fp64_max_abs_dP"] = max(abs(R_by_seed[s]["B1_fp64"]["persistence_cycles"] - R_by_seed[s]["B1"]["persistence_cycles"]) for s in SEEDS)
    q["mixed_case_per_seed"] = {s: C_by_seed[s]["mixed"]["case"] for s in SEEDS}
    q["mixed_class_per_seed"] = {s: C_by_seed[s]["mixed"]["class"] for s in SEEDS}
    q["mixed_W_ratio_per_seed"] = {s: C_by_seed[s]["mixed"]["W_ratio"] for s in SEEDS}
    q["case_E_per_seed"] = {s: C_by_seed[s]["case_E_all_similar"] for s in SEEDS}
    return q


def figures(rows, snaps, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": A.SURF, "savefig.facecolor": A.SURF, "font.size": 9})

    def save(fig, name):
        p = fig_dir / f"B2-E01S_{name}.png"
        if p.exists():
            raise FileExistsError(p)
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)

    # per-arm small multiples: L2 and collapse front vs PDE evaluations, one line per seed
    for key, ylab, name, logy in (("L2_exact", "L2_exact", "L2_vs_pde_evals", True),
                                  ("collapse_time_s", "collapse time [s]", "collapse_front", False)):
        fig, axs = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
        for ax, arm in zip(axs, ARMS):
            for s in SEEDS:
                pts = [r for r in snaps if r["arm"] == arm and r["seed"] == s]
                ax.plot([r["pde_evaluations"] for r in pts], [r[key] for r in pts], color=A.COLOR[arm], lw=2, ls=LS[s],
                        marker=MARK[s], ms=5, mec=A.SURF, mew=1.2, label=f"seed {s}")
            if logy:
                ax.set_yscale("log")
            A.style(ax, "PDE evaluations", ylab if arm == "B1" else "", A.LABEL[arm])
            ax.legend(frameon=False, fontsize=7, labelcolor=A.INK2)
        fig.suptitle(f"{ylab} vs PDE evaluations, per seed (seeds shown separately, not pooled)", color=A.INK,
                     fontsize=11, x=0.01, ha="left")
        fig.tight_layout()
        save(fig, name)

    # final metrics per seed (dot strip per arm)
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (key, lab) in zip(axs, (("persistence_cycles", "persistence [cycles]"), ("L2_exact", "L2_exact"),
                                    ("train_seconds", "training seconds"))):
        for i, arm in enumerate(ARMS):
            for s in SEEDS:
                v = [r[key] for r in rows if r["arm"] == arm and r["seed"] == s][0]
                ax.scatter(i, v, s=60, color=A.COLOR[arm], marker=MARK[s], edgecolors=A.SURF, linewidths=1.2, zorder=3)
            vals = [r[key] for r in rows if r["arm"] == arm]
            ax.annotate(f"{np.mean(vals):.3g}", (i, np.mean(vals)), xytext=(10, 0), textcoords="offset points",
                        color=A.INK2, fontsize=8, va="center")
        ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(ARMS)
        A.style(ax, "", lab, f"{lab} (final, 640k evaluations)")
    from matplotlib.lines import Line2D
    axs[0].legend([Line2D([], [], color=A.INK2, marker=MARK[s], ls="") for s in SEEDS], [f"seed {s}" for s in SEEDS],
                  frameon=False, fontsize=8, labelcolor=A.INK2)
    fig.tight_layout()
    save(fig, "final_by_seed")

    # Pareto per seed
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for arm in ARMS:
        for s in SEEDS:
            r = [x for x in rows if x["arm"] == arm and x["seed"] == s][0]
            for ax, key in ((axs[0], "L2_exact"), (axs[1], "persistence_cycles")):
                hollow = arm == "modal"
                ax.scatter(r["train_seconds"], r[key], s=55, marker=MARK[s], color=A.SURF if hollow else A.COLOR[arm],
                           edgecolors=A.COLOR[arm], linewidths=1.8, zorder=3,
                           label=f"{arm} s{s}" if ax is axs[0] else None)
    axs[0].set_yscale("log")
    A.style(axs[0], "training seconds (640,000 PDE evaluations)", "L2_exact", "Accuracy vs compute, all seeds")
    A.style(axs[1], "training seconds (640,000 PDE evaluations)", "persistence [cycles]", "Persistence vs compute, all seeds")
    axs[0].legend(frameon=False, fontsize=7, labelcolor=A.INK2, ncol=2)
    fig.tight_layout()
    save(fig, "pareto")


def main():
    tab = assert_safe_output(REPO / "results_batch2" / "tables", strict=True, purpose="B2-E01S")
    fig_dir = assert_safe_output(REPO / "results_batch2" / "figures", strict=True, purpose="B2-E01S")
    torch.set_num_threads(4)
    rows, snaps, R_by_seed, C_by_seed = [], [], {}, {}
    for s in SEEDS:
        R = {}
        for arm in ARMS:
            r, sn, _, _ = evaluate_run(arm, s)
            R[arm] = r; rows.append(r); snaps.extend(sn)
            print(f"seed {s} {arm:8s} L2e {r['L2_exact']:.4f} P {r['persistence_cycles']:.2f} tc {r['collapse_time_s']:.3f} "
                  f"w {r['fit_w']:.2f} decay {r['fit_decay']:.2f} res {r['PDE_residual_rel']:.3f} W {r['train_seconds']:.0f}s")
        R_by_seed[s] = R
        C_by_seed[s] = classify_seed(R, s)
    # seed 1234 must reproduce the committed B2-E01 table exactly (same evaluator, same checkpoints)
    ref = {r["arm"]: r for r in csv.DictReader(open(tab / "B2-E01_RESULTS.csv"))}
    for arm in ARMS:
        for k in ("L2_exact", "persistence_cycles", "fit_decay", "PDE_residual_rel"):
            assert abs(float(ref[arm][k]) - R_by_seed[1234][arm][k]) <= 1e-9 * max(1.0, abs(R_by_seed[1234][arm][k])), (arm, k)
    out = {"per_seed": {str(s): C_by_seed[s] for s in SEEDS}, "phase_b": phase_b(R_by_seed, C_by_seed)}
    cols = list(rows[0].keys()) + [k for k in rows[2] if k not in rows[0]]
    with open(tab / "B2-E01S_SEED_RESULTS.csv", "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["class_vs_B1", "case"]); w.writeheader()
        for r in rows:
            c = C_by_seed[r["seed"]].get(r["arm"], {})
            w.writerow({**r, "class_vs_B1": c.get("class", "reference"), "case": c.get("case", "reference")})
    with open(tab / "B2-E01S_SEED_SNAPSHOTS.csv", "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(snaps[0].keys())); w.writeheader(); w.writerows(snaps)
    summ = summary(rows)
    with open(tab / "B2-E01S_SEED_SUMMARY.csv", "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)
    with open(tab / "B2-E01S_SEED_CLASSIFICATION.json", "x") as f:
        json.dump(out, f, indent=2, default=float)
    figures(rows, snaps, fig_dir)
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
