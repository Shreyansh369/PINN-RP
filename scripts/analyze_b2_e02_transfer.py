"""B2-E02 section 10/11: full-field transfer evaluation and FINAL B2-E02 decision. Evaluation only.

B1 (frozen, re-run in the same batch) vs B1 + Adam->L-BFGS, seeds 1234-1236, 640,000-evaluation budget.
Uses the frozen B2-E01 evaluator for every snapshot and the final model, plus the strong PDE residual of u
at every snapshot. Transfer success (pre-registered, all 3 seeds): P >= P_B1 + 3.98 cycles,
L2 <= 0.9 L2_B1, |frequency error| <= 2 %, PDE residual of u <= 1.1 x B1's.

    python scripts/analyze_b2_e02_transfer.py
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

import analyze_b2_e01 as A  # noqa: E402
from beampinn.evaluation.metrics import physics_metrics  # noqa: E402
from physref.persistence import passes_persistence_gate  # noqa: E402
from physref.safety import assert_safe_output  # noqa: E402

RUNS = REPO / "results_batch2" / "runs"
SEEDS = (1234, 1235, 1236)
ARMS = ("B1", "LBFGS")
LABEL = {"B1": "B1 full field: Adam + decay", "LBFGS": "B1 full field: Adam -> L-BFGS"}
COLOR = {"B1": "#2a78d6", "LBFGS": "#eb6834"}
MARK = {1234: "o", 1235: "s", 1236: "^"}
LS = {1234: "-", 1235: "--", 1236: ":"}
DP_THR = 3.98                                  # docs/hypotheses/B2-E02.md section 9 (2 x Phase-A B1 range 1.99)


def eid(arm, seed):
    return f"B2-E02T-{arm}-s{seed}"


def run_dirs(arm, seed):
    d = RUNS / eid(arm, seed)
    rid = next((d / "checkpoints").iterdir()).name
    return d / "checkpoints" / rid, d / "logs" / rid


def eval_ckpt(path):
    cfg, model, view, prob, blob = A.load("B1", path)
    row, tr = A.snapshot_metrics(view, prob)
    bm, refs, c2, g = prob
    row["PDE_residual_rel"] = physics_metrics(view, bm, refs["exact"], c2, g)["PDE_residual_rel"]
    row["pde_evaluations"] = int(blob["acc"]["pde_evaluations"])
    row["train_seconds"] = float(blob["acc"]["train_seconds"])
    return row, tr, blob


def evaluate(arm, seed):
    ck, lg = run_dirs(arm, seed)
    met = json.load(open(lg / "metrics.json"))
    snaps = sorted(ck.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    S = []
    for p in snaps:
        r, _, _ = eval_ckpt(p)
        r.update(arm=arm, seed=seed, checkpoint=p.name)
        S.append(r)
    r, tr, blob = eval_ckpt(ck / "final.pt")
    assert abs(r["L2_exact"] / met["L2_exact"] - 1) < 1e-6, "final re-evaluation does not match the run record"
    if all(x["pde_evaluations"] != r["pde_evaluations"] for x in S):
        S.append({**r, "arm": arm, "seed": seed, "checkpoint": "final.pt"})
    S.sort(key=lambda x: x["pde_evaluations"])
    acc = blob["acc"]
    F = {"arm": arm, "seed": seed, "experiment_id": eid(arm, seed), **r,
         "IC_error_max": met["IC_error_max"], "BC_error_max": met["BC_error_max"], "parameters": met["parameters"],
         "peak_rss_mb": met["peak_rss_mb"], "optimizer_steps": acc["optimizer_steps"],
         "lbfgs_closure_evals": acc.get("lbfgs_closure_evals", 0), "lbfgs_stop_reason": acc.get("lbfgs_stop_reason", ""),
         "switch_pde_evaluations": acc.get("switch_pde_evaluations", ""),
         "forward_passes": acc.get("forward_passes", acc["optimizer_steps"]), "backward_passes": acc["grad_evaluations"],
         "decay_error": abs(r["fit_decay"] - 3.54), "passes_persistence_gate": passes_persistence_gate(r),
         "CF_AUC": float(np.mean([x["collapse_time_s"] for x in S]))}
    return F, S, tr


def decide(F):
    per = {}
    for s in SEEDS:
        b, x = F[("B1", s)], F[("LBFGS", s)]
        ok = (x["persistence_cycles"] >= b["persistence_cycles"] + DP_THR and x["L2_exact"] <= 0.9 * b["L2_exact"]
              and abs(x["frequency_error_exact"]) <= 0.02 and x["PDE_residual_rel"] <= 1.1 * b["PDE_residual_rel"])
        per[s] = {"transfer_success": bool(ok), "dP": x["persistence_cycles"] - b["persistence_cycles"],
                  "L2_ratio": x["L2_exact"] / b["L2_exact"], "res_ratio": x["PDE_residual_rel"] / b["PDE_residual_rel"],
                  "W_ratio": x["train_seconds"] / b["train_seconds"], "freq_err": x["frequency_error_exact"]}
    passed = all(p["transfer_success"] for p in per.values())
    # section 11 final decision (modal label of LBFGS = DOMINANCE -> accuracy and dominance)
    decision = "A" if passed else "E"
    return {"dP_threshold": DP_THR, "per_seed": per, "transfer_passed": passed, "final_decision": decision,
            "modal_label_LBFGS": "DOMINANCE (B2-E02_CLASSIFICATION.json)"}


def figures(F, SN, TR, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": A.SURF, "savefig.facecolor": A.SURF, "font.size": 9})

    def save(fig, name):
        p = fig_dir / f"B2-E02T_{name}.png"
        if p.exists():
            raise FileExistsError(p)
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)

    for key, ylab, name, logy in (("L2_exact", "L2_exact", "L2_vs_progress", True),
                                  ("collapse_time_s", "collapse time [s] (1.0 = full window)", "collapse_front", False),
                                  ("PDE_residual_rel", "PDE residual of u (relative)", "pde_residual", True),
                                  ("fit_decay", "fitted decay [1/s] (exact 3.54)", "decay_rate", False)):
        fig, ax = plt.subplots(figsize=(7.5, 4.3))
        for arm in ARMS:
            for s in SEEDS:
                pts = SN[(arm, s)]
                ax.plot([r["pde_evaluations"] for r in pts], [r[key] for r in pts], color=COLOR[arm], lw=2, ls=LS[s],
                        marker=MARK[s], ms=5, mec=A.SURF, mew=1.2, label=f"{LABEL[arm]}, seed {s}")
        if logy:
            ax.set_yscale("log")
        A.style(ax, "PDE (collocation) evaluations", ylab, f"Full-field transfer: {ylab}")
        ax.legend(frameon=False, fontsize=6.5, labelcolor=A.INK2)
        save(fig, name)
    fig, axs = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    for ax, s in zip(axs, SEEDS):
        t, up, vp, ue, ve = TR[("B1", s)]
        ax.plot(t, ue, color=A.INK, lw=1, label="exact")
        ax.plot(t, up, color=COLOR["B1"], lw=1.4, label=LABEL["B1"])
        t2, up2, *_ = TR[("LBFGS", s)]
        ax.plot(t2, up2, color=COLOR["LBFGS"], lw=1.4, label=LABEL["LBFGS"])
        A.style(ax, "t [s]" if s == SEEDS[-1] else "", "u(L/2,t) [m]", f"Mid-span displacement, seed {s}")
        ax.legend(frameon=False, fontsize=7.5, labelcolor=A.INK2, loc="upper right")
    fig.tight_layout(); save(fig, "displacement_trace")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    for arm in ARMS:
        for s in SEEDS:
            x = F[(arm, s)]
            for ax, key in ((axs[0], "L2_exact"), (axs[1], "persistence_cycles")):
                ax.scatter(x["train_seconds"], x[key], s=60, marker=MARK[s], color=COLOR[arm], edgecolors=A.SURF,
                           linewidths=1.2, zorder=3, label=f"{arm} s{s}" if ax is axs[0] else None)
    axs[0].set_yscale("log")
    A.style(axs[0], "training seconds (<= 640,000 PDE evaluations)", "L2_exact", "Full-field compute Pareto: accuracy")
    A.style(axs[1], "training seconds (<= 640,000 PDE evaluations)", "persistence [cycles]", "Full-field compute Pareto: persistence")
    axs[0].legend(frameon=False, fontsize=7, labelcolor=A.INK2, ncol=2)
    fig.tight_layout(); save(fig, "pareto")


def main():
    out = assert_safe_output(REPO / "results_batch2", strict=True, purpose="B2-E02T")
    fig_dir = assert_safe_output(REPO / "results_batch2" / "figures", strict=True, purpose="B2-E02T")
    torch.set_num_threads(4)
    F, SN, TR, rows, snaps = {}, {}, {}, [], []
    for arm in ARMS:
        for s in SEEDS:
            f, S, tr = evaluate(arm, s)
            F[(arm, s)], SN[(arm, s)], TR[(arm, s)] = f, S, tr
            rows.append(f); snaps.extend(S)
            print(f"{arm:5s} s{s}: L2e {f['L2_exact']:.4f} P {f['persistence_cycles']:.2f} tc {f['collapse_time_s']:.3f} "
                  f"freq {f['frequency_error_exact']:+.4f} decay {f['fit_decay']:.2f} res {f['PDE_residual_rel']:.4f} "
                  f"W {f['train_seconds']:.0f}s E {f['pde_evaluations']}")
    ref = {(r["arm"], int(r["seed"])): r for r in csv.DictReader(open(out / "tables" / "B2-E01S_SEED_RESULTS.csv"))}
    for s in SEEDS:                      # frozen full-field control must equal Phase-A B1 of the same seed
        for k in ("L2_exact", "persistence_cycles", "fit_decay"):
            a, b = float(ref[("B1", s)][k]), F[("B1", s)][k]
            assert abs(a - b) <= 1e-9 * max(1.0, abs(b)), ("B1 control changed", s, k, a, b)
    cls = decide(F)
    with open(out / "B2-E02T_RESULTS.csv", "x", newline="") as f:
        cols = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    with open(out / "B2-E02T_SNAPSHOTS.csv", "x", newline="") as f:
        cols = list(dict.fromkeys(k for r in snaps for k in r))
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(snaps)
    with open(out / "B2-E02T_CLASSIFICATION.json", "x") as f:
        json.dump(cls, f, indent=2, default=float)
    figures(F, SN, TR, fig_dir)
    print(json.dumps(cls, indent=2, default=float))


if __name__ == "__main__":
    main()
