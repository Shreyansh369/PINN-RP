"""B2-E03 analysis: collocation coverage vs optimizer mechanism isolation. Evaluation only - no training.
Implements docs/hypotheses/B2-E03.md sections 3, 5-10 EXACTLY (all thresholds below are copied from it).

    python scripts/analyze_b2_e03.py

Writes (never overwrites): results_batch2/B2-E03_{RESULTS,SNAPSHOTS}.csv, results_batch2/B2-E03_CLASSIFICATION.json,
results_batch2/figures/B2-E03_*.png
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

import analyze_b2_e01 as A  # noqa: E402  (frozen snapshot evaluator)
from beampinn.evaluation.metrics import PDE_NT, PDE_NX, grid, physics_metrics  # noqa: E402
from beampinn.losses.residuals import pde_residual  # noqa: E402
from physref.persistence import passes_persistence_gate  # noqa: E402
from physref.safety import assert_safe_output  # noqa: E402

RUNS = REPO / "results_batch2" / "runs"
SEEDS = (1234, 1235, 1236)
ARMS = ("B1", "LBFGS-F", "LBFGS-R", "LBFGS-4X", "ADAM-FULL")
STAGE2 = ARMS[1:]
COLOR = {"B1": "#2a78d6", "LBFGS-F": "#eb6834", "LBFGS-R": "#1baf7a", "LBFGS-4X": "#eda100", "ADAM-FULL": "#e87ba4"}
LABEL = {"B1": "B1 (Adam mini-batch)", "LBFGS-F": "L-BFGS fixed 640", "LBFGS-R": "L-BFGS resampled",
         "LBFGS-4X": "L-BFGS 2,560 reservoir", "ADAM-FULL": "full-batch Adam fixed 640"}
MARK = {1234: "o", 1235: "s", 1236: "^"}
LS = {1234: "-", 1235: "--", 1236: ":"}
DP_IB1, BUDGET, ADAM_FULL_E, COMPUTE_TOL = 3.98, 640_000, 623_360, 0.03


def eid(arm, seed):
    return f"B2-E03-{arm}-s{seed}"


def run_dirs(arm, seed):
    d = RUNS / eid(arm, seed)
    rid = next((d / "checkpoints").iterdir()).name
    return d, d / "checkpoints" / rid, d / "logs" / rid


# ----------------------------------------------------------------- residuals
_DEN = {}


def utt_den():
    if "v" not in _DEN:
        from beampinn.physics.benchmarks import get_benchmark
        bm = get_benchmark("FE-D-M1"); ex = bm.reference("exact")
        _, _, X, T = grid(bm.L, bm.t_end, PDE_NX, PDE_NT)
        _DEN["v"] = float(np.sqrt(np.mean(ex.u(X, T, 0, 2) ** 2)))
        _DEN["grid"] = (np.unique(X), np.unique(T))
    return _DEN["v"]


def r_train(model, prob, x, t):
    bm, refs, c2, g = prob
    dt = next(model.parameters()).dtype
    out = []
    for i in range(0, len(x), 1024):
        xx = x[i:i + 1024].to(dt).clone().requires_grad_(True)
        tt = t[i:i + 1024].to(dt).clone().requires_grad_(True)
        out.append(pde_residual(model, xx, tt, c2, g).detach().double())
    r = torch.cat(out).numpy().ravel()
    return float(np.sqrt(np.mean(r ** 2)) / utt_den())


def check_independent(x, t):
    """STOP if any training point coincides with a dense-grid node (section 7)."""
    utt_den()
    gx, gt = _DEN["grid"]
    xs, ts = x.numpy().ravel(), t.numpy().ravel()
    dx = np.min(np.abs(xs[:, None] - gx[None, :]), axis=1)
    dtt = np.min(np.abs(ts[:, None] - gt[None, :]), axis=1)
    bad = np.sum((dx <= 1e-12) & (dtt <= 1e-12))
    assert bad == 0, f"STOP: {bad} training points coincide with dense-grid nodes"


def training_sets(blob, final_coll, is_switch):
    """(primary S, secondary S or None) per section 7."""
    if is_switch or ("collocation" in blob and blob["collocation"]["active_set"] is not None):
        coll = final_coll if is_switch else blob["collocation"]
        sets, mode = coll["sets"], coll["mode"]
        if mode == "reservoir":
            act = sets[0] if is_switch else sets[coll["active_set"]]
            return (torch.cat([s[0] for s in sets]), torch.cat([s[1] for s in sets])), act
        act = sets[0] if (is_switch or mode == "fixed") else sets[coll["active_set"]]
        return act, None
    x, t, _ = blob["sampler"]["data"]["f"]
    return (x, t), None


# ----------------------------------------------------------------- evaluate
def eval_point(path, final_coll, is_switch=False):
    cfg, model, view, prob, blob = A.load("B1", path)
    row, tr = A.snapshot_metrics(view, prob)
    bm, refs, c2, g = prob
    row["R_dense"] = physics_metrics(view, bm, refs["exact"], c2, g)["PDE_residual_rel"]
    (xs, ts), sec = training_sets(blob, final_coll, is_switch)
    check_independent(xs, ts)
    row["R_train"] = r_train(model, prob, xs, ts)
    row["n_train_points"] = int(len(xs))
    row["R_train_active"] = r_train(model, prob, *sec) if sec is not None else row["R_train"]
    row["G"] = row["R_dense"] / row["R_train"]
    row["dR"] = row["R_dense"] - row["R_train"]
    row["pde_evaluations"] = int(blob["acc"]["pde_evaluations"])
    row["train_seconds"] = float(blob["acc"]["train_seconds"])
    row["checkpoint"] = path.name
    return row, tr, blob


def evaluate(arm, seed):
    d, ck, lg = run_dirs(arm, seed)
    met = json.load(open(lg / "metrics.json"))
    fin_blob = torch.load(ck / "final.pt", weights_only=False)
    fcoll = fin_blob.get("collocation")
    S = []
    for p in sorted(ck.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1])):
        r, _, _ = eval_point(p, fcoll)
        r.update(arm=arm, seed=seed); S.append(r)
    sw = None
    if (ck / "switch.pt").exists():
        sw, _, _ = eval_point(ck / "switch.pt", fcoll, is_switch=True)
        sw.update(arm=arm, seed=seed); S.append(sw)
    f, tr, blob = eval_point(ck / "final.pt", fcoll)
    assert abs(f["L2_exact"] / met["L2_exact"] - 1) < 1e-6, "final re-evaluation does not match the run record"
    if all(x["pde_evaluations"] != f["pde_evaluations"] or x["checkpoint"] == "switch.pt" for x in S):
        S.append({**f, "arm": arm, "seed": seed})
    S.sort(key=lambda x: (x["pde_evaluations"], x["checkpoint"] != "switch.pt"))
    acc = blob["acc"]
    F = {"arm": arm, "seed": seed, "experiment_id": eid(arm, seed), **f,
         "IC_error_max": met["IC_error_max"], "BC_error_max": met["BC_error_max"], "parameters": met["parameters"],
         "model_size_bytes": met["model_size_bytes"], "peak_rss_mb": met["peak_rss_mb"],
         "optimizer_steps": acc["optimizer_steps"], "stage2_closure_evals": acc.get("stage2_closure_evals", 0),
         "lbfgs_iterations": acc.get("lbfgs_iterations", ""), "stage2_stop_reason": acc.get("stage2_stop_reason", ""),
         "forward_passes": acc.get("forward_passes", acc["optimizer_steps"]), "backward_passes": acc["grad_evaluations"],
         "n_point_sets": acc.get("n_point_sets", ""), "n_distinct_training_points_stage2": acc.get("n_distinct_training_points_stage2", ""),
         "init_checksum": json.load(open(d / "init_checksum.json"))["init_checksum"],
         "passes_persistence_gate": passes_persistence_gate(f)}
    if fcoll is not None:
        F["switch_events"] = json.dumps([(e["closure_index"], e["pde_evaluations"], e["set"]) for e in fcoll["events"]])
    if sw is not None:
        F.update({f"switch_{k}": sw[k] for k in ("R_train", "R_dense", "G", "persistence_cycles", "L2_exact")})
    return F, S, tr, fcoll


# ----------------------------------------------------------------- decisions (sections 3, 8, 9)
def MB(y, f):
    return y["L2_exact"] <= 0.80 * f["L2_exact"] and y["R_dense"] <= 0.80 * f["R_dense"] and \
        y["persistence_cycles"] >= f["persistence_cycles"] - 0.5


def MW(y, f):
    return y["L2_exact"] >= 1.25 * f["L2_exact"] or y["R_dense"] >= 1.25 * f["R_dense"] or \
        y["persistence_cycles"] <= f["persistence_cycles"] - 1.0


def RB1(y, b):
    return y["L2_exact"] <= 1.10 * b["L2_exact"] and y["R_dense"] <= 1.10 * b["R_dense"] and \
        y["persistence_cycles"] >= b["persistence_cycles"] - 1.0 and abs(y["frequency_error_exact"]) <= 0.02


def IB1(y, b):
    return y["persistence_cycles"] >= b["persistence_cycles"] + DP_IB1 and y["L2_exact"] <= 0.9 * b["L2_exact"] and \
        y["R_dense"] <= 1.1 * b["R_dense"] and abs(y["frequency_error_exact"]) <= 0.02


def OS(y):
    return (y["R_train"] < y["switch_R_train"] and y["R_dense"] > y["switch_R_dense"] and y["G"] > y["switch_G"]
            and y["persistence_cycles"] - y["switch_persistence_cycles"] <= 0.5)


def dominates(x, y):
    le = (x["L2_exact"] <= y["L2_exact"] and x["persistence_cycles"] >= y["persistence_cycles"]
          and x["train_seconds"] <= y["train_seconds"] and x["peak_rss_mb"] <= y["peak_rss_mb"])
    lt = (x["L2_exact"] < y["L2_exact"] or x["persistence_cycles"] > y["persistence_cycles"]
          or x["train_seconds"] < y["train_seconds"] or x["peak_rss_mb"] < y["peak_rss_mb"])
    return le and lt


def decide(F):
    per = {}
    for s in SEEDS:
        b, f = F[("B1", s)], F[("LBFGS-F", s)]
        row = {}
        for arm in ARMS:
            y = F[(arm, s)]
            row[arm] = {"RB1": RB1(y, b), "IB1": IB1(y, b), "G": y["G"], "L2": y["L2_exact"], "R_dense": y["R_dense"],
                        "P": y["persistence_cycles"]}
            if arm in STAGE2:
                row[arm]["OS"] = OS(y)
            if arm in ("LBFGS-R", "LBFGS-4X", "ADAM-FULL"):
                row[arm].update(MB=MB(y, f), MW=MW(y, f), G_ratio_vs_F=y["G"] / f["G"], L2_ratio_vs_F=y["L2_exact"] / f["L2_exact"],
                                Rd_ratio_vs_F=y["R_dense"] / f["R_dense"], dP_vs_F=y["persistence_cycles"] - f["persistence_cycles"])
        row["pareto_nondominated"] = [a for a in ARMS if not any(dominates(F[(o, s)], F[(a, s)]) for o in ARMS if o != a)]
        per[s] = row
    n = lambda arm, key: sum(bool(per[s][arm][key]) for s in SEEDS)
    cov = {a: n(a, "MB") for a in ("LBFGS-R", "LBFGS-4X")}
    opt = n("ADAM-FULL", "MB")
    Cov3, Cov2 = max(cov.values()) == 3, max(cov.values()) >= 2
    Opt3, Opt2 = opt == 3, opt >= 2
    if Cov3 and not Opt2:
        case = "A"
    elif Opt3 and not Cov2:
        case = "B"
    elif Cov2 and Opt2:
        case = "C"
    else:
        case = "D"
    F_OS = n("LBFGS-F", "OS")
    h1_arm = {a: (cov[a], sum(per[s][a]["G_ratio_vs_F"] <= 0.5 for s in SEEDS), sum(not per[s][a]["OS"] for s in SEEDS))
              for a in ("LBFGS-R", "LBFGS-4X")}
    if F_OS >= 2 and any(m == 3 and g == 3 and na >= 2 for m, g, na in h1_arm.values()):
        H1 = "supported"
    elif F_OS >= 2 and any(m >= 2 and g >= 2 and na >= 2 for m, g, na in h1_arm.values()):
        H1 = "partially supported"
    elif max(cov.values()) == 0 or F_OS < 2:
        H1 = "falsified" + (" (F overfitting signature not reproduced)" if F_OS < 2 else "")
    else:
        H1 = "not supported (criteria not met)"
    H2 = "supported" if opt == 3 else ("partially supported" if opt == 2 else ("falsified" if opt == 0 else "not supported (1/3)"))
    rbR, rb4, rbF = n("LBFGS-R", "RB1"), n("LBFGS-4X", "RB1"), n("LBFGS-F", "RB1")
    if rbR == 3 and rb4 == 3 and rbF == 0 and opt < 2:
        H3 = "supported"
    elif (3 - rbR) >= 2 or (3 - rb4) >= 2:
        H3 = "falsified"
    else:
        H3 = "not supported (criteria not met)"
    H4 = "supported" if all(v < 2 for v in list(cov.values()) + [opt]) else "not supported"
    chunk_flag = rbF >= 2                         # section 4: B2-E02 fixed-set L-BFGS failed RB1 in 3/3 seeds
    return {"per_seed": per, "counts": {"MB_LBFGS-R": cov["LBFGS-R"], "MB_LBFGS-4X": cov["LBFGS-4X"], "MB_ADAM-FULL": opt,
                                        "OS_LBFGS-F": F_OS, **{f"RB1_{a}": n(a, "RB1") for a in ARMS},
                                        **{f"IB1_{a}": n(a, "IB1") for a in ARMS},
                                        **{f"OS_{a}": n(a, "OS") for a in STAGE2}},
            "CASE": case, "H1": H1, "H2": H2, "H3": H3, "H4": H4, "chunking_confounded": chunk_flag}


# ----------------------------------------------------------------- integrity checks (section 11)
def integrity(F, COLL):
    out = {}
    for s in SEEDS:
        cks = {F[(a, s)]["init_checksum"] for a in ARMS}
        assert len(cks) == 1, f"STOP: initial weights differ for seed {s}: {cks}"
        for a in STAGE2:
            e = F[(a, s)]["pde_evaluations"]
            assert e <= BUDGET, f"STOP: {a} s{s} exceeds the budget ({e})"
            assert abs(e - ADAM_FULL_E) <= COMPUTE_TOL * ADAM_FULL_E, f"STOP: unequal compute {a} s{s}: {e}"
        assert F[("B1", s)]["pde_evaluations"] == BUDGET
        # set0 identical across stage-2 arms and equal to the B2-E02 transfer fixed set
        s0 = COLL[("LBFGS-F", s)]["sets"][0]
        for a in STAGE2:
            x, t = COLL[(a, s)]["sets"][0]
            assert torch.equal(x, s0[0]) and torch.equal(t, s0[1]), f"STOP: set0 differs ({a}, s{s})"
        e02 = next((RUNS / f"B2-E02T-LBFGS-s{s}" / "checkpoints").iterdir()) / "final.pt"
        x2, t2, _ = torch.load(e02, weights_only=False)["sampler"]["data"]["f"]
        assert torch.equal(x2, s0[0]) and torch.equal(t2, s0[1]), f"STOP: set0 differs from the B2-E02 fixed set (s{s})"
        out[s] = {"init_checksum": cks.pop(), "set0_equals_B2E02": True}
    ref = {(r["arm"], int(r["seed"])): r for r in csv.DictReader(open(REPO / "results_batch2" / "tables" / "B2-E01S_SEED_RESULTS.csv"))}
    for s in SEEDS:                                                          # frozen B1 control
        for k in ("L2_exact", "persistence_cycles", "fit_decay"):
            a, b = float(ref[("B1", s)][k]), F[("B1", s)][k]
            assert abs(a - b) <= 1e-9 * max(1.0, abs(b)), ("STOP: B1 control changed", s, k, a, b)
    return out


# ----------------------------------------------------------------- figures
def figures(F, SN, TR, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": A.SURF, "savefig.facecolor": A.SURF, "font.size": 9})

    def save(fig, name):
        p = fig_dir / f"B2-E03_{name}.png"
        if p.exists():
            raise FileExistsError(p)
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)

    def progress(key, ylab, name, logy=False):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.0), sharey=True)
        for ax, s in zip(axs, SEEDS):
            for arm in ARMS:
                pts = [r for r in SN[(arm, s)]]
                ax.plot([r["pde_evaluations"] for r in pts], [r[key] for r in pts], color=COLOR[arm], lw=2,
                        marker="o", ms=4, mec=A.SURF, mew=1, label=LABEL[arm])
            ax.axvline(320_000, color=A.MUTED, lw=1)
            if logy:
                ax.set_yscale("log")
            A.style(ax, "PDE evaluations (grey: stage-2 switch)", ylab if s == SEEDS[0] else "", f"seed {s}")
        axs[0].legend(frameon=False, fontsize=7, labelcolor=A.INK2)
        fig.suptitle(f"{ylab} vs PDE evaluations", color=A.INK, fontsize=11, x=0.01, ha="left")
        fig.tight_layout(); save(fig, name)

    progress("L2_exact", "L2_exact", "L2_vs_pde_evals", True)
    progress("persistence_cycles", "persistence [cycles]", "persistence_vs_pde_evals")
    progress("collapse_time_s", "collapse-front time [s]", "collapse_front")
    progress("R_train", "R_train (training points)", "R_train", True)
    progress("R_dense", "R_dense (independent 51x501 grid)", "R_dense", True)
    progress("G", "G = R_dense / R_train", "residual_gap_ratio", True)
    progress("dR", "dR = R_dense - R_train", "residual_gap_difference")
    progress("frequency_error_exact", "fitted frequency error", "frequency_error")
    progress("fit_decay", "fitted decay [1/s] (exact 3.54)", "decay_rate")

    # KEY plot: R_train vs R_dense trajectories (stage 2 from the switch state)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)
    for ax, s in zip(axs, SEEDS):
        lo, hi = 1e-3, 1.0
        ax.plot([lo, hi], [lo, hi], color=A.MUTED, lw=1)
        ax.annotate("R_dense = R_train", (lo * 1.5, lo * 1.9), color=A.INK2, fontsize=7)
        for arm in ARMS:
            pts = [r for r in SN[(arm, s)] if r["pde_evaluations"] >= 256_000]
            ax.plot([r["R_train"] for r in pts], [r["R_dense"] for r in pts], color=COLOR[arm], lw=2, marker="o", ms=4,
                    mec=A.SURF, mew=1, label=LABEL[arm])
            ax.scatter([pts[-1]["R_train"]], [pts[-1]["R_dense"]], s=60, color=COLOR[arm], edgecolors=A.INK, zorder=4)
        ax.set_xscale("log"); ax.set_yscale("log")
        A.style(ax, "R_train", "R_dense" if s == SEEDS[0] else "", f"seed {s}: trajectory from 256k (dot = final)")
    axs[0].legend(frameon=False, fontsize=7, labelcolor=A.INK2)
    fig.tight_layout(); save(fig, "Rtrain_vs_Rdense")

    # temporal displacement and error (final, seed 1234)
    fig, axs = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    t, up, vp, ue, ve = TR[("B1", 1234)]
    axs[0].plot(t, ue, color=A.INK, lw=1, label="exact")
    for arm in ARMS:
        tt, u, *_ = TR[(arm, 1234)]
        axs[0].plot(tt, u, color=COLOR[arm], lw=1.3, label=LABEL[arm])
        axs[1].plot(tt, u - ue, color=COLOR[arm], lw=1.1, label=LABEL[arm])
    A.style(axs[0], "", "u(L/2,t) [m]", "Mid-span displacement, final models (seed 1234)")
    A.style(axs[1], "t [s]", "u - u_exact [m]", "Mid-span displacement error")
    axs[0].legend(frameon=False, fontsize=7, labelcolor=A.INK2, ncol=3)
    fig.tight_layout(); save(fig, "displacement_and_error")

    for xkey, xlab, name in (("train_seconds", "training seconds (same-batch)", "wallclock_vs_accuracy"),
                             ("peak_rss_mb", "peak RAM [MB]", "memory_vs_accuracy")):
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
        for arm in ARMS:
            for s in SEEDS:
                x = F[(arm, s)]
                for ax, key in ((axs[0], "L2_exact"), (axs[1], "persistence_cycles")):
                    ax.scatter(x[xkey], x[key], s=60, marker=MARK[s], color=COLOR[arm], edgecolors=A.SURF, linewidths=1.2,
                               zorder=3, label=f"{arm} s{s}" if ax is axs[0] else None)
        axs[0].set_yscale("log")
        A.style(axs[0], xlab, "L2_exact (lower better)", f"Accuracy vs {xlab.split(' (')[0]}")
        A.style(axs[1], xlab, "persistence [cycles] (higher better)", f"Persistence vs {xlab.split(' (')[0]}")
        axs[0].legend(frameon=False, fontsize=6.5, labelcolor=A.INK2, ncol=3)
        fig.tight_layout(); save(fig, name)

    # accuracy / compute / memory Pareto: L2 vs wall-clock, marker area ~ RAM; non-dominated ringed
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, s in zip(axs, SEEDS):
        nd = [a for a in ARMS if not any(dominates(F[(o, s)], F[(a, s)]) for o in ARMS if o != a)]
        for arm in ARMS:
            x = F[(arm, s)]
            ax.scatter(x["train_seconds"], x["L2_exact"], s=x["peak_rss_mb"] / 6, color=COLOR[arm],
                       edgecolors=A.INK if arm in nd else A.SURF, linewidths=1.8 if arm in nd else 1, zorder=3, label=LABEL[arm])
            ax.annotate(f"{arm}\n{x['peak_rss_mb']:.0f} MB", (x["train_seconds"], x["L2_exact"]), xytext=(8, 0),
                        textcoords="offset points", color=A.INK2, fontsize=7, va="center")
        ax.set_yscale("log")
        A.style(ax, "training seconds", "L2_exact" if s == SEEDS[0] else "",
                f"seed {s}: area = peak RAM; dark ring = non-dominated (L2, P, W, RAM)")
    axs[0].legend(frameon=False, fontsize=7, labelcolor=A.INK2, markerscale=0.5)
    fig.tight_layout(); save(fig, "pareto_accuracy_compute_memory")


def main():
    out = assert_safe_output(REPO / "results_batch2", strict=True, purpose="B2-E03")
    fig_dir = assert_safe_output(REPO / "results_batch2" / "figures", strict=True, purpose="B2-E03")
    torch.set_num_threads(4)
    F, SN, TR, COLL, rows, snaps = {}, {}, {}, {}, [], []
    for s in SEEDS:
        for arm in ARMS:
            f, S, tr, coll = evaluate(arm, s)
            F[(arm, s)], SN[(arm, s)], TR[(arm, s)], COLL[(arm, s)] = f, S, tr, coll
            rows.append(f); snaps.extend(S)
            print(f"s{s} {arm:9s} L2e {f['L2_exact']:.4f} P {f['persistence_cycles']:.2f} Rd {f['R_dense']:.4f} "
                  f"Rt {f['R_train']:.4f} G {f['G']:.1f} freq {f['frequency_error_exact']:+.4f} decay {f['fit_decay']:.2f} "
                  f"E {f['pde_evaluations']} W {f['train_seconds']:.0f}s RAM {f['peak_rss_mb']:.0f}")
    integ = integrity(F, COLL)
    cls = decide(F)
    cls["integrity"] = integ
    with open(out / "B2-E03_RESULTS.csv", "x", newline="") as f:
        cols = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    with open(out / "B2-E03_SNAPSHOTS.csv", "x", newline="") as f:
        cols = list(dict.fromkeys(k for r in snaps for k in r))
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(snaps)
    with open(out / "B2-E03_CLASSIFICATION.json", "x") as f:
        json.dump(cls, f, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
    figures(F, SN, TR, fig_dir)
    print(json.dumps({k: cls[k] for k in ("CASE", "H1", "H2", "H3", "H4", "chunking_confounded", "counts")}, indent=2, default=float))


if __name__ == "__main__":
    main()
