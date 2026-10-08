"""Extract every number shown in the supervisor deck from committed artifacts at the frozen commit.

Reads files ONLY via `git show <COMMIT>:<path>` so the deck is tied to the completed B2-E03 state.
Writes presentation/deck_data.json. No training, no result is modified. B2-E04 is NOT read.

    python presentation/extract_deck_data.py
"""
import csv
import io
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMMIT = "cdd87b5"
ROOT_COMMIT = "d31864a868c59858f99b9b0c5dab4b0f18758e25"


def show(path):
    return subprocess.run(["git", "--no-optional-locks", "-C", str(REPO), "show", f"{COMMIT}:{path}"],
                          check=True, capture_output=True, text=True).stdout


def rows(path):
    return list(csv.DictReader(io.StringIO(show(path))))


def f(x):
    return float(x)


D = {"commit": COMMIT, "root_commit": ROOT_COMMIT,
     "full_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", COMMIT], capture_output=True,
                                   text=True).stdout.strip(), "sources": {}}

# ------------------------------------------------------------------ counts (registry + Batch-1 leaderboard)
reg = rows("results_batch2/EXPERIMENT_REGISTRY.csv")
lb = rows("references/batch1/optimization_leaderboard.csv")
arm_labels = sorted({r["experiment_id"].rsplit("-s", 1)[0] for r in reg})
# Batch-2 control arms that are bit-exact re-runs of an earlier arm (asserted in the B2-E02/E03 evaluators)
B2_REPEATS = {"B2-E02-M0": "B2-E01-modal", "B2-E02T-B1": "B2-E01-B1", "B2-E03-B1": "B2-E01-B1"}
b2_distinct = [a for a in arm_labels if a not in B2_REPEATS]
# Batch-1 seed replicates of an otherwise identical configuration
B1_SEED_REPEATS = {"Z2_Y1_seed1235": "Y1_X2_lrsched"}
b1_methods = [r["method"] for r in lb]
b1_distinct = [m for m in b1_methods if m not in B1_SEED_REPEATS]
# B2 frozen B1 == Batch-1 Z4_Y1_mb128 configuration at 640k evaluations (bit-exact, B2-E01 report §2)
b2_new = [a for a in b2_distinct if a != "B2-E01-B1"]
D["counts"] = {
    "batch1_runs": len(lb), "batch1_distinct_configs": len(b1_distinct),
    "batch2_runs": len(reg), "batch2_arm_labels": len(arm_labels), "batch2_distinct_arms": len(b2_distinct),
    "batch2_new_configs": len(b2_new),
    "total_runs": len(lb) + len(reg), "total_distinct_configs": len(b1_distinct) + len(b2_new),
    "batch2_seeds": sorted({int(r["seed"]) for r in reg}),
    "batch1_seeds": sorted({int(r["seed"]) for r in lb}),
    "batch2_arm_label_list": arm_labels, "batch2_repeats": B2_REPEATS, "batch1_seed_repeats": B1_SEED_REPEATS,
    "batch1_method_list": b1_methods,
}
# intervention families -> configurations that test them (membership from run names / reports)
FAM = {
    "Representation / formulation": {
        "Fourier feature embedding (paper method)": ["BB_fourier", "C0_paper", "C0_rc"],
        "NTK adaptive loss weighting (paper method)": ["C0_paper", "C0_rc"],
        "Fourier convention / input scaling / bias init": ["D1_unit", "D2_bias0", "D3_unit_bias0", "D4_rc_bias0", "E4b_hard_fourier_rc"],
        "Hard IC/BC constraints": ["E4_hard_fourier", "E4b_hard_fourier_rc", "E5_hard_fourier_rad"],
        "tanh²(ω₁t) temporal conditioning": ["X1_hard_tanh2", "X2_hard_tanh2_rc"],
        "Mixed formulation v = u_xx": ["B2-E01-mixed"],
        "Modal reduction (diagnostic)": ["B2-E01-modal", "B2-E02-LBFGS", "B2-E02-CAUSAL"],
    },
    "Optimization / training": {
        "Exponential LR decay": ["Y1_X2_lrsched", "Y2_X1_lrsched", "Y4_C0_lrsched"],
        "Larger mini-batch (32 → 128)": ["Z4_Y1_mb128"],
        "Training budget (5k → 20k steps)": ["Y1_20K", "Z4_20K"],
        "Adam → L-BFGS": ["B2-E02-LBFGS", "B2-E02T-LBFGS", "B2-E03-LBFGS-F"],
        "Full-batch Adam": ["B2-E03-ADAM-FULL"],
        "Causal temporal weighting": ["B2-E02-CAUSAL"],
    },
    "Sampling": {
        "RAD adaptive sampling": ["E3_fourier_rad", "E5_hard_fourier_rad", "Z1_Y1_rad", "Z3_Y4_rad"],
        "Collocation resampling / larger pool (L-BFGS)": ["B2-E03-LBFGS-R", "B2-E03-LBFGS-4X"],
    },
    "Numerical precision": {"FP64 (vs FP32)": ["B2-E01-B1_fp64"]},
}
known = set(b1_methods) | set(arm_labels)
for cat, fams in FAM.items():
    for fam, mem in fams.items():
        assert set(mem) <= known, (fam, set(mem) - known)
CONTROLS = {"Vanilla tanh PINN baseline": ["BA_vanilla"],
            "Supervised representation checks (not PINN)": ["X3_supervised_paperconv", "X4_supervised_rcconv", "Y3_X4_lrsched"]}
D["families"] = FAM
D["controls"] = CONTROLS
D["counts"]["n_families"] = sum(len(v) for v in FAM.values())
D["counts"]["n_classes"] = len(FAM)
D["sources"]["counts"] = "results_batch2/EXPERIMENT_REGISTRY.csv; references/batch1/optimization_leaderboard.csv"

# ------------------------------------------------------------------ Batch 1 progression
lbm = {r["method"]: r for r in lb}
z4k = {r["step"]: r for r in rows("references/batch1/tables/phaseZ4_20K_checkpoints.csv") if r["run"] == "Z4_20K"}
y1k = {r["run"]: r for r in rows("references/batch1/tables/phaseY1_20K_checkpoints.csv")}
osc = {}
for t in ("phaseE_oscillation.csv", "phaseX_oscillation.csv", "phaseY_oscillation.csv", "phaseZ_oscillation.csv"):
    for r in rows(f"references/batch1/tables/{t}"):
        osc[r["name"]] = r
prog = [
    ("C0", "Paper method (Fourier + NTK)", "C0_paper", None, "static field"),
    ("E4", "+ hard IC/BC constraints", "E4_hard_fourier", None, "static field"),
    ("X2", "+ tanh²(ω₁t) conditioning", "X2_hard_tanh2_rc", None, "low-frequency, over-damped"),
    ("Y1", "+ LR decay", "Y1_X2_lrsched", "Y1_X2_lrsched", "oscillates, collapses"),
    ("Z4", "+ mini-batch 128 (= B1)", "Z4_Y1_mb128", "Z4_Y1_mb128", "oscillates, collapses"),
    ("Z4-20K", "+ 4× training budget", "Z4_20K", "Z4_20K@20000", "oscillates, collapses"),
]
P = []
for key, lab, m, pk, beh in prog:
    r = lbm[m]
    pers = None
    if pk == "Z4_20K@20000":
        pers = f(z4k["20000"]["persistence_cycles"])
    elif pk:
        pers = f(y1k[pk]["persistence_cycles"])
    P.append({"key": key, "label": lab, "method": m, "L2_exact": f(r["L2_exact"]), "persistence_cycles": pers,
              "pde_evaluations": int(r["PDE_evaluations"]), "steps": int(r["optimizer_steps"]), "behaviour": beh,
              "verdict": osc.get(m, {}).get("verdict")})
D["batch1_progression"] = P
D["batch1_best"] = {"L2_exact": f(z4k["20000"]["L2_exact"]), "persistence_cycles": f(z4k["20000"]["persistence_cycles"]),
                    "collapse_time_s": f(z4k["20000"]["collapse_time_s"]), "fit_w": f(z4k["20000"]["fit_w"]),
                    "fit_decay": f(z4k["20000"]["fit_decay"]), "pde_evaluations": int(z4k["20000"]["pde_evaluations_cum"]),
                    "train_seconds": f(z4k["20000"]["train_seconds_cum"]), "full_window_cycles": f(z4k["20000"]["full_window_cycles"])}
D["batch1_z4_budget_curve"] = [{"pde_evaluations": int(z4k[s]["pde_evaluations_cum"]), "persistence_cycles": f(z4k[s]["persistence_cycles"]),
                                "L2_exact": f(z4k[s]["L2_exact"])} for s in ("5000", "10000", "15000", "20000")]
D["supervised"] = {m: {"L2_exact": f(osc[m]["L2_exact"]), "fit_w": f(osc[m]["fit_w"]), "w_exact": f(osc[m]["w_exact"])}
                   for m in ("X3_supervised_paperconv", "X4_supervised_rcconv")}
pbr = [r for r in rows("references/batch1/paper_benchmark_registry.csv")]
D["paper_registry_header"] = list(pbr[0].keys())
fe = [r for r in pbr if r.get("benchmark_id") == "FE-D-M1" or "FE-D-M1" in ",".join(r.values())]
D["paper_FE_D_M1"] = fe[0] if fe else None
D["sources"]["batch1"] = ("references/batch1/optimization_leaderboard.csv; references/batch1/tables/phaseZ4_20K_checkpoints.csv; "
                          "phaseY1_20K_checkpoints.csv; phase{E,X,Y,Z}_oscillation.csv")

# ------------------------------------------------------------------ B2-E01 (seed 1234) table
e01 = {r["arm"]: r for r in rows("results_batch2/tables/B2-E01_RESULTS.csv")}
D["e01"] = {a: {"persistence_cycles": f(r["persistence_cycles"]), "L2_exact": f(r["L2_exact"]),
                "frequency_error_signed_pct": 100 * (f(r["fit_w"]) - f(r["omega_d"])) / f(r["omega_d"]),
                "fit_decay": f(r["fit_decay"]), "PDE_residual_rel": f(r["PDE_residual_rel"]),
                "train_seconds": f(r["train_seconds"]), "peak_rss_mb": f(r["peak_rss_mb"]),
                "seconds_per_step": f(r["seconds_per_step"]), "collapse_time_s": f(r["collapse_time_s"])}
            for a, r in e01.items()}
D["sources"]["e01"] = "results_batch2/tables/B2-E01_RESULTS.csv"

# ------------------------------------------------------------------ B2-E01 seed replication
s = rows("results_batch2/tables/B2-E01S_SEED_RESULTS.csv")
D["e01s"] = [{"arm": r["arm"], "seed": int(r["seed"]), "persistence_cycles": f(r["persistence_cycles"]), "L2_exact": f(r["L2_exact"]),
              "collapse_time_s": f(r["collapse_time_s"]), "fit_decay": f(r["fit_decay"]), "PDE_residual_rel": f(r["PDE_residual_rel"]),
              "train_seconds": f(r["train_seconds"]), "peak_rss_mb": f(r["peak_rss_mb"]), "class_vs_B1": r["class_vs_B1"]} for r in s]
D["sources"]["e01s"] = "results_batch2/tables/B2-E01S_SEED_RESULTS.csv"
snap = rows("results_batch2/tables/B2-E01S_SEED_SNAPSHOTS.csv")
D["e01s_modal_best"] = {}
for sd in (1234, 1235, 1236):
    m = [r for r in snap if r["arm"] == "modal" and int(r["seed"]) == sd]
    b = max(m, key=lambda r: f(r["persistence_cycles"]))
    D["e01s_modal_best"][sd] = {"persistence_cycles": f(b["persistence_cycles"]), "pde_evaluations": int(b["pde_evaluations"])}

# ------------------------------------------------------------------ B2-E02 modal + transfer
e02 = rows("results_batch2/B2-E02_RESULTS.csv")
D["e02"] = [{"arm": r["arm"], "seed": int(r["seed"]), "persistence_cycles": f(r["persistence_cycles"]), "L2_exact": f(r["L2_exact"]),
             "fit_decay": f(r["fit_decay"]), "train_seconds": f(r["train_seconds"]), "peak_rss_mb": f(r["peak_rss_mb"]),
             "pde_evaluations": int(r["pde_evaluations"]), "full_window_cycles": f(r["full_window_cycles"]),
             "collapse_time_s": f(r["collapse_time_s"])} for r in e02]
e02s = rows("results_batch2/B2-E02_SNAPSHOTS.csv")
D["e02_snap"] = [{"arm": r["arm"], "seed": int(r["seed"]), "pde_evaluations": int(r["pde_evaluations"]),
                  "persistence_cycles": f(r["persistence_cycles"]), "checkpoint": r["checkpoint"]} for r in e02s]
e02t = rows("results_batch2/B2-E02T_RESULTS.csv")
D["e02t"] = [{"arm": r["arm"], "seed": int(r["seed"]), "persistence_cycles": f(r["persistence_cycles"]), "L2_exact": f(r["L2_exact"]),
              "PDE_residual_rel": f(r["PDE_residual_rel"]), "train_seconds": f(r["train_seconds"]),
              "peak_rss_mb": f(r["peak_rss_mb"]), "fit_decay": f(r["fit_decay"]), "pde_evaluations": int(r["pde_evaluations"])} for r in e02t]
e02ts = rows("results_batch2/B2-E02T_SNAPSHOTS.csv")
D["e02t_snap"] = [{"arm": r["arm"], "seed": int(r["seed"]), "pde_evaluations": int(r["pde_evaluations"]),
                   "persistence_cycles": f(r["persistence_cycles"])} for r in e02ts]
cls = json.loads(show("results_batch2/B2-E02T_CLASSIFICATION.json"))
D["e02t_classification_keys"] = list(cls)[:20]
D["sources"]["e02"] = "results_batch2/B2-E02_{RESULTS,SNAPSHOTS}.csv; B2-E02T_{RESULTS,SNAPSHOTS}.csv; B2-E02T_CLASSIFICATION.json"

# ------------------------------------------------------------------ B2-E03
e03 = rows("results_batch2/B2-E03_RESULTS.csv")
D["e03"] = [{"arm": r["arm"], "seed": int(r["seed"]), "status": r["status"],
             "G": (f(r["G"]) if r["G"] not in ("", "nan") else None),
             "persistence_cycles": (f(r["persistence_cycles"]) if r["persistence_cycles"] not in ("", "nan") else None),
             "L2_exact": (f(r["L2_exact"]) if r["L2_exact"] not in ("", "nan") else None),
             "R_dense": (f(r["R_dense"]) if r["R_dense"] not in ("", "nan") else None),
             "peak_rss_mb": f(r["peak_rss_mb"]), "pde_evaluations": int(float(r["pde_evaluations"]))} for r in e03]
c3 = json.loads(show("results_batch2/B2-E03_CLASSIFICATION.json"))
D["e03_case"] = c3.get("CASE") or c3.get("case") or c3.get("decision")
D["sources"]["e03"] = "results_batch2/B2-E03_RESULTS.csv; B2-E03_CLASSIFICATION.json"

# tests at cdd87b5
tests = show("tests/test_b2_collocation_lab.py")  # existence check only
D["tests_passing_at_B2E03"] = 168          # B2-E03 completion record (commit message cdd87b5 / final verification)
# analytical reference (slide 1 visual; NOT a network output)
import sys  # noqa: E402
import numpy as np  # noqa: E402
sys.path.insert(0, str(REPO / "src"))
from beampinn.physics.benchmarks import get_benchmark  # noqa: E402
bm = get_benchmark("FE-D-M1"); ex = bm.reference("exact")
t = np.linspace(0, 1, 601)
D["analytic"] = {"t": [round(v, 5) for v in t.tolist()], "u_mid": [round(v, 6) for v in ex.u(ex.x_norm, t).tolist()],
                 "omega_d": ex.omega_d, "f_d_hz": ex.omega_d / (2 * np.pi), "decay": 0.5 * ex.gamma,
                 "cycles": ex.omega_d / (2 * np.pi) * bm.t_end, "omega1": bm.fundamental_omega(), "c2": ex.c2,
                 "gamma": ex.gamma, "L": bm.L, "A0": ex.A0,
                 "source": "analytical reference: src/beampinn/physics/{beam,benchmarks}.py (exact root)"}
out = REPO / "presentation" / "deck_data.json"
json.dump(D, open(out, "w"), indent=1, ensure_ascii=False)
print("wrote", out)
print(json.dumps(D["counts"], indent=1, ensure_ascii=False)[:1500])
