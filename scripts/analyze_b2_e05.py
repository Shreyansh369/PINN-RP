"""B2-E05 Block-3 evaluator: implements docs/hypotheses/B2-E05.md sections 6-8 exactly (no new choices).

Reads the immutable run directories results_batch2/runs/B2-E05-<arm>-s<seed>/ and (re)generates the DERIVED
tables results_batch2/B2-E05_RESULTS.csv, results_batch2/B2-E05_CLASSIFICATION.json and
results_batch2/reports/B2-E05_REPORT.md. Regenerating these derived files as more runs finish is deliberate;
run outputs themselves are never touched.

    python scripts/analyze_b2_e05.py
"""
import csv
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from physref.safety import assert_safe_output  # noqa: E402

RUNS = REPO / "results_batch2" / "runs"
OUT_CSV = REPO / "results_batch2" / "B2-E05_RESULTS.csv"
OUT_JSON = REPO / "results_batch2" / "B2-E05_CLASSIFICATION.json"
OUT_MD = REPO / "results_batch2" / "reports" / "B2-E05_REPORT.md"
B2E01_B1 = "B2-E01-B1-s{seed}"

# pre-registration section 6 constants
W = 129.3246            # omega_d [rad/s]
ZETA = 3.54             # gamma / 2 [1/s]
D_MAX = 200.0
LOGGED_B1_L2 = {1234: 0.52639}
ARMS = [("W1000", 1.0), ("W250", 0.25), ("W100", 0.10), ("W050", 0.05)]


def J(d, E, T_w, Delta=0.0, w=W, zeta=ZETA):
    lam = zeta + d
    g = -math.expm1(-2.0 * lam * T_w) / (2.0 * lam)
    return ((d * d + Delta) ** 2 + 4.0 * w * w * d * d + E) * g


def d_star(E, T_w):
    if E <= 0:
        return 0.0
    s = minimize_scalar(lambda d: J(d, E, T_w), bounds=(-ZETA + 1e-9, D_MAX), method="bounded",
                        options={"xatol": 1e-10, "maxiter": 5000})
    return float(s.x)


def load(arm, seed):
    base = RUNS / f"B2-E05-{arm}-s{seed}"
    ms = glob.glob(str(base / "logs" / "*" / "metrics.json"))
    if not ms:
        return None
    m = json.load(open(ms[0]))
    m["_dir"] = str(base.relative_to(REPO))
    return m


def predict(m):
    Rk, T_w, d = m["R_k"], m["T_w"], m["decay_error"]
    dk = [d_star(R * R, T_w) for R in Rk]
    dk_i = [d_star(max(R * R - 4 * W * W * d * d, 0.0), T_w) for R in Rk]
    return {"predicted_d*": float(np.mean(dk)), "predicted_d*_k": dk,
            "predicted_d*_sens_i_E_minus_4w2d2": float(np.mean(dk_i)),
            "predicted_d*_sens_ii_median": float(np.median(dk))}


def rho(pred, meas):
    return None if meas <= 0 else pred / meas


def cost_per_eval(m):
    return (m["train_seconds"] + m.get("handoff_seconds", 0.0) + m.get("rk_seconds", 0.0)) / m["pde_evaluations"]


def harness(m, seed):
    out = {"W1000_L2_exact": m["L2_exact"], "logged_B1_L2_exact": LOGGED_B1_L2.get(seed)}
    if seed in LOGGED_B1_L2:
        out["match_4_significant_digits"] = f"{m['L2_exact']:.4g}" == f"{LOGGED_B1_L2[seed]:.4g}"
    hist = glob.glob(str(RUNS / B2E01_B1.format(seed=seed) / "logs" / "*" / "history.csv"))
    mine = glob.glob(str(REPO / m["_dir"] / "logs" / "*" / "history.csv"))
    if hist and mine:
        def losses(p):
            return {int(r["step"]): float(r["loss"]) for r in csv.DictReader(open(p)) if r.get("loss")}
        a, b = losses(hist[0]), losses(mine[0])
        steps = [s for s in (1000, 2000, 3000, 4000, 5000) if s in a and s in b]
        out["loss_at_steps"] = {s: {"logged_B2-E01": a[s], "W1000": b[s], "rel_diff": abs(a[s] - b[s]) / abs(a[s])}
                                for s in steps}
        common = sorted(set(a) & set(b))
        out["loss_identical_all_logged_steps"] = all(a[s] == b[s] for s in common) if common else None
    return out


def gate(rows):
    d = {r["T_w"]: r["decay_error"] for r in rows}
    p = {r["T_w"]: r["predicted_d*"] for r in rows}
    seq = [d[T] for _, T in ARMS]
    mono = all(x > y for x, y in zip(seq, seq[1:]))
    rs = {T: rho(p[T], d[T]) for T in (0.10, 0.05)}
    inside = lambda r, lo, hi: r is not None and lo <= r <= hi
    if not mono:
        label = "FAIL (falsifier 1)"
    elif all(inside(rs[T], 0.5, 2.0) for T in rs):
        label = "PASS"
    elif all(inside(rs[T], 0.2, 5.0) for T in rs):
        label = "PARTIAL"
    else:
        label = "FAIL (falsifier 2)"
    sp = spearmanr([T for _, T in ARMS], seq)
    return {"H-E05a_monotone": mono, "d_by_T_w": {str(T): d[T] for _, T in ARMS},
            "predicted_by_T_w": {str(T): p[T] for _, T in ARMS},
            "rho_pred_over_meas": {str(T): rs[T] for T in rs}, "spearman_Tw_vs_d": float(sp.statistic),
            "label": label}


COLS = ["arm", "seed", "T_w", "N_windows", "L2_exact", "L2_late_exact", "persistence_cycles", "persistence_cycles_vel",
        "amp_ratio_t0.1", "amp_ratio_t0.25", "amp_ratio_t0.5", "amp_ratio_t0.75", "amp_ratio_t1", "fit_w", "fit_lam",
        "decay_error", "frequency_error_exact", "phase_error_exact", "PDE_residual_rel", "R_k_mean", "R_k_max", "R_k",
        "predicted_d*", "predicted_d*_sens_i_E_minus_4w2d2", "predicted_d*_sens_ii_median", "rho_pred_over_meas",
        "train_seconds", "handoff_seconds", "rk_seconds", "cost_per_eval_s", "cost_ratio_vs_W1000",
        "pde_evaluations", "peak_rss_mb", "parameters", "handoff_projection_error_max", "status", "run_dir"]


def main():
    rows = []
    for seed in (1234, 1235, 1236):
        ref = load("W1000", seed)
        for arm, T in ARMS:
            m = load(arm, seed)
            if m is None:
                continue
            r = {k: m.get(k) for k in COLS}
            r.update(arm=arm, seed=seed, run_dir=m["_dir"], R_k=json.dumps([round(x, 3) for x in m["R_k"]]))
            pr = predict(m)
            r.update({k: v for k, v in pr.items() if k in COLS})
            rr = rho(pr["predicted_d*"], m["decay_error"])
            r["rho_pred_over_meas"] = rr
            r["cost_per_eval_s"] = cost_per_eval(m)
            r["cost_ratio_vs_W1000"] = cost_per_eval(m) / cost_per_eval(ref) if ref else None
            r["_pred_k"] = pr["predicted_d*_k"]
            rows.append(r)
    for p in (OUT_CSV, OUT_JSON, OUT_MD):
        assert_safe_output(p, strict=True, purpose="B2-E05 evaluator")
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    cls = {"pre_registration": "docs/hypotheses/B2-E05.md", "constants": {"w": W, "zeta": ZETA, "d_max": D_MAX},
           "runs_found": [f"{r['arm']}-s{r['seed']}" for r in rows]}
    s1234 = [r for r in rows if r["seed"] == 1234]
    w1000 = load("W1000", 1234)
    if w1000:
        cls["harness_check_s1234"] = harness(w1000, 1234)
    if len(s1234) == len(ARMS):
        cls["gate_seed1234"] = gate(s1234)
        cls["H-E05d_cost_ratio_vs_W1000"] = {r["arm"]: r["cost_ratio_vs_W1000"] for r in s1234}
        cls["H-E05d_within_1.5x"] = all(r["cost_ratio_vs_W1000"] <= 1.5 for r in s1234)
    else:
        cls["gate_seed1234"] = "pending: " + ", ".join(a for a, _ in ARMS if a not in {r["arm"] for r in s1234})
    with open(OUT_JSON, "w") as f:
        json.dump(cls, f, indent=2, default=str)
    lines = ["# B2-E05 report (auto-generated by scripts/analyze_b2_e05.py; derived from run dirs)", "",
             "Pre-registration: docs/hypotheses/B2-E05.md. B1 reference on THIS machine = W1000 of the same seed.", "",
             "| arm | seed | T_w | L2_exact | L2_late | persist. (cyc) | pers. vel | amp 0.25 | amp 0.5 | amp 1.0 | fit_lam | d meas | d* pred | rho | R_k mean | proj err max | train s | cost x W1000 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f3 = lambda v: "-" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v))
    for r in rows:
        lines.append("| " + " | ".join(f3(r[k]) for k in ["arm", "seed", "T_w", "L2_exact", "L2_late_exact",
                     "persistence_cycles", "persistence_cycles_vel", "amp_ratio_t0.25", "amp_ratio_t0.5", "amp_ratio_t1",
                     "fit_lam", "decay_error", "predicted_d*", "rho_pred_over_meas", "R_k_mean",
                     "handoff_projection_error_max", "train_seconds", "cost_ratio_vs_W1000"]) + " |")
    lines += ["", "## Classification", "", "```", json.dumps(cls, indent=2, default=str), "```", ""]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
