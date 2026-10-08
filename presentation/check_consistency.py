"""Read-only consistency audit of PINN_Research_Progress_Supervisor.pptx against committed artifacts at cdd87b5.

1. Every number registered by build_deck.js (displayed_numbers.json) appears verbatim in the deck text.
2. Key numbers are RECOMPUTED here, independently of the deck builder, from the CSVs at cdd87b5 and compared.
3. Method counts are recomputed from the experiment registry and the Batch-1 leaderboard.
4. Scope / wording rules (no B2-E04, modal marked diagnostic, no optimized-PINN claim, railway = future,
   failures qualified "on this benchmark", B1 = full-field reference, forbidden words).

    python presentation/check_consistency.py     -> presentation/consistency_check.txt (exit 1 on any failure)
"""
import csv
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMMIT = "cdd87b5"
DECK = HERE / "PINN_Research_Progress_Supervisor.pptx"
LOG = []
FAIL = []


def check(name, ok, detail=""):
    LOG.append(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def show(path):
    return subprocess.run(["git", "--no-optional-locks", "-C", str(REPO), "show", f"{COMMIT}:{path}"],
                          check=True, capture_output=True, text=True).stdout


def rows(path):
    return list(csv.DictReader(io.StringIO(show(path))))


# ------------------------------------------------------------------ deck text per slide
z = zipfile.ZipFile(DECK)
slide_files = sorted([n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)],
                     key=lambda n: int(re.findall(r"\d+", n)[0]))
SLIDES = []
for n in slide_files:
    xml = z.read(n).decode("utf8")
    paras = re.findall(r"<a:p>.*?</a:p>", xml, re.S)
    txt = "\n".join("".join(re.findall(r"<a:t>(.*?)</a:t>", p, re.S)) for p in paras)
    txt = txt.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'")
    SLIDES.append(txt)
ALL = "\n".join(SLIDES)
# chart data labels (formatted as the charts display them: 0.00) are added for the presence check
CHART_VALS = []
for n in z.namelist():
    if re.fullmatch(r"ppt/charts/chart\d+\.xml", n):
        for v in re.findall(r"<c:v>([-0-9.eE]+)</c:v>", z.read(n).decode("utf8")):
            CHART_VALS.append(f"{float(v):.2f}")
ALL_WITH_CHARTS = ALL + "\n" + " ".join(CHART_VALS)
check("slide count", len(SLIDES) == 14, f"{len(SLIDES)} slides")

# ------------------------------------------------------------------ 1. registered numbers present
shown = json.load(open(HERE / "displayed_numbers.json"))
miss = [s for s in shown if s["text"] not in ALL_WITH_CHARTS]
check("every registered displayed number is present in the deck", not miss, f"{len(shown)} checked; missing: {[m['label'] for m in miss]}")

# ------------------------------------------------------------------ 2. independent recomputation
f = float
SE = (1234, 1235, 1236)


def rg(v, d):
    a, b = f"{min(v):.{d}f}", f"{max(v):.{d}f}"
    return a if a == b else f"{a}–{b}"


e01 = {r["arm"]: r for r in rows("results_batch2/tables/B2-E01_RESULTS.csv")}
exp = {
    "B2-E01 B1 persistence 3.15": f"{f(e01['B1']['persistence_cycles']):.2f}" == "3.15",
    "B2-E01 modal persistence 7.62": f"{f(e01['modal']['persistence_cycles']):.2f}" == "7.62",
    "B2-E01 mixed persistence 1.67": f"{f(e01['mixed']['persistence_cycles']):.2f}" == "1.67",
    "B2-E01 L2 0.526/0.675/0.270": [f"{f(e01[a]['L2_exact']):.3f}" for a in ("B1", "mixed", "modal")] == ["0.526", "0.675", "0.270"],
    "B2-E01 decay 8.98/13.24/5.40": [f"{f(e01[a]['fit_decay']):.2f}" for a in ("B1", "mixed", "modal")] == ["8.98", "13.24", "5.40"],
    "B2-E01 residual 0.110/2.57/0.074": [f"{f(e01['B1']['PDE_residual_rel']):.3f}", f"{f(e01['mixed']['PDE_residual_rel']):.2f}", f"{f(e01['modal']['PDE_residual_rel']):.3f}"] == ["0.110", "2.57", "0.074"],
    "B2-E01 time 822/1404/374/190": [round(f(e01[a]['train_seconds'])) for a in ("B1", "B1_fp64", "mixed", "modal")] == [822, 1404, 374, 190],
    "B2-E01 RAM 934/1178/730/704": [round(f(e01[a]['peak_rss_mb'])) for a in ("B1", "B1_fp64", "mixed", "modal")] == [934, 1178, 730, 704],
    "B2-E01 freq err -0.96/-2.0/-0.04 %": [round(100 * (f(e01[a]['fit_w']) - f(e01[a]['omega_d'])) / f(e01[a]['omega_d']), n) for a, n in (("B1", 2), ("mixed", 1), ("modal", 2))] == [-0.96, -2.0, -0.04],
}
for k, v in exp.items():
    check(k, v)
for s_ in ("3.15", "0.526", "−0.96 %", "−2.0 %", "−0.04 %", "8.98", "13.24", "5.40", "0.110", "2.57", "0.074", "1,404", "1,178"):
    check(f"B2-E01 table value '{s_}' shown", s_ in SLIDES[6])

s = rows("results_batch2/tables/B2-E01S_SEED_RESULTS.csv")
g = lambda arm, k: [f(r[k]) for r in s if r["arm"] == arm]  # noqa: E731
ratio = [f(next(r for r in s if r["arm"] == "modal" and int(r["seed"]) == sd)["persistence_cycles"]) /
         f(next(r for r in s if r["arm"] == "B1" and int(r["seed"]) == sd)["persistence_cycles"]) for sd in SE]
check("modal/B1 persistence ratios ×2.42, ×2.13, ×1.19", [f"{x:.2f}" for x in ratio] == ["2.42", "2.13", "1.19"] and "×2.42, ×2.13, ×1.19" in SLIDES[7])
check("modal collapse 0.296–0.370 s", rg(g("modal", "collapse_time_s"), 3) == "0.296–0.370" and "0.296–0.370 s" in SLIDES[7])
check("B1 seed range 3.11–5.10", rg(g("B1", "persistence_cycles"), 2) == "3.11–5.10" and "3.11–5.10" in SLIDES[7])

e02 = rows("results_batch2/B2-E02_RESULTS.csv")
a2 = lambda arm, k: [f(r[k]) for r in e02 if r["arm"] == arm]  # noqa: E731
check("modal L-BFGS full window 20.58 in 3/3", all(f"{x:.2f}" == "20.58" for x in a2("LBFGS", "persistence_cycles")))
check("modal L-BFGS L2 0.038–0.093", rg(a2("LBFGS", "L2_exact"), 3) == "0.038–0.093" and "0.038–0.093" in SLIDES[8])
check("M0 L2 0.27–0.33", rg(a2("M0", "L2_exact"), 2) == "0.27–0.33" and "0.27–0.33" in SLIDES[8])
check("M0 persistence 6.1–7.6", rg(a2("M0", "persistence_cycles"), 1) == "6.1–7.6" and "6.1–7.6" in SLIDES[8])
check("L-BFGS decay 3.62–4.10", rg(a2("LBFGS", "fit_decay"), 2) == "3.62–4.10" and "3.62–4.10" in SLIDES[8])
wr = [f(next(r for r in e02 if r["arm"] == "LBFGS" and int(r["seed"]) == sd)["train_seconds"]) /
      f(next(r for r in e02 if r["arm"] == "M0" and int(r["seed"]) == sd)["train_seconds"]) for sd in SE]
check("modal L-BFGS relative wall-clock 0.91–0.93", rg(wr, 2) == "0.91–0.93" and "0.91–0.93" in SLIDES[8])
check("causal persistence 2.1–4.6, L2 0.56–1.56", rg(a2("CAUSAL", "persistence_cycles"), 1) == "2.1–4.6" and rg(a2("CAUSAL", "L2_exact"), 2) == "0.56–1.56")

t = rows("results_batch2/B2-E02T_RESULTS.csv")
tr = {(r["arm"], int(r["seed"])): r for r in t}
want = {1234: ("−1.6", "×1.28", "×1.55"), 1235: ("−2.4", "×1.54", "×2.38"), 1236: ("−2.6", "×1.44", "×1.22")}
for sd, (dp, l2, rr) in want.items():
    b, l = tr[("B1", sd)], tr[("LBFGS", sd)]
    calc = (f"−{abs(f(l['persistence_cycles']) - f(b['persistence_cycles'])):.1f}", f"×{f(l['L2_exact']) / f(b['L2_exact']):.2f}",
            f"×{f(l['PDE_residual_rel']) / f(b['PDE_residual_rel']):.2f}")
    check(f"transfer seed {sd} {want[sd]}", calc == (dp, l2, rr) and all(x in SLIDES[9] for x in calc), str(calc))

e03 = rows("results_batch2/B2-E03_RESULTS.csv")
G = lambda arm: [f(r["G"]) for r in e03 if r["arm"] == arm and r["G"] not in ("", "nan")]  # noqa: E731
for arm, d, txt in (("LBFGS-F", 0, "35–76"), ("ADAM-FULL", 0, "4–6"), ("LBFGS-R", 1, "1.8–2.8"), ("LBFGS-4X", 1, "1.2–1.4"), ("B1", 1, "1.1")):
    check(f"B2-E03 G {arm} = {txt}×", rg(G(arm), d) == txt and f"{txt}×" in SLIDES[10], rg(G(arm), d))
c3 = json.loads(show("results_batch2/B2-E03_CLASSIFICATION.json"))
check("B2-E03 CASE D", "CASE D" in SLIDES[10] and json.dumps(c3).count('"D"') >= 1)

lb = rows("references/batch1/optimization_leaderboard.csv")
lbm = {r["method"]: r for r in lb}
check("C0 L2 3.26", f"{f(lbm['C0_paper']['L2_exact']):.2f}" == "3.26" and "3.26" in SLIDES[2])
z4 = {r["step"]: r for r in rows("references/batch1/tables/phaseZ4_20K_checkpoints.csv") if r["run"] == "Z4_20K"}
check("Batch-1 best 0.263 / 8.09 / 0.393 s", (f"{f(z4['20000']['L2_exact']):.3f}", f"{f(z4['20000']['persistence_cycles']):.2f}", f"{f(z4['20000']['collapse_time_s']):.3f}") == ("0.263", "8.09", "0.393") and "0.263" in SLIDES[11])
pbr = [r for r in rows("references/batch1/paper_benchmark_registry.csv") if r["benchmark_id"] == "FE-D-M1"][0]
check("paper L2 4.64e-4 from registry", pbr["paper_error"] == "4.64e-4" and "4.64e-4" in SLIDES[2])
check("paper citation present", "Söyleyici" in SLIDES[2] and "109804" in SLIDES[2] and "Engineering Applications of Artificial Intelligence" in SLIDES[2])

# ------------------------------------------------------------------ 3. counts
reg = rows("results_batch2/EXPERIMENT_REGISTRY.csv")
labels = {r["experiment_id"].rsplit("-s", 1)[0] for r in reg}
check("Batch-2 runs = 42", len(reg) == 42)
check("Batch-2 arm labels = 14, distinct = 11 (3 bit-exact control re-runs)", len(labels) == 14 and len(labels - {"B2-E02-M0", "B2-E02T-B1", "B2-E03-B1"}) == 11)
check("Batch-1 runs = 26, distinct = 25", len(lb) == 26 and len({r["method"] for r in lb} - {"Z2_Y1_seed1235"}) == 25)
check("total runs 68 and distinct configurations 35 shown", "68" in SLIDES[4] and "35" in SLIDES[4])
check("Batch-2 seeds = {1234,1235,1236}", sorted({int(r["seed"]) for r in reg}) == [1234, 1235, 1236])
fam = json.load(open(HERE / "deck_data.json"))["families"]
nf = sum(len(v) for v in fam.values())
check("16 families in 4 classes shown", nf == 16 and len(fam) == 4 and "16" in SLIDES[4] and "in 4 classes" in SLIDES[4])
reg_ids = {r["experiment_id"] for r in reg}
check("no B2-E04 run in the registry at cdd87b5", not any("E04" in i for i in reg_ids))

# ------------------------------------------------------------------ 4. scope and wording
check("no B2-E04 content in the deck", "E04" not in ALL)
check("future experiments not described as completed",
      all(w not in ALL.lower() for w in ("we validated on railway", "railway results", "deployed")))
for i, txt in enumerate(SLIDES, 1):
    low = txt.lower()
    if "modal" in low and i not in (1,):
        check(f"slide {i}: modal content marked diagnostic", "diagnostic" in low or "diag." in low)
    if "optimized pinn" in low:
        ok = all(("not" in seg or "no " in seg or "towards" in seg) for seg in [low])
        check(f"slide {i}: 'optimized PINN' only in negated/future form", ok)
    if "railway" in low and i not in (1, 14):
        check(f"slide {i}: railway framed as future", any(w in low for w in ("future", "not yet", "no railway", "target application", "not yet started")))
check("FAILED tags qualified 'ON THIS BENCHMARK'", ALL.count("FAILED ON THIS BENCHMARK") >= 1 and not re.search(r"FAILED(?! ON THIS BENCHMARK)", ALL))
check("full-field reference is B1 (0.526 / 3.15 cycles)", "L2 0.526" in SLIDES[11] and "3.15 cycles" in SLIDES[11] and "B1" in SLIDES[11])
check("no completion percentage", not re.search(r"\d+\s?% (complete|optimi[sz]ed|done)", ALL, re.I))
FORBIDDEN = ["revolutionary", "state-of-the-art", "state of the art", "guaranteed", "optimal", "novel architecture"]
check("forbidden words absent", not [w for w in FORBIDDEN if w in ALL.lower()], str([w for w in FORBIDDEN if w in ALL.lower()]))
# 'breakthrough' and 'solves' appear only in the sentences the brief itself mandates
check("'breakthrough' only in the mandated takeaway", [l for l in ALL.splitlines() if "breakthrough" in l.lower()] ==
      ["The next breakthrough must improve the full-field solution, not merely the diagnostic modal problem."])
check("'solved' not used", "solved" not in ALL.lower())
check("ROOT commit cited", "d31864a" in ALL)

out = HERE / "consistency_check.txt"
out.write_text("\n".join(LOG) + f"\n\n{'ALL CHECKS PASSED' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}\n")
print("\n".join(LOG[-5:]))
print("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
