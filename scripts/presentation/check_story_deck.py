"""QA for the research-story deck. Read-only; writes only its report under results_batch2/presentation/.

    python scripts/presentation/check_story_deck.py

1. Every displayed number is recomputed from a frozen Batch-1 source (references/batch1/*, byte-identical to
   ROOT @ d31864a) or from the frozen-checkpoint extraction, formatted as on the slide, and must appear on it.
2. Scope checks: paper number labelled as published; no failed Batch-1 run called successful; no Mode-2
   result; railway only as future work; no published method called novel; every chart slide asks a question.
"""
import csv
import json
import re
import sys
import zipfile
from pathlib import Path

RP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RP / "src"))
from physref.safety import assert_safe_output  # noqa: E402

PRES = RP / "results_batch2" / "presentation"
DECK = PRES / "final_research_story_presentation.pptx"
B1 = RP / "references" / "batch1"
D = json.loads((PRES / "data" / "deck_data.json").read_text())


def text(n, notes=False):
    z = zipfile.ZipFile(DECK)
    part = f"ppt/notesSlides/notesSlide{n}.xml" if notes else f"ppt/slides/slide{n}.xml"
    t = " ".join(re.findall(r"<a:t>([^<]*)</a:t>", z.read(part).decode("utf8")))
    return t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')


def n_slides():
    return len([n for n in zipfile.ZipFile(DECK).namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)])


def has_chart(n):
    rels = zipfile.ZipFile(DECK).read(f"ppt/slides/_rels/slide{n}.xml.rels").decode()
    return "chart" in rels


def chart_titles(n):
    z = zipfile.ZipFile(DECK)
    rels = z.read(f"ppt/slides/_rels/slide{n}.xml.rels").decode()
    out = ""
    for c in re.findall(r'charts/(chart\d+\.xml)', rels):
        out += " ".join(re.findall(r"<a:t>([^<]*)</a:t>", z.read(f"ppt/charts/{c}").decode()))
    return out


LB = {r["run_id"].split("__")[0]: r for r in csv.DictReader(open(B1 / "optimization_leaderboard.csv"))}
s3 = lambda k: f"{float(LB[k]['L2_exact']):#.3g}"            # 3 significant figures, as on the slides
tab = lambda f: list(csv.DictReader(open(B1 / "tables" / f)))
ck = lambda f, run, step: [r for r in tab(f) if r["run"] == run and int(r["step"]) == step][0]
X = {r["name"]: r for r in tab("phaseX_oscillation.csv")}
reg = [r for r in csv.DictReader(open(B1 / "paper_benchmark_registry.csv")) if r["benchmark_id"] == "FE-D-M1"][0]
rep = lambda name: (B1 / "reports" / name).read_text()
f1 = lambda v: f"{float(v):.1f}"
Y1_5, Y1_20 = ck("phaseY1_20K_checkpoints.csv", "Y1_20K", 5000), ck("phaseY1_20K_checkpoints.csv", "Y1_20K", 20000)
Z4_5, Z4_20 = ck("phaseZ4_20K_checkpoints.csv", "Z4_20K", 5000), ck("phaseZ4_20K_checkpoints.csv", "Z4_20K", 20000)

CHECKS = [  # (slide, shown, recomputed, source)
    (3, "4.64 × 10⁻⁴", "4.64 × 10⁻⁴" if float(reg["paper_error"]) == 4.64e-4 else reg["paper_error"], "paper_benchmark_registry.csv FE-D-M1 paper_error (published)"),
    (3, "20.6 Hz", f"{D['f_d']:.1f} Hz", "exact reference (beampinn)"),
    (4, "26", str(len(LB)), "optimization_leaderboard.csv rows"),
    (5, "L2 3.26", "L2 " + s3("C0_paper"), "optimization_leaderboard.csv C0_paper"),
    (5, "L2 0.909", "L2 " + s3("X2_hard_tanh2_rc"), "optimization_leaderboard.csv X2"),
    (5, "L2 0.526", "L2 " + s3("Z4_Y1_mb128"), "optimization_leaderboard.csv Z4"),
    (5, "L2 0.263", "L2 " + s3("Z4_20K"), "optimization_leaderboard.csv Z4_20K"),
    (5, "≈ 3.2 cycles", f"≈ {f1(Z4_5['persistence_cycles'])} cycles", "tables/phaseZ4_20K_checkpoints.csv 5k (= Z4)"),
    (5, "≈ 8.1 cycles", f"≈ {f1(Z4_20['persistence_cycles'])} cycles", "tables/phaseZ4_20K_checkpoints.csv 20k"),
    (6, "8.4 × 10³", f"{abs(D['conditioning']['range_t2'][0]) / 1e3:.1f} × 10³", "N*(t) from exact solution = profiles/phaseE_ansatz_conditioning.txt (−8368.7)"),
    (7, "129.25", f"{float(X['X3_supervised_paperconv']['fit_w']):.2f}", "tables/phaseX_oscillation.csv X3"),
    (7, "129.37", f"{float(X['X4_supervised_rcconv']['fit_w']):.2f}", "tables/phaseX_oscillation.csv X4"),
    (7, "129.32", f"{float(X['X4_supervised_rcconv']['w_exact']):.2f}", "tables/phaseX_oscillation.csv w_exact"),
    (7, "1.2 → 3.1 → 3.2 → 8.1", " → ".join(f1(r["persistence_cycles"]) for r in (Y1_5, Y1_20, Z4_5, Z4_20)), "checkpoint tables Y1/Z4 5k, 20k"),
    (7, "0.16M → 2.56M", f"{float(Y1_5['pde_evaluations_cum']) / 1e6:.2f}M → {float(Z4_20['pde_evaluations_cum']) / 1e6:.2f}M", "checkpoint tables"),
    (7, "1.6 cycles", "1.6 cycles" if "1.6 cycles (0.080 s)" in rep("PHASE_Y1_20K_BUDGET_DIAGNOSTIC.md") else "?", "reports/PHASE_Y1_20K_BUDGET_DIAGNOSTIC.md (Z1)"),
    (7, "L2 0.263", "L2 " + s3("Z4_20K"), "optimization_leaderboard.csv"),
    (8, "2.56M PDE evaluations", f"{float(Z4_20['pde_evaluations_cum']) / 1e6:.2f}M PDE evaluations", "tables/phaseZ4_20K_checkpoints.csv"),
    (8, "L2 0.263", f"L2 {float(Z4_20['L2_exact']):.3f}", "tables/phaseZ4_20K_checkpoints.csv"),
    (8, "≈ 8.1 cycles", f"≈ {f1(Z4_20['persistence_cycles'])} cycles", "tables/phaseZ4_20K_checkpoints.csv"),
    (8, "0.393 s", f"{float(Z4_20['collapse_time_s']):.3f} s", "tables/phaseZ4_20K_checkpoints.csv"),
    (8, "129.2 rad/s", f"{f1(Z4_20['fit_w'])} rad/s", "tables/phaseZ4_20K_checkpoints.csv fit_w"),
    (8, "2.88 × 10⁷", "2.88 × 10⁷" if "2.88e7 PDE" in rep("STAGE01_CORRECTIONS.md") else "?", "reports/STAGE01_CORRECTIONS.md (paper schedule)"),
    (8, "≈ 20.6", f"≈ {D['f_d']:.1f}", "exact reference: 1 s × f_d"),
    (11, "241,601", "241,601" if "241 601" in rep("PHASE_Z4_20K_BUDGET_DIAGNOSTIC.md") else "?", "reports/PHASE_Z4_20K_BUDGET_DIAGNOSTIC.md"),
    (11, "4.386e-4", f"{float([r for r in tab('stage01_reference_comparison.csv') if r['grid'] == '201x2001' and r['pred'] == 'exact' and r['ref'] == 'paper'][0]['L2']):.3e}".replace("e-04", "e-4"), "tables/stage01_reference_comparison.csv"),
]
for k in LB:   # appendix B: every L2 at 3 significant figures
    CHECKS.append((12, s3(k), s3(k), f"optimization_leaderboard.csv {k} (3 s.f., must appear in the matrix)"))

lines, fail = [], 0
for n, shown, rec, src in CHECKS:
    ok = shown == rec and shown in text(n)
    fail += not ok
    lines.append(f"{'OK  ' if ok else 'FAIL'} slide {n:2d}: {shown!r:26} source {rec!r:26} [{src}]")
for k in ("Z4", "Z4_20K"):
    ok = D["checks"][k]["match"]
    fail += not ok
    lines.append(f"{'OK  ' if ok else 'FAIL'} trace {k}: collapse from frozen checkpoint {D['checks'][k]['collapse_time_s']} s == frozen table {D['checks'][k]['frozen_collapse_time_s']} s")
for k, (lo, hi) in (("range_t2", (-8368.7, -1.02)), ("range_tanh2", (-1.93, -0.158))):
    r = D["conditioning"][k]
    ok = abs(r[0] - lo) < 0.05 and abs(r[1] - hi) < 0.01
    fail += not ok
    lines.append(f"{'OK  ' if ok else 'FAIL'} conditioning {k}: {r[0]:.2f}..{r[1]:.3f} vs frozen profile {lo}..{hi}")

# ------------------------------------------------------------ scope checks
N = n_slides()
S = {n: text(n) for n in range(1, N + 1)}
A = {n: S[n] + " " + text(n, True) for n in range(1, N + 1)}


def scope(ok, msg):
    global fail
    fail += not ok
    lines.append(f"{'OK  ' if ok else 'FAIL'} {msg}")


scope(N == 13 and len([n for n in S if "APPENDIX" not in S[n]]) == 10, f"{N} slides: 10 main + 3 appendix")
paper = [n for n in S if "4.64" in S[n]]
scope(all(re.search(r"published|paper", S[n], re.I) for n in paper) and "not our result" in S[3], f"paper 4.64e-4 appears on slides {paper}, always attributed to the paper")
scope(not any(re.search(r"\b(successful|succeeded|solved|we beat|outperform|state of the art|state-of-the-art|best PINN)\b", A[n], re.I) for n in A), "no Batch-1 result labelled successful; no superiority claims")
scope("NO VALIDATED OPTIMIZED PINN YET" in S[7], "slide 7 shows the explicit status box")
scope(not any(re.search(r"mode[- ]?2", A[n], re.I) for n in A), "no Mode-2 content anywhere")
rail = [n for n in S if re.search(r"rail", S[n], re.I)]
scope(rail == [10] and "no railway results exist yet" in S[10] and "FUTURE" in S[10], f"railway appears only on slide(s) {rail}, marked future work with no results")
nov = [(n, m.start()) for n in A for m in re.finditer(r"novel(?!TY_GAP)", A[n], re.I)]   # the file name NOVELTY_GAP.md is not a claim
scope(all(re.search(r"(not|honesty about) ?(claim|novel)", A[n][max(0, i - 40):i + 10], re.I) for n, i in nov), f"'novel' used only in negated/honesty context ({len(nov)} occurrences)")
scope("RESEARCH HYPOTHESIS" in S[9] and "Not an established contribution" in S[9], "slide 9 labels the solver strategy a research hypothesis")
scope("we benchmark them, we do not claim them" in S[9] and "benchmarked, not claimed" in S[13], "prior-art components explicitly not claimed (slides 9, 13)")
for n in range(1, N + 1):
    if has_chart(n) and n not in (1, 3):     # slide 3 chart is the beam mode-shape drawing, not a data graph
        scope("?" in S[n] + chart_titles(n), f"slide {n} (data graph) poses an explicit question (slide text or chart title)")
final = "We did not simply add more components to a PINN. We used controlled experiments to determine where the solver fails."
scope(final in text(10, True), "final message present verbatim in slide-10 speaker notes")
scope(all(len(text(n, True)) > 80 for n in range(1, N + 1)), "every slide has speaker notes")

lines.append("ALL CHECKS PASSED" if fail == 0 else f"{fail} FAILURE(S)")
out = assert_safe_output(PRES / "qa_check_output.txt", strict=True, purpose="QA report")
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(1 if fail else 0)
