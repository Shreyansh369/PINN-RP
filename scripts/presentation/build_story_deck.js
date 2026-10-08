// Batch-1 research-story deck (10 slides + 3 appendix), rebuilt from scratch.
// Evidence: frozen Batch-1 artifacts only (references/batch1/*, frozen checkpoints via extract_deck_data.py).
//   NODE_PATH=<dir with pptxgenjs> node scripts/presentation/build_story_deck.js <pptx skill apply_theme.js>
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const RP = path.resolve(__dirname, "..", "..");
const OUTDIR = path.join(RP, "results_batch2", "presentation");
const OUT = path.join(OUTDIR, "final_research_story_presentation.pptx");
const D = JSON.parse(fs.readFileSync(path.join(OUTDIR, "data", "deck_data.json"), "utf8"));
const { applyTheme } = require(process.argv[2]);

// ---- frozen leaderboard (references/batch1/optimization_leaderboard.csv) for the appendix matrix
const LB = (() => {
  const lines = fs.readFileSync(path.join(RP, "references", "batch1", "optimization_leaderboard.csv"), "utf8").trim().split("\n");
  const h = lines[0].split(",");
  const m = {};
  for (const l of lines.slice(1)) {
    const c = l.split(",");
    const r = Object.fromEntries(h.map((k, i) => [k, c[i]]));
    m[r.run_id.split("__")[0]] = r;
  }
  return m;
})();
const l2 = (name) => Number(LB[name].L2_exact).toPrecision(3);   // 3 significant figures everywhere

const THEME = {
  name: "Physics Reference Research",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "17242E", lt1: "FFFFFF", dk2: "3C5566", lt2: "EEF2F4",
    accent1: "E8743B", accent2: "2A9D8F", accent3: "6C5CB5", accent4: "B5473A",
    accent5: "8A9AA6", accent6: "B7861A", hlink: "2A9D8F", folHlink: "8A9AA6",
  },
};
const HEX = THEME.colors;
const EXACT = "AEB8C0";            // exact-solution grey (hex-only chart option)
const STAGE = { C0: HEX.accent4, X2: HEX.accent6, Z4: HEX.accent2, Z4_20K: HEX.accent1, Y1: HEX.accent3 };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Can We Make Physics-Informed AI Actually Solve Physics Efficiently?";
pres.subject = "Batch-1 reproducibility and diagnosis of a beam-vibration PINN; Batch-2 research direction";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333, MX = 0.6, CW = W - 2 * MX;

// ------------------------------------------------------------------ layouts
const FOOT = "Batch 1 = frozen evidence (PINN-replication @ d31864a) · Batch 2 = research in progress (PINN-RP)";
const titlePh = (color) => ({ placeholder: { options: { name: "title", type: "title", x: MX, y: 0.6, w: CW, h: 0.7, fontFace: THEME.headFontFace, fontSize: 30, bold: true, color, valign: "top", align: "left", margin: 0 }, text: "" } });
const kickerPh = { placeholder: { options: { name: "kicker", type: "body", x: MX, y: 0.3, w: CW, h: 0.3, fontSize: 12, bold: true, color: C.accent1, charSpacing: 1, margin: 0 }, text: "" } };
const foot = { text: { text: FOOT, options: { x: MX, y: 7.08, w: 10.5, h: 0.28, fontSize: 10, color: C.accent5, margin: 0 } } };
const num = { x: W - MX - 0.6, y: 7.08, w: 0.6, h: 0.28, fontSize: 10, color: C.accent5, align: "right" };
pres.defineSlideMaster({
  title: "TITLE_DARK", background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.85, w: CW, h: 1.5, fontFace: THEME.headFontFace, fontSize: 38, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: MX, y: 2.5, w: CW, h: 0.5, fontSize: 20, color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({ title: "CONTENT", background: { color: C.background1 }, objects: [kickerPh, titlePh(C.text1), foot], slideNumber: num });
pres.defineSlideMaster({ title: "CLOSING_DARK", background: { color: C.text1 }, objects: [kickerPh, titlePh(C.background1), foot], slideNumber: num });

// ------------------------------------------------------------------ helpers
const T = (s, text, o) => s.addText(text, { isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: "top", ...o });
const box = (s, x, y, w, h, fill, name, line, radius = 0.08) =>
  s.addShape(radius ? pres.shapes.ROUNDED_RECTANGLE : pres.shapes.RECTANGLE, { x, y, w, h, ...(radius ? { rectRadius: radius } : {}), fill: { color: fill }, line: line ? { color: line, width: 1.25 } : { type: "none" }, objectName: name });
const dashed = (s, x, y, w, h, color, name, fill) =>
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: fill ? { color: fill } : { type: "none" }, line: { color, width: 1.25, dashType: "dash" }, objectName: name });
const arrowR = (s, x, y, w, color, name) => s.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color, width: 1.75, endArrowType: "triangle" }, objectName: name });
const arrowD = (s, x, y, h, color, name) => s.addShape(pres.shapes.LINE, { x, y, w: 0, h, line: { color, width: 1.5, endArrowType: "triangle" }, objectName: name });
const AX = { catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 10, valAxisLabelFontSize: 10,
  catAxisTitleColor: HEX.dk2, valAxisTitleColor: HEX.dk2, catAxisTitleFontFace: "+mn-lt", valAxisTitleFontFace: "+mn-lt", catAxisTitleFontSize: 11, valAxisTitleFontSize: 11,
  titleFontFace: "+mn-lt", titleColor: HEX.dk1, titleFontSize: 13, legendFontFace: "+mn-lt", legendFontSize: 11, legendColor: HEX.dk2,
  valGridLine: { color: "E3E8EB", size: 0.5 }, catGridLine: { style: "none" } };
const notes = (s, a) => s.addNotes(a.join("\n\n"));

// trace chart (scatter with lines): exact + model (+ optional collapse marker)
function traceChart(s, key, x, y, w, h, o = {}) {
  const t = D.t, tmax = o.tmax ?? 1.0;
  const idx = t.map((v, i) => i).filter((i) => t[i] <= tmax + 1e-9);
  const xs = idx.map((i) => t[i]);
  const ex = idx.map((i) => D.traces_mm.exact[i]), md = idx.map((i) => D.traces_mm[key][i]);
  const nul = (k) => Array(k).fill(null);
  const data = [{ name: "t", values: [...xs] }, { name: "Exact solution", values: [...ex] }, { name: o.label || key, values: [...md] }];
  const colors = [EXACT, STAGE[key]];
  if (o.collapse != null) {
    data[0].values.push(o.collapse, o.collapse);
    data[1].values.push(null, null); data[2].values.push(null, null);
    data.push({ name: "collapse", values: [...nul(xs.length), -100, 100] });
    colors.push(HEX.dk1);
  }
  s.addChart(pres.charts.SCATTER, data, {
    x, y, w, h, objectName: o.name || `${key} trace`, ...AX,
    chartColors: colors, lineSize: o.lineSize ?? 1.5, lineDataSymbol: "none", showLegend: false,
    valAxisMinVal: -100, valAxisMaxVal: 100, valAxisMajorUnit: 50, valAxisLabelFormatCode: "General", valAxisCrossesAt: -100,
    catAxisMinVal: 0, catAxisMaxVal: tmax, catAxisMajorUnit: o.xMajor ?? 0.2,
    catAxisHidden: !!o.hideX, showValAxisTitle: !!o.yTitle, valAxisTitle: o.yTitle || "",
    showCatAxisTitle: !!o.xTitle, catAxisTitle: o.xTitle || "", showTitle: !!o.title, title: o.title || "",
    plotArea: { fill: { color: HEX.lt1 } },
  });
}

// ================================================================== 1 — the question
pres.addSection({ title: "The question" });
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "The question" });
  s.addText("Can We Make Physics-Informed AI Actually Solve Physics Efficiently?", { placeholder: "title" });
  s.addText("From beam-vibration reproducibility to a compute-aware Physics Reference Layer", { placeholder: "body" });
  const steps = ["Physical system", "Physics equations", "PINN", "Verified physical reference", "AI application"];
  const bw = 2.05, gap = (CW - 5 * bw) / 4, y = 3.75, bh = 1.05;
  steps.forEach((st, i) => {
    const x = MX + i * (bw + gap);
    const goal = i === 3;
    box(s, x, y, bw, bh, goal ? C.accent1 : "263746", `Pipeline step ${i + 1}`);
    T(s, st, { x: x + 0.12, y, w: bw - 0.24, h: bh, fontSize: 17, bold: true, color: C.background1, align: "center", valign: "middle", objectName: `Pipeline step ${i + 1} text` });
    if (i < 4) arrowR(s, x + bw + 0.08, y + bh / 2, gap - 0.16, C.accent5, `Pipeline arrow ${i + 1}`);
  });
  T(s, "the missing piece we are researching", { x: MX + 3 * (bw + gap) - 0.2, y: y + bh + 0.12, w: bw + 0.4, h: 0.3, fontSize: 12, italic: true, color: C.accent1, align: "center", objectName: "Goal caption" });
  T(s, "First testbed: Euler–Bernoulli beam vibration", { x: MX, y: 5.75, w: 8, h: 0.4, fontSize: 16, color: C.background2, objectName: "Testbed footer" });
  T(s, "[Author name] · [Institution] · October 2026", { x: MX, y: 6.6, w: 8, h: 0.3, fontSize: 12, color: C.accent5, objectName: "Author line" });
  notes(s, [
    "NOTICE: the pipeline. Physics-based AI needs a trustworthy physical reference between the equations and the application. That reference is what we are researching.",
    "MEANING: our first testbed is a vibrating beam. It is simple enough to have an exact solution, so every claim can be checked, and hard enough (fast, long-lasting vibration) to expose PINN failures.",
    "NEXT: why this matters, and why we looked at physics-informed neural networks at all.",
  ]);
}

// ================================================================== 2 — why this matters
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "The question" });
  s.addText("WHY THIS PROBLEM MATTERS", { placeholder: "kicker" });
  s.addText("Engineering needs fast, trustworthy physics", { placeholder: "title" });
  // map: x = cost per answer (low → high), y = physical trust (low → high)
  const ox = 1.1, oy = 6.45, mw = 6.6, mh = 4.75;
  s.addShape(pres.shapes.LINE, { x: ox, y: oy, w: mw, h: 0, line: { color: C.text2, width: 1.5, endArrowType: "triangle" }, objectName: "Cost axis" });
  s.addShape(pres.shapes.LINE, { x: ox, y: oy - mh, w: 0, h: mh, line: { color: C.text2, width: 1.5, beginArrowType: "triangle" }, objectName: "Trust axis" });
  T(s, "computational cost per answer  →", { x: ox + mw - 3.6, y: oy + 0.08, w: 3.6, h: 0.3, fontSize: 12, color: C.text2, align: "right", objectName: "Cost axis label" });
  T(s, "physical trust  →", { x: ox - 0.5, y: oy - mh + 1.6, w: 2.2, h: 0.3, fontSize: 12, color: C.text2, rotate: 270, objectName: "Trust axis label" });
  // simulation (top-right)
  box(s, 4.95, 1.75, 2.6, 1.25, C.background2, "Simulation card");
  T(s, [{ text: "Physics simulation", options: { bold: true, breakLine: true } }, { text: "accurate, verifiable — expensive per query", options: { fontSize: 12, color: C.text2 } }], { x: 5.1, y: 1.85, w: 2.35, h: 1.1, objectName: "Simulation text" });
  // pure ML (bottom-left)
  box(s, 1.45, 4.95, 2.6, 1.2, C.background2, "Pure ML card");
  T(s, [{ text: "Pure ML", options: { bold: true, breakLine: true } }, { text: "fast — but can violate physics", options: { fontSize: 12, color: C.text2 } }], { x: 1.6, y: 5.05, w: 2.35, h: 1.0, objectName: "Pure ML text" });
  // target (top-left)
  dashed(s, 1.45, 1.75, 2.6, 1.25, HEX.accent1, "Target zone", "FDF0E9");
  T(s, [{ text: "TARGET", options: { bold: true, color: C.accent1, breakLine: true } }, { text: "accurate + cheap + verified", options: { fontSize: 12, color: C.text2 } }], { x: 1.6, y: 1.85, w: 2.35, h: 1.0, objectName: "Target text" });
  // PINN (middle) with arrow to target
  box(s, 4.25, 3.65, 2.3, 1.0, C.text1, "PINN card");
  T(s, [{ text: "PINN", options: { bold: true, breakLine: true } }, { text: "physics inside a neural network", options: { fontSize: 12 } }], { x: 4.4, y: 3.72, w: 2.05, h: 0.9, color: C.background1, objectName: "PINN text" });
  s.addShape(pres.shapes.LINE, { x: 3.45, y: 3.1, w: 0.8, h: 0.55, line: { color: C.accent1, width: 2, beginArrowType: "triangle", dashType: "dash" }, objectName: "PINN to target arrow" });
  T(s, "?", { x: 3.25, y: 3.3, w: 0.4, h: 0.45, fontSize: 26, bold: true, color: C.accent1, fontFace: THEME.headFontFace, objectName: "Question mark" });
  // right: the research question
  const rx = 8.45, rw = W - MX - rx;
  T(s, "The research question", { x: rx, y: 1.75, w: rw, h: 0.4, fontSize: 18, bold: true, fontFace: THEME.headFontFace, objectName: "Question heading" });
  ["HIGH PHYSICAL ACCURACY", "LOW COMPUTATIONAL COST", "PHYSICAL VERIFICATION"].forEach((q, i) => {
    const y = 2.35 + i * 1.0;
    box(s, rx, y, rw, 0.7, i === 2 ? C.accent2 : C.text1, `Goal ${i + 1}`);
    T(s, q, { x: rx, y, w: rw, h: 0.7, fontSize: 17, bold: true, color: C.background1, align: "center", valign: "middle", objectName: `Goal ${i + 1} text` });
    if (i < 2) T(s, "+", { x: rx, y: y + 0.68, w: rw, h: 0.34, fontSize: 18, bold: true, color: C.accent1, align: "center", valign: "middle", objectName: `Plus ${i + 1}` });
  });
  T(s, "PINNs promise all three at once. Do they deliver on a real vibration problem?", { x: rx, y: 5.4, w: rw, h: 0.9, fontSize: 16, bold: true, color: C.accent1, objectName: "Slide conclusion" });
  notes(s, [
    "NOTICE: the map. Classical simulation is trustworthy but costs a full solve per answer. Pure machine learning is fast but has no guarantee of obeying physics. A PINN puts the governing equations into the network's training objective, so in principle it can be both. The orange box is where we want to be.",
    "MEANING: the question is not 'can a network fit a curve'. It is whether we can get accuracy, low cost and verification together. The literature is honest about cost: PINNs are typically much slower to train than classical solvers but fast to evaluate once trained (Grossmann et al., 2024).",
    "NEXT: the published method we started from.",
  ]);
}

// ================================================================== 3 — where we started
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "The question" });
  s.addText("WHERE WE STARTED", { placeholder: "kicker" });
  s.addText("A 2025 paper: a PINN for beam vibration", { placeholder: "title" });
  const lx = MX, lw = 6.3;
  // beam
  T(s, "1  Beam", { x: lx, y: 1.55, w: 2, h: 0.35, fontSize: 14, bold: true, color: C.accent1, objectName: "Beam label" });
  const bx = lx + 0.45, by = 1.95, bw = 5.4;
  box(s, bx - 0.18, by - 0.1, 0.18, 0.9, C.text2, "Clamp left", null, 0);
  box(s, bx + bw, by - 0.1, 0.18, 0.9, C.text2, "Clamp right", null, 0);
  s.addChart(pres.charts.LINE, [{ name: "Mode-1 shape", labels: D.mode_shape.map((_, i) => String(i)), values: D.mode_shape.map((v) => -v) }], {
    x: bx - 0.05, y: by - 0.05, w: bw + 0.1, h: 0.8, objectName: "Beam mode shape", chartColors: [HEX.accent1], lineSize: 2.5, lineDataSymbol: "none",
    showLegend: false, catAxisHidden: true, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, valAxisMinVal: -1.1, valAxisMaxVal: 0.1 });
  arrowD(s, lx + 3.15, 2.85, 0.35, C.accent5, "Arrow beam to PDE");
  T(s, "2  Physics equation", { x: lx, y: 3.05, w: 3, h: 0.35, fontSize: 14, bold: true, color: C.accent1, objectName: "PDE label" });
  box(s, lx, 3.3, lw, 0.95, C.background2, "PDE card");
  T(s, [{ text: "EI·u_xxxx + ρA·u_tt + b·u_t = 0", options: { bold: true, fontFace: "Cambria", fontSize: 22, breakLine: true } },
    { text: "bending stiffness · inertia · damping", options: { fontSize: 12, color: C.text2 } }], { x: lx + 0.25, y: 3.38, w: lw - 0.5, h: 0.85, align: "center", objectName: "PDE text" });
  arrowD(s, lx + 3.15, 4.3, 0.35, C.accent5, "Arrow PDE to PINN");
  T(s, "3  PINN", { x: lx, y: 4.5, w: 3, h: 0.35, fontSize: 14, bold: true, color: C.accent1, objectName: "PINN label" });
  box(s, lx, 4.75, lw, 0.85, C.text1, "PINN card");
  T(s, "A neural network u(x, t), trained until it satisfies the equation, the initial shape and the clamped ends", { x: lx + 0.25, y: 4.75, w: lw - 0.5, h: 0.85, fontSize: 14, color: C.background1, valign: "middle", objectName: "PINN text" });
  T(s, "Our test case: clamped–clamped, damped beam, first mode at 20.6 Hz, i.e. 20.6 vibration cycles in the 1-second window.", { x: lx, y: 5.85, w: lw, h: 0.6, fontSize: 13, color: C.text2, objectName: "Test case" });
  // right: published baseline
  const rx = 7.35, rw = W - MX - rx;
  box(s, rx, 1.55, rw, 5.0, C.background2, "Published baseline card");
  box(s, rx + 0.3, 1.8, 2.5, 0.4, C.text1, "Baseline tag", null, 0.06);
  T(s, "PUBLISHED BASELINE", { x: rx + 0.3, y: 1.8, w: 2.5, h: 0.4, fontSize: 12, bold: true, color: C.background1, align: "center", valign: "middle", charSpacing: 1, objectName: "Baseline tag text" });
  T(s, "Söyleyici & Ünver, Engineering Applications of Artificial Intelligence 141 (2025) 109804", { x: rx + 0.3, y: 2.35, w: rw - 0.6, h: 0.5, fontSize: 12, italic: true, color: C.text2, objectName: "Citation" });
  T(s, "Fourier features + NTK adaptive weighting", { x: rx + 0.3, y: 2.95, w: rw - 0.6, h: 0.45, fontSize: 18, bold: true, fontFace: THEME.headFontFace, objectName: "Method name" });
  T(s, "The paper's contribution: a PINN for beam vibration covering high-frequency cases, several boundary conditions and experimental validation.", { x: rx + 0.3, y: 3.45, w: rw - 0.6, h: 0.8, fontSize: 13, color: C.text2, objectName: "Paper scope" });
  T(s, "4.64 × 10⁻⁴", { x: rx + 0.3, y: 4.35, w: rw - 0.6, h: 0.75, fontSize: 40, bold: true, fontFace: THEME.headFontFace, objectName: "Paper L2" });
  T(s, "relative L2 error reported by the paper for our test case", { x: rx + 0.3, y: 5.1, w: rw - 0.6, h: 0.35, fontSize: 13, color: C.text2, objectName: "Paper L2 label" });
  T(s, "Published value — not our result.", { x: rx + 0.3, y: 5.75, w: rw - 0.6, h: 0.4, fontSize: 14, bold: true, color: C.accent4, objectName: "Not our result" });
  notes(s, [
    "NOTICE: the chain beam → equation → PINN. The network takes position and time and outputs displacement; training penalises any violation of the beam equation, the initial shape and the clamped ends.",
    "MEANING: the reference paper (Söyleyici & Ünver, EAAI 2025) adds two published techniques: Fourier features, which help networks represent fast oscillations, and NTK-based adaptive weighting of the loss terms. It reports a relative L2 error of 4.64e-4 for our test case. That is the paper's number, not ours, and Fourier + NTK is the paper's method, not our contribution.",
    "Our test case: clamped–clamped damped beam, first mode, 20.6 Hz, so the model must reproduce about 20.6 decaying cycles in one second.",
    "NEXT: what we actually did with it.",
  ]);
}

// ================================================================== 4 — what we did (ladder)
pres.addSection({ title: "Batch 1: what we did and found" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Batch 1: what we did and found" });
  s.addText("WHAT WE ACTUALLY DID", { placeholder: "kicker" });
  s.addText("An experiment ladder: one question per step", { placeholder: "title" });
  const rungs = [
    ["Published Fourier + NTK", "The paper's method — our starting point"],
    ["Baseline reproduction", "Does a faithful implementation reproduce it?"],
    ["D · Configuration diagnostics", "Which implementation choices matter?"],
    ["E · Hard constraints + RAD", "Can hard constraints remove the failure?"],
    ["X · Conditioning + representation", "Is representation the problem?"],
    ["Y · Optimizer", "Is optimization the problem?"],
    ["Z · Sampling / seed / batch", "Can sampling or batch size fix propagation?"],
    ["Budget study · Y1-20K, Z4-20K", "Does more physics computation extend persistence?"],
  ];
  const y0 = 1.55, rh = 0.6, gap = 0.07, lx = MX + 0.65, lw = 3.6;
  s.addShape(pres.shapes.LINE, { x: MX + 0.21, y: y0 + 0.3, w: 0, h: 7 * (rh + gap), line: { color: C.accent5, width: 2 }, objectName: "Ladder rail" });
  rungs.forEach(([lab, q], i) => {
    const y = y0 + i * (rh + gap), first = i === 0, last = i === rungs.length - 1;
    s.addShape(pres.shapes.OVAL, { x: MX, y: y + 0.09, w: 0.42, h: 0.42, fill: { color: first ? C.text1 : last ? C.accent1 : C.accent2 }, line: { color: C.background1, width: 1.5 }, objectName: `Rung ${i + 1} node` });
    T(s, String(i), { x: MX, y: y + 0.09, w: 0.42, h: 0.42, fontSize: 13, bold: true, color: C.background1, align: "center", valign: "middle", objectName: `Rung ${i + 1} number` });
    box(s, lx, y, lw, rh, first ? C.text1 : C.background2, `Rung ${i + 1} label box`, null, 0.06);
    T(s, lab, { x: lx + 0.15, y, w: lw - 0.3, h: rh, fontSize: 14, bold: true, color: first ? C.background1 : C.text1, valign: "middle", objectName: `Rung ${i + 1} label` });
    T(s, q, { x: lx + lw + 0.3, y, w: 5.0, h: rh, fontSize: 17, color: first ? C.text2 : C.text1, italic: first, valign: "middle", objectName: `Rung ${i + 1} question` });
  });
  const rx = 10.0, rw = W - MX - rx;
  box(s, rx, 1.55, rw, 1.75, C.text1, "Method card");
  T(s, [
    { text: "26", options: { fontSize: 40, bold: true, fontFace: THEME.headFontFace, color: C.accent1, breakLine: true } },
    { text: "controlled training runs", options: { fontSize: 14, bold: true, breakLine: true } },
    { text: "one change at a time", options: { fontSize: 14 } },
  ], { x: rx + 0.25, y: 1.65, w: rw - 0.5, h: 1.6, color: C.background1, objectName: "Run count" });
  T(s, [
    { text: "Kept identical across runs: ", options: { bold: true } },
    { text: "the beam, the exact reference solution, the evaluation grid, the seed (one controlled exception) and the compute accounting." },
  ], { x: rx, y: 3.55, w: rw, h: 1.7, fontSize: 13, color: C.text2, objectName: "Controls note" });
  notes(s, [
    "NOTICE: read the ladder top to bottom; each rung asks one question and changes one thing relative to the rung before.",
    "MEANING: first we checked whether a faithful implementation of the paper reproduces its behaviour. When it did not, we did not pile on tricks; we isolated causes: implementation choices (D), hard constraints and adaptive sampling (E), conditioning versus representation (X), the optimizer (Y), sampling, seed and batch size (Z), and finally how much physics computation the solver gets (the 20 000-step budget runs). 26 runs in total, single seed 1234 except one seed-change control (Z2).",
    "NEXT: what these runs actually produced — the clearest single picture of Batch 1.",
  ]);
}

// ================================================================== 5 — what happened (Figure 1)
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Batch 1: what we did and found" });
  s.addText("WHAT HAPPENED", { placeholder: "kicker" });
  s.addText("From a static field to eight cycles — then collapse", { placeholder: "title" });
  T(s, "How close does each stage get to the real vibration?  (mid-span displacement u(L/2, t), mm)", { x: MX, y: 1.35, w: 8.2, h: 0.3, fontSize: 13, italic: true, color: C.text2, objectName: "Figure question" });
  // legend
  s.addShape(pres.shapes.LINE, { x: 9.0, y: 1.5, w: 0.45, h: 0, line: { color: EXACT, width: 2.5 }, objectName: "Legend exact line" });
  T(s, "exact solution", { x: 9.52, y: 1.37, w: 1.4, h: 0.26, fontSize: 11, color: C.text2, objectName: "Legend exact" });
  s.addShape(pres.shapes.LINE, { x: 10.85, y: 1.38, w: 0, h: 0.26, line: { color: C.text1, width: 1.25 }, objectName: "Legend collapse line" });
  T(s, "collapse (amplitude < 50 % of exact)", { x: 10.95, y: 1.37, w: 1.8, h: 0.26, fontSize: 11, color: C.text2, fit: "shrink", objectName: "Legend collapse" });
  const ch = D.checks;
  const rows = [
    ["C0", "C0", "paper method: Fourier + NTK · 20k steps", "Static collapse", "never vibrates · L2 " + l2("C0_paper"), null],
    ["X2", "X2", "hard constraints + tanh²(ω₁t) · 5k steps", "Conditioning improved", "vibrates, but over-damped · L2 " + l2("X2_hard_tanh2_rc"), null],
    ["Z4", "Z4", "+ LR schedule, batch 128 · 5k steps", "≈ 3.2 cycles", "longer persistence · L2 " + l2("Z4_Y1_mb128"), ch.Z4.collapse_time_s],
    ["Z4_20K", "Z4-20K", "same recipe, 4× compute · 20k steps", "≈ 8.1 cycles", "then still collapses · L2 " + l2("Z4_20K"), ch.Z4_20K.collapse_time_s],
  ];
  const y0 = 1.72, hs = [1.12, 1.12, 1.12, 1.62], g = 0.08, cx = 2.75, cw = 7.75;
  let y = y0;
  rows.forEach(([key, name, cfg, ann, sub, tc], i) => {
    const h = hs[i], last = i === 3;
    T(s, name, { x: MX, y: y + 0.12, w: 2.1, h: 0.4, fontSize: 20, bold: true, color: STAGE[key], fontFace: THEME.headFontFace, objectName: `${name} label` });
    T(s, cfg, { x: MX, y: y + 0.52, w: 2.1, h: 0.6, fontSize: 11, color: C.text2, objectName: `${name} config` });
    traceChart(s, key, cx, y, cw, h, { hideX: !last, xTitle: last ? "time t [s]" : "", collapse: tc, name: `${name} vs exact` });
    T(s, ann, { x: cx + cw + 0.15, y: y + 0.18, w: W - MX - cx - cw - 0.15, h: 0.4, fontSize: 15, bold: true, color: STAGE[key], objectName: `${name} annotation` });
    T(s, sub, { x: cx + cw + 0.15, y: y + 0.58, w: W - MX - cx - cw - 0.15, h: 0.5, fontSize: 12, color: C.text2, objectName: `${name} annotation detail` });
    y += h + g;
  });
  notes(s, [
    "NOTICE: grey is the exact beam response — 20.6 decaying cycles. Each coloured line is one stage of our investigation, same time axis.",
    "C0 (red), the paper's Fourier + NTK method implemented faithfully: it never vibrates; it sits near a static deflection (relative L2 3.26 after 20 000 steps). X2 (amber): with exact hard constraints and a better-conditioned time factor it starts from the right shape and vibrates, but dies out far too fast (L2 0.909). Z4 (teal): with a decaying learning rate and larger batches it follows the first ~3.2 cycles (L2 0.526). Z4-20K (orange): the same recipe with four times the physics computation follows ~8.1 cycles, then still collapses at 0.393 s (L2 0.263).",
    "MEANING: every step moved the failure later. None reproduced the full window. All four curves are inference from frozen checkpoints; the collapse times shown equal the frozen Batch-1 values exactly.",
    "NEXT: why each of these failures happened.",
  ]);
}

// ================================================================== 6 — why did it fail
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Batch 1: what we did and found" });
  s.addText("WHY DID IT FAIL?", { placeholder: "kicker" });
  s.addText("Three different failures, found one at a time", { placeholder: "title" });
  const pw = (CW - 0.6) / 3, top = 1.5, ph = 5.4;
  const P = [
    ["A", "SOFT TRAINING FAILURE", HEX.accent4, "Does the published setup vibrate?"],
    ["B", "CONDITIONING FAILURE", HEX.accent6, "How large must the network output be?"],
    ["C", "LONG-TIME PROPAGATION FAILURE", HEX.accent1, "Where in time is the vibration lost?"],
  ];
  P.forEach(([L, h, col, q], i) => {
    const x = MX + i * (pw + 0.3);
    box(s, x, top, pw, ph, C.background2, `Panel ${L} card`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: top + 0.2, w: 0.42, h: 0.42, fill: { color: col }, line: { type: "none" }, objectName: `Panel ${L} badge` });
    T(s, L, { x: x + 0.2, y: top + 0.2, w: 0.42, h: 0.42, fontSize: 15, bold: true, color: C.background1, align: "center", valign: "middle", objectName: `Panel ${L} letter` });
    T(s, h, { x: x + 0.75, y: top + 0.2, w: pw - 0.9, h: 0.42, fontSize: 14, bold: true, color: col, valign: "middle", objectName: `Panel ${L} heading` });
    T(s, q, { x: x + 0.2, y: top + 0.75, w: pw - 0.4, h: 0.3, fontSize: 12, italic: true, color: C.text2, objectName: `Panel ${L} question` });
  });
  const cy = top + 1.1, chh = 1.95, cwid = pw - 0.3;
  // A
  const xA = MX;
  traceChart(s, "C0", xA + 0.15, cy, cwid, chh, { tmax: 0.3, xMajor: 0.1, xTitle: "t [s]", name: "Panel A chart" });
  T(s, [{ text: "Fourier + NTK  →  static / incorrect attractor", options: { bold: true } }], { x: xA + 0.2, y: cy + chh + 0.15, w: pw - 0.4, h: 0.55, fontSize: 14, objectName: "Panel A flow" });
  T(s, "The published baseline did not reproduce sustained vibration in our controlled implementation.", { x: xA + 0.2, y: top + ph - 1.2, w: pw - 0.4, h: 1.0, fontSize: 13, color: C.text2, objectName: "Panel A caption" });
  // B: conditioning chart (Figure 3)
  const xB = MX + pw + 0.3, cd = D.conditioning;
  s.addChart(pres.charts.SCATTER, [
    { name: "t", values: cd.t },
    { name: "(t/T)²", values: cd.absN_t2 },
    { name: "tanh²(ω₁t)", values: cd.absN_tanh2 },
  ], { x: xB + 0.15, y: cy, w: cwid, h: chh, objectName: "Panel B conditioning chart", ...AX,
    chartColors: [HEX.accent4, HEX.accent2], lineSize: 2, lineDataSymbol: "none", showLegend: false,
    valAxisLogScaleBase: 10, valAxisMinVal: 0.1, valAxisMaxVal: 10000, valAxisLabelFormatCode: "General", showValAxisTitle: true, valAxisTitle: "required |N|",
    catAxisMinVal: 0, catAxisMaxVal: 1, catAxisMajorUnit: 0.25, showCatAxisTitle: true, catAxisTitle: "t [s]" });
  const rng = cd.range_t2[0];
  T(s, [
    { text: "(t/T)²  →  required |N| ≈ 8.4 × 10³", options: { bold: true, color: C.accent4, breakLine: true } },
    { text: "tanh²(ω₁t)  →  required |N| = O(1)", options: { bold: true, color: C.accent2 } },
  ], { x: xB + 0.2, y: cy + chh + 0.15, w: pw - 0.4, h: 0.6, fontSize: 14, objectName: "Panel B flow" });
  T(s, "Hard constraints fixed IC/BC satisfaction, but the original time factor was poorly conditioned.", { x: xB + 0.2, y: top + ph - 1.2, w: pw - 0.4, h: 1.0, fontSize: 13, color: C.text2, objectName: "Panel B caption" });
  // C
  const xC = MX + 2 * (pw + 0.3);
  traceChart(s, "Z4_20K", xC + 0.15, cy, cwid, chh, { collapse: D.checks.Z4_20K.collapse_time_s, xMajor: 0.25, xTitle: "t [s]", name: "Panel C chart" });
  const chips = ["correct first cycles", "amplitude decay", "collapse"];
  chips.forEach((c, i) => {
    const cwc = (pw - 0.4 - 2 * 0.25) / 3, x = xC + 0.2 + i * (cwc + 0.25);
    box(s, x, cy + chh + 0.15, cwc, 0.55, i === 2 ? C.accent1 : C.background1, `Chip ${i + 1}`, null, 0.06);
    T(s, c, { x, y: cy + chh + 0.15, w: cwc, h: 0.55, fontSize: 11, bold: true, color: i === 2 ? C.background1 : C.text1, align: "center", valign: "middle", objectName: `Chip ${i + 1} text` });
    if (i < 2) arrowR(s, x + cwc + 0.02, cy + chh + 0.425, 0.21, C.accent5, `Chip arrow ${i + 1}`);
  });
  T(s, "More compute moves collapse later, but does not remove it.", { x: xC + 0.2, y: top + ph - 1.2, w: pw - 0.4, h: 1.0, fontSize: 13, color: C.text2, objectName: "Panel C caption" });
  notes(s, [
    "NOTICE: three panels, three different failure mechanisms, each found by a controlled experiment.",
    `A — the paper's soft-constraint training (initial and boundary conditions as penalties, NTK weights). In our implementation it settled into a static, non-vibrating solution while the NTK weights grew to roughly 1e14–1e15. We call this an observed optimization failure; we do not claim the NTK weights cause it.`,
    `B — we then built the initial and boundary conditions into the network exactly ('hard constraints'). The original time factor (t/T)² forces the network to output about ${Math.round(-rng).toLocaleString("en-US")} in magnitude near t = 0 — a badly conditioned target. Replacing it with tanh²(ω₁t), where ω₁ comes from the beam equation, brings the required output to order one (at most 1.9). The chart is the required network output computed from the exact solution.`,
    "C — with conditioning fixed, the model learns the first cycles correctly, then the amplitude decays too fast and the vibration collapses. More physics computation moves this collapse later but has not removed it.",
    "NEXT: what these experiments let us conclude.",
  ]);
}

// ================================================================== 7 — what we learned
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Batch 1: what we did and found" });
  s.addText("WHAT WE LEARNED", { placeholder: "kicker" });
  s.addText("Claim → evidence", { placeholder: "title" });
  const R = [
    ["Representation is not the main limit", [{ text: "Trained directly on the exact solution (no physics loss), the same network recovers ", options: {} }, { text: "129.25 and 129.37 rad/s", options: { bold: true } }, { text: " (X3, X4); exact 129.32 rad/s.", options: {} }]],
    ["Conditioning matters", [{ text: "(t/T)² hard constraint needs |N| ≈ 8.4 × 10³; ", options: {} }, { text: "tanh²(ω₁t) needs O(1)", options: { bold: true } }, { text: ". Only the tanh² variants started to vibrate.", options: {} }]],
    ["Compute matters", [{ text: "Persistence 1.2 → 3.1 → 3.2 → 8.1 cycles", options: { bold: true } }, { text: "  (Y1 5k → Y1 20k → Z4 5k → Z4 20k; 0.16M → 2.56M PDE evaluations).", options: {} }]],
    ["RAD did not solve the failure", [{ text: "Z1 (Y1 + residual-adaptive sampling): ", options: {} }, { text: "1.6 cycles", options: { bold: true } }, { text: ", then collapse — no long-time oscillation.", options: {} }]],
  ];
  const y0 = 1.5, rh = 0.88, gap = 0.12, cw = 4.3, ex = MX + cw + 0.75;
  R.forEach(([claim, ev], i) => {
    const y = y0 + i * (rh + gap);
    box(s, MX, y, cw, rh, C.text1, `Claim ${i + 1} card`);
    T(s, `${i + 1}`, { x: MX + 0.2, y, w: 0.4, h: rh, fontSize: 24, bold: true, color: C.accent1, fontFace: THEME.headFontFace, valign: "middle", objectName: `Claim ${i + 1} number` });
    T(s, claim, { x: MX + 0.65, y, w: cw - 0.8, h: rh, fontSize: 16, bold: true, color: C.background1, valign: "middle", objectName: `Claim ${i + 1}` });
    arrowR(s, MX + cw + 0.1, y + rh / 2, 0.5, C.accent5, `Claim ${i + 1} arrow`);
    box(s, ex, y, W - MX - ex, rh, C.background2, `Evidence ${i + 1} card`);
    T(s, ev, { x: ex + 0.25, y, w: W - MX - ex - 0.5, h: rh, fontSize: 14, valign: "middle", objectName: `Evidence ${i + 1}` });
  });
  const sy = y0 + 4 * (rh + gap) + 0.12;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: sy, w: CW, h: 1.0, rectRadius: 0.08, fill: { color: "FBEDEA" }, line: { color: C.accent4, width: 2 }, objectName: "Status box" });
  T(s, [{ text: "5   STATUS: NO VALIDATED OPTIMIZED PINN YET", options: { bold: true, fontSize: 20, color: C.accent4, fontFace: THEME.headFontFace, breakLine: true } },
    { text: "Best tested (Z4-20K): 8.1 of 20.6 cycles, L2 0.263, still over-damped. Full-window reproduction was not achieved.", options: { fontSize: 13, color: C.text1 } }],
  { x: MX + 0.3, y: sy, w: CW - 0.6, h: 1.0, valign: "middle", objectName: "Status text" });
  notes(s, [
    "NOTICE: each claim on the left is tied to one measured piece of evidence on the right.",
    "1 — Representation: when we trained the same network directly on the exact solution (a diagnostic, not a solver), it reproduced the 20.6 Hz vibration. So network capacity is not what blocks us. 2 — Conditioning: how the constraints are built into the network changes the required output by more than three orders of magnitude, and only the well-conditioned version vibrated. 3 — Compute: persistence grew from 1.2 to 8.1 cycles as physics computation grew. 4 — Adaptive sampling (RAD) did not fix the collapse.",
    "5 — and the honest status: no optimized PINN has been validated. The best tested model reproduces 8.1 of 20.6 cycles.",
    "NEXT: the compute story in one graph.",
  ]);
}

// ================================================================== 8 — the compute story (Figure 2)
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Batch 1: what we did and found" });
  s.addText("THE COMPUTE STORY", { placeholder: "kicker" });
  s.addText("More physics computation helps — but not enough yet", { placeholder: "title" });
  const Y = D.budget.Y1, Z = D.budget.Z4, nul = (k) => Array(k).fill(null);
  const xs = [...Y.map((p) => p.pde_evaluations_cum / 1e6), ...Z.map((p) => p.pde_evaluations_cum / 1e6), 0, 3.0];
  s.addChart(pres.charts.SCATTER, [
    { name: "x", values: xs },
    { name: "Y1 recipe, batch 32 (5k–20k steps)", values: [...Y.map((p) => p.persistence_cycles), ...nul(6)] },
    { name: "Z4 recipe, batch 128 (5k–20k steps)", values: [...nul(4), ...Z.map((p) => p.persistence_cycles), null, null] },
    { name: "Full-window target (20.6 cycles)", values: [...nul(8), 20.6, 20.6] },
  ], { x: MX, y: 1.5, w: 7.4, h: 5.35, objectName: "Persistence vs PDE evaluations chart", ...AX,
    showTitle: true, title: "Does additional physics computation delay dynamic collapse?",
    chartColors: [HEX.accent3, HEX.accent2, HEX.accent5], lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 8,
    valAxisMinVal: 0, valAxisMaxVal: 22, valAxisMajorUnit: 4, valAxisLabelFormatCode: "General",
    catAxisMinVal: 0, catAxisMaxVal: 3.0, catAxisMajorUnit: 0.5,
    showValAxisTitle: true, valAxisTitle: "cycles reproduced before collapse", showCatAxisTitle: true, catAxisTitle: "cumulative PDE evaluations [millions]",
    showLegend: true, legendPos: "b" });
  const rx = 8.35, rw = W - MX - rx, tw = (rw - 0.25) / 2;
  box(s, rx, 1.5, tw, 1.45, C.background2, "Target tile");
  T(s, [{ text: "≈ 20.6", options: { fontSize: 34, bold: true, fontFace: THEME.headFontFace, color: C.text2, breakLine: true } }, { text: "FULL-WINDOW TARGET (cycles)", options: { fontSize: 11, bold: true, color: C.text2 } }], { x: rx + 0.15, y: 1.6, w: tw - 0.3, h: 1.3, objectName: "Target tile text" });
  box(s, rx + tw + 0.25, 1.5, tw, 1.45, "FDF0E9", "Best tile");
  T(s, [{ text: "≈ 8.1", options: { fontSize: 34, bold: true, fontFace: THEME.headFontFace, color: C.accent1, breakLine: true } }, { text: "BEST TESTED (cycles)", options: { fontSize: 11, bold: true, color: C.accent1 } }], { x: rx + tw + 0.4, y: 1.6, w: tw - 0.3, h: 1.3, objectName: "Best tile text" });
  const z = Z[3];
  box(s, rx, 3.15, rw, 1.65, C.text1, "Z4-20K card");
  T(s, [
    { text: "Z4-20K, the best tested model", options: { bold: true, color: C.accent1, breakLine: true } },
    { text: `${(z.pde_evaluations_cum / 1e6).toFixed(2)}M PDE evaluations · L2 ${z.L2_exact.toFixed(3)}`, options: { breakLine: true } },
    { text: `≈ ${z.persistence_cycles.toFixed(1)} cycles · collapse at ${z.collapse_time_s.toFixed(3)} s`, options: { breakLine: true } },
    { text: `frequency ≈ ${z.fit_w.toFixed(1)} rad/s (exact 129.32)` },
  ], { x: rx + 0.25, y: 3.25, w: rw - 0.5, h: 1.5, fontSize: 14, color: C.background1, objectName: "Z4-20K facts" });
  T(s, [
    { text: "Not the same budget as the paper. ", options: { bold: true } },
    { text: "The paper reports L2 ≈ 4.64 × 10⁻⁴ after its full schedule (≈ 2.88 × 10⁷ PDE evaluations by our reading of it); our best tested is L2 0.263 at 2.56 × 10⁶." },
  ], { x: rx, y: 5.0, w: rw, h: 1.05, fontSize: 12, color: C.text2, objectName: "Paper comparison note" });
  T(s, "Clearly helps; the full-window threshold has not been crossed.", { x: rx, y: 6.2, w: rw, h: 0.65, fontSize: 15, bold: true, color: C.text1, objectName: "Slide conclusion" });
  notes(s, [
    "NOTICE: horizontal axis is how many times the physics equation was evaluated during training; vertical axis is how many vibration cycles the model reproduces before it collapses. The grey line at the top is the target: all 20.6 cycles.",
    "MEANING: along both recipes, more physics computation monotonically delays the collapse — from 1.2 cycles at 0.16 million evaluations to 8.1 cycles at 2.56 million. At equal computation (0.64 million) the small-batch and large-batch recipes land in the same place (3.1 vs 3.2 cycles), so batch size mainly changed wall-clock time, not the solution. The last block of computation gained about half as much as the previous blocks, so we do not extrapolate.",
    "The paper's 4.64e-4 comes from its full training schedule, roughly ten times more PDE evaluations than our largest run, so the numbers are not a like-for-like comparison. Our controlled implementation did not reproduce the published result; the investigation identified implementation, conditioning and optimization sensitivities.",
    "NEXT: what this means for how we continue — Batch 2.",
  ]);
}

// ================================================================== 9 — Batch 2 (Figure 5)
pres.addSection({ title: "Batch 2 and beyond" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Batch 2 and beyond" });
  s.addText("WHAT WE ARE DOING NOW", { placeholder: "kicker" });
  s.addText("Batch 2: From PINN Optimization to a General Physics Solver", { placeholder: "title" });
  const st = ["PHYSICAL PROBLEM", "PHYSICS ANALYSIS", "CONDITIONING", "FORMULATION", "REPRESENTATION", "TRAINING STRATEGY", "PHYSICS SOLVER", "VERIFICATION", "PHYSICS REFERENCE"];
  const y0 = 1.5, bh = 0.43, gap = 0.17, bw = 3.5;
  st.forEach((t, i) => {
    const y = y0 + i * (bh + gap), last = i === st.length - 1, first = i === 0;
    box(s, MX, y, bw, bh, last ? C.accent1 : first ? C.text1 : C.background2, `Stage ${i + 1}`, null, 0.06);
    T(s, t, { x: MX, y, w: bw, h: bh, fontSize: 13, bold: true, color: last || first ? C.background1 : C.text1, align: "center", valign: "middle", charSpacing: 1, objectName: `Stage ${i + 1} text` });
    if (!last) arrowD(s, MX + bw / 2, y + bh + 0.01, gap - 0.03, C.accent5, `Stage arrow ${i + 1}`);
  });
  const rx = 5.0, rw = W - MX - rx, cw2 = (rw - 0.3) / 2;
  box(s, rx, 1.5, cw2, 1.45, C.background2, "NOT card");
  T(s, [{ text: "NOT", options: { bold: true, color: C.accent4, fontSize: 14, breakLine: true } }, { text: "one PINN recipe for every PDE", options: { fontSize: 17, bold: true } }], { x: rx + 0.25, y: 1.62, w: cw2 - 0.5, h: 1.25, objectName: "NOT text" });
  box(s, rx + cw2 + 0.3, 1.5, cw2, 1.45, "E3F3F1", "BUT card");
  T(s, [{ text: "BUT", options: { bold: true, color: C.accent2, fontSize: 14, breakLine: true } }, { text: "choose a solving strategy based on the physics and the compute budget", options: { fontSize: 17, bold: true } }], { x: rx + cw2 + 0.55, y: 1.62, w: cw2 - 0.5, h: 1.25, objectName: "BUT text" });
  T(s, "“We are not trying to make PINNs more complicated. We are trying to make physics solving more efficient and adaptive.”", { x: rx, y: 3.15, w: rw, h: 0.95, fontSize: 18, italic: true, fontFace: THEME.headFontFace, color: C.text1, objectName: "Key quote" });
  T(s, [
    { text: "Building blocks already in the literature — we benchmark them, we do not claim them: ", options: { bold: true } },
    { text: "mixed formulations · modal PINNs · temporal decomposition · hard constraints and tanh²-type conditioning · adaptive sampling · loss balancing · optimizers · automated PINN design." },
  ], { x: rx, y: 4.25, w: rw, h: 1.0, fontSize: 13, color: C.text2, objectName: "Prior art note" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx, y: 5.45, w: rw, h: 1.4, rectRadius: 0.08, fill: { color: C.background1 }, line: { color: C.accent1, width: 1.5, dashType: "dash" }, objectName: "Hypothesis box" });
  T(s, [
    { text: "RESEARCH HYPOTHESIS — UNDER INVESTIGATION", options: { bold: true, color: C.accent1, fontSize: 13, charSpacing: 1, breakLine: true } },
    { text: "Problem-dependent, compute-aware selection of the solving strategy, with built-in verification. Not an established contribution.", options: { fontSize: 14, breakLine: true } },
    { text: "Status: preparation done; first controlled Batch-2 diagnostics started (single seed, provisional — not presented as evidence).", options: { fontSize: 12, color: C.text2 } },
  ], { x: rx + 0.25, y: 5.55, w: rw - 0.5, h: 1.2, objectName: "Hypothesis text" });
  notes(s, [
    "NOTICE: the left column is the pipeline Batch 2 studies. The middle stages — conditioning, formulation, representation, training strategy — are exactly where Batch 1 found the solver to be sensitive.",
    "MEANING: Batch 1 showed there is no single setting that 'fixes' the PINN; what helped depended on the physics (conditioning) and on the compute budget. So Batch 2 asks a broader question: can the solving strategy be chosen from measurable properties of the problem and the available compute, with every answer verified against physics before it is returned?",
    "Honesty about novelty: mixed formulations, modal PINNs, temporal decomposition, hard constraints and tanh²-type conditioning, adaptive sampling, loss balancing, optimizers and automated PINN design all have prior art (Batch-2 literature matrix, 81 works). We benchmark them. The candidate gap — problem-dependent, compute-aware strategy selection with verification — is a research hypothesis under investigation, not a result.",
    "Status: Batch-2 preparation and pre-registration are done and the first controlled diagnostics have started; they are single-seed and provisional and are not presented as evidence today.",
    "NEXT: where this leads.",
  ]);
}

// ================================================================== 10 — railway to vertical AI (closing)
{
  const s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "Batch 2 and beyond" });
  s.addText("WHERE THIS LEADS", { placeholder: "kicker" });
  s.addText("From railway to vertical AI", { placeholder: "title" });
  T(s, "We did not simply add components to a PINN. Controlled experiments located where the solver fails: the network can represent the dynamics; conditioning and optimization limit it; physics-computation budget measurably matters. Next: a problem-aware, compute-aware Physics Reference Layer — not another fixed PINN recipe.",
    { x: MX, y: 1.4, w: CW, h: 0.95, fontSize: 15, color: C.background1, objectName: "Final message" });
  // hub
  const hx = 3.25, hy = 4.0, r = 0.85;
  const br = ["RAILWAY", "STRUCTURAL", "THERMAL", "FLUID", "ENERGY", "MANUFACTURING"];
  const pw = 1.6, ph = 0.4, rx_ = 2.15, ry_ = 1.12;
  br.forEach((b, i) => {
    const a = (-90 + i * 60) * Math.PI / 180, cx = hx + rx_ * Math.cos(a), cy = hy + ry_ * Math.sin(a);
    const rail = i === 0;
    s.addShape(pres.shapes.LINE, { x: Math.min(hx, cx), y: Math.min(hy, cy), w: Math.abs(cx - hx), h: Math.abs(cy - hy), flipV: (cx < hx) !== (cy < hy), line: { color: rail ? HEX.accent1 : "4A5E6D", width: 1.25 }, objectName: `Branch ${b} line` });
    if (rail) box(s, cx - pw / 2, cy - ph / 2, pw, ph, C.accent1, `Branch ${b}`, null, 0.06);
    else dashed(s, cx - pw / 2, cy - ph / 2, pw, ph, HEX.accent5, `Branch ${b}`, HEX.dk1);
    T(s, b, { x: cx - pw / 2, y: cy - ph / 2, w: pw, h: ph, fontSize: 11, bold: true, color: rail ? C.background1 : C.background2, align: "center", valign: "middle", objectName: `Branch ${b} text` });
  });
  s.addShape(pres.shapes.OVAL, { x: hx - r, y: hy - r, w: 2 * r, h: 2 * r, fill: { color: C.accent1 }, line: { type: "none" }, objectName: "Hub" });
  T(s, "PHYSICS REFERENCE LAYER", { x: hx - r + 0.12, y: hy - r, w: 2 * r - 0.24, h: 2 * r, fontSize: 13, bold: true, color: C.background1, align: "center", valign: "middle", objectName: "Hub text" });
  // stage timeline
  const tx = 6.9, tw = W - MX - tx;
  const TL = [["CURRENT", "beam physics research", true], ["NEXT", "general solver research", false], ["FUTURE", "railway Physics Reference Layer", false], ["LATER", "other vertical AI applications", false]];
  TL.forEach(([k, v, now], i) => {
    const y = 2.6 + i * 0.68;
    box(s, tx, y, 1.45, 0.52, now ? C.accent1 : "263746", `Stage ${k}`, null, 0.06);
    T(s, k, { x: tx, y, w: 1.45, h: 0.52, fontSize: 13, bold: true, color: C.background1, align: "center", valign: "middle", charSpacing: 1, objectName: `Stage ${k} text` });
    T(s, v, { x: tx + 1.65, y, w: tw - 1.65, h: 0.52, fontSize: 16, color: now ? C.background1 : C.background2, bold: now, valign: "middle", objectName: `Stage ${k} detail` });
  });
  // railway path
  const ry0 = 5.6;
  T(s, "RAILWAY PATH  ·  future work — no railway results exist yet", { x: MX, y: ry0, w: CW, h: 0.3, fontSize: 11, bold: true, color: C.accent1, charSpacing: 1, objectName: "Railway path label" });
  const rp = ["Beam vibration", "Rail dynamics", "Vehicle / track dynamics", "Axle / bearing dynamics", "Sensor-informed physics", "Predictive maintenance"];
  const ng = 0.28, nw = (CW - 5 * ng) / 6;
  rp.forEach((n, i) => {
    const x = MX + i * (nw + ng), now = i === 0;
    if (now) box(s, x, ry0 + 0.38, nw, 0.62, C.accent2, `Rail node ${i + 1}`, null, 0.06);
    else dashed(s, x, ry0 + 0.38, nw, 0.62, HEX.accent5, `Rail node ${i + 1}`, HEX.dk1);
    T(s, n, { x: x + 0.08, y: ry0 + 0.38, w: nw - 0.16, h: 0.62, fontSize: 12, bold: now, color: C.background1, align: "center", valign: "middle", objectName: `Rail node ${i + 1} text` });
    if (i < 5) arrowR(s, x + nw + 0.03, ry0 + 0.69, ng - 0.06, C.accent5, `Rail arrow ${i + 1}`);
  });
  notes(s, [
    "FINAL MESSAGE: We did not simply add more components to a PINN. We used controlled experiments to determine where the solver fails. We found that the network can represent the target dynamics, identified conditioning and optimization limitations, and measured the effect of physics-computation budget. That motivates the next phase: a problem-aware, compute-aware Physics Reference Layer rather than another fixed PINN recipe.",
    "NOTICE: the Physics Reference Layer sits between physics and application AI. Railway is the first application we will target; structural, thermal, fluid, energy and manufacturing are later candidates. The timeline on the right is the honest status: today we are doing beam physics research; next is general solver research; a railway Physics Reference Layer is future work; other verticals come later.",
    "The railway path along the bottom is a plan, not a result. Only the first box — beam vibration — has been studied, and even there no validated optimized PINN exists yet. No railway model has been trained.",
  ]);
}

// ================================================================== appendix
pres.addSection({ title: "Appendix" });
{ // A — technical PDE and PINN
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appendix" });
  s.addText("APPENDIX A · BACKUP", { placeholder: "kicker" });
  s.addText("Technical setup: PDE and PINN architecture", { placeholder: "title" });
  const hw = (CW - 0.4) / 2;
  const row = (a, b) => [{ text: a, options: { bold: true } }, b];
  const opt = (x, w) => ({ x, y: 1.55, w, colW: [1.9, w - 1.9], fontSize: 12, color: C.text1, rowH: 0.36, border: { type: "solid", pt: 0.75, color: C.background2 }, margin: [0.03, 0.08, 0.03, 0.08] });
  s.addTable([
    [{ text: "Physics (benchmark FE-D-M1)", options: { bold: true, colspan: 2, color: C.background1, fill: { color: C.text1 } } }],
    row("Equation (paper Eq. 49)", "c²·u_xxxx + u_tt + γ·u_t = 0  (EI·u_xxxx + ρA·u_tt + b·u_t = 0 divided by ρA)"),
    row("Coefficients", "c² = 43.73², γ = 7.08"),
    row("Domain", "x ∈ [0, 2.75] m, t ∈ [0, 1] s"),
    row("Boundary conditions", "clamped–clamped: u = u_x = 0 at both ends"),
    row("Initial conditions", "u(x,0) = Mode-1 shape, A₀ = 0.08 m; u_t(x,0) = 0"),
    row("Mode 1", "ω_d = 129.32 rad/s (20.6 Hz); decay 3.54 s⁻¹"),
    row("References", "paper-faithful (printed root 4.7300) and exact-physics (exact root); they differ by 4.386e-4 relative L2"),
    row("Evaluation", "201 × 2001 grid; relative L2; persistence = first t ≥ P/2 where local amplitude < 50 % of exact"),
  ], opt(MX, hw));
  s.addTable([
    [{ text: "PINN (published baseline and our variants)", options: { bold: true, colspan: 2, color: C.background1, fill: { color: C.text1 } } }],
    row("Network", "6 × 200 tanh, 241,601 parameters"),
    row("Fourier features", "spatio-temporal multiscale, m = 100; σ_x = 1; σ_t = 10, 1"),
    row("Paper training", "NTK trace weights; Adam lr 1e-4; 640 points per term, mini-batch 32; 45,000 epochs"),
    row("Hard constraints", "u = u₀(x) + g(t)·Φ(x)·A₀·N(x,t), Φ = 16x²(L−x)²/L⁴"),
    row("Time factor", "g = (t/T)² (original) or tanh²(ω₁t), ω₁ = 129.37 rad/s from the PDE and BCs"),
    row("LR schedule (Y, Z)", "1e-3 · 0.9^(step/1000)"),
    row("Hardware", "CPU, 1 thread per run, float32, seed 1234"),
    row("Budgets", "5,000–20,000 steps; 1.6e5–2.56e6 PDE evaluations"),
  ], opt(MX + hw + 0.4, hw));
  notes(s, ["Backup only. Exact definitions behind the main talk. Sources: references/batch1/reports/STAGE0_AUDIT.md, STAGE01_CORRECTIONS.md, CONVENTIONS.md, PHASE_X_DIAGNOSTIC.md, PHASE_Y1_20K_BUDGET_DIAGNOSTIC.md."]);
}
{ // B — experiment matrix (values read from the frozen leaderboard)
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appendix" });
  s.addText("APPENDIX B · BACKUP", { placeholder: "kicker" });
  s.addText("Detailed experiment matrix (26 runs, frozen)", { placeholder: "title" });
  const M = [
    ["BA", "BA_vanilla", "plain MLP (20k)", "no oscillation"],
    ["BB", "BB_fourier", "+ Fourier features (20k)", "static / low-freq."],
    ["C0", "C0_paper", "paper: Fourier + NTK (20k)", "static"],
    ["C0-rc", "C0_rc", "C0, reference-code convention (20k)", "static"],
    ["D1", "D1_unit", "C0 + unit-normalised inputs", "static"],
    ["D2", "D2_bias0", "C0 + zero biases", "low-freq., wrong ω"],
    ["D3", "D3_unit_bias0", "C0 + unit inputs + zero biases", "low-freq."],
    ["D4", "D4_rc_bias0", "C0, ref.-code conv. + zero biases", "low-freq."],
    ["E3", "E3_fourier_rad", "BB + RAD", "offset, no oscillation"],
    ["E4", "E4_hard_fourier", "hard constraints, (t/T)²", "slow decay, no oscillation"],
    ["E5", "E5_hard_fourier_rad", "E4 + RAD", "no oscillation"],
    ["E4b", "E4b_hard_fourier_rc", "E4, ref.-code convention", "no oscillation"],
    ["X1", "X1_hard_tanh2", "E4 with tanh²(ω₁t)", "static / low-freq."],
    ["X2", "X2_hard_tanh2_rc", "E4b with tanh²(ω₁t)", "112 rad/s, over-damped"],
    ["X3", "X3_supervised_paperconv", "supervised fit (diagnostic)", "129.25 rad/s"],
    ["X4", "X4_supervised_rcconv", "supervised fit (diagnostic)", "129.37 rad/s"],
    ["Y1", "Y1_X2_lrsched", "X2 + LR schedule", "1.2 cycles, collapse"],
    ["Y2", "Y2_X1_lrsched", "X1 + LR schedule", "static / low-freq."],
    ["Y3", "Y3_X4_lrsched", "X4 + LR schedule (diagnostic)", "129.32 rad/s"],
    ["Y4", "Y4_C0_lrsched", "C0 + LR schedule", "static"],
    ["Z1", "Z1_Y1_rad", "Y1 + RAD", "1.6 cycles, collapse"],
    ["Z2", "Z2_Y1_seed1235", "Y1, seed 1235", "1.1 cycles, collapse"],
    ["Z3", "Z3_Y4_rad", "Y4 + RAD", "static"],
    ["Z4", "Z4_Y1_mb128", "Y1, mini-batch 128", "3.2 cycles, collapse"],
    ["Y1-20K", "Y1_20K", "Y1, 20k steps", "3.1 cycles, collapse"],
    ["Z4-20K", "Z4_20K", "Z4, 20k steps", "8.1 cycles, collapse"],
  ];
  const hw = (CW - 0.3) / 2, cols = [0.75, 2.6, 0.7, hw - 4.05];
  const hdr = ["Run", "Change vs parent (5k steps unless noted)", "L2", "Outcome"].map((t, j) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text1 }, align: j === 2 ? "right" : "left" } }));
  const rows = (part) => [hdr, ...part.map(([id, key, ch, out]) => {
    const hl = id === "Z4-20K", sup = /supervised|diagnostic/.test(ch);
    const f = { fill: { color: hl ? "FCEADF" : HEX.lt1 }, bold: hl, italic: sup && !hl };
    return [{ text: id, options: f }, { text: ch, options: f }, { text: l2(key), options: { ...f, align: "right" } }, { text: out, options: f }];
  })];
  const o = (x) => ({ x, y: 1.5, w: hw, colW: cols, fontSize: 10.5, color: C.text1, rowH: 0.36, border: { type: "solid", pt: 0.5, color: C.background2 }, margin: [0.02, 0.06, 0.02, 0.06] });
  s.addTable(rows(M.slice(0, 13)), o(MX));
  s.addTable(rows(M.slice(13)), o(MX + hw + 0.3));
  T(s, "L2 = relative L2 error vs the exact solution (= paper-faithful reference at this precision). Italic rows are supervised diagnostics, not physics-trained solvers. Persistence in cycles uses the frozen collapse metric.", { x: MX, y: 6.65, w: CW, h: 0.35, fontSize: 10, color: C.text2, objectName: "Matrix caption" });
  notes(s, ["Backup only. L2 values are read at build time from references/batch1/optimization_leaderboard.csv (frozen, byte-identical to ROOT). Outcomes from the Phase A1, D, E, X, Y, Z and budget reports."]);
}
{ // C — literature frontier / novelty map (Batch-2 literature matrix)
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appendix" });
  s.addText("APPENDIX C · BACKUP", { placeholder: "kicker" });
  s.addText("Literature frontier: what already exists", { placeholder: "title" });
  const G = [
    ["Mixed / auxiliary formulations", "Lyu et al. 2022 (MIM); A-PINN for EB beams 2026"],
    ["Modal / reduced-order PINNs", "Zhang, Vlachas & Chatzi 2026 (RO-PINN)"],
    ["Temporal decomposition", "Krishnapriyan et al. 2021; Penwarden et al. 2023; AT-PINN-HC (Chen et al. 2025)"],
    ["Hard constraints, tanh²-type conditioning", "Sukumar & Srivastava 2022; Lu et al. 2021 (hPINN); AT-PINN-HC auxiliary functions 2025"],
    ["Adaptive sampling", "RAR (Lu et al. 2021); RAD (Wu et al. 2023); R3 (Daw et al. 2023)"],
    ["Loss balancing", "Wang et al. 2021 (LR annealing); NTK (Wang et al. 2022); ReLoBRaLo; SA-PINN 2023"],
    ["Optimizers and precision", "Adam + L-BFGS / NNCG (Rathore et al. 2024); MultiAdam 2023; SOAP 2025; FP64 (Xu et al. 2025)"],
    ["Representation", "Fourier features (Tancik 2020; Wang 2021); SIREN 2020; SPINN 2023"],
    ["Automated PINN design", "Auto-PINN 2022; PINNsAgent 2025; Lang-PINN 2025"],
  ];
  const gw = (CW - 0.5) / 3, gh = 1.2;
  G.forEach(([h, p], i) => {
    const x = MX + (i % 3) * (gw + 0.25), y = 1.5 + Math.floor(i / 3) * (gh + 0.18);
    box(s, x, y, gw, gh, C.background2, `Lit card ${i + 1}`);
    T(s, [{ text: h, options: { bold: true, fontSize: 13, breakLine: true } }, { text: p, options: { fontSize: 11, color: C.text2 } }], { x: x + 0.18, y: y + 0.1, w: gw - 0.36, h: gh - 0.2, objectName: `Lit card ${i + 1} text` });
  });
  const by = 1.5 + 3 * (gh + 0.18) + 0.05;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: by, w: CW, h: 1.05, rectRadius: 0.08, fill: { color: C.background1 }, line: { color: C.accent1, width: 1.5, dashType: "dash" }, objectName: "Gap box" });
  T(s, [
    { text: "Our status: every component above has prior art — benchmarked, not claimed. ", options: { bold: true, breakLine: true } },
    { text: "Candidate gap (RESEARCH HYPOTHESIS): selecting the strategy from measurable physics properties before training, under a compute budget, with verification — closest prior work selects by search or LLM agents (Auto-PINN, PINNsAgent, Lang-PINN).", options: { color: C.text2 } },
  ], { x: MX + 0.25, y: by + 0.08, w: CW - 0.5, h: 0.9, fontSize: 12, objectName: "Gap text" });
  notes(s, ["Backup only. From the Batch-2 literature audit (PINN-RP docs/LITERATURE_MATRIX.md and docs/NOVELTY_GAP.md): 81 works tabulated, 65 confirmed by web search, the rest from bibliographic memory and flagged. Abstract-level reading; full texts must be checked before any publication claim."]);
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("written", OUT);
})();
