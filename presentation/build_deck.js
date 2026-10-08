// Supervisor-facing progress deck, PINN-RP. Every number comes from presentation/deck_data.json
// (extracted by presentation/extract_deck_data.py from committed artifacts at cdd87b5). No B2-E04 content.
//
//   NODE_PATH=<dir with pptxgenjs, react-icons, react, react-dom, sharp> node presentation/build_deck.js <apply_theme.js>
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const HERE = __dirname;
const D = JSON.parse(fs.readFileSync(path.join(HERE, "deck_data.json"), "utf8"));
const OUT = path.join(HERE, "PINN_Research_Progress_Supervisor.pptx");
const SHOWN = [];                                   // displayed numbers, checked by check_consistency.py

// ------------------------------------------------------------------------------------------ theme
const THEME = {
  name: "PINN-RP Engineering",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2430", lt1: "FFFFFF", dk2: "14213D", lt2: "EEF2F5",
    accent1: "1F6F8B", accent2: "2E7D5B", accent3: "B7791F", accent4: "B03A2E", accent5: "5E6B7A", accent6: "7FA7BA",
    hlink: "1F6F8B", folHlink: "5E6B7A",
  },
};
const H = THEME.colors;                              // hex (charts, grid lines, shadows only)
const MUTED = "5E6B7A";
const STATUS = {
  ACHIEVED: { color: "accent2", label: "ACHIEVED" },
  DIAGNOSTIC: { color: "accent3", label: "DIAGNOSTIC" },
  FAILED: { color: "accent4", label: "FAILED ON THIS BENCHMARK" },
  NOTYET: { color: "accent5", label: "NOT YET DONE" },
  BENCH: { color: "accent1", label: "BENCHMARK" },
  FUTURE: { color: "accent5", label: "FUTURE WORK" },
  PARTIAL: { color: "accent3", label: "PARTIAL" },
};

// ---------------------------------------------------------------------------------------- helpers
const fx = (v, d) => Number(v).toFixed(d);
const rng = (arr, d) => { const a = Math.min(...arr), b = Math.max(...arr); return fx(a, d) === fx(b, d) ? fx(a, d) : `${fx(a, d)}–${fx(b, d)}`; };
const thou = (n) => Number(n).toLocaleString("en-US");
const show = (label, text, source) => { SHOWN.push({ label, text, source }); return text; };
const bySeed = (rows, arm) => [1234, 1235, 1236].map((s) => rows.find((r) => r.arm === arm && r.seed === s));
const SEEDS = [1234, 1235, 1236];

async function icon(Comp, color = "FFFFFF") {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";                         // 13.333 x 7.5 in
  pres.author = "PINN-RP";
  pres.title = "Towards a Compute-Efficient Physics-Informed Neural Network";
  pres.subject = "Research progress report (completed work through B2-E03, commit cdd87b5)";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;
  const W = 13.333;

  // ----------------------------------------------------------------------------------- layouts
  pres.defineSlideMaster({
    title: "TITLE_DARK", background: { color: H.dk2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.55, w: 7.6, h: 1.9, fontFace: "Cambria", fontSize: 36, bold: true, color: C.background1, valign: "top", align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "subtitle", type: "body", x: 0.7, y: 3.55, w: 7.6, h: 0.7, fontSize: 22, color: "C9DCE6", align: "left", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT", background: { color: H.lt1 },
    margin: [0.5, 0.6, 0.6, 0.6],
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 10.1, h: 0.85, fontFace: "Cambria", fontSize: 30, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "source", type: "body", x: 0.6, y: 6.98, w: 11.6, h: 0.36, fontSize: 10, color: C.accent5, valign: "middle", align: "left", margin: 0 }, text: "" } },
    ],
    slideNumber: { x: 12.35, y: 6.98, w: 0.4, h: 0.36, fontSize: 10, color: "5E6B7A", align: "right" },
  });
  pres.defineSlideMaster({
    title: "CLOSING_DARK", background: { color: H.dk2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 0.45, w: 11.9, h: 0.8, fontFace: "Cambria", fontSize: 32, bold: true, color: C.background1, align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "source", type: "body", x: 0.7, y: 6.98, w: 11.4, h: 0.36, fontSize: 10, color: "C9DCE6", valign: "middle", align: "left", margin: 0 }, text: "" } },
    ],
    slideNumber: { x: 12.35, y: 6.98, w: 0.4, h: 0.36, fontSize: 10, color: "C9DCE6", align: "right" },
  });

  // ------------------------------------------------------------------------------- composites
  const pill = (s, st, x, y, w = 2.6, name = "status") => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.38, rectRadius: 0.19, fill: { color: C[STATUS[st].color] || STATUS[st].color }, line: { type: "none" }, objectName: name + "-bg" });
    s.addText(STATUS[st].label, { x, y, w, h: 0.38, fontSize: 11, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: name, charSpacing: 1 });
  };
  const statusPill = (s, st) => pill(s, st, W - 0.6 - 2.6, 0.58);
  const schemeOf = (st) => ({ accent1: C.accent1, accent2: C.accent2, accent3: C.accent3, accent4: C.accent4, accent5: C.accent5 })[STATUS[st].color];
  const tag = (s, st, x, y, w) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.3, rectRadius: 0.15, fill: { color: schemeOf(st) }, line: { type: "none" } });
    s.addText(STATUS[st].label, { x, y, w, h: 0.3, fontSize: 10, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
  };
  const card = (s, x, y, w, h, fill = C.background2, name = "card") =>
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { type: "none" }, objectName: name });
  const iconDot = async (s, Comp, x, y, d, fill) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { type: "none" } });
    s.addImage({ data: await icon(Comp), x: x + d * 0.24, y: y + d * 0.24, w: d * 0.52, h: d * 0.52 });
  };
  const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt",
    titleFontFace: "+mn-lt", valAxisTitleFontFace: "+mn-lt", catAxisTitleFontFace: "+mn-lt",
    catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, legendFontSize: 11,
    dataLabelFontSize: 10, dataLabelColor: "1B2430", titleFontSize: 13, titleColor: "1B2430", valAxisTitleColor: MUTED, catAxisTitleColor: MUTED,
    valAxisTitleFontSize: 11, catAxisTitleFontSize: 11 };
  const quiet = () => ({ valGridLine: { color: "D9E0E6", size: 0.5 }, catGridLine: { style: "none" }, valAxisLineShow: false });

  const SRC = (txt) => `Source: ${txt} · commit cdd87b5`;
  const content = (title, section, src) => {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
    s.addText(title, { placeholder: "title" });
    s.addText(SRC(src), { placeholder: "source" });
    return s;
  };

  // ============================================================================ 1. TITLE
  pres.addSection({ title: "Introduction" });
  {
    const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
    s.addText("Towards a Compute-Efficient Physics-Informed Neural Network", { placeholder: "title" });
    s.addText("From Beam-Vibration Benchmark to Railway Physics", { placeholder: "subtitle" });
    s.addText([
      { text: "[Presenter name]", options: { bold: true, color: C.background1, breakLine: true } },
      { text: "[Programme / institution]", options: { color: "C9DCE6" } },
    ], { x: 0.7, y: 4.55, w: 6.5, h: 0.8, fontSize: 16, margin: 0, isTextBox: true, objectName: "presenter" });
    s.addText("Physics-informed neural networks · Euler–Bernoulli beam vibration · Railway engineering (target application)",
      { x: 0.7, y: 5.55, w: 7.6, h: 0.5, fontSize: 14, color: "C9DCE6", margin: 0, isTextBox: true });
    s.addText(`Progress report · completed experiments through B2-E03 · PINN-RP commit ${D.commit}`,
      { x: 0.7, y: 6.15, w: 7.6, h: 0.4, fontSize: 12, color: "9FB6C4", margin: 0, isTextBox: true });
    // analytical reference curve (NOT a network output)
    const a = D.analytic;
    const step = 2;
    const t = a.t.filter((_, i) => i % step === 0), u = a.u_mid.filter((_, i) => i % step === 0);
    s.addText("Analytical reference: mid-span displacement of the damped fixed–fixed beam (Mode 1)",
      { x: 8.75, y: 1.55, w: 4.0, h: 0.6, fontSize: 12, color: "C9DCE6", margin: 0, isTextBox: true });
    s.addChart(pres.charts.SCATTER, [{ name: "t [s]", values: t }, { name: "u(L/2, t) [m]", values: u }], {
      x: 8.6, y: 2.15, w: 4.2, h: 3.0, chartColors: ["7FC3DC"], lineSize: 1.25, lineDataSymbol: "none", showLegend: false,
      ...chartText, catAxisLabelColor: "9FB6C4", valAxisLabelColor: "9FB6C4", valAxisLabelFormatCode: "0.00", catAxisLabelFormatCode: "0.0",
      catAxisHidden: true, showValAxisTitle: true, valAxisTitle: "u(L/2, t) [m]", valAxisTitleColor: "9FB6C4",
      valGridLine: { style: "none" }, catGridLine: { style: "none" },
      valAxisMaxVal: 0.1, valAxisMinVal: -0.1, valAxisMajorUnit: 0.05, catAxisMinVal: 0, catAxisMaxVal: 1, catAxisMajorUnit: 0.2, catAxisCrossesAt: -0.1,
    });
    s.addText("t from 0 to 1 s", { x: 8.75, y: 5.12, w: 4.0, h: 0.3, fontSize: 11, color: "9FB6C4", align: "center", margin: 0, isTextBox: true });
    s.addText(show("analytic cycles", `${fx(a.f_d_hz, 2)} Hz · ${fx(a.cycles, 2)} cycles in 1 s · decay ${fx(a.decay, 2)} s⁻¹`, "analytic"),
      { x: 8.75, y: 5.5, w: 4.0, h: 0.4, fontSize: 12, color: "C9DCE6", margin: 0, isTextBox: true });
    s.addNotes("Title. Presenter name and institution are placeholders: the repository metadata does not identify the presenter, so they were not guessed. The curve is the closed-form analytical reference solution (exact root of the clamped-beam eigenproblem), not a network prediction. Scope: only completed experiments through B2-E03 (commit cdd87b5) are reported.");
  }

  // ============================================================================ 2. PROBLEM
  {
    const s = content("The problem: accurate dynamics at low compute", "Introduction", `analytical reference (src/beampinn/physics); B2-E01 B1 seed 1234 (results_batch2/tables/B2-E01_RESULTS.csv, figures/B2-E01_displacement_trace.png)`);
    // objective
    card(s, 0.6, 1.4, 5.7, 1.55, C.background2, "objective");
    s.addText("Objective", { x: 0.85, y: 1.5, w: 5.2, h: 0.35, fontSize: 14, bold: true, color: C.accent1, margin: 0, isTextBox: true });
    const objs = ["Lower physical error", "Better dynamic fidelity", "Lower computational cost"];
    objs.forEach((o, i) => {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.85 + i * 1.82, y: 1.95, w: 1.62, h: 0.8, rectRadius: 0.08, fill: { color: C.background1 }, line: { color: C.accent1, width: 1 } });
      s.addText(o, { x: 0.85 + i * 1.82, y: 1.95, w: 1.62, h: 0.8, fontSize: 14, bold: true, color: C.text2, align: "center", valign: "middle", margin: 2, isTextBox: true });
      if (i < 2) s.addText("+", { x: 0.85 + i * 1.82 + 1.62, y: 1.95, w: 0.2, h: 0.8, fontSize: 16, bold: true, color: C.accent1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    });
    s.addText("Tracked on every run", { x: 0.6, y: 3.2, w: 5.7, h: 0.35, fontSize: 14, bold: true, color: C.text2, margin: 0, isTextBox: true });
    const mets = [[fa.FaSquareRootAlt, "PDE residual"], [fa.FaWaveSquare, "Frequency error"], [fa.FaChartLine, "Damping / decay rate"],
      [fa.FaHourglassHalf, "Persistence (cycles)"], [fa.FaMemory, "Peak memory"], [fa.FaStopwatch, "Wall-clock time"]];
    for (let i = 0; i < mets.length; i++) {
      const cx = 0.6 + (i % 2) * 2.9, cy = 3.65 + Math.floor(i / 2) * 0.62;
      await iconDot(s, mets[i][0], cx, cy, 0.48, C.accent1);
      s.addText(mets[i][1], { x: cx + 0.6, y: cy, w: 2.25, h: 0.48, fontSize: 14, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
    }
    s.addText("A low scalar loss is not enough: u ≡ 0 also satisfies the homogeneous beam equation, so a collapsed late-time field can carry a small residual. Dynamics are therefore measured directly.",
      { x: 0.6, y: 5.6, w: 5.7, h: 1.05, fontSize: 13, italic: true, color: MUTED, margin: 0, isTextBox: true });
    // flow
    const fl = [["PDE", "c²u_xxxx + u_tt + γu_t = 0"], ["PINN", "network u_θ(x, t)"], ["Prediction", "vibration u(x, t)"]];
    fl.forEach((f, i) => {
      const x = 6.85 + i * 2.0;
      card(s, x, 1.4, 1.7, 0.95, C.background2);
      s.addText([{ text: f[0], options: { bold: true, breakLine: true, fontSize: 14, color: C.text2 } }, { text: f[1], options: { fontSize: 11, color: MUTED } }],
        { x, y: 1.4, w: 1.7, h: 0.95, align: "center", valign: "middle", margin: 2, isTextBox: true });
      if (i < 2) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 1.72, y: 1.75, w: 0.26, h: 0.25, fill: { color: C.accent5 }, line: { type: "none" } });
    });
    const fm = [["Correct early oscillation", C.accent2], ["Artificial damping", C.accent3], ["Collapse", C.accent4]];
    s.addText("Observed failure mode", { x: 6.85, y: 2.55, w: 5.9, h: 0.3, fontSize: 13, bold: true, color: C.text2, margin: 0, isTextBox: true });
    fm.forEach((f, i) => {
      const x = 6.85 + i * 2.0;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.9, w: 1.7, h: 0.55, rectRadius: 0.08, fill: { color: f[1] }, line: { type: "none" } });
      s.addText(f[0], { x, y: 2.9, w: 1.7, h: 0.55, fontSize: 12, bold: true, color: C.background1, align: "center", valign: "middle", margin: 2, isTextBox: true });
      if (i < 2) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 1.72, y: 3.05, w: 0.26, h: 0.25, fill: { color: C.accent5 }, line: { type: "none" } });
    });
    s.addImage({ path: path.join(HERE, "figures", "fig_B1_midspan_trace_s1234.png"), x: 6.75, y: 3.65, w: 6.0, h: 6.0 * 392 / 1489, altText: "B1 mid-span displacement vs exact, seed 1234" });
    const b1 = D.e01.B1;
    s.addText(show("B1 decay/collapse", `B1 (seed 1234): decay ${fx(b1.fit_decay, 2)} s⁻¹ vs exact ${fx(D.analytic.decay, 2)} s⁻¹; collapse at ${fx(b1.collapse_time_s, 3)} s (${fx(b1.persistence_cycles, 2)} of ${fx(D.analytic.cycles, 2)} cycles)`, "e01"),
      { x: 6.85, y: 5.35, w: 5.9, h: 0.6, fontSize: 13, color: C.text1, margin: 0, isTextBox: true });
    s.addNotes("The goal is not a lower training loss but correct physics per unit compute. Every run reports PDE residual, frequency and decay error, persistence (cycles before the local amplitude falls below 50 % of exact), memory and wall-clock. The trace is the real B1 control (seed 1234): the first cycles are right, then the response is over-damped and collapses.");
  }

  // ============================================================================ 3. BENCHMARK
  {
    const s = content("Starting point: the published benchmark", "Introduction", "Söyleyici & Ünver, EAAI 141 (2025) 109804; references/batch1/ (paper_benchmark_registry.csv, optimization_leaderboard.csv, PHASE_A1_CHECKPOINT.md)");
    statusPill(s, "BENCH");
    card(s, 0.6, 1.4, 7.1, 1.75);
    s.addText([
      { text: "Cem Söyleyici & Hakkı Özgür Ünver", options: { bold: true, breakLine: true } },
      { text: "“A Physics-Informed Deep Neural Network based beam vibration framework for simulation and parameter identification”", options: { italic: true, breakLine: true } },
      { text: "Engineering Applications of Artificial Intelligence, 141 (2025) 109804 · DOI 10.1016/j.engappai.2024.109804", options: { color: MUTED } },
    ], { x: 0.85, y: 1.5, w: 6.65, h: 1.55, fontSize: 15, color: C.text1, valign: "top", margin: 0, isTextBox: true, paraSpaceAfter: 6 });
    const p = D.paper_FE_D_M1, a = D.analytic;
    const rows = [
      ["Method", "Spatio-temporal Fourier feature mapping + NTK adaptive loss weighting"],
      ["Network", "6 × 200 tanh · Adam lr 1e-4 · 45,000 epochs"],
      ["Benchmark FE-D-M1", show("benchmark", `Fixed–fixed damped beam · L = ${a.L} m · Mode 1 · ${fx(a.f_d_hz, 2)} Hz · ${fx(a.cycles, 2)} cycles in 1 s`, "analytic")],
      ["Published accuracy", show("paper L2", `Relative L2 = ${p.paper_error} (paper's reported value)`, "paper registry")],
    ];
    s.addTable(rows.map((r) => [{ text: r[0], options: { bold: true, color: C.text2 } }, { text: r[1] }]), {
      x: 0.6, y: 3.4, w: 7.1, colW: [2.0, 5.1], fontSize: 13, color: C.text1, border: { type: "solid", color: "D9E0E6", pt: 0.75 }, rowH: 0.56, valign: "middle", margin: 0.08 });
    // right
    const c0 = D.batch1_progression.find((r) => r.key === "C0");
    card(s, 8.1, 1.4, 4.65, 2.6, C.background2);
    s.addText("Our controlled re-implementation of the paper method (C0)", { x: 8.35, y: 1.5, w: 4.2, h: 0.6, fontSize: 13, bold: true, color: C.text2, margin: 0, isTextBox: true });
    s.addText(show("C0 L2", fx(c0.L2_exact, 2), "batch1 leaderboard"), { x: 8.35, y: 2.1, w: 2.0, h: 0.95, fontSize: 48, bold: true, color: C.accent4, margin: 0, isTextBox: true, fontFace: "Cambria" });
    s.addText("relative L2 · static (non-oscillating) field · NTK weights grew to ~10¹⁴–10¹⁵", { x: 10.2, y: 2.15, w: 2.45, h: 0.95, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "middle" });
    const paperEvals = 45000 * 640;
    s.addText(show("C0 budget", `${thou(c0.pde_evaluations)} PDE evaluations = ${fx(100 * c0.pde_evaluations / paperEvals, 1)} % of the paper's ${thou(paperEvals)} (45,000 epochs × 640)`, "batch1 leaderboard; paper registry"),
      { x: 8.35, y: 3.1, w: 4.2, h: 0.75, fontSize: 12, color: MUTED, margin: 0, isTextBox: true });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.1, y: 4.2, w: 4.65, h: 1.85, rectRadius: 0.08, fill: { color: C.background1 }, line: { color: C.accent1, width: 1.25 } });
    s.addText([
      { text: "This is our incumbent, not our novelty.", options: { bold: true, breakLine: true, color: C.accent1 } },
      { text: "Research question: can we reproduce it, diagnose why it loses dynamic fidelity, and improve physical accuracy per unit compute?", options: { color: C.text1 } },
    ], { x: 8.35, y: 4.3, w: 4.2, h: 1.65, fontSize: 15, valign: "middle", margin: 0, isTextBox: true, paraSpaceAfter: 8 });
    s.addNotes("The paper reports relative L2 = 4.64e-4 with Fourier features + NTK weighting. In our controlled implementation at our budget (2.2 % of the paper's evaluation count), the paper-faithful configuration C0 settled on a static field (L2 3.26) while the NTK weights grew to ~1e14–1e15. This is an observation under our budget, not a claim that the paper is wrong.");
  }

  // ============================================================================ 4. INFRASTRUCTURE
  pres.addSection({ title: "What we built and tested" });
  {
    const s = content("Research infrastructure achieved", "What we built and tested", "configs/root_provenance.json; results_batch2/EXPERIMENT_REGISTRY.csv; tests/ (pytest at cdd87b5); docs/hypotheses/B2-E0{1,2,3}.md");
    statusPill(s, "ACHIEVED");
    const items = [
      [fa.FaLock, "Frozen ROOT provenance", "Batch-1 repo frozen at d31864a; byte-identical after every phase"],
      [fa.FaCodeBranch, "Independent PINN-RP", "All new work in a separate repository; ROOT never written"],
      [fa.FaClipboardList, "Experiment registry", show("registry runs", `${D.counts.batch2_runs} Batch-2 runs, each with config hash, git SHA, seed`, "registry")],
      [fa.FaBook, "Benchmark + exact reference", "Closed-form damped Mode-1 solution; paper-faithful and exact-root references"],
      [fa.FaRulerCombined, "Fixed evaluation", "201 × 2001 grid; frozen persistence metric; 51 × 501 residual grid"],
      [fa.FaTachometerAlt, "Compute accounting", "PDE evaluations, forward/backward passes, peak RAM, wall-clock"],
      [fa.FaFileSignature, "Pre-registration", "Hypotheses and decision thresholds committed before any training"],
      [fa.FaRandom, "Seed replication", "Seeds 1234–1236 for every Batch-2 arm; bit-exact control re-runs"],
      [fa.FaSave, "Checkpoints + safety", "Snapshots every 128k evaluations; runtime guard blocks writes to ROOT"],
      [fa.FaCheckDouble, "Automated tests", show("tests", `${D.tests_passing_at_B2E03} tests passing at B2-E03 completion`, "pytest at cdd87b5")],
    ];
    for (let i = 0; i < items.length; i++) {
      const col = i % 5, row = Math.floor(i / 5);
      const x = 0.6 + col * 2.45, y = 1.5 + row * 2.65;
      card(s, x, y, 2.25, 2.45);
      await iconDot(s, items[i][0], x + 0.2, y + 0.2, 0.62, C.accent2);
      s.addText(items[i][1], { x: x + 0.2, y: y + 0.95, w: 1.9, h: 0.55, fontSize: 14, bold: true, color: C.text2, margin: 0, isTextBox: true, valign: "top" });
      s.addText(items[i][2], { x: x + 0.2, y: y + 1.5, w: 1.9, h: 0.85, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
    }
    s.addNotes("Infrastructure is a result in itself: every number in this deck is reproducible from a registered run, a frozen evaluator and a committed pre-registration. ROOT (the Batch-1 repository) is frozen at d31864a and was verified byte-identical after every phase. 168 tests passed at the B2-E03 completion commit (re-verified from a clean export of cdd87b5).");
  }

  // ============================================================================ 5. COUNTS
  {
    const s = content("What we tested", "What we built and tested", "results_batch2/EXPERIMENT_REGISTRY.csv (42 runs); references/batch1/optimization_leaderboard.csv (26 runs); counting rules in speaker notes");
    const k = D.counts;
    const stats = [
      [show("total runs", String(k.total_runs), "registry+leaderboard"), "completed training runs", `${k.batch1_runs} Batch 1 + ${k.batch2_runs} Batch 2`],
      [show("distinct configs", String(k.total_distinct_configs), "registry+leaderboard"), "distinct configurations", `${k.batch1_distinct_configs} Batch 1 + ${k.batch2_new_configs} new in Batch 2`],
      [show("families", String(k.n_families), "family table"), "intervention families", `in ${k.n_classes} classes`],
      [show("seeds", String(k.batch2_seeds.length), "registry"), "seeds per Batch-2 arm", `${k.batch2_seeds.join(", ")}`],
    ];
    stats.forEach((st, i) => {
      const y = 1.4 + i * 1.33;
      card(s, 0.6, y, 3.4, 1.18);
      s.addText(st[0], { x: 0.8, y, w: 1.25, h: 1.18, fontSize: 40, bold: true, color: C.accent1, fontFace: "Cambria", valign: "middle", margin: 0, isTextBox: true });
      s.addText([{ text: st[1], options: { bold: true, breakLine: true, color: C.text2 } }, { text: st[2], options: { color: MUTED, fontSize: 12 } }],
        { x: 2.05, y, w: 1.9, h: 1.18, fontSize: 14, valign: "middle", margin: 0, isTextBox: true });
    });
    const cats = Object.entries(D.families);
    const colX = [4.3, 6.45, 8.6, 10.75], colW = 2.0;
    cats.forEach(([cat, fams], i) => {
      const x = colX[i];
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.4, w: colW, h: 0.7, rectRadius: 0.08, fill: { color: C.text2 }, line: { type: "none" } });
      s.addText(cat, { x, y: 1.4, w: colW, h: 0.7, fontSize: 13, bold: true, color: C.background1, align: "center", valign: "middle", margin: 3, isTextBox: true });
      card(s, x, 2.2, colW, 4.5);
      s.addText(Object.keys(fams).map((f, j, arr) => ({ text: f, options: { bullet: true, breakLine: j < arr.length - 1 } })),
        { x: x + 0.08, y: 2.3, w: colW - 0.16, h: 4.3, fontSize: 13, color: C.text1, valign: "top", margin: 2, isTextBox: true, paraSpaceAfter: 6 });
    });
    s.addNotes(`Counting rules (from the registry and the Batch-1 leaderboard, not estimated): a run = one completed training run. Distinct configurations count seed replicates once (Batch 1: Z2 = Y1 at seed 1235) and bit-exact control re-runs once (B2-E02 M0 = B2-E01 modal; B2-E02T B1 and B2-E03 B1 = B2-E01 B1). Batch-2 B1 is the Batch-1 Z4 configuration, so Batch 2 adds ${k.batch2_new_configs} new configurations. The ${k.total_distinct_configs} configurations include the vanilla baseline and three supervised representation checks, which are controls and are not counted as intervention families. The superseded legacy notebook study is not counted. Batch-1 runs are single-seed (1234; one run at 1235).`);
  }

  // ============================================================================ 6. BATCH 1
  pres.addSection({ title: "Results" });
  {
    const s = content("Batch 1: reproduction and first optimizations", "Results", "references/batch1/optimization_leaderboard.csv; tables/phaseZ4_20K_checkpoints.csv, phaseY1_20K_checkpoints.csv, phaseX_oscillation.csv; reports/PHASE_E_SCREEN.md");
    const P = D.batch1_progression;
    const evalLbl = (n) => (n >= 1e6 ? `${fx(n / 1e6, 2)}M` : `${n / 1000}k`);
    s.addChart(pres.charts.BAR, [{ name: "relative L2 (exact)", labels: P.map((r) => `${r.key} (${evalLbl(r.pde_evaluations)})`), values: P.map((r) => Number(fx(r.L2_exact, 3))) }], {
      x: 0.5, y: 1.35, w: 6.6, h: 3.85, barDir: "col", chartColors: [H.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
      showTitle: true, title: "Relative L2 error, Batch-1 progression (lower is better)", showLegend: false, ...chartText, ...quiet(),
      valAxisMinVal: 0, valAxisMaxVal: 3.6, valAxisMajorUnit: 1, showCatAxisTitle: true, catAxisTitle: "configuration (PDE evaluations)",
    });
    P.forEach((r) => show(`b1 L2 ${r.key}`, fx(r.L2_exact, 2), "leaderboard"));
    const pers = P.filter((r) => r.persistence_cycles !== null);
    s.addText([{ text: "Persistence (cycles of ", options: {} }, { text: fx(D.analytic.cycles, 2), options: {} }, { text: "): ", options: {} },
      ...pers.map((r, i) => ({ text: show(`b1 pers ${r.key}`, `${r.key} ${fx(r.persistence_cycles, 2)}`, "checkpoints") + (i < pers.length - 1 ? " → " : ""), options: { bold: true } })),
      { text: " · C0, E4, X2: no sustained oscillation", options: { color: MUTED } }],
      { x: 0.6, y: 5.3, w: 6.5, h: 0.6, fontSize: 13, color: C.text1, margin: 0, isTextBox: true });
    s.addText("Budgets differ (labels): E4/X2/Y1 at 160k, C0/Z4 at 640k, Z4-20K at 2.56M evaluations. Z4 = the frozen B1 reference used in Batch 2.",
      { x: 0.6, y: 5.95, w: 6.5, h: 0.65, fontSize: 12, italic: true, color: MUTED, margin: 0, isTextBox: true });
    const sup = D.supervised;
    const rowsF = [
      ["FAILED", "Paper method (Fourier + NTK)", "Static field under the tested budget (C0 L2 3.26)"],
      ["ACHIEVED", "Hard IC/BC constraints + tanh²(ω₁t)", "IC/BC exact by construction (4.8e-6 / 1.4e-5); required network output O(1) instead of ≈ −8,369"],
      ["ACHIEVED", "LR decay · mini-batch 128 · budget", show("b1 progression", `Persistence ${fx(pers[0].persistence_cycles, 2)} → ${fx(pers[1].persistence_cycles, 2)} → ${fx(pers[2].persistence_cycles, 2)} cycles`, "checkpoints")],
      ["ACHIEVED", "Supervised capacity check", show("supervised w", `Network recovers ω = ${fx(sup.X3_supervised_paperconv.fit_w, 2)} / ${fx(sup.X4_supervised_rcconv.fit_w, 2)} rad/s (exact ${fx(sup.X4_supervised_rcconv.w_exact, 2)}): capacity is not the bottleneck`, "phaseX_oscillation")],
      ["FAILED", "RAD adaptive sampling", "Did not solve this benchmark failure under the tested budget"],
    ];
    rowsF.forEach((r, i) => {
      const y = 1.4 + i * 1.05;
      tag(s, r[0] === "FAILED" ? "FAILED" : "ACHIEVED", 7.45, y + 0.05, 2.35);
      s.addText(r[1], { x: 9.95, y, w: 2.8, h: 0.4, fontSize: 13, bold: true, color: C.text2, margin: 0, isTextBox: true, valign: "top" });
      s.addText(r[2], { x: 7.45, y: y + 0.42, w: 5.3, h: 0.58, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
    });
    s.addNotes("Batch 1 (single seed) moved the full-field PINN from a static field to sustained early oscillation. Hard constraints make IC/BC exact; tanh²(ω₁t) removes an ill-conditioned target (required N* ≈ −ω²/2 = −8,369 with (t/T)² vs −1.9 to −0.16 with tanh²). LR decay, mini-batch 128 and more budget progressively extend persistence. Supervised controls show the network can represent the 20.6 Hz solution. Unsuccessful interventions are reported as 'did not solve this benchmark failure under the tested budget', not as generally ineffective. Best Batch-1 run (Z4-20K): L2 0.263, 8.09 of 20.58 cycles at 2.56M evaluations, still over-damped and collapsing at 0.393 s.");
  }

  // ============================================================================ 7. B2-E01
  {
    const s = content("B2-E01: one change at a time, matched compute", "Results", "results_batch2/tables/B2-E01_RESULTS.csv (seed 1234, 640,000 PDE evaluations per arm); reports/B2-E01_REPORT.md");
    const e = D.e01, arms = ["B1", "B1_fp64", "mixed", "modal"];
    const hdr = ["Metric", "B1", "B1 FP64", "Mixed v = u_xx", "Modal q(t)"];
    const fmt = {
      "Persistence [cycles]": (r) => fx(r.persistence_cycles, 2),
      "L2 (exact)": (r) => fx(r.L2_exact, 3),
      "Frequency error": (r) => `${r.frequency_error_signed_pct < 0 ? "−" : "+"}${Math.abs(r.frequency_error_signed_pct) >= 0.1 ? fx(Math.abs(r.frequency_error_signed_pct), r.frequency_error_signed_pct > -1.5 && r.frequency_error_signed_pct < 1.5 ? 2 : 1) : fx(Math.abs(r.frequency_error_signed_pct), 2)} %`,
      "Decay rate [s⁻¹] (exact 3.54)": (r) => fx(r.fit_decay, 2),
      "u PDE residual": (r) => fx(r.PDE_residual_rel, r.PDE_residual_rel >= 1 ? 2 : 3),
      "Training time [s]": (r) => thou(Math.round(r.train_seconds)),
      "Peak RAM [MB]": (r) => thou(Math.round(r.peak_rss_mb)),
    };
    const body = Object.entries(fmt).map(([k, f]) => [{ text: k, options: { bold: true, color: C.text2 } },
      ...arms.map((a) => ({ text: show(`e01 ${k} ${a}`, f(e[a]), "e01"), options: { align: "right", bold: a === "modal", color: a === "modal" ? C.accent3 : C.text1 } }))]);
    s.addTable([hdr.map((h, i) => ({ text: h, options: { bold: true, color: C.background1, fill: { color: i === 4 ? C.accent3 : C.text2 }, align: i ? "right" : "left" } })), ...body], {
      x: 0.6, y: 1.4, w: 7.4, colW: [2.6, 1.1, 1.15, 1.35, 1.2], fontSize: 13, border: { type: "solid", color: "D9E0E6", pt: 0.75 }, rowH: 0.5, valign: "middle", margin: 0.06 });
    s.addText("Seed 1234 · every arm 640,000 PDE evaluations · 4 arms run concurrently on 4 vCPU", { x: 0.6, y: 5.5, w: 7.4, h: 0.35, fontSize: 12, italic: true, color: MUTED, margin: 0, isTextBox: true });
    const cards = [
      ["FAILED", "FP64 precision", show("fp64 ratio", `No accuracy change; ${fx(e.B1_fp64.train_seconds / e.B1.train_seconds, 2)}× wall-clock, +${fx(100 * (e.B1_fp64.peak_rss_mb / e.B1.peak_rss_mb - 1), 0)} % RAM`, "e01")],
      ["FAILED", "Mixed formulation", show("mixed ratio", `${fx(e.mixed.train_seconds / e.B1.train_seconds, 2)}× wall-clock, but worse physics: u-residual ${fx(e.mixed.PDE_residual_rel, 2)} vs ${fx(e.B1.PDE_residual_rel, 3)}`, "e01")],
      ["DIAGNOSTIC", "Modal reduction", show("modal e01", `${fx(e.modal.persistence_cycles, 2)} cycles, L2 ${fx(e.modal.L2_exact, 3)} — exact spatial mode supplied`, "e01")],
    ];
    cards.forEach((c, i) => {
      const y = 1.4 + i * 1.32;
      card(s, 8.35, y, 4.4, 1.17);
      tag(s, c[0], 8.55, y + 0.15, c[0] === "DIAGNOSTIC" ? 1.45 : 2.6);
      s.addText(c[1], { x: c[0] === "DIAGNOSTIC" ? 10.15 : 11.3, y: y + 0.1, w: 2.5, h: 0.4, fontSize: 13, bold: true, color: C.text2, margin: 0, isTextBox: true, valign: "middle" });
      s.addText(c[2], { x: 8.55, y: y + 0.52, w: 4.05, h: 0.6, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
    });
    card(s, 8.35, 5.4, 4.4, 1.3, C.background1);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.35, y: 5.4, w: 4.4, h: 1.3, rectRadius: 0.08, fill: { color: C.background1 }, line: { color: C.accent3, width: 1.25 } });
    s.addText("Removing the spatial learning burden substantially improves temporal learning.", { x: 8.55, y: 5.45, w: 4.0, h: 1.2, fontSize: 15, bold: true, italic: true, color: C.text2, valign: "middle", margin: 0, isTextBox: true });
    s.addNotes("FP64 and mixed are labelled 'failed on this benchmark': they did not improve the physical solution here. FP64 is identical to FP32 to the reported digits at 1.71× the wall-clock. Mixed halves the cost per step but its displacement violates the beam equation (u-residual 2.57). Modal supplies the exact spatial mode shape and learns only q(t): it is a diagnostic, not a competitor.");
  }

  // ============================================================================ 8. SEEDS
  {
    const s = content("Three-seed replication: what is robust", "Results", "results_batch2/tables/B2-E01S_SEED_RESULTS.csv, B2-E01S_SEED_SNAPSHOTS.csv; reports/B2-E01_SEED_REPLICATION.md");
    const r = D.e01s;
    const arms = [["B1", "B1"], ["B1_fp64", "B1 FP64"], ["mixed", "Mixed"], ["modal", "Modal (diag.)"]];
    const series = SEEDS.map((sd) => ({ name: `seed ${sd}`, labels: arms.map((a) => a[1]), values: arms.map((a) => Number(fx(r.find((x) => x.arm === a[0] && x.seed === sd).persistence_cycles, 2))) }));
    s.addChart(pres.charts.BAR, series, {
      x: 0.5, y: 1.35, w: 6.9, h: 5.4, barDir: "col", barGrouping: "clustered", chartColors: ["1F6F8B", "7FA7BA", "B7C9D3"],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", showTitle: true, title: "Persistence [cycles before collapse] by arm and seed",
      showLegend: true, legendPos: "b", ...chartText, ...quiet(), valAxisMinVal: 0, valAxisMaxVal: 9, valAxisMajorUnit: 3,
    });
    const g = (arm, key) => bySeed(r, arm).map((x) => x[key]);
    const ratio = (num, den, key) => SEEDS.map((sd) => r.find((x) => x.arm === num && x.seed === sd)[key] / r.find((x) => x.arm === den && x.seed === sd)[key]);
    const l2diff = Math.max(...SEEDS.map((sd) => Math.abs(r.find((x) => x.arm === "B1_fp64" && x.seed === sd).L2_exact / r.find((x) => x.arm === "B1" && x.seed === sd).L2_exact - 1)));
    const mb = Object.values(D.e01s_modal_best);
    const pr = ratio("modal", "B1", "persistence_cycles");
    const items = [
      ["FAILED", "FP64 ≡ FP32 in 3/3 seeds", show("fp64 seeds", `max relative L2 difference ${l2diff.toExponential(1)}; ${rng(ratio("B1_fp64", "B1", "train_seconds"), 2)}× wall-clock`, "e01s")],
      ["FAILED", "Mixed: cheaper, never more accurate", show("mixed seeds", `${rng(ratio("mixed", "B1", "train_seconds"), 2)}× wall-clock; L2 ${rng(g("mixed", "L2_exact"), 3)} vs B1 ${rng(g("B1", "L2_exact"), 3)}`, "e01s")],
      ["DIAGNOSTIC", "Modal: longer persistence in 3/3 seeds", show("modal seeds", `×${fx(pr[0], 2)}, ×${fx(pr[1], 2)}, ×${fx(pr[2], 2)} vs B1; collapses at ${rng(g("modal", "collapse_time_s"), 3)} s, not 1 s`, "e01s")],
      ["DIAGNOSTIC", "Modal improves, then degrades", show("modal peak", `peaks at ${rng(mb.map((m) => m.persistence_cycles), 2)} cycles (384k–512k), then retreats by 640k`, "e01s snapshots")],
      ["BENCH", "B1 seed variability", show("b1 seeds", `${rng(g("B1", "persistence_cycles"), 2)} cycles; L2 ${rng(g("B1", "L2_exact"), 3)}`, "e01s")],
    ];
    items.forEach((it, i) => {
      const y = 1.4 + i * 1.07;
      tag(s, it[0], 7.75, y + 0.04, it[0] === "FAILED" ? 2.55 : 1.5);
      s.addText(it[1], { x: 7.75, y: y + 0.38, w: 5.0, h: 0.32, fontSize: 13, bold: true, color: C.text2, margin: 0, isTextBox: true });
      s.addText(it[2], { x: 7.75, y: y + 0.68, w: 5.0, h: 0.34, fontSize: 12, color: C.text1, margin: 0, isTextBox: true });
    });
    s.addNotes("Robust across seeds: FP64 is not material; mixed is cheaper but never more accurate; modal has longer persistence than B1 in every seed, but the magnitude varies (seed 1236 gain is within the pre-registered comparability band, because B1 itself persisted longest there). The modal front reaches its best state at 384k–512k evaluations and then retreats: a reproducible late-training degradation. Not presented as a universal result: one benchmark, three seeds.");
  }

  // ============================================================================ 9. B2-E02 modal optimizer
  {
    const s = content("B2-E02: the optimizer test (modal diagnostic)", "Results", "results_batch2/B2-E02_RESULTS.csv; figures/B2-E02_q_of_t.png (seed 1234); reports/B2-E02_REPORT.md");
    statusPill(s, "DIAGNOSTIC");
    s.addImage({ path: path.join(HERE, "figures", "fig_modal_M0_vs_LBFGS_q_s1234.png"), x: 0.55, y: 1.4, w: 6.3, h: 6.3 * 774 / 1489, altText: "Modal q(t): Adam control vs Adam to L-BFGS, seed 1234" });
    const e = D.e02;
    const ar = (arm, key) => e.filter((r) => r.arm === arm).map((r) => r[key]);
    const wrel = (arm) => SEEDS.map((sd) => e.find((r) => r.arm === arm && r.seed === sd).train_seconds / e.find((r) => r.arm === "M0" && r.seed === sd).train_seconds);
    const rows = [
      ["M0 Adam + decay", `${rng(ar("M0", "persistence_cycles"), 1)}`, rng(ar("M0", "L2_exact"), 2), `${rng(ar("M0", "fit_decay"), 1)}`, "1.00"],
      ["Adam → L-BFGS", `${rng(ar("LBFGS", "persistence_cycles"), 2)}`, rng(ar("LBFGS", "L2_exact"), 3), `${rng(ar("LBFGS", "fit_decay"), 2)}`, rng(wrel("LBFGS"), 2)],
      ["Causal weighting", `${rng(ar("CAUSAL", "persistence_cycles"), 1)}`, rng(ar("CAUSAL", "L2_exact"), 2), "fit not meaningful", rng(wrel("CAUSAL"), 2)],
    ];
    rows.forEach((r) => r.slice(1).forEach((v, j) => show(`e02 ${r[0]} ${j}`, v, "e02")));
    const hdr = ["Arm (3 seeds)", "Persistence [cycles]", "L2", "Decay [s⁻¹]", "Rel. wall-clock"];
    s.addTable([hdr.map((h, i) => ({ text: h, options: { bold: true, color: C.background1, fill: { color: C.text2 }, align: i ? "right" : "left" } })),
      ...rows.map((r, i) => r.map((v, j) => ({ text: v, options: { align: j ? "right" : "left", bold: i === 1, color: i === 1 ? C.accent2 : C.text1 } })))], {
      x: 7.1, y: 1.4, w: 5.65, colW: [1.65, 1.15, 0.95, 0.95, 0.95], fontSize: 12, border: { type: "solid", color: "D9E0E6", pt: 0.75 }, rowH: 0.55, valign: "middle", margin: 0.05 });
    card(s, 7.1, 3.75, 2.2, 1.55);
    s.addText([{ text: show("modal full window", fx(ar("LBFGS", "persistence_cycles")[0], 2), "e02"), options: { fontSize: 32, bold: true, color: C.accent2, fontFace: "Cambria", breakLine: true } },
      { text: "cycles = full 1-s window, 3/3 seeds", options: { fontSize: 12, color: C.text1 } }], { x: 7.25, y: 3.8, w: 1.95, h: 1.45, valign: "middle", margin: 0, isTextBox: true });
    card(s, 9.5, 3.75, 3.25, 1.55);
    s.addText([{ text: show("modal L2 drop", `${rng(ar("M0", "L2_exact"), 2)} → ${rng(ar("LBFGS", "L2_exact"), 3)}`, "e02"), options: { fontSize: 18, bold: true, color: C.accent2, breakLine: true } },
      { text: "relative L2: Adam control → Adam → L-BFGS", options: { fontSize: 12, color: C.text1 } }], { x: 9.65, y: 3.8, w: 3.0, h: 1.45, valign: "middle", margin: 0, isTextBox: true });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.1, y: 5.5, w: 5.65, h: 1.2, rectRadius: 0.08, fill: { color: C.accent3 }, line: { type: "none" } });
    const ram = SEEDS.map((sd) => e.find((r) => r.arm === "LBFGS" && r.seed === sd).peak_rss_mb / e.find((r) => r.arm === "M0" && r.seed === sd).peak_rss_mb - 1);
    s.addText([{ text: "Diagnostic only — not yet a valid full-field solution.", options: { bold: true, breakLine: true } },
      { text: show("modal ram", `Exact spatial mode supplied · ${thou(ar("LBFGS", "pde_evaluations")[0])} vs ${thou(ar("M0", "pde_evaluations")[0])} PDE evaluations · +${fx(100 * Math.min(...ram), 0)}–${fx(100 * Math.max(...ram), 0)} % peak RAM`, "e02"), options: { fontSize: 12 } }],
      { x: 7.3, y: 5.55, w: 5.3, h: 1.1, fontSize: 15, color: C.background1, valign: "middle", margin: 0, isTextBox: true });
    s.addNotes("Pre-registered modal screen, three seeds, same initialisation and budget. Adam for 320k evaluations, then full-batch L-BFGS on a fixed 640-point set: full-window persistence in all seeds, L2 down 3.5–7×, decay close to exact (3.54). Causal weighting was worse in every seed. Wall-clock ratios are within-batch. This result uses the exact spatial mode shape: it shows the temporal sub-problem is solvable by better optimization, not that the full-field PINN is solved.");
  }

  // ============================================================================ 10. TRANSFER
  {
    const s = content("Transfer to the full field: the key negative result", "Results", "results_batch2/B2-E02T_RESULTS.csv, B2-E02T_SNAPSHOTS.csv, B2-E02T_CLASSIFICATION.json; reports/B2-E02_REPORT.md §4");
    statusPill(s, "FAILED");
    const sn = D.e02t_snap;
    const lev = [128000, 256000, 384000, 512000];
    const pick = (arm, sd, ev) => sn.find((r) => r.arm === arm && r.seed === sd && r.pde_evaluations === ev);
    const endOf = (arm, sd) => sn.filter((r) => r.arm === arm && r.seed === sd).sort((a, b) => b.pde_evaluations - a.pde_evaluations)[0];
    const labels = ["128k", "256k", "384k", "512k", "end"];
    const ser = [];
    for (const sd of SEEDS) {
      ser.push({ name: `B1 s${sd}`, labels, values: [...lev.map((ev) => Number(fx(pick("B1", sd, ev).persistence_cycles, 2))), Number(fx(endOf("B1", sd).persistence_cycles, 2))] });
    }
    for (const sd of SEEDS) {
      ser.push({ name: `Adam→L-BFGS s${sd}`, labels, values: [...lev.map((ev) => Number(fx(pick("LBFGS", sd, ev).persistence_cycles, 2))), Number(fx(endOf("LBFGS", sd).persistence_cycles, 2))] });
    }
    s.addChart(pres.charts.LINE, ser, {
      x: 0.5, y: 1.35, w: 6.6, h: 5.0, chartColors: ["1F6F8B", "1F6F8B", "1F6F8B", "B03A2E", "B03A2E", "B03A2E"], lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 6,
      showTitle: true, title: "Full field: persistence vs training progress (switch at 320k)", showLegend: true, legendPos: "b", ...chartText, ...quiet(),
      valAxisMinVal: 0, valAxisMaxVal: 6, valAxisMajorUnit: 1, showValAxisTitle: true, valAxisTitle: "persistence [cycles]", showCatAxisTitle: true, catAxisTitle: "PDE evaluations",
    });
    s.addText("end = 640,000 (B1) / 623,360 (Adam → L-BFGS) evaluations; identical up to 256k", { x: 0.6, y: 6.4, w: 6.5, h: 0.35, fontSize: 11, italic: true, color: MUTED, margin: 0, isTextBox: true });
    const t = D.e02t;
    const row = (sd) => {
      const b = t.find((r) => r.arm === "B1" && r.seed === sd), l = t.find((r) => r.arm === "LBFGS" && r.seed === sd);
      const dP = l.persistence_cycles - b.persistence_cycles;
      return [String(sd), show(`dP ${sd}`, `${dP < 0 ? "−" : "+"}${fx(Math.abs(dP), 1)}`, "e02t"), show(`L2r ${sd}`, `×${fx(l.L2_exact / b.L2_exact, 2)}`, "e02t"), show(`Rr ${sd}`, `×${fx(l.PDE_residual_rel / b.PDE_residual_rel, 2)}`, "e02t")];
    };
    s.addTable([["Seed", "Persistence change [cycles]", "L2 vs B1", "PDE residual vs B1"].map((h, i) => ({ text: h, options: { bold: true, color: C.background1, fill: { color: C.text2 }, align: i ? "right" : "left" } })),
      ...SEEDS.map((sd) => row(sd).map((v, j) => ({ text: v, options: { align: j ? "right" : "left", color: j ? C.accent4 : C.text1, bold: j > 0 } })))], {
      x: 7.45, y: 1.4, w: 5.3, colW: [0.8, 1.7, 1.3, 1.5], fontSize: 13, border: { type: "solid", color: "D9E0E6", pt: 0.75 }, rowH: 0.5, valign: "middle", margin: 0.06 });
    const wr = SEEDS.map((sd) => t.find((r) => r.arm === "LBFGS" && r.seed === sd).train_seconds / t.find((r) => r.arm === "B1" && r.seed === sd).train_seconds);
    const rr = SEEDS.map((sd) => t.find((r) => r.arm === "LBFGS" && r.seed === sd).peak_rss_mb / t.find((r) => r.arm === "B1" && r.seed === sd).peak_rss_mb);
    s.addText(show("transfer cost", `Cheaper in time (${rng(wr, 2)}× wall-clock) but ${rng(rr, 2)}× peak RAM, and worse on every accuracy metric in 3/3 seeds.`, "e02t"),
      { x: 7.45, y: 3.55, w: 5.3, h: 0.8, fontSize: 13, color: C.text1, margin: 0, isTextBox: true });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.45, y: 4.55, w: 5.3, h: 2.1, rectRadius: 0.08, fill: { color: C.text2 }, line: { type: "none" } });
    s.addText("The optimizer that solves the isolated temporal problem does not automatically solve the coupled space–time problem.",
      { x: 7.7, y: 4.6, w: 4.85, h: 2.0, fontSize: 18, bold: true, color: C.background1, valign: "middle", margin: 0, isTextBox: true, fontFace: "Cambria" });
    s.addNotes("Same Adam → L-BFGS controller, same accounting, applied to the frozen full-field B1 configuration (pre-registered transfer test). It fails the transfer criteria in every seed: persistence −1.6 to −2.6 cycles, L2 ×1.28–1.54, u-residual ×1.22–2.38 relative to B1. After the switch the L-BFGS front stops advancing while B1 keeps advancing. Decision E: the improvement exists only in the modal diagnostic.");
  }

  // ============================================================================ 11. B2-E03
  {
    const s = content("B2-E03: sparse collocation alone is not the root cause", "Results", "results_batch2/B2-E03_RESULTS.csv, B2-E03_CLASSIFICATION.json (CASE D); reports/B2-E03_REPORT.md");
    const e = D.e03;
    const arms = [["LBFGS-F", "L-BFGS fixed 640"], ["ADAM-FULL", "Adam fixed 640"], ["LBFGS-R", "L-BFGS resampled"], ["LBFGS-4X", "L-BFGS 2,560 pool"], ["B1", "B1"]];
    const val = (arm, sd, k) => { const r = e.find((x) => x.arm === arm && x.seed === sd); return r && r[k] !== null ? r[k] : null; };
    const gser = SEEDS.map((sd) => ({ name: `seed ${sd}`, labels: arms.map((a) => a[1]), values: arms.map((a) => { const v = val(a[0], sd, "G"); return v === null ? null : Number(fx(v, 1)); }) }));
    s.addChart(pres.charts.BAR, gser, {
      x: 0.5, y: 1.35, w: 6.2, h: 3.55, barDir: "col", barGrouping: "clustered", chartColors: ["1F6F8B", "7FA7BA", "B7C9D3"],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", showTitle: true, title: "Residual gap G = dense / training residual (log scale)",
      showLegend: true, legendPos: "b", ...chartText, ...quiet(), valAxisLogScaleBase: 10, valAxisMinVal: 1, valAxisMaxVal: 100,
    });
    const pser = SEEDS.map((sd) => ({ name: `seed ${sd}`, labels: arms.map((a) => a[1]), values: arms.map((a) => { const v = val(a[0], sd, "persistence_cycles"); return v === null ? null : Number(fx(v, 2)); }) }));
    s.addChart(pres.charts.BAR, pser, {
      x: 6.75, y: 1.35, w: 6.1, h: 3.55, barDir: "col", barGrouping: "clustered", chartColors: ["1F6F8B", "7FA7BA", "B7C9D3"],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", showTitle: true, title: "Persistence at end of training [cycles]",
      showLegend: true, legendPos: "b", ...chartText, ...quiet(), valAxisMinVal: 0, valAxisMaxVal: 6, valAxisMajorUnit: 2,
    });
    const gr = (arm) => SEEDS.map((sd) => val(arm, sd, "G")).filter((v) => v !== null);
    const txt = arms.map((a) => `${a[1]} ${rng(gr(a[0]), a[0] === "LBFGS-F" || a[0] === "ADAM-FULL" ? 0 : 1)}×`);
    arms.forEach((a) => show(`G ${a[0]}`, `${rng(gr(a[0]), a[0] === "LBFGS-F" || a[0] === "ADAM-FULL" ? 0 : 1)}×`, "e03"));
    s.addText([{ text: "Residual gap G: ", options: { bold: true } }, { text: txt.join(" · ") }],
      { x: 0.6, y: 5.0, w: 12.15, h: 0.45, fontSize: 13, color: C.text1, margin: 0, isTextBox: true });
    s.addText("L-BFGS 2,560 pool, seed 1234: diverged at 576,640 evaluations (no bar; excluded by pre-registered Amendment 1).", { x: 0.6, y: 5.42, w: 12.15, h: 0.35, fontSize: 11, italic: true, color: MUTED, margin: 0, isTextBox: true });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.85, w: 12.15, h: 0.95, rectRadius: 0.08, fill: { color: C.text2 }, line: { type: "none" } });
    s.addText([{ text: "More collocation coverage removed the train/dense residual gap without restoring dynamic propagation. ", options: { bold: true } },
      { text: "Sparse collocation alone is not the root cause — no stage-2 variant beat B1 (pre-registered CASE D).", options: {} }],
      { x: 0.85, y: 5.88, w: 11.7, h: 0.9, fontSize: 15, color: C.background1, valign: "middle", margin: 0, isTextBox: true });
    s.addNotes("Four single-change arms from the identical 320k state: L-BFGS on the fixed set, L-BFGS with fresh points every 50 closures, L-BFGS on a 4× larger pool, and full-batch Adam on the fixed set. The gap between dense-grid and training residual is a coverage × optimizer interaction (35–76× only with fixed points AND L-BFGS). Yet the 2,560-point pool has almost no gap and still stalls or diverges, and Adam on the fixed set also stalls. Resampled L-BFGS advanced in 2/3 seeds and matched B1 in only one. Value: it stopped us from optimizing collocation density, which would have been the wrong direction.");
  }

  // ============================================================================ 12. WHERE ARE WE
  pres.addSection({ title: "Status and next steps" });
  {
    const s = content("Where we are: no optimized full-field PINN yet", "Status and next steps", "B2-E01 seed replication (B2-E01S_SEED_RESULTS.csv); B2-E02_RESULTS.csv; references/batch1/tables/phaseZ4_20K_checkpoints.csv");
    const ladder = [
      ["Benchmark established", "ACHIEVED"], ["Reproducibility established", "ACHIEVED"], ["Failure mechanisms narrowed (refinement ongoing)", "ACHIEVED"],
      ["Candidate optimization mechanisms identified", "PARTIAL"], ["Reproducible full-field improvement", "NOTYET"], ["Compute-efficient optimized PINN", "NOTYET"], ["Railway validation", "NOTYET"],
    ];
    ladder.forEach((l, i) => {
      const y = 1.4 + i * 0.68;
      s.addShape(pres.shapes.OVAL, { x: 0.6, y: y + 0.08, w: 0.48, h: 0.48, fill: { color: schemeOf(l[1]) }, line: { type: "none" } });
      s.addText(String(i + 1), { x: 0.6, y: y + 0.08, w: 0.48, h: 0.48, fontSize: 14, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
      s.addText(l[0], { x: 1.25, y, w: 3.85, h: 0.64, fontSize: 14, color: C.text1, bold: i === 4, valign: "middle", margin: 0, isTextBox: true });
      tag(s, l[1] === "NOTYET" && i === 6 ? "NOTYET" : l[1], 5.15, y + 0.17, 1.75);
    });
    const b1 = D.e01s.filter((r) => r.arm === "B1");
    const ml = D.e02.filter((r) => r.arm === "LBFGS");
    const best = D.batch1_best;
    card(s, 7.3, 1.4, 5.45, 2.15);
    s.addText("Full-field reference (B1, 640k evaluations)", { x: 7.5, y: 1.5, w: 5.1, h: 0.4, fontSize: 14, bold: true, color: C.text2, margin: 0, isTextBox: true });
    s.addText([{ text: show("B1 L2 s1234", `L2 ${fx(D.e01.B1.L2_exact, 3)}`, "e01"), options: { breakLine: true } }, { text: show("B1 P s1234", `${fx(D.e01.B1.persistence_cycles, 2)} cycles`, "e01") }],
      { x: 7.5, y: 1.9, w: 2.6, h: 1.2, fontSize: 26, bold: true, color: C.accent1, fontFace: "Cambria", margin: 0, isTextBox: true, valign: "middle" });
    s.addText(show("B1 3 seeds", `seed 1234; across 3 seeds L2 ${rng(b1.map((r) => r.L2_exact), 3)}, ${rng(b1.map((r) => r.persistence_cycles), 2)} of ${fx(D.analytic.cycles, 2)} cycles`, "e01s"),
      { x: 10.15, y: 1.95, w: 2.5, h: 1.2, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "middle" });
    card(s, 7.3, 3.7, 5.45, 2.15);
    s.addText("Best modal diagnostic (Adam → L-BFGS)", { x: 7.5, y: 3.8, w: 5.1, h: 0.4, fontSize: 14, bold: true, color: C.text2, margin: 0, isTextBox: true });
    s.addText([{ text: show("modal L2 range", `L2 ${rng(ml.map((r) => r.L2_exact), 3)}`, "e02"), options: { breakLine: true } }, { text: show("modal cycles", `${fx(ml[0].persistence_cycles, 2)} cycles`, "e02") }],
      { x: 7.5, y: 4.2, w: 3.1, h: 1.2, fontSize: 26, bold: true, color: C.accent3, fontFace: "Cambria", margin: 0, isTextBox: true, valign: "middle" });
    s.addText("3/3 seeds · exact spatial mode supplied — not the optimized PINN", { x: 10.65, y: 4.25, w: 2.0, h: 1.2, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "middle" });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.3, y: 6.0, w: 5.45, h: 0.75, rectRadius: 0.08, fill: { color: C.text2 }, line: { type: "none" } });
    s.addText("Closing this gap on the coupled full field is the central open problem.", { x: 7.5, y: 6.0, w: 5.1, h: 0.75, fontSize: 14, bold: true, color: C.background1, valign: "middle", margin: 0, isTextBox: true });
    s.addText(show("batch1 best", `Batch-1 best full-field run: L2 ${fx(best.L2_exact, 3)}, ${fx(best.persistence_cycles, 2)} cycles at 4× compute (${fx(best.pde_evaluations / 1e6, 2)}M evaluations, 1 seed), still collapsing at ${fx(best.collapse_time_s, 3)} s.`, "phaseZ4_20K"),
      { x: 0.6, y: 6.25, w: 6.4, h: 0.6, fontSize: 12, italic: true, color: MUTED, margin: 0, isTextBox: true });
    s.addNotes("No completion percentage is given because there is no defensible way to compute one. The validated full-field reference is still B1 at matched compute. The modal diagnostic shows what correct temporal learning looks like when the spatial mode is supplied; achieving comparable dynamics while learning the spatial field, without extra compute, is the open problem.");
  }

  // ============================================================================ 13. RAILWAY
  {
    const s = content("Path to the railway application", "Status and next steps", "planned validation path (not executed); literature context: docs/LITERATURE_MATRIX.md (H2, H3: Kapoor et al., beams on Winkler foundation, moving loads)");
    statusPill(s, "FUTURE");
    const steps = [
      ["Beam-vibration benchmark", "current work (in progress)", "BENCH"],
      ["Full-field optimized PINN", "not yet achieved", "NOTYET"],
      ["Railway vibration / structural dynamics", "future", "FUTURE"],
      ["Wheelset / axle / bearing vibration", "future", "FUTURE"],
      ["Physics-informed condition monitoring", "future", "FUTURE"],
    ];
    for (let i = 0; i < steps.length; i++) {
      const y = 1.4 + i * 1.07;
      const fill = i === 0 ? C.accent1 : C.background2;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 5.6, h: 0.8, rectRadius: 0.08, fill: { color: fill }, line: i === 0 ? { type: "none" } : { color: C.accent5, width: 1, dashType: "dash" } });
      s.addText([{ text: steps[i][0], options: { bold: true, breakLine: true, fontSize: 15 } }, { text: steps[i][1], options: { fontSize: 12 } }],
        { x: 0.8, y, w: 5.2, h: 0.8, color: i === 0 ? C.background1 : C.text1, valign: "middle", margin: 0, isTextBox: true });
      if (i < steps.length - 1) s.addShape(pres.shapes.DOWN_ARROW, { x: 3.25, y: y + 0.82, w: 0.3, h: 0.22, fill: { color: C.accent5 }, line: { type: "none" } });
    }
    await iconDot(s, fa.FaTrain, 6.8, 1.45, 0.8, C.accent1);
    s.addText("Why start from a beam benchmark", { x: 7.8, y: 1.55, w: 4.9, h: 0.6, fontSize: 18, bold: true, color: C.text2, margin: 0, isTextBox: true, fontFace: "Cambria" });
    s.addText([
      { text: "Exact solution available, so dynamic fidelity (frequency, damping, persistence) is measurable", options: { bullet: true, breakLine: true } },
      { text: "Euler–Bernoulli beam models are standard building blocks of track and bridge dynamics in the PINN literature (e.g. beams on Winkler foundations, moving loads)", options: { bullet: true, breakLine: true } },
      { text: "A method must first preserve dynamics on controlled physics before it is trusted on noisy railway measurements", options: { bullet: true } },
    ], { x: 6.8, y: 2.45, w: 5.95, h: 2.9, fontSize: 14, color: C.text1, valign: "top", margin: 0, isTextBox: true, paraSpaceAfter: 10 });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.8, y: 5.55, w: 5.95, h: 1.15, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.accent5, width: 1 } });
    s.addText("No railway experiment, dataset or deployment has been performed yet. Railway work starts only after a validated full-field method exists.",
      { x: 7.0, y: 5.6, w: 5.55, h: 1.05, fontSize: 14, bold: true, color: C.text2, valign: "middle", margin: 0, isTextBox: true });
    s.addNotes("Railway is the intended first vertical application. The beam benchmark is the controlled step: exact physics, measurable fidelity. Moving to wheelset/axle/bearing vibration and condition monitoring requires (1) a validated full-field method, (2) real railway data and (3) more complex dynamics. None of these steps has been executed.");
  }

  // ============================================================================ 14. TAKEAWAY
  {
    const s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "Status and next steps" });
    s.addText("Takeaways", { placeholder: "title" });
    s.addText("Source: B2-E01–B2-E03 reports and CSVs; frozen baseline commit cdd87b5 · ROOT d31864a unchanged", { placeholder: "source" });
    const msgs = [
      ["1", "We have not found the final optimized PINN yet."],
      ["2", "We have ruled out several obvious explanations — precision, derivative order alone, collocation density, the L-BFGS switch alone — and identified a strong gap between temporal-only and coupled space–time learning."],
      ["3", "The next breakthrough must improve the full-field solution, not merely the diagnostic modal problem."],
    ];
    msgs.forEach((m, i) => {
      const y = 1.45 + i * 1.3;
      s.addShape(pres.shapes.OVAL, { x: 0.7, y: y + 0.08, w: 0.7, h: 0.7, fill: { color: C.accent1 }, line: { type: "none" } });
      s.addText(m[0], { x: 0.7, y: y + 0.08, w: 0.7, h: 0.7, fontSize: 24, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, fontFace: "Cambria" });
      s.addText(m[1], { x: 1.65, y, w: 11.0, h: 1.0, fontSize: i === 1 ? 19 : 22, bold: true, color: C.background1, valign: "middle", margin: 0, isTextBox: true, fontFace: "Cambria" });
    });
    const goals = ["Lower error", "Longer physical fidelity", "Less compute", "Transfer to railway physics"];
    s.addText("Research objective", { x: 0.7, y: 5.4, w: 4, h: 0.35, fontSize: 14, color: "C9DCE6", margin: 0, isTextBox: true });
    goals.forEach((g, i) => {
      const x = 0.7 + i * 3.0;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 5.85, w: 2.65, h: 0.75, rectRadius: 0.08, fill: { color: H.dk2 }, line: { color: "7FC3DC", width: 1.25 } });
      s.addText(g, { x, y: 5.85, w: 2.65, h: 0.75, fontSize: 15, bold: true, color: C.background1, align: "center", valign: "middle", margin: 2, isTextBox: true });
      if (i < 3) s.addText("+", { x: x + 2.65, y: 5.85, w: 0.35, h: 0.75, fontSize: 20, bold: true, color: "7FC3DC", align: "center", valign: "middle", margin: 0, isTextBox: true });
    });
    s.addNotes("Central message: we progressed from reproducing a difficult PINN benchmark to systematically identifying why it fails; tested multiple formulation and optimization strategies; demonstrated substantial gains in an isolated diagnostic problem; and narrowed the remaining problem to achieving those gains on the coupled full-field physics without increasing compute.");
  }

  await pres.writeFile({ fileName: OUT });
  const { applyTheme } = require(process.argv[2]);
  await applyTheme(OUT, THEME);
  fs.writeFileSync(path.join(HERE, "displayed_numbers.json"), JSON.stringify(SHOWN, null, 1));
  console.log("wrote", OUT, "slides:", 14, "displayed numbers:", SHOWN.length);
})().catch((e) => { console.error(e); process.exit(1); });
