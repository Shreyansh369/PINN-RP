# Research-story presentation: summary

**Deck:** `final_research_story_presentation.pptx`. It has 10 main slides and 3 appendix slides, 16:9, with speaker notes on every slide. The main talk runs about 8–12 minutes.

**Built by:** `scripts/presentation/build_story_deck.js`, with data from `scripts/presentation/extract_deck_data.py`.

**QA:** `scripts/presentation/check_story_deck.py`, output in `qa_check_output.txt`. The result is **ALL CHECKS PASSED** (70 checks). See `QA_REPORT.md`.

**Figures:** see `FIGURES.md`.

This deck **replaces** the earlier Batch-1 deck. That deck is in ROOT and is untouched; this one does not reuse its structure. No model was trained for it. All evidence comes from frozen Batch-1 artifacts:
- `references/batch1/*`, byte-identical to ROOT @ `d31864a`;
- inference on frozen, git-tracked Batch-1 checkpoints, read from ROOT read-only.

Batch-2 results that already exist in this repository (B2-E01, single seed, provisional) are **not** used as evidence. They are only mentioned in the slide-9 status line.

## Narrative

Physical problem → can a PINN solve vibration physics? → the 2025 Fourier + NTK paper → independent reproduction → unexpected failure in our controlled implementation → controlled diagnosis → conditioning identified → representation verified → optimization/propagation limits quantified → no validated optimized PINN yet → Batch 2: compute-aware, physics-aware solver research → Physics Reference Layer → railway first → other verticals later.

## Slide by slide

| # | Title | One message | Main visual |
|---|---|---|---|
| 1 | Can We Make Physics-Informed AI Actually Solve Physics Efficiently? | The ambition: a verified physical reference between the equations and the application. | Pipeline: physical system → equations → PINN → **verified physical reference** → AI application. Footer: "First testbed: Euler–Bernoulli beam vibration". |
| 2 | Engineering needs fast, trustworthy physics | Simulation is trustworthy but costly; pure ML is fast but unverified. Can a PINN reach both? | Cost-vs-trust map with a target zone. Question: high accuracy + low cost + physical verification. |
| 3 | A 2025 paper: a PINN for beam vibration | Beam → PDE → PINN. The paper's method is Fourier features + NTK. 4.64 × 10⁻⁴ is **the paper's** number. | Beam schematic, PDE card `EI·u_xxxx + ρA·u_tt + b·u_t = 0`, **PUBLISHED BASELINE** card. |
| 4 | An experiment ladder: one question per step | The whole programme in 15 s: 8 rungs, one plain-language question each, 26 controlled runs. | Ladder: published → reproduction → D → E → X → Y → Z → budget study. |
| 5 | From a static field to eight cycles — then collapse | Exact vs C0 vs X2 vs Z4 vs Z4-20K: each stage pushes the failure later; none reaches the full window. | **Figure 1:** stacked mid-span traces with annotations and collapse markers. |
| 6 | Three different failures, found one at a time | (A) soft training → static attractor; (B) (t/T)² badly conditioned, tanh²(ω₁t) O(1); (C) more compute moves the collapse later but does not remove it. | Three panels; panel B is **Figure 3** (required network output). |
| 7 | Claim → evidence | Representation is not the limit; conditioning matters; compute matters; RAD did not fix it. **STATUS: NO VALIDATED OPTIMIZED PINN YET.** | Claim/evidence rows and a red status box. |
| 8 | More physics computation helps — but not enough yet | Persistence rises monotonically with PDE evaluations: best 8.1 of 20.6 cycles. Paper budget ≠ our budget. | **Figure 2:** cycles before collapse vs cumulative PDE evaluations, with the full-window target line. |
| 9 | Batch 2: From PINN Optimization to a General Physics Solver | Not one recipe for every PDE, but a strategy chosen from the physics and the compute budget. Components have prior art. The strategy-selection idea is a **research hypothesis**. | **Figure 5:** Batch-2 pipeline. NOT/BUT cards, key quote, prior-art note, hypothesis box. |
| 10 | From railway to vertical AI | Final message. Physics Reference Layer hub; railway highlighted first. CURRENT / NEXT / FUTURE / LATER. Railway path is **future work, no results**. | Hub diagram, stage timeline, railway path. |
| A | Technical setup: PDE and PINN architecture | Backup: exact definitions. | Two tables. |
| B | Detailed experiment matrix (26 runs, frozen) | Backup: every run, its change, L2 and outcome. | Two tables. |
| C | Literature frontier: what already exists | Backup: prior art per component (Batch-2 literature matrix). The candidate gap is a hypothesis. | 3 × 3 card grid and a gap box. |

Figure 4 (the method/experiment ladder) is slide 4.

## Every number used, with its source

`B1/` = `references/batch1/` (byte-identical copies of ROOT `results_optimization/` tables and reports, ROOT `optimization_leaderboard.csv` and `paper_benchmark_registry.csv`).

| Slide | Value | Meaning | Source |
|---|---|---|---|
| 3, 8 | 4.64 × 10⁻⁴ | relative L2 **reported by the paper** (FE-D-M1) | `B1/paper_benchmark_registry.csv` (paper_error); Söyleyici & Ünver, EAAI 141 (2025) 109804 |
| 3, 8, A | 20.6 Hz / 20.6 cycles | Mode-1 damped frequency × 1 s window | exact reference (`beampinn`), `deck_data.json` f_d = 20.59 |
| 4, B | 26 | controlled training runs | `B1/optimization_leaderboard.csv` (26 rows) |
| 5, B | L2 3.26 / 0.909 / 0.526 / 0.263 | C0, X2, Z4, Z4-20K relative L2 (exact) | `B1/optimization_leaderboard.csv` |
| 5, 7, 8 | ≈ 3.2 / ≈ 8.1 cycles | Z4 (5k) and Z4-20K persistence | `B1/tables/phaseZ4_20K_checkpoints.csv` |
| 5, 6 | collapse markers 0.153 s / 0.393 s | Z4, Z4-20K collapse time | recomputed from frozen checkpoints; equal to `phaseZ4_20K_checkpoints.csv` (0.15325 / 0.39325 s) |
| 6, 7 | ≈ 8.4 × 10³ (8 369) | required \|N\| for (t/T)² | `deck_data.json` (exact solution); equals `profiles/phaseE_ansatz_conditioning.txt` (−8368.7) and `B1/reports/PHASE_E_SCREEN.md` |
| 6, 7 | O(1) (max 1.9) | required \|N\| for tanh²(ω₁t) | same; frozen range [−1.9, −0.16] |
| 7 | 129.25 / 129.37 / 129.32 rad/s | X3, X4 supervised fitted ω; exact ω_d | `B1/tables/phaseX_oscillation.csv` |
| 7 | 1.2 → 3.1 → 3.2 → 8.1 cycles | Y1 5k, Y1 20k, Z4 5k, Z4 20k | `B1/tables/phaseY1_20K_checkpoints.csv`, `phaseZ4_20K_checkpoints.csv` |
| 7 | 0.16M → 2.56M | cumulative PDE evaluations | same tables |
| 7 | 1.6 cycles | Z1 (Y1 + RAD) persistence | `B1/reports/PHASE_Y1_20K_BUDGET_DIAGNOSTIC.md` |
| 8 | 2.56M, L2 0.263, 0.393 s, 129.2 rad/s | Z4-20K | `B1/tables/phaseZ4_20K_checkpoints.csv` |
| 8 | 2.88 × 10⁷ | PDE evaluations in the paper's stated schedule (our reading; never run) | `B1/reports/STAGE01_CORRECTIONS.md`, `STAGE0_AUDIT.md` |
| 8 (chart) | 8 checkpoint points | persistence vs PDE evaluations | both checkpoint tables |
| A | c² = 43.73², γ = 7.08, L = 2.75 m, A₀ = 0.08 m | PDE and geometry | paper Eq. 49; `B1/reports/STAGE0_AUDIT.md` |
| A | 241,601 parameters; ω₁ = 129.37 rad/s; decay 3.54 s⁻¹ | model and reference | `B1/reports/PHASE_Z4_20K_BUDGET_DIAGNOSTIC.md`, `PHASE_X_DIAGNOSTIC.md`, `phaseY_oscillation.csv` |
| A | 4.386e-4 | L2 between paper-faithful and exact references | `B1/tables/stage01_reference_comparison.csv` |
| B | 26 L2 values (3 s.f.) and outcomes | every Batch-1 run | `B1/optimization_leaderboard.csv`; phase reports |
| notes | ~1e14–1e15 | C0 NTK weights | `B1/reports/PHASE_A1_CHECKPOINT.md` |
| notes | 3.1 vs 3.2 cycles at 0.64M | batch size: same state at equal compute | `B1/reports/PHASE_Y1_20K_BUDGET_DIAGNOSTIC.md` |
| C | 81 works / 65 search-confirmed | literature audit size | `docs/LITERATURE_MATRIX.md` |

**Rounding rule:**
- relative L2 to 3 significant figures, everywhere;
- cycles to 1 decimal;
- collapse time to 3 decimals;
- frequencies as in the frozen tables.

The brief suggested "~4 cycles" for Z4. The frozen value is **3.2**, and that is what the deck shows.

## Final scientific claims

**Established:**
1. Our controlled implementation of the published Fourier + NTK method did not reproduce the published result. It settled on a static/incorrect attractor. This is an observed optimization failure; NTK causation is not claimed.
2. Hard constraints fixed IC/BC satisfaction, but the original (t/T)² time factor was poorly conditioned. tanh²(ω₁t) reduced the required network output from about 8.4 × 10³ to O(1).
3. The network can represent the 20.6 Hz dynamics (supervised diagnostics).
4. Training strategy and physics-computation budget measurably extend persistence, from 1.2 to 8.1 cycles.
5. RAD did not resolve the collapse.

**Not established:**
- no validated optimized PINN;
- no full-window reproduction;
- no superiority over the paper;
- no claim that the paper is wrong;
- no Mode-2 result;
- no railway result;
- no novelty for any individual component.

**Batch-2 position:** problem-dependent, compute-aware strategy selection with verification is a **research hypothesis under investigation**. It is not a contribution.

## Limitations (as stated in the notes)

- Single seed (1234), except Z2 at 5k steps.
- CPU, float32.
- Budgets far below the paper's schedule.
- The persistence threshold (50 %) is fixed but arbitrary.
- The literature audit is abstract-level; full texts must be read before any publication claim.

## Placeholders to fill

Slide 1: `[Author name] · [Institution]`.
