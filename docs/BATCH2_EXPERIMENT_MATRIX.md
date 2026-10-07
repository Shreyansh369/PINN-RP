# Batch-2 candidate experiment matrix (PRE-REGISTERED; nothing approved, nothing run)

The canonical benchmark is FE-D-M1, with the Batch-1 physics, references, grids and metrics unchanged.

Default controls, unless an experiment lists its own:
- seed 1234, float32, 1 CPU thread;
- the B1 sampler (640 points redrawn per epoch, mini-batch 128);
- Adam, lr = 1e-3·0.9^(step/1000);
- evaluation on the 201×2001 grid against both references, plus the frozen persistence metric;
- no concurrent jobs during timed runs, or concurrency declared.

Parent baselines are hash-locked:
- **B0** = `configs/frozen/baseline_fourier_ntk.yaml`
- **B1** = `configs/frozen/batch1_hard_tanh2.yaml`

Every arm changes **one** thing relative to its parent.

Cost estimates use measured per-step costs. B2-PROF-001 (this machine: Xeon @ 2.8 GHz) gives the
residual+backward time; to this add the optimizer step and sampling. Batch-1 measured 611 s for B1 at 5k
steps on a 2.1 GHz Xeon, about 122 ms/step **including** optimizer and sampling. Every estimate is an
estimate, and actuals are recorded.

| ID | Phase | Axis | Arms (one change each) | Parent | Budget | Seeds (stage 1 / 2) | Est. CPU (stage 1) | Hypothesis | Priority |
|---|---|---|---|---|---|---|---|---|---|
| **E01** | B2-5 | formulation / mechanism | B1 (re-run, reproducibility), **mixed**, **modal** | B1 | 5k steps = 6.4e5 E | 1234 / 1235, 1236 | ≈ 15 + 8 + 4 min training + ≈ 3×5 min evaluation ≈ **45 min** | H2, H3 | **1 (first)** |
| E02 | B2-8 | precision | B1-fp64 | B1 | 5k steps | 1234 / 1235, 1236 | ≈ 20–40 min (measure) | H4 | 2 |
| E03 | B2-6 | temporal decomposition | K = 2, 4, 8 windows, exact (u, u_t) hand-over, transfer initialisation; overlap 0 | B1 | total 2.56e6 E (= B1-20k) | 1234 / 2 more | ≈ 45 min per K | H5 | 3 |
| E04 | B2-7 | sampling | R3 (non-causal), same point count 640 | B1 | 6.4e5 E | 1234 / 2 more | ≈ 15 min | H6 | 4 |
| E05a | B2-4 | conditioning | (i) residual nondimensionalised by ω₁²A0; (ii) network input τ = ω₁t (modal scaling) | B1 | 6.4e5 E | 1234 | ≈ 2 × 15 min | H1 | 5 |
| E05b | B2-8 | optimizer | Adam → L-BFGS (FP64 if E02 says so); then MultiAdam / SOAP if affordable | B1-20k state (Batch-1 checkpoint read-only via `git show`) and the best E01–E04 arm | 500 L-BFGS iterations | 1234 | ≈ 30 min | H7 | 6 |
| E06 | B2-7 | loss balancing | ReLoBRaLo, ConFIG, LRA on the **mixed** arm (the only multi-term arm); B0 already contains NTK | mixed | 6.4e5 E | 1234 | ≈ 3 × 8 min | (loss-balancing question, §15.7) | 7 (only if mixed is promising) |
| E07 | B2-8 | representation | MLP (no Fourier), SIREN, separable (SPINN-like) at matched parameters (±5 %) | B1 | 6.4e5 E | 1234 | ≈ 3 × 15 min | (§15.8) | 8 |
| E08 | B2-8 | derivative computation | reverse vs forward profiling at batch 32/128/512 and depth 4/6; memory per case in subprocesses | — | profiling only | — | ≈ 10 min | (§15.9) | 9 (cheap; can run any time) |
| E09 | B2-9 | compute-aware comparison | aggregate E01–E08 at 3 seeds; Pareto fronts | — | — | — | analysis | H8 | after E01–E08 |
| E10 | B2-11/12 | combination | only components with independent benefit; full ablation | winners | per ablation | 3 | TBD | — | after E09 |
| E11 | B2-13 | cross-PDE | heat (Tier 2), wave with N_c = 20.6 and Klein–Gordon (Tier 3), Allen–Cahn or Burgers (Tier 4) | per-tier B0/B1 analogues | TBD | 3 | TBD | H0, H1 | later |
| E12 | B2-14 | railway transfer | Timoshenko / moving-load beam (Kapoor et al. setups) | validated method | TBD | 3 | TBD | — | later |

---

## E01 — Failure-mechanism attribution by equivalent reformulation (FIRST EXPERIMENT)

Spec: `configs/batch2/B2-E01_mechanism_attribution.yaml`. Runner:
`experiments/run_batch2.py --spec … --arm {B1,mixed,modal}`. Arm trainer: `src/physref/arms.py`.

- **Hypotheses:** H2 (temporal propagation dominates) and H3 (lower derivative order helps per compute).
- **Independent variable:** the formulation of the same physics:
  - strong (x,t)→u;
  - mixed (x,t)→(u, v = u_xx);
  - modal t→q(t) with the exact mode-1 shape.
- **Dependent variables:**
  - primary: P, plus velocity persistence and t_c.
  - secondary:
    - L2e and L2p;
    - R_pde (full-field residual of the displacement u, for every arm);
    - fitted ω and decay;
    - amplitude and max|u_t| ratios;
    - IC/BC errors.
  - cost: W, peak RSS, parameters, inference µs/point.
  - mechanism: P at the 1k-step snapshots (front-speed curve dt_c/dE).
- **Controlled variables** (identical in all arms):
  - seed, sampler, collocation count, mini-batch, optimizer, schedule, precision, threads;
  - temporal representation (Fourier σ_t = (10, 1) on standardised t; 6×200 tanh; m = 100);
  - time factor tanh²(ω₁t) with the same ω₁;
  - evaluation grid and metrics.

  Parameter counts: B1 241,601; modal 241,601; mixed 242,002 (+0.2 %, the second head).
- **Expected mechanism:**
  - If the collapse is propagation-limited, the modal arm shows a front at a similar t_c (H2).
  - If the 4th-order operator conditioning (D1 theory) matters, the mixed arm advances the front faster per
    evaluation (H3).
- **Failure criteria:**
  - **Reproducibility gate for the B1 re-run:**
    - The run key must equal `…f78aa7f1da`. This is checked at dry-run and already passes.
    - Metrics must reproduce Batch-1 Z4 (L2e 0.526, P 3.2 cycles, R_pde 0.110) within |ΔL2e| ≤ 5 % relative
      and |ΔP| ≤ 0.5 cycles.

    Bit-identity is **not** required. The CPU differs (Xeon 2.8 GHz vs 2.1 GHz), and so does the torch wheel
    (`2.14.0+cu130` run on CPU vs the Batch-1 CPU build); float32 kernels can round differently.

    Outside tolerance: **STOP** and report as an environment/determinism finding before interpreting any arm.
    Either way, every Batch-2 comparison uses the **re-run** B1, never the Batch-1 numbers.
  - Any arm diverges (non-finite loss): report and exclude it from the conclusions. No re-tuning.
- **Computational budget:** 5,000 steps × 128 = 6.4e5 PDE evaluations per arm.
  - Stage 1 is one seed, ≈ 45 min total CPU.
  - Stage 2 is two more seeds for all three arms, ≈ 1.5 h.
  - Arms may run concurrently, 1 thread each, with concurrency declared. Timing comparisons then use the
    per-step medians measured in the same concurrent setting.
- **Interpretation rule:** the table in `docs/BATCH2_RESEARCH_HYPOTHESES.md` H2/H3, applied as written.
- **Outputs:**
  - `results_batch2/runs/B2-E01-<arm>-s<seed>/` (record.json, history, metrics.json, snapshots);
  - report `results_batch2/reports/B2-E01_REPORT.md`;
  - figures under `results_batch2/figures/`.

## E02 — Precision (FP64)

- **Hypothesis:** H4.
- **Independent variable:** `precision` float32 → float64. Nothing else changes.
- **Dependent variables:** as E01, plus per-step W (FP64 cost).
- **Controlled variables:** everything else in B1. Initialisation, Fourier draws and sampling come from the
  same seeded generators; values differ only by dtype.
- **Mechanism:** reduced cancellation in u_xxxx (4 nested derivatives) and in late-time small residuals.
- **Failure criterion:** none beyond divergence.
- **Budget:** 5k steps; W expected ≈ 1.5–3× float32 on CPU (to be measured).
- **Interpretation:** H4 table.

## E03 — Temporal domain decomposition

- **Hypothesis:** H5.
- **Independent variable:** number of windows K (2, 4, 8); overlap 0; hard hand-over of (u, u_t).
- **Dependent variables:**
  - P over the full window;
  - L2e per window and overall;
  - interface jump of (u, u_t) (should be 0 by construction for a hard hand-over);
  - error accumulation vs k;
  - total W including the hand-over overhead.
- **Controlled variables:** total E matched to B1-20k (2.56e6), split equally over windows; B1 recipe per
  window; per-window time normalisation inside the network (AT-PINN uses it, so it is declared, not novel).
- **Mechanism:** each window covers N_c/K cycles, below the ≈ 3–8 cycles B1 reaches at matched E.
- **Failure criterion:** collapse inside any window.
- **Budget:** about 45 min per K.
- **Interpretation:** H5. Comparison with AT-PINN-HC is required before any claim.

**Implementation status:** plan generation and interface residuals are implemented and tested
(`physref/temporal.py`). The windowed trainer will be written only after E01 approval decides whether E03 is
prioritised.

## E04 — R3 sampling

- **Hypothesis:** H6.
- **Independent variable:** sampler uniform-redraw → R3 (retain |r| > mean |r|, resample the rest).
  Population 640; update every epoch.
- **Dependent variables:** P, t_c curve, and retained-point time histogram vs front position (mechanism
  check); all E01 metrics.
- **Controlled variables:** everything else in B1. Candidate residual evaluations = 640 per update, counted
  in E.
- **Failure criterion:** none beyond divergence.
- **Budget:** 6.4e5 E (+ R3 overhead counted).

**Implementation status:** the sampler is implemented and tested (`physref/sampling.py`); integration into
the trainer is pending approval.

## E05a — Conditioning transformations

- **Arm (i), residual scaling:** r → r/(ω₁²A0). Under Adam this is expected to be ≈ no-op; it is a control
  for the conditioning analyzer.
- **Arm (ii), network coordinates:** the network sees τ = ω₁t ∈ [0, 129] (and ξ = β₁x) instead of
  standardised t. This changes what the Fourier features cover; the change must be declared and measured
  (prompt §15.8).
- **Hypothesis:** H1 (analyzer validity).

## E05b — Optimizer refinement

- **Hypothesis:** H7.
- Adam → L-BFGS (strong-Wolfe line search, history 50).
- Started (a) from the Batch-1 Z4-20K `final.pt`, read via `git show d31864a:<path>` into `results_batch2/`
  and never symlinked; and (b) from the best E01–E04 arm.
- Cost counted in residual evaluations, including line-search evaluations.

## E06 — Loss balancing (conditional)

- Applies only to multi-term formulations. B1 has a single loss term.
- Run only if E01-mixed is promising.
- Question (§15.7): under which measurable condition does dynamic balancing improve accuracy per unit
  compute? Measured conditions:
  - the init gradient-norm ratio of the two mixed terms (`physref.conditioning.init_term_statistics`);
  - the gradient-alignment score (A4).

## E07 — Representation efficiency

- Arms at matched parameters: plain tanh MLP, SIREN, separable per-axis networks.
- Metrics: accuracy per parameter and per evaluation, inference µs/point, checkpoint bytes.
- No bandwidth increase is made specifically for Mode 2.

## E08 — Derivative computation (profiling only)

- Extends B2-PROF-001 across batch sizes and depths.
- Measures per-case peak memory in a subprocess per case.
- Includes the mixed formulation with forward-mode. JAX Taylor mode is listed as future work.

---

## Execution order and gates

1. **E01** (stage 1) → report → **STOP for approval.**
2. Depending on H2/H3:
   - H2 supported → E02 + E03 + E04;
   - H2 falsified → E02 + E06 + E07.
3. E02 always runs before any conclusion that attributes the collapse to non-precision causes.
4. E09 runs only after ≥ 3 seeds for every arm entering the comparison.
5. No combination experiment (E10) before E09 shows independent benefit for each component (prompt §25).
