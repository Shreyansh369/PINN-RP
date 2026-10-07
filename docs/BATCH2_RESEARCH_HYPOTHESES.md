# Batch-2 research hypotheses (PRE-REGISTERED 2026-10-07, before any Batch-2 training)

These hypotheses were written **before** any Batch-2 data exist. Do not edit a hypothesis, prediction or
interpretation rule after its experiment starts. Corrections are appended as dated amendments with reasons.

Metrics are the frozen Batch-1 definitions (`physref/persistence.py`, `beampinn/evaluation/metrics.py`):

| Symbol | Definition |
|---|---|
| **P** | persistence, in cycles; the full window is 20.58 |
| **t_c** | collapse time |
| **L2e** | L2_exact |
| **R_pde** | PDE_residual_rel |
| **E** | PDE (collocation-residual) evaluations used in training |
| **W** | training wall-clock (`train_seconds`) on the declared hardware |

## H0 — Central program hypothesis

> There is no single universally optimal PINN training configuration. The most compute-efficient strategy
> depends on measurable structure of the problem: derivative order, conditioning, spectral content, temporal
> propagation length (N_c), damping, constraints and budget.

**Operationalisation.** This hypothesis is decided only at Phase B2-13, by gap D2 (`docs/NOVELTY_GAP.md`).

**Support:**
- the Pareto-optimal strategy differs between ≥ 2 PDE tiers, **and**
- pre-registered invariant-based rules choose a Pareto-optimal (or within-seed-noise) strategy on held-out
  tiers, including the selection cost.

**Falsification:**
- one strategy is Pareto-optimal on all tiers, **or**
- the rules do no better than the fixed best-practice recipe (L7) or than random selection at equal total
  compute.

## H1 — Physics analysis is necessary but not sufficient (analyzer validity)

> Pre-training invariants (N_c, ζ, max derivative order, coefficient spread, N* scale, Fourier coverage)
> correlate with the observed failure mode. No single invariant suffices.

**Existing evidence (Batch 1).**
- The N* scale (−8.4e3 → −0.5) predicted the E → X conditioning improvement.
- Fourier coverage (0/100 features at ω_d for B1) did **not** predict failure to reproduce ω: B1 got ω to −0.1 %.

**Test.** Evaluated across the E01–E05 results and the Tier-2/3 PDEs.

**Falsified if** no invariant ranks the strategies better than chance (Spearman ρ with Pareto rank,
95 % bootstrap CI including 0) across ≥ 3 problems.

## H2 — The collapse is dominated by temporal propagation

> The Batch-1 collapse front is a temporal-propagation phenomenon. It is governed by how far correct
> dynamics must propagate from the exactly-enforced IC (N_c ≈ 20.6 cycles). The 4th-order spatial operator
> is not the dominant cause.

**Experiment:** E01, modal arm vs B1 at matched E = 6.4e5.
- The modal arm keeps B1's temporal representation, time factor, optimizer, schedule, mini-batch,
  precision and seed.
- It removes x entirely: the exact mode shape is used.

**Expected mechanism.** If propagation dominates, a pure 1-D temporal problem over the same 20.6 cycles shows
a collapse front at a comparable t_c per evaluation.

| Outcome (seed 1234 first; then 3 seeds) | Pre-registered interpretation |
|---|---|
| P_modal ≤ 1.5 × P_B1 (B1 at 5k: 3.2 cycles) | **H2 supported.** Temporal propagation dominates; prioritise E03 (temporal decomposition) and E04 (R3) |
| P_modal = full window (20.6) and L2e_modal < 0.1 | **H2 falsified.** Removing the spatial operator removes the collapse; prioritise E01-mixed and formulation work |
| in between | partial. Report the ratio; run the stage-2 seeds before concluding |

## H3 — Lowering derivative order improves persistence per unit compute

> The mixed formulation (max order 2) reaches equal or higher P than strong B1 at matched E, at lower W.

**Experiment:** E01, mixed arm vs B1 at E = 6.4e5.

**Measured cost prior (B2-PROF-001).** One residual+backward evaluation costs 70 ms (mixed) vs 154 ms (B1).
At matched E the mixed arm should therefore need ≈ 0.45–0.6 × W.

**Supported if:**
- P_mixed ≥ P_B1 − (seed spread); **and**
- W_mixed ≤ 0.75 W_B1; **and**
- IC/BC errors stay at B1 levels (exact by construction for u).

**Falsified if** P_mixed < P_B1 by more than the seed spread at matched E.

**Then** run the wall-clock-matched comparison: mixed at E' = E × W_B1/W_mixed. If P_mixed(E') > P_B1(E),
mixed is better per unit wall-clock even though it is worse per evaluation. Both results are reported.

**Note (no novelty claim).** A-PINN EB (2026) is the closest prior work. Our contribution would be the
matched-compute measurement on a long-window damped problem, not the formulation.

## H4 — Precision is not the cause of the collapse in the Adam regime

> Unlike the L-BFGS premature-convergence mechanism of Xu et al. 2025 (FP32 → FP64 removes failure modes),
> the Batch-1 Adam-trained collapse front is not a precision artefact.

**Experiment:** E02. B1 in float64 at 5k steps vs B1 float32 (Batch-1 Z4: P = 3.2, L2e = 0.526).

**Supported if** |P_fp64 − P_fp32| ≤ seed spread.

**Falsified if** P_fp64 ≥ 2 × P_fp32. In that case precision becomes a first-class axis, and every later arm
is re-run in FP64 before conclusions are drawn.

## H5 — Exact window transfer converts the front into bounded per-window error

> Temporal decomposition with an exact IC hand-over (u, u_t) lets each window cover its N_c/K cycles. Total
> error then grows with K, roughly linearly in phase, instead of collapsing.

**Experiment:** E03, K ∈ {2, 4, 8} at total E matched to B1-20k (2.56e6).
- The B1 recipe is used per window.
- Each window is initialised by transfer from the previous one.

**Supported if** P = full window for some K with L2e < L2e(B1-20k) = 0.263 at ≤ equal total E.

**Falsified if** every K collapses inside a window, or accumulated error exceeds B1's.

**Comparator:** AT-PINN-HC (G8). It must be re-implemented or cited with its published numbers before any
claim.

## H6 — R3 tracks the propagation front where RAD did not

> R3's retained population concentrates at the collapse front, so t_c advances faster per evaluation than
> with uniform redraw (B1) and than with RAD (Batch-1 Z1).

**Experiment:** E04 at E = 6.4e5.

**Supported if:**
- P_R3 > P_B1 + seed spread; **and**
- the retained-point time distribution peaks within ±1 period of the measured front (mechanism check).

**Falsified otherwise.** A gain without the mechanism signature is reported as unexplained.

## H7 — Second-order refinement improves accuracy per unit compute only after the front covers the window

> Adam → L-BFGS (and NNCG/SOAP if affordable) improves L2e and R_pde when started from a state that already
> persists. Started from a collapsed state, it does not extend P.

**Experiment:** E05b. It is run only after an arm passes persistence, or on B1-20k as a control.

**Falsified if** L-BFGS from the B1-20k state extends P by more than 2 cycles. That would contradict the
hypothesis, which is informative either way.

## H8 — Lower-cost formulations shift the accuracy–compute Pareto front

> At least one formulation or temporal strategy dominates B1 on the (L2e, W) and (P, W) Pareto fronts.

**Experiment:** E09. It aggregates E01–E05 at 3 seeds.

**Falsified if** B1 is non-dominated against all arms on both fronts.

## Global rules

1. **One change per arm** relative to its declared parent. Config diffs are printed and stored.
2. **Matched PDE evaluations are the primary comparison.** Matched wall-clock is secondary, measured on one
   machine, one thread, with no concurrent jobs, or with concurrency declared.
3. **Seeds.**
   - Stage 1 uses seed 1234.
   - Stage 2 (1235, 1236) runs for any arm whose stage-1 result would change a decision.
   - "Seed spread" is the max − min over the 3 seeds of the arm in question, or of B1 where B1 is the
     comparator.
4. **Success requires every one of:**
   - correct frequency;
   - correct amplitude;
   - plausible damping;
   - correct velocity dynamics;
   - persistence over the full window;
   - low R_pde;
   - low IC/BC error;
   - acceptable cost.

   A falling L2 alone never counts as success (prompt §19).
5. **No metric, threshold or interpretation rule changes after results are seen.**
