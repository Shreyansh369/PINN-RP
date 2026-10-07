# B2-E02 — Temporal-optimisation screen (modal laboratory) and full-field transfer

## FINAL RESEARCH DECISION: **E — the improvement exists only in the modal diagnostic and fails full-field transfer.**

- **Modal laboratory.** Adam→L-BFGS earned the pre-registered label **DOMINANCE in 3/3 seeds**:
  full-window persistence (20.58 cycles), 3.5–7× lower L2, decay within 0.08–0.56 1/s of exact, 1.9–7.4× lower
  u-residual, at fewer PDE evaluations and 0.91–0.93× the wall-clock. Its peak RAM was 30 % higher.
- **Causal weighting** was worse in all seeds.
- **Full field.** With the same accounting on the frozen B1 configuration, Adam→L-BFGS is **worse than B1
  in all three seeds**: persistence −1.6 to −2.6 cycles, L2 ×1.28–1.54, u-residual ×1.2–2.4.

**Nothing here is a novel method:** Adam→L-BFGS (Rathore et al. 2024), causal weighting (Wang et al.
2024) and modal reduction are all established. The modal problem is a diagnostic laboratory, never a
competitor, because the exact spatial mode shape is supplied.

## Provenance

| Item | Value |
|---|---|
| Pre-registration | `docs/hypotheses/B2-E02.md`. Draft `450f7dd` (formulas, arms, accounting); final `8147008` (§9 from Phase A). Committed and pushed before any B2-E02 run (first run 21:28 UTC) |
| Phase A (seed replication of B2-E01) | `results_batch2/reports/B2-E01_SEED_REPLICATION.md` |
| Modal runs | 9 = 3 arms × seeds 1234–1236, launch commit `8147008`, `git_dirty=False`. One batch per seed: M0, LBFGS and CAUSAL concurrently, one thread each |
| Transfer runs | 6 = B1/LBFGS × 3 seeds, launch commit `bf4e905`, triggered by the pre-registered gate (`19621a8`). Batches: {1234, 1235} with 4 processes, {1236} with 2 |
| Frozen-control checks | M0 reproduces the Phase-A modal runs and transfer-B1 reproduces the Phase-A B1 runs **bit-exactly**, per seed (asserted in both evaluators) |
| Evaluators | `scripts/analyze_b2_e02.py`, `scripts/analyze_b2_e02_transfer.py`; frozen B2-E01 snapshot evaluator plus the u-residual at every snapshot |
| Outputs | `results_batch2/B2-E02_{RESULTS,SNAPSHOTS}.csv`, `B2-E02_CLASSIFICATION.json`, `B2-E02T_{RESULTS,SNAPSHOTS}.csv`, `B2-E02T_CLASSIFICATION.json`, `results_batch2/figures/B2-E02_*.png`, `B2-E02T_*.png` |
| Hardware | 4-vCPU Xeon @ 2.8 GHz, CPU only (peak VRAM 0), float32, torch 2.14.0 |

---

## 1. Phase A in brief (feeds the design)

Seeds 1234–1236 of all four B2-E01 arms:
- FP64 is not material in 3/3 seeds.
- Mixed is cheaper and never more accurate (B, A, B).
- Modal beats B1 in every seed (persistence ×1.19–2.42) but collapses at 0.30–0.37 s.
- B1's own persistence varies by 2.0 cycles across seeds.
- **In all three seeds the modal front reaches its best state at 384k–512k evaluations and then retreats**,
  i.e. a late-training regression.

Thresholds derived from pre-registered formulas:
- modal ΔP_thr = **3.06 cycles**;
- transfer ΔP_thr = **3.98 cycles**.

## 2. Candidate selection (from the pre-registration, §4)

| Arm | Change vs M0 | Sub-hypothesis tested |
|---|---|---|
| M0 (control) | none: frozen B2-E01 modal arm | — |
| LBFGS | Adam (bitwise = M0) for 320,000 evaluations, then full-batch L-BFGS (strong Wolfe, PyTorch default tolerances, history 50) on a fixed 640-point set | H-E02a: first-order stochastic optimisation cannot resolve the late-time residual |
| CAUSAL | causal loss weights, 16 bins, ε = 1/(16·L_tol) from the 1 %·ω₁²A0 rule (no tuning) | H-E02b: late times are fitted to a low-amplitude solution before early dynamics are resolved |

Every other mechanism was excluded with stated reasons: single loss term, shown not material, excluded by
the PI, or too costly.

## 3. Modal laboratory results (640,000-evaluation budget)

### 3.1 Final models, per seed (no pooling)

| Arm | Seed | L2_exact | P [cycles] | P_v | t_c [s] | freq err | decay [1/s] (3.54) | phase err | PDE residual of u | IC / BC | PDE evals | steps + L-BFGS evals | W [s] | W/M0 | RAM [MB] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 1234 | 0.2705 | 7.62 | 7.85 | 0.370 | 0.04 % | 5.40 | 0.033 | 0.0743 | 4.8e-6 / 2.8e-5 | 640,000 | 5000 | 236 | 1.00 | 704 |
| M0 | 1235 | 0.3043 | 6.62 | 6.80 | 0.322 | 0.21 % | 5.89 | 0.012 | 0.0458 | 4.8e-6 / 1.4e-5 | 640,000 | 5000 | 233 | 1.00 | 704 |
| M0 | 1236 | 0.3295 | 6.09 | 5.86 | 0.296 | 0.02 % | 6.05 | 0.038 | 0.0707 | 4.8e-6 / 1.4e-5 | 640,000 | 5000 | 233 | 1.00 | 705 |
| **LBFGS** | 1234 | **0.0382** | **20.58** | 20.58 | **1.000** | 0.02 % | **3.70** | 0.005 | **0.0101** | 4.8e-6 / 2.8e-5 | 623,360 | 2500 + 474 | 216 | 0.92 | 916 |
| **LBFGS** | 1235 | **0.0706** | **20.58** | 20.58 | **1.000** | 0.05 % | **3.62** | 0.050 | **0.0242** | 4.8e-6 / 2.8e-5 | 623,360 | 2500 + 474 | 217 | 0.93 | 909 |
| **LBFGS** | 1236 | **0.0934** | **20.58** | 20.58 | **1.000** | 0.09 % | **4.10** | 0.005 | **0.0241** | 4.8e-6 / 2.8e-5 | 623,360 | 2500 + 474 | 211 | 0.91 | 916 |
| CAUSAL | 1234 | 0.8656 | 4.14 | 3.85 | 0.201 | 199 %* | 8.45 | 0.019 | 0.665 | 4.8e-6 / 2.8e-5 | 640,000 | 5000 | 239 | 1.01 | 706 |
| CAUSAL | 1235 | 0.5597 | 4.64 | 4.85 | 0.225 | 97 %* | −11.96* | 2.99 | 0.361 | 4.8e-6 / 1.4e-5 | 640,000 | 5000 | 236 | 1.01 | 706 |
| CAUSAL | 1236 | 1.5551 | 2.10 | 1.91 | 0.102 | 75 %* | 1.06 | 0.72 | 1.278 | 4.8e-6 / 2.8e-5 | 640,000 | 5000 | 234 | 1.00 | 707 |

\* The global damped-cosine fit is meaningless for the CAUSAL traces: amplitude growth and spurious
late oscillation (`figures/B2-E02_q_of_t.png`).

All arms have 241,601 parameters.

**Passes.** Forward = backward passes = 5,000 (M0, CAUSAL) and 2,974 (LBFGS: 2,500 Adam steps + 474
closures, line search included).

**L-BFGS budget.** L-BFGS used 474 of its 500 permitted closures. The pre-registered overshoot guard left 26
unused, so 623,360 ≤ 640,000. It stopped on budget in all seeds, never on tolerance.

### 3.2 Pre-registered classification (`B2-E02_CLASSIFICATION.json`)

| Arm | Accuracy improvement (§8.1) | Efficiency (§8.2) | Dominance (§8.3) | Worse (§8.4) | Label |
|---|---|---|---|---|---|
| LBFGS | 3/3 (ΔP +13.0, +14.0, +14.5 vs threshold 3.06; L2 ratio 0.14, 0.23, 0.28) | 3/3† | 3/3 (W 0.91–0.93) | 0/3 | **DOMINANCE** |
| CAUSAL | 0/3 | 0/3 | 0/3 | 3/3 | **NO IMPROVEMENT** |

† **Caveat on my efficiency metric.** For seeds 1235 and 1236, E_reach is met at 256k evaluations. At that
point the LBFGS arm is still **bitwise identical to M0** (the switch is at 320k). Because of M0's own
late-training regression, M0's 256k state already beats its own 640k final state.

So the efficiency flag is genuinely attributable to L-BFGS only for seed 1234 (reached at 384k, after the
switch). The **DOMINANCE label does not depend on it**: it uses the accuracy criteria plus full-budget
wall-clock, both of which hold in 3/3 seeds.

**Not captured by "dominance" as pre-registered:** LBFGS needs **≈ 30 % more peak RAM** (909–916 vs 704 MB),
from the full-batch closure and L-BFGS history.

### 3.3 Collapse front

| Arm | CF_AUC per seed | CF_regressions |
|---|---|---|
| M0 | 0.303 / 0.321 / 0.322 | 0 / 0 / 2 |
| LBFGS | 0.510 / 0.574 / 0.637 | 0 / 0 / 0 |
| CAUSAL | 0.130 / 0.167 / 0.149 | 0 / 0 / 1 |

The LBFGS front tracks M0 until the 320k switch. It then advances to the full window by 512k–623k in every
seed (`figures/B2-E02_collapse_front.png`). The M0 retreat after 384k–512k does not occur.

### 3.4 q(t) error decomposition (§7; best candidate LBFGS vs M0)

| Term (relative L2 of q) | M0 (s1234 / 1235 / 1236) | LBFGS | Change |
|---|---|---|---|
| total L2 of q | 0.271 / 0.305 / 0.330 | 0.038 / 0.071 / 0.094 | −78 % |
| err_decay | **0.266 / 0.314 / 0.328** | 0.030 / 0.016 / 0.099 | −84 % |
| err_amplitude | 0.027 / 0.050 / 0.016 | 0.005 / 0.029 / 0.029 | −33 % |
| err_frequency | 0.009 / 0.054 / 0.004 | 0.006 / 0.012 / 0.023 | −40 % (both small) |
| err_phase | 0.033 / 0.012 / 0.038 | 0.005 / 0.049 / 0.005 | mixed |
| err_non_damped_cosine (collapse distortion) | 0.071 / 0.053 / 0.058 | 0.015 / 0.009 / 0.040 | −65 % |
| local decay before collapse [1/s] | 5.22 / 5.89 / 5.97 | 3.81 / 3.70 / 3.78 | (exact 3.54) |
| local frequency error before collapse | 4.0 % / 1.2 % / 2.5 % | 0.08 % / 0.05 % / 0.04 % | — |

**Pre-registered mechanism category: B (amplitude decay).**
- Freq+phase reduction 34 %; **decay+amplitude reduction 79 %**.
- The control's error is dominated by over-damping (err_decay ≈ the whole L2), and L-BFGS removes most of
  it.
- It also removes the late-training regression and the seed spread of P (std 0.78 → 0).

The decomposition is pre-registered for the best candidate and M0 only. The code also emits a category for
CAUSAL ("C"); it is not meaningful there, because CAUSAL is worse on every term.

### 3.5 Verdict on the modal hypotheses

**H-E02a (optimisation conditioning / stochastic first-order limits): SUPPORTED in the modal diagnostic.**
At fixed network size, a deterministic quasi-Newton refinement stage converts the collapsing, over-damped
solution into a full-window solution with near-exact decay, in every seed.

**H-E02b (temporal causality): NOT SUPPORTED.**
- The pre-registered causal weighting degrades every metric in every seed.
- q(t) shows spurious late oscillation and amplitude growth: the late bins receive almost no weight
  (ε rule), so they are left effectively unconstrained.
- No ε variants were tried; per the pre-registration it stays rejected.

## 4. Full-field transfer (§10)

### 4.1 Final models

| Arm | Seed | L2_exact | P [cycles] | t_c [s] | freq err | decay [1/s] | PDE residual of u | PDE evals | W [s] | W/B1 | RAM [MB] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 1234 | 0.526 | 3.15 | 0.153 | 0.96 % | 8.98 | 0.110 | 640,000 | 1045 | 1.00 | 922 |
| B1 | 1235 | 0.549 | 3.11 | 0.151 | 0.91 % | 9.47 | 0.097 | 640,000 | 1029 | 1.00 | 907 |
| B1 | 1236 | 0.402 | 5.10 | 0.248 | 0.33 % | 7.03 | 0.096 | 640,000 | 1021 | 1.00 | 905 |
| B1+LBFGS | 1234 | 0.671 | 1.60 | 0.077 | 0.05 % | 15.64 | 0.171 | 623,360 | 910 | 0.87 | 1483 |
| B1+LBFGS | 1235 | 0.843 | 0.72 | 0.035 | 5.7 % | 24.08 | 0.231 | 623,360 | 916 | 0.89 | 1479 |
| B1+LBFGS | 1236 | 0.578 | 2.53 | 0.123 | 2.0 % | 10.55 | 0.117 | 623,360 | 905 | 0.89 | 1475 |

**Transfer criteria** (all 3 seeds required): ΔP ≥ 3.98, L2 ratio ≤ 0.9, frequency error ≤ 2 %, residual
ratio ≤ 1.1.

| Seed | ΔP | L2 ratio | residual ratio | Result |
|---|---|---|---|---|
| 1234 | −1.56 | 1.28 | 1.55 | **FAIL** |
| 1235 | −2.39 | 1.54 | 2.38 | **FAIL** |
| 1236 | −2.57 | 1.44 | 1.22 | **FAIL** |

The criteria fail in every seed, so `transfer_passed = false` and the §11 decision is **E**.

### 4.2 What happens on the full field (descriptive; not a tested causal claim)

**The trajectory.** Up to 256k evaluations both arms are identical. After the switch, the L-BFGS front stops
advancing (`figures/B2-E02T_collapse_front.png`):

| Seed | 256k | after the switch → 623k |
|---|---|---|
| 1234 | 1.63 | 1.65 → 1.60 cycles |
| 1235 | 0.71 | stays at 0.72 |
| 1236 | 2.14 | 2.16 → 2.53 |

Over the same budget B1 keeps advancing to 3.1–5.1 cycles. The L-BFGS dense-grid residual *rises* after the
switch: s1235 0.166 → 0.231; s1234 0.140 → 0.171.

**Overfitting signature.** Residual RMS relative to the exact RMS(u_tt), after L-BFGS:

| Problem | Fixed-set residual (its 640 training points) | Dense-grid u-residual | Dense/fixed |
|---|---|---|---|
| Full field | 0.003–0.005 | 0.117–0.231 | **36–78×** |
| Modal | 0.002–0.004 | 0.010–0.024 | 3–11× |

This is consistent with **overfitting to the fixed collocation set**:
- 640 points sample the 1-D modal time interval densely (≈ 31 per cycle);
- the same 640 points are sparse in the 2-D (x,t) domain of the 4th-order full-field problem.

The mechanism suggested by the data is therefore that the modal gain came from deterministic second-order
refinement **on an adequately sampled problem**. On the full field the same fixed-set refinement optimises a
sparse sample. This interpretation has **not** been tested (for example by resampling the L-BFGS set, using
more points, or a stochastic quasi-Newton method). It is a hypothesis for review, not a result.

## 5. Answers to the B2-E02 questions

1. **Is the remaining temporal collapse (modal) at least partly an optimisation / training-control failure?**
   **Yes, in the modal diagnostic.** At the same network size, physics, initialisation and ≤ budget, changing
   only the optimizer after 320k evaluations removes the collapse in 3/3 seeds. The fix acts mainly on
   **amplitude decay** (category B) and on stability (no late regression; zero seed spread in P).
2. **A substantially better accuracy-per-compute point?** In the modal diagnostic, yes: dominance on accuracy
   and wall-clock, with fewer evaluations and +30 % RAM. On the full field, **no**: cheaper (0.87–0.89×
   wall-clock) but worse on every accuracy and physics metric, with 1.6× the RAM.
3. **Does it survive restoring the spatial field?** **No.** The "physics difficulty → strategy" link observed
   in the temporal laboratory does not transfer unchanged. It is the transfer test, not the modal result, that
   determines the method question.

## 6. Limitations

- **Three seeds per arm.** The modal LBFGS effect is far outside the seed spread (ΔP ≥ +13 vs threshold
  3.06); the transfer failure is consistent in sign across all seeds.
- **One switch point (50/50), one L-BFGS configuration (PyTorch defaults), one fixed-set size (640).** These
  were declared, not tuned. The transfer failure may depend on them, which is untested.
- **Efficiency metric weakness** (§3.2 †): for 2/3 seeds it is satisfied by the control's own early state.
  Dominance does not depend on it.
- **Memory** is not part of the pre-registered dominance definition. L-BFGS costs +30 % (modal) and +60 %
  (full field) peak RAM.
- **Wall-clock** differs between sessions by ≈ 1.3×; only same-batch ratios are compared. Transfer batch 1
  had 4 processes and batch 2 had 2.
- **Modal results are diagnostic only.** The exact spatial mode shape is supplied.

## 7. Recommendations (NOT executed; each needs PI approval)

1. **Diagnose the transfer failure before any new method.** One single-change arm at a time, each testing
   the fixed-set overfitting explanation:
   - (a) L-BFGS on a **resampled** collocation set every k closures;
   - (b) L-BFGS on a **larger** fixed set (e.g. 2,560 points; fewer closures at the same budget);
   - (c) full-batch Adam on the same fixed 640 points (separates "fixed set" from "quasi-Newton").
2. **Report memory as a fourth Pareto axis** in future pre-registrations.
3. **Re-define E_reach** in future pre-registrations as "first point *after the arm diverges from the
   control*" (or against the control's best-so-far state), so it cannot be satisfied by the control's own
   regression.
4. **No novelty claim**:
   - L-BFGS refinement is established;
   - the modal success is a diagnostic result;
   - the transfer failed.

   The problem-adaptive selection idea remains a hypothesis. This experiment shows that the right control for
   the temporal sub-problem is **not automatically** right for the coupled space-time problem. That is
   relevant evidence for, not against, the need for problem-aware selection, but it is no evidence that such
   a selector works.

**STOPPED. Waiting for review of B2-E02.**
