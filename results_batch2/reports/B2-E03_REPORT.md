# B2-E03 — Collocation coverage vs quasi-Newton optimizer: why does L-BFGS solve the modal problem but fail on the full field?

## PRE-REGISTERED OUTCOME: **CASE D**. The fixed-collocation explanation is insufficient (H4 supported).

| Hypothesis | Verdict (`B2-E03_CLASSIFICATION.json`) |
|---|---|
| H1 (coverage) | **not supported**. Only LBFGS-R is "materially better than LBFGS-F", and in 1/3 seeds |
| H2 (optimizer: full-batch Adam beats L-BFGS on the same points) | **falsified** (0/3) |
| H3 (coverage dominant: R and 4X both restore B1 level) | **falsified** |
| H4 (none of R, 4X, ADAM-FULL materially better in ≥ 2/3 seeds) | **supported** |
| Chunking-confound flag (§4) | not raised: LBFGS-F fails RB1 in 3/3 seeds, as in B2-E02 |

**The mechanistic answer** (§6) **separates two phenomena that B2-E02 had observed together:**

1. **The train/dense residual gap** (36–78× in B2-E02, reproduced here) is a **coverage × optimizer
   interaction**:
   - it requires a fixed sparse set;
   - L-BFGS amplifies it about 10× relative to Adam on the identical points;
   - it disappears with resampling or a 4× larger reservoir.
2. **The stalled collapse front, i.e. the physical dynamics failure, is *not* explained by that gap.**
   - Removing the gap (LBFGS-4X) leaves the front stalled.
   - Adam on the fixed set also stalls.
   - The only stage-2 arm that moves the front after the switch is LBFGS-R (fresh points every 50
     closures), and only in 2/3 seeds. It never beats B1, which keeps advancing in every seed.

**Nothing here is novel:** L-BFGS, Adam, full-batch optimisation, collocation resampling and larger
collocation sets are established. This is a bottleneck-identification study.

---

## 1. Provenance and integrity

| Item | Value |
|---|---|
| Pre-registration | `docs/hypotheses/B2-E03.md`, committed and pushed in `cdcb7a4` before any B2-E03 run |
| Amendment 1 | `64f039e`, written after LBFGS-4X s1234 diverged and **before** any metric of any other arm or seed was inspected. It classifies diverged runs (no threshold introduced) |
| Training code | identical for all 15 runs (`git diff cdcb7a4 64f039e -- src experiments configs` = empty); `git_dirty=False` |
| Runs | 15 = 5 arms × seeds 1234–1236. One batch per seed: 5 concurrent single-thread processes on 4 vCPU (declared oversubscription) |
| **Initialisation** | sha256 identical across the 5 arms of every seed: s1234 `9c5c91e3…`, s1235 `d5d35861…`, s1236 `8c963bb2…` (asserted) |
| **set0** | identical across all stage-2 arms and **identical to the B2-E02 transfer fixed set** (asserted) |
| **Compute** | B1 640,000; every completed stage-2 arm 623,360 (474 closures / full-batch steps); LBFGS-4X s1234 diverged at 576,640 (Amendment 1). Never above 640,000 (asserted) |
| **Dense grid** | 51×501 uniform grid, independent of training points: no coincident node (asserted for every evaluated point) |
| **Frozen B1** | bit-exact vs the Phase-A B1 runs of the same seeds (asserted) |
| Evaluator | `scripts/analyze_b2_e03.py`, written and committed with the pre-registration |
| Outputs | `results_batch2/B2-E03_{RESULTS,SNAPSHOTS}.csv`, `B2-E03_CLASSIFICATION.json`, `results_batch2/figures/B2-E03_*.png` (14 figures) |

**Incident (no effect on results).** A `pkill` in my own wait command twice terminated the *waiting shell*,
never a training process. All runs completed or diverged on their own; the run logs show this.

## 2. Arms (one stage-2 change each; stage 1 = B1's first 320,000 evaluations, bitwise)

| Arm | Stage 2 (320k → 623k evaluations) | Distinct stage-2 points |
|---|---|---|
| B1 | B1 continues: Adam, mini-batch 128, fresh 640-point epoch every 5 steps | ≈ 2,500 epochs |
| LBFGS-F | L-BFGS on fixed set0 (640) | 640 |
| LBFGS-R | L-BFGS; fresh 640-point draw every 50 closures | 6,400 (10 sets) |
| LBFGS-4X | L-BFGS; 2,560-point reservoir (4 sets), active 640-subset cycled every 50 closures | 2,560 |
| ADAM-FULL | the same Adam state and schedule; 474 full-batch steps on set0 | 640 |

L-BFGS iterations: 433–443 per completed run (474 closures including line search).

## 3. Final results (per seed; no pooling)

Abbreviations: R_d = dense residual, R_t = training-set residual, G = R_d/R_t, freq err = frequency error,
W = training time.

| Seed | Arm | L2_exact | P [cycles] | R_d | R_t | G | freq err | decay [1/s] (exact 3.54) | W [s] | RAM [MB] |
|---|---|---|---|---|---|---|---|---|---|---|
| 1234 | B1 | 0.526 | 3.15 | 0.110 | 0.102 | 1.1 | 0.96 % | 8.98 | 1297 | 923 |
| 1234 | LBFGS-F | 0.671 | 1.60 | 0.170 | 0.0048 | **35.3** | 0.05 % | 15.63 | 1169 | 1495 |
| 1234 | LBFGS-R | **0.529** | **3.60** | **0.078** | 0.042 | 1.8 | 1.30 % | 9.18 | 1158 | 1489 |
| 1234 | LBFGS-4X | **diverged** at closure 401 / 576,640 evals (last finite 512k: L2 0.578, P 2.66) | | | | | | | 1095 | 1492 |
| 1234 | ADAM-FULL | 0.631 | 1.67 | 0.113 | 0.029 | 4.0 | 1.41 % | 12.69 | 1131 | 1383 |
| 1235 | B1 | 0.549 | 3.11 | 0.097 | 0.088 | 1.1 | 0.91 % | 9.47 | 1313 | 922 |
| 1235 | LBFGS-F | 0.843 | 0.72 | 0.231 | 0.0030 | **76.3** | 5.68 % | 24.07 | 1198 | 1493 |
| 1235 | LBFGS-R | 0.755 | 1.19 | 0.149 | 0.053 | 2.8 | 5.10 % | 17.10 | 1177 | 1491 |
| 1235 | LBFGS-4X | 0.804 | 0.69 | 0.204 | 0.173 | 1.2 | 1.07 % | 24.46 | 1186 | 1487 |
| 1235 | ADAM-FULL | 0.824 | 1.12 | 0.188 | 0.033 | 5.7 | 5.90 % | 20.38 | 1136 | 1389 |
| 1236 | B1 | 0.402 | 5.10 | 0.096 | 0.090 | 1.1 | 0.33 % | 7.03 | 1316 | 923 |
| 1236 | LBFGS-F | 0.579 | 2.52 | 0.117 | 0.0027 | **43.2** | 1.99 % | 10.55 | 1154 | 1486 |
| 1236 | LBFGS-R | 0.515 | 3.16 | **0.068** | 0.035 | 2.0 | 1.13 % | 9.10 | 1164 | 1492 |
| 1236 | LBFGS-4X | 0.623 | 2.14 | 0.115 | 0.081 | 1.4 | 0.29 % | 10.96 | 1173 | 1490 |
| 1236 | ADAM-FULL | 0.602 | 2.17 | 0.105 | 0.017 | 6.3 | 1.92 % | 11.19 | 1145 | 1391 |

Notes on the columns:
- For LBFGS-4X, R_t is measured on the whole 2,560-point reservoir.
- IC/BC errors stay at the hard-ansatz level (≤ 3e-5) in every completed run.
- Parameters: 241,601 in every arm.
- **No run in any arm passes the full-window persistence gate.**

**Classification flags per seed** (vs LBFGS-F unless stated):

| Arm | MB (1234 / 1235 / 1236) | RB1 vs B1 | IB1 vs B1 | OS (overfitting signature) |
|---|---|---|---|---|
| LBFGS-F | — | ✗ ✗ ✗ | ✗ ✗ ✗ | **✓ ✓ ✓** |
| LBFGS-R | **✓** ✗ ✗ (L2 ratio 0.79 / 0.90 / 0.89; R_d ratio 0.46 / 0.64 / 0.58) | **✓** ✗ ✗ | ✗ ✗ ✗ | ✗ ✓ ✗ |
| LBFGS-4X | diverged ✗ ✗ (L2 ratio — / 0.95 / 1.08) | ✗ ✗ ✗ | ✗ ✗ ✗ | — ✗ ✓ |
| ADAM-FULL | ✗ ✗ ✗ (L2 ratio 0.94 / 0.98 / 1.04) | ✗ ✗ ✗ | ✗ ✗ ✗ | ✓ ✓ ✓ |

Counts: MB R 1/3, 4X 0/3, ADAM-FULL 0/3. This gives Cov2 = false and Opt2 = false, so **CASE D**.

## 4. Trajectories after the switch (`B2-E03_SNAPSHOTS.csv`)

Columns are persistence P and G at the switch (320k) → 384k → 512k → 623k:

| Seed | Arm | P | G |
|---|---|---|---|
| 1234 | LBFGS-F | 1.69 → 1.65 → 1.61 → 1.60 | 1.0 → 5.7 → 18.8 → 35.3 |
| 1234 | LBFGS-R | 1.69 → 2.17 → 3.12 → **3.60** | 1.0 → 2.6 → 2.0 → 1.8 |
| 1234 | LBFGS-4X | 1.69 → 2.17 → 2.66 → diverged | 1.0 → 1.1 → 1.2 → — |
| 1234 | ADAM-FULL | 1.69 → 1.67 → 1.67 → 1.67 | 1.0 → 2.0 → 3.2 → 4.0 |
| 1235 | LBFGS-F | 1.16 → 0.73 → 0.73 → 0.72 | 1.1 → 12.5 → 44.6 → 76.3 |
| 1235 | LBFGS-R | 1.16 → 1.04 → 1.16 → 1.19 | 1.1 → 3.6 → 3.3 → 2.8 |
| 1235 | LBFGS-4X | 1.16 → 1.04 → 0.69 → 0.69 | 1.0 → 1.1 → 1.2 → 1.2 (R_t rises 0.139 → 0.173) |
| 1235 | ADAM-FULL | 1.16 → 1.15 → 1.13 → 1.12 | 1.1 → 2.3 → 4.1 → 5.7 |
| 1236 | LBFGS-F | 2.16 → 2.16 → 2.18 → 2.52 | 1.0 → 8.1 → 25.1 → 43.2 |
| 1236 | LBFGS-R | 2.16 → 2.14 → 2.61 → **3.16** | 1.0 → 4.2 → 2.8 → 2.0 |
| 1236 | LBFGS-4X | 2.16 → 2.14 → 2.13 → 2.14 | 1.0 → 1.4 → 1.4 → 1.4 |
| 1236 | ADAM-FULL | 2.16 → 2.18 → 2.17 → 2.17 | 1.0 → 2.5 → 4.5 → 6.3 |

From 384k to 640k, B1 keeps advancing: 1234 2.16 → 3.15; 1235 1.61 → 3.11; 1236 2.65 → 5.10. B1 has no switch snapshot.

Key figures:
- `B2-E03_Rtrain_vs_Rdense.png` (key plot);
- `B2-E03_collapse_front.png`;
- `B2-E03_residual_gap_ratio.png`;
- `B2-E03_R_train.png` and `B2-E03_R_dense.png`.

## 5. Memory and compute (Pareto, §9)

**Peak RAM:**

| Arm | Peak RAM [MB] | vs B1 |
|---|---|---|
| B1 | 922–923 | — |
| ADAM-FULL | 1,383–1,391 | +50 %: full-batch 640-point 4th-order graph |
| L-BFGS arms | 1,486–1,495 | +61 % |

**Wall-clock** is within-batch, with 5 processes on 4 vCPU:
- B1: 1,297–1,316 s at 640,000 evaluations;
- stage-2 arms: 1,095–1,198 s at 623,360.

**Non-dominated arms** on (L2, P, W, RAM):

| Seed | Non-dominated |
|---|---|
| 1234 | B1, LBFGS-R, ADAM-FULL |
| 1235 | B1, LBFGS-R, LBFGS-4X, ADAM-FULL |
| 1236 | B1, LBFGS-F, LBFGS-R, ADAM-FULL |

**B1 is non-dominated in 3/3 seeds and never dominated on accuracy.** The stage-2 arms appear only because
they are cheaper in wall-clock: they are faster but less accurate, at +50–61 % memory. With memory counted, no
stage-2 arm is an accuracy/compute/memory improvement over B1.

## 6. The mechanistic answer: why L-BFGS solves the modal problem but fails on the full field

**What the data establish (3 seeds, pre-registered measurements):**

1. **The fixed-set overfitting signature is real and reproducible.** LBFGS-F shows OS in 3/3 seeds:
   - the training residual falls 22–44× (to 0.003–0.005);
   - the dense residual rises;
   - G grows to 35–76;
   - the front stalls.

   This reproduces B2-E02 with different chunking (50 vs ≤ 200 closures), so it is not a chunking artefact.
2. **The gap is a coverage × optimizer interaction.** On the **identical** fixed set, full-batch Adam reaches
   G = 4–6, about 10× less than L-BFGS. With coverage changed (fresh points or 4× reservoir), L-BFGS reaches
   G = 1.2–2.8, close to B1's 1.1. Both the fixed sparse set and the quasi-Newton optimizer are needed for
   the extreme gap.
3. **The gap is not what breaks the dynamics:**
   - LBFGS-4X has essentially no gap (G 1.2–1.4) yet stalls (s1235, s1236) or diverges (s1234).
   - ADAM-FULL has a modest gap and stalls in 3/3 seeds.
   - Closing the train/dense gap is therefore **not sufficient** to restore front propagation.
4. **No stage-2 variant matches B1's dynamics.** Every stage-2 arm stays more over-damped than B1 in
   every seed (decay 9.1–24.5 vs 7.0–9.5 1/s). The only stage-2 front that advances is LBFGS-R in 2/3
   seeds, and it reaches B1 level in only 1/3 (seed 1234: P 3.60 vs 3.15, L2 0.529 vs 0.526, R_d 0.078 vs
   0.110).

**Answer to the boxed question, at the strength the evidence allows:**

- **In the modal problem**, 640 fixed points in 1-D give ≈ 31 points per period. Fixed-set full-batch
  refinement generalises (dense/train 3–11× in B2-E02), and L-BFGS's curvature use removes the
  over-damping error.
- **On the full field**, the same 640 points are sparse in 2-D (≈ 5.6 per period per axis). L-BFGS
  overfits them (G 35–76): **collocation coverage and the quasi-Newton optimizer interact** to produce the
  residual gap.
- **However, the dynamic failure on the full field is not primarily caused by that interaction.** Fixing
  the gap does not fix the dynamics, and the optimizer swap (full-batch Adam) does not fix them either.
- **Neither "coverage" nor "quasi-Newton optimisation" is the dominant cause of the stalled front.** The
  pre-registered verdict is CASE D: another mechanism must be identified.

**What the remaining evidence suggests** (a hypothesis for review; **not** a conclusion, and not tested
here):

- Every stage-2 policy that holds its collocation objective **stationary** for long periods stalls:
  - LBFGS-F, ADAM-FULL: one set;
  - LBFGS-4X: a revisited finite reservoir.
- The only stage-2 policy whose front still advances is the one that keeps drawing **fresh points**
  (LBFGS-R), and the B1 control draws fresh points every 5 steps and advances in every seed.
- This is consistent with front propagation on the full field depending on a **continually refreshed
  collocation measure** (stochastic, non-stationary sampling), rather than on coverage density or the
  optimizer class. This interpretation is close to the "propagation failure" literature (Daw et al. 2023)
  and would need its own pre-registered test.

There are two confounds that test must remove:
- **LBFGS-R and LBFGS-4X also keep stale curvature history across set switches** (pre-registered: no
  reset). In s1235 LBFGS-4X even *increased* its reservoir residual (0.139 → 0.173), so it failed to
  optimise the larger objective at all.
- **ADAM-FULL performs 5.3× fewer parameter updates than B1's stage 2** at equal evaluations (474 vs 2,500).

## 7. Limitations

- **Three seeds.** LBFGS-R's benefit is seed-dependent (1/3 MB), and B1's own seed range is 2 cycles.
- **One divergence.** LBFGS-4X s1234 is excluded from success counts by Amendment 1, a rule written after
  the divergence but before any other result was seen.
- **Design confounds.**
  - The L-BFGS history is not reset at set switches (R, 4X).
  - Iteration counts differ between B1 (2,500 mini-batch updates in stage 2) and every stage-2 arm (474
    closures / steps).
  - The 4X arm cycles subsets, so "denser coverage" is never a single, fixed, fully evaluated objective.
- **One switch point, one resampling period (50 closures), one reservoir factor (4×)**, all declared a
  priori. Sensitivity was not tested.
- **Wall-clock under 5-way oversubscription** is comparable only within a batch.

## 8. Recommendations (NOT executed; each needs pre-registration and approval)

1. **Test the sampling-freshness hypothesis directly at the stage-2 point.** Change one variable at a time
   against B1's continued Adam:
   - (a) B1 Adam continued on a **fixed** 640-point set with mini-batches of 128 (freshness removed,
     optimizer and update count kept);
   - (b) L-BFGS-R with the history **reset** at every resample (removes the stale-curvature confound);
   - (c) L-BFGS with a fresh draw **every closure batch**, i.e. stochastic L-BFGS.

   If (a) stalls like ADAM-FULL, while B1 advances, freshness, not optimizer class, is the operative
   variable.
2. **Keep the residual-gap diagnostic (G) in every future run.** It reliably exposes fixed-set
   overfitting; but, as shown here, a small G does not imply correct dynamics.
3. **No novelty claim.** The project's possible contribution, a problem-adaptive strategy selector, gets one
   relevant data point: the same optimizer interacts with collocation design differently in 1-D and 2-D.
   The deciding variable for the full field has not yet been identified.

**STOPPED. Waiting for review of B2-E03.**
