# B2-E01 Phase A — seed replication (seeds 1234, 1235, 1236)

**Type:** robustness check of B2-E01. All four arms were re-run with seeds 1235 and 1236 using exactly the
same configuration, budget (640,000 PDE evaluations), snapshot cadence, evaluator and classification rules.
**No B2-E02 training has started.**

## Provenance

| Item | Value |
|---|---|
| Configuration | `configs/batch2/B2-E01_mechanism_attribution.yaml`, sha256 `a515aaea…e9a6`, identical to the B2-E01 registry entry; spec, frozen baselines, `src/` and `experiments/` unchanged since the pre-registration commit `24813bf` (verified by `git diff` before launch) |
| Seed 1235 launch commit | `c7c1e4a` |
| Seed 1236 launch commit | `ce9cd92`. The only differences are additive B2-E02 preparation files, a new runner branch and the provenance dirty-flag rule; `git diff` shows no change under `src/beampinn`, `physref/arms.py`, `formulations/`, `persistence.py`, `safety.py` or `configs/` |
| Execution | 4 concurrent single-thread processes per seed batch (same policy as B2-E01) |
| Evaluator | `scripts/analyze_b2_e01_seeds.py` imports the frozen B2-E01 evaluator. Its seed-1234 re-evaluation reproduces `B2-E01_RESULTS.csv` exactly (asserted) |
| Tables | `results_batch2/tables/B2-E01S_SEED_{RESULTS,SNAPSHOTS,SUMMARY}.csv`, `B2-E01S_SEED_CLASSIFICATION.json` |
| Figures | `results_batch2/figures/B2-E01S_{L2_vs_pde_evals,collapse_front,final_by_seed,pareto}.png` |

**Wall-clock caveat.** Absolute training times of the seed-1235 and seed-1236 batches are ≈ 1.3× those of
seed 1234 for every arm. Machine load differed between sessions. Compute comparisons therefore use the
**within-batch ratio to B1**, which is stable: mixed 0.46–0.47×, modal 0.23–0.24×, FP64 1.62–1.71×.

## 1. Per-seed results (final models; seeds are NOT pooled)

Abbreviations: P = persistence, P_v = velocity persistence, t_c = collapse time, freq err = fitted
frequency below ω_d, W = training time.

| Arm | Seed | L2_exact | P [cycles] | P_v | t_c [s] | freq err | decay [1/s] (exact 3.54) | PDE residual of u | W/B1 | RAM [MB] | Class vs B1 | Case |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 1234 | 0.526 | 3.15 | 3.36 | 0.153 | 0.96 % | 8.98 | 0.110 | 1.00 | 934 | ref. | gate PASS (bit-exact) |
| B1 | 1235 | 0.549 | 3.11 | 2.88 | 0.151 | 0.91 % | 9.47 | 0.097 | 1.00 | 926 | ref. | — |
| B1 | 1236 | 0.402 | **5.10** | 4.86 | 0.248 | 0.33 % | 7.03 | 0.096 | 1.00 | 904 | ref. | — |
| B1-FP64 | 1234 | 0.526 | 3.15 | 3.36 | 0.153 | 0.96 % | 8.98 | 0.110 | 1.71 | 1178 | COMPARABLE | not material |
| B1-FP64 | 1235 | 0.549 | 3.11 | 2.88 | 0.151 | 0.91 % | 9.47 | 0.097 | 1.62 | 1176 | COMPARABLE | not material |
| B1-FP64 | 1236 | 0.402 | 5.10 | 4.86 | 0.248 | 0.33 % | 7.03 | 0.096 | 1.65 | 1170 | COMPARABLE | not material |
| mixed | 1234 | 0.675 | 1.67 | 1.86 | 0.081 | 2.01 % | 13.24 | 2.57 | 0.46 | 730 | WORSE | B |
| mixed | 1235 | 0.623 | 2.13 | 1.89 | 0.103 | 1.27 % | 11.54 | 1.12 | 0.47 | 730 | COMPARABLE | **A** |
| mixed | 1236 | 0.628 | 2.13 | 1.90 | 0.104 | 1.53 % | 11.53 | 1.65 | 0.46 | 730 | WORSE | B |
| modal (diag.) | 1234 | 0.270 | 7.62 | 7.85 | 0.370 | 0.04 % | 5.40 | 0.074 | 0.23 | 704 | BETTER | partial |
| modal (diag.) | 1235 | 0.304 | 6.62 | 6.80 | 0.322 | 0.21 % | 5.89 | 0.046 | 0.24 | 704 | BETTER | partial |
| modal (diag.) | 1236 | 0.329 | 6.09 | 5.86 | 0.296 | 0.02 % | 6.05 | 0.071 | 0.23 | 705 | COMPARABLE | H2-support |

Every arm takes 5,000 optimizer steps (= 640,000 PDE evaluations), and IC/BC errors are at the hard-ansatz
level (≤ 3e-5; ~1e-14 in FP64). Parameters: 241,601 (B1, FP64, modal), 242,002 (mixed). No run passes the
full-window persistence gate.

## 2. Seed statistics (mean ± sample std [min, max]; n = 3)

| Arm | L2_exact | P [cycles] | t_c [s] | decay [1/s] | PDE residual of u |
|---|---|---|---|---|---|
| B1 | 0.492 ± 0.079 [0.402, 0.549] | 3.79 ± 1.14 [3.11, 5.10] | 0.184 ± 0.055 | 8.49 ± 1.29 | 0.101 ± 0.008 |
| B1-FP64 | identical to B1 to 6 digits | identical | identical | identical | identical |
| mixed | 0.642 ± 0.029 [0.623, 0.675] | 1.97 ± 0.27 [1.67, 2.13] | 0.096 ± 0.013 | 12.10 ± 0.99 | 1.78 ± 0.74 |
| modal (diag.) | 0.301 ± 0.030 [0.270, 0.329] | 6.78 ± 0.78 [6.09, 7.62] | 0.329 ± 0.038 | 5.78 ± 0.34 | 0.064 ± 0.016 |

Full statistics, including CV and ranges of every principal metric, are in `B2-E01S_SEED_SUMMARY.csv`.

## 3. Phase-B questions

**1. Are the B2-E01 classification rules confirmed?**

The rules were re-run per seed, against the same-seed B1:

| | Seed 1234 | Seed 1235 | Seed 1236 |
|---|---|---|---|
| FP64 | not material | not material | not material |
| mixed | B | **A** | B |
| modal | partial | partial | **H2-support** |
| Case E | no | no | no |

**2. Is the modal improvement stable?**

- **The direction is stable**: modal beats B1 in every seed on persistence (×2.42, ×2.13, ×1.19), L2
  (×0.51, ×0.55, ×0.82), frequency error (≤ 0.21 % vs 0.33–0.96 %) and decay (5.4–6.1 vs 7.0–9.5 1/s).
- **The magnitude is not stable**. Case C ("dramatically better") is reached in no seed. In seed 1236, where
  B1 happened to persist longest (5.10 cycles), the modal gain is within the 1-cycle comparability band.

**3. Is the ~0.37 s temporal collapse stable?**

- **The collapse itself is**: the modal diagnostic collapses in every seed and never covers the window.
- **The location is not "0.37 s"**: it lies at 0.296–0.370 s (6.1–7.6 cycles), and 0.37 s was the seed-1234
  value.
- **A new, reproducible observation (snapshots, `B2-E01S_SEED_SNAPSHOTS.csv`).** In all three seeds the modal
  front first advances to its best state and then **retreats** by the end of training:

  | Seed | Best state (at) | Final state (640k) | Front retreat |
  |---|---|---|---|
  | 1234 | P 8.16, L2 0.243 (512k) | P 7.62, L2 0.270 | 0.026 s |
  | 1235 | P 8.60, L2 0.229 (384k) | P 6.62, L2 0.304 | 0.096 s |
  | 1236 | P 8.62, L2 0.233 (384k) | P 6.09, L2 0.329 | 0.123 s |

  In the full-field arms (B1, mixed) the front is still advancing monotonically at 640k in all seeds. In the
  temporal-only problem, late training therefore degrades a previously better solution. This is consistent
  with an optimisation / training-control component of the collapse. It is an observation, not a tested
  causal claim.

**4. Does FP64 remain behaviourally identical?**

**Yes, in all three seeds.** The maximum relative L2 difference is 5.4e-6, ΔP = 0 and fitted frequency and
decay are identical to the reported digits. FP64 costs 1.62–1.71× the wall-clock and +26 % RAM. **Precision
is ruled out as a cause for this configuration.**

**5. Does mixed remain a non-dominating cheap/worse trade-off?**

**Yes, it is non-dominating in all seeds.** It is always 0.46–0.47× cheaper and never better: WORSE in 2/3
seeds, COMPARABLE (Case A by the pre-registered thresholds) in seed 1235. In every seed its displacement
violates the beam equation: PDE residual 1.12–2.57 vs B1's 0.096–0.110. Its fitted decay (11.5–13.2) is the
most over-damped of all arms.

The pre-registered single-seed "Case B" is therefore **not robust as a label** (B, A, B). What is robust is
that mixed is cheaper, never more accurate, and physically less consistent.

**6. Seed variance.**

- **B1's persistence varies by 2.0 cycles across seeds** (3.11–5.10; CV 0.30), and L2 by 0.15.
- B2-E01 justified its 1-cycle / 25 % thresholds with the Batch-1 seed spread of the Y1 recipe (0.1 cycles).
  That spread is **≈ 20× smaller** than the spread now measured for the B1 (Z4) recipe.
- Consequently the single-seed B2-E01 margins were less safe than stated. The modal–B1 gap in seed 1236
  (+0.99 cycles) is inside B1's own seed range.
- The modal arm is less variable in P (range 1.53) and in L2 (0.059) than B1.

## 4. Does replication invalidate B2-E01? (STOP condition 1)

**No.** Every qualitative B2-E01 conclusion holds in all seeds:
- precision is not a cause;
- mixed is cheaper but never more accurate and physically inconsistent;
- removing the spatial field improves persistence, L2, frequency and decay, but the temporal-only problem
  still collapses;
- no arm covers the window.

What changes is the **strength** of two statements:
- the modal advantage is 1.2–2.4× in persistence, not a fixed ≈ 2.4×;
- the collapse location is 0.30–0.37 s.

The B2-E01 report carries an amendment pointing here.

## 5. Consequences for B2-E02 (feeds `docs/hypotheses/B2-E02.md` §9)

- Phase-A modal P range = 1.53 cycles, so the pre-registered formula gives
  **ΔP_thr = max(1.0, 2 × 1.53) = 3.06 cycles** for B2-E02 accuracy claims.
- Phase-A B1 P range = 1.99 cycles, so the transfer threshold is **max(1.0, 2 × 1.99) = 3.98 cycles**.
- The control's own late-training regression (front retreat in 3/3 seeds) is the concrete failure B2-E02
  targets.
