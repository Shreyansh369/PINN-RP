# B2-E01 — Failure-mechanism attribution at matched compute (Mode 1, FE-D-M1)

**Type:** mechanistic diagnostic / benchmark study. **No method is claimed as novel.** Mixed formulations,
modal reduction, hard constraints, Fourier features and FP64 are all established
(`docs/NOVELTY_GAP.md` §A/E).

**Scope:** stage 1, single seed (1234). Every conclusion below is **provisional** until seeds 1235 and 1236
are run; they are not approved.

**Pre-registration:** `docs/hypotheses/B2-E01.md`, committed in `24813bf` before the first run started at
01:31 UTC. The classification below was computed by `scripts/analyze_b2_e01.py` with exactly those
thresholds. Nothing was changed after the results were seen.

**Outcome at a glance:**

| Arm | Pre-registered outcome | What it means |
|---|---|---|
| Reproducibility gate (B1) | **PASS (bit-exact)** | Batch-1 behaviour is reproduced inside PINN-RP |
| FP64 | **not material** (Case D not triggered) | precision does not contribute to the Batch-1 failure |
| mixed | **Case B** | cheaper (0.46× wall-clock) but less accurate, with wrong dynamics |
| modal | **partial** | removing the spatial field gives 2.4× more persistence, but the temporal-only problem still collapses at 37 % of the window |
| all arms | **Case E not triggered** | the arms do not all fail similarly |

---

## 1. What was run

| Arm | ID | Change vs B1 (`configs/frozen/batch1_hard_tanh2.yaml`) | Run key / record |
|---|---|---|---|
| B1 (reference) | `B2-E01-B1-s1234` | none | `Z4_20K__s1234__f78aa7f1da` (= Batch-1 Z4 @ 5k) |
| B1-FP64 (precision control) | `B2-E01-B1_fp64-s1234` | float32 → float64; FP32 initial state cast to float64 | `Z4_20K__s1234__e153dd7abf` |
| mixed | `B2-E01-mixed-s1234` | strong → mixed (v = u_xx); 2-output head | arm trainer |
| modal (DIAGNOSTIC) | `B2-E01-modal-s1234` | u = A0 φ̂₁(x) q_θ(t), q learned; B1's temporal draws and initialisation | arm trainer |

Shared settings:
- 5,000 Adam steps × mini-batch 128 = **640,000 collocation (PDE) points per arm** (actual count);
- lr 1e-3·0.9^(s/1000);
- 640 points redrawn per epoch;
- tanh²(ω₁t) hard ansatz;
- seed 1234; 1 thread per process.

Execution details:
- All four arms ran **concurrently** on a 4-vCPU Xeon @ 2.8 GHz with torch 2.14.0, as declared in the
  pre-registration.
- Per-step times agree with the solo profile B2-PROF-001 to within 7 %: B1 164 vs 154 ms, mixed 75 vs 70 ms,
  modal 38 vs 36 ms. Concurrency therefore did not distort the cost comparison.
- No training used the exact reference solution; references were used only for evaluation.

## 2. Reproducibility gate (B1 re-run vs Batch-1 Z4 @ 5k)

**PASS, bit-exact.**

| | Batch 1 (ROOT, 2.1 GHz Xeon) | B2-E01 B1 (this run) |
|---|---|---|
| L2_exact | 0.5263868153173645 | **0.5263868153173645** |
| PDE residual | 0.11022354745212025 | **0.11022354745212025** |
| Persistence | 3.1543 cycles (collapse 0.15325 s) | **3.1543 cycles (0.15325 s)** |
| ω_fit / decay | 128.078 rad/s / 8.9814 1/s | 128.078 / 8.9814 |
| Training seconds | 593 s (solo) | 822 s (4 concurrent, other CPU) |

The identical run key and bit-identical metrics confirm that the PINN-RP import, the guard-patched I/O and the
new environment reproduce Batch 1 exactly. **All comparisons below use this re-run**, including its wall-clock.

## 3. Results (final models, 640,000 PDE evaluations each)

Source: `results_batch2/tables/B2-E01_RESULTS.csv`.

### 3.1 Accuracy and dynamics

The exact damped frequency is ω_d = 129.325 rad/s; the exact decay rate is 3.54 1/s; the full window is
20.58 cycles.

| Metric | B1 | B1-FP64 | mixed | modal (diag.) |
|---|---|---|---|---|
| L2_exact | 0.5264 | 0.5264 | 0.6750 | **0.2705** |
| L2_paper | 0.5264 | 0.5264 | 0.6750 | 0.2705 |
| late-window L2_exact (t ≥ 0.5 s) | 1.067 | 1.067 | 1.042 | 0.824 |
| **persistence, displacement [cycles]** | 3.15 | 3.15 | 1.67 | **7.62** |
| persistence, velocity [cycles] | 3.36 | 3.36 | 1.86 | 7.85 |
| collapse time [s] | 0.153 | 0.153 | 0.081 | 0.370 |
| fitted ω [rad/s] (error vs ω_d) | 128.08 (−0.96 %) | 128.08 (−0.96 %) | 126.73 (−2.0 %) | **129.28 (−0.04 %)** |
| fitted decay [1/s] | 8.98 | 8.98 | 13.24 | 5.40 |
| phase error [rad] | 0.018 | 0.018 | 0.029 | 0.033 |
| amplitude error (fit) | 0.075 | 0.075 | 0.027 | 0.027 |
| amplitude ratio (std) | 0.69 | 0.69 | 0.54 | 0.83 |
| max \|u_t\| ratio | 0.93 | 0.93 | 0.89 | 1.00 |
| **PDE residual (strong, of u)** | 0.110 | 0.110 | **2.57** | 0.074 |
| IC error (max, dimensionless) | 4.8e-6 | 8.9e-15 | 4.8e-6 | 4.8e-6 |
| BC error (max, dimensionless) | 1.4e-5 | 2.6e-14 | 1.4e-5 | 2.8e-5 |
| full-window persistence gate | FAIL | FAIL | FAIL | FAIL |

Mixed-only diagnostics, on the 51×501 grid:
- link residual RMS(v − u_xx)/RMS(u_xx,exact) = **0.056**;
- the mixed-form PDE residual RMS(c²v_xx + u_tt + γu_t)/RMS(u_tt) = **0.070**.

### 3.2 Compute (actual counts)

| Quantity | B1 | B1-FP64 | mixed | modal (diag.) |
|---|---|---|---|---|
| optimizer steps | 5,000 | 5,000 | 5,000 | 5,000 |
| PDE (collocation) evaluations | 640,000 | 640,000 | 640,000 | 640,000 |
| residual-vector evaluations | 640,000 | 640,000 | 1,280,000 (r_link + r_pde) | 640,000 (ODE) |
| forward passes (batched residual graph) | 5,000 | 5,000 | 5,000 | 5,000 |
| backward passes (training) | 5,000 | 5,000 | 5,000 | 5,000 |
| extra diagnostic backward passes* | 21 | 21 | 0 | 0 |
| highest network derivative order | 4 (x) | 4 (x) | 2 | 2 (t only) |
| **training wall-clock** | **822 s** | 1,404 s (1.71×) | **374 s (0.46×)** | 190 s (0.23×) |
| seconds per step | 0.164 | 0.281 | 0.075 | 0.038 |
| peak RAM (training) | 934 MB | 1,178 MB | 730 MB | 704 MB |
| peak VRAM | 0 (CPU only) | 0 | 0 | 0 |
| parameters | 241,601 | 241,601 | 242,002 | 241,601 |
| model size | 0.97 MB | 1.94 MB | 0.97 MB | 0.97 MB |
| inference (201×2001 grid) | 23.2 µs/point | 44.1 µs/point | 22.7 µs/point | 16.2 µs/point† |

\* B1's frozen per-term gradient-norm logging (`log_grad_norms`), unchanged from Batch 1 and excluded from
the wall-clock.

† Evaluated pointwise through the generic grid predictor. Reusing q(t) across x was **not** exploited, so this
is an upper bound for the modal model.

### 3.3 Trajectory vs PDE evaluations

Source: `results_batch2/tables/B2-E01_SNAPSHOTS.csv`. Each cell is L2_exact / persistence (cycles).

| PDE evaluations | B1 | B1-FP64 | mixed | modal (diag.) |
|---|---|---|---|---|
| 128,000 | 0.869 / 0.71 | 0.869 / 0.71 | 0.896 / 0.69 | 0.553 / 3.10 |
| 256,000 | 0.724 / 1.63 | 0.724 / 1.63 | 0.846 / 0.69 | 0.372 / 5.14 |
| 384,000 | 0.617 / 2.16 | 0.617 / 2.16 | 0.789 / 0.72 | 0.265 / 7.15 |
| 512,000 | 0.559 / 2.68 | 0.559 / 2.68 | 0.699 / 1.62 | **0.243 / 8.16** |
| 640,000 | 0.526 / 3.15 | 0.526 / 3.15 | 0.675 / 1.67 | 0.270 / 7.62 |

## 4. Pre-registered classification (`B2-E01_CLASSIFICATION.json`)

| Arm | Class vs B1 (§7.1) | ΔP [cycles] | L2e ratio | Dynamics diagnostics (§7.1) | W ratio | Case (§7.3) |
|---|---|---|---|---|---|---|
| B1-FP64 | COMPARABLE | 0.00 | 1.000005 | pass | 1.71 | **not material** (D not triggered) |
| mixed | **WORSE** | −1.49 | 1.28 | **fail**: frequency error 2.0 % > 2 % | **0.46** | **B** |
| modal | BETTER | +4.47 | 0.514 | pass | 0.23 | **partial** |

**Why modal is "partial" and not Case C.**
- Case C requires P_modal ≥ 2·P_B1 **and** L2e_modal ≤ 0.5·L2e_B1.
- The persistence criterion is met: 7.62 ≥ 6.31.
- The L2 criterion is missed narrowly: 0.2705 vs a threshold of 0.2632.
- H2-support requires P_modal ≤ 1.5·P_B1 = 4.73. It is not met either.

The pre-registered rule therefore gives "partial", and that label is kept.

**Case E** (all arms fail similarly): **not triggered**. The mixed and modal arms differ from B1 by more than
1 cycle.

## 5. Answers to the four questions (provisional, single seed)

**Q1. Does reducing the spatial derivative order improve training efficiency?**
- **Per step: yes.** The mixed arm needs 0.46× the wall-clock at matched evaluations (75 vs 164 ms/step) and
  0.78× the peak RAM.
- **Per unit of accuracy: no, not at this budget.**
  - At 640,000 evaluations the mixed arm is WORSE: persistence 1.67 vs 3.15 cycles, L2_exact 0.675 vs 0.526.
  - Its dynamics are also wrong: −2.0 % frequency, decay 13.2 vs 8.98 1/s (more over-damped).
  - At *equal wall-clock* the comparison is mixed at 374 s vs B1's snapshot nearest in time. B1 reaches
    ≈ 2,300 steps by 374 s, with L2 ≈ 0.70 and P ≈ 1.6–2.2 cycles between its 2k and 3k snapshots. Mixed at
    374 s gives L2 0.675 and P 1.67, so the two are **roughly on par per second**. This is inferred from
    snapshot interpolation, not a matched run.
  - The pre-registered Case-B follow-up (mixed at E' = 640,000 × 822/374 ≈ 1.41 M evaluations) has **not**
    been run.

**Q2. Is temporal propagation the dominant difficulty once spatial complexity is removed?**

**Partly, and it remains a major difficulty on its own.**
- With the spatial field removed entirely (exact mode shape; only q(t) learned), the same temporal
  representation, optimizer and budget give 2.4× the persistence (7.62 vs 3.15 cycles) and about half the L2.
- The frequency is almost exact (−0.04 %).
- **But the pure temporal problem still collapses at 0.37 s (7.6 of 20.6 cycles).** It stays over-damped
  (5.40 vs 3.54 1/s) and regresses between 512k and 640k evaluations (P 8.16 → 7.62, L2 0.243 → 0.270).

So both mechanisms contribute:
- spatial-field learning roughly halves the persistence reached per evaluation;
- a temporal-propagation limit remains when space is removed.

Observation, not a pre-registered test: the modal arm at 512k–640k evaluations reaches about the state Batch-1
Z4-20K reached with the full field at 2.56M evaluations (P 8.1, L2 0.263, decay 5.40 1/s), with ≈ 4× fewer
evaluations and ≈ 12× less training time (190 s vs Batch-1's 2,290 s; different machines).

**Q3. Does numerical precision materially affect the failure? No.**
- FP64 started from identical initial weights and follows the FP32 run's trajectory: after 5,000 steps the
  parameters differ by ≤ 1.5e-5 relative.
- L2_exact agrees to 5e-6 relative, and persistence, frequency and decay are identical to the reported digits.
- FP64 cost 1.71× the wall-clock and 1.26× the RAM.
- The only change is that the hard-constraint IC/BC errors drop from ~1e-5 to ~1e-14. This shows the FP32
  values are float32 rounding of the ansatz, not a physics violation.

For this Adam-trained, hard-constrained configuration, precision is **not** a contributing mechanism (Case D
not triggered). This differs from the FP64 mechanism reported for L-BFGS-trained failure modes (Xu et al.
2025), which this experiment did not test.

**Q4. Can a mathematically equivalent reformulation reach a better accuracy/compute point? Not at this budget.**
- On the (L2_exact, W) and (P, W) planes (`figures/B2-E01_pareto.png`), mixed is **cheaper but less accurate**:
  a different Pareto point that does not dominate B1. B1-FP64 is dominated by B1.
- The modal arm is the lowest-cost, most accurate point, but it solves a **reduced problem** (the exact mode
  shape is supplied). By pre-registration it is excluded from Q4 and is not evidence of PINN superiority.

## 6. Mechanistic observations (descriptive; they suggest hypotheses, they do not test them)

1. **The mixed arm satisfies its own equations but not the beam equation.** Its link and mixed-PDE residuals are
   small (5.6 %, 7.0 %), yet the strong residual of its displacement is 2.57, 23× B1's. Equivalence holds only
   when v = u_xx exactly; a 5.6 % L2 mismatch in v − u_xx is amplified by the two further x-derivatives that
   the strong form applies. The weak point of this formulation is therefore **derivative control of the
   auxiliary link**, not the cost of the residual. Any follow-up must state this as a new hypothesis and test
   it in isolation.
2. **The mixed training loss is unstable.** It shows spikes of up to ≈ 10× (`figures/B2-E01_convergence.png`), and
   the collapse front did not move for the first 384k evaluations (P ≈ 0.7). Both terms carry weight 1 after
   the pre-declared dimensional scaling, and no balancing was used.
3. **The collapse is front-like in every arm.** Correct oscillation is followed by an abrupt amplitude loss; it
   is not a uniform error (`figures/B2-E01_displacement_trace.png`, `B2-E01_velocity_trace.png`). The front
   advances steadily with evaluations for B1 (+0.044, +0.026, +0.025, +0.023 s per 128k), faster for modal
   (+0.100, +0.097, +0.049 s) until it retreats after 512k (−0.026 s) (`figures/B2-E01_collapse_front.png`).
4. **Over-damping is a common signature.** Every arm's fitted decay exceeds the physical 3.54 1/s (5.40–13.2).
   The fitted models put too much decay into the early cycles before the front.

## 7. Decision-tree outcome (pre-registered actions; nothing beyond them was executed)

| Case | Triggered? | Pre-registered action |
|---|---|---|
| A (mixed comparable + cheaper) | no | — |
| **B (mixed cheaper, less accurate)** | **yes** | Pareto trade-off quantified (§3–§5); not rejected. The recommended follow-up is the wall-clock-matched mixed run (E' ≈ 1.41M evaluations), **not run** |
| C (modal dramatically better) | no: L2 criterion missed by 2.8 % relative | — |
| modal "partial" | **yes** | report the ratio (P ×2.42, L2 ×0.514); stage-2 seeds recommended before concluding |
| D (FP64 material) | **no** | precision is not a required branch for this configuration |
| E (all similar) | no | — |

## 8. Recommendations for review (NOT executed; each needs approval)

1. **Stage-2 seeds (1235, 1236) for all four arms.** About 45 min of wall-clock running concurrently.
   Decision-relevant because the modal L2 sits 2.8 % from the Case-C threshold and the mixed loss is unstable.
2. **Pre-registered Case-B follow-up:** mixed at matched wall-clock (≈ 1.41M evaluations, ≈ 822 s). This tests
   whether the per-step saving buys accuracy.
3. **Before any temporal method (E03/E04):** the modal arm shows that the temporal-only problem collapses by
   itself. It is therefore the cheapest testbed (38 ms/step) for temporal-propagation mechanisms. **Proposed**:
   run future temporal-axis screens on the modal diagnostic first, then confirm on the full field. This is a
   proposal for review only, not an approved experiment.
4. **The mixed link-derivative observation (§6.1)** should be written as a separate hypothesis before any arm
   addresses it. For example, enforcing the link in a derivative-aware or weak sense would be a new
   single-change arm; candidates come from the literature (gPINN-type terms, weak forms), and none is novel.

None of the following were run, per instruction: additional optimizers, R3, temporal decomposition, SIREN,
SPINN, Mode 2, railway physics.

## 9. Limitations and deviations

- **Single seed.** The thresholds (1 cycle, 25 %) are ≥ 8× the Batch-1 seed spread, but stage-1 results remain
  provisional.
- **Concurrency.** The arms ran concurrently, as declared. FP64 trained alone for its last ≈ 9 min (01:46:45–01:55:46 UTC). Per-step
  times match the solo profile within 7 %.
- **`git_dirty=True` in the run registry.** At allocation time the only untracked files were the runs' own
  launch logs. The code was exactly commit `24813bf`.
- **Analysis re-run.** The first analysis run wrote the three tables and then failed at the figures because
  `matplotlib` was not installed in the venv. Those three files, produced seconds earlier by the same
  deterministic script, were deleted and regenerated after installing `matplotlib==3.11.2` (the Batch-1 pin).
  No result changed; the re-evaluation is asserted against each run's own `metrics.json` (≤ 1e-6 relative).
- **The modal arm is a diagnostic.** It uses the exact mode shape, which contains the FE-D-M1 solution, so its
  numbers must never be presented as full-field PINN results.
- **Figure overlap.** In the L2, front and convergence figures, the B1 and B1-FP64 curves coincide, so B1 is
  hidden under FP64. The tables give both.
- Inference latency for the modal arm does not exploit q(t) reuse (upper bound).

## 10. Files

| Path | Content |
|---|---|
| `docs/hypotheses/B2-E01.md` | pre-registration |
| `configs/batch2/B2-E01_mechanism_attribution.yaml` | spec |
| `results_batch2/runs/B2-E01-*-s1234/` | `record.json` (git SHA, config hash, environment, hardware), configs, histories, `metrics.json`, `step_{1..5}000.pt`, `final.pt`; launch logs |
| `results_batch2/tables/B2-E01_RESULTS.csv` | one row per arm |
| `results_batch2/tables/B2-E01_SNAPSHOTS.csv` | per-snapshot table |
| `results_batch2/tables/B2-E01_CLASSIFICATION.json` | pre-registered classification |
| `results_batch2/figures/B2-E01_L2_vs_pde_evals.png` | L2 vs PDE evaluations |
| `results_batch2/figures/B2-E01_runtime_vs_pde_evals.png` | wall-clock vs PDE evaluations |
| `results_batch2/figures/B2-E01_convergence.png` | training-loss convergence |
| `results_batch2/figures/B2-E01_displacement_trace.png` | mid-span displacement |
| `results_batch2/figures/B2-E01_velocity_trace.png` | mid-span velocity |
| `results_batch2/figures/B2-E01_pareto.png` | accuracy/persistence vs compute |
| `results_batch2/figures/B2-E01_collapse_front.png` | collapse front vs evaluations |
| `scripts/analyze_b2_e01.py` | analysis (evaluation only) |

**STOPPED. Waiting for review of B2-E01.**
