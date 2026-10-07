# Compute-cost model and accounting framework (Batch 2)

Goal (prompt §17): optimise for **physical accuracy + physics consistency + robustness + computational cost**.
Cost is always **measured**, never inferred from FLOP counts alone (prompt §15.9, §27).

## 1. Decision: no composite score yet

The literature review found no accepted weighting between error and compute for PINNs. Grossmann et al.
(L9) report error and time separately. Batch 2 therefore reports:

1. **Pareto fronts** (non-dominated sets; `physref.compute_cost.pareto_front`):
   - (L2e, W_train);
   - (−P, W_train);
   - (L2e, peak memory);
   - (L2e, inference µs/point);
   - (L2e, E).
2. **Matched-budget comparisons**: equal E (primary) and equal W (secondary) groups
   (`accuracy_at_matched_budget`).
3. **Hard physics gates** applied before any ranking. A model that fails the persistence or IC/BC gates is
   reported but never ranked as a solution.

A composite metric may be proposed later only with a stated justification. One acceptable form is a
user-specified budget constraint ("best L2e subject to W ≤ W_max"), which is a constrained choice on the
Pareto front, not invented weights.

## 2. Measured quantities (every major run)

| Quantity | Symbol | Definition | Where recorded |
|---|---|---|---|
| parameter count | P_θ | trainable parameters | metrics.json `parameters` |
| PDE evaluations | **E** | collocation residual points that entered a training loss (all formulations: one per point; the mixed formulation counts each point once although it evaluates 2 residuals) | `pde_evaluations` |
| residual evaluations by component | E_r | mixed: 2 per point (r_link, r_pde); modal: 1 ODE residual per t | arm logs |
| optimizer updates | S | Adam steps (plus L-BFGS iterations and line-search evaluations separately) | `optimizer_steps` |
| derivative order | k_max | highest network derivative in the residual | `derivative_order_max` |
| forward / backward passes | — | 1 forward + 1 backward through the derivative graph per step (Adam) | implied by S; L-BFGS counts reported |
| candidate evaluations | E_c | residuals evaluated only for sampling (RAD/R3) | `candidate_evaluations` |
| training wall-clock | **W** | time of training work only (steps + weighting + sampling updates); excludes validation, checkpointing and final evaluation | `train_seconds` |
| peak memory | M | peak RSS during training (separately for evaluation) | `peak_rss_mb`, `peak_rss_eval_mb` |
| inference latency | — | µs per point on the 201×2001 grid (chunked, no grad) and single-point median | `inference_us_per_point`, `inference_single_point_us` |
| checkpoint size | — | bytes of state_dict | `model_size_bytes` |
| hardware | — | CPU model, threads, RAM, GPU, torch version | record.json |
| precision | — | float32 / float64 | config |

## 3. Cost model used for planning (empirical, per-formulation)

    W ≈ S · ( c_res(form, k_max, P_θ, B) + c_opt(P_θ) + c_samp )  +  S/n_w · c_w

| Symbol | Meaning |
|---|---|
| B | mini-batch |
| c_res | residual + backward cost per step |
| c_opt | optimizer step (Adam ≈ O(P_θ), negligible here) |
| c_samp | sampling cost (≈ 0 for uniform redraw; ≈ one forward per candidate for RAD/R3) |
| c_w | cost of a weighting update every n_w steps (NTK: 0.67 s at 32 rows per term on 6×200, Batch-1 STAGE01 §7) |

**Measured c_res** (B2-PROF-001; Xeon @ 2.8 GHz, 1 thread, float32, B = 128, 6×200, m = 100; 30 repetitions,
median [p10, p90]):

| Formulation / strategy | P_θ | k_max | c_res | relative |
|---|---|---|---|---|
| B1 strong, nested reverse | 241,601 | 4 | 154 ms [145, 170] | 1.00 |
| B1 strong, nested forward (functorch JVP) | 241,601 | 4 | 750 ms [706, 809] | 4.86 |
| mixed (u, v), nested reverse | 242,002 | 2 | 70 ms [60, 89] | 0.45 |
| modal q(t), nested reverse | 241,601 | 2 (t) | 36 ms [33, 38] | 0.23 |

**Cross-check against Batch 1.** Z4 (B1 at 5k steps, B = 128) took 611 s = 122 ms/step on a 2.1 GHz Xeon,
including the optimizer and sampling. That is consistent in magnitude with c_res = 154 ms here: the ±20 %
machine jitter documented in Batch 1 and the different CPU/torch build make the per-machine comparison
approximate. **Planning estimates use this table. Every claim uses the runs' own measured W.**

**Limitation.** The profile records process-level max RSS only. Per-formulation peak memory will be measured
in E08 with one subprocess per case.

## 4. Inference cost (edge relevance)

| Model | Inference work per query (x,t) |
|---|---|
| full-field B1 | 2 Fourier encodings + 3 trunk passes (1 spatial + 2 temporal mappings) + head |
| modal | mode shape (closed form, a few transcendental calls) × q(t), where q needs 2 trunk passes on a scalar t |

For a fixed t, q(t) is computed once and reused for every x. Grid inference is then ≈ N_t trunk passes
instead of N_x·N_t. On the 201×2001 grid that is up to 201× fewer trunk passes. The real number is measured
in E01 (`inference_metrics`); the arithmetic above is not a claim.

## 5. Budget-matching rules

1. **Primary: matched E.** Arms differ in c_res, so matched E means different W, and both are reported.
2. **Secondary: matched W.** For an arm cheaper per step, a second run with E' = E·W_B1/W_arm tests whether
   the saving buys accuracy (H3 follow-up).
3. **Parameters matched within ±5 %** for representation comparisons (E07).
4. **Selection or search cost is charged.** For H0/D2, the selector's own trial runs are added to W.
5. **Concurrency.** Timed comparisons run either solo or with the concurrency level declared. The per-step
   median from the same setting is used. Batch 1 measured no per-step slowdown at 4 concurrent 1-thread
   runs, but this is re-checked on the new machine.

## 6. Accuracy-side quantities (gates first, then ranking)

| Quantity | Role |
|---|---|
| persistence P (displacement and velocity) | **gate** (full window) |
| IC / BC errors | **gate** (≤ 1e-4 dimensionless for hard constraints; soft forms report the value) |
| L2_exact, L2_paper (both always reported) | ranking |
| R_pde | ranking / consistency |
| frequency, decay, phase, amplitude errors | diagnostics (extractor p95 ≈ 1.4e-5 at target-level error) |

## 7. Compute claims we will NOT make

- A speed-up from theoretical FLOPs or derivative-order arguments without a measured W on stated hardware.
- A speed-up of a model that fails a physics gate ("a faster physically wrong model is not an improvement").
- A cross-machine wall-clock comparison without re-running the baseline on the same machine.
