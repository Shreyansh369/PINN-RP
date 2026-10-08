# 24-hour sprint brief v2 — damping-bias law, windowed training, resource-aware window selection

Supersedes SPRINT_24H_CLAUDE_CODE.md (the normalised-loss fix was tested tonight and FAILED; see §1).
Run with Claude Code at the root of PINN-RP. Follow the repo's own rules: pre-register hypotheses in
docs/hypotheses/, add experiment IDs to configs/batch2/APPROVED_EXPERIMENTS.txt only with PI approval,
write only under results_batch2/.

---

## 0. The claim this sprint tests (Paper 1 candidate)

Q1 — Damping-bias law. If a PINN leaves an unresolved residual R (per unit amplitude, units s^-2) on a
damped mode with frequency w and modal decay zeta, the residual-loss minimiser has excess decay

    d*(T_w) = argmin_d  [ (d^2 + Delta)^2 + 4 w^2 d^2 + E ] * (1 - exp(-2 (zeta + d) T_w)) / (2 (zeta + d))

    long windows (zeta*T_w >> 1):  lambda* ~ sqrt(zeta^2 + (R / 2w)^2)          (full-window B1)
    short windows (zeta*T_w << 1): d*      ~ R^2 * T_w / (8 w^2)                 (time-marching)

Q2 — Design rule. Given a damping tolerance delta (1/s) and the measured R, the window length
T_w <= 8 w^2 delta / R^2 meets the tolerance; the compute cost grows with the number of windows T / T_w.
R is measurable WITHOUT the reference solution: R = rms(residual) / rms(field) over a window.

Q3 — Amplitude homogeneity sets the sign of the bias (exploratory evidence tonight, §1): a loss of
degree 2 in amplitude (standard) over-damps; dividing by window energy (degree 0) makes the solution grow.

Prior art (all verified tonight): AT-PINN (Chen, Lai, Yang, Thin-Walled Struct. 196, 2024) and
AT-PINN-HC (Chen et al., CMAME 436, 2025) do time-marching with hard constraints on Euler-Bernoulli beams
and report large error reductions, but give no relation between segment length and damping error.
Causal weighting (Wang et al.), R3 sampling (Daw et al., ICML 2023), Adam+L-BFGS conditioning (Rathore
et al., ICML 2024), GUA (arXiv 2609.01558), Per-Loss Adapters (arXiv 2605.10136), PINNForge
(arXiv 2609.23023) and Leiteritz & Pflueger (arXiv 2112.05620, penalty on max d/dt of the residual)
do not derive or use a damping-bias law. Energy-dissipation penalties exist for Allen-Cahn
(arXiv 2411.08760), KdV, EM waves (arXiv 2512.23396). NOVELTY IS NOT TIME-MARCHING. Novelty is Q1+Q2:
the law, its validation on a full-field PDE, and window selection from a reference-free measurement.

---

## 1. What was measured tonight (EXPLORATORY: seed 1234 only, scratch runs, not results of record)

Harness: homogeneity_prototype.py subclasses beampinn Trainer (read-only import), changes only the PDE
loss term. With the standard loss it reproduced the logged B2-E01 B1 L2 at every checkpoint to 4
digits (steps 1-2500). Budget for all rows: 5,000 steps x 128 = 640,000 PDE evaluations.

| Variant | L2_exact | persistence (cycles) | fit decay (true 3.54) | PDE_residual_rel | note |
|---|---|---|---|---|---|
| B1 (logged) | 0.526 | 3.15 | 8.98 | 0.110 | over-damped |
| normalised, degree 0 (p=1) | 5.48 @2500 | (20.58: metric blind to growth) | growing | - | amplitude ratio 1.8 / 9.3 / 61 at t = 0.05 / 0.4 / 1.0 s |
| fixed exponent p=0.5 | 0.818 | 2.22 | 10.20 | 0.143 | flat amplitude ratio 0.39 over 0.1-0.3 s, late spurious growth |
| energy-rate loss mu=1 | 0.929 @2500 | - | - | - | B1 was 0.654 @2500 |
| energy-rate loss mu=10 | 4.49 @2500 | - | - | - | collapsed to a static field (energy balance holds trivially) |
| feedback controller on p | 0.604 | 2.63 | 11.19 | 0.071 | sensor latched onto post-collapse growth; p driven to 0 |

Energy-balance sensor rel = (dE/dt + gamma int u_t^2)/E, 8-pt Gauss-Legendre in x, h = T/960 (exact
solution scores 0.06 1/s): B1 final -6.85, normalised run +7.13, static run -0.02. It is
reference-free but blind to static fields and contaminated after collapse.

Conclusions carried forward: (a) loss amplitude-homogeneity changes the learned damping in both
directions (Q3); (b) reweighting or energy terms in one global window did not beat B1; (c) the
remaining lever with a quantitative prediction is window length (Q1/Q2).

---

## 2. Pre-registration (write BEFORE any training): docs/hypotheses/B2-E05.md

- H-E05a: excess decay d = fit_lam - 3.54 decreases monotonically as T_w decreases (1.0, 0.25, 0.10, 0.05 s).
- H-E05b: with R measured per window from the field and residual alone, the predicted d*(T_w) from the
  formula in §0 matches the measured d within a factor of 2 for T_w <= 0.1 s.
- H-E05c: at an equal total budget (640k PDE evaluations), the best T_w beats B1 on decay error AND
  persistence AND L2 in 3/3 seeds.
- H-E05d (cost): wall-clock per PDE evaluation stays within 1.5x of B1.
- Falsifiers: if d does not decrease with T_w, or measured d differs from the prediction by more than
  5x, Q1 is rejected for this problem and Paper 1 falls back to the diagnostic manuscript.

---

## 3. Implementation (new files only; frozen code untouched)

src/physref/windowed.py — WindowedTrainer:

- Split [0, T] into N = T / T_w windows [t_k, t_{k+1}]. Train windows sequentially.
- Window ansatz (hard IC and BC exactly, same tanh^2 factor as B1, local time tau = t - t_k):

      u_k(x, t) = a_k(x) + tau * b_k(x) + tanh^2(w1 * tau) * Phi(x) * A0 * N_k(x, tau)

  with a_k = u_{k-1}(x, t_k), b_k = d/dt u_{k-1}(x, t_k) for k >= 1, and a_0 = u0(x), b_0 = 0.
  Represent a_k, b_k by projection onto the first 8 analytic fixed-fixed mode shapes (the E04 code in
  scripts/analyze_b2_e04.py has a stable ff_mode_stable); this gives exact 4th derivatives in x and
  satisfies the BCs. Check the projection error at each hand-off and log it.
- N_k = same architecture as B1 (st_fourier, depth 6, width 200, D4 Fourier), time input normalised to
  [0, T_w] (AT-PINN also normalises each segment). Initialise N_k from N_{k-1} (transfer). Same Adam +
  exp-decay schedule restarted per window ("reactivating optimiser" in AT-PINN; cite it).
- Budget: total 640,000 PDE evaluations split equally: steps_per_window = 5000 / N, mini-batch 128,
  collocation 640 points per window per epoch (same density per unit time as B1 is NOT preserved;
  log both and add one arm with density preserved if time allows).
- Evaluation: stitch the windows into one field u(x, t) and run the SAME evaluate_full + persistence as
  B1. Add the amplitude ratio R(t) at t = 0.1, 0.25, 0.5, 0.75, 1.0 s (persistence cannot see growth).
- Measured R per window: R_k = rms over window of (residual) / rms over window of (u), on a dense grid,
  computed from the trained window only (no reference). Log it.

configs/batch2/B2-E05_window_law.yaml — arms:
- W1000: T_w = 1.0 s (N=1; must reproduce B1 at 640k; harness check)
- W250, W100, W050: T_w = 0.25, 0.10, 0.05 s
- seeds 1234 first; then 1235, 1236 for W100 and W050 (and the best arm).

---

## 4. Order of work (hours are wall-clock on 1 CPU thread per run; ~15-25 min per run at 640k)

| Block | Hours | Task | Output |
|---|---|---|---|
| 1 | 0-1 | Write B2-E05.md hypotheses; implement WindowedTrainer + unit tests (IC/BC exact, hand-off continuity of u and u_t, W1000 == B1 bit-exact) | tests pass |
| 2 | 1-3 | Seed 1234: W1000, W250, W100, W050 (two at a time) | first d(T_w) curve |
| 3 | 3-4 | Compute predicted d*(T_w) from measured R_k (formula §0, scipy minimize on the window loss) vs measured d | prediction vs measurement plot |
| 4 | 4-8 | Seeds 1235, 1236 for W100, W050 (+ best) | 3-seed table |
| 5 | 8-10 | Pareto: decay error, persistence, L2 vs wall-clock and PDE evaluations for every arm | Pareto figure |
| 6 | 10-12 | Window selection rule: from a short pilot (first window only, 1/10 budget) measure R, choose T_w = 8 w^2 delta / R^2 for delta = 1 s^-1, run it, check the tolerance is met | rule validated or rejected |
| 7 | 12-24 | Manuscript: replace Sections 4-5 with the law, the T_w sweep and the rule; keep tonight's homogeneity runs as Section 6 (mechanism) | Paper 1 draft |

Decision gate after Block 3:
- PASS: d falls with T_w and the prediction is within 2x for T_w <= 0.1 -> Paper 1 = law + rule + 3-seed validation.
- PARTIAL: d falls with T_w but the prediction is off by 2-5x -> report as an empirical law with the
  model as a qualitative explanation; still a stronger paper than the diagnostic draft.
- FAIL: no monotone trend -> Paper 1 = the diagnostic manuscript (already drafted) + tonight's
  homogeneity results as the mechanism section.

---

## 5. Metrics table for every run (one CSV row per run)

arm, seed, T_w, N_windows, L2_exact, L2_late_exact, persistence_cycles, persistence_cycles_vel,
amp_ratio_t0.1/0.25/0.5/0.75/1.0, fit_w, fit_lam, decay_error = fit_lam - 3.54, frequency_error,
phase_error, PDE_residual_rel, R_k per window (mean, max), predicted_d*, train_seconds,
pde_evaluations, peak_rss_mb, parameters, handoff_projection_error_max.

---

## 6. What to tell the supervisor (only what the evidence supports)

- Proven tonight: the learned damping of this PINN moves in both directions with the amplitude scaling
  of its loss (seed 1234, exploratory); naive reweighting and energy penalties do not beat B1.
- Proposed and testable within 24 h: a damping-bias law that predicts how much time-marching reduces
  artificial damping, and a reference-free rule to choose window length for a damping tolerance and
  compute budget.
- Not claimed: an optimised PINN, until Block 4 passes in 3/3 seeds.
