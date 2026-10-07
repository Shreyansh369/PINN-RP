# Batch-2 readiness report — PINN-RP (2026-10-07)

**Status: PREPARATION COMPLETE. NO TRAINING EXECUTED. WAITING FOR PI APPROVAL.**

Training is mechanically blocked by `physref.gate`: `configs/batch2/APPROVED_EXPERIMENTS.txt` is empty.
Approving an experiment means adding its IDs to that file (for example `B2-E01-B1-s1234`,
`B2-E01-mixed-s1234`, `B2-E01-modal-s1234`).

---

## 0. Batch-1 frozen evidence inventory (read from committed reports; values verbatim)

| # | Batch-1 observation | Exact evidence | Source (`references/batch1/…`) |
|---|---|---|---|
| 1 | The published Fourier+NTK (C0) settles on a static field | L2 3.26 at 20k steps; amplitude ratio 0.06; 0 cycles | `reports/PHASE_A1_CHECKPOINT.md`, `tables/phaseA_summary.csv` |
| 2 | NTK weights become extreme | λ up to 8.5e14 / 9.9e14 (C0); 1e7–4e15 across D1–D4 | `reports/PHASE_A1_CHECKPOINT.md`, `reports/PHASE_D_DIAGNOSTIC.md` |
| 3 | Convention/initialisation variants do not recover 20.6 Hz | D1–D4 STATIC or LOW-FREQ; Y4 (C0+schedule) L2 2.75; Z3 (Y4+RAD) 2.60 | `PHASE_D_DIAGNOSTIC.md`, `PHASE_Y_…`, `PHASE_Z_DIAGNOSTIC.md` |
| 4 | The soft formulation admits static / low-dynamic modes | as rows 1 and 3 | same |
| 5 | Hard constraints fix IC/BC, but (t/T)² is ill-conditioned | IC/BC errors 4.8e-6 / 1.4e-5; required N* ≈ −8.4e3 | `PHASE_E_SCREEN.md` |
| 6 | tanh²(ω₁t) restores conditioning | N* ∈ [−1.9, −0.16]; X2 L2 0.909 vs E4b 1.61 | `PHASE_X_DIAGNOSTIC.md` |
| 7–8 | The representation can carry 20.6 Hz | supervised X3/X4 L2 0.463/0.316; Y3 0.120 (frequency error 2.7e-5) | `PHASE_X_…`, `PHASE_Y_…` |
| 9 | The LR schedule helps but is insufficient | Y1 0.774 vs X2 0.909 | `PHASE_Y_OPTIMIZER_DIAGNOSTIC.md` |
| 10 | RAD does not fix the collapse | Z1: 1.6 vs 1.2 cycles | `PHASE_Z_DIAGNOSTIC.md`, `PHASE_Y1_20K_…` |
| 11 | Seed does not change the failure mode | Z2: 1.1 cycles | same |
| 12 | More physics computation extends persistence | 1.2 → 3.1 → 3.2 → 8.1 cycles over 1.6e5 → 6.4e5 → 6.4e5 → 2.56e6 PDE evaluations. **Batch size per se is NOT established** (equal state at equal evaluations; only a 0.57× wall-clock advantage) | `PHASE_Y1_20K_…`, `PHASE_Z4_20K_BUDGET_DIAGNOSTIC.md` |
| 13 | Z4-20K | L2_exact 0.263; ω 129.2 rad/s (−0.1 %); decay 5.40 vs 3.54 1/s (over-damped); collapse 0.393 s = 8.1 of 20.6 cycles; R_pde 0.041; 2,290 s; the last 6.4e5 evaluations gave half the advance of the previous blocks | `PHASE_Z4_20K_BUDGET_DIAGNOSTIC.md` |
| 14 | No validated optimized PINN | "Full-window reproduction is not achieved. There is no validated optimized PINN." | same |

Additional frozen facts used in Batch 2:
- the reference-rounding floor L2_paper(exact) = 4.386e-4, which is 94.5 % of the 4.64e-4 target
  (`reports/STAGE01_CORRECTIONS.md` §3);
- the frequency-extractor resolution, p95 ≈ 1.4e-5 (same report, §4).

---

## Answers to the thirteen required questions

### 1. What does the current literature already establish?

Every individual mechanism on the Batch-2 list is published. That includes:
- Fourier features and NTK weighting;
- LRA, GradNorm, ReLoBRaLo, SA and ConFIG;
- RAR, RAD and R3;
- hard constraints and auxiliary-function ansätze;
- mixed and weak formulations;
- modal and reduced-order PINNs;
- time-marching and temporal decomposition with transfer and exact continuity;
- Adam→L-BFGS, NNCG, SOAP, ENGD and KFAC;
- forward- and Taylor-mode AD;
- FP64 as a fix for failure modes;
- operator learning.

For EB beams specifically, **auxiliary-variable PINNs (A-PINN EB 2026)**, **time-marching with hard
constraints (AT-PINN-HC 2025)**, **reduced-order PINNs (RO-PINN 2026)** and **operator learning (SpectONet
2026)** already exist.

Automated PINN design by search (Auto-PINN, AutoPINN) and by LLM agents (PINNsAgent, Lang-PINN, EvoPINN)
is active.

See `docs/LITERATURE_MATRIX.md`: 81 works, 65 confirmed by search.

### 2. What is the real unresolved problem?

Not "a better PINN trick". On a long-window, lightly damped, high-frequency, 4th-order dynamic problem with
exact IC/BC and proven representation adequacy, it is still unknown **which mechanism limits the solver**.
Batch 1 measured a collapse front that advances with physics budget. The candidates are:
- temporal propagation;
- spatial-operator conditioning;
- precision;
- optimizer;
- sampling.

Consequently it is also unknown **how to choose a solver strategy from measurable problem properties so that
compute is spent where it matters.** Today that choice is made by trial-and-error (search or agents) or by a
fixed recipe.

### 3. Which optimisation directions are genuinely promising?

1. Mechanism attribution by equivalent reformulation at matched compute (E01), together with a precision
   control (E02).
2. Temporal decomposition with exact (u, u_t) hand-over (E03), if E01 implicates propagation.
3. Propagation-tracking sampling, R3 (E04).
4. Second-order refinement (E05b), but after persistence is achieved.
5. Lower-order formulations, which are measured to be 2.2× cheaper per step (E01 mixed).

### 4. Which approaches are likely to improve…

| Target | Likely (to be tested) | Basis |
|---|---|---|
| accuracy | FP64 + L-BFGS refinement (E02/E05b); temporal decomposition (E03) | A9, A2, G7–G8 |
| stability | hard constraints (already in B1); exact window hand-over | Batch 1; G5 |
| training cost | mixed formulation (0.45× per step, measured); modal reduction where applicable (0.23×) | B2-PROF-001 |
| inference cost | modal or separable representations (q(t) reused across x) | F6, F9; to be measured |
| generalisation | operator learning for parametric families; invariant-based strategy rules | K1–K4; H0 |

### 5. Which approaches are redundant or low-value now?

- new loss-weighting schemes (B1 has a single loss term);
- re-tuning or renaming RAD;
- causal weighting (rejected in Batch 1);
- larger Fourier bandwidth for Mode 2;
- nested functorch forward-mode AD (measured 4.9× slower);
- ENGD at 2.4e5 parameters;
- KAN-PINNs;
- mixed precision or quantisation before physical validity;
- any "Fourier + NTK + more tricks" stack.

### 6. Which candidate methods are too close to existing literature?

| Candidate | Too close to |
|---|---|
| mixed EB formulation | A-PINN EB |
| modal PINN | RO-PINN, classical modal analysis |
| temporal windows with hard constraints | AT-PINN-HC |
| the tanh²(ω₁t) time factor | AT-PINN-HC auxiliary functions |
| "adaptive PINN configuration" in general terms | Auto-PINN / AutoPINN / Lang-PINN |
| any second-order optimizer | A2–A7 |

See `docs/NOVELTY_GAP.md` §E.

### 7. What exact research gap is defensible?

**D1 (near term, analysis contribution):** controlled attribution of long-window dynamic PINN failure on a
4th-order structural problem. It uses mathematically equivalent reformulations (strong / mixed / modal),
precision and optimizer arms at matched PDE evaluations and matched wall-clock, scored by a frozen
persistence metric, and is replicated on a 2nd-order wave problem with matched N_c.

**D2 (program-level, provisional):** pre-training, physics-invariant-driven strategy selection.
- It uses pre-registered, falsifiable rules.
- It is validated out-of-sample on accuracy-vs-compute Pareto fronts, with the selector's own cost charged.
- It is compared against a fixed best-practice recipe and AutoML/agent-style search at equal compute.

D2's distinctness from L1–L5/L12 is **provisional** pending full-text review.

### 8. What is the smallest first experiment capable of testing it?

**B2-E01, stage 1**: three arms × one seed × 6.4e5 PDE evaluations.
- **B1 re-run**: reproducibility check plus same-machine cost.
- **mixed**: max derivative order 2.
- **modal**: no spatial operator; same temporal representation.

All arms use the same seed, sampler, optimizer, schedule, precision and temporal representation. Estimated
cost is ≈ 45 min on one CPU core per arm group (§J). It directly decides H2 (propagation vs spatial operator)
and H3 (order reduction per unit compute).

### 9. What would constitute success?

For **E01**: a decisive, pre-registered classification of H2 and H3, reproduced at 3 seeds.

**Program success** for any method requires every one of the following on the full window:
- frequency correct;
- amplitude correct;
- damping plausible;
- velocity dynamics correct;
- **persistence over the full window** (20.58 cycles, displacement and velocity);
- low R_pde;
- IC/BC at the hard-constraint level;
- measured cost no worse than B1 at matched accuracy.

Then L2_paper < 4.64e-4 must hold **together with** L2_exact ≤ L2_paper, so that no pass can come from
reference rounding.

### 10. What would falsify the hypothesis?

- **H2** is falsified if the modal arm reproduces the full window (P = 20.58, L2e < 0.1) at 6.4e5 evaluations
  while B1 collapses.
- **H3** is falsified if mixed persistence is below B1's by more than the seed spread at matched evaluations.
- **H4** is falsified if FP64 at least doubles persistence.
- **H0/D2** is falsified if one strategy is Pareto-optimal on every tier, or if invariant-based rules do no
  better than the fixed recipe or than random choice at equal total compute.

See `docs/BATCH2_RESEARCH_HYPOTHESES.md`.

### 11. How could the methodology generalise beyond beams?

The analyzer is written for linear constant-coefficient PDEs Σ a_ij ∂_x^i ∂_t^j u = 0
(`physref.conditioning.LinearPDESpec`). Its invariants and diagnostics apply unchanged to:
- the heat equation (Tier 2: N_c is replaced by the number of diffusion times);
- wave and Klein–Gordon equations (Tier 3: N_c is directly comparable);
- linearisations of nonlinear PDEs (Tier 4, via PINNacle problems).

The formulation catalogue (strong / mixed / first-order / weak / modal) and the temporal strategies are
generic. Rules learnt on Tiers 1–2 are tested on Tiers 3–4.

### 12. How could it eventually support railway AI?

Through the Physics Reference Layer contract (`docs/PHYSICS_REFERENCE_LAYER_ARCHITECTURE.md`): railway
models consume **verification-labelled** physics references with cost metadata, never bare predictions.

Pathway:
1. EB → Timoshenko and beam-on-foundation (Kapoor et al. 2023);
2. moving loads (Kapoor et al. 2024);
3. sparse-sensor state estimation (Haywood-Alexander et al. 2024);
4. parameter identification for condition monitoring;
5. axle/bearing SHM with multi-sensor fusion.

Compute-aware selection matters at fleet scale and at the edge. **No railway claim is made now.**

### 13. What evidence is required before making a publication claim?

1. The gap is confirmed by **full-text** review of AT-PINN-HC, A-PINN EB, RO-PINN, Auto-PINN, AutoPINN,
   PINNsAgent, Lang-PINN, the 2026 method-selection review, FP64 and the overfitting paper.
2. A precise mathematical formulation (taxonomy §2–§3).
3. A reproducible implementation (frozen configs, run keys, manifests).
4. Controlled baselines B0 and B1, re-run on the same hardware.
5. One-change ablations.
6. ≥ 3 seeds, with effects larger than the seed spread.
7. ≥ 3 PDE classes, including a held-out tier.
8. Independent test conditions (unseen PDE parameters and windows).
9. Measured accuracy-vs-compute Pareto fronts, with search cost charged.
10. Robustness: seeds, budgets and precision.
11. Transparent limitations and negative results.
12. No novelty claim for any component in `NOVELTY_GAP.md` §A/E.
13. Released code, configs and provenance.

---

## BATCH-2 READINESS REPORT

### A. ROOT verification

| Item | Value |
|---|---|
| ROOT | `/home/user/PINN-replication` (https://github.com/Shreyansh369/PINN-replication) |
| Branch | `claude/tender-ramanujan-v3w90d` (unchanged) |
| Frozen commit (initial HEAD) | `d31864a868c59858f99b9b0c5dab4b0f18758e25` |
| Final HEAD | `d31864a868c59858f99b9b0c5dab4b0f18758e25` (`scripts/check_root_untouched.py`) |
| Status | clean before and after |
| Working-tree checksums | 486 files sha256, identical before and after (session check) |
| ROOT modified | **NO** |

### B. PINN-RP verification

- Existing repository with an empty history at start: branch `claude/tender-ramanujan-v3w90d`, no commits,
  remote https://github.com/Shreyansh369/PINN-RP.
- It was never re-initialised or re-cloned.
- Batch-2 commits B2-001 … B2-006 are on that branch.
- 149 files were imported from `d31864a`, with 5 declared modifications (`docs/ROOT_PROVENANCE.md`).
- Tests: 103 Batch-1 + 51 Batch-2 = **154 passed**.

### C. Literature frontier

The frontier is:
- second-order and preconditioned optimisation (SOAP, NNCG, KFAC, SSBFGS);
- precision and overfitting re-interpretations of failure modes (2025–2026);
- Taylor-mode derivatives (STDE);
- separable and operator models;
- domain-specific structural PINNs (AT-PINN-HC, A-PINN EB, RO-PINN, SpectONet);
- **automated, agentic PINN design** (Lang-PINN at ICLR 2026, PINNsAgent, EvoPINN).

### D. Existing methods

See the matrix: A1–L12 with status, compute implications and baseline suitability.

### E. Real research gaps

- D1: mechanism attribution at matched compute (pursue).
- D2: invariant-driven, search-free, compute-inclusive strategy selection (program-level; provisional).
- D3: a collapse-front budget law (by-product).
- D4: the verification-gated contract (engineering).

### F. Candidate research directions

E01 → (E02, E03, E04 or E02, E06, E07, decided by the E01 outcome) → E05 → E09 Pareto → E10 combination and
ablation → E11 cross-PDE → E12 railway transfer (`docs/BATCH2_EXPERIMENT_MATRIX.md`).

### G. Proposed solver architecture

Analyzer → conditioning → formulation → representation → training controller → solver → verification →
reference API.
- The analyzer, formulations (strong, mixed, modal), verification and API schema are implemented.
- The selectors are manual until H1 and H0 evidence exists.

See `docs/PHYSICS_REFERENCE_LAYER_ARCHITECTURE.md`.

### H. Publication strategy

1. **Paper 1 (analysis, D1 + D3):** "What limits long-window physics-informed training of structural
   vibration? A controlled, compute-matched attribution". Tier 1 + Tier 3 wave. Target: CMAME / JCP /
   Eng. Appl. AI, or an ML-for-science workshop first.
2. **Paper 2 (method, D2), only if H0/H1 hold:** invariant-driven compute-aware strategy selection across
   ≥ 4 PDE tiers vs recipe and AutoML/agent baselines.
3. **Paper 3 (application):** a railway Physics Reference Layer prototype after Tier 5 validation.

### I. First experiment

**B2-E01 stage 1**: arms B1 (re-run), mixed and modal; seed 1234; 5,000 steps × 128 = 6.4e5 PDE
evaluations each.
- Spec: `configs/batch2/B2-E01_mechanism_attribution.yaml`.
- Dry-run verified: the B1 arm reproduces the Batch-1 Z4 run key `f78aa7f1da`.

### J. Estimated computation

Measured per-step costs (B2-PROF-001) plus ≈ 10 % for the optimizer and sampling:

| Arm | Estimated training | Estimated evaluation (201×2001 + residual grid + persistence) |
|---|---|---|
| B1 | ≈ 14 min | ≈ 5 min |
| mixed | ≈ 7 min | ≈ 5 min |
| modal | ≈ 4 min | ≈ 5 min |

- **Stage 1 ≈ 40–45 min sequential**, or ≈ 20 min with 3 concurrent single-thread runs (concurrency declared).
- Stage 2 (2 more seeds): ≈ 1.5 h sequential.
- E02 (FP64): ≈ 20–40 min.
- Peak RAM: < 1.5 GB per run (Batch-1 measured 0.8–1.4 GB).

### K. Exact success / failure criteria for E01

1. **Reproducibility gate:** the B1 re-run matches Batch-1 Z4 within |ΔL2e| ≤ 5 % and |ΔP| ≤ 0.5 cycles,
   with the run key identical. Failing this means **STOP**.
2. **H2:**
   - supported if P_modal ≤ 1.5 × P_B1;
   - falsified if P_modal = 20.58 cycles and L2e_modal < 0.1;
   - otherwise partial, and stage 2 runs before any conclusion.
3. **H3:**
   - supported if P_mixed ≥ P_B1 − (seed spread) **and** W_mixed ≤ 0.75 W_B1 **and** IC/BC errors stay at
     the B1 level;
   - falsified if P_mixed < P_B1 − (seed spread).
4. Divergence of any arm is reported, with no re-tuning.
5. No metric or threshold changes after results are seen. Every decision is recorded in
   `results_batch2/reports/B2-E01_REPORT.md`, followed by a **STOP for approval**.

---

## Final repository verification (at the time of writing)

```
ROOT INITIAL SHA:   d31864a868c59858f99b9b0c5dab4b0f18758e25
ROOT FINAL SHA:     d31864a868c59858f99b9b0c5dab4b0f18758e25
ROOT MODIFIED:      NO
ROOT INITIAL STATUS: clean
ROOT FINAL STATUS:   clean
PINN-RP INITIAL SHA: none (repository existed with no commits)
TRAINING EXECUTED:  NO
```

The final SHAs and file lists are printed in the session's closing message.
