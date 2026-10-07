# Novelty-gap analysis — Batch 2 (2026-10-07)

Standard (master prompt §13, §37):

- A combination of known techniques is **not** novel.
- Terminology is not novelty.
- A candidate is pursued only after answering the eight questions below.
- Identifiers refer to `docs/LITERATURE_MATRIX.md`.

**Caveat.** The audit read abstracts and search summaries; arxiv.org full texts were not reachable from this
environment. Every "remains unsolved" statement below is **provisional** until the closest papers have been read
in full: AT-PINN-HC (G8), A-PINN EB (E3), RO-PINN (F9), Auto-PINN/AutoPINN (L1–L2), PINNsAgent/Lang-PINN (L3–L4),
the 2026 method-selection review (L12), FP64 (A9) and the overfitting paper (L8).

---

## A. ESTABLISHED (use as baselines or components; never claim)

| Idea | Evidence |
|---|---|
| Fourier features / spatio-temporal multiscale Fourier PINN | F1, F2; it is the **B0** paper method |
| NTK-based loss weighting | B1; **B0** |
| Gradient-statistics weighting (LRA), GradNorm, SA-PINN | A8, B2, B4 |
| Residual-adaptive sampling (RAR, RAD, RAR-D) | C1, C2 (RAD also negative in Batch 1) |
| Hard constraints by distance / auxiliary functions | Sukumar & Srivastava 2022 (S), Lu et al. hPINN 2021 (S), D5 |
| Adam → L-BFGS | A2 |
| Mixed / auxiliary-variable formulations for high-order PDEs | E1, E2, E4, **E3 for the EB beam** |
| Weak / variational formulations | E5–E7 |
| Modal superposition and reduced-order PINNs | classical; F9, F10 |
| Temporal decomposition, time-marching, transfer learning between windows | G1–G5, **G7–G8 for structural vibration** |
| Causal residual weighting | B6 (rejected in Batch 1) |
| Nondimensionalization | D3, H2 |
| Forward-mode / Taylor-mode AD for PINN derivatives | I1, I2, I4 |
| Operator learning (DeepONet, FNO, PINO, PI-DeepONet) | K1–K4 |
| Benchmarks of PINN variants | L6 (PINNacle) |

## B. PARTIALLY EXPLORED

| Idea | What exists | What appears open (provisional) |
|---|---|---|
| Second-order / preconditioned optimizers for PINNs | NNCG, SOAP, ENGD, KFAC, SSBFGS (A2–A7) | behaviour on **long-window, high-frequency** dynamics with exact IC/BC (most benchmarks are short-window convection/wave) |
| Precision as a failure-mode cause | FP64 paper (A9): failure modes vanish with FP64 + L-BFGS | whether it explains **Adam-trained**, hard-constrained, **4th-order** collapse fronts (A9 uses L-BFGS on standard failure modes) |
| Conditioning diagnosis via condition numbers | D1 (theory), D2 (condition-number diagnosis + preconditioner) | a pre-training physics-invariant analysis tied to **strategy choice** and validated across PDE classes |
| Variable scaling + temporal segmentation | D4 (VS-PINN, STVS-PINN) | — (close to any "conditioning + decomposition" combination) |
| Frequency-informed hard-constraint time factors | D5 (case-by-case auxiliary functions), D6 (Batch-1 tanh²) | a stated **rule** (choose g with g''(0) ≍ ω₁² so that N* = O(1)) with a conditioning argument; small on its own |
| Automated PINN design | L1–L5 (search, AutoML, LLM agents) | selection from **measured invariants without trial-and-error**, with falsifiable rules and Pareto (accuracy vs compute) validation |
| "Propagation failure" as the failure mechanism | C3 (R3), G1, L8 (overfitting), A9 (precision) | a **quantitative** law of collapse-front advance vs physics budget, and attribution among the competing mechanisms on one controlled problem |

## C. OPEN / UNDEREXPLORED (provisional)

1. **Attribution of long-window dynamic failure to a mechanism** using *mathematically equivalent* reformulations
   at **matched compute** on the same problem, measured with a frozen persistence metric. The mechanisms are
   temporal propagation, 4th-order spatial operator conditioning, precision, optimizer and representation.
2. **Physics-invariant-driven, compute-aware strategy selection with pre-registered, falsifiable rules.** The
   invariants are dimensionless groups (N_cycles, ζ, β₁L), derivative order, coefficient spread, Fourier
   coverage and N* conditioning. The rules are validated out-of-sample across PDE tiers by accuracy-per-compute
   Pareto dominance, not best-case accuracy.
3. **Collapse-front speed per PDE evaluation** as a compute-scaling observable for dynamic PINNs (Batch 1:
   0.057 → 0.393 s over 1.6e5 → 2.56e6 evaluations, with a decelerating last increment).
4. A **verification-gated output contract** (Physics Reference Layer) that downstream AI systems consume. This is
   an engineering contribution, so its scientific novelty weight is low.

## D. CANDIDATE RESEARCH GAPS (each answers the eight questions)

### D1. Mechanism attribution of the dynamic collapse via equivalent reformulations at matched compute

1. **What is new?** A controlled decomposition of one failure (the Batch-1 collapse front) into contributions
   from:
   - the spatial operator order (strong vs mixed);
   - the spatial dimension (full field vs exact-mode-shape modal reduction, i.e. a pure temporal problem);
   - precision (FP32 vs FP64);
   - the optimizer (Adam vs Adam→L-BFGS).

   All arms are matched in PDE evaluations and scored with a frozen persistence metric.
2. **Closest prior work:** Krishnapriyan et al. 2021 (G1); Daw et al. 2023 (C3); Xu et al. 2025 (A9);
   Andersen & Matsubara 2026 (L8); Rathore et al. 2024 (A2).
3. **What they solve:**
   - G1: soft-constraint ill-conditioning; time marching helps.
   - C3: propagation failure; R3 sampling.
   - A9: FP32 + L-BFGS premature convergence.
   - L8: collocation overfitting.
   - A2: operator-induced ill-conditioning; second-order methods.

   Mostly on 1st/2nd-order convection, reaction and wave benchmarks with short windows.
4. **What remains:** no controlled attribution on a 4th-order, lightly damped, 20-cycle structural problem
   where the IC/BC are exact (hard), representation adequacy is proven (Batch-1 X3/X4) and budget scaling is
   measured (Batch-1 Y1-20K, Z4-20K).
5. **Why it matters:** the attribution decides which solver axis a compute-aware selector should spend budget
   on (temporal decomposition, formulation, precision or optimizer). Without it, any "adaptive selector" is a
   guess.
6. **How it can be falsified:** H2/H3/H4 in `docs/BATCH2_RESEARCH_HYPOTHESES.md`. Each hypothesis has a
   pre-registered prediction. For example, if the modal arm (no spatial operator) shows the same persistence
   per PDE evaluation as B1, then "4th-order spatial conditioning dominates" is falsified.
7. **Benchmark:** FE-D-M1 (Tier 1) first, then a 2nd-order wave equation with matched N_cycles (Tier 3), to
   separate "4th order" from "many cycles".
8. **Evidence that would make it publishable:**
   - ≥ 3 seeds per arm;
   - matched PDE evaluations **and** matched wall-clock;
   - an effect larger than the seed spread;
   - replication on ≥ 2 PDE classes;
   - mechanism-consistent intermediate measurements (front speed, gradient alignment, residual time
     profile);
   - all negative arms reported.

   Realistic venue class: an analysis paper (e.g. CMAME, JCP or a NeurIPS/ICML workshop) rather than a
   methods paper.

**Novelty risk: MEDIUM.** It is a diagnostic contribution, and its value depends on rigour and generality.

### D2. Physics-invariant-driven, compute-aware strategy selection (the "solver strategy" hypothesis)

1. **What is new?** The *selection procedure*, not a component. Before training, measurable invariants map to
   a recommended formulation, representation, temporal strategy, precision and optimizer through **explicit,
   pre-registered rules**. The rules are learned from controlled Tier-1/2 experiments and **tested
   out-of-sample** on unseen PDE tiers by whether the recommended strategy lies on (or near) the measured
   accuracy-vs-compute Pareto front.
2. **Closest prior work:**
   - Auto-PINN (L1) and AutoPINN (L2): search-based HPO/NAS, the latter resource-aware.
   - PINNsAgent (L3) and Lang-PINN (L4): LLM-agent selection with execution feedback and experience databases.
   - The Expert's guide (L7): a fixed best-practice recipe.
   - PINNacle (L6): empirical method comparison.
   - The 2026 method-selection review (L12).
3. **What they solve:**
   - L1–L5 find good configurations by trial-and-error (search or agent loops) and report accuracy.
   - L7 gives one recipe.
   - L6 compares methods without a predictive rule.
4. **What remains (provisional):**
   - whether *cheap pre-training measurements* predict the best strategy;
   - whether that prediction **saves** the search cost the others pay;
   - whether rules transfer across PDE classes.

   Compute accounting that charges the selector's own search cost is rarely reported.
5. **Why it matters:** this is the Physics Reference Layer use case. Many PDE instances must be solved under a
   budget, so search per instance is unaffordable at the edge and in fleet-scale railway monitoring.
6. **How it can be falsified:** if, on held-out PDEs, rule-selected strategies are not better on the Pareto
   front than either (a) the fixed L7 recipe or (b) random selection from the same strategy set at equal total
   compute (selection cost included), the hypothesis fails.
7. **Benchmark:** a tiered suite.
   - Tier 1: EB beam (FE-D-M1, then a Mode-2 stress test).
   - Tier 2: 1-D heat equation.
   - Tier 3: 1-D wave with N_cycles matched to the beam; Klein–Gordon.
   - Tier 4: Allen–Cahn or Burgers (PINNacle-compatible).
   - Tier 5: Timoshenko / moving load (railway).
8. **Evidence:**
   - ≥ 3 PDE tiers held out from rule fitting;
   - ≥ 3 seeds;
   - selection cost charged;
   - comparison against L7 and an AutoML baseline at equal compute;
   - released code and configs.

**Novelty risk: HIGH–MEDIUM.** The conceptual overlap with L1–L5 is real. The defensible difference is
predictive, invariant-based, search-free selection judged on compute-inclusive Pareto fronts. **No final name
is assigned until full-text review of L1–L5 and L12 confirms the difference.**

### D3. Collapse-front dynamics as a compute-scaling law (sub-goal of D1/D2)

1. **What is new?** Model the front position t_c(E) vs PDE evaluations E, test whether its speed depends on the
   invariants (N_cycles, ζ) and on the strategy, and use dt_c/dE as a budget predictor.
2. **Closest prior work:** C3 (R3 propagation); G3 (time sweeping with a loss tolerance); B6 (causal weights);
   scaling-law studies (arXiv:2603.12556, single-layer PINNs).
3. **What they solve:** C3/G3/B6 manipulate propagation but do not report a budget law.
4. **What remains:** a measured, predictive front-speed law.
5. **Why it matters:** it enables budget allocation ("how many evaluations to cover T?") inside the selector.
6. **How it can be falsified:** if the fitted law from 3 budgets mis-predicts a 4th budget by more than seed
   variability, it fails.
7. **Benchmark:** FE-D-M1 at 4 budgets × 3 seeds, then the wave equation.
8. **Evidence:** predictive validation on held-out budgets and problems.

**Novelty risk: MEDIUM.**

### D4. Verification-gated Physics Reference Layer contract

1. **What is new?** A machine-readable contract that refuses to label outputs "verified" unless declared
   physics gates pass (`src/physref/reference_layer.py`).
2. **Closest prior work:** digital-twin and UQ literature (not systematically searched in this audit).
3. **What they solve:** uncertainty quantification, data assimilation.
4. **What remains:** the integration standard for AI consumers.
5. **Why it matters:** the business objective.
6. **How it can be falsified:** a user study or a downstream-task benefit. It is not a scientific falsification.
7. **Benchmark:** downstream railway monitoring tasks (later).
8. **Evidence:** an engineering demonstration.

**Novelty risk: LOW as science.** Pursue as an engineering deliverable, not as a paper claim.

## E. TOO CLOSE TO EXISTING WORK (no novelty claim permitted)

| Idea | Too close to | Allowed use |
|---|---|---|
| "Mixed/auxiliary formulation for the EB beam" | E1, **E3 (A-PINN EB, 2026)** | arm in D1; efficiency measurement only |
| "Modal PINN t → q(t)" | classical modal analysis, **F9 (RO-PINN)** | diagnostic arm in D1 (it contains the exact solution for FE-D-M1, so it is not a fair competitor) |
| "Time-marching / temporal windows with transfer and continuity" | G1–G5, **G7–G8 (AT-PINN/-HC with EB beams)** | axis E03; G8 must be a comparator |
| "tanh²(ω₁t) time factor" as a method | **D5 (AT-PINN-HC auxiliary functions)**, D6 | conditioning rule inside D2 only |
| "Adaptive selection of PINN configuration" stated generically | L1–L5 | only the specific D2 form above |
| "R3 for beams" | C3 | sampling arm only |
| "Second-order optimizer for PINNs" | A2–A7 | optimizer arm only |
| "Nondimensionalization engine" | D3, H2, L7 | analyzer component (`physref.conditioning`) |
| Any "Fourier + NTK + X" stack | B0 + literature | forbidden (prompt §14, §36) |

## F. NOT WORTH PURSUING (now)

| Idea | Reason |
|---|---|
| New loss-weighting variant (NTK/LRA/GradNorm-like) | crowded field. B1 has a single loss term (hard constraints), so weighting cannot act there |
| Re-tuning RAD or calling RAD something else | tested in Batch 1 (Z1/Z3), no fix; renaming is forbidden |
| Causal weighting | rejected in Batch 1 (legacy REPORT §5.3) |
| Larger Fourier bandwidth to make Mode 2 easier | prompt §15.8; masks the representation variable |
| Nested `torch.func` forward-mode AD for this architecture | measured 4.9× slower than reverse mode (B2-PROF-001). Revisit only with JAX Taylor mode (I1) or separable structure (F6) |
| Mixed precision / quantisation now | no physically valid model exists yet (prompt §15.10); A9 suggests the opposite direction (more precision) |
| KAN-based PINNs | slower per parameter; no compute-aware motivation |
| ENGD at 2.4e5 parameters | O(P²) Gramian; infeasible on the current CPU budget |
| Railway models | prompt §21: only after a validated structural benchmark |

## Decision

Pursue **D1 first** (it is cheap, decisive and a prerequisite for D2), with D3 measured as a by-product.
Keep **D2** as the program-level hypothesis, judged at the end of Phase B2-13.
Treat D4 as engineering.

No component in sections A or E will be claimed as novel.
