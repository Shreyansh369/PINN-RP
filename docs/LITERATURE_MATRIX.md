# Literature matrix — Batch 2 audit (2026-10-07)

Scope: the methods relevant to a **compute-aware, physics-aware, problem-adaptive** neural PDE solver,
with the damped Euler–Bernoulli beam (4th order in x, 20.6 cycles, ζ = 0.027) as the first test case.

**How to read the identifier column**

- **S**: title, authors and venue/arXiv ID were confirmed by a web search in this audit.
- **M**: from bibliographic memory, not re-confirmed here. Check the DOI before citing it in a manuscript.

arxiv.org itself was not reachable from this environment (egress policy), so no PDF was read in full.
"Reported" numbers are taken from abstracts or search summaries and are **not** reproduced by us.

**Relevance codes:** H = high, M = medium, L = low.
- **HF**: high-frequency / long-time dynamics.
- **4th**: fourth-order PDEs.
- **Edge**: edge inference.
- **Gen**: general-purpose solving.

**Status codes:**
- **EST**: established (widely used, multiple independent follow-ups).
- **PART**: partially explored.
- **NEW**: recent; few or no independent replications.

**Baseline?**: should this be a Batch-2 baseline or comparator?

---

## A. Optimization

| ID | Method | Authors, year | Venue / identifier | PDEs tested | Principal idea | Compute implication | Reported result (abstract-level) | Code | HF | 4th | Edge | Gen | Status | Baseline? | Novelty risk if we "propose" it |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | Adam | Kingma & Ba, 2015 | ICLR 2015, arXiv:1412.6980 (M) | generic | adaptive first-order moments | O(P) per step | — | yes | — | — | — | H | EST | yes (B0/B1 use it) | n/a |
| A2 | Adam → L-BFGS; NysNewton-CG (NNCG) | Rathore, Lei, Frangella, Lu, Udell, 2024 | ICML 2024, PMLR 235:42159; arXiv:2402.01868 (S) | convection, reaction, wave | the PINN loss is ill-conditioned through the differential operator; second-order methods help | L-BFGS: ~m·P memory; NNCG: Hessian-vector products (≈2–4× a gradient each) | Adam+L-BFGS beats either alone; NNCG improves further | yes | M | M | L | H | EST (Adam+L-BFGS) / PART (NNCG) | Adam→L-BFGS: **yes** | very high |
| A3 | MultiAdam | Yao, Su, Hao, Liu, Su, Zhu, 2023 | ICML 2023, PMLR 202:39702; arXiv:2306.02816 (S) | 2nd-order PDEs | per-loss-term, parameter-wise scale-invariant moments | ≈ Adam × number of loss terms (memory) | 1–2 orders of magnitude over baselines | yes | M | L | L | M | PART | candidate (multi-term arms only) | high |
| A4 | SOAP + gradient-alignment analysis | Wang, Bhartari, Li, Perdikaris, 2025 | NeurIPS 2025; arXiv:2502.00604 (S) | 10 benchmarks incl. turbulent flow | directional gradient conflicts between terms; SOAP approximates a Newton preconditioner | Shampoo-type preconditioner; periodic eigendecompositions per layer | 2–10× accuracy over Adam | yes (jaxpi) | H | L | L | H | NEW | candidate | high |
| A5 | Energy natural gradient (ENGD) | Müller & Zeinhofer, 2023 | ICML 2023, PMLR 202:25471; arXiv:2302.13163 (S) | Poisson, heat | Newton-like step in function space (Gramian of the PDE operator) | O(P²) Gram + linear solve: limited to small nets | errors orders of magnitude below Adam at equal time (small nets) | yes | L | M | L | M | PART | no (cost at 2.4e5 params) | high |
| A6 | KFAC for PINNs | Dangel, Müller, Zeinhofer, 2024 | NeurIPS 2024; arXiv:2405.15603 (S) | various | Kronecker-factored Gauss–Newton that includes the differential operator (via Taylor mode) | per-layer factor inversions; scales to larger nets than ENGD | reaches ENGD-level accuracy at larger scale | yes | M | M | L | H | NEW | later candidate | high |
| A7 | Self-scaled BFGS / Broyden | Kiyani, Shukla, Urbán, Darbon, Karniadakis | arXiv:2501.16371 (S, ID from search index); related arXiv:2604.05230 (S) | Burgers, Allen–Cahn, KS, Ginzburg–Landau, Helmholtz, stiff ODEs | quasi-Newton with history-based rescaling | ≈ L-BFGS | orders of magnitude without adaptive weights | partial | M | L | L | H | NEW | candidate (with A2) | high |
| A8 | Learning-rate annealing (gradient statistics) | Wang, Teng, Perdikaris, 2021 | SIAM J. Sci. Comput. 43(5):A3055; arXiv:2001.04536 (S); DOI 10.1137/20M1318043 (M) | Helmholtz, Klein–Gordon, flow | balance term weights by gradient-magnitude ratios | one extra gradient per term per update | large gains on stiff multi-term losses | yes | M | L | L | M | EST | as a loss-balancing comparator | very high |
| A9 | FP64 rescues failure modes | Xu et al., 2025 | NeurIPS 2025; arXiv:2505.10949 (S) | convection, reaction, wave ("failure modes") | FP32 makes L-BFGS stop prematurely (spurious "failure phase"); FP64 removes the failure modes | ≈2–4× slower on GPUs; ~1–2× on CPU | vanilla PINN + FP64 + L-BFGS solves the failure modes | yes | **H** | M | L (opposite direction) | H | NEW | **yes: precision arm** | n/a (we test it, never claim it) |

## B. Loss balancing

| ID | Method | Authors, year | Venue / identifier | Principal idea | Compute implication | Reported | Code | HF | 4th | Gen | Status | Baseline? | Notes for us |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | NTK-based weighting | Wang, Yu, Perdikaris, 2022 | J. Comput. Phys. 449:110768; arXiv:2007.14527 (S); DOI 10.1016/j.jcp.2021.110768 (M) | equalise convergence rates by NTK-trace ratios | per-row Jacobian traces; superlinear in rows (Batch-1 measured 0.67 s per update, 6×200) | fixes wave-equation imbalance | yes | M | L | M | EST | **B0 uses it** | Batch 1: weights reached 1e7–1e15; static attractor |
| B2 | GradNorm | Chen, Badrinarayanan, Lee, Rabinovich, 2018 | ICML 2018, PMLR 80; arXiv:1711.02257 (S) | tune weights so that per-task gradient norms train at equal relative rates | one gradient per task per step | multitask vision | yes | L | L | M | EST | comparator only | not PINN-specific |
| B3 | ReLoBRaLo | Bischof & Kraus | arXiv:2110.09813 "Multi-Objective Loss Balancing for Physics-Informed Deep Learning" (S) | softmax of relative loss progress, with random lookback to the initial losses | negligible (loss values only) | competitive with LRA/GradNorm at lower cost | yes | L | L | M | PART | comparator | cheap, so useful for a compute-aware comparison |
| B4 | Self-adaptive PINN (SA-PINN) | McClenny & Braga-Neto, 2023 | J. Comput. Phys. 474:111722; arXiv:2009.04544 (S) | per-point trainable weights maximised (min-max) | +1 parameter per collocation point | fewer epochs, lower L2 on benchmarks | yes | M | L | M | EST | comparator | conflicts with epoch redraw (weights are tied to fixed points) |
| B5 | ConFIG (conflict-free updates) | Liu, Chu, Thuerey, 2025 | ICLR 2025; arXiv:2408.11104 (S) | update direction with positive dot product with every term's gradient | one gradient per term; small linear solve | better accuracy and runtime than weighting | yes (pip `conflictfree`) | M | L | H | NEW | candidate (multi-term arms) | acts on directions, not weights |
| B6 | Causal weighting | Wang, Sankaran, Perdikaris, 2024 | CMAME 421:116813; arXiv:2203.07404 (S); DOI 10.1016/j.cma.2024.116813 (M) | time-ordered residual weights w_i = exp(−ε Σ_{k<i} L_k) | temporal binning of the residual | chaotic and turbulent benchmarks | yes | **H** | L | H | EST | — | **rejected in Batch 1 (legacy REPORT §5.3)**; not re-proposed |

## C. Adaptive sampling

| ID | Method | Authors, year | Venue / identifier | Principal idea | Compute implication | Code | HF | Status | Notes for us |
|---|---|---|---|---|---|---|---|---|---|
| C1 | RAR | Lu, Meng, Mao, Karniadakis, 2021 (DeepXDE) | SIAM Review 63(1):208; arXiv:1907.04502 (M) | add points where the residual is largest | candidate-residual evaluations | yes | M | EST | — |
| C2 | RAD, RAR-D | Wu, Zhu, Tan, Kartha, Lu, 2023 | CMAME 403:115671; arXiv:2207.10289 (S) | resample from p ∝ \|r\|^k / mean + c | ~10⁴ candidate residuals per update | yes | M | EST | **RAD tested in Batch 1 (Z1/Z3): no fix.** Never renamed or re-proposed |
| C3 | R3 (retain–resample–release), causal R3 | Daw, Bu, Wang, Perdikaris, Karpatne, 2023 | ICML 2023, PMLR 202; arXiv:2207.02338 (S) | keep high-residual points and resample the rest; frames failure as *propagation* from IC/BC | no candidate pool; residuals of the current points only | yes | **H** | EST | conceptually matches the Batch-1 "collapse front"; implemented in `physref/sampling.py` (non-causal) |
| C4 | Time-sweeping collocation | Penwarden, Jagtap, Zhe, Karniadakis, Kirby, 2023 | J. Comput. Phys. (2023); arXiv:2302.14227 (S) | advance the collocation front with a loss tolerance | fewer points per iteration | yes | H | PART | see G3 |

## D. Mathematical conditioning

| ID | Method | Authors, year | Venue / identifier | Principal idea | Code | HF | 4th | Status | Notes for us |
|---|---|---|---|---|---|---|---|---|---|
| D1 | Operator-preconditioning view of PINN training | De Ryck, Bonnet, Mishra, de Bézenac, 2024 | ICLR 2024; arXiv:2310.05801 (S) | GD rate is governed by the conditioning of the Hermitian square of the PDE operator (for 4th order, effectively order 8) | — | M | **H** | PART | theory that predicts why u_xxxx hurts; motivates mixed/modal arms |
| D2 | Preconditioning for PINNs (condition-number diagnosis) | Liu et al. (Tsinghua group), 2024 | arXiv:2402.00531 (S), ICML 2024 (M) | diagnose with the condition number; precondition the discretised operator | yes | M | M | PART | closest prior art to a "conditioning analyzer" |
| D3 | Nondimensionalization / term scaling | many; e.g. Kapoor et al. 2023 (nondimensional EB/Timoshenko); 2025 "direct term scaling" (S, details unverified) | — | scale variables and terms by characteristic scales | — | M | M | EST | standard practice; never a novelty |
| D4 | Variable-scaling PINNs (VS-PINN); STVS-PINN | Ko & Park 2024, arXiv:2406.06287 (S: ID; authors M); STVS-PINN (Springer chapter, 2025/26) (S) | — | stretch coordinates to reduce gradient stiffness; STVS adds temporal segmentation | partial | M | L | PART | combines scaling and temporal segmentation: **closest to a conditioning + decomposition combination** |
| D5 | Hard-constraint auxiliary functions for vibration (AT-PINN-HC) | Chen et al., 2025 | CMAME (2025) 117691; DOI 10.1016/j.cma.2024.117691 (S) | case-by-case auxiliary functions (trig for BC displacement, exponential for IC displacement/velocity) + time-marching | — | **H** | **H** | NEW | **closest prior art to the Batch-1 tanh²(ω₁t) time factor**; includes an EB beam |
| D6 | Batch-1 time-factor conditioning (t/T)² → tanh²(ω₁t) | this project, Batch 1 | `references/batch1/reports/PHASE_X_DIAGNOSTIC.md` | required output N* ≈ −ω²/g''(0): −8.4e3 → −0.5 | yes | H | H | internal | prior evidence only; not claimed (prompt §15.1) |
| D7 | Exact BC imposition with distance functions | Sukumar & Srivastava, 2022 | CMAME 389:114333 (vol. M); arXiv:2104.08426 (S) | trial function = (approximate distance function) × network, via R-functions and transfinite interpolation; the loss then contains only the residual | yes | M | M | EST | foundation of the B1 hard ansatz family |
| D8 | hPINN (hard constraints via penalty / augmented Lagrangian) | Lu, Pestourie, Yao, Wang, Verdugo, Johnson, 2021 | SIAM J. Sci. Comput. 43(6):B1105 (vol. M), DOI 10.1137/21M1397908; arXiv:2102.04626 (S) | constraints enforced by penalty/AL in inverse design | yes | L | L | EST | — |
| D9 | Scaling laws and pathologies of single-layer PINNs | 2026 | arXiv:2603.12556 (S) | empirical scaling laws; optimisation, not approximation, is the bottleneck | ? | M | L | NEW | supports the Batch-1 "optimisation not capacity" finding |

## E. PDE reformulation

| ID | Method | Authors, year | Venue / identifier | PDEs | Principal idea | Code | 4th | Status | Notes for us |
|---|---|---|---|---|---|---|---|---|---|
| E1 | Deep mixed residual method (MIM) | Lyu, Zhang, Chen, Chen, 2022 | J. Comput. Phys. 452:110930 (M) | high-order elliptic incl. biharmonic | write a high-order PDE as a lower-order system; the network outputs u and its derivatives | yes | **H** | EST | mixed formulation = established |
| E2 | Auxiliary PINN (A-PINN) | Yuan, Ni, Deng, Hao, 2022 | J. Comput. Phys. 462:111260 (M) | integro-differential | auxiliary outputs replace integrals and derivatives | yes | M | EST | — |
| E3 | A-PINN for continuous EB beams | (authors unverified), 2026 | arXiv:2601.00866; Applied Soft Computing 2026 (S) | EB beam vibration | auxiliary outputs for lower-order derivatives + adaptive Adam/L-BFGS switching | ? | **H** | NEW | **closest prior art to our mixed arm**; reports ~40 % MSE reduction and 30–40 % fewer epochs vs PINN |
| E4 | Fourier heuristic PINN for biharmonic (coupled scheme) | 2025 | arXiv:2509.15004 (S) | biharmonic | Δu = v coupling + Fourier features | ? | H | NEW | — |
| E5 | VPINN, hp-VPINN | Kharazmi, Zhang, Karniadakis 2019; 2021 | arXiv:1912.00873 (M); CMAME 374:113547, arXiv:2003.05385 (S) | elliptic, advection–diffusion | Petrov–Galerkin weak form; test functions take derivatives off the network | yes | **H** (integration by parts halves the order) | EST | weak form for u_xxxx needs only u_xx·φ_xx |
| E6 | Weak adversarial networks (WAN) | Zang, Bao, Ye, Zhou, 2020 | J. Comput. Phys. 411:109409 (M) | high-dimensional | adversarial test functions | yes | M | EST | — |
| E7 | Deep Ritz | E & Yu, 2018 | Commun. Math. Stat. 6:1 (M) | variational (static) | energy minimisation | yes | H (static beams) | EST | not directly for damped dynamics |

## F. Representation

| ID | Method | Authors, year | Venue / identifier | Idea | Compute | Code | HF | Edge | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| F1 | Random Fourier features | Tancik et al., 2020 | NeurIPS 2020; arXiv:2006.10739 (M) | input embedding fights spectral bias | +2m inputs | yes | H | M | EST | — |
| F2 | Spatio-temporal multiscale Fourier PINN | Wang, Wang, Perdikaris, 2021 | CMAME 384:113938; arXiv:2012.10047 (S) | separate x/t embeddings, multiscale σ; NTK eigenvector-bias analysis | — | yes | H | M | EST | **B0/B1 architecture (paper Eqs. 38–43)** |
| F3 | SIREN | Sitzmann et al., 2020 | NeurIPS 2020 (S) | sine activations with principled initialisation | ≈ MLP | yes | H | M | EST | candidate representation arm |
| F4 | sf-PINN (sinusoidal spaces) | Wong, Ooi, Gupta, Ong | IEEE Trans. Artif. Intell. (2022); arXiv:2109.09338 (S) | first-layer sinusoidal mapping | ≈ MLP | yes | M | M | EST | — |
| F5 | Adaptive activations | Jagtap, Kawaguchi, Karniadakis, 2020 | J. Comput. Phys. 404:109136 (M) | trainable activation slope | +1 parameter per layer | yes | M | M | EST | low priority |
| F6 | Separable PINN (SPINN) | Cho, Nam, Yang, Yun, Hong, Park, 2023 | NeurIPS 2023; arXiv:2306.15969 (S) | per-axis networks, outer-product field, forward-mode AD | 62× wall-clock, 1394× FLOPs vs PINN at the same collocation count | yes | M | **H** | EST | strong efficiency comparator; separable ansatz ≈ learned modal basis |
| F7 | PirateNets | Wang, Li, Chen, Perdikaris, 2024 | JMLR 25(402):1–51; arXiv:2402.00326 (S) | adaptive residual connections; initialised shallow | ≈ MLP | yes (jaxpi) | M | L | NEW | — |
| F8 | KAN-based PINNs (PIKAN) | multiple; unified benchmark 2026 | arXiv:2602.15068 (S) | learnable univariate edge functions | slower per parameter | partial | M | L | NEW | low priority for compute-aware goals |
| F9 | Modal / reduced-order PINN (RO-PINN) | Zhang, Vlachas, Chatzi, 2026 | arXiv:2608.17131 (S) | embed reduced (modal) equations in the PINN loss; adaptive basis | very low dimension | ? | H | **H** | NEW | **closest prior art to the modal arm**; modal integration is NOT our novelty |
| F10 | ModalPINN (periodic flows) | (authors M) | (M) | modal decomposition in the architecture | — | ? | H | H | PART | — |

## G. Temporal / spatial decomposition

| ID | Method | Authors, year | Venue / identifier | Idea | Code | HF | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| G1 | Curriculum regularisation; seq2seq time marching | Krishnapriyan, Gholami, Zhe, Kirby, Mahoney, 2021 | NeurIPS 2021; arXiv:2109.01050 (S) | the soft-constraint loss is hard to optimise, not under-expressive; march in time | yes | H | EST | supports "optimisation, not capacity" (Batch-1 X3/X4 agree) |
| G2 | bc-PINN (backward compatible) | Mattey & Ghosh, 2022 | CMAME 390:114474 (M) | retrain one net on successive segments while keeping earlier ones | yes | H | EST | — |
| G3 | Stacked decomposition + transfer learning + time sweeping | Penwarden et al., 2023 | J. Comput. Phys.; arXiv:2302.14227 (S) | unify time-marching PINNs and XPINNs | yes | H | EST | — |
| G4 | XPINN | Jagtap & Karniadakis, 2020 | Commun. Comput. Phys. 28(5) (M) | space-time domain decomposition with interface losses | yes | M | EST | — |
| G5 | Exact temporal continuity in sequential PINNs | Roy, Castonguay (M), 2024 | CMAME (2024), S0045782524004535 (S: venue) | hard-enforce the window IC from the previous window | ? | H | NEW | our `interface="hard_ic"` follows this idea |
| G6 | FBPINN; multilevel FBPINN | Moseley, Markham, Nissen-Meyer 2023; Dolean, Heinlein, Mishra, Moseley 2024 | Adv. Comput. Math (M); CMAME; arXiv:2306.05486 (S) | overlapping partition-of-unity subdomain networks; multilevel for global communication | yes | **H** | EST | strong for high frequency |
| G7 | AT-PINN (advanced time-marching) | Chen et al., 2024 | Thin-Walled Structures (S0263823123009011) (S) | per-segment normalisation, reactivating optimiser, transfer learning, sine activation | ? | **H** | NEW | structural vibration, long durations |
| G8 | AT-PINN-HC | Chen et al., 2025 | CMAME 117691 (S) | G7 + hard constraints with auxiliary functions | ? | **H** | NEW | **EB beam included; reports 1–4 orders of magnitude lower error, up to 78 % fewer iterations.** Must be a comparator for any temporal-decomposition claim |
| G9 | Pre-training for evolution equations | (2022) | arXiv:2212.00798 (S) | pre-train on short time, extend | ? | M | PART | — |

## H. Structural and mechanical PINNs

| ID | Work | Authors, year | Venue / identifier | Physics | Key point | Relevance |
|---|---|---|---|---|---|---|
| H1 | **Target paper** | Söyleyici & Ünver, 2025 | EAAI 141:109804; DOI 10.1016/j.engappai.2024.109804 (S, ROOT) | EB beam SS/FE/CF, damped/undamped, parameter ID | Fourier + NTK | B0; Batch 1 could not reproduce it at short budgets |
| H2 | Complex beam systems | Kapoor, Wang, Núñez, Dollevoet, 2023 | IEEE TNNLS 35(5):5981; DOI 10.1109/TNNLS.2023.3310585 (S) | EB, Timoshenko, double beam on a Winkler foundation | **nondimensional** equations; forward error < 1e-3 % | railway-relevant group (TU Delft); nondimensionalization baseline |
| H3 | Moving-load PIML | Kapoor et al., 2024 | J. Phys.: Conf. Ser.; arXiv:2304.00369 (S) | beam under a moving load | Dirac load approximated by a Gaussian | railway transfer path |
| H4 | A-PINN EB beam | 2026 | arXiv:2601.00866 (S) | EB vibration | auxiliary lower-order outputs | closest to the mixed arm |
| H5 | AT-PINN / AT-PINN-HC | Chen et al., 2024/2025 | see G7/G8 (S) | EB beam, panels, plates | time-marching + hard constraints | closest to the decomposition axis |
| H6 | SpectONet | 2026 | arXiv:2607.25790 (S) | EB beam operator learning | physics-guided DeepONet with Chebyshev–Gauss–Lobatto sensors; real bridge data | operator direction for beams |
| H7 | Response estimation and system ID | Haywood-Alexander et al., 2024 | arXiv:2410.01340 (S) | MDOF / SHM | PINNs for state and parameter estimation | sensor-informed stage |
| H8 | RO-PINN | Zhang, Vlachas, Chatzi, 2026 | arXiv:2608.17131 (S) | frame structure | reduced-order PINN with adaptive basis | closest to the modal arm |
| H9 | Railway PINNs (track irregularity; truss railroad bridge damage) | 2025 | arXiv:2502.00194 (S, ID); others (S: titles) | train–track, bridge–train LTV | PINNs with in-service responses | application stage only |

## I. High-order derivative efficiency

| ID | Method | Authors, year | Venue / identifier | Idea | Measured / reported compute | Status | Notes for us |
|---|---|---|---|---|---|---|---|
| I1 | Stochastic Taylor Derivative Estimator (STDE) | Shi, Hu, Lin, Kawaguchi, 2024 | NeurIPS 2024 (oral); arXiv:2412.00088 (S) | univariate Taylor-mode jets contract arbitrary derivative tensors; avoids the O(2^(k−1)L) nested graph | >1000× speed-up, >30× memory vs first-order-AD randomisation (high dimension) | NEW | JAX Taylor mode; PyTorch has no native Taylor mode |
| I2 | Forward-mode AD in SPINN | Cho et al., 2023 | see F6 | forward mode is efficient when the input dimension per axis is 1 | 62× wall-clock (separable) | EST | needs separable structure |
| I3 | CAN-PINN (coupled automatic–numerical differentiation) | Chiu et al., 2022 | CMAME (M); arXiv:2110.15832 (S) | numerical stencils + AD | faster training | PART | — |
| I4 | Taylor mode inside KFAC | Dangel et al., 2024 | see A6 | operator graph as a weight-shared forward network | — | NEW | — |
| I5 | **Our measurement (B2-PROF-001)** | this project | `results_batch2/tables/B2-PROF-001_formulation_step_cost.csv` | nested `torch.func.jvp` vs nested reverse mode, 6×200, mini-batch 128, CPU | **forward 750 ms vs reverse 154 ms per step (4.9× SLOWER)**; mixed 70 ms; modal 36 ms | measured | theoretical FLOP savings do not carry over to nested functorch JVPs on CPU |

## J. Physics + data hybridisation

| ID | Method | Authors, year | Venue / identifier | Notes |
|---|---|---|---|---|
| J1 | PINN (forward and inverse) | Raissi, Perdikaris, Karniadakis, 2019 | J. Comput. Phys. 378:686; DOI 10.1016/j.jcp.2018.10.045 (M) | foundational; sparse data plus physics |
| J2 | gPINN | Yu, Lu, Meng, Karniadakis, 2022 | CMAME 393:114823; arXiv:2111.02801 (S) | residual-gradient loss; better with few points; +1 derivative order (costly for 4th order) |
| J3 | System ID and response estimation | Haywood-Alexander et al., 2024 | see H7 | SHM-relevant |
| J4 | Kapoor inverse beams | 2023 | see H2 | noisy-data inverse problems |

## K. Operator learning and the "general solver" direction

| ID | Method | Authors, year | Venue / identifier | Idea | Edge | Gen | Notes |
|---|---|---|---|---|---|---|---|
| K1 | DeepONet | Lu, Jin, Pang, Zhang, Karniadakis, 2021 | Nat. Mach. Intell. 3:218 (M) | branch/trunk operator learning | H | H | — |
| K2 | Physics-informed DeepONet | Wang, Wang, Perdikaris, 2021 | Science Advances 7(40) eabi8605 (S) | train operators from physics only; ~10³ PDEs solved in a fraction of a second | H | H | amortises solver cost over parameters |
| K3 | FNO | Li et al., 2021 | ICLR 2021; arXiv:2010.08895 (M) | spectral convolution operator | H | H | — |
| K4 | PINO | Li, Zheng, Kovachki, et al., 2024 | ACM/IMS J. Data Sci.; arXiv:2111.03794 (S) | data at coarse resolution + physics at fine resolution | H | H | — |
| K5 | SpectONet (beam) | 2026 | see H6 | — | H | M | — |

## L. Automation, benchmarks and meta-studies (closest to the "problem-adaptive solver" idea)

| ID | Work | Authors, year | Venue / identifier | What it does | Why it matters for our novelty |
|---|---|---|---|---|---|
| L1 | Auto-PINN | Wang et al., 2022/2023 | arXiv:2205.13748; NeurIPS-W 2023 (S) | step-wise decoupled HPO / NAS for PINNs; training loss as the search objective | **selection by search**. Our idea differs only if selection comes from measured physics invariants before training |
| L2 | AutoPINN (resource-aware) | (authors M) | (S: title) | AutoML + PINN with resource constraints | **compute-aware selection by search**: direct overlap with "compute-aware" |
| L3 | PINNsAgent | 2025 | arXiv:2501.12053 (S) | LLM multi-agent; database of past runs; planner proposes architectures | **experience-driven selection** |
| L4 | Lang-PINN | 2025/2026 | ICLR 2026; arXiv:2510.05158 (S) | LLM agents: PDE parsing → PINN architecture selection → code → feedback | **end-to-end automated selection**; reports MSE gains of 3–5 orders of magnitude |
| L5 | EvoPINN | 2026 | arXiv:2607.26490 (S, ID) | agentic discovery of executable algorithms for PINNs | same family |
| L6 | PINNacle benchmark | Hao et al., 2024 | NeurIPS 2024 D&B; arXiv:2306.08827 (S) | 20+ PDEs, ~10 PINN methods; guidance on domain decomposition and reweighting | **benchmark source for Tiers 2–4** |
| L7 | Expert's guide to training PINNs | Wang, Sankaran, Wang, Perdikaris, 2023 | arXiv:2308.08468 (S) | best-practice pipeline (non-dimensionalisation, Fourier features, RWF, causal weighting, LRA, curriculum) | **an established "recipe" baseline**: any "pipeline" claim must beat it |
| L8 | PINNs failure modes are overfitting | Andersen & Matsubara, 2026 | arXiv:2605.30910 (S) | failure = overfitting to collocation points; double-backprop regularisation; 23× fewer points | competing mechanism; Batch-1 redraws points every epoch (fresh points), which partly controls for this |
| L9 | Can PINNs beat FEM? / PINN vs FEM cost | Grossmann, Komorowska, Latz, Schönlieb, 2024 | IMA J. Appl. Math.; arXiv:2302.04107 (S) | PINNs are 1–3 orders of magnitude slower to train than FEM, but faster to evaluate | **frames the compute-aware question honestly**; classical solvers are the efficiency reference |
| L10 | Adaptive PINNs survey | 2025 | arXiv:2503.18181 (S) | taxonomy of adaptive methods | taxonomy cross-check |
| L11 | Difficulty-aware task sampler (meta-learning PINNs) | (2024/25) | (S: description only) | allocates collocation budget across tasks by difficulty | budget-aware, but across tasks |
| L12 | PINN + neural-operator review incl. method selection | 2026 | Neural Networks (S: title via search; details unverified) | review of architectures, pathologies and **method selection** | must be read before any selection claim |

---

## Cross-cutting observations

1. **Every individual mechanism on our list is established or partially explored**: Fourier features, NTK
   weighting, hard constraints, mixed formulation, modal reduction, temporal decomposition, R3, L-BFGS/NNCG/SOAP,
   ReLoBRaLo, forward/Taylor-mode AD, FP64. A Batch-2 contribution therefore cannot be "a new component".
2. **The closest prior art to each Batch-2 axis on the beam itself is recent (2025–2026)**:
   - mixed: A-PINN EB 2026;
   - modal: RO-PINN 2026;
   - temporal + hard constraints: AT-PINN-HC 2025;
   - operator: SpectONet 2026.
   They must be comparators, not citations to be outdone by terminology.
3. **Automated strategy selection is an active area**: AutoML (Auto-PINN, AutoPINN) and LLM agents (PINNsAgent, Lang-PINN,
   EvoPINN). The distinguishing question left open is whether **measurable, pre-training physics invariants
   *predict*** which mechanism dominates and which strategy is most compute-efficient, without trial-and-error
   search. None of the works above was found to pre-register falsifiable selection rules and validate them
   out-of-sample on accuracy-per-compute Pareto fronts. This rests on abstract-level reading only and must be
   re-checked against full texts (L1–L5, L12) before any claim.
4. **Two recent papers attack the failure-mode narrative itself**: FP64 (A9) and overfitting (L8). Batch 1
   used float32 + Adam with per-epoch redraws. A **precision arm** is mandatory before attributing the
   collapse to anything else.
5. **Measured efficiency contradicts naive theory** in our setting (I5). Every efficiency claim in Batch 2 is
   measured on the target hardware.

## Count

**81 distinct works** are tabulated. Rows that repeat a work across categories (for example AT-PINN-HC in D5, G8 and
H5) are counted once, and the internal Batch-1 rows D6 and I5 are not counted.

- **65** were confirmed by search in this audit (S).
- **16** are from bibliographic memory (M): Adam, DeepXDE/RAR, MIM, A-PINN (Yuan), VPINN, WAN, Deep Ritz,
  Tancik et al., adaptive activations, ModalPINN, bc-PINN, XPINN, FBPINN (Moseley), Raissi et al., DeepONet and FNO.
- A DOI marked (M) inside an (S) row means the title/ID was confirmed but the DOI was not.
