# Physics-solver taxonomy — mechanisms, mathematics and what each acts on

This document classifies the **mechanisms** of a neural physics solver by the mathematical object each
one changes, using the damped Euler–Bernoulli beam (FE-D-M1) as the worked instance. It is the reference for
`docs/BATCH2_EXPERIMENT_MATRIX.md`: every experiment axis is one row here.

Notation:
- u(x,t): displacement [m] on x ∈ [0,L], t ∈ [0,T], with L = 2.75 m and T = 1 s.
- θ: network parameters (P ≈ 2.4e5).
- r: the PDE residual.
- Collocation points x_i are redrawn every epoch (Batch-1 sampler).

## 0. The problem (frozen, Batch-1 definition)

    c² u_xxxx + u_tt + γ u_t = 0,      c² = 43.73² = 1912.31,  γ = 7.08          (paper Eq. 49)
    u = u_x = 0 at x = 0, L            (fixed–fixed)
    u(x,0) = A0 φ₁(x)/φ₁(L/2),  u_t(x,0) = 0,     A0 = 0.08 m, φ₁ = exact-root mode shape

Exact solution: u = A0 φ̂₁(x) q(t), with q'' + γ q' + ω₁² q = 0, q(0) = 1, q'(0) = 0.
- ω₁ = β₁²c, β₁L = 4.730040745, so ω₁ = 129.373 rad/s.
- ω_d = 129.325 rad/s (20.58 Hz).

Dimensionless invariants (`results_batch2/reports/B2-DIAG-001_conditioning_FE-D-M1.md`):

| group | value | meaning |
|---|---|---|
| N_c = ω_d T / 2π | **20.58** | cycles the solution must propagate through |
| ζ = γ / 2ω₁ | 0.0274 | damping ratio (light damping: the envelope falls to 2.9 % at T) |
| β₁L | 4.730 | spatial mode content (a single half-wave family) |
| ω₁ / (γ/2) | 36.5 | ratio of oscillation rate to decay rate (temporal stiffness) |

No change of variables alters these groups. Rescaling moves coefficient magnitudes around, but it cannot
shorten the temporal propagation length N_c.

## 1. The pipeline and the object each stage changes

| Stage | Mechanism family | Mathematical object changed | Primary pathology addressed | Literature (matrix IDs) |
|---|---|---|---|---|
| Physics analysis | invariants, scales, derivative orders, spectra | none (measurement) | — | D1, D2 |
| Conditioning | nondimensionalisation, variable/term scaling, residual normalisation, ansatz time factor | coordinates, coefficient magnitudes, required network-output scale | Hessian/NTK spectrum spread from scale disparity | D3–D6 |
| Formulation | strong / mixed / first-order-in-time / weak / modal | the residual operator 𝒜 and its order | conditioning of 𝒜*𝒜 (D1); derivative-graph cost | E1–E7, F9 |
| Constraints | soft penalty / hard ansatz (distance or auxiliary functions) | feasible set of the trial space | term imbalance; static attractors | D5, D7, D8 |
| Representation | MLP / Fourier / SIREN / separable / modal basis | function class and its NTK eigenbasis | spectral bias | F1–F10 |
| Sampling | uniform / RAD / R3 / time-sweeping | the empirical measure of the loss | propagation failure; overfitting to points | C1–C4, L8 |
| Loss balancing | NTK / LRA / GradNorm / ReLoBRaLo / SA / ConFIG | the weights or directions of the multi-term gradient | rate imbalance; gradient conflict | B1–B5 |
| Temporal strategy | global / windows / time-marching / causal | domain topology over t; information flow | long-time propagation | G1–G8, B6 |
| Optimizer | Adam / L-BFGS / NNCG / SOAP / KFAC / ENGD | the preconditioner applied to ∇θ L | ill-conditioning (A2, D1) | A1–A8 |
| Precision | FP32 / FP64 / mixed | arithmetic rounding of residuals and updates | premature stopping; cancellation | A9 |
| Derivative computation | reverse / forward / Taylor mode / mixed | the cost of evaluating ∂^k u; not the maths | compute and memory | I1–I5 |
| Verification | residual, IC/BC, frequency, decay, persistence gates | none (measurement) | silent failure | — |

## 2. Formulations of the beam (all mathematically equivalent for smooth solutions)

| Form | Unknowns | Residuals | Highest derivative of a network | Derivative evaluations per point (reverse mode) | Status in PINN-RP |
|---|---|---|---|---|---|
| **Strong** (B0, B1) | u | r = c²u_xxxx + u_tt + γu_t | 4 (x), 2 (t) | 4 nested x-grads + 2 t-grads | implemented (beampinn) |
| **Mixed (curvature)** | u, v | r₁ = c²β₁²(v − u_xx), r₂ = c²v_xx + u_tt + γu_t | 2 (x), 2 (t) | 2 + 2 x-grads, 2 t-grads (one shared trunk) | **implemented** (`physref/formulations/mixed.py`) |
| First-order in time | u, w | w − u_t, w_t + c²u_xxxx + γw | 4 (x), 1 (t) | 4 x + 2 t-grads | not implemented (no order reduction in x) |
| Mixed + first-order | u, v, w | combine the above | 2 (x), 1 (t) | — | candidate |
| **Weak (Petrov–Galerkin)** | u | ∫∫ (c²u_xx ψ_xx + u_tt ψ + γ u_t ψ) dx dt = 0 for every test ψ with ψ = ψ_x = 0 at x = 0, L | 2 (x) on the network | 2 x-grads + quadrature over test functions | not implemented (E5) |
| **Modal (Galerkin, n modes)** | q_n(t) | q_n'' + γq_n' + ω_n²q_n = 0 | 0 (x), 2 (t) | 2 t-grads of a 1-D network | **implemented** (`physref/formulations/modal.py`) |

**Mixed: equivalence.**

- If r₁ ≡ 0 then v = u_xx and r₂ = c²u_xxxx + u_tt + γu_t, so the solution set is unchanged.
  `test_mixed_equals_strong_when_link_is_satisfied` verifies this with any network for u.
- At a clamped end the moment v is not zero, so v carries no BC.
- v(x,0) = u₀'' is enforced exactly by the ansatz v = u₀'' + g(t)β₁²A0 N_v, consistent with u(x,0) = u₀.

**Scaling of r₁ (declared, not tuned).** For a modal field |u_xx| ~ β₁²A0, so c²β₁²(v − u_xx) has the scale
c²β₁⁴A0 = ω₁²A0, the same as each term of r₂. The two loss terms have equal weight 1, and no other weight is
introduced.

**Modal: derivation.**

1. Substitute u = Σ A0 φ_n(x) q_n(t), where φ_n'''' = β_n⁴ φ_n (exact eigenfunctions of the fixed–fixed
   operator).
2. Project onto φ_m. Orthogonality decouples the modes:
   q_n'' + γ q_n' + c²β_n⁴ q_n = 0, with q_n(0) = ⟨u₀,φ_n⟩/(A0⟨φ_n,φ_n⟩) and q_n'(0) = 0.
3. Numerical check (`B2-DIAG-001`): ω_n² from Galerkin quadrature equals c²β_n⁴ to < 1e-6 relative.
   q(0) = (1, 5.6e-15, −1.2e-13).

**For FE-D-M1 the one-mode ansatz contains the exact solution.** The modal arm is a diagnostic that removes
the spatial operator entirely. It is never a fair full-field competitor.

Its full-field PDE residual is **identically** φ̂₁(x) × (ODE residual)
(`test_modal_field_pde_residual_is_shape_times_ode_residual`). Measuring the modal arm therefore measures
the temporal-propagation difficulty of the same 20.6-cycle signal in isolation.

## 3. Conditioning mechanisms (§15.1)

### 3.1 Coefficient spread under candidate scalings

Values from `B2-DIAG-001`; coefficients are divided by the u_tt coefficient.

| scaling | a₄₀ | a₀₂ | a₀₁ | ξ range | τ range | max/min coefficient |
|---|---|---|---|---|---|---|
| physical (1 m, 1 s) | 1912 | 1 | 7.08 | 2.75 | 1 | 1912 |
| domain (L, T) | 33.4 | 1 | 7.08 | 1 | 1 | 33.4 |
| modal (1/β₁, 1/ω₁) | 1 | 1 | 0.0547 | 4.73 | **129.4** | 18.3 |

Batch-1 models already differentiate with respect to physical coordinates and normalise inputs inside the
network. They keep the residual in physical units, with characteristic magnitude ω₁²A0 = 1339.
Nondimensionalising the **residual** rescales the loss by a constant, and Adam is invariant to that rescaling
to first order. Nondimensionalising the **coordinates the network sees** changes the representation.
These two effects must be tested separately; the matrix does so in E05.

### 3.2 Hard-ansatz time factor

The ansatz is u = u₀ + g(t)Φ(x)A0N(x,t), with g(0) = g'(0) = 0. Matching u_tt(x,0) = −ω₁²u₀ requires

    N*(x, 0) = − (ω₁² / g''(0)) · u₀(x)/(Φ(x)A0)

| g | g''(0) | −ω₁²/g''(0) | Batch-1 measured N* range |
|---|---|---|---|
| (t/T)² | 2/T² | **−8.4e3** | about −8.4e3 (E4: ill-conditioned) |
| tanh²(ω₁t) | 2ω₁² | **−0.5** | [−1.9, −0.16] (X1/X2: conditioning restored) |

**Rule (conditioning, not novelty).** Choose g with g''(0) ≍ ω₁² so that N* = O(1). This is one instance of
"place the problem's characteristic time scale into the ansatz". AT-PINN-HC (D5) studies auxiliary functions
case by case.

### 3.3 Representation–frequency coverage

The expected number of temporal Fourier features with physical frequency ≥ ω_d (out of m = 100) is:
- 3.96 under the B0 convention (3 in the seeded draw);
- 0.019 under the B1 convention (0 in the seeded draw).

B1 still reproduces ω to −0.1 % in the cycles it covers, so the trunk synthesises the frequency. Coverage is
therefore **not** sufficient to predict success, and this is a warning against naive selection rules.

### 3.4 Operator conditioning (theory, D1)

GD on the PINN loss is governed by the spectrum of 𝒜*𝒜 restricted to the model's tangent space. For the
strong beam operator 𝒜 = c²∂_x⁴ + ∂_t² + γ∂_t, the spatial part contributes eigenvalues ∝ (c²β_n⁴)², which
grow like n⁸ over modes. The mixed form splits this into two second-order operators, ∝ n⁴ each. The modal form
removes it. **Whether this matters for the observed failure is exactly what E01 measures (H3).**

## 4. Temporal strategies (§15.4)

| Strategy | Domain over t | Information path IC → late t | Overhead | Status |
|---|---|---|---|---|
| global (B0, B1) | [0,T] at once | through the optimiser only (the residual is local in t) | none | baseline |
| causal weights | [0,T] with temporal gating | forced ordering of the residual | binning | rejected (Batch 1) |
| R3 sampling | [0,T] with an evolving point population | points accumulate at the front | ≈ 0 | implemented (non-causal) |
| windows, hard IC transfer | [t_k, t_{k+1}] sequentially | exact: IC_k := terminal state of k−1 | K trainings | planned (`physref/temporal.py`) |
| windows + transfer learning | as above, initialised from θ_{k−1} | as above | fewer steps per window | planned |
| overlapping windows (XPINN/FBPINN-like) | overlaps with interface or PoU | interface residuals | interface terms | planned |

For a 2nd-order-in-time PDE the interface must transmit **both** u and u_t
(`physref.temporal.interface_residuals`).

**Error accumulation.** Window k inherits the IC error e_{k−1}. With a lightly damped oscillator the error
amplitude is roughly preserved, ×exp(−γΔt/2) per window. Window errors therefore *add up* rather than decay,
and phase errors accumulate linearly. E03 measures this.

## 5. Optimizer and precision (§15.6, §15.10)

| Choice | Per-step cost (P parameters) | Memory | Known effect | Batch-1 evidence |
|---|---|---|---|---|
| Adam, constant lr | 1 gradient | 3P | slow on ill-conditioned losses | B0 static; E/X runs fail |
| Adam + exp-decay lr | 1 gradient | 3P | better | Y1: helps, not sufficient |
| Adam → L-BFGS | 1 gradient + line search (≈ 1–20 evaluations) | (2m+3)P | large gains in A2 | untested |
| SOAP / NNCG | ≈ 2–6 gradient-equivalents | O(P)–O(layer²) | A2, A4 gains | untested |
| FP64 | ≈ 1–2× on CPU (measure) | 2× | A9: removes failure modes for L-BFGS | **untested: mandatory arm** |

## 6. Derivative computation (§15.9)

Measured in B2-PROF-001: 6×200 network, mini-batch 128, float32, one CPU thread. Each row is residual +
backward with **no optimizer step**.

| Strategy | Formulation | Highest network derivative | Median per step | vs B1 |
|---|---|---|---|---|
| nested reverse (Batch 1) | strong | 4 | **154 ms** | 1.00 |
| nested forward (`torch.func.jvp`) | strong | 4 | 750 ms | 4.9× slower |
| nested reverse | mixed (u, v) | 2 | **70 ms** | 2.2× faster |
| nested reverse | modal q(t) | 2 (t only, 1-D input) | **36 ms** | 4.3× faster |

Reverse-mode nesting builds a graph whose size grows roughly geometrically with derivative order (STDE, I1:
O(2^(k−1)L)). Lowering the order from 4 to 2 is therefore a real wall-clock saving here.

Forward mode would be asymptotically favourable for a scalar input. In functorch, however, nesting it
re-traces the network once per order. A fair forward-mode test needs Taylor-mode jets (JAX `jet`), which are
out of scope for now.

## 7. Verification layer (frozen definitions)

| Gate | Definition | Source |
|---|---|---|
| Persistence | no collapse in [P/2, T]: R(t) = A_pred/A_exact over a one-period sliding window, collapse when R < 0.5; checked for displacement and velocity | Batch 1 `budget_diagnostic.py` → `physref/persistence.py` (verbatim) |
| L2 (both references) | rel-L2 on the 201×2001 grid vs paper-faithful and exact references | Batch 1 `metrics.py` |
| PDE residual | RMS(r)/RMS(u_tt,ref) on a 51×501 grid | Batch 1 |
| IC/BC | dimensionless max errors | Batch 1 |
| Frequency / decay | damped-cosine fit at mid-span. A diagnostic, not a gate (extractor p95 ≈ 1.4e-5) | Batch 1 |
| Primary paper gate | L2_paper < 4.64e-4, reported together with L2_exact (floor 4.386e-4 from reference rounding) | Batch 1 STAGE01 §3 |
