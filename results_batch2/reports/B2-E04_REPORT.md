# B2-E04 — What goes wrong inside the optimization when the spatial field is restored? (analysis only, no training)

## PRE-REGISTERED OUTCOME: **CASE F — no clear mechanism among A–D**

| Mechanism | R (reproducible) | S (discriminating) | O (not only after collapse) | Verdict |
|---|---|---|---|---|
| A: spectral / modal | 0/3 seeds | S2 2/3 | 3/3* | **NOT SUPPORTED** |
| B: temporal-window gradient conflict | 2/3 | S1 0/3, S2 0/3 | 3/3 | **PARTIAL** (reproducible, but **not discriminating**) |
| C: space/time operator gradients | 0/3 | 0/3 | 3/3* | **NOT SUPPORTED** |
| D: residual-Jacobian conditioning | 0/3 | 0/3 | 0/3 | **NOT SUPPORTED** |

Source: `results_batch2/B2-E04_CLASSIFICATION.json`; the rules are in `docs/hypotheses/B2-E04.md` §7–9.

\* O for A and C is satisfied only through the initial state, where the criterion holds trivially (§5). It plays no
role, because R fails.

**The statement the evidence supports:**

> When the spatial field is restored, the collapse remains a pure **loss of first-mode amplitude**: a₁(t) → 0
> while higher spatial modes do **not** grow (≤ 5·10⁻⁵ of the exact energy).
>
> After the L-BFGS switch, two things change in the full field.
> - The temporal-window gradients start to conflict. **This happens identically in the successful modal problem**,
>   so it does not explain the divergence.
> - The physical residual behind the front acquires 21–44 % non-fundamental spatial content, against 3–18 % for the
>   Adam continuation. This cannot happen in the modal problem. It is reproducible in 2/3 seeds, but it stays
>   **below** the pre-registered pathology threshold (0.5). It appears **coincident with**, not before, the
>   degradation.
>
> Operator/Jacobian conditioning does not deteriorate after the switch. Space/time operator-gradient conflict is
> absent after the first 128k evaluations.

**Nothing here is novel.** Spectral bias, gradient conflict, causality and operator conditioning are established
concepts (§8). This is a benchmark-specific, pre-registered attribution study.

---

## 1. Objective

The question is why the optimization, which works when the spatial degrees of freedom are supplied analytically
(modal problem), fails when they are restored (full field). Four candidate mechanisms were tested on **existing
checkpoints only**:

- **A.** Modal/spectral competition: redistribution of field or residual energy into higher spatial modes.
- **B.** Temporal gradient conflict or late-window starvation.
- **C.** Conflict or scale imbalance between the temporal operator part (u_tt + γu_t) and the spatial operator part
  (c2·u_xxxx).
- **D.** Degradation of the residual-Jacobian conditioning.

For each mechanism the criteria require that it be reproducible, discriminate the failing trajectory from the
successful one, and appear before or at, not only after, the degradation.

## 2. Existing evidence used (B2-E01–E03)

- FP64 is not material. Mixed (derivative order 2) is never more accurate.
- Modal Adam (M0) collapses at 0.30–0.37 s with a late front retreat. Modal Adam→L-BFGS reaches the full window in
  3/3 seeds.
- The same Adam→L-BFGS on the full field is worse than B1 in 3/3 seeds.
- The train/dense residual gap is a coverage × optimizer interaction and does **not** explain the stalled front
  (B2-E03, CASE D).

## 3. Method (exactly as pre-registered; deviations listed at the end)

**Integrity and provenance**

| Item | Value |
|---|---|
| Pre-registration | `docs/hypotheses/B2-E04.md`, committed and pushed in `5c10536` together with the evaluator `scripts/analyze_b2_e04.py`, **before** any diagnostic of a trained checkpoint was computed |
| Amendment 1 | `7b9424a`. The dry-run's first check found catastrophic cancellation in φ₈ evaluated as cosh − s·sinh (error 2.7e-5). It was replaced by the algebraically identical stable form; no diagnostic value had been seen. No definition or threshold changed |
| States | 75 = per seed: 10 full-field (FF), 9 modal (MO), 6 mixed (MX); seeds 1234–1236 |
| Levels | 0 (reconstructed init, checksum-verified), 128k, 256k, 320k (FF only: the B2-E03 `switch.pt`, bit-identical across the 4 E03 arms and equal to B1's Adam state), 384k, 512k, 640k (Adam) / 623,360 (L-BFGS arms) |
| Unavailable levels | 64k does not exist; 320k does not exist for MO/MX. No substitute was used (§3.2 of the pre-registration) |
| Integrity assertions (all passed) | <ul><li>shared pre-switch states bit-identical (transfer = B1, modal L-BFGS = M0 at 128k/256k);</li><li>reconstructed inits match the recorded sha256 (`9c5c91e3…`, `d5d35861…`, `8c963bb2…`);</li><li>the MO init equals the B1 temporal branch;</li><li>the MX init u equals the B1 init u;</li><li>every checkpoint's evaluation count equals its level;</li><li>recomputed L2/t_c equal the previously reported snapshot tables (81 rows cross-checked: E01S, E02, E02T, E03)</li></ul> |
| Checkpoint safety | 66 source files hashed before and after (`results_batch2/B2-E04/checkpoint_manifest_{before,after}.json`): **0 changed**. All set read-only (0444) before the run. Diagnostics run on float64 copies; no optimizer was built |
| Dry-run (`FF_b1-128k_s1234`) | <ul><li>basis orthogonality 1.3e-15;</li><li>exact solution → a₁ = A0·q_ex (4.7e-16), non-fundamental energy 7.6e-30;</li><li>residual split = frozen `pde_residual` (1.3e-17);</li><li>window-loss gradient vs central finite difference 2.8e-10, vs direct autograd 1.2e-15;</li><li>Jacobian chunk invariance 0 (bitwise); Jᵀr/N identity 7.4e-16;</li><li>QR-route σ vs `numpy.linalg.svd` 1.1e-14;</li><li>166 s per FF state</li></ul> |
| Compute | 3 single-thread processes; peak RSS ≤ 4.2 GB each; about 1.2 h wall time |

**Diagnostics** (details in pre-registration §4–6)

- **A: field modal decomposition.** Grid: 64 Gauss–Legendre x-nodes × 4001 t-nodes. Projections onto the 8
  exact-root fixed–fixed eigenfunctions give a_n(t) and energies relative to the exact energy, ρ_n. Also reported:
  the non-fundamental energy ρ_nonfund (modes 2–8 plus the beyond-8 remainder), the first-mode collapse time t_c1,
  and the post-front character.
- **B: residual spectrum.** The physical residual r = c2·u_xxxx + u_tt + γu_t on 64 GL × 501 t is projected onto the
  same basis. Reported: the per-mode energy fraction, the non-fundamental share h_r = 1 − σ₁, the centroid and the
  dominant mode. Regions: W₁–W₄, FULL, PRE = [0, t_c) (behind the front) and POST = [t_c, 1].
- **C: temporal-window gradients.** Deterministic quadrature (16 GL × 256 t-midpoints) of each problem's own
  training residual. Reported: the gradients g₁–g₄ of the window losses for W₁ = [0, 0.25] … W₄ = (0.75, 1];
  the 4×4 cosine matrix, min C, the negative fraction and the starvation ratio ‖g₄‖/‖g₁‖; front-relative
  BEHIND/FRONT/AHEAD windows (±P_d around t_c).
- **D: space/time operator gradients.**
  - FF: r_time = u_tt + γu_t and r_space = c2·u_xxxx.
  - MO: their Galerkin images A0(q'' + γq') and A0·ω₁²q.
  - Reported: C_st = cos(∇⟨r_time²⟩, ∇⟨r_space²⟩), ‖∇L_space‖/‖∇L_time‖, and the decomposition of the *actual*
    training gradient, g = g_T + g_S, with the cancellation index χ.
- **E: residual Jacobian.**
  - J = ∂r/∂θ, computed exactly in float64 (640 × 241,601) on a fixed fresh uniform set (seed 20261008).
  - Singular values via in-place Householder QR of Jᵀ and the SVD of R.
  - "Estimated residual-Jacobian spectral condition" κ̂ = σ₁/σ_res, with τ_res = 1e-10.
  - Also reported: numerical ranks r(τ) and spectral decay.

## 4. Results (raw numbers; per seed; no pooling)

Full tables:
- `results_batch2/B2-E04_DIAGNOSTICS.csv`: unified record per trajectory/seed/level;
- `B2-E04_MODAL_SPECTRUM.csv`;
- `B2-E04_RESIDUAL_SPECTRUM.csv`;
- `B2-E04_GRADIENT_CONFLICT.csv`;
- `B2-E04_CONDITIONING.csv`;
- `results_batch2/B2-E04/B2-E04_DESCRIPTIVE.json`.

Figures: `results_batch2/figures/B2-E04/` (30 PNGs).

### 4.1 Physical outcome being explained (frozen metrics, recomputed and cross-checked)

Collapse time t_c [s] at 384k / 512k / end:

| Seed | B1 (Adam continues) | Transfer (L-BFGS) | M0 (modal Adam) | Modal L-BFGS |
|---|---|---|---|---|
| 1234 | 0.105 / 0.130 / 0.153 | 0.080 / 0.078 / 0.078 | 0.347 / 0.396 / 0.370 | 0.389 / 0.761 / **1.000** |
| 1235 | 0.078 / 0.105 / 0.151 | 0.035 / 0.035 / 0.035 | 0.418 / 0.370 / 0.322 | 0.493 / 0.881 / **1.000** |
| 1236 | 0.129 / 0.177 / 0.248 | 0.105 / 0.106 / 0.123 | 0.419 / 0.346 / 0.296 | 0.637 / **1.000** / **1.000** |

At 320k the full-field state is shared: 0.082 / 0.056 / 0.105 s.

**Pre-registered degradation events:**
- transfer falls behind B1 by more than P_d/2 at 384k, 384k and 512k (s1234, s1235, s1236);
- M0 front retreat at 640k, 512k and 512k.

### 4.2 Diagnostic A: the collapse is first-mode amplitude loss, not spectral redistribution

| Quantity (all 72 trained states) | Full field (B1 + transfer) | Mixed | Modal |
|---|---|---|---|
| Post-front character | **AMPLITUDE_LOSS in 27/27 FF and 15/15 MX trained states** | AMPLITUDE_LOSS | AMPLITUDE_LOSS, or NOT_COLLAPSED (modal L-BFGS) |
| First-mode energy retained ahead of the front, ρ₁(POST) | 0.013–0.105 | 0.013–0.067 | 0.064–0.200 |
| Total content ahead of the front, ρ_tot(POST) | 0.013–0.105 (= ρ₁) | 0.013–0.067 | = ρ₁ |
| Non-fundamental energy ahead of the front, ρ_nonfund(POST) | **≤ 1.2·10⁻⁵** | ≤ 3·10⁻⁴ | 0 (structural) |
| Non-fundamental energy behind the front, ρ_nonfund(PRE) | ≤ 4.9·10⁻⁵ (modes 5–8: ≤ 1.2·10⁻⁶; beyond 8: ≤ 4·10⁻¹⁰) | ≤ 5.4·10⁻⁵ | 0 |
| \|t_c1 − t_c\| (first-mode vs displacement collapse) | ≤ 5·10⁻⁴ s (2 trace steps) | ≤ 2.5·10⁻⁴ s | 0 |

**Answers to the five questions of pre-registration §4.4:**

1. **Does a₁ disappear?** Yes. Beyond the front a₁ keeps only 1–10 % of the exact energy, and the first-mode
   collapse time coincides with the displacement collapse time.
2. **Does energy migrate into higher modes?** **No.** Modes 2–8 carry ≤ 1.2·10⁻⁵ of the exact energy ahead of the
   front.
3. **Is total modal content preserved while a₁ decays?** No. The total content falls exactly with a₁
   (ρ_tot = ρ₁).
4. **Spurious high-frequency spatial content?** Negligible in the field: modes 5–8 ≤ 1.2·10⁻⁶ and beyond-8
   ≤ 4·10⁻¹⁰.
5. **Amplitude loss or redistribution?** **Pure amplitude loss**, in every trained state of every seed and both
   full-field trajectories, and also for mixed.

This is the "particularly important result" anticipated in the task: **a₁(t) → 0 while higher modes do not grow**.
The full-field failure is therefore not "energy moved into other modes". It is the same over-damped first-mode
decay already seen in the modal problem. Plot A: `B2-E04_01_modal_coefficients_s*.png`.

**Pre-registered rule artefact (disclosed).** At the *initial* state the field overshoots (ρ₁(POST) = 13–38), so
"missing" energy = 0, and the §4.2 rule labels it REDISTRIBUTION by default. This is why A's O-ordering reads
"precedes" (first pathological level = 0k). It has no influence on the verdict, because R = 0/3.

### 4.3 Diagnostic B: residual spectrum, the one full-field-specific change after the switch

Non-fundamental residual share **behind the front**, h_r(PRE). Mode 1 remains the dominant residual mode in every
trained state.

| Seed | switch 320k | B1 384k / 512k / 640k | Transfer 384k / 512k / 623k | Mixed 640k |
|---|---|---|---|---|
| 1234 | 0.048 | 0.073 / 0.117 / 0.098 | **0.208** / 0.131 / 0.115 | 0.982 |
| 1235 | 0.037 | 0.055 / 0.060 / 0.119 | **0.436 / 0.443 / 0.443** | 0.943 |
| 1236 | 0.077 | 0.179 / 0.178 / 0.176 | **0.379 / 0.402 / 0.367** | 0.984 |

**Composition of the transfer's residual behind the front at 384k:**
- symmetric higher modes: mode 3 up to 0.11, mode 5 up to 0.18, mode 7 up to 0.08;
- antisymmetric content σ_even: 0.09–0.17, against 0.01–0.02 at the switch.

**Each operator part individually stays ≥ 88 % mode 1.**
- r_time is 100 % mode 1.
- r_space is 88–99 % mode 1.

The non-fundamental residual therefore arises in the **sum**: once the mode-1 parts nearly cancel, the remaining
c2·u_xxxx contribution of field content too small to see (≤ 5·10⁻⁵ of the energy) becomes visible. That content is
amplified by (β_n/β₁)⁴ ≈ 29 (mode 3) and ≈ 178 (mode 5).

**Status under the pre-registered rules:**
- The paired increase over B1 exceeds the 0.1 margin at ≥ 2 of 3 levels in **2/3 seeds** (1235, 1236), so S2_A
  holds.
- The absolute criterion h_r(PRE) ≥ 0.5 is never reached (maximum 0.443), so R_A = 0/3 and **A is NOT SUPPORTED**.
- In the modal problem this shift is impossible by construction.

**Mixed:** h_r = 0.69–0.98, i.e. its strong residual is dominated by higher modes. Its optimized objective is
different, so this is reported only descriptively.

### 4.4 Diagnostic C: temporal-window gradient conflict (W₁–W₄; front-relative)

min C over the 6 window pairs, at 384k / 512k / end:

| Seed | B1 | **Transfer** | M0 | **Modal L-BFGS (successful)** |
|---|---|---|---|---|
| 1234 | 0.58 / −0.13 / 0.55 | **−0.71 / −0.65 / −0.56** | −0.57 / −0.21 / 0.22 | **−0.41 / −0.19 / −0.04** |
| 1235 | 0.70 / 0.74 / 0.35 | −0.05 / 0.00 / 0.02 | 0.10 / 0.09 / −0.01 | **−0.26 / −0.46 / −0.11** |
| 1236 | 0.38 / 0.13 / 0.78 | **−0.17 / −0.29 / −0.33** | 0.00 / 0.17 / −0.48 | **−0.24 / −0.50 / −0.67** |

Late-window starvation ‖g₄‖/‖g₁‖ at the same levels:

| Seed | B1 | Transfer | M0 | Modal L-BFGS |
|---|---|---|---|---|
| 1234 | 0.55 / 0.36 / 0.80 | 0.23 / 0.10 / 0.09 | 0.18 / 0.50 / 0.17 | **0.024 / 0.097 / 0.051** |
| 1235 | 1.94 / 0.53 / 0.75 | 0.30 / 0.22 / 0.22 | 0.53 / 1.17 / 0.44 | **0.020 / 0.090 / 0.043** |
| 1236 | 0.56 / 1.64 / 0.69 | 0.10 / 0.09 / 0.09 | 1.20 / 0.84 / 0.13 | **0.017 / 0.018 / 0.013** |

**What the tables show:**
- **L-BFGS induces window-gradient conflict and late-window gradient reduction in *both* problems.**
- Both are **stronger in the successful modal L-BFGS runs**: starvation 0.013–0.097 there, against 0.085–0.30 in the
  failing full-field runs.
- The criterion holds for the transfer in 2/3 seeds (R), but equally for modal L-BFGS. So S1 = 0/3 and S2 = 0/3, and
  B is **PARTIAL: reproducible but not discriminating**.
- Behind-front/front conflict (cos ≤ −0.1) also occurs in **B1 itself** (s1234: −0.10, −0.36, −0.40), whose front
  keeps advancing.
- In s1236 the conflict appears at 384k, before the 512k fall-behind event. It appears at the same level in the
  successful modal run.
- Plots: `B2-E04_04_window_cosine_s*.png` (Plot B), `_04b_minC.png`, `_04c_cos_behind_front.png`,
  `_05_window_gradient_norms.png`, `_05b_starvation.png`.

### 4.5 Diagnostic D: space/time operator gradients

C_st and ‖∇L_space‖/‖∇L_time‖ at 384k / 512k / end:

| Seed | Transfer C_st | Transfer ratio | B1 C_st | B1 ratio | Modal L-BFGS C_st | Modal L-BFGS ratio |
|---|---|---|---|---|---|---|
| 1234 | 0.92 / 0.89 / 0.88 | 0.94 / 0.90 / 0.89 | 0.65 / 0.87 / 0.48 | 1.05 / 1.12 / 1.10 | 0.99 / 0.99 / 0.99 | 0.98 / 0.98 / 0.98 |
| 1235 | 0.54 / 0.49 / 0.47 | 1.09 / 1.07 / 1.04 | 0.53 / 0.45 / 0.71 | 1.60 / 1.40 / 1.00 | 0.99 / 0.99 / 0.99 | 0.97 / 0.97 / 0.97 |
| 1236 | 0.93 / 0.93 / 0.93 | 0.88 / 0.87 / 0.88 | 0.89 / 0.76 / 0.82 | 1.13 / 1.21 / 1.66 | 0.99 / 0.99 / 0.99 | 0.99 / 0.99 / 0.99 |

**Space/time gradient conflict:**
- There is **no space/time gradient conflict** in any trained state: C_st ≥ 0.007 throughout.
- There is **no scale imbalance after training**: the ratio is 0.87–2.0 in FF and 0.92–1.5 in MO. C is
  **NOT SUPPORTED** (falsified per §10).

**The one large difference sits at initialization:**
- the full-field ratio is **1.6–3.4·10⁴**;
- the modal ratio is 61–560;
- so the spatial-operator gradient initially dominates about 60–500× more when the field is learned;
- by 128k it has disappeared in every seed (FF ratio 1.2–2.0).

**Training-gradient decomposition (D2):**
- After the switch, the cancellation index χ falls in **both** problems:
  - FF from 0.78–0.87 to 0.62–0.78;
  - MO from 0.88–0.95 to 0.58–0.82.
- So it does not discriminate either.
- C_TS is negative in most FF states (−0.05 to −0.54). That is, the temporal and spatial parts of the true training
  gradient partly cancel; this holds equally for B1 and the transfer.

### 4.6 Diagnostic E: residual-Jacobian conditioning (Plot D: `B2-E04_09_condition_trajectory.png`)

| | Full field (640 × 241,601) | Modal (640 × 241,601) |
|---|---|---|
| log10 κ̂ | init 4.05–4.23 → 4.56–5.25 trained | 9.93–10.0 (**saturated** at 1/τ_res) |
| r(1e-10) / r(1e-6) | **640 / 640** (full row rank) | 426–507 / 276–352 |
| σ₁₀₀/σ₁ | 0.024–0.159 | 0.003–0.013 |
| Δlog10 κ̂ after the switch (transfer vs 320k) | −0.08 … +0.07 (no deterioration) | — |
| Δlog10 κ̂, B1 320k → 640k | +0.06 … +0.25 | — |

**Reading the table:**
- **Conditioning does not deteriorate when the full-field L-BFGS run fails.**
- The *successful* modal problem has the far worse sampled Jacobian condition. That is structural: its 640 rows
  are samples of a function of t alone, so they are nearly linearly dependent.
- D is **NOT SUPPORTED** (falsified per §10: no 10× deterioration and no 25 % rank loss in any seed).
- Within B1, κ̂ increases slowly as the solution improves (Spearman κ̂ vs L2: ρ = −0.61, p = 0.007, n = 18 states).
  Conditioning gets mildly worse as accuracy improves; it is not a precursor of failure.

**Limitation (estimator).** With N = 640 rows the full-field Jacobian is row-limited: r = N in every state. The
full-field rank therefore cannot reveal degeneration beyond 640 directions. FF-vs-MO comparisons of absolute κ̂ are
confounded by input dimension (2-D vs 1-D rows). This is an exact property of the pre-registered estimator, stated
rather than "fixed" after seeing it.

## 5. Temporal ordering

Pre-registered: training-time ordering relative to the fall-behind event (384k / 384k / 512k), and physical-time
location relative to the front.

| Mechanism | First appearance on the transfer trajectory | Relative to the event | Same change in the successful modal L-BFGS? |
|---|---|---|---|
| A (field redistribution) | never in a trained state (init artefact only) | — | structurally impossible |
| A (residual non-fundamental share behind the front, below threshold) | 384k: the first post-switch level, ≤ 100 closures after the switch | **coincides** (s1234, s1235); **precedes** by one level (s1236) | structurally impossible |
| B (window conflict) | 384k (s1234, s1236); not met in s1235 post-switch | coincides / precedes | **yes, at the same level, with stronger starvation** |
| C (space/time) | only at init (imbalance 10⁴, transient); absent after 128k | — | weaker at init; absent afterwards |
| D (conditioning) | never | absent | no |

**What can be concluded about ordering:**
- No quantity shows a pathology **before** the switch that differs between the trajectories. They share the
  identical 320k state, which then leads to two different outcomes.
- What matters therefore arises within the first ≤ 100 L-BFGS closures. The 128k snapshot spacing cannot resolve
  whether it precedes or follows the front stall inside that interval.

**Lead–lag association** (§9.6, descriptive): Spearman between each primary scalar at level k and the front advance
t_c(k+1) − t_c(k), pooled over seeds (n = 12–15 per trajectory; about 30 tests in total).
- There is no consistent association across trajectories.
- Isolated |ρ| ≈ 0.6 with p ≈ 0.02 appear in different, unrelated scalars per trajectory:
  - FF-LBFGS: |log10 ratio|;
  - MO-ADAM: min C, with the **opposite** sign to a conflict mechanism, and r(1e-6).
- About 1.5 such values are expected by chance.
- Source: `results_batch2/B2-E04/B2-E04_DESCRIPTIVE.json`.

## 6. Modal vs full field (step 9)

| Mechanism | Modal (φ₁ supplied) | Full field (φ learned) | Evidence |
|---|---|---|---|
| Modal energy redistribution (field) | impossible by construction | **absent**: ≤ 5·10⁻⁵ of exact energy; collapse = pure a₁ loss | §4.2; 27/27 FF and 15/15 MX trained states AMPLITUDE_LOSS |
| High-mode residual dominance | impossible (σ₁ = 1) | **not dominant** (mode 1 dominant everywhere). The non-fundamental share behind the front rises 3–18 % → 21–44 % after the L-BFGS switch (2/3 seeds beyond margin); below the 0.5 threshold | §4.3 |
| Temporal gradient conflict | **present after L-BFGS** (min C to −0.67; starvation 0.013–0.097), yet successful | present after L-BFGS in 2/3 seeds (min C to −0.71; starvation 0.085–0.30) | §4.4; not discriminating |
| Space/time gradient conflict | none (C_st 0.69–0.99; ratio 0.92–1.5 after 128k) | none (C_st 0.007–0.93 over all trained states, 0.47–0.93 after the switch; ratio 0.87–2.0); 10⁴ imbalance at init only | §4.5 |
| Jacobian conditioning degradation | saturated by 1-D row redundancy; stable | well conditioned (κ̂ ~ 10⁵, full row rank); no post-switch change | §4.6 |

**Spatial coupling test** (what changes when φ₁ is no longer supplied):
1. A transient 10⁴ spatial-operator gradient dominance at initialization, gone by 128k.
2. Row-rank structure of the Jacobian (dimension-driven).
3. The possibility of non-fundamental residual content, which becomes material (21–44 %) only after the L-BFGS
   switch.

**Temporal propagation test** (what stays difficult without the spatial field). The modal Adam control shows the
same pure first-mode amplitude loss. Its late front retreat (512k–640k) is preceded by **no** consistent criterion:
- B is met in 2/3 seeds at inconsistent positions: s1234 at 384k–512k, before its 640k retreat; s1236 only at 640k,
  after its 512k retreat; s1235 never;
- C and D are never met.

**Optimization test** (what changes after the switch):

| Change | Full field | Modal problem |
|---|---|---|
| Window-gradient conflict and late-window reduction | yes | yes, stronger |
| Training-gradient cancellation χ | ↓ | ↓ |
| Jacobian conditioning | unchanged | unchanged |
| Non-fundamental residual behind the front | ↑ | cannot occur |

## 7. Mechanism classification

**CASE F: no clear mechanism.**
- None of A–D satisfies the pre-registered conjunction of reproducible, discriminating and not-only-after-collapse.
- **B is PARTIAL.** Temporal-window conflict is reproducible on the failing full-field runs, but the successful
  modal L-BFGS runs show it equally or more strongly.
- **A narrowly misses on magnitude, not on direction.** Its discriminating paired test (S2) holds in 2/3 seeds, but
  the absolute criterion is never reached.

## 8. Literature interpretation (`docs/LITERATURE_MATRIX.md`; targeted search 2026-10-08)

**Gradient conflict between loss components** is established:
- Wang, Teng & Perdikaris 2021 (A8);
- ConFIG, Liu, Chu & Thuerey ICLR 2025 (B5; arXiv:2408.11104);
- Wang et al. NeurIPS 2025 (A4; arXiv:2502.00604). It argues that second-order/quasi-Newton preconditioning resolves
  directional conflicts through implicit alignment.

Our temporal-window conflict *appears* after the quasi-Newton switch. It coexists with **success** in the modal
problem, which is consistent with conflict per se not being harmful. We do not claim more.

**Temporal causality / propagation failure** (causal weighting B6; R3, Daw et al. 2023, C3) describes a
front-propagation failure of exactly the observed type: pure amplitude loss beyond a front. B2-E02 already found
causal weighting worse on this benchmark. B2-E04 adds that the collapsed region carries **no** spurious
higher-mode field energy: the trivial (decayed) solution is reached through first-mode amplitude, not through
spatial redistribution.

**Operator / Jacobian conditioning** is established:
- De Ryck et al. ICLR 2024 (D1): the operator's Hermitian square, effectively order 8 for u_xxxx;
- Rathore et al. ICML 2024 (A2): Hessian spectrum, Adam + L-BFGS;
- Liu et al. 2024 (D2; arXiv:2402.00531);
- arXiv:2405.01957 (Jacobian condition; search-level only).

These works predict that the 4th-order spatial operator worsens conditioning. Our data show the predicted **initial**
spatial-operator dominance (10⁴), but **no** post-switch conditioning degradation of the sampled residual Jacobian.
So, on this benchmark and with this estimator, conditioning does not separate the successful and failing L-BFGS
stages.

**Spectral bias / NTK eigen-disparity:**
- Wang, Yu & Perdikaris JCP 2022 (B1; arXiv:2007.14527);
- Wang, Wang & Perdikaris 2021 (F2).

Spectral bias would predict difficulty with high-frequency content. The field contains essentially none, and its
failure is a low-mode amplitude loss. A "spatial spectral bias" account is therefore not supported here. The
temporal oscillation (20.6 cycles) is the high-frequency direction, and it is common to both problems.

**What is new, and what is not:**
- None of the measured mechanisms is new.
- The specific combination is benchmark-specific evidence, not found in the matrix: paired modal vs full-field runs,
  same init, same switch, pre-registered and ordered.
- It shows that two of the most commonly invoked explanations (conflict, conditioning) do not discriminate success
  from failure in this case.

## 9. What was ruled out (on this benchmark, at this resolution)

1. **Field spectral redistribution.** Energy does not migrate into higher spatial modes; collapse is pure a₁
   amplitude loss (27/27 FF and 15/15 MX trained states).
2. **Space/time operator-gradient conflict or scale imbalance** after the first 128k evaluations. C_st > 0 and the
   ratio ≈ 1, in both problems.
3. **Residual-Jacobian conditioning degradation** as the trigger of the L-BFGS failure. κ̂ is unchanged after the
   switch, and the successful modal problem is the worse-conditioned one.
4. **Temporal-window gradient conflict or late-window starvation as a discriminating cause.** These are present, and
   stronger, in the successful modal L-BFGS runs.
5. **(Carried from B2-E01–E03.)** Precision, derivative order alone, collocation density, and L-BFGS alone.

**Not ruled out:**
- The full-field-specific growth of non-fundamental residual content behind the front after the switch (sub-threshold;
  2/3 seeds).
- Anything happening at a finer temporal resolution than the 128k snapshots, i.e. inside the first 100 closures.

## 10. Recommended next experiment (ONE direction; not executed; needs pre-registration and approval)

**Isolate the spatial composition of the objective at the L-BFGS stage, keeping the full-field network: a
modal-restricted (Galerkin test-function) residual objective.**

**Why this one.** The only quantity that changes after the switch **only** in the full field is the non-fundamental
spatial content of the residual. It must be tested causally rather than by correlation.

**Design.** From the identical shared 320k state of each seed, run two arms:
- the existing transfer arm (strong residual, full);
- one arm whose L-BFGS stage minimizes only the projection of the strong residual onto φ₁, i.e. ⟨r, φ₁⟩(t). This is
  the modal-problem objective with the spatial field still learned.

Everything else is identical: network, init, switch, closures, budget and accounting.

**Interpretation:**

| Restricted arm (K = 1) | Then the obstacle is… |
|---|---|
| propagates like modal L-BFGS | the non-fundamental residual components of the full-field objective |
| fails like the transfer | the spatial representation / coupling inside the network itself; future work should go there (representation), not to the objective, the sampler or the optimizer |

This is an established idea (Petrov–Galerkin / VPINN, matrix E5) used here as a **diagnostic contrast**, not a
method.

**Caveat.** It is motivated by a signal that stayed **below** the pre-registered threshold (§4.3). The test must be
pre-registered with its own falsification criterion. The B2-E03 sampling-freshness recommendation remains open and
is not superseded.

---

### Deviations and transparency notes

- **Amendment 1** (stable φ_n evaluation, before any diagnostic value): §3.
- **The descriptive Spearman analysis** (pre-registered §9.6) was implemented after the classification had been
  computed (`descriptive` command). It is descriptive only and does not enter the CASE.
- **The pre-registered post-front rule labels the over-shooting initial state "REDISTRIBUTION"** (missing = 0). This is
  disclosed in §4.2 and has no effect on the verdict.
- **The diagnostics are evaluated on float64 copies of float32-trained weights**, i.e. the exact-arithmetic version
  of the trained function. FP64 was shown not to be material for training in B2-E01.
- **Unequal end levels.** L-BFGS end states are at 623,360 evaluations vs 640,000 for Adam (−2.6 %); flagged in
  every paired comparison.
- **Three seeds.** No significance claims; consistency counts only.

**STOPPED. B2-E05 is not designed or run. Waiting for review of the B2-E04 evidence.**
