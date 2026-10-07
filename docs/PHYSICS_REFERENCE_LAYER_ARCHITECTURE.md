# Physics Reference Layer — proposed architecture (RESEARCH HYPOTHESIS, not a claimed system)

This document maps the conceptual pipeline (prompt §9, §16, §22) onto concrete modules. For each stage it
states what **exists**, what is **planned**, and what is a **hypothesis requiring evidence**.

**No name is given to the solver strategy.** Naming waits until the novelty audit (`docs/NOVELTY_GAP.md`,
gap D2) is resolved with full-text review and experiments.

## 1. Layered view

```
 APPLICATION AI  (railway condition monitoring, digital twins, design assistants, ...)
        │  query: problem spec + inputs (+ observations) + budget
        ▼
 PHYSICS REFERENCE LAYER (API)        physref.reference_layer.PhysicsReference  [schema v0.1, implemented]
        │  returns ONLY verification-labelled results
        ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ 1 PHYSICS ANALYZER        invariants, orders, scales, spectra    physref.conditioning [implemented, beam]│
 │ 2 CONDITIONING ANALYZER   scalings, ansatz N*, Fourier coverage  physref.conditioning [implemented]      │
 │ 3 FORMULATION SELECTOR    strong / mixed / modal / weak / ...     formulations/* [strong, mixed, modal]   │
 │ 4 REPRESENTATION SELECTOR MLP / Fourier / SIREN / separable / modal          [Fourier, modal; others planned]│
 │ 5 TRAINING CONTROLLER     sampling, balancing, temporal plan, optimizer, precision, budget                │
 │                           sampling.R3 [impl.], temporal plans [impl.], arms trainer [impl.], gate [impl.] │
 │ 6 PHYSICS SOLVER          training / inference                    beampinn.Trainer, physref.arms          │
 │ 7 VERIFICATION ENGINE     residual, IC/BC, spectral, persistence  beampinn.evaluation, physref.persistence │
 └──────────────────────────────────────────────────────────────────────────────┘
        ▼
 PDE / ODE / OBSERVATIONS  →  VERIFIED PHYSICAL RESPONSE
```

Selectors 3–5 are **not automated**. Today a human chooses arms from a pre-registered matrix. Automating the
choice from analyzer outputs is hypothesis H0/H1 and is built only from rules that experiments support.

## 2. Stage contracts

| Stage | Input | Output | Exists now | Evidence needed before automation |
|---|---|---|---|---|
| Physics analyzer | linear PDE spec (term orders and coefficients, domain, BC type, fundamental wavenumber, amplitude) | scales, invariants (N_c, ζ, β₁L), term magnitudes, max orders, spectral content | yes (`LinearPDESpec`, `analyze`) | generalise to nonlinear and multi-field PDEs (Tier 4) |
| Conditioning analyzer | analyzer output + candidate representation | coefficient spread per scaling, N* per ansatz time factor, Fourier coverage, init loss/grad ratios | yes | H1: which indicators predict failure (E01–E05) |
| Formulation selector | invariants + budget | formulation choice | manual | H2/H3 (E01) |
| Representation selector | invariants + coverage | architecture | manual | E07 |
| Training controller | all of the above | sampler, weighting, temporal plan, optimizer, precision, budget | components exist; no controller | E02–E06, E09 |
| Solver | config | trained model + cost record | yes (approval-gated) | — |
| Verification | model + references (or residual-only when no reference exists) | gate results | yes for the beam | reference-free gates for Tiers 2–5 (residual, conservation, spectral consistency) |
| Reference API | verification + solution | `PhysicsReference` JSON | schema + validation | downstream integration tests |

## 3. Output contract (`src/physref/reference_layer.py`, schema 0.1.0)

```json
{
  "solution":            {"quantity": "u", "units": "m", "grid": {"x": "...", "t": "..."}, "values": "..."},
  "physical_parameters": {"pde": "c2 u_xxxx + u_tt + gamma u_t = 0", "c2": 1912.3129, "gamma": 7.08,
                          "bc": "fixed-fixed", "L": 2.75, "T": 1.0, "ic": "A0 phi_1, A0 = 0.08 m"},
  "residual":            {"PDE_residual_rel": 0.041, "grid": "51x501"},
  "constraint_error":    {"IC_error_max": 4.8e-6, "BC_error_max": 1.4e-5},
  "uncertainty":         {"method": "none"},
  "convergence_status":  "failed_verification",
  "verification":        {"gates": {"persistence_full_window": {"passed": false, "value_cycles": 8.1, "required": 20.58},
                                    "ic_bc": {"passed": true}}},
  "compute_cost":        {"train_seconds": 2290, "pde_evaluations": 2560000, "parameters": 241601,
                          "inference_us_per_point": "...", "hardware": "..."},
  "model_metadata":      {"formulation": "strong", "representation": "spatio-temporal Fourier 6x200",
                          "precision": "float32"},
  "provenance":          {"experiment_id": "...", "git_sha": "...", "config_sha256": "...",
                          "root_frozen_commit": "d31864a..."},
  "warnings":            ["collapse at t = 0.393 s"],
  "schema_version":      "0.1.0"
}
```

The illustrative numbers above are B1 (Batch-1 Z4-20K). Under this contract **B1 would be labelled
`failed_verification`**. That is the intended behaviour: a downstream AI system must never receive B1 as a
trusted reference.

Rules enforced by `validate()` (tested in `tests/test_b2_metrics_and_tools.py`):

- `verified` requires every declared gate to pass;
- solution, parameters, residual and constraint error must be populated;
- provenance must contain experiment_id, git_sha and config_sha256;
- uncertainty is `{"method": "none"}` until a validated UQ method exists. Error bars are never fabricated.

## 4. Railway pathway (prompt §21) — future work only

```
general solver strategy (Tiers 1–4 validated)
  → validated structural benchmark (EB, Mode 1 and Mode 2 stress test)
  → rail/beam dynamics: Timoshenko beam, beam on a Winkler/Pasternak foundation (Kapoor et al. 2023 setups)
  → vehicle/track: moving load (Kapoor et al. 2024), moving mass, coupled vehicle–track ODE/PDE
  → sensor-informed physics: sparse accelerometers/strain → state estimation (Haywood-Alexander et al. 2024)
  → condition monitoring / parameter estimation: stiffness/damping identification, track irregularity
  → axle/bearing SHM: vibration + temperature + acoustic-emission fusion with physics-referenced features
```

**No railway claim is made in Batch 2 preparation.** Each arrow requires a validated preceding stage.

## 5. What would make this architecture scientifically meaningful (and not just plumbing)

- The selectors (3–5) must be **driven by analyzer outputs through rules that experiments established**
  (H1, H0).
- The verification engine must **refuse** unverified results, with no silent fallback.
- Compute cost must be part of the returned record, so that consumers can trade accuracy for budget
  explicitly.
