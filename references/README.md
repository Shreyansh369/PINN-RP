# Literature traceability (prompt §30)

One file per literature method that is **implemented** in this repository (as code, frozen baseline or
dry-run arm). Every file records:

- the citation and DOI/arXiv;
- the mathematical mechanism;
- implementation notes and the exact source followed;
- deviations and the reasons for them;
- computational implications;
- novelty status.

A method listed only in `docs/LITERATURE_MATRIX.md` has no file until it is implemented.

| File | Method | Code | Role |
|---|---|---|---|
| fourier_features_ntk.md | spatio-temporal Fourier PINN + NTK weighting | `beampinn/models/networks.py`, `beampinn/losses/weighting.py` | baseline B0 (frozen) |
| hard_constraints_time_factor.md | hard IC/BC ansatz with a time factor | `beampinn/models/constraints.py` | part of B1 (frozen) |
| adam_exp_decay.md | Adam with exponential LR decay | `beampinn/optimization/optimizers.py` | part of B1 (frozen) |
| rad.md | residual-based adaptive distribution | `beampinn/sampling/rad.py` | tested in Batch 1 (negative) |
| mixed_formulation.md | auxiliary-variable (mixed) residual | `physref/formulations/mixed.py` | E01 arm |
| modal_reduction.md | Galerkin modal reduction | `physref/formulations/modal.py` | E01 diagnostic arm |
| r3_sampling.md | retain–resample–release sampling | `physref/sampling.py` | E04 arm (dry-run) |
| temporal_decomposition.md | temporal windows with exact hand-over | `physref/temporal.py` | E03 (planning only) |
| forward_mode_autodiff.md | forward-mode derivatives | `physref/derivatives.py` | E08 profiling |
| nondimensionalization_conditioning.md | characteristic scales / nondimensionalisation | `physref/conditioning.py` | analyzer |

`batch1/` holds the frozen Batch-1 evidence (reports, tables, run configs, run metrics, leaderboard) with
`IMPORT_MANIFEST.json`.
