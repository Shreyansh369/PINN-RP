# Residual-based adaptive distribution (RAD)

**Citation.** Wu, Zhu, Tan, Kartha & Lu, *CMAME* 403 (2023) 115671, arXiv:2207.10289.

**Mechanism.** Every n steps, draw the PDE collocation set from p(x) ∝ |r(x)|^k / E|r|^k + c over a
candidate pool.

**Implementation.** `src/beampinn/sampling/rad.py` (Batch 1):
- k = 1, c = 1;
- 5 000–10 000 candidates;
- multinomial without replacement;
- detached residuals;
- epoch-aligned updates.

**Deviations.** Epoch alignment (updates only at epoch boundaries) keeps the Batch-1 sampler semantics.

**Computational implications.** Candidate residual evaluations (4th-order derivatives) every update.
Measured RAD overhead is in the Batch-1 Z1 run records.

**Batch-1 outcome.** It did not resolve the late-time collapse (Z1: 1.6 vs 1.2 cycles at 5k steps;
Z3: no rescue of B0).

**Novelty status.** ESTABLISHED. Tested and negative here. It must not be renamed or re-proposed.
