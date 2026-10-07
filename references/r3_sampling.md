# Retain–Resample–Release (R3) sampling

**Citation.** Daw, Bu, Wang, Perdikaris & Karpatne, "Mitigating Propagation Failures in Physics-informed
Neural Networks using Retain-Resample-Release (R3) Sampling", ICML 2023, PMLR 202; arXiv:2207.02338.

**Mechanism.** Each update:
- **retain** the collocation points with |r| > mean |r|;
- **release** the rest;
- **resample** the released count uniformly.

The population accumulates at high-residual regions, such as a propagation front.

**Exact source followed.** The algorithm as described in the abstract and the paper's method summary. The
paper's causal extension (a time gate) is **not** implemented.

**Implementation.** `src/physref/sampling.py` (`R3Sampler`); tested for size preservation, retention of
high-residual points and determinism. It is not yet wired into a trainer (E04, pending approval).

**Deviations.** Non-causal variant only. The update cadence (every epoch) is our choice and declared in E04.

**Computational implications.** One residual evaluation of the current population per update (640 points);
no candidate pool, unlike RAD.

**Novelty status.** ESTABLISHED. It is distinct from RAD (tested in Batch 1) and must not be described as
RAD.
