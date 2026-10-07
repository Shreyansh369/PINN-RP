"""physref: Batch-2 research scaffolding for a compute-aware physics reference layer (PINN-RP).

This package holds Batch-2 infrastructure (ROOT-write guard, provenance, frozen baselines,
conditioning diagnostics, alternative formulations, derivative strategies, sampling, metrics,
cost accounting, output contract). It contains NO claimed novel method; see docs/NOVELTY_GAP.md.
The frozen Batch-1 package `beampinn` is imported unchanged except `beampinn/utils/io.py`.
"""
__version__ = "0.1.0-b2prep"
