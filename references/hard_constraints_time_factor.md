# Hard IC/BC ansatz with a frequency-informed time factor (part of baseline B1)

**Citations (closest literature):**
- Sukumar & Srivastava, *CMAME* (2022), arXiv:2104.08426: distance-function hard BCs.
- Lu et al., *SIAM J. Sci. Comput.* 43(6) (2021) B1105, arXiv:2102.04626: hard constraints.
- Chen et al., AT-PINN-HC, *CMAME* (2025) 117691, DOI 10.1016/j.cma.2024.117691: auxiliary functions for
  vibration IC/BC hard constraints, including an EB beam.

**Mechanism.**

    u = u0(x) + g(t) Φ(x) A0 N(x,t),   Φ = 16x²(L−x)²/L⁴,   g(0) = g'(0) = 0

All six IC/BC conditions hold for any N, so only the PDE residual is trained.
- Batch 1 replaced g = (t/T)² by g = tanh²(ω₁t), where ω₁ comes from the PDE coefficient and the eigenproblem.
- The required network output near t = 0 is N* ≈ −ω₁²/g''(0): −8.4e3 for (t/T)², −0.5 for tanh²(ω₁t).

**Exact source followed.** Our own derivation (Batch 1, `src/beampinn/models/constraints.py`). The
literature above was identified in this audit **after** Batch 1. No source implementation was copied.

**Deviations.** The base u0 is the known IC shape (a problem-specific, declared prior). ω₁ uses the exact
fixed–fixed root and is never fitted to the solution.

**Computational implications.** A few elementwise ops per point (negligible). It removes the multi-term
loss, so NTK weighting is unnecessary.

**Batch-1 outcome.**
- Conditioning was restored (X phase).
- With the LR schedule and mini-batch 128 (Z4-20K): 8.1 of 20.6 cycles, then collapse.

**Novelty status.**
- PARTIALLY EXPLORED / TOO CLOSE (AT-PINN-HC).
- Allowed use: a conditioning rule inside the analyzer ("choose g with g''(0) ≍ ω₁²"). Not a standalone
  claim.
