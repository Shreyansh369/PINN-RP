# Galerkin modal reduction (modal PINN arm)

**Citations:**
- Classical modal superposition (structural dynamics textbooks).
- Zhang, Vlachas & Chatzi, "Reduced-Order Physics-Informed Neural Network with Adaptive Basis Refinement
  for Structural Identification", arXiv:2608.17131 (2026).
- ModalPINN (periodic flows) — reference from memory.

**Mechanism.**

    u = Σ A0 φ_n(x) q_n(t)   with   q_n'' + γq_n' + c²β_n⁴ q_n = 0,   q_n(0) = ⟨u₀,φ_n⟩/(A0⟨φ_n,φ_n⟩),   q_n'(0) = 0

**Exact source followed.** Our derivation (`docs/PHYSICS_SOLVER_TAXONOMY.md` §2). The numerical Galerkin
check is `galerkin_modal_system`.

**Implementation.** `src/physref/formulations/modal.py`.
- `TemporalFourierNet` = the B1 temporal branch.
- `ModalHardQ`: q = 1 + tanh²(ω₁t) N(t).
- `ModalField` exposes (x,t)→u for the unchanged Batch-1 evaluation.

**Deviations.** None relevant. The exact mode shape is used, which for FE-D-M1 contains the exact solution.
**This makes the arm a diagnostic, not a competitor** (stated in code, docs and every report).

**Computational implications.**
- Measured 36 ms vs 154 ms per step.
- Inference reuses q(t) across all x (expected ≈ N_x-fold fewer trunk passes on a grid; to be measured).

**Novelty status.** ESTABLISHED. It is never claimed.
