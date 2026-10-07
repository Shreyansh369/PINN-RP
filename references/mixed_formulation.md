# Mixed (auxiliary-variable) formulation of the Euler–Bernoulli beam

**Citations:**
- Lyu, Zhang, Chen & Chen, "MIM: A deep mixed residual method for solving high-order PDEs", *J. Comput.
  Phys.* 452 (2022) 110930 (DOI from memory; verify).
- Yuan et al., "A-PINN: auxiliary physics informed neural networks…", *J. Comput. Phys.* (2022).
- **Closest:** "A-PINN: Auxiliary physics-informed neural networks for structural vibration analysis in
  continuous Euler–Bernoulli beam", arXiv:2601.00866, *Applied Soft Computing* (2026). Authors are not
  verified in this audit.
- Coupled scheme for the biharmonic equation: arXiv:2509.15004.

**Mechanism.**

    v = u_xx;   r_link = c²β₁²(v − u_xx);   r_pde = c²v_xx + u_tt + γu_t

The highest network derivative falls from 4 to 2. Equivalence holds when r_link ≡ 0 (tested).

**Exact source followed.** None copied. This is the standard construction, derived in
`docs/PHYSICS_SOLVER_TAXONOMY.md` §2.

**Implementation.** `src/physref/formulations/mixed.py`.
- `TwoHeadFourierPINN` is the Batch-1 network with a 2-output head; the u-head equals the B1 head at
  initialisation.
- `MixedHardFF`: B1 hard ansatz for u; v = u₀'' + g(t)β₁²A0N_v.
- `mixed_residuals`.

**Deviations from A-PINN EB (as described in its abstract).**
- We keep the B1 ansatz, representation, optimizer and schedule fixed, to isolate the formulation (one
  change).
- No Adam/L-BFGS switching.
- r_link uses a dimensional scaling, not a tuned weight.

**Computational implications (measured, B2-PROF-001).**
- 70 ms vs 154 ms per residual+backward step: 0.45×.
- Parameters +0.2 %.

**Novelty status.** ESTABLISHED / TOO CLOSE. Used as an E01 arm. Only the matched-compute measurement could
be a contribution.
