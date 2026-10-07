# Temporal domain decomposition with exact hand-over

**Citations:**
- Krishnapriyan et al., NeurIPS 2021, arXiv:2109.01050 (seq2seq time marching).
- Mattey & Ghosh, *CMAME* 390 (2022) (bc-PINN).
- Penwarden et al., *J. Comput. Phys.* (2023), arXiv:2302.14227 (stacked decomposition, transfer learning,
  time sweeping).
- Roy et al., *CMAME* (2024), "Exact enforcement of temporal continuity in sequential PINNs".
- Chen et al., AT-PINN (*Thin-Walled Structures*, 2024) and AT-PINN-HC (*CMAME*, 2025), structural vibration.

**Mechanism.** Split [0,T] into K windows. Each window's IC (u, u_t) is the previous window's terminal state,
optionally enforced exactly. Weights are initialised by transfer from the previous window.

**Implementation (planning only).** `src/physref/temporal.py`:
- `make_plan`, `validate_plan`;
- `interface_residuals`, which checks both u and u_t continuity.

The windowed trainer is not yet written (E03 is conditional on E01).

**Deviations.** To be declared in the E03 configuration.

**Computational implications.** K sequential trainings, so the total E is matched to the baseline. Error
accumulates across windows (lightly damped: errors roughly add).

**Novelty status.** ESTABLISHED (AT-PINN-HC is directly on EB beams). Never claimed.
