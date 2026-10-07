# Nondimensionalisation and conditioning analysis

**Citations:**
- Standard dimensional analysis.
- Kapoor, Wang, Núñez & Dollevoet, *IEEE TNNLS* 35(5) (2023) 5981, DOI 10.1109/TNNLS.2023.3310585
  (nondimensional EB/Timoshenko PINNs).
- Wang, Sankaran, Wang & Perdikaris, "An Expert's Guide to Training PINNs", arXiv:2308.08468.
- De Ryck et al., ICLR 2024, arXiv:2310.05801 (operator preconditioning).
- Liu et al., arXiv:2402.00531 (condition-number diagnosis).

**Mechanism.**
- Characteristic scales (1/β₁, 1/ω₁, A0) map the PDE to O(1) coefficients.
- The dimensionless invariants (N_c, ζ, β₁L) are scale-independent.
- Further diagnostics: the ansatz output scale N*, Fourier coverage, and init loss/gradient ratios.

**Implementation.** `src/physref/conditioning.py` (analysis only; it chooses nothing). Report:
`results_batch2/reports/B2-DIAG-001_conditioning_FE-D-M1.md`.

**Deviations.** Not applicable. This is a diagnostic tool, not a reproduction.

**Computational implications.** Negligible (closed form, plus one forward/backward at initialisation for
the term statistics).

**Novelty status.** ESTABLISHED as practice. Any contribution would come only from validated predictive use
(gap D2, hypothesis H1).
