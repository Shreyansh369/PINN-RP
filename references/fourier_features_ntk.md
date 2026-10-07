# Spatio-temporal Fourier-feature PINN + NTK loss weighting (baseline B0)

**Citations:**
- Söyleyici & Ünver, *Eng. Appl. Artif. Intell.* 141 (2025) 109804, DOI 10.1016/j.engappai.2024.109804 (the
  target paper; Eqs. 37–49).
- Wang, Wang & Perdikaris, *CMAME* 384 (2021) 113938, arXiv:2012.10047 (spatio-temporal multiscale Fourier
  features).
- Wang, Yu & Perdikaris, *J. Comput. Phys.* 449 (2022) 110768, arXiv:2007.14527 (NTK weighting).
- Tancik et al., NeurIPS 2020, arXiv:2006.10739.

**Mechanism.**
- Separate random Fourier embeddings γ(x) and γ(t), with B ~ N(0, σ²), feed a shared tanh trunk. The
  outputs are combined by Hadamard products H_x ⊙ H_t and a linear head.
- Loss-term weights are λ_i = Σ_j tr K_j / tr K_i, where K_i is the NTK block of term i, recomputed every
  100 steps.

**Exact source followed.** Paper Eqs. 37–43, 48 and 49, Table 4 #12. The reference code
`Wave1D_NTK_ST_mFF` (MultiscalePINNs) supplies details the paper does not specify.

**Implementation.** Batch 1, frozen: `src/beampinn/models/networks.py`, `src/beampinn/losses/weighting.py`.
Configuration: `configs/frozen/baseline_fourier_ntk.yaml` (= Batch-1 run `C0_paper__s1234__c41ccb1cdf`).

**Deviations (all from Batch 1, documented in `references/batch1/reports/STAGE01_CORRECTIONS.md` §5).**
These choices are not specified by the paper:
- L_u composition: 320 IC + 320 BC points;
- NTK trace computed exactly on 32 rows per term;
- N(0,1) bias initialisation.

**Computational implications.**
- NTK update: 0.67 s per update on 6×200 (≈ 9 % overhead at every-100 steps).
- Trace cost grows superlinearly with rows.

**Batch-1 outcome.** Static attractor. NTK weights reached 1e7–1e15 depending on the variant (A1/D reports);
L2 3.26 at 20k steps.

**Novelty status.** ESTABLISHED. It is a published-method baseline and never our method.
