# Figures used in the research-story deck, and their sources

Every graph is a **native PowerPoint chart** built from data that is either:
- extracted from frozen artifacts by `scripts/presentation/extract_deck_data.py` and saved in `data/deck_data.json`, or
- read directly from frozen tables at build time.

No pre-rendered Batch-1 image is reused. Each graph poses an explicit question; the QA script checks for it.

| # | Slide | Question the figure answers | Data | Source artifact |
|---|---|---|---|---|
| 1 | 5 | *How close does each stage get to the real vibration?* | mid-span displacement u(L/2, t), t ∈ [0, 1] s, 2 001 points: exact, C0, X2, Z4, Z4-20K. Collapse markers for Z4 and Z4-20K. | Inference on frozen, git-tracked checkpoints in ROOT (read-only): `C0_paper__s1234__c41ccb1cdf/final.pt` (20k), `X2_hard_tanh2_rc__s1234__2c89a6ce41/final.pt` (5k), `Z4_Y1_mb128__s1234__f78aa7f1da/final.pt` (5k), `Z4_20K__s1234__fa8fa7fe7e/step_20000.pt`. Exact curve: analytical reference (`beampinn.physics`). The recomputed collapse times (0.15325 s, 0.39325 s) equal `references/batch1/tables/phaseZ4_20K_checkpoints.csv` exactly. |
| 2 | 8 | *Does additional physics computation delay dynamic collapse?* | cycles before collapse vs cumulative PDE evaluations at 5k/10k/15k/20k for the Y1 and Z4 recipes, plus the 20.6-cycle target line | `references/batch1/tables/phaseY1_20K_checkpoints.csv`, `phaseZ4_20K_checkpoints.csv` |
| 3 | 6 (panel B) | *How large must the network output be?* | required network output \|N*(t)\| at mid-span for g = (t/T)² and g = tanh²(ω₁t), log scale | Computed from the exact solution with the formula of Batch-1 `experiments/phaseE_ansatz_conditioning.py`. Ranges [−8368.7, −1.02] and [−1.93, −0.158] equal frozen `profiles/phaseE_ansatz_conditioning.txt`. |
| 4 | 4 | *What did we test, and in what order?* | experiment ladder (diagram) | Batch-1 phase reports (`references/batch1/reports/`); run count from `optimization_leaderboard.csv` |
| 5 | 9 | *What is Batch 2?* | Batch-2 pipeline (diagram) | `docs/PHYSICS_REFERENCE_LAYER_ARCHITECTURE.md`, `docs/NOVELTY_GAP.md` |
| — | 6 (panels A, C) | *Does the published setup vibrate? Where in time is the vibration lost?* | C0 vs exact (0–0.3 s); Z4-20K vs exact (0–1 s) with collapse | same as Figure 1 |
| — | 3 | (schematic) | fixed–fixed Mode-1 shape | analytical (`beampinn.physics`) |
| — | 2, 10 | (schematics) | cost/trust map; Physics Reference Layer hub and roadmap | conceptual diagrams, no data |

## Previous images, deliberately not used

The earlier ROOT deck used the following images. They were replaced because they did not answer one clear question for a non-specialist audience:
- `fig_C0_midspan_static.png`;
- `fig_X3/X4_supervised_trace.png`;
- `fig_Z4_20K_step20k_trace.png`;
- the NTK-weight chart.

ROOT figures were not modified.
