# ROOT provenance (Batch 1 → Batch 2 fork record)

**ROOT is READ-ONLY.** Nothing in this repository writes to it. Every output path passes
`physref.safety.assert_safe_output` (§7), and `scripts/check_root_untouched.py` verifies ROOT
before and after work.

## 1. ROOT identity

| Field | Value |
|---|---|
| ROOT repository | https://github.com/Shreyansh369/PINN-replication |
| ROOT absolute path (this machine) | `/home/user/PINN-replication` |
| ROOT branch at fork | `claude/tender-ramanujan-v3w90d` (the remote default `HEAD` also resolves to this commit) |
| **Frozen ROOT commit** | **`d31864a868c59858f99b9b0c5dab4b0f18758e25`** |
| Frozen tree | `73a939135a9799baea5a7fc4df151406c54f5fe7` |
| Commit | "Merge Batch 1: frozen Fourier/NTK beam-PINN reproducibility study and final presentation (#1)", 2026-10-07 05:49:30 +0530, Shreyansh Kandpal; parents `5790b53` (main) and `6b31d79` (Batch-1 branch head; identical tree) |
| Z4-20K state | Z4-20K report commit `d7e1461`; presentation built from it in `6b31d79`; both contained in `d31864a` |
| ROOT git status before Batch 2 | clean: no modified, staged or untracked files (`git status --porcelain` empty) |
| Fork / import time | 2026-10-07T00:21:26Z (initial audit) |

The machine-readable record is `configs/root_provenance.json`.

How the frozen commit was chosen: `d31864a` is the merge of the complete Batch-1 PR (#1). Its tree is
identical to `6b31d79` ("Add final presentation built from frozen artifacts"), which follows the final
Batch-1 experiment report `d7e1461` (Z4-20K, CASE B). No later Batch-1 commit exists on any ROOT branch
(`git ls-remote`: the only other refs are `main` = `5790b53`, the older legacy state, and
`claude/pinn-beam-vibration-opt-a946fl` = `6b31d79`).

## 2. ROOT project purpose and Batch-1 completion state

ROOT is a replication and controlled optimization study of Söyleyici & Ünver, "A Physics-Informed Deep
Neural Network based beam vibration framework for simulation and parameter identification",
*Engineering Applications of Artificial Intelligence* 141 (2025) 109804, DOI 10.1016/j.engappai.2024.109804.

Batch 1 is **complete and frozen**. Its final state (`references/batch1/reports/PHASE_Z4_20K_BUDGET_DIAGNOSTIC.md`):
**no validated optimized PINN.** Batch-1 findings are frozen evidence; see `docs/BATCH2_READY_FOR_TRAINING.md` §1
for the inventory with exact sources.

## 3. Relationship between ROOT and PINN-RP

- PINN-RP is an **independent** repository (https://github.com/Shreyansh369/PINN-RP) that had no commits when
  Batch 2 started. Its git history starts at Batch 2 and is **not** a fork of ROOT's history. Provenance is
  recorded by commit SHA, blob IDs and sha256 (`references/batch1/IMPORT_MANIFEST.json`).
- The frozen Batch-1 package `beampinn` is carried over so that Batch-2 baselines are bit-compatible with
  Batch-1 (the frozen baseline YAMLs reproduce Batch-1 run keys; `tests/test_b2_frozen.py`).
- New Batch-2 code lives in `src/physref/`.

## 4. Import procedure (non-destructive)

1. ROOT was only read: `git rev-parse`, `git --no-optional-locks status`, `git ls-tree`, `git show`,
   `git ls-remote`, `git archive`. No checkout, reset, fetch, branch change or index refresh. *(The initial
   audit used two plain `git status` calls; plain `git status` may refresh the stat cache in `.git/index`.
   The `.git/index` file was excluded from the integrity snapshot for that reason; every later call used
   `--no-optional-locks`. HEAD, branch, refs and the working tree are verified unchanged.)*
2. `git archive d31864a` was extracted to a scratch staging directory **outside both repositories**
   (session scratchpad). The staged copy passed its own 103 tests there. The staging directory was never
   used as an output path.
3. Selected files were copied into PINN-RP (§5). PINN-RP's `.git` was never re-initialised, re-cloned or
   overwritten.
4. `scripts/build_import_manifest.py` rebuilds `references/batch1/IMPORT_MANIFEST.json` from `git show
   d31864a:<path>` blobs and classifies every imported file as `identical` or `modified`.
   `tests/test_b2_frozen.py::test_import_manifest_matches_files` enforces it.

## 5. Imported files (149, all from `d31864a`)

| ROOT path | PINN-RP path | Status |
|---|---|---|
| `src/beampinn/**` (25 files) | `src/beampinn/**` | identical except `utils/io.py` |
| `tests/*.py` (6) | `tests/` | identical except `test_hard_rad_mode2.py` |
| `configs/phase{A,D,E,X,Y,Z}/*.json` (26) | `configs/frozen/batch1_json/` | identical |
| `results_optimization/reports/*.md` (11) | `references/batch1/reports/` | identical |
| `results_optimization/tables/*.csv` (14) | `references/batch1/tables/` | identical |
| `results_optimization/configs/*.json` (30 executed run configs) | `references/batch1/run_configs/` | identical |
| `results_optimization/logs/*/metrics.json` (30) | `references/batch1/run_metrics/<run>.metrics.json` | identical |
| `optimization_leaderboard.csv`, `paper_benchmark_registry.csv` | `references/batch1/` | identical |
| `REPORT.md` (legacy study) | `references/batch1/LEGACY_REPORT.md` | identical |
| `presentation_summary.md` | `references/batch1/` | identical |
| `experiments/update_leaderboard.py` | `experiments/` | modified |
| `requirements.txt`, `pyproject.toml` | repo root | modified |

## 6. Declared modifications of imported files (5)

| File | Change | Why |
|---|---|---|
| `src/beampinn/utils/io.py` | default output root `results_optimization/` → `results_batch2/runs/`; `run_paths` calls the Batch-2 ROOT guard | Batch-2 output isolation (master prompt §5–6) |
| `experiments/update_leaderboard.py` | default board/log paths → `results_batch2/` | same |
| `tests/test_hard_rad_mode2.py` | reads Batch-1 run configs from `references/batch1/run_configs/` | `results_optimization/` is not imported |
| `requirements.txt` | added `pyyaml`, `pytest` | Batch-2 tooling; Batch-1 pins unchanged |
| `pyproject.toml` | project name/version/description | PINN-RP packaging; pytest settings unchanged |

No physics, model, loss, sampler, metric or trainer code was changed. All 103 imported Batch-1 tests pass in PINN-RP.

## 7. Intentionally NOT copied

| ROOT content | Size | Reason |
|---|---|---|
| `results_optimization/checkpoints/**` (39 `.pt`) | 108 MB | Not needed to start Batch 2. Batch-1 numbers are taken from frozen reports/metrics. If a checkpoint is needed (for example, to re-evaluate Z4-20K under a new diagnostic), read it **read-only** with `git show d31864a:<path>` into `results_batch2/`; never symlink. |
| `results_optimization/figures/**`, `logs/**/history.csv` | 6 MB | Reproducible from ROOT on demand; reports cite them |
| `results/**` (legacy notebook outputs) | 4 MB | Legacy study, superseded |
| `beam_pinn_research.ipynb`, `notebook_src/**` | 0.4 MB | Legacy study |
| `experiments/*` except `update_leaderboard.py` | — | Batch-1 phase scripts hard-code `results_optimization/` output paths. Batch-2 uses `experiments/run_batch2.py`. The frozen persistence metric was ported verbatim to `src/physref/persistence.py` |
| presentation (`.pptx`, `presentation_build/`, `presentation_figures/`, `presentation_tables/`) | 1 MB | Batch-1 deliverable; summary imported |

## 8. Data provenance

There are no external datasets. Every reference solution is closed-form (`beampinn/physics/beam.py`,
`benchmarks.py`; dual paper-faithful and exact-physics references). Batch-1 run records are imported as
read-only evidence copies with sha256 verification. No experiment can overwrite them: they are outside
`results_batch2/`, which is the only place the strict guard allows writes to.

## 9. Runtime protection summary

`physref.safety.assert_safe_output` aborts, printing the attempted path, when an output path:

- is inside ROOT (the recorded path, `$PINNRP_ROOT_PATH`, or **any** git checkout whose origin remote is PINN-replication);
- reaches ROOT through a symlink;
- contains a `results/` or `results_optimization/` component; or
- in strict mode (the experiment runner), lies outside `results_batch2/`.

Tests: `tests/test_b2_safety.py`.
