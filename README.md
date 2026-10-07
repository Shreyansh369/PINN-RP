# PINN-RP — Batch 2: toward a compute-aware Physics Reference Layer

Batch-2 research repository. The long-term goal is a **Physics Reference Layer**: physically
constrained, verifiable, compute-efficient predictions that downstream AI systems can query.
PINNs are the first technology under study, the damped Euler–Bernoulli beam is the controlled
laboratory, and railway engineering is the first application domain (not yet started).

> **Status (2026-10-07): PREPARATION PHASE. No model has been trained in Batch 2.**
> All training is refused by `physref.gate` until the PI approves specific experiment IDs in
> `configs/batch2/APPROVED_EXPERIMENTS.txt`. Start with
> [`docs/BATCH2_READY_FOR_TRAINING.md`](docs/BATCH2_READY_FOR_TRAINING.md).

## Relationship to Batch 1 (ROOT)

Batch 1 is the frozen, **read-only** repository
[PINN-replication](https://github.com/Shreyansh369/PINN-replication) @ `d31864a`. Its findings are
treated as frozen evidence; it established **no** validated optimized PINN. See
[`docs/ROOT_PROVENANCE.md`](docs/ROOT_PROVENANCE.md). The guard in `src/physref/safety.py` aborts on
any output path inside ROOT, through a symlink into ROOT, or in a ROOT-style result tree.

## Layout

```
src/beampinn/           frozen Batch-1 package (identical to ROOT except utils/io.py output root)
src/physref/            Batch-2 infrastructure: safety, provenance, frozen baselines, conditioning
                        analyzer, mixed/modal formulations, derivative strategies, R3 sampling,
                        temporal plans, frozen persistence metric, cost model, output contract, arms
experiments/            run_batch2.py (dry-run by default; approval-gated) + Batch-1 leaderboard writer
configs/frozen/         B0 baseline_fourier_ntk.yaml, B1 batch1_hard_tanh2.yaml (hash-locked, read-only)
                        + batch1_json/ (all Batch-1 phase configs, read-only evidence)
configs/literature/     literature-method configuration notes (none trainable yet)
configs/batch2/         pre-registered Batch-2 experiment specs; APPROVED_EXPERIMENTS.txt (empty)
references/             <method>.md literature traceability; batch1/ frozen evidence + IMPORT_MANIFEST
docs/                   provenance, literature matrix, novelty gap, taxonomy, architecture,
                        experiment matrix, cost model, hypotheses, readiness report
results_batch2/         the ONLY place new outputs may be written (runs/checkpoints/figures/reports/tables)
scripts/                ROOT check, import manifest, baseline freezing, profiling, conditioning report
tests/                  103 Batch-1 tests + Batch-2 tests (safety, frozen integrity, equivalence, ...)
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/                       # all tests (no training)
python scripts/check_root_untouched.py        # ROOT unchanged?
python experiments/run_batch2.py --spec configs/batch2/B2-E01_mechanism_attribution.yaml --arm mixed   # dry run
```

## Frozen reference baselines (neither is our method)

| ID | File | What | Batch-1 result |
|---|---|---|---|
| B0 | `configs/frozen/baseline_fourier_ntk.yaml` | Published Fourier + NTK (paper-faithful C0) | static field, L2 3.26 @ 20k steps |
| B1 | `configs/frozen/batch1_hard_tanh2.yaml` | hard IC/BC + tanh²(ω₁t) + D4 Fourier + exp-decay LR + mini-batch 128 (Z4-20K) | L2_exact 0.263, 8.1 of 20.6 cycles, then collapse (FAIL) |
