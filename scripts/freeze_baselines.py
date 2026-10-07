"""ONE-TIME generator of configs/frozen/{baseline_fourier_ntk,batch1_hard_tanh2}.yaml from the
exact executed Batch-1 run configs (references/batch1/run_configs/). Refuses to overwrite an
existing frozen file. Kept for provenance; re-running it after freezing is a no-op error.

    python scripts/freeze_baselines.py
"""
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "configs" / "frozen"
SRC = REPO / "references" / "batch1" / "run_configs"

SPECS = {
    "baseline_fourier_ntk": dict(
        baseline_id="B0", run_id="C0_paper__s1234__c41ccb1cdf",
        title="Published-method baseline: paper-faithful Fourier features + NTK weighting (C0)",
        role="Published-method reference (Soyleyici & Unver 2025, EAAI 141:109804). NOT our method.",
        batch1_result=dict(source="references/batch1/run_metrics/C0_paper__s1234__c41ccb1cdf.metrics.json",
                           L2_paper=3.2588899126058393, L2_exact=3.258904583751413, steps=20000,
                           pde_evaluations=640000, train_seconds=1575.17,
                           behaviour="static field (0 cycles); NTK weights grew to ~1e14-1e15 (Phase A/D reports)")),
    "batch1_hard_tanh2": dict(
        baseline_id="B1", run_id="Z4_20K__s1234__fa8fa7fe7e",
        title="Batch-1 best candidate: hard IC/BC ansatz + tanh^2(w1 t) + D4 Fourier convention + exp-decay LR + mini-batch 128 (Z4-20K)",
        role="Strongest Batch-1 diagnostic configuration. NOT a validated method and NOT our claimed method.",
        batch1_result=dict(source="references/batch1/reports/PHASE_Z4_20K_BUDGET_DIAGNOSTIC.md",
                           L2_exact=0.2633298147534262, L2_paper=0.2633223957613705, steps=20000,
                           pde_evaluations=2560000, train_seconds=2290.09, fit_w_rad_s=129.2,
                           collapse_time_s=0.393, persistence_cycles=8.1, full_window_cycles=20.6,
                           behaviour="correct frequency in reproduced cycles; collapses at 0.393 s; over-damped (decay 5.40 vs 3.54 1/s); FAIL")),
}


def main():
    hashes_path = FROZEN / "FROZEN_HASHES.json"
    hashes = json.load(open(hashes_path)) if hashes_path.exists() else {}
    for name, spec in SPECS.items():
        out = FROZEN / f"{name}.yaml"
        if out.exists():
            sys.exit(f"{out} already frozen; refusing to overwrite")
        cfg = json.load(open(SRC / f"{spec['run_id']}.json"))
        doc = {"frozen": True, "baseline_id": spec["baseline_id"], "title": spec["title"],
               "role": spec["role"], "source_root_commit": "d31864a868c59858f99b9b0c5dab4b0f18758e25",
               "source_run_id": spec["run_id"],
               "source_run_config": f"results_optimization/configs/{spec['run_id']}.json (ROOT)",
               "batch1_result": spec["batch1_result"], "config": cfg}
        header = (f"# FROZEN REFERENCE BASELINE {spec['baseline_id']} - DO NOT EDIT.\n"
                  f"# Integrity: sha256 in FROZEN_HASHES.json; verified by physref.frozen and tests.\n"
                  f"# The `config` block is the exact executed Batch-1 config of run {spec['run_id']};\n"
                  f"# loading it reproduces that run key.\n")
        out.write_text(header + yaml.safe_dump(doc, sort_keys=False, width=110))
        hashes[out.name] = hashlib.sha256(out.read_bytes()).hexdigest()
        os.chmod(out, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    hashes_path.write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    main()
