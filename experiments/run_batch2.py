"""Batch-2 experiment runner. DRY-RUN BY DEFAULT.

    python experiments/run_batch2.py --spec configs/batch2/B2-E01_mechanism_attribution.yaml --arm mixed [--seed 1234]
    ... --execute     # refused unless the experiment ID is in configs/batch2/APPROVED_EXPERIMENTS.txt

Every executed run: approval gate -> strict ROOT guard -> exclusive run dir under
results_batch2/runs/<experiment_id>/ (IDs never reused) -> record.json (git SHA, config hash,
environment, hardware, ROOT provenance) -> training -> metrics.json.
"""
import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

from physref.frozen import to_experiment_config  # noqa: E402
from physref.gate import approved_ids, require_approval  # noqa: E402


def plan(spec_path, arm, seed):
    spec = yaml.safe_load(open(spec_path))
    if arm not in spec["arms"]:
        raise SystemExit(f"arm {arm!r} not in {list(spec['arms'])}")
    cfg, doc = to_experiment_config(spec["base_frozen"])
    cfg.train.max_steps, cfg.train.budget_label = spec["max_steps"], "B2"
    cfg.seed = seed
    cfg.validate()
    exp_id = f"{spec['experiment']}-{arm}-s{seed}"
    steps = cfg.total_steps()
    return spec, cfg, exp_id, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    spec, cfg, exp_id, steps = plan(a.spec, a.arm, a.seed)
    print(f"experiment {exp_id} | arm {a.arm}: {spec['arms'][a.arm]['change']}")
    print(f"  base {spec['base_frozen']} (B1 run key {cfg.run_id()}), {steps:,} steps x mini-batch "
          f"{cfg.sampler.mini_batch} = {steps * cfg.sampler.mini_batch:,} PDE evaluations, {cfg.precision}, "
          f"{cfg.threads} thread(s)")
    print(f"  approved for training: {exp_id in approved_ids()}")
    if not a.execute:
        print("  DRY RUN - nothing trained, nothing written.")
        return
    require_approval(exp_id)
    from physref.provenance import allocate_run_dir, record
    spec_sha = hashlib.sha256(open(a.spec, "rb").read()).hexdigest()
    run_dir = allocate_run_dir(exp_id, record(exp_id, a.spec, spec_sha, a.seed, spec["arms"][a.arm]["change"]))
    if a.arm == "B1":
        from beampinn.training.trainer import Trainer
        tr = Trainer(cfg, root=run_dir)
        tr.run()
        return
    from physref.arms import ArmTrainer
    tr = ArmTrainer(cfg, a.arm, run_dir.parent, exp_id, strict=True)
    if tr.run(steps, snapshot_every=spec.get("snapshot_every")) == "completed":
        tr.finalise()


if __name__ == "__main__":
    main()
