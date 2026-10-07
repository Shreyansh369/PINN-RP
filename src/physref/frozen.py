"""Frozen reference baselines (configs/frozen/*.yaml). Integrity-checked on every load.

The YAML files are byte-for-byte protected by configs/frozen/FROZEN_HASHES.json and are
chmod read-only in the working tree. Each one reproduces the exact Batch-1 run key of the run
it was frozen from, so the baseline identity is verifiable (tests/test_b2_frozen.py)."""
import hashlib
import json

import yaml

from .paths import PINNRP

FROZEN_DIR = PINNRP / "configs" / "frozen"
HASHES = FROZEN_DIR / "FROZEN_HASHES.json"
BASELINES = {"baseline_fourier_ntk": "B0", "batch1_hard_tanh2": "B1"}


class FrozenIntegrityError(RuntimeError):
    pass


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_frozen(name):
    path = FROZEN_DIR / f"{name}.yaml"
    want = json.load(open(HASHES))[path.name]
    got = sha256(path)
    if got != want:
        raise FrozenIntegrityError(f"{path} was modified: sha256 {got} != frozen {want}")
    return yaml.safe_load(open(path))


def to_experiment_config(name):
    """Frozen YAML -> beampinn ExperimentConfig (the exact Batch-1 configuration)."""
    from beampinn.config import ExperimentConfig
    doc = load_frozen(name)
    return ExperimentConfig.from_dict(doc["config"]), doc
