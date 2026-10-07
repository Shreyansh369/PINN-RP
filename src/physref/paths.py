"""Canonical locations for PINN-RP (Batch 2) and the frozen ROOT (Batch 1)."""
import json
import os
from pathlib import Path

PINNRP = Path(__file__).resolve().parents[2]
RESULTS_B2 = PINNRP / "results_batch2"
RESULT_SUBDIRS = ("runs", "checkpoints", "figures", "reports", "tables")
ROOT_RECORD = PINNRP / "configs" / "root_provenance.json"

# Directory names that belong to ROOT-derived output trees; Batch 2 never writes into them.
FORBIDDEN_OUTPUT_DIRNAMES = frozenset({"results", "results_optimization"})
# Remote-name fragment identifying a clone of the frozen ROOT repository.
ROOT_REMOTE_MARKER = "pinn-replication"


def root_record():
    with open(ROOT_RECORD) as f:
        return json.load(f)


def root_candidates():
    """Every path that must be treated as ROOT: $PINNRP_ROOT_PATH and the recorded path."""
    out = []
    env = os.environ.get("PINNRP_ROOT_PATH")
    if env:
        out.append(Path(env))
    try:
        out.append(Path(root_record()["root_path"]))
    except (OSError, KeyError, ValueError):
        pass
    return out
