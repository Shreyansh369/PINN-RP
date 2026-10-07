"""Training approval gate. Batch 2 is in the PREPARATION phase: no training is approved.

Training is allowed only for an experiment ID listed in configs/batch2/APPROVED_EXPERIMENTS.txt,
a file that is edited only after explicit written approval from the PI."""
from .paths import PINNRP

APPROVALS = PINNRP / "configs" / "batch2" / "APPROVED_EXPERIMENTS.txt"


class TrainingNotApproved(PermissionError):
    pass


def approved_ids():
    if not APPROVALS.exists():
        return set()
    ids = set()
    for line in APPROVALS.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.add(line)
    return ids


def require_approval(experiment_id):
    if experiment_id not in approved_ids():
        raise TrainingNotApproved(
            f"{experiment_id} is not in {APPROVALS.name}; Batch 2 training requires explicit approval")
