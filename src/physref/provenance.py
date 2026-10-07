"""Experiment records: unique IDs, code version, environment, hardware. Never overwrites."""
import csv
import datetime as _dt
import json
import platform
import subprocess
import sys
from pathlib import Path

from .paths import PINNRP, RESULTS_B2, root_record
from .safety import assert_safe_output

REGISTRY = RESULTS_B2 / "EXPERIMENT_REGISTRY.csv"
REGISTRY_FIELDS = ["experiment_id", "created_utc", "config_path", "config_sha256", "git_sha",
                   "git_dirty", "seed", "status", "notes"]


def git_state(repo=PINNRP):
    """(HEAD sha or 'no-commits', dirty flag). Uses --no-optional-locks so it never writes.

    dirty = any modified/staged TRACKED file, or any untracked file OUTSIDE results_batch2/.
    Untracked experiment outputs under results_batch2/ (run dirs, launch logs of concurrently
    running experiments) do not mark the CODE as dirty. (B2-E01 recorded git_dirty=True only
    because of such outputs; changed 2026-10-07 before the B2-E01 seed replication.)"""
    def run(*a):
        return subprocess.run(["git", "--no-optional-locks", "-C", str(repo), *a],
                              capture_output=True, text=True)
    head = run("rev-parse", "HEAD")
    sha = head.stdout.strip() if head.returncode == 0 else "no-commits"
    lines = [l for l in run("status", "--porcelain", "--untracked-files=all").stdout.splitlines() if l.strip()]
    dirty = any(not (l.startswith("??") and l[3:].startswith("results_batch2/")) for l in lines)
    return sha, dirty


def environment():
    import numpy
    import scipy
    import torch
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__,
            "torch_threads": torch.get_num_threads(), "cuda_available": torch.cuda.is_available()}


def hardware():
    from beampinn.profiling.resources import hardware_info
    return hardware_info()


def record(experiment_id, config_path, config_sha256, seed, notes=""):
    sha, dirty = git_state()
    return {"experiment_id": experiment_id,
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "config_path": str(config_path), "config_sha256": config_sha256,
            "git_sha": sha, "git_dirty": dirty, "seed": seed, "status": "allocated", "notes": notes,
            "environment": environment(), "hardware": hardware(),
            "root_provenance": root_record()}


def registered_ids(registry=REGISTRY):
    if not Path(registry).exists():
        return set()
    with open(registry) as f:
        return {r["experiment_id"] for r in csv.DictReader(f)}


def allocate_run_dir(experiment_id, rec, registry=REGISTRY, base=RESULTS_B2 / "runs"):
    """Create results_batch2/runs/<experiment_id>/ EXCLUSIVELY (never reuses or overwrites) and
    append the record to the registry. Raises FileExistsError if the ID was ever used."""
    if experiment_id in registered_ids(registry):
        raise FileExistsError(f"experiment id {experiment_id} already registered; IDs are never reused")
    d = assert_safe_output(Path(base) / experiment_id, strict=True, purpose=experiment_id)
    d.mkdir(parents=True, exist_ok=False)
    with open(d / "record.json", "x") as f:
        json.dump(rec, f, indent=2, default=str)
    new = not Path(registry).exists()
    with open(registry, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(rec)
    return d
