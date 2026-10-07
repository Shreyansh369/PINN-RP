"""Build references/batch1/IMPORT_MANIFEST.json: for every file imported from the frozen ROOT
commit, record its ROOT path, git blob id, sha256 at the frozen commit, and whether the PINN-RP
copy is identical or MODIFIED (with reason). Reads ROOT only through `git show <commit>:<path>`
(read-only object access; no checkout, no index refresh).

    python scripts/build_import_manifest.py
"""
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REC = json.load(open(REPO / "configs" / "root_provenance.json"))
ROOT, COMMIT = REC["root_path"], REC["root_frozen_commit"]

MAPPING = [  # (ROOT dir or file, PINN-RP dir or file)
    ("src/beampinn", "src/beampinn"),
    ("tests", "tests"),
    ("configs", "configs/frozen/batch1_json"),
    ("results_optimization/reports", "references/batch1/reports"),
    ("results_optimization/tables", "references/batch1/tables"),
    ("results_optimization/configs", "references/batch1/run_configs"),
    ("experiments/update_leaderboard.py", "experiments/update_leaderboard.py"),
    ("optimization_leaderboard.csv", "references/batch1/optimization_leaderboard.csv"),
    ("paper_benchmark_registry.csv", "references/batch1/paper_benchmark_registry.csv"),
    ("REPORT.md", "references/batch1/LEGACY_REPORT.md"),
    ("presentation_summary.md", "references/batch1/presentation_summary.md"),
    ("requirements.txt", "requirements.txt"),
    ("pyproject.toml", "pyproject.toml"),
]
MODIFIED = {
    "src/beampinn/utils/io.py": "default output root results_optimization/ -> results_batch2/runs/; run_paths calls the Batch-2 ROOT-write guard",
    "experiments/update_leaderboard.py": "default board/log paths moved to results_batch2/",
    "tests/test_hard_rad_mode2.py": "Batch-1 run configs read from references/batch1/run_configs/ instead of results_optimization/configs/",
    "requirements.txt": "added pyyaml and pytest (Batch-2 tooling); Batch-1 pins unchanged",
    "pyproject.toml": "package discovery and project name for PINN-RP; Batch-1 pytest settings unchanged",
}


def git(*a):
    return subprocess.run(["git", "--no-optional-locks", "-C", ROOT, *a], capture_output=True, check=True).stdout


def main():
    tree = git("ls-tree", "-r", COMMIT).decode().splitlines()
    blobs = {line.split("\t", 1)[1]: line.split()[2] for line in tree}
    files = []
    for src, dst in MAPPING:
        for path, blob in sorted(blobs.items()):
            if path == src or path.startswith(src + "/"):
                rel = path[len(src):].lstrip("/")
                pin = f"{dst}/{rel}" if rel else dst
                if (src == "configs" and not rel.startswith("phase")) or path.endswith(".gitkeep"):
                    continue
                data = git("show", f"{COMMIT}:{path}")
                cur = (REPO / pin).read_bytes()
                st = "identical" if cur == data else "modified"
                e = {"root_path": path, "pinnrp_path": pin, "git_blob": blob,
                     "root_sha256": hashlib.sha256(data).hexdigest(), "status": st}
                if st == "modified":
                    if pin not in MODIFIED:
                        raise SystemExit(f"undeclared modification: {pin}")
                    e.update(pinnrp_sha256=hashlib.sha256(cur).hexdigest(), reason=MODIFIED[pin])
                files.append(e)
    metrics = sorted((REPO / "references/batch1/run_metrics").glob("*.metrics.json"))
    for m in metrics:
        rid = m.name[: -len(".metrics.json")]
        path = f"results_optimization/logs/{rid}/metrics.json"
        data = git("show", f"{COMMIT}:{path}")
        assert m.read_bytes() == data, m
        files.append({"root_path": path, "pinnrp_path": str(m.relative_to(REPO)), "git_blob": blobs[path],
                      "root_sha256": hashlib.sha256(data).hexdigest(), "status": "identical"})
    out = {"root_repository": REC["root_repository"], "root_frozen_commit": COMMIT,
           "method": "git archive of the frozen commit into a scratch staging dir outside both repos; "
                     "selected files copied into PINN-RP; this manifest is built from `git show` blobs",
           "n_files": len(files), "modified_files": sorted(e["pinnrp_path"] for e in files if e["status"] == "modified"),
           "files": files}
    (REPO / "references/batch1/IMPORT_MANIFEST.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(files)} files, modified: {out['modified_files']}")


if __name__ == "__main__":
    main()
