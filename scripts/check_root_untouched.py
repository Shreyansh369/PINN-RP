"""Verify that ROOT (frozen Batch-1 repo) is unchanged: HEAD == recorded frozen commit, branch
unchanged, and a clean working tree. Read-only: git is called with --no-optional-locks so not
even the index stat cache is refreshed. Exit code 1 on any deviation.

    python scripts/check_root_untouched.py [--root /path/to/PINN-replication]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REC = Path(__file__).resolve().parents[1] / "configs" / "root_provenance.json"


def git(root, *a):
    r = subprocess.run(["git", "--no-optional-locks", "-C", str(root), *a], capture_output=True, text=True)
    return r.stdout.strip(), r.returncode


def main():
    rec = json.load(open(REC))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=rec["root_path"])
    a = ap.parse_args()
    head, rc = git(a.root, "rev-parse", "HEAD")
    if rc:
        print(f"ROOT not found at {a.root}"); return 2
    branch, _ = git(a.root, "branch", "--show-current")
    status, _ = git(a.root, "status", "--porcelain=v1", "--untracked-files=all")
    ok = head == rec["root_frozen_commit"] and branch == rec["root_branch_at_fork"] and status == ""
    print(json.dumps({"root_path": a.root, "head": head, "expected_head": rec["root_frozen_commit"],
                      "branch": branch, "expected_branch": rec["root_branch_at_fork"],
                      "status": status or "clean", "ROOT_MODIFIED": "NO" if ok else "YES"}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
