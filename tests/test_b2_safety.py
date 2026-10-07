"""ROOT write-protection and output-path safety (Batch-2 guard)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from physref.paths import PINNRP, RESULTS_B2, root_record
from physref.safety import RootWriteError, assert_safe_output, check_output_path

ROOT = Path(root_record()["root_path"])
needs_root = pytest.mark.skipif(not ROOT.exists(), reason="ROOT checkout not present on this machine")


@needs_root
@pytest.mark.parametrize("sub", ["", "results_optimization", "results_optimization/checkpoints/x",
                                 "src/beampinn", "new_dir/deeper"])
def test_root_and_subpaths_rejected(sub):
    with pytest.raises(RootWriteError) as e:
        assert_safe_output(ROOT / sub)
    assert str(ROOT) in str(e.value)                     # attempted path is printed


def test_env_root_override_rejected(tmp_path, monkeypatch):
    fake_root = tmp_path / "frozen_root"
    fake_root.mkdir()
    monkeypatch.setenv("PINNRP_ROOT_PATH", str(fake_root))
    with pytest.raises(RootWriteError):
        assert_safe_output(fake_root / "out")


def test_symlink_into_root_rejected(tmp_path, monkeypatch):
    fake_root = tmp_path / "frozen_root"
    (fake_root / "results_optimization").mkdir(parents=True)
    monkeypatch.setenv("PINNRP_ROOT_PATH", str(fake_root))
    link = tmp_path / "innocent_name"
    link.symlink_to(fake_root / "results_optimization")
    v = check_output_path(link / "run1")
    assert any("symlink" in s for s in v) and any("inside ROOT" in s for s in v)
    with pytest.raises(RootWriteError):
        assert_safe_output(link / "run1")


def test_any_clone_of_root_repo_rejected(tmp_path):
    clone = tmp_path / "somewhere" / "my_clone"
    (clone / ".git").mkdir(parents=True)
    (clone / ".git" / "config").write_text('[remote "origin"]\n\turl = https://github.com/Shreyansh369/PINN-replication\n')
    with pytest.raises(RootWriteError):
        assert_safe_output(clone / "sub" / "out")


@pytest.mark.parametrize("name", ["results", "results_optimization"])
def test_root_style_result_dirs_rejected_even_in_pinnrp(name):
    with pytest.raises(RootWriteError):
        assert_safe_output(PINNRP / name / "run")


def test_strict_mode_requires_results_batch2(tmp_path):
    assert check_output_path(tmp_path / "x", strict=False) == []
    assert check_output_path(tmp_path / "x", strict=True)
    for sub in ("runs", "checkpoints", "figures", "reports", "tables"):
        assert check_output_path(RESULTS_B2 / sub / "E", strict=True) == []
    with pytest.raises(RootWriteError):
        assert_safe_output(PINNRP / "src" / "x", strict=True)


def test_beampinn_run_paths_uses_guard(tmp_path, monkeypatch):
    from beampinn.utils.io import RESULTS, run_paths
    assert RESULTS == PINNRP / "results_batch2" / "runs"
    fake_root = tmp_path / "frozen_root"; fake_root.mkdir()
    monkeypatch.setenv("PINNRP_ROOT_PATH", str(fake_root))
    with pytest.raises(RootWriteError):
        run_paths("x", fake_root)
    assert not (fake_root / "configs").exists()          # nothing created before aborting


@needs_root
def test_root_is_unchanged_at_frozen_commit():
    r = subprocess.run([sys.executable, str(PINNRP / "scripts" / "check_root_untouched.py")],
                       capture_output=True, text=True)
    rep = json.loads(r.stdout)
    assert r.returncode == 0 and rep["ROOT_MODIFIED"] == "NO", r.stdout
