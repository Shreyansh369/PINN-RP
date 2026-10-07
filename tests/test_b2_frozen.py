"""Frozen-baseline integrity, import integrity, config reproducibility, run-key integrity."""
import hashlib
import json
import os
import shutil
import stat

import pytest
import yaml

from physref import frozen
from physref.paths import PINNRP


@pytest.mark.parametrize("name,run_id", [("baseline_fourier_ntk", "C0_paper__s1234__c41ccb1cdf"),
                                         ("batch1_hard_tanh2", "Z4_20K__s1234__fa8fa7fe7e")])
def test_frozen_yaml_reproduces_batch1_run_key(name, run_id):
    cfg, doc = frozen.to_experiment_config(name)
    assert doc["frozen"] is True and doc["source_run_id"] == run_id
    assert cfg.run_id() == run_id                      # exact Batch-1 identity
    rec = json.load(open(PINNRP / "references" / "batch1" / "run_configs" / f"{run_id}.json"))
    assert rec == doc["config"]                        # verbatim copy of the executed config


def test_frozen_files_are_hash_locked_and_read_only():
    hashes = json.load(open(frozen.HASHES))
    assert set(hashes) == {"baseline_fourier_ntk.yaml", "batch1_hard_tanh2.yaml"}
    for fn, h in hashes.items():
        p = frozen.FROZEN_DIR / fn
        assert hashlib.sha256(p.read_bytes()).hexdigest() == h
        assert not (p.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)), f"{fn} is writable"


def test_tampering_is_detected(tmp_path, monkeypatch):
    d = tmp_path / "frozen"; d.mkdir()
    for fn in ("baseline_fourier_ntk.yaml", "FROZEN_HASHES.json"):
        shutil.copy(frozen.FROZEN_DIR / fn, d / fn)
    p = d / "baseline_fourier_ntk.yaml"
    os.chmod(p, 0o644)
    p.write_text(p.read_text().replace("lr: 0.0001", "lr: 0.001"))
    monkeypatch.setattr(frozen, "FROZEN_DIR", d)
    monkeypatch.setattr(frozen, "HASHES", d / "FROZEN_HASHES.json")
    with pytest.raises(frozen.FrozenIntegrityError):
        frozen.load_frozen("baseline_fourier_ntk")


def test_baselines_are_what_the_protocol_says():
    b0, _ = frozen.to_experiment_config("baseline_fourier_ntk")
    assert (b0.loss.weighting, b0.loss.hard_constraints, b0.model.two_pi, b0.model.input_norm) == ("ntk", "none", True, "physical")
    b1, _ = frozen.to_experiment_config("batch1_hard_tanh2")
    assert b1.loss.hard_constraints == "ff_tanh2" and b1.loss.weighting == "fixed"
    assert (b1.model.two_pi, b1.model.input_norm, b1.model.bias_init) == (False, "standardize", "zeros")   # D4
    assert b1.optim.schedule == "exp_decay" and b1.sampler.mini_batch == 128


def test_import_manifest_matches_files():
    """Every imported file is byte-identical to the frozen ROOT blob, except the files the
    manifest declares as MODIFIED (whose current sha256 must equal the recorded one)."""
    man = json.load(open(PINNRP / "references" / "batch1" / "IMPORT_MANIFEST.json"))
    assert man["root_frozen_commit"] == "d31864a868c59858f99b9b0c5dab4b0f18758e25"
    n_mod = 0
    for e in man["files"]:
        cur = hashlib.sha256((PINNRP / e["pinnrp_path"]).read_bytes()).hexdigest()
        if e["status"] == "identical":
            assert cur == e["root_sha256"], e["pinnrp_path"]
        else:
            n_mod += 1
            assert e["status"] == "modified" and cur == e["pinnrp_sha256"] and e["reason"], e["pinnrp_path"]
    assert n_mod == len(man["modified_files"])


def test_batch1_frozen_numbers_match_leaderboard():
    import csv
    rows = {r["run_id"]: r for r in csv.DictReader(open(PINNRP / "references" / "batch1" / "optimization_leaderboard.csv"))}
    z = rows["Z4_20K__s1234__fa8fa7fe7e"]
    doc = frozen.load_frozen("batch1_hard_tanh2")
    assert float(z["L2_exact"]) == doc["batch1_result"]["L2_exact"]
