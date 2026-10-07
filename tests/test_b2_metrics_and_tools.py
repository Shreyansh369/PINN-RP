"""Conditioning analyzer, frozen persistence metric, R3 sampler, temporal plans, output contract,
cost utilities, approval gate, experiment-ID integrity."""
import json
import math

import numpy as np
import pytest
import torch

from beampinn.physics.benchmarks import get_benchmark
from physref import conditioning as C
from physref.persistence import passes_persistence_gate, persistence

BM = get_benchmark("FE-D-M1")
EX = BM.reference("exact")


def test_conditioning_scales_match_physics():
    spec, ref = C.beam_spec()
    s = C.characteristic_scales(spec)
    assert abs(s["omega1_rad_s"] - EX.omega) < 1e-9 and abs(s["omega_d_rad_s"] - EX.omega_d) < 1e-9
    assert abs(s["cycles_in_window"] - 20.5826) < 1e-3 and s["max_x_order"] == 4
    nd = C.nondimensionalization_candidates(spec)["modal (Lx=1/beta1, Tc=1/w1)"]
    assert abs(nd["coeffs"]["a40"] - 1) < 1e-12 and abs(nd["tau_range"] - EX.omega) < 1e-9
    h = C.hard_ansatz_conditioning(spec)
    assert abs(h["t2"]["N_star_scale"] + EX.omega ** 2 / 2) < 1e-6 and h["tanh2"]["N_star_scale"] == -0.5


def test_fourier_coverage_consistent_with_batch1_seeded_draws():
    """Expected counts vs the seeded counts measured in Batch 1 (stage01 table): paper 3/100, rc 0/100."""
    spec, _ = C.beam_spec()
    wd = EX.omega_d
    paper = C.fourier_coverage(10.0, wd, "physical", True, spec.T)["expected_count"]
    rc = C.fourier_coverage(10.0, wd, "standardize", False, spec.T)["expected_count"]
    assert 1.0 < paper < 8.0 and rc < 0.1
    from beampinn.models.networks import SpatioTemporalFourierPINN
    from physref.frozen import to_experiment_config
    for name, expect in (("baseline_fourier_ntk", 3), ("batch1_hard_tanh2", 0)):
        cfg, _ = to_experiment_config(name)
        net = SpatioTemporalFourierPINN(cfg.model, BM.L, BM.t_end, cfg.seed)
        f = net.enc_t[0].angular_frequencies() / net.tt.scale
        assert int((f >= wd).sum()) == expect


def test_spectral_content_finds_fundamental():
    t = np.linspace(0, 1, 20001)
    sc = C.spectral_content(EX.u(EX.x_norm, t), t[1] - t[0])
    assert abs(sc["dominant_Hz"] - EX.omega_d / (2 * math.pi)) < 0.1


def test_persistence_exact_passes_and_fast_decay_collapses(probe_factory):
    p, _ = persistence(probe_factory(EX), EX, BM.t_end)
    assert passes_persistence_gate(p) and abs(p["persistence_cycles"] - p["full_window_cycles"]) < 1e-9
    import dataclasses
    fast = dataclasses.replace(EX, gamma=40.0)               # envelope e^{-20 t}: ratio 0.5 at ~0.04 s
    p2, _ = persistence(probe_factory(fast), EX, BM.t_end)
    assert not passes_persistence_gate(p2) and p2["collapse_time_s"] < 0.1


def test_r3_retains_high_residual_points_and_keeps_size():
    from physref.sampling import R3Sampler
    s = R3Sampler(1000, BM.L, BM.t_end, seed=1)
    st = s.update(lambda x, t: (t > 0.8).double() * 10 + 0.01)
    assert s.x.shape == (1000, 1) and st["retained"] == int((st["retained"]))
    assert float(s.t[: st["retained"]].min()) > 0.8           # retained = the high-residual band
    s2 = R3Sampler(1000, BM.L, BM.t_end, seed=1); s2.update(lambda x, t: (t > 0.8).double() * 10 + 0.01)
    assert torch.equal(s.t, s2.t)                               # deterministic


def test_temporal_plan_and_interface_continuity(probe_factory):
    from physref.temporal import interface_residuals, make_plan
    p = make_plan(1.0, 4, overlap=0.05)
    assert [round(w.t0, 3) for w in p.windows] == [0.0, 0.2, 0.45, 0.7]
    assert all(abs(c - 20.58 / 4) < 1.5 for c in p.cycles_per_window(EX.omega_d))
    with pytest.raises(ValueError):
        make_plan(1.0, 4, overlap=0.3)
    probe = probe_factory(EX)
    x = torch.linspace(0, BM.L, 11, dtype=torch.float64).reshape(-1, 1)
    du, dut = interface_residuals(probe, probe, x, 0.25)
    assert float(du.abs().max()) == 0.0 and float(dut.abs().max()) == 0.0


def test_reference_layer_contract_rules():
    from physref.reference_layer import PhysicsReference
    base = dict(solution={"quantity": "u", "units": "m", "values": [0.0]}, physical_parameters={"c2": 1912.3},
                residual={"PDE_residual_rel": 0.04}, constraint_error={"IC_error_max": 1e-6},
                provenance={"git_sha": "x", "config_sha256": "y", "experiment_id": "z"})
    assert json.loads(PhysicsReference(**base).to_json())["uncertainty"] == {"method": "none"}
    with pytest.raises(ValueError):          # cannot claim 'verified' without passing gates
        PhysicsReference(**base, convergence_status="verified",
                         verification={"gates": {"persistence": {"passed": False}}}).validate()
    with pytest.raises(ValueError):
        PhysicsReference(**{**base, "provenance": {}}).validate()


def test_pareto_front():
    from physref.compute_cost import pareto_front
    pts = [(0.5, 10, "a"), (0.3, 20, "b"), (0.6, 30, "c"), (0.3, 25, "d")]
    assert [p[2] for p in pareto_front(pts)] == ["a", "b"]


def test_training_gate_only_approved_ids():
    from physref.gate import TrainingNotApproved, approved_ids, require_approval
    want = {f"B2-E01-{a}-s{s}" for a in ("B1", "B1_fp64", "mixed", "modal") for s in (1234, 1235, 1236)}
    want |= {f"B2-E02-{a}-s{s}" for a in ("M0", "LBFGS", "CAUSAL") for s in (1234, 1235, 1236)}
    want |= {f"B2-E02T-{a}-s{s}" for a in ("B1", "LBFGS") for s in (1234, 1235, 1236)}
    assert approved_ids() == want
    with pytest.raises(TrainingNotApproved):
        require_approval("B2-E03-K4-s1234")


def test_experiment_ids_never_reused(tmp_path, monkeypatch):
    from physref import provenance
    import physref.safety as safety
    monkeypatch.setattr(safety, "RESULTS_B2", tmp_path)
    reg = tmp_path / "REG.csv"
    rec = {"experiment_id": "T-1", "created_utc": "now", "config_path": "c", "config_sha256": "h",
           "git_sha": "g", "git_dirty": False, "seed": 1, "status": "allocated", "notes": ""}
    d = provenance.allocate_run_dir("T-1", rec, registry=reg, base=tmp_path / "runs")
    assert (d / "record.json").exists()
    with pytest.raises(FileExistsError):
        provenance.allocate_run_dir("T-1", rec, registry=reg, base=tmp_path / "runs")
