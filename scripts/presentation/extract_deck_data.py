"""Extract the data behind the Batch-1 story deck from FROZEN artifacts only. No training.

Reads (read-only):
  * frozen Batch-1 checkpoints in ROOT (git-tracked, commit d31864a / 6b31d79): inference only;
  * frozen Batch-1 tables copied into references/batch1/tables (byte-identical to ROOT);
  * the analytical exact reference (beampinn.physics).
Writes ONLY to results_batch2/presentation/data/ (checked by physref.safety in strict mode).

    PYTHONDONTWRITEBYTECODE=1 python scripts/presentation/extract_deck_data.py [ROOT_PATH]
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

RP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RP / "src"))
from beampinn.config import ExperimentConfig  # noqa: E402
from beampinn.evaluation.metrics import predict  # noqa: E402
from beampinn.models.networks import build_model  # noqa: E402
from beampinn.physics.beam import modal_time  # noqa: E402
from beampinn.physics.benchmarks import get_benchmark  # noqa: E402
from beampinn.training.trainer import DTYPES, build_hard, resolve_problem  # noqa: E402
from physref.safety import assert_safe_output  # noqa: E402

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/user/PINN-replication")
CK = ROOT / "results_optimization" / "checkpoints"
OUT = assert_safe_output(RP / "results_batch2" / "presentation" / "data", strict=True, purpose="deck data")
OUT.mkdir(parents=True, exist_ok=True)
TB = RP / "references" / "batch1" / "tables"

# (label, run directory, checkpoint file, frozen collapse-time source for the self-check)
RUNS = [
    ("C0", "C0_paper__s1234__c41ccb1cdf", "final.pt", None),
    ("X2", "X2_hard_tanh2_rc__s1234__2c89a6ce41", "final.pt", None),
    ("Z4", "Z4_Y1_mb128__s1234__f78aa7f1da", "final.pt", ("phaseZ4_20K_checkpoints.csv", "Z4_20K", 5000)),
    ("Z4_20K", "Z4_20K__s1234__fa8fa7fe7e", "step_20000.pt", ("phaseZ4_20K_checkpoints.csv", "Z4_20K", 20000)),
]


def load(run, fname):
    blob = torch.load(CK / run / fname, weights_only=False, map_location="cpu")
    cfg = ExperimentConfig.from_dict(blob["config"])
    bm, refs, c2, g, _ = resolve_problem(cfg)
    model = build_model(cfg, bm).to(DTYPES[cfg.precision])
    if cfg.loss.hard_constraints in ("ff_tsq", "ff_tanh2"):
        model = build_hard(model, cfg, bm, refs).to(DTYPES[cfg.precision])
    model.load_state_dict(blob["model"])
    model.eval()
    return cfg, bm, refs, model, int(blob.get("step", -1))


def local_amp(y, t, P):           # identical to Batch-1 experiments/budget_diagnostic.py
    dt = t[1] - t[0]; h = max(1, int(round(P / 2 / dt)))
    return np.array([0.5 * (y[max(0, i - h):i + h + 1].max() - y[max(0, i - h):i + h + 1].min()) for i in range(len(y))])


def collapse_time(up, ue, t, P):  # identical definition: first t >= P/2 with A_pred/A_exact < 0.5
    R = local_amp(up, t, P) / local_amp(ue, t, P)
    idx = np.where((t >= P / 2) & (R < 0.5))[0]
    return float(t[idx[0]]) if len(idx) else float(t[-1])


torch.set_num_threads(4)
t = np.linspace(0.0, 1.0, 4001)
plot_idx = np.arange(0, 4001, 2)          # 2001 points for the slide charts
out = {"t": [round(float(v), 5) for v in t[plot_idx]], "traces_mm": {}, "checks": {}}
ref = get_benchmark("FE-D-M1").reference("exact")
ue = ref.u(ref.x_norm, t)
P = 2 * np.pi / ref.omega_d
out["traces_mm"]["exact"] = [round(float(v) * 1e3, 3) for v in ue[plot_idx]]
out["omega_d"], out["f_d"], out["x_mid"] = ref.omega_d, ref.omega_d / (2 * np.pi), ref.x_norm
for label, run, fname, frozen in RUNS:
    cfg, bm, refs, model, step = load(run, fname)
    assert abs(refs["exact"].x_norm - ref.x_norm) < 1e-12
    with torch.no_grad():
        pass
    up = predict(model, np.full_like(t, ref.x_norm)[:, None], t[:, None]).ravel()
    out["traces_mm"][label] = [round(float(v) * 1e3, 3) for v in up[plot_idx]]
    tc = collapse_time(up, ue, t, P)
    chk = {"run": run, "checkpoint": fname, "step": step, "mini_batch": cfg.sampler.mini_batch,
           "collapse_time_s": tc, "cycles": tc * out["f_d"],
           "amp_ratio_std": float(up.std() / ue.std())}
    if frozen:
        r = [x for x in csv.DictReader(open(TB / frozen[0])) if x["run"] == frozen[1] and int(x["step"]) == frozen[2]][0]
        chk["frozen_collapse_time_s"] = float(r["collapse_time_s"])
        chk["match"] = abs(float(r["collapse_time_s"]) - tc) < 1e-9
    out["checks"][label] = chk
    print(label, chk)

# conditioning: required network output N*(t) at mid-span (Batch-1 experiments/phaseE_ansatz_conditioning.py)
tc_ = np.linspace(1e-6, 1.0, 200001)
q = modal_time(ref.omega, ref.gamma, tc_)
N_t2, N_th = (q - 1) / tc_ ** 2, (q - 1) / np.tanh(ref.omega * tc_) ** 2
out["conditioning"] = {
    "range_t2": [float(N_t2.min()), float(N_t2.max())], "range_tanh2": [float(N_th.min()), float(N_th.max())],
    "t": [], "absN_t2": [], "absN_tanh2": []}
sel = np.unique(np.concatenate([np.linspace(0, 20000, 201), np.linspace(20000, 200000, 181)]).astype(int))
for i in sel:
    out["conditioning"]["t"].append(round(float(tc_[i]), 5))
    out["conditioning"]["absN_t2"].append(float(abs(N_t2[i])))
    out["conditioning"]["absN_tanh2"].append(float(abs(N_th[i])))
print("N* ranges", out["conditioning"]["range_t2"], out["conditioning"]["range_tanh2"])


def ck(name, run):
    return [{k: (float(v) if k not in ("run",) else v) for k, v in r.items() if k in
             ("run", "step", "pde_evaluations_cum", "collapse_time_s", "persistence_cycles", "L2_exact", "fit_w", "fit_decay", "train_seconds_cum")}
            for r in csv.DictReader(open(TB / name)) if r["run"] == run]


out["budget"] = {"Y1": ck("phaseY1_20K_checkpoints.csv", "Y1_20K"), "Z4": ck("phaseZ4_20K_checkpoints.csv", "Z4_20K")}
x = np.linspace(0, ref.L, 61)
ms = ref.mode_shape(x); out["mode_shape"] = [round(float(v), 4) for v in ms / ms.max()]
(OUT / "deck_data.json").write_text(json.dumps(out, indent=1))
print("written", OUT / "deck_data.json")
