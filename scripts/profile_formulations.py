"""B2-PROF-001: measured per-step cost of residual + backward for each formulation / derivative
strategy at the B1 configuration (6x200, m=100, mini-batch 128, float32, 1 thread).
PROFILING ONLY: gradients are computed and discarded; NO optimizer step, NO parameter update,
NO training. Writes results_batch2/tables/B2-PROF-001_formulation_step_cost.csv (+ .json record).

    python scripts/profile_formulations.py [--reps 30] [--batch 128]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from beampinn.models.networks import build_model, count_parameters  # noqa: E402
from beampinn.training.trainer import DTYPES, build_hard, resolve_problem, setup_torch  # noqa: E402
from physref.derivatives import derivatives  # noqa: E402
from physref.formulations.mixed import MixedHardFF, TwoHeadFourierPINN, mixed_residuals  # noqa: E402
from physref.formulations.modal import ModalHardQ, TemporalFourierNet, modal_residual  # noqa: E402
from physref.frozen import to_experiment_config  # noqa: E402
from physref.provenance import environment, git_state, hardware  # noqa: E402
from physref.safety import assert_safe_output  # noqa: E402

EXP = "B2-PROF-001"


def strong_only(model, x, t, c2, g, strategy):
    if strategy == "reverse":
        D = derivatives(model, x, t, "reverse", 4, 2)
        return c2 * D["u_xxxx"] + D["u_tt"] + g * D["u_t"]
    from physref.derivatives import _nested_jvp
    x, t = x.detach(), t.detach()
    u4 = _nested_jvp(lambda z: model(z, t), x, 4)
    ut = _nested_jvp(lambda z: model(x, z), t, 1)
    utt = _nested_jvp(lambda z: model(x, z), t, 2)
    return c2 * u4 + utt + g * ut


def timeit(fn, params, reps):
    import resource
    for _ in range(3):                     # warm-up
        loss = fn(); torch.autograd.grad(loss, params, allow_unused=True)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        loss = fn()
        torch.autograd.grad(loss, params, allow_unused=True)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return {"median_ms": 1e3 * ts[len(ts) // 2], "p10_ms": 1e3 * ts[len(ts) // 10],
            "p90_ms": 1e3 * ts[(9 * len(ts)) // 10],
            "maxrss_mb_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    cfg, _ = to_experiment_config("batch1_hard_tanh2")
    setup_torch(cfg)
    dt = DTYPES[cfg.precision]
    bm, refs, c2, g, _ = resolve_problem(cfg)
    ex, w1 = refs["exact"], bm.fundamental_omega()
    gen = torch.Generator().manual_seed(0)
    x = (torch.rand(a.batch, 1, generator=gen, dtype=torch.float64) * bm.L).to(dt)
    t = (torch.rand(a.batch, 1, generator=gen, dtype=torch.float64) * bm.t_end).to(dt)
    b1 = build_hard(build_model(cfg, bm), cfg, bm, refs).to(dt)
    mixed = MixedHardFF(TwoHeadFourierPINN(cfg.model, bm.L, bm.t_end, cfg.seed), ex, bm.L, bm.t_end, "tanh2", w1).to(dt)
    m = cfg.model
    modal = ModalHardQ(TemporalFourierNet(bm.t_end, m.m_fourier, m.sigma_t, m.depth, m.width, cfg.seed), 1.0, bm.t_end, "tanh2", w1).to(dt)
    w2 = c2 * ex.beta ** 4
    cases = {
        "B1 strong, reverse (Batch-1)": (b1, lambda: 0.5 * (strong_only(b1, x, t, c2, g, "reverse") ** 2).mean(), 4),
        "B1 strong, forward (JVP)": (b1, lambda: 0.5 * (strong_only(b1, x, t, c2, g, "forward") ** 2).mean(), 4),
        "mixed (u,v), reverse": (mixed, lambda: sum(0.5 * (r ** 2).mean() for r in mixed_residuals(mixed, x, t, c2, g, ex.beta)), 2),
        "modal q(t), reverse": (modal, lambda: 0.5 * ((ex.A0 * modal_residual(modal, t, w2, g)) ** 2).mean(), 2),
    }
    out_dir = assert_safe_output(REPO / "results_batch2" / "tables", strict=True, purpose=EXP)
    rows = []
    for name, (model, fn, order) in cases.items():
        params = [p for p in model.parameters() if p.requires_grad]
        r = timeit(fn, params, a.reps)
        rows.append({"experiment_id": EXP, "case": name, "parameters": count_parameters(model),
                     "max_derivative_order": order, "batch": a.batch, "precision": cfg.precision,
                     "threads": torch.get_num_threads(), **r})
        print(f"{name:32s} params {rows[-1]['parameters']:7d}  median {r['median_ms']:8.1f} ms  "
              f"[p10 {r['p10_ms']:.1f}, p90 {r['p90_ms']:.1f}]")
    p = out_dir / f"{EXP}_formulation_step_cost.csv"
    if p.exists():
        sys.exit(f"{p} exists; results are never overwritten (use a new experiment ID)")
    with open(p, "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sha, dirty = git_state()
    json.dump({"experiment_id": EXP, "kind": "profiling (no training)", "git_sha": sha, "git_dirty": dirty,
               "reps": a.reps, "environment": environment(), "hardware": hardware()},
              open(out_dir / f"{EXP}_record.json", "x"), indent=2, default=str)


if __name__ == "__main__":
    main()
