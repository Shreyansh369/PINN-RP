"""EXPLORATORY PROTOTYPE (not a result of record).

Tests the differentiable amplitude-normalised residual loss on the frozen B1 configuration
at the B2-E01 budget (5,000 steps x mini-batch 128 = 640,000 PDE evaluations), float32, 1 thread.
Imports the user's beampinn package READ-ONLY; writes only under OUT_ROOT (scratch), never the repo.

Modes (only the PDE loss term changes; model, init, sampler, optimiser, schedule, seed = B1):
  std  : B1 loss 0.5*mean(r^2)                       (harness check: must reproduce B1 at 640k)
  norm : 0.5*mean( r_i^2 * D_ref / (D_{k(i)} + eps) ),  D_k = mean of u^2 on a fixed probe grid in
         time window k (K windows ~ 1 period each), kept IN the autograd graph
  det  : same as norm but D_k detached (ablation: frozen-weight normalisation)

  erate: B1 loss + MU * mean_j ( (dE/dt + gamma*int u_t^2) / (E + eps) )^2 on a probe stencil, where
         E = 0.5*int(u_t^2 + c2*u_xx^2) dx (energy of r = c2 u_xxxx + u_tt + gamma u_t). Degree-0 homogeneous
         in amplitude: zero for the exact solution, penalises excess decay AND growth.

  pow  : 0.5*mean( r_i^2 * (D_ref/(D_k+eps))^P )  with fixed exponent P in [0,1] (P=0 std, P=1 norm); arg5 = P
  ctrl : as pow, but P adapted every CTRL_EVERY steps from the measured energy-balance mismatch
         rel = (dE/dt + gamma*int u_t^2)/E on late times (no reference solution used):
         P <- clip(P - KAPPA * rel_ema, 0, 1)   (rel<0: excess decay -> raise P; rel>0: growth -> lower P)

Usage: python -I norm_proto.py <repo_src_dir> <out_root> <mode> <seed> [mu]
"""
import json
import math
import sys
import time
from pathlib import Path

REPO_SRC, OUT_ROOT, MODE, SEED = sys.argv[1], Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
MU = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
CTRL_EVERY, KAPPA, EMA = 50, 0.005, 0.8
sys.path.insert(0, REPO_SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from physref.frozen import to_experiment_config  # noqa: E402
from physref.persistence import persistence  # noqa: E402
from beampinn.training.trainer import Trainer  # noqa: E402
from beampinn.losses.residuals import term_residuals, reduce_loss  # noqa: E402
from beampinn.utils.io import write_json  # noqa: E402

K, PNX, PNT = 20, 8, 12          # windows; probe points per window in x and t
EPS_REL = 1e-6


class ProtoTrainer(Trainer):
    def __init__(self, cfg, root, mode):
        super().__init__(cfg, root)
        self.mode = mode
        L, T = float(self.bm.L), float(self.bm.t_end)
        self.T = T
        dt = next(self.model.parameters()).dtype
        xs = torch.linspace(0.0, L, PNX + 2, dtype=dt)[1:-1]
        edges = torch.linspace(0.0, T, K + 1, dtype=dt)
        px, pt = [], []
        for k in range(K):
            ts = edges[k] + (edges[k + 1] - edges[k]) * (torch.arange(PNT, dtype=dt) + 0.5) / PNT
            X, Tm = torch.meshgrid(xs, ts, indexing="ij")
            px.append(X.reshape(-1)); pt.append(Tm.reshape(-1))
        self.px = torch.cat(px).reshape(-1, 1)
        self.pt = torch.cat(pt).reshape(-1, 1)
        with torch.no_grad():                      # reference scale = IC displacement energy (constant)
            u0 = self.model.u0(xs.reshape(-1, 1))
            self.D_ref = float((u0 ** 2).mean())
        self.eps = EPS_REL * self.D_ref
        self.D_log = []
        with torch.no_grad():
            w1 = float(self.bm.fundamental_omega(self.cfg.benchmark.pde_coeffs))
            self.e_eps = 1e-6 * self.D_ref * w1 ** 2          # tiny floor on the energy scale 0.5*w1^2*u^2
        self.E_log = []
        self.P = MU if self.mode == "pow" else 0.0
        self.rel_ema = None
        self.P_log = []

    def energy_rate_loss(self, n_centres=32):
        """Central-difference energy balance on n_centres random times (3-point stencils) x PNX interior x."""
        dt_ = next(self.model.parameters()).dtype
        L, T = float(self.bm.L), self.T
        h = T / 960.0                                                     # exact solution: rms rel 0.06 1/s
        tc = h + (T - 2 * h) * torch.rand(n_centres, dtype=dt_)
        ts = torch.stack([tc - h, tc, tc + h], 1).reshape(-1)            # (3n,)
        xi, wi = np.polynomial.legendre.leggauss(PNX)                     # Gauss-Legendre in x
        xs = torch.tensor(0.5 * L * (xi + 1.0), dtype=dt_)
        wq = torch.tensor(0.5 * L * wi, dtype=dt_).reshape(PNX, 1, 1)
        X, Tm = torch.meshgrid(xs, ts, indexing="ij")
        x = X.reshape(-1, 1).clone().requires_grad_(True)
        t = Tm.reshape(-1, 1).clone().requires_grad_(True)
        u = self.model(x, t)
        g = lambda y, v: torch.autograd.grad(y, v, torch.ones_like(y), create_graph=True)[0]
        u_t = g(u, t)
        u_x = g(u, x)
        u_xx = g(u_x, x)
        e = 0.5 * (u_t ** 2 + self.c2 * u_xx ** 2)
        e = (wq * e.reshape(PNX, n_centres, 3)).sum(0)                     # (n,3) energy integral
        diss = (wq * (self.gamma * u_t ** 2).reshape(PNX, n_centres, 3)).sum(0)[:, 1]
        dEdt = (e[:, 2] - e[:, 0]) / (2 * h)
        rel = (dEdt + diss) / (e[:, 1] + self.e_eps)
        return (rel ** 2).mean()

    def measure_rel(self, n_centres=64, t_lo=0.25):
        """Energy-balance mismatch on late times; returns float, no parameter gradient."""
        dt_ = next(self.model.parameters()).dtype
        L, T = float(self.bm.L), self.T
        h = T / 960.0
        tc = t_lo * T + h + ((1 - t_lo) * T - 2 * h) * torch.rand(n_centres, dtype=dt_)
        ts = torch.stack([tc - h, tc, tc + h], 1).reshape(-1)
        xi, wi = np.polynomial.legendre.leggauss(PNX)
        xs = torch.tensor(0.5 * L * (xi + 1.0), dtype=dt_)
        wq = torch.tensor(0.5 * L * wi, dtype=dt_).reshape(PNX, 1, 1)
        X, Tm = torch.meshgrid(xs, ts, indexing="ij")
        x = X.reshape(-1, 1).clone().requires_grad_(True)
        t = Tm.reshape(-1, 1).clone().requires_grad_(True)
        u = self.model(x, t)
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x))[0]
        u_t = u_t.detach()
        e = (wq * (0.5 * (u_t ** 2 + self.c2 * u_xx ** 2)).reshape(PNX, n_centres, 3)).sum(0)
        diss = (wq * (self.gamma * u_t ** 2).reshape(PNX, n_centres, 3)).sum(0)[:, 1]
        rel = ((e[:, 2] - e[:, 0]) / (2 * h) + diss) / (e[:, 1] + self.e_eps)
        return float(rel.median())

    def window_energy(self):
        u = self.model(self.px, self.pt).reshape(K, -1)
        return (u ** 2).mean(1)

    def train_step(self, log_row=None):
        if self.mode == "std":
            return super().train_step(log_row)
        if self.mode == "erate":
            return self.erate_step()
        cfg = self.cfg
        batch = self.sampler.next_batch()
        res = term_residuals(self.model, batch, self.c2, self.gamma, cfg.loss.pde_scale)
        D = self.window_energy()
        if self.mode == "det":
            D = D.detach()
        if self.mode == "ctrl" and self.step % CTRL_EVERY == 0 and self.step > 0:
            rel = self.measure_rel()
            self.rel_ema = rel if self.rel_ema is None else EMA * self.rel_ema + (1 - EMA) * rel
            self.P = min(1.0, max(0.0, self.P - KAPPA * self.rel_ema))
            self.P_log.append([self.step, rel, self.rel_ema, self.P])
        parts = {}
        for name, r in res.items():
            if batch[name]["spec"].kind == "pde":
                t = batch[name]["t"].reshape(-1)
                k = torch.clamp((t / self.T * K).long(), 0, K - 1)
                w = self.D_ref / (D[k] + self.eps)
                if self.mode in ("pow", "ctrl"):
                    w = w ** self.P
                parts[name] = 0.5 * (w * r.reshape(-1) ** 2).mean()
            else:
                parts[name] = reduce_loss(r, cfg.loss.reduction)
        total = sum(self.weighting.lam[k_] * parts[k_] for k_ in parts)
        if not torch.isfinite(total):
            return None, parts
        self.opt.zero_grad(set_to_none=True)
        total.backward()
        self.opt.step()
        self.sched.step()
        mb = cfg.sampler.mini_batch
        self.acc["optimizer_steps"] += 1
        self.acc["grad_evaluations"] += 1
        self.acc["training_points"] += mb * len(res)
        self.acc["pde_evaluations"] += sum(mb for b in batch.values() if b["spec"].kind == "pde")
        if self.step % 500 == 0:
            self.D_log.append([self.step] + [float(v) for v in D.detach()])
        return float(total.detach()), parts


def _erate_step(self):
    cfg = self.cfg
    batch = self.sampler.next_batch()
    res = term_residuals(self.model, batch, self.c2, self.gamma, cfg.loss.pde_scale)
    parts = {k: reduce_loss(r, cfg.loss.reduction) for k, r in res.items()}
    le = self.energy_rate_loss()
    total = sum(self.weighting.lam[k_] * parts[k_] for k_ in parts) + MU * le
    if not torch.isfinite(total):
        return None, parts
    self.opt.zero_grad(set_to_none=True)
    total.backward()
    self.opt.step()
    self.sched.step()
    mb = cfg.sampler.mini_batch
    self.acc["optimizer_steps"] += 1
    self.acc["grad_evaluations"] += 1
    self.acc["training_points"] += mb * len(res)
    self.acc["pde_evaluations"] += sum(mb for b in batch.values() if b["spec"].kind == "pde")
    if self.step % 250 == 0:
        self.E_log.append([self.step, float(parts[next(iter(parts))].detach()), float(le.detach())])
    return float(total.detach()), parts


ProtoTrainer.erate_step = _erate_step


def main():
    cfg, _ = to_experiment_config("batch1_hard_tanh2")
    cfg.train.max_steps, cfg.train.budget_label = 5000, "B2"
    cfg.seed = SEED
    cfg.name = f"PROTO_{MODE}" + (f"_mu{MU:g}" if MODE in ("erate", "pow") else "")
    cfg.train.snapshot_every = 0
    cfg.validate()
    out = OUT_ROOT / (f"{MODE}_s{SEED}" + (f"_mu{MU:g}" if MODE in ("erate", "pow") else ""))
    out.mkdir(parents=True, exist_ok=True)
    tr = ProtoTrainer(cfg, out, MODE)
    t0 = time.time()
    status = tr.run(resume=False)
    wall = time.time() - t0
    mpath = tr.paths["logs"] / "metrics.json"
    m = json.load(open(mpath)) if mpath.exists() else {}
    p, _ = persistence(tr.model, tr.refs["exact"], tr.bm.t_end)
    keep = ["L2_exact", "L2_late_exact", "fit_w", "fit_lam", "frequency_error_exact", "damping_error_exact",
            "amplitude_error_exact", "phase_error_exact", "PDE_residual_rms", "PDE_residual_rel",
            "train_seconds", "pde_evaluations", "peak_rss_mb", "parameters"]
    summary = {"mode": MODE, "seed": SEED, "mu": MU, "status": status, "wall_s": wall,
               **{k: m.get(k) for k in keep}, **p, "D_ref": tr.D_ref}
    write_json(out / "summary.json", summary)
    if tr.P_log:
        np.savetxt(out / "ctrl_log.csv", np.array(tr.P_log), delimiter=",", header="step,rel,rel_ema,P")
    if tr.E_log:
        np.savetxt(out / "erate_log.csv", np.array(tr.E_log), delimiter=",")
    if tr.D_log:
        np.savetxt(out / "window_energy.csv", np.array(tr.D_log), delimiter=",")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
