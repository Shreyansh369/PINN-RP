"""B2-E05 window-law trainer: sequential time windows with an exact modal (u, u_t) hand-off.

Pre-registration: docs/hypotheses/B2-E05.md. Spec: configs/batch2/B2-E05_window_law.yaml.
Built on the frozen B1 configuration via read-only imports of beampinn (nothing frozen is edited):

    window k, local time tau = t - t_k in [0, T_w]:
        u_k(x, tau) = a_k(x) + tau * b_k(x) + tanh^2(w1 tau) * Phi(x) * A0 * N_k(x, tau)

  k = 0 : a_0 = u0 (exact IC), b_0 = 0 -> beampinn HardConstrainedFF verbatim (W1000 == B1).
  k >= 1: a_k, b_k = 8-mode fixed-fixed projections of u_{k-1}(., T_w), d/dt u_{k-1}(., T_w).
Every window reuses beampinn Trainer.train_step (same loss, sampler, Adam + exp-decay, accounting);
the network's time input is standardised over [0, T_w]; N_k starts from N_{k-1}; the optimiser and
schedule restart per window (AT-PINN "reactivating" optimiser). Steps are split equally.
"""
import copy
import dataclasses
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn

from beampinn.evaluation.metrics import evaluate_full
from beampinn.losses.residuals import pde_residual
from beampinn.models.networks import build_model, count_parameters
from beampinn.optimization.optimizers import build_optimizer
from beampinn.physics.beam import eigen_root
from beampinn.profiling.resources import peak_rss_mb, reset_peak_rss
from beampinn.sampling.samplers import PaperEpochSampler
from beampinn.training.trainer import DTYPES, Trainer, build_hard
from beampinn.utils.io import save_checkpoint, write_history, write_json

from .persistence import persistence

N_MODES = 8                       # brief section 3
N_GL = 64                         # Gauss-Legendre nodes for the hand-off projection
R_NX, R_NT = 51, 51               # measured-R grid per window (docs/hypotheses/B2-E05.md section 6)
JUMP_NX = 201                     # uniform grid for the logged hand-off jumps
AMP_TIMES = (0.1, 0.25, 0.5, 0.75, 1.0)


# ------------------------------------------------------------------ mode basis
def ff_mode_np(beta, L, x):
    """Fixed-fixed U = cosh z - cos z - s (sinh z - sin z), z = beta x, cancellation-free (same identity
    as scripts/analyze_b2_e04.ff_mode_stable, B2-E04 Amendment 1)."""
    x = np.asarray(x, dtype=np.float64)
    bl, z = beta * L, beta * x
    oms = (math.cos(bl) - math.sin(bl) - math.exp(-bl)) / (math.sinh(bl) - math.sin(bl))
    return np.exp(-z) + oms * np.sinh(z) - np.cos(z) + (1.0 - oms) * np.sin(z)


class FFModeBasis(nn.Module):
    """First n fixed-fixed mode shapes phi_n(x) (max |phi_n| = 1), evaluated in float64 inside the graph
    (differentiable; exact phi'''' = beta^4 phi) and returned in the input dtype. Shape (len(x), n)."""

    def __init__(self, L, n=N_MODES):
        super().__init__()
        self.L, self.n = float(L), n
        self.betas = [eigen_root("fixed-fixed", k) / self.L for k in range(1, n + 1)]
        xs = np.linspace(0.0, self.L, 20001)
        self.oms, self.norm = [], []
        for b in self.betas:
            bl = b * self.L
            self.oms.append((math.cos(bl) - math.sin(bl) - math.exp(-bl)) / (math.sinh(bl) - math.sin(bl)))
            U = ff_mode_np(b, self.L, xs)
            self.norm.append(float(U[np.argmax(np.abs(U))]))

    def forward(self, x):
        xd = x.double()
        cols = []
        for b, oms, nrm in zip(self.betas, self.oms, self.norm):
            z = b * xd
            cols.append((torch.exp(-z) + oms * torch.sinh(z) - torch.cos(z) + (1.0 - oms) * torch.sin(z)) / nrm)
        return torch.cat(cols, dim=1).to(x.dtype)

    def numpy(self, x):
        return np.stack([ff_mode_np(b, self.L, x) / nrm for b, nrm in zip(self.betas, self.norm)], 0)


def gauss_legendre(n, L):
    xi, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * L * (xi + 1.0), 0.5 * L * w


def project(F, Phi, w):
    """L2(0, L) projection of samples F (n_gl,) onto modes Phi (n_modes, n_gl) with GL weights w.
    Returns (coefficients, relative remainder ||F - P F|| / ||F||)."""
    m = (Phi ** 2) @ w
    a = (Phi * w) @ F / m
    rem = math.sqrt(float(w @ (F - Phi.T @ a) ** 2) / max(float(w @ F ** 2), 1e-300))
    return a, rem


# ------------------------------------------------------------------ ansatz
class HandoffFF(nn.Module):
    """u_k(x, tau) = sum_n a_n phi_n(x) + tau sum_n b_n phi_n(x) + tanh^2(w1 tau) Phi(x) A0 N(x, tau).
    For ANY N: u(x,0) = a(x), u_tau(x,0) = b(x) (tanh^2 has a double root at 0) and u = u_x = 0 at x = 0, L
    (every phi_n and Phi = 16 x^2 (L-x)^2 / L^4 have double roots there)."""

    def __init__(self, net, basis, a, b, L, A0, omega_1):
        super().__init__()
        self.net, self.basis = net, basis
        self.L, self.A0, self.omega_1 = float(L), float(A0), float(omega_1)
        dt = next(net.parameters()).dtype
        self.register_buffer("a", torch.as_tensor(np.asarray(a, np.float64), dtype=dt).reshape(-1, 1))
        self.register_buffer("b", torch.as_tensor(np.asarray(b, np.float64), dtype=dt).reshape(-1, 1))

    def g(self, t):
        return torch.tanh(self.omega_1 * t) ** 2

    def phi(self, x):
        return 16.0 * x ** 2 * (self.L - x) ** 2 / self.L ** 4

    def forward(self, x, t):
        P = self.basis(x)
        return P @ self.a + t * (P @ self.b) + self.g(t) * self.phi(x) * self.A0 * self.net(x, t)


class StitchedField(nn.Module):
    """Global field u(x, t) from the trained windows: t in window k = floor(t / T_w) (t = T -> last
    window), evaluated at tau = t - t_k. Differentiable in x and t (piecewise)."""

    def __init__(self, windows, T_w):
        super().__init__()
        self.windows = nn.ModuleList(windows)
        self.T_w, self.N = float(T_w), len(windows)

    def window_index(self, t):
        k = torch.floor(t.detach().double() / self.T_w).long()
        return torch.clamp(k, 0, self.N - 1)

    def forward(self, x, t):
        if self.N == 1:
            return self.windows[0](x, t)
        k = self.window_index(t).reshape(-1)
        out = torch.zeros_like(t)
        for j in torch.unique(k).tolist():
            m = k == j
            out[m] = self.windows[j](x[m], t[m] - j * self.T_w)
        return out


def amplitude_ratios(trace, times=AMP_TIMES):
    t, R = trace[0], trace[5]
    return {f"amp_ratio_t{s:g}": float(np.interp(s, t, R)) for s in times}


# ------------------------------------------------------------------ trainer
class WindowedTrainer(Trainer):
    """beampinn Trainer whose model, sampler and optimiser are rebuilt on [0, T_w] and re-used window by
    window. train_step is NOT overridden: the per-step computation and accounting are B1's."""

    def __init__(self, cfg, root, T_w):
        super().__init__(cfg, root)
        T = float(self.bm.t_end)
        n = T / float(T_w)
        self.N = int(round(n))
        if self.N < 1 or abs(n - self.N) > 1e-9:
            raise ValueError(f"T_w = {T_w} does not divide T = {T}")
        total = cfg.total_steps()
        spe = cfg.sampler.n_per_term // cfg.sampler.mini_batch
        if total % self.N or (total // self.N) % spe:
            raise ValueError(f"{total} steps do not split into {self.N} windows of whole epochs ({spe} steps)")
        self.T_w, self.steps_per_window = float(T_w), total // self.N
        dt = DTYPES[cfg.precision]
        self.bm_w = dataclasses.replace(self.bm, t_end=self.T_w)
        self.omega_1 = self.bm.fundamental_omega(cfg.benchmark.pde_coeffs)
        self.A0 = float(self.refs["exact"].A0)
        # identical draws to B1 (build_model uses its own seeded generators); only the time normalisation differs
        self.net = build_model(cfg, self.bm_w).to(dt)
        self.model = build_hard(self.net, cfg, self.bm_w, self.refs).to(dt)
        self.sampler = PaperEpochSampler(self.bm_w, lambda x: self.ic_fn(x), cfg.sampler, cfg.loss.grouping,
                                         cfg.seed, dt, data_fn=self.refs["exact"].u)
        self._new_optimizer()
        self.basis = FFModeBasis(self.bm.L).to(dt)
        self.xg, self.wg = gauss_legendre(N_GL, float(self.bm.L))
        self.Pg = self.basis.numpy(self.xg)
        self.window, self.windows, self.window_log = 0, [], []
        self.acc.update(handoff_seconds=0.0, rk_seconds=0.0, windows_completed=0)

    def _new_optimizer(self):
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt, self.sched = build_optimizer(self.params, self.cfg.optim)

    # ------------------------------------------------------------ per-window
    def _end_state(self, model):
        """u and u_t at tau = T_w on the GL nodes (float64 numpy)."""
        dt = next(model.parameters()).dtype
        x = torch.tensor(self.xg, dtype=dt).reshape(-1, 1)
        t = torch.full_like(x, self.T_w).requires_grad_(True)
        u = model(x, t)
        ut = torch.autograd.grad(u, t, torch.ones_like(u))[0]
        return u.detach().double().numpy().ravel(), ut.detach().double().numpy().ravel()

    def handoff(self, model):
        """Coefficients (a, b) of the next window + logged projection errors and jumps."""
        F, G = self._end_state(model)
        a, eu = project(F, self.Pg, self.wg)
        b, ev = project(G, self.Pg, self.wg)
        dt = next(model.parameters()).dtype
        xu = np.linspace(0.0, float(self.bm.L), JUMP_NX)
        x = torch.tensor(xu, dtype=dt).reshape(-1, 1)
        t = torch.full_like(x, self.T_w).requires_grad_(True)
        u = model(x, t)
        ut = torch.autograd.grad(u, t, torch.ones_like(u))[0]
        Pu = self.basis.numpy(xu)
        ref = self.refs["exact"]
        ju = np.abs(u.detach().double().numpy().ravel() - Pu.T @ a).max() / ref.A0
        jv = np.abs(ut.detach().double().numpy().ravel() - Pu.T @ b).max() / (ref.omega * ref.A0)
        return a, b, {"proj_err_u": eu, "proj_err_ut": ev, "jump_u_max_over_A0": float(ju),
                      "jump_ut_max_over_wA0": float(jv)}

    def measured_R(self, model):
        """R_k = rms(PDE residual) / rms(u) on a 51 x 51 grid of the window (no reference)."""
        dt = next(model.parameters()).dtype
        X, Tt = np.meshgrid(np.linspace(0.0, float(self.bm.L), R_NX), np.linspace(0.0, self.T_w, R_NT), indexing="ij")
        xs = torch.tensor(X.reshape(-1, 1), dtype=dt)
        ts = torch.tensor(Tt.reshape(-1, 1), dtype=dt)
        rs, us = [], []
        for i in range(0, len(xs), 1024):
            x = xs[i:i + 1024].clone().requires_grad_(True)
            t = ts[i:i + 1024].clone().requires_grad_(True)
            rs.append(pde_residual(model, x, t, self.c2, self.gamma).detach().double())
            with torch.no_grad():
                us.append(model(xs[i:i + 1024], ts[i:i + 1024]).double())
        r, u = torch.cat(rs).numpy(), torch.cat(us).numpy()
        return float(np.sqrt(np.mean(r ** 2)) / np.sqrt(np.mean(u ** 2))), float(np.sqrt(np.mean(r ** 2)))

    def train_window(self):
        """B1's inner loop (same log cadence -> same grad-norm diagnostics) for steps_per_window steps."""
        cfg = self.cfg
        end = (self.window + 1) * self.steps_per_window
        while self.step < end:
            log_now = self.step == 0 or (self.step + 1) % cfg.train.log_every == 0
            row = {} if log_now else None
            diag0, t0 = self.acc["diag_seconds"], time.perf_counter()
            loss, parts = self.train_step(row)
            self.acc["train_seconds"] += time.perf_counter() - t0 - (self.acc["diag_seconds"] - diag0)
            if loss is None:
                self.status = "diverged"
                self.history.append({"step": self.step + 1, "window": self.window, "status": "non-finite loss"})
                return False
            self.step += 1
            first = self.step == self.window * self.steps_per_window + 1
            if row is not None or first or self.step == end:
                row = row or {}
                row.update(step=self.step, window=self.window, epoch=self.sampler.epoch, loss=loss,
                           lr=self.opt.param_groups[0]["lr"], train_seconds=self.acc["train_seconds"],
                           pde_evaluations=self.acc["pde_evaluations"])
                row.update({f"L_{k}": float(v.detach()) for k, v in parts.items()})
                self.history.append(row)
        return True

    # ------------------------------------------------------------ checkpoint
    def _wblob(self):
        return {"run_id": self.run_id, "T_w": self.T_w, "N": self.N, "window": self.window, "step": self.step,
                "net": self.net.state_dict(), "model": self.model.state_dict(),
                "windows": [w.state_dict() for w in self.windows],
                "coefs": [(w.a.double().numpy().ravel(), w.b.double().numpy().ravel())
                          if isinstance(w, HandoffFF) else None for w in self.windows],
                "sampler": self.sampler.state_dict(), "acc": dict(self.acc), "history": list(self.history),
                "window_log": list(self.window_log), "status": self.status, "torch_rng": torch.get_rng_state()}

    def _make_window_model(self, net, coef):
        if coef is None:
            return build_hard(net, self.cfg, self.bm_w, self.refs).to(next(net.parameters()).dtype)
        return HandoffFF(net, self.basis, coef[0], coef[1], self.bm.L, self.A0, self.omega_1)

    def try_resume(self):
        p = self.paths["ckpt"] / "latest.pt"
        if not p.exists():
            return False
        b = torch.load(p, weights_only=False)
        if b["run_id"] != self.run_id or b["T_w"] != self.T_w:
            raise RuntimeError("checkpoint belongs to a different configuration")
        self.windows = []
        for sd, coef in zip(b["windows"], b["coefs"]):
            w = self._make_window_model(copy.deepcopy(self.net), coef)
            w.load_state_dict(sd)
            self.windows.append(w.requires_grad_(False))
        self.net.load_state_dict(b["net"])
        self.window, self.step, self.status = b["window"], b["step"], b["status"]
        self.sampler.load_state_dict(b["sampler"]); self.acc = b["acc"]
        self.history, self.window_log = b["history"], b["window_log"]
        torch.set_rng_state(b["torch_rng"])
        if self.window < self.N and self.window > 0:          # rebuild the pending window's ansatz
            self.model = self._make_window_model(self.net, b["next_coef"])
        self._new_optimizer()
        return True

    # ------------------------------------------------------------ main loop
    def run(self, resume=True, final_eval=True, stop_after_window=None):
        """Train all windows (checkpoint after each). stop_after_window ends a SEGMENT (status 'interrupted')."""
        self.cfg.to_json(self.paths["config"])
        if resume:
            self.try_resume()
        if self.status in ("completed", "diverged"):
            return self.status
        reset_peak_rss()
        self.acc["segments"] += 1
        self.status = "running"
        while self.window < self.N:
            if not self.train_window():
                break
            t0 = time.perf_counter()
            Rk, rms_r = self.measured_R(self.model)
            self.acc["rk_seconds"] += time.perf_counter() - t0
            frozen = copy.deepcopy(self.model).requires_grad_(False)
            self.windows.append(frozen)
            log = {"window": self.window, "t_start": self.window * self.T_w, "R_k": Rk, "rms_residual": rms_r,
                   "loss_end": self.history[-1].get("loss"), "step_end": self.step}
            next_coef = None
            if self.window + 1 < self.N:
                t0 = time.perf_counter()
                a, b, err = self.handoff(frozen)
                self.acc["handoff_seconds"] += time.perf_counter() - t0
                log.update(err)
                next_coef = (a, b)
                self.model = self._make_window_model(self.net, next_coef)   # N_{k+1} starts from N_k
                self._new_optimizer()                                        # fresh Adam + schedule
            self.window_log.append(log)
            self.window += 1
            self.acc["windows_completed"] = self.window
            self.acc["peak_rss_mb"] = max(self.acc["peak_rss_mb"], peak_rss_mb())
            blob = self._wblob()
            blob["next_coef"] = next_coef
            save_checkpoint(self.paths["ckpt"] / "latest.pt", blob)
            write_history(self.paths["logs"] / "history.csv", self.history)
            write_history(self.paths["logs"] / "windows.csv", self.window_log)
            if stop_after_window is not None and self.window >= stop_after_window and self.window < self.N:
                self.status = "interrupted"
                return self.status
        if self.status == "running":
            self.status = "completed"
        if self.status == "completed" and final_eval:
            self.finalise()
        return self.status

    def stitched(self):
        return StitchedField(self.windows, self.T_w)

    def finalise(self):
        train_peak = max(self.acc["peak_rss_mb"], peak_rss_mb())
        reset_peak_rss()
        field = self.stitched()
        t0 = time.perf_counter()
        metrics, _ = evaluate_full(field, self.bm, self.refs, self.c2, self.gamma)
        pers, trace = persistence(field, self.refs["exact"], self.bm.t_end)
        eval_s = time.perf_counter() - t0
        save_checkpoint(self.paths["ckpt"] / "final.pt", self._wblob())
        Rk = [w["R_k"] for w in self.window_log]
        perr = [max(w["proj_err_u"], w["proj_err_ut"]) for w in self.window_log if "proj_err_u" in w]
        lam_exact = 0.5 * self.gamma
        out = {"run_id": self.run_id, "status": self.status, "seed": self.cfg.seed, "T_w": self.T_w,
               "N_windows": self.N, "steps_per_window": self.steps_per_window,
               "collocation_per_epoch": self.cfg.sampler.n_per_term,
               "collocation_density_per_s_per_epoch": self.cfg.sampler.n_per_term / self.T_w,
               "parameters": count_parameters(self.net),
               "parameters_stored": sum(p.numel() for p in field.parameters()),
               "precision": self.cfg.precision, "threads": self.cfg.threads, "final_eval_seconds": eval_s,
               **self.acc, **metrics, **pers, **amplitude_ratios(trace),
               "decay_error": metrics["fit_lam"] - lam_exact, "lam_exact": lam_exact,
               "R_k": Rk, "R_k_mean": float(np.mean(Rk)), "R_k_max": float(np.max(Rk)),
               "handoff_projection_error_max": float(max(perr)) if perr else 0.0,
               "window_log": self.window_log}
        out["peak_rss_mb"] = train_peak
        out["peak_rss_eval_mb"] = peak_rss_mb()
        write_json(self.paths["logs"] / "metrics.json", out)
        return out


RESULT_COLUMNS = ["arm", "seed", "T_w", "N_windows", "L2_exact", "L2_late_exact", "persistence_cycles",
                  "persistence_cycles_vel"] + [f"amp_ratio_t{s:g}" for s in AMP_TIMES] + [
                  "fit_w", "fit_lam", "decay_error", "frequency_error_exact", "phase_error_exact",
                  "PDE_residual_rel", "R_k_mean", "R_k_max", "R_k", "predicted_d*", "train_seconds",
                  "handoff_seconds", "pde_evaluations", "peak_rss_mb", "parameters",
                  "handoff_projection_error_max", "status"]


def results_row(arm, metrics):
    """One brief-section-5 row (predicted_d* is filled in by the Block-3 evaluator)."""
    row = {k: metrics.get(k) for k in RESULT_COLUMNS}
    row["arm"], row["R_k"], row["predicted_d*"] = arm, json.dumps(metrics.get("R_k")), None
    return row
