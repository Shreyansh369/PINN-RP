"""B2-E02 modal laboratory: temporal-optimisation controllers on the FROZEN B2-E01 modal diagnostic.

The control is `physref.arms.ArmTrainer(arm="modal")` UNCHANGED (B2-E01). Candidates subclass it and
change ONLY the training controller; network, initialisation (same seed -> bit-identical weights),
sampler, physics residual, precision and evaluation are inherited. Pre-registration:
docs/hypotheses/B2-E02.md. Nothing here is a novel method:

  'adam'        control: Adam + lr 1e-3*0.9^(s/1000), mini-batch 128 (B2-E01 modal arm).
  'adam_lbfgs'  Adam (identical to the control) for the first half of the PDE-evaluation budget, then
                full-batch L-BFGS (strong-Wolfe) on a FIXED 640-point collocation set for the second
                half (Rathore et al., ICML 2024; established two-stage practice).
  'adam_causal' control with the temporal causal loss weighting of Wang, Sankaran & Perdikaris
                (CMAME 2024): w_i = exp(-eps * sum_{k<i} L_k) over N_t time bins (stop-gradient).

PDE-evaluation accounting (pre-registered):
  Adam step     : mini_batch (128) collocation points, 1 forward + 1 backward.
  L-BFGS closure: n_lbfgs (640) points, 1 forward + 1 backward, INCLUDING every line-search
                  evaluation. The stage-2 budget is never exceeded: each L-BFGS call gets
                  max_eval <= remaining - max_ls, so unused evaluations (< max_ls+1) are reported.
"""
import math
import time

import torch

from beampinn.losses.residuals import reduce_loss
from beampinn.profiling.resources import peak_rss_mb, reset_peak_rss
from beampinn.utils.io import write_history

from .arms import ArmTrainer
from .formulations.modal import modal_residual

CONTROLLERS = ("adam", "adam_lbfgs", "adam_causal")


def causal_epsilon(omega1, A0, n_bins, tol_frac=0.01):
    """Pre-registered, physics-scaled causal tolerance (no tuning): a bin counts as 'converged' when
    its residual RMS is tol_frac * w1^2 A0, i.e. L_tol = 0.5 (tol_frac w1^2 A0)^2; eps = 1/(n_bins L_tol),
    so that n_bins converged bins in front of the last one give a weight of exp(-1)."""
    L_tol = 0.5 * (tol_frac * omega1 ** 2 * A0) ** 2
    return 1.0 / (n_bins * L_tol)


class CausalModalTrainer(ArmTrainer):
    def __init__(self, base_cfg, root, experiment_id, strict=True, n_bins=16, tol_frac=0.01):
        super().__init__(base_cfg, "modal", root, experiment_id, strict)
        self.n_bins = int(n_bins)
        self.eps = causal_epsilon(self.omega_1, self.refs["exact"].A0, self.n_bins, tol_frac)
        self.T = float(self.bm.t_end)
        self.last_w = None

    def loss(self, batch):
        t = batch["f"]["t"]
        r = self.refs["exact"].A0 * modal_residual(self.model, t, self.omega2, self.gamma)
        idx = torch.clamp((t.detach().reshape(-1) / self.T * self.n_bins).long(), 0, self.n_bins - 1)
        sq = 0.5 * r.reshape(-1) ** 2
        Lb = torch.zeros(self.n_bins, dtype=sq.dtype).index_add(0, idx, sq)
        cnt = torch.zeros(self.n_bins, dtype=sq.dtype).index_add(0, idx, torch.ones_like(sq))
        nonempty = cnt > 0
        Lb = torch.where(nonempty, Lb / cnt.clamp(min=1), torch.zeros_like(Lb))
        cum = torch.cumsum(Lb.detach(), 0) - Lb.detach()               # sum_{k<i} L_k
        w = torch.exp(-self.eps * cum)
        total = (w * Lb)[nonempty].sum() / nonempty.sum()
        self.last_w = w.detach()
        return total, {"ode_causal": total, "ode_plain": reduce_loss(r).detach(),
                       "w_min": w.min().detach(), "w_last_bin": w[-1].detach()}


class AdamLBFGSModalTrainer(ArmTrainer):
    """Stage 1 = the control's Adam loop (bit-identical up to the switch); stage 2 = L-BFGS."""

    def __init__(self, base_cfg, root, experiment_id, strict=True, switch_evals=320_000, total_evals=640_000,
                 n_lbfgs=640, history_size=50, max_ls=25, snapshot_evals=128_000):
        super().__init__(base_cfg, "modal", root, experiment_id, strict)
        mb = self.cfg.sampler.mini_batch
        if switch_evals % mb:
            raise ValueError("switch must be a whole number of Adam steps")
        self.switch_step = switch_evals // mb
        self.total_evals, self.n_lbfgs, self.max_ls = int(total_evals), int(n_lbfgs), int(max_ls)
        self.history_size, self.snapshot_evals = int(history_size), int(snapshot_evals)
        self.lbfgs_log = []

    def run(self, max_steps=None, resume=False, stop_at_step=None, snapshot_every=None):
        # ---- stage 1: the control loop, unchanged, up to the switch
        st = super().run(self.switch_step, resume=False, snapshot_every=snapshot_every)
        if st != "completed":
            return st
        self.acc["switch_step"] = self.step
        self.acc["switch_pde_evaluations"] = self.acc["pde_evaluations"]
        # ---- stage 2: fixed collocation set = the next full epoch draw of the frozen sampler
        self.sampler._draw()
        t_fix = self.sampler.data["f"][1].to(next(self.model.parameters()).dtype)
        assert t_fix.shape[0] == self.n_lbfgs, (t_fix.shape, self.n_lbfgs)
        A0 = self.refs["exact"].A0
        opt = torch.optim.LBFGS(self.params, lr=1.0, max_iter=10 ** 9, max_eval=1, history_size=self.history_size,
                                line_search_fn="strong_wolfe")           # PyTorch default tolerances (declared)
        acc = self.acc
        budget_evals = (self.total_evals - acc["pde_evaluations"]) // self.n_lbfgs
        used = {"n": 0}
        last = {}

        def closure():
            opt.zero_grad(set_to_none=True)
            r = A0 * modal_residual(self.model, t_fix, self.omega2, self.gamma)
            loss = reduce_loss(r)
            loss.backward()
            used["n"] += 1
            acc["pde_evaluations"] += self.n_lbfgs
            acc["residual_evaluations"] += self.n_lbfgs
            acc["forward_passes"] += 1
            acc["backward_passes"] += 1
            last["loss"] = float(loss.detach())
            return loss

        reset_peak_rss()
        next_snap = (acc["pde_evaluations"] // self.snapshot_evals + 1) * self.snapshot_evals
        reason = "budget"
        while True:
            remaining = budget_evals - used["n"]
            if remaining <= self.max_ls + 1:
                break
            chunk_evals = max(1, min((next_snap - acc["pde_evaluations"]) // self.n_lbfgs, remaining - self.max_ls - 1))
            opt.param_groups[0]["max_eval"] = chunk_evals
            n0 = used["n"]
            t0 = time.perf_counter()
            opt.step(closure)
            acc["train_seconds"] += time.perf_counter() - t0
            st_ = opt.state[opt._params[0]]
            self.lbfgs_log.append({"pde_evaluations": acc["pde_evaluations"], "closure_evals": used["n"],
                                   "loss": last.get("loss"), "n_iter": st_.get("n_iter"), "train_seconds": acc["train_seconds"]})
            self.history.append({"step": self.step, "stage": "lbfgs", "loss": last.get("loss"), "lr": 1.0, **acc})
            if not math.isfinite(last.get("loss", float("nan"))):
                self.status, reason = "diverged", "non-finite loss"
                break
            if acc["pde_evaluations"] >= next_snap:
                self.step = acc["pde_evaluations"] // self.cfg.sampler.mini_batch   # equivalent-step label
                self.save(f"step_{self.step}.pt")
                next_snap += self.snapshot_evals
            if used["n"] - n0 < chunk_evals:          # L-BFGS returned early: tolerance reached
                reason = "lbfgs_tolerance"
                break
        self.step = acc["pde_evaluations"] // self.cfg.sampler.mini_batch
        acc["lbfgs_closure_evals"] = used["n"]
        acc["lbfgs_budget_evals"] = int(budget_evals)
        acc["lbfgs_stop_reason"] = reason
        acc["peak_rss_mb"] = max(acc["peak_rss_mb"], peak_rss_mb())
        if self.status == "running" or self.status == "completed":
            self.status = "completed"
        self.save("latest.pt")
        write_history(self.paths["logs"] / "history.csv", self.history)
        return self.status

    def _blob(self):
        b = super()._blob()
        b["lbfgs_log"] = list(self.lbfgs_log)
        return b


def build_trainer(controller, base_cfg, root, experiment_id, strict=True, **kw):
    if controller == "adam":
        return ArmTrainer(base_cfg, "modal", root, experiment_id, strict)
    if controller == "adam_lbfgs":
        return AdamLBFGSModalTrainer(base_cfg, root, experiment_id, strict, **kw)
    if controller == "adam_causal":
        return CausalModalTrainer(base_cfg, root, experiment_id, strict, **kw)
    raise ValueError(controller)
