"""B2-E02 full-field TRANSFER test (docs/hypotheses/B2-E02.md section 10): the Adam -> L-BFGS controller
that passed the modal gate, applied to the frozen full-field B1 configuration with the SAME accounting.

Stage 1: the frozen Batch-1 `Trainer` loop, UNCHANGED, stopped at 320,000 PDE evaluations (2,500 steps
          x 128) - bitwise identical to B1's first 2,500 steps (tested).
Stage 2: full-batch torch L-BFGS (lr 1, history 50, strong Wolfe, PyTorch default tolerances, float32) on
          a FIXED set of 640 (x, t) collocation points = the next epoch draw of the frozen sampler; loss =
          the B1 loss (pde_only, half-mean of the strong residual; hard IC/BC ansatz unchanged).
Accounting: every closure evaluation (incl. line search) = 640 PDE evaluations, 1 forward, 1 backward;
          each LBFGS call gets max_eval <= remaining - max_ls - 1, so 640,000 is never exceeded.
No network, ansatz, physics, precision or sampler change. Not a novel method (Rathore et al. 2024).
"""
import math
import time

import torch

from beampinn.losses.residuals import pde_residual, reduce_loss
from beampinn.profiling.resources import peak_rss_mb
from beampinn.training.trainer import Trainer


class AdamLBFGSFullFieldTrainer(Trainer):
    def __init__(self, cfg, root, switch_evals=320_000, total_evals=640_000, n_lbfgs=640, history_size=50,
                 max_ls=25, snapshot_evals=128_000):
        super().__init__(cfg, root=root)
        mb = cfg.sampler.mini_batch
        if switch_evals % mb:
            raise ValueError("switch must be a whole number of Adam steps")
        if cfg.loss.grouping != "pde_only" or cfg.loss.hard_constraints != "ff_tanh2":
            raise ValueError("defined for the hard-constrained B1 configuration only")
        self.switch_step = switch_evals // mb
        self.total_evals, self.n_lbfgs, self.max_ls = int(total_evals), int(n_lbfgs), int(max_ls)
        self.history_size, self.snapshot_evals = int(history_size), int(snapshot_evals)
        self.lbfgs_log = []

    def run_two_stage(self):
        st = self.run(resume=False, final_eval=False, stop_at_step=self.switch_step)
        if st != "interrupted" or self.step != self.switch_step:
            return st
        acc = self.acc
        acc["switch_step"] = self.step
        acc["switch_pde_evaluations"] = acc["pde_evaluations"]
        self.sampler._draw()
        dt = next(self.model.parameters()).dtype
        x_fix, t_fix, _ = self.sampler.data["f"]
        x_fix, t_fix = x_fix.to(dt), t_fix.to(dt)
        assert x_fix.shape[0] == self.n_lbfgs, (x_fix.shape, self.n_lbfgs)
        opt = torch.optim.LBFGS(self.params, lr=1.0, max_iter=10 ** 9, max_eval=1, history_size=self.history_size,
                                line_search_fn="strong_wolfe")
        budget = (self.total_evals - acc["pde_evaluations"]) // self.n_lbfgs
        used, last = {"n": 0}, {}
        cfg = self.cfg

        def closure():
            opt.zero_grad(set_to_none=True)
            x = x_fix.clone().requires_grad_(True)
            t = t_fix.clone().requires_grad_(True)
            r = pde_residual(self.model, x, t, self.c2, self.gamma, cfg.loss.pde_scale)
            loss = reduce_loss(r, cfg.loss.reduction)
            loss.backward()
            used["n"] += 1
            acc["pde_evaluations"] += self.n_lbfgs
            acc["training_points"] += self.n_lbfgs
            acc["grad_evaluations"] += 1
            last["loss"] = float(loss.detach())
            return loss

        self.status = "running"
        next_snap = (acc["pde_evaluations"] // self.snapshot_evals + 1) * self.snapshot_evals
        reason = "budget"
        while True:
            remaining = budget - used["n"]
            if remaining <= self.max_ls + 1:
                break
            chunk = max(1, min((next_snap - acc["pde_evaluations"]) // self.n_lbfgs, remaining - self.max_ls - 1))
            opt.param_groups[0]["max_eval"] = chunk
            n0 = used["n"]
            t0 = time.perf_counter()
            opt.step(closure)
            acc["train_seconds"] += time.perf_counter() - t0
            self.history.append({"step": self.step, "stage": "lbfgs", "loss": last.get("loss"),
                                 "train_seconds": acc["train_seconds"], "pde_evaluations": acc["pde_evaluations"]})
            self.lbfgs_log.append({"pde_evaluations": acc["pde_evaluations"], "closure_evals": used["n"],
                                   "loss": last.get("loss"), "train_seconds": acc["train_seconds"]})
            if not math.isfinite(last.get("loss", float("nan"))):
                self.status, reason = "diverged", "non-finite loss"
                break
            if acc["pde_evaluations"] >= next_snap:
                self.step = acc["pde_evaluations"] // cfg.sampler.mini_batch      # equivalent-step label
                self.save(f"step_{self.step}.pt")
                next_snap += self.snapshot_evals
            if used["n"] - n0 < chunk:
                reason = "lbfgs_tolerance"
                break
        self.step = acc["pde_evaluations"] // cfg.sampler.mini_batch
        acc["lbfgs_closure_evals"] = used["n"]
        acc["lbfgs_budget_evals"] = int(budget)
        acc["lbfgs_stop_reason"] = reason
        acc["forward_passes"] = acc["optimizer_steps"] + used["n"]
        acc["backward_passes"] = acc["grad_evaluations"]
        acc["peak_rss_mb"] = max(acc["peak_rss_mb"], peak_rss_mb())
        if self.status == "running":
            self.status = "completed"
        self._snapshot()
        if self.status == "completed":
            self.finalise()
        return self.status

    def _blob(self):
        b = super()._blob()
        b["lbfgs_log"] = list(getattr(self, "lbfgs_log", []))
        return b
