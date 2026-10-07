"""B2-E03 collocation-vs-optimizer mechanism isolation on the FROZEN full-field B1 problem.
Pre-registration: docs/hypotheses/B2-E03.md. Nothing here is a novel method.

Every non-B1 arm = the frozen Batch-1 `Trainer` Adam loop UNCHANGED for 2,500 steps (320,000 PDE
evaluations; bitwise identical to B1's first 2,500 steps), then a stage 2 that differs ONLY in
(optimizer, collocation-point policy):

  arm            stage-2 optimizer                         collocation policy (640 active points)
  LBFGS-F        torch L-BFGS (lr 1, hist 50, strong Wolfe) fixed: set0 = next epoch draw of the frozen sampler
  LBFGS-R        same                                      resample: a FRESH epoch draw every 50 closures
  LBFGS-4X       same                                      reservoir: 4 epoch draws (set0..set3; 2,560 points)
                                                           fixed at the switch, active subset cycles 0,1,2,3,0..
                                                           every 50 closures
  ADAM-FULL      the SAME Adam optimizer/schedule state,   fixed: set0 (identical to LBFGS-F)
                 full-batch steps on the active set

Accounting (identical for every arm): one closure / full-batch step on the active set = 640 PDE
evaluations + 1 forward + 1 backward (line-search evaluations included). Drawing points costs nothing.
L-BFGS stage: chunks of <= 50 closures for ALL L-BFGS arms (so that only the point set differs); each
LBFGS.step call gets max_eval <= remaining - max_ls - 1, so the 640,000 budget is never exceeded.
ADAM-FULL takes exactly `adam_full_steps` (474 = the B2-E02 L-BFGS closure count) full-batch steps.
L-BFGS history is NOT reset when the active set changes (the only difference between F/R/4X is which
points the closure evaluates).
"""
import hashlib
import math
import time

import torch

from beampinn.losses.residuals import pde_residual, reduce_loss
from beampinn.profiling.resources import peak_rss_mb
from beampinn.training.trainer import Trainer

ARMS = {"LBFGS-F": ("lbfgs", "fixed"), "LBFGS-R": ("lbfgs", "resample"),
        "LBFGS-4X": ("lbfgs", "reservoir"), "ADAM-FULL": ("adam_full", "fixed")}


def state_checksum(model):
    """sha256 over every tensor of the state_dict (name, dtype, shape, bytes) - machine-verifiable init."""
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        t = v.detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


class CollocationLabTrainer(Trainer):
    def __init__(self, cfg, root, arm, switch_evals=320_000, total_evals=640_000, n_active=640, n_reservoir_sets=4,
                 switch_every=50, history_size=50, max_ls=25, snapshot_evals=128_000, adam_full_steps=474):
        super().__init__(cfg, root=root)
        if arm not in ARMS:
            raise ValueError(arm)
        if cfg.loss.grouping != "pde_only" or cfg.loss.hard_constraints != "ff_tanh2":
            raise ValueError("defined for the hard-constrained B1 configuration only")
        if cfg.sampler.n_per_term != n_active:
            raise ValueError("active set size must equal the sampler epoch size (one epoch draw per set)")
        mb = cfg.sampler.mini_batch
        if switch_evals % mb:
            raise ValueError("switch must be a whole number of Adam steps")
        self.arm, (self.optimizer_kind, self.point_mode) = arm, ARMS[arm]
        self.switch_step = switch_evals // mb
        self.total_evals, self.n_active, self.max_ls = int(total_evals), int(n_active), int(max_ls)
        self.n_reservoir_sets, self.switch_every = int(n_reservoir_sets), int(switch_every)
        self.history_size, self.snapshot_evals, self.adam_full_steps = int(history_size), int(snapshot_evals), int(adam_full_steps)
        self.init_checksum = state_checksum(self.model)
        self.coll = {"mode": self.point_mode, "sets": [], "active_set": None, "events": []}

    # --------------------------------------------------------------- point sets
    def _draw_set(self):
        self.sampler._draw()
        x, t, _ = self.sampler.data["f"]
        self.coll["sets"].append((x.clone(), t.clone()))
        return len(self.coll["sets"]) - 1

    def _activate(self, idx, closure_index):
        self.coll["active_set"] = idx
        self.coll["events"].append({"closure_index": closure_index, "pde_evaluations": self.acc["pde_evaluations"],
                                    "set": idx})
        dt = next(self.model.parameters()).dtype
        x, t = self.coll["sets"][idx]
        self._x, self._t = x.to(dt), t.to(dt)

    def _loss(self):
        x = self._x.clone().requires_grad_(True)
        t = self._t.clone().requires_grad_(True)
        r = pde_residual(self.model, x, t, self.c2, self.gamma, self.cfg.loss.pde_scale)
        return reduce_loss(r, self.cfg.loss.reduction)

    def _count(self):
        a = self.acc
        a["pde_evaluations"] += self.n_active
        a["training_points"] += self.n_active
        a["grad_evaluations"] += 1

    def _maybe_snapshot(self, next_snap):
        if self.acc["pde_evaluations"] >= next_snap:
            self.step = self.acc["pde_evaluations"] // self.cfg.sampler.mini_batch      # equivalent-step label
            self.save(f"step_{self.step}.pt")
            return next_snap + self.snapshot_evals
        return next_snap

    # --------------------------------------------------------------------- run
    def run_two_stage(self):
        st = self.run(resume=False, final_eval=False, stop_at_step=self.switch_step)
        if st != "interrupted" or self.step != self.switch_step:
            return st
        acc = self.acc
        acc["switch_step"], acc["switch_pde_evaluations"] = self.step, acc["pde_evaluations"]
        self.save("switch.pt")                     # state at the switch (metadata snapshot; no effect on training)
        n_sets = self.n_reservoir_sets if self.point_mode == "reservoir" else 1
        for _ in range(n_sets):
            self._draw_set()
        self._activate(0, 0)
        self.status = "running"
        next_snap = (acc["pde_evaluations"] // self.snapshot_evals + 1) * self.snapshot_evals
        if self.optimizer_kind == "adam_full":
            used = self._run_adam_full(next_snap)
            reason = "fixed_steps"
        else:
            used, reason = self._run_lbfgs(next_snap)
        self.step = acc["pde_evaluations"] // self.cfg.sampler.mini_batch
        acc["stage2_closure_evals"] = used
        acc["lbfgs_closure_evals"] = used if self.optimizer_kind == "lbfgs" else 0
        acc["stage2_stop_reason"] = reason
        if self.optimizer_kind == "lbfgs":
            acc["lbfgs_iterations"] = int(self._lbfgs.state[self._lbfgs._params[0]].get("n_iter", 0))
        acc["forward_passes"] = acc["optimizer_steps"] + used
        acc["backward_passes"] = acc["grad_evaluations"]
        acc["n_point_sets"] = len(self.coll["sets"])
        acc["n_distinct_training_points_stage2"] = len(self.coll["sets"]) * self.n_active
        acc["init_checksum"] = self.init_checksum
        acc["peak_rss_mb"] = max(acc["peak_rss_mb"], peak_rss_mb())
        if self.status == "running":
            self.status = "completed"
        self._snapshot()
        if self.status == "completed":
            self.finalise()
        return self.status

    def _run_adam_full(self, next_snap):
        acc, used = self.acc, 0
        for _ in range(self.adam_full_steps):
            t0 = time.perf_counter()
            loss = self._loss()
            if not torch.isfinite(loss):
                self.status = "diverged"; break
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step(); self.sched.step()
            acc["train_seconds"] += time.perf_counter() - t0
            self._count(); used += 1
            if used % 50 == 0 or used == self.adam_full_steps:
                self.history.append({"step": self.step, "stage": "adam_full", "loss": float(loss.detach()),
                                     "train_seconds": acc["train_seconds"], "pde_evaluations": acc["pde_evaluations"],
                                     "lr": self.opt.param_groups[0]["lr"]})
            next_snap = self._maybe_snapshot(next_snap)
        acc["adam_full_steps"] = used
        return used

    def _run_lbfgs(self, next_snap):
        acc = self.acc
        opt = torch.optim.LBFGS(self.params, lr=1.0, max_iter=10 ** 9, max_eval=1, history_size=self.history_size,
                                line_search_fn="strong_wolfe")
        budget = (self.total_evals - acc["pde_evaluations"]) // self.n_active
        used, last = {"n": 0}, {}

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = self._loss()
            loss.backward()
            used["n"] += 1
            self._count()
            last["loss"] = float(loss.detach())
            return loss

        reason = "budget"
        next_switch = self.switch_every
        self._lbfgs = opt
        while True:
            remaining = budget - used["n"]
            if remaining <= self.max_ls + 1:
                break
            to_snap = max(1, (next_snap - acc["pde_evaluations"]) // self.n_active)
            chunk = max(1, min(next_switch - used["n"], to_snap, remaining - self.max_ls - 1))
            opt.param_groups[0]["max_eval"] = chunk
            n0 = used["n"]
            t0 = time.perf_counter()
            opt.step(closure)
            acc["train_seconds"] += time.perf_counter() - t0
            self.history.append({"step": self.step, "stage": "lbfgs", "loss": last.get("loss"),
                                 "train_seconds": acc["train_seconds"], "pde_evaluations": acc["pde_evaluations"],
                                 "active_set": self.coll["active_set"]})
            if not math.isfinite(last.get("loss", float("nan"))):
                self.status, reason = "diverged", "non-finite loss"
                break
            next_snap = self._maybe_snapshot(next_snap)
            if used["n"] - n0 < chunk:
                reason = "lbfgs_tolerance"
                break
            if used["n"] >= next_switch:
                next_switch = (used["n"] // self.switch_every + 1) * self.switch_every
                if self.point_mode == "resample":
                    self._activate(self._draw_set(), used["n"])
                elif self.point_mode == "reservoir":
                    self._activate((self.coll["active_set"] + 1) % self.n_reservoir_sets, used["n"])
        return used["n"], reason

    def _blob(self):
        b = super()._blob()
        b["collocation"] = {"mode": self.coll["mode"], "active_set": self.coll["active_set"],
                            "events": list(self.coll["events"]),
                            "sets": [(x.clone(), t.clone()) for x, t in self.coll["sets"]]}
        b["init_checksum"] = getattr(self, "init_checksum", None)
        return b
