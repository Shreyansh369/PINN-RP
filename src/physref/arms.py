"""Single-axis Batch-2 arms built on the frozen B1 configuration (docs/BATCH2_EXPERIMENT_MATRIX.md).

Each arm changes EXACTLY ONE thing relative to B1 (frozen Z4 recipe) and reuses the Batch-1
sampler (paper_epoch, pde_only), optimizer (Adam + exp-decay), precision, seed handling and the
Batch-1 full-field evaluation + frozen persistence metric:
  'mixed' : formulation strong -> mixed (u, v = u_xx); 2-output head; max derivative order 2.
  'modal' : field (x,t)->u replaced by t->q(t) with the exact mode-1 shape (Galerkin reduction);
            temporal representation = the B1 temporal branch.
Training is refused unless physref.gate approves the experiment ID (preparation phase: none).
"""
import copy
import time

import numpy as np
import torch

from beampinn.evaluation.metrics import evaluate_full
from beampinn.losses.residuals import reduce_loss
from beampinn.models.networks import count_parameters, model_size_bytes
from beampinn.optimization.optimizers import build_optimizer
from beampinn.profiling.resources import peak_rss_mb, peak_vram_mb, reset_peak_rss
from beampinn.sampling.samplers import PaperEpochSampler
from beampinn.training.trainer import DTYPES, resolve_problem, setup_torch
from beampinn.utils.io import save_checkpoint, write_history, write_json

from .formulations.mixed import DisplacementView, MixedHardFF, TwoHeadFourierPINN, mixed_residuals
from .formulations.modal import ModalField, ModalHardQ, TemporalFourierNet, modal_residual
from .persistence import persistence
from .safety import assert_safe_output

ARMS = ("mixed", "modal")


class ArmTrainer:
    def __init__(self, base_cfg, arm, root, experiment_id, strict=True):
        if arm not in ARMS:
            raise ValueError(arm)
        if base_cfg.loss.hard_constraints != "ff_tanh2":
            raise ValueError("arms are defined relative to B1 (hard ff_tanh2 ansatz)")
        self.cfg, self.arm, self.experiment_id = copy.deepcopy(base_cfg), arm, experiment_id
        setup_torch(self.cfg)
        dt = DTYPES[self.cfg.precision]
        self.bm, self.refs, self.c2, self.gamma, self.ic_fn = resolve_problem(self.cfg)
        ex = self.refs["exact"]
        self.omega_1 = self.bm.fundamental_omega(self.cfg.benchmark.pde_coeffs)
        m = self.cfg.model
        if arm == "mixed":
            net2 = TwoHeadFourierPINN(m, self.bm.L, self.bm.t_end, self.cfg.seed)
            self.model = MixedHardFF(net2, ex, self.bm.L, self.bm.t_end, "tanh2", self.omega_1).to(dt)
            self.view = DisplacementView(self.model)
        else:
            qnet = TemporalFourierNet(self.bm.t_end, m.m_fourier, m.sigma_t, m.depth, m.width, self.cfg.seed, m.two_pi,
                                      burn_spatial_sigmas=m.sigma_x)
            self.model = ModalHardQ(qnet, 1.0, self.bm.t_end, "tanh2", self.omega_1).to(dt)
            self.view = ModalField(self.model, ex).to(dt)
            self.omega2 = self.c2 * ex.beta ** 4          # Galerkin: w1^2 = c2 beta1^4 (tests)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt, self.sched = build_optimizer(self.params, self.cfg.optim)
        self.sampler = PaperEpochSampler(self.bm, self.ic_fn, self.cfg.sampler, "pde_only", self.cfg.seed, dt)
        base = assert_safe_output(root, strict=strict, purpose=experiment_id) / experiment_id
        self.paths = {"ckpt": base / "checkpoints", "logs": base / "logs"}
        for p in self.paths.values():
            p.mkdir(parents=True, exist_ok=True)
        self.step, self.history, self.status = 0, [], "initialised"
        # forward_passes / backward_passes: one batched forward of the residual graph and one
        # backward (loss.backward) per optimizer step; residual_evaluations counts residual vectors
        # (mixed: 2 per point - r_link and r_pde; modal: 1 ODE residual per t point).
        self.acc = dict(optimizer_steps=0, pde_evaluations=0, residual_evaluations=0, forward_passes=0,
                        backward_passes=0, train_seconds=0.0, peak_rss_mb=0.0)

    # ------------------------------------------------------------------ loss
    def loss(self, batch):
        b = batch["f"]
        if self.arm == "mixed":
            r_link, r_pde = mixed_residuals(self.model, b["x"], b["t"], self.c2, self.gamma, self.refs["exact"].beta)
            parts = {"link": reduce_loss(r_link), "pde": reduce_loss(r_pde)}
        else:
            r = self.refs["exact"].A0 * modal_residual(self.model, b["t"], self.omega2, self.gamma)
            parts = {"ode": reduce_loss(r)}
        return sum(parts.values()), parts

    # ------------------------------------------------------------ checkpoint
    def _blob(self):
        return {"experiment_id": self.experiment_id, "arm": self.arm, "config": self.cfg.to_dict(),
                "step": self.step, "model": self.model.state_dict(), "opt": self.opt.state_dict(),
                "sched": self.sched.state_dict(), "sampler": self.sampler.state_dict(),
                "acc": dict(self.acc), "history": list(self.history), "status": self.status,
                "torch_rng": torch.get_rng_state()}

    def save(self, name="latest.pt"):
        save_checkpoint(self.paths["ckpt"] / name, self._blob())

    def try_resume(self):
        p = self.paths["ckpt"] / "latest.pt"
        if not p.exists():
            return False
        b = torch.load(p, weights_only=False)
        if b["experiment_id"] != self.experiment_id or b["arm"] != self.arm:
            raise RuntimeError("checkpoint belongs to a different experiment/arm")
        self.model.load_state_dict(b["model"]); self.opt.load_state_dict(b["opt"])
        self.sched.load_state_dict(b["sched"]); self.sampler.load_state_dict(b["sampler"])
        self.acc, self.history, self.step, self.status = b["acc"], b["history"], b["step"], b["status"]
        torch.set_rng_state(b["torch_rng"])
        return True

    # ------------------------------------------------------------------ run
    def run(self, max_steps, resume=True, stop_at_step=None, snapshot_every=None):
        if resume:
            self.try_resume()
        if self.status == "completed":
            return self.status
        reset_peak_rss()
        self.status = "running"
        while self.step < max_steps:
            t0 = time.perf_counter()
            batch = self.sampler.next_batch()
            total, parts = self.loss(batch)
            if not torch.isfinite(total):
                self.status = "diverged"; break
            self.opt.zero_grad(set_to_none=True)
            total.backward()
            self.opt.step(); self.sched.step()
            self.acc["train_seconds"] += time.perf_counter() - t0
            self.acc["optimizer_steps"] += 1
            self.acc["pde_evaluations"] += self.cfg.sampler.mini_batch
            self.acc["residual_evaluations"] += self.cfg.sampler.mini_batch * len(parts)
            self.acc["forward_passes"] += 1
            self.acc["backward_passes"] += 1
            self.step += 1
            if self.step % self.cfg.train.log_every == 0 or self.step == 1:
                self.history.append({"step": self.step, "loss": float(total.detach()),
                                     **{f"L_{k}": float(v.detach()) for k, v in parts.items()},
                                     "lr": self.opt.param_groups[0]["lr"], **self.acc})
            if snapshot_every and self.step % snapshot_every == 0:
                self.save(f"step_{self.step}.pt")
            if stop_at_step is not None and self.step >= stop_at_step and self.step < max_steps:
                self.status = "interrupted"; break
        if self.status == "running":
            self.status = "completed"
        self.acc["peak_rss_mb"] = max(self.acc["peak_rss_mb"], peak_rss_mb())
        self.save("latest.pt")
        write_history(self.paths["logs"] / "history.csv", self.history)
        return self.status

    def finalise(self):
        metrics, _ = evaluate_full(self.view, self.bm, self.refs, self.c2, self.gamma)
        pers, _ = persistence(self.view, self.refs["exact"], self.bm.t_end)
        self.save("final.pt")
        out = {"experiment_id": self.experiment_id, "arm": self.arm, "status": self.status,
               "parameters": count_parameters(self.model), "model_size_bytes": model_size_bytes(self.model),
               "derivative_order_max": 2, "precision": self.cfg.precision, "peak_vram_mb": peak_vram_mb(),
               **self.acc, **metrics, **pers}
        write_json(self.paths["logs"] / "metrics.json", out)
        return out


def fp32_initial_state(cfg):
    """Initial state_dict of the B-family model built EXACTLY as in FP32 (Fourier draws, weights,
    biases, buffers). Used by the precision control so that FP64 differs from FP32 ONLY in arithmetic
    precision, not in the random draws (torch.randn consumes generators differently per dtype)."""
    from beampinn.models.networks import build_model
    from beampinn.training.trainer import build_hard
    c32 = copy.deepcopy(cfg)
    c32.precision = "float32"
    old = torch.get_default_dtype()
    setup_torch(c32)
    bm, refs, _, _, _ = resolve_problem(c32)
    model = build_model(c32, bm).to(torch.float32)
    if c32.loss.hard_constraints in ("ff_tsq", "ff_tanh2"):
        model = build_hard(model, c32, bm, refs).to(torch.float32)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.set_default_dtype(old)
    return sd


def load_cast_state(model, sd32):
    """Load a float32 state_dict into a (float64) model, casting floating tensors."""
    tgt = model.state_dict()
    model.load_state_dict({k: (v.to(tgt[k].dtype) if v.is_floating_point() else v) for k, v in sd32.items()})
