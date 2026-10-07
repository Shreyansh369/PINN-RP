"""Retain-Resample-Release (R3) collocation sampling (Daw et al., ICML 2023), axis 15.5.

Distinct from RAD (tested in Batch 1): R3 keeps an evolving population. Each update:
  retain   points whose |residual| exceeds the population mean (|r| > mean |r|),
  release  all others,
  resample the released count uniformly in the domain.
The causal extension of the paper (causal gate on time) is NOT implemented here.
Dry-run implementation: pure point management; the residual function is injected."""
import torch


class R3Sampler:
    def __init__(self, n, L, T, seed=0, dtype=torch.float64):
        self.n, self.L, self.T, self.dtype = n, float(L), float(T), dtype
        self.gen = torch.Generator().manual_seed(seed)
        self.x, self.t = self._uniform(n)
        self.history = []

    def _uniform(self, k):
        x = torch.rand(k, 1, generator=self.gen, dtype=self.dtype) * self.L
        t = torch.rand(k, 1, generator=self.gen, dtype=self.dtype) * self.T
        return x, t

    def update(self, residual_fn):
        """residual_fn(x, t) -> residual vector (n, 1). Returns stats; mutates the population."""
        with torch.no_grad():
            r = residual_fn(self.x, self.t).detach().abs().reshape(-1)
        keep = r > r.mean()
        n_keep = int(keep.sum())
        xn, tn = self._uniform(self.n - n_keep)
        self.x = torch.cat([self.x[keep], xn])
        self.t = torch.cat([self.t[keep], tn])
        st = {"retained": n_keep, "resampled": self.n - n_keep, "mean_abs_r": float(r.mean()),
              "retained_t_mean": float(self.t[:n_keep].mean()) if n_keep else float("nan")}
        self.history.append(st)
        return st
