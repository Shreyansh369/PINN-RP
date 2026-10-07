"""Compute-cost accounting record and Pareto utilities (docs/COMPUTE_COST_MODEL.md).

No composite cost/error score is defined: reports use Pareto fronts over measured quantities.
"""
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class CostRecord:
    experiment_id: str
    arm: str
    # accuracy / physics (lower is better unless noted)
    L2_exact: float = float("nan")
    L2_paper: float = float("nan")
    persistence_cycles: float = float("nan")      # HIGHER is better
    frequency_error_exact: float = float("nan")
    decay_error_rel: float = float("nan")
    PDE_residual_rel: float = float("nan")
    IC_error_max: float = float("nan")
    BC_error_max: float = float("nan")
    # cost (all measured)
    parameters: int = 0
    optimizer_steps: int = 0
    pde_evaluations: int = 0                      # residual points used for training
    derivative_order_max: int = 0
    train_seconds: float = float("nan")
    peak_rss_mb: float = float("nan")
    inference_us_per_point: float = float("nan")
    checkpoint_bytes: int = 0
    precision: str = "float32"
    hardware: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def pareto_front(points, minimize=("error", "cost")):
    """points: list of (error, cost, label). Returns the non-dominated subset (both minimised)."""
    front = []
    for e, c, lab in points:
        dominated = any((e2 <= e and c2 <= c) and (e2 < e or c2 < c) for e2, c2, _ in points)
        if not dominated:
            front.append((e, c, lab))
    return sorted(front, key=lambda p: p[1])


def accuracy_at_matched_budget(records, budget_key="pde_evaluations", tol=0.05):
    """Group records whose budget agrees within `tol` (relative) for matched comparisons."""
    groups = []
    for r in sorted(records, key=lambda r: getattr(r, budget_key)):
        b = getattr(r, budget_key)
        if groups and abs(b - groups[-1][0]) <= tol * max(b, 1):
            groups[-1][1].append(r)
        else:
            groups.append((b, [r]))
    return groups
