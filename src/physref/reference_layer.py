"""Physics Reference Layer output contract (prototype schema, docs/PHYSICS_REFERENCE_LAYER_ARCHITECTURE.md).

A downstream AI system receives a PhysicsReference, never a bare prediction. Fields that the
solver cannot certify are explicitly 'unverified', never omitted. Uncertainty is NOT yet
implemented: the field exists and is reported as {"method": "none"} until a validated method
exists (no fabricated error bars)."""
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

SCHEMA_VERSION = "0.1.0"
CONVERGENCE_STATES = ("verified", "unverified", "failed_verification", "not_converged")


@dataclass
class PhysicsReference:
    solution: Dict[str, Any]                  # {"quantity", "units", "points"|"grid", "values"}
    physical_parameters: Dict[str, Any]       # PDE coefficients, BC type, geometry, IC
    residual: Dict[str, float]                # e.g. {"PDE_residual_rel": ..., "grid": "51x501"}
    constraint_error: Dict[str, float]        # IC/BC errors (dimensionless definitions in metadata)
    uncertainty: Dict[str, Any] = field(default_factory=lambda: {"method": "none"})
    convergence_status: str = "unverified"
    verification: Dict[str, Any] = field(default_factory=dict)   # gates + pass/fail per gate
    compute_cost: Dict[str, Any] = field(default_factory=dict)
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def validate(self):
        if self.convergence_status not in CONVERGENCE_STATES:
            raise ValueError(self.convergence_status)
        for k in ("solution", "physical_parameters", "residual", "constraint_error"):
            if not getattr(self, k):
                raise ValueError(f"{k} must be populated")
        if self.convergence_status == "verified":
            gates = self.verification.get("gates", {})
            if not gates or not all(g.get("passed") for g in gates.values()):
                raise ValueError("'verified' requires every declared verification gate to pass")
        for k in ("git_sha", "config_sha256", "experiment_id"):
            if k not in self.provenance:
                raise ValueError(f"provenance.{k} missing")
        return self

    def to_json(self):
        return json.dumps(asdict(self.validate()), indent=2, default=float)
