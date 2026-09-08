"""Hierarchical active-parameter accounting (sparse-research Phase 9).

The base Phase 11 accounting (``parameter_accounting``) measures total vs
top-k-active parameters from real safetensors headers. This module
extends the hierarchy one level down, the TurboSparse/PowerInfer
compound:

    total
      = always-on (attention, norms, embeddings, MTP, vision, router)
      + dense/shared FFN (always on)
      + routed experts (capacity; only top-k of E computed per token)
          -> routed active/token = routed * top_k / num_experts
              -> neuron-active/token = routed_active * (1 - sparsity)
                  where sparsity is MEASURED per layer by the activation
                  census (dReLU counterfactual), not assumed.

Every number carries its definition id so cross-paper comparisons
(TurboSparse, PowerInfer) are normalized explicitly, never silently.
The intra-expert claim is fail-closed: it exists only when the caller
supplies a census digest plus the per-layer measured sparsities —
assuming sparsity without measuring it is exactly the failure mode the
program forbids.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from chowder.parameter_accounting import (
    ParameterAccounting,
    ParameterAccountingError,
)

SPARSE_ACCOUNTING_VERSION = 1
#: Definition id for the effective-active number. Comparisons with other
#: papers must state BOTH definition ids; there is no default.
EFFECTIVE_DEFINITION_ID = "chowder/effective-active-v1"


@dataclass(frozen=True)
class HierarchicalActiveBreakdown:
    """Measured hierarchical split for one sparse model directory.

    All counts are parameters-per-token unless named ``_capacity``.
    ``neuron_*`` fields are ``None`` unless census evidence was supplied.
    """

    model_dir: str
    definition_id: str
    total_parameters: int
    always_on_parameters: int
    dense_shared_ffn_parameters: int
    routed_expert_capacity: int
    routed_active_parameters: int
    num_experts: int
    top_k: int
    census_digest: str | None
    neuron_sparsity_measured: float | None
    neuron_sparsity_min: float | None
    neuron_sparsity_max: float | None
    neuron_active_parameters: int | None
    total_effective_active: int | None

    def __post_init__(self) -> None:
        for label in (
            "total_parameters",
            "always_on_parameters",
            "dense_shared_ffn_parameters",
            "routed_expert_capacity",
            "routed_active_parameters",
            "num_experts",
            "top_k",
        ):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ParameterAccountingError(f"{label} must be a non-negative int")
        if self.total_parameters <= 0:
            raise ParameterAccountingError("an empty accounting is not evidence")
        for label in ("num_experts", "top_k"):
            if getattr(self, label) <= 0:
                raise ParameterAccountingError(f"{label} must be positive")
        if self.top_k > self.num_experts:
            raise ParameterAccountingError("top_k cannot exceed num_experts")
        for label in (
            "census_digest",
            "neuron_sparsity_measured",
            "neuron_active_parameters",
            "total_effective_active",
        ):
            if (getattr(self, label) is None) != (
                getattr(self, "census_digest") is None
            ):
                raise ParameterAccountingError(
                    "intra-expert fields require census evidence: pass "
                    "census_digest, per-layer sparsities together or not at all"
                )
        if self.neuron_sparsity_measured is not None:
            for label in ("neuron_sparsity_measured", "neuron_sparsity_min", "neuron_sparsity_max"):
                value = getattr(self, label)
                if value is None or not (0.0 <= value < 1.0):
                    raise ParameterAccountingError(
                        f"{label} must be a measured fraction in [0, 1)"
                    )
            if self.neuron_active_parameters <= 0:
                raise ParameterAccountingError(
                    "neuron-active parameters must be positive when census evidence is present"
                )
        if self.total_effective_active is not None and not (
            0 < self.total_effective_active <= self.total_parameters
        ):
            raise ParameterAccountingError(
                f"total_effective_active {self.total_effective_active} outside "
                f"(0, total {self.total_parameters}]"
            )

    # -- derived ------------------------------------------------------------

    @property
    def routed_active_fraction(self) -> float:
        return self.top_k / self.num_experts

    @property
    def intra_expert_active_fraction(self) -> float | None:
        if self.neuron_sparsity_measured is None:
            return None
        return 1.0 - self.neuron_sparsity_measured

    def effective_label(self) -> str:
        """The honest compound label, or raise when census evidence is absent."""
        if self.total_effective_active is None:
            raise ParameterAccountingError(
                "no effective label without census-measured intra-expert "
                "sparsity; the top-k-only label is the honest maximum"
            )
        b = self.total_effective_active / 1e9
        return (
            f"A{b:.1f}B effective ({self.total_effective_active:,} "
            f"parameters/token; top-{self.top_k} of {self.num_experts} "
            f"experts x {self.intra_expert_active_fraction:.2f} intra-expert "
            f"active fraction, census {self.census_digest[:12]}...)"
        )

    def formula_document(self) -> dict[str, str]:
        """The explicit formulas behind every number (normalized defs)."""
        doc = {
            "definition_id": self.definition_id,
            "always_on": (
                "attention_and_deltanet + layernorm + embedding + mtp + "
                "vision + router + other (computed for every token)"
            ),
            "dense_shared_ffn": (
                "dense_ffn + shared_expert (computed for every token)"
            ),
            "routed_capacity": "all routed expert parameters (capacity, not per-token)",
            "routed_active": (
                f"routed_capacity * top_k / num_experts = "
                f"{self.routed_expert_capacity} * {self.top_k} / "
                f"{self.num_experts} = {self.routed_active_parameters}"
            ),
        }
        if self.neuron_active_parameters is not None:
            doc["neuron_active"] = (
                "routed_active * (1 - measured dReLU sparsity) = "
                f"{self.routed_active_parameters} * "
                f"(1 - {self.neuron_sparsity_measured:.4f}) = "
                f"{self.neuron_active_parameters}  [per-expert neurons are "
                "uniform 3-row groups: gate row + up row + down column, so "
                "an inactive neuron removes exactly its 3*H share]"
            )
            doc["total_effective_active"] = (
                "always_on + dense_shared_ffn + neuron_active = "
                f"{self.always_on_parameters} + {self.dense_shared_ffn_parameters} "
                f"+ {self.neuron_active_parameters} = {self.total_effective_active}"
            )
        else:
            doc["total_effective_active"] = (
                "not claimed: no census evidence; top-k-only is the upper bound"
            )
        doc["cross_paper_rule"] = (
            "do not compare with TurboSparse/PowerInfer numbers unless the "
            "other side's definition id is stated; counts here are "
            "parameters whose multiply is actually performed per token"
        )
        return doc

    def to_dict(self) -> dict[str, Any]:
        return {
            "sparse_accounting_version": SPARSE_ACCOUNTING_VERSION,
            "model_dir": self.model_dir,
            "definition_id": self.definition_id,
            "total_parameters": self.total_parameters,
            "always_on_parameters": self.always_on_parameters,
            "dense_shared_ffn_parameters": self.dense_shared_ffn_parameters,
            "routed_expert_capacity": self.routed_expert_capacity,
            "routed_active_parameters": self.routed_active_parameters,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "census_digest": self.census_digest,
            "neuron_sparsity_measured": self.neuron_sparsity_measured,
            "neuron_sparsity_min": self.neuron_sparsity_min,
            "neuron_sparsity_max": self.neuron_sparsity_max,
            "neuron_active_parameters": self.neuron_active_parameters,
            "total_effective_active": self.total_effective_active,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HierarchicalActiveBreakdown":
        return cls(
            model_dir=data["model_dir"],
            definition_id=data["definition_id"],
            total_parameters=data["total_parameters"],
            always_on_parameters=data["always_on_parameters"],
            dense_shared_ffn_parameters=data["dense_shared_ffn_parameters"],
            routed_expert_capacity=data["routed_expert_capacity"],
            routed_active_parameters=data["routed_active_parameters"],
            num_experts=data["num_experts"],
            top_k=data["top_k"],
            census_digest=data["census_digest"],
            neuron_sparsity_measured=data["neuron_sparsity_measured"],
            neuron_sparsity_min=data["neuron_sparsity_min"],
            neuron_sparsity_max=data["neuron_sparsity_max"],
            neuron_active_parameters=data["neuron_active_parameters"],
            total_effective_active=data["total_effective_active"],
        )


_ALWAYS_ON_CATEGORIES = (
    "attention_and_deltanet",
    "layernorm",
    "embedding",
    "mtp",
    "vision",
    "router",
    "other",
)
_DENSE_SHARED_CATEGORIES = ("dense_ffn", "shared_expert")


def hierarchical_active_breakdown(
    accounting: ParameterAccounting,
    *,
    census_digest: str | None = None,
    per_layer_drelu_sparsity: Sequence[float] | None = None,
) -> HierarchicalActiveBreakdown:
    """Build the Phase 9 hierarchy from a real base accounting.

    ``per_layer_drelu_sparsity`` values are the census's measured
    ``sparsity_drelu_counterfactual`` per decoder layer (simple mean,
    min, max reported; layers weighted equally — the census records the
    per-layer values so finer weighting can be recomputed offline).
    """
    if not accounting.is_sparse:
        raise ParameterAccountingError(
            "hierarchical breakdown is for sparse (routed-expert) models; "
            "a dense model's active count is its total (see base accounting)"
        )
    if accounting.router_geometry is None or accounting.active_parameters is None:
        raise ParameterAccountingError(
            "base accounting lacks measured routing geometry; rebuild it "
            "from a real sparse model directory first"
        )
    cats = accounting.categories

    def params_of(names: tuple[str, ...]) -> int:
        return sum(cats.get(n).parameters for n in names if n in cats)

    always_on = params_of(_ALWAYS_ON_CATEGORIES)
    dense_shared = params_of(_DENSE_SHARED_CATEGORIES)
    routed_capacity = cats["routed_expert"].parameters
    geometry = accounting.router_geometry
    # Routed active/token: the base module computes active as
    # total - routed - router; cross-check our own composition instead of
    # reusing it, so the hierarchy is internally consistent by
    # construction (and disagreement is loud, not silent).
    routed_active = round(routed_capacity * geometry.top_k / geometry.num_experts)
    composed = always_on + dense_shared + routed_active
    if composed != accounting.active_parameters:
        raise ParameterAccountingError(
            f"hierarchy composition {composed:,} != base accounting active "
            f"{accounting.active_parameters:,}; definitions have drifted"
        )

    have_census = census_digest is not None
    have_sparsities = per_layer_drelu_sparsity is not None
    if have_census != have_sparsities:
        raise ParameterAccountingError(
            "census evidence requires BOTH census_digest and "
            "per_layer_drelu_sparsity (all-or-nothing, fail closed)"
        )
    neuron_active: int | None = None
    effective: int | None = None
    sparsity = sparsity_min = sparsity_max = None
    if have_census:
        values = [float(v) for v in per_layer_drelu_sparsity or []]
        if not values:
            raise ParameterAccountingError(
                "per_layer_drelu_sparsity was empty; a census over zero "
                "layers is not evidence"
            )
        if any(not (0.0 <= v < 1.0) for v in values):
            raise ParameterAccountingError(
                "measured sparsities must lie in [0, 1); got "
                f"{[v for v in values if not (0.0 <= v < 1.0)][:3]}"
            )
        if not isinstance(census_digest, str) or len(census_digest) < 12:
            raise ParameterAccountingError(
                "census_digest must be a real content digest (>= 12 chars)"
            )
        sparsity = sum(values) / len(values)
        sparsity_min = min(values)
        sparsity_max = max(values)
        neuron_active = round(routed_active * (1.0 - sparsity))
        effective = always_on + dense_shared + neuron_active

    return HierarchicalActiveBreakdown(
        model_dir=accounting.model_dir,
        definition_id=EFFECTIVE_DEFINITION_ID,
        total_parameters=accounting.total_parameters,
        always_on_parameters=always_on,
        dense_shared_ffn_parameters=dense_shared,
        routed_expert_capacity=routed_capacity,
        routed_active_parameters=routed_active,
        num_experts=geometry.num_experts,
        top_k=geometry.top_k,
        census_digest=census_digest,
        neuron_sparsity_measured=sparsity,
        neuron_sparsity_min=sparsity_min,
        neuron_sparsity_max=sparsity_max,
        neuron_active_parameters=neuron_active,
        total_effective_active=effective,
    )


def write_hierarchical_accounting(
    breakdown: HierarchicalActiveBreakdown, output_path: str | Path
) -> str:
    """Atomically persist the breakdown + formula document as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **breakdown.to_dict(),
        "formulas": breakdown.formula_document(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)
    return str(path)
