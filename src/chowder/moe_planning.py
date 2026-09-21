from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping


def _finite_nonnegative(value: float, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


@dataclass(frozen=True)
class ExpertImportance:
    """Aggregated evidence for one routed expert in one MoE layer.

    The fields intentionally remain separate rather than storing only one
    pre-computed score. Chowder can change the scoring rule later without
    rerunning the expensive teacher trace.
    """

    layer: int
    expert: int
    selected_tokens: int
    router_mass: float
    gated_activation: float
    output_norm: float

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        if self.expert < 0:
            raise ValueError("expert must be non-negative")
        if self.selected_tokens < 0:
            raise ValueError("selected_tokens must be non-negative")
        _finite_nonnegative(self.router_mass, label="router_mass")
        _finite_nonnegative(self.gated_activation, label="gated_activation")
        _finite_nonnegative(self.output_norm, label="output_norm")


@dataclass(frozen=True)
class ImportanceWeights:
    router_mass: float = 1.0
    gated_activation: float = 1.0
    output_norm: float = 1.0
    selected_tokens: float = 0.0

    def __post_init__(self) -> None:
        values = {
            "router_mass": self.router_mass,
            "gated_activation": self.gated_activation,
            "output_norm": self.output_norm,
            "selected_tokens": self.selected_tokens,
        }
        for name, value in values.items():
            _finite_nonnegative(value, label=f"weight {name}")
        if not any(float(value) > 0 for value in values.values()):
            raise ValueError("at least one importance weight must be positive")


@dataclass(frozen=True)
class LayerPruningPlan:
    layer: int
    total_experts: int
    keep_experts: tuple[int, ...]
    remove_experts: tuple[int, ...]

    @property
    def retention_fraction(self) -> float:
        return len(self.keep_experts) / self.total_experts


@dataclass(frozen=True)
class PruningPlan:
    requested_retention_fraction: float
    layers: tuple[LayerPruningPlan, ...]
    scores: Mapping[tuple[int, int], float]

    @property
    def total_experts(self) -> int:
        return sum(layer.total_experts for layer in self.layers)

    @property
    def kept_experts(self) -> int:
        return sum(len(layer.keep_experts) for layer in self.layers)

    @property
    def actual_retention_fraction(self) -> float:
        if self.total_experts == 0:
            return 0.0
        return self.kept_experts / self.total_experts


def _normalized(values: Mapping[int, float]) -> dict[int, float]:
    """Min/max normalize one layer's metric to [0, 1].

    A constant metric contains no ranking information, so every expert gets
    zero contribution from that metric instead of an arbitrary equal bonus.
    """
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if math.isclose(lo, hi, rel_tol=0.0, abs_tol=1e-15):
        return {key: 0.0 for key in values}
    span = hi - lo
    return {key: (value - lo) / span for key, value in values.items()}


def score_experts(
    records: Iterable[ExpertImportance],
    *,
    weights: ImportanceWeights | None = None,
) -> dict[tuple[int, int], float]:
    """Score experts using within-layer normalized evidence.

    Normalizing within each layer prevents one naturally high-magnitude layer
    from dominating another merely because its activation scale differs. The
    output is deterministic and intended for candidate planning, not as a
    claim that this weighting is already optimal.
    """
    weights = weights or ImportanceWeights()
    by_layer: dict[int, list[ExpertImportance]] = {}
    seen: set[tuple[int, int]] = set()
    for record in records:
        key = (record.layer, record.expert)
        if key in seen:
            raise ValueError(f"duplicate expert importance record for layer={record.layer} expert={record.expert}")
        seen.add(key)
        by_layer.setdefault(record.layer, []).append(record)

    scores: dict[tuple[int, int], float] = {}
    for layer, layer_records in by_layer.items():
        router = _normalized({r.expert: r.router_mass for r in layer_records})
        activation = _normalized({r.expert: r.gated_activation for r in layer_records})
        output = _normalized({r.expert: r.output_norm for r in layer_records})
        selected = _normalized({r.expert: float(r.selected_tokens) for r in layer_records})
        for record in layer_records:
            scores[(layer, record.expert)] = (
                weights.router_mass * router[record.expert]
                + weights.gated_activation * activation[record.expert]
                + weights.output_norm * output[record.expert]
                + weights.selected_tokens * selected[record.expert]
            )
    return scores


def build_uniform_pruning_plan(
    records: Iterable[ExpertImportance],
    *,
    retention_fraction: float,
    minimum_survivors_per_layer: int,
    weights: ImportanceWeights | None = None,
) -> PruningPlan:
    """Build a dry-run per-layer expert pruning plan.

    This function never mutates or writes model weights. It is deliberately a
    planning primitive so Chowder can regression-test expert selection before
    a model-specific Qwen writer exists.
    """
    fraction = float(retention_fraction)
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("retention_fraction must be finite and in (0, 1]")
    if minimum_survivors_per_layer <= 0:
        raise ValueError("minimum_survivors_per_layer must be positive")

    materialized = tuple(records)
    if not materialized:
        raise ValueError("at least one expert importance record is required")
    scores = score_experts(materialized, weights=weights)

    by_layer: dict[int, list[ExpertImportance]] = {}
    for record in materialized:
        by_layer.setdefault(record.layer, []).append(record)

    layer_plans: list[LayerPruningPlan] = []
    for layer in sorted(by_layer):
        layer_records = by_layer[layer]
        total = len(layer_records)
        if minimum_survivors_per_layer > total:
            raise ValueError(
                f"layer {layer} has only {total} experts but minimum_survivors_per_layer="
                f"{minimum_survivors_per_layer}"
            )
        requested = max(minimum_survivors_per_layer, math.ceil(total * fraction))
        requested = min(total, requested)

        # Highest score wins. Expert id is the stable tie-breaker, making a
        # plan reproducible even when multiple experts have identical evidence.
        ranked = sorted(
            layer_records,
            key=lambda record: (-scores[(layer, record.expert)], record.expert),
        )
        keep = tuple(sorted(record.expert for record in ranked[:requested]))
        keep_set = set(keep)
        remove = tuple(sorted(record.expert for record in layer_records if record.expert not in keep_set))
        layer_plans.append(
            LayerPruningPlan(
                layer=layer,
                total_experts=total,
                keep_experts=keep,
                remove_experts=remove,
            )
        )

    return PruningPlan(
        requested_retention_fraction=fraction,
        layers=tuple(layer_plans),
        scores=scores,
    )
