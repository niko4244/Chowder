from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class OptimizationDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ExperimentStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    REJECTED = "rejected"


def _finite(value: float, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


@dataclass(frozen=True)
class MetricTarget:
    name: str
    minimum: float | None = None
    maximum: float | None = None
    weight: float = 1.0
    regression_tolerance: float = 0.0
    direction: OptimizationDirection = OptimizationDirection.MAXIMIZE

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("metric target name must be a non-empty string")
        minimum = None if self.minimum is None else _finite(self.minimum, label=f"{self.name} minimum")
        maximum = None if self.maximum is None else _finite(self.maximum, label=f"{self.name} maximum")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"metric target {self.name!r} minimum cannot exceed maximum")
        if _finite(self.weight, label=f"{self.name} weight") <= 0:
            raise ValueError(f"metric target {self.name!r} weight must be positive")
        if _finite(
            self.regression_tolerance,
            label=f"{self.name} regression_tolerance",
        ) < 0:
            raise ValueError(
                f"metric target {self.name!r} regression_tolerance cannot be negative"
            )

    def utility_delta(self, baseline: float, candidate: float) -> float:
        base = _finite(baseline, label=f"{self.name} baseline")
        value = _finite(candidate, label=f"{self.name} candidate")
        raw = value - base
        return raw if self.direction is OptimizationDirection.MAXIMIZE else -raw

    def target_met(self, value: float) -> bool:
        value = _finite(value, label=f"{self.name} value")
        if self.minimum is not None and value < self.minimum:
            return False
        if self.maximum is not None and value > self.maximum:
            return False
        return True


@dataclass(frozen=True)
class Goal:
    metrics: tuple[MetricTarget, ...]
    gpu_hour_budget: float
    max_parallel_candidates: int = 4
    minimum_promotion_gain: float = 0.0
    require_protocol_match: bool = False

    def __post_init__(self) -> None:
        if not self.metrics:
            raise ValueError("goal must contain at least one metric target")
        names = tuple(target.name for target in self.metrics)
        if len(names) != len(set(names)):
            raise ValueError("goal metric target names must be unique")
        budget = _finite(self.gpu_hour_budget, label="goal gpu_hour_budget")
        if budget < 0:
            raise ValueError("goal gpu_hour_budget cannot be negative")
        if (
            not isinstance(self.max_parallel_candidates, int)
            or isinstance(self.max_parallel_candidates, bool)
            or self.max_parallel_candidates <= 0
        ):
            raise ValueError("goal max_parallel_candidates must be a positive integer")
        _finite(self.minimum_promotion_gain, label="goal minimum_promotion_gain")

    def target(self, name: str) -> MetricTarget | None:
        return next((m for m in self.metrics if m.name == name), None)


@dataclass(frozen=True)
class Hypothesis:
    observation: str
    suspected_cause: str
    intervention: str
    expected_deltas: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value in (
            ("observation", self.observation),
            ("suspected_cause", self.suspected_cause),
            ("intervention", self.intervention),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"hypothesis {label} must be a non-empty string")
        for name, value in self.expected_deltas.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("hypothesis expected delta names must be non-empty strings")
            _finite(value, label=f"hypothesis expected delta {name}")


@dataclass
class Experiment:
    experiment_id: str
    parent_id: str | None
    hypothesis: Hypothesis
    config_patch: dict[str, Any]
    estimated_gpu_hours: float
    status: ExperimentStatus = ExperimentStatus.PLANNED
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_id, str) or not self.experiment_id.strip():
            raise ValueError("experiment_id must be a non-empty string")
        if self.parent_id is not None and (
            not isinstance(self.parent_id, str) or not self.parent_id.strip()
        ):
            raise ValueError("parent_id must be None or a non-empty string")
        estimate = _finite(self.estimated_gpu_hours, label="experiment estimated_gpu_hours")
        if estimate <= 0:
            raise ValueError("experiment estimated_gpu_hours must be positive")


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    metrics: Mapping[str, float]
    gpu_hours: float
    artifact_ref: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_id, str) or not self.experiment_id.strip():
            raise ValueError("experiment result experiment_id must be a non-empty string")
        hours = _finite(self.gpu_hours, label="experiment result gpu_hours")
        if hours < 0:
            raise ValueError("experiment result gpu_hours cannot be negative")
        if not self.metrics:
            raise ValueError("experiment result must contain metrics")
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("experiment result metric names must be non-empty strings")
            _finite(value, label=f"experiment result metric {name}")


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    score: float
    regressions: Mapping[str, float]
    unmet_targets: tuple[str, ...]
    missing_metrics: tuple[str, ...]
    goal_met: bool
    reason: str
