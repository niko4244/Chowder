from __future__ import annotations

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


@dataclass(frozen=True)
class MetricTarget:
    name: str
    minimum: float | None = None
    maximum: float | None = None
    weight: float = 1.0
    regression_tolerance: float = 0.0
    direction: OptimizationDirection = OptimizationDirection.MAXIMIZE

    def utility_delta(self, baseline: float, candidate: float) -> float:
        raw = candidate - baseline
        return raw if self.direction is OptimizationDirection.MAXIMIZE else -raw

    def target_met(self, value: float) -> bool:
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

    def target(self, name: str) -> MetricTarget | None:
        return next((m for m in self.metrics if m.name == name), None)


@dataclass(frozen=True)
class Hypothesis:
    observation: str
    suspected_cause: str
    intervention: str
    expected_deltas: Mapping[str, float] = field(default_factory=dict)


@dataclass
class Experiment:
    experiment_id: str
    parent_id: str | None
    hypothesis: Hypothesis
    config_patch: dict[str, Any]
    estimated_gpu_hours: float
    status: ExperimentStatus = ExperimentStatus.PLANNED
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    metrics: Mapping[str, float]
    gpu_hours: float
    artifact_ref: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    score: float
    regressions: Mapping[str, float]
    unmet_targets: tuple[str, ...]
    missing_metrics: tuple[str, ...]
    goal_met: bool
    reason: str
