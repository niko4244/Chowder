from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from .memory import HardwareProfile
from .models import Experiment
from .resources import ResourceUsage


@dataclass(frozen=True)
class CostEstimate:
    gpu_hours: float
    peak_vram_gb: float | None = None
    confidence: float = 0.5
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        hours = float(self.gpu_hours)
        if not math.isfinite(hours) or hours < 0:
            raise ValueError("cost estimate gpu_hours must be finite and non-negative")
        if self.peak_vram_gb is not None:
            peak = float(self.peak_vram_gb)
            if not math.isfinite(peak) or peak < 0:
                raise ValueError("cost estimate peak_vram_gb must be finite and non-negative")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("cost estimate confidence must be finite and in [0, 1]")


@dataclass(frozen=True)
class ExecutionContext:
    hardware: HardwareProfile
    work_dir: str
    seed: int
    resolved_config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingArtifact:
    """Output of a training backend before independent evaluation."""

    run_id: str
    experiment_id: str
    artifact_ref: str
    gpu_hours: float
    telemetry: Mapping[str, float | int | str] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    resource_usage: ResourceUsage | None = None

    def __post_init__(self) -> None:
        hours = float(self.gpu_hours)
        if not math.isfinite(hours) or hours < 0:
            raise ValueError("training artifact gpu_hours must be finite and non-negative")
        for label, value in (
            ("run_id", self.run_id),
            ("experiment_id", self.experiment_id),
            ("artifact_ref", self.artifact_ref),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"training artifact {label} must be a non-empty string")
        if self.resource_usage is not None and not math.isclose(
            hours,
            self.resource_usage.gpu_hours,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("training artifact gpu_hours disagrees with resource usage")


@runtime_checkable
class TrainingExecutor(Protocol):
    name: str

    def profile(self, experiment: Experiment, context: ExecutionContext) -> CostEstimate:
        ...

    def run(self, experiment: Experiment, context: ExecutionContext) -> TrainingArtifact:
        ...

    def cancel(self, run_id: str) -> None:
        ...


@dataclass(frozen=True)
class EvaluationOutcome:
    """Independent benchmark evidence before lifecycle cost is finalized."""

    run_id: str
    experiment_id: str
    source_artifact_ref: str
    metrics: Mapping[str, float]
    gpu_hours: float
    evidence: Mapping[str, Any] = field(default_factory=dict)
    resource_usage: ResourceUsage | None = None

    def __post_init__(self) -> None:
        hours = float(self.gpu_hours)
        if not math.isfinite(hours) or hours < 0:
            raise ValueError("evaluation outcome gpu_hours must be finite and non-negative")
        for label, value in (
            ("run_id", self.run_id),
            ("experiment_id", self.experiment_id),
            ("source_artifact_ref", self.source_artifact_ref),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"evaluation outcome {label} must be a non-empty string")
        if not self.metrics:
            raise ValueError("evaluation outcome must contain metrics")
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("evaluation metric names must be non-empty strings")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"evaluation metric {name!r} must be finite")
        if self.resource_usage is not None and not math.isclose(
            hours,
            self.resource_usage.gpu_hours,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("evaluation outcome gpu_hours disagrees with resource usage")


@runtime_checkable
class EvaluationExecutor(Protocol):
    name: str

    def profile(self, experiment: Experiment, context: ExecutionContext) -> CostEstimate:
        ...

    def evaluate(
        self,
        *,
        experiment: Experiment,
        artifact: TrainingArtifact,
        context: ExecutionContext,
    ) -> EvaluationOutcome:
        ...

    def cancel(self, run_id: str) -> None:
        ...
