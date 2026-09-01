from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from .memory import HardwareProfile
from .models import Experiment, ExperimentResult


@dataclass(frozen=True)
class CostEstimate:
    gpu_hours: float
    peak_vram_gb: float | None = None
    confidence: float = 0.5
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionContext:
    hardware: HardwareProfile
    work_dir: str
    seed: int


@dataclass(frozen=True)
class TrainingArtifact:
    """Output of a training backend before independent evaluation.

    Training telemetry is intentionally *not* stored in ``ExperimentResult``.
    That prevents train loss, throughput, or backend-specific statistics from
    being mistaken for benchmark evidence by the promotion gate.
    """

    run_id: str
    experiment_id: str
    artifact_ref: str
    gpu_hours: float
    telemetry: Mapping[str, float | int | str] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class TrainingExecutor(Protocol):
    """Backend contract. Decision logic must never depend on framework internals."""

    name: str

    def profile(self, experiment: Experiment, context: ExecutionContext) -> CostEstimate:
        ...

    def run(self, experiment: Experiment, context: ExecutionContext) -> TrainingArtifact:
        ...

    def cancel(self, run_id: str) -> None:
        ...


@runtime_checkable
class EvaluationExecutor(Protocol):
    """Independent evaluation contract used to produce promotion evidence."""

    name: str

    def evaluate(
        self,
        *,
        experiment: Experiment,
        artifact: TrainingArtifact,
        context: ExecutionContext,
    ) -> ExperimentResult:
        ...
