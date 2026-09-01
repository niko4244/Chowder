from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

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


@runtime_checkable
class TrainingExecutor(Protocol):
    """Backend contract. Decision logic must never depend on framework internals."""

    name: str

    def profile(self, experiment: Experiment, context: ExecutionContext) -> CostEstimate:
        ...

    def run(self, experiment: Experiment, context: ExecutionContext) -> ExperimentResult:
        ...

    def cancel(self, run_id: str) -> None:
        ...
