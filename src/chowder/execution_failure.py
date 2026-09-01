from __future__ import annotations

import math
import traceback
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4

from .executors import ExecutionContext
from .models import Experiment
from .resources import ResourceUsage


class ExecutionStage(str, Enum):
    PREPARE = "prepare"
    LAUNCH = "launch"
    TRAIN = "train"
    FINALIZE = "finalize"
    EVALUATE = "evaluate"


class ExecutionFailure(RuntimeError):
    """Structured failure emitted or normalized at the execution boundary."""

    def __init__(
        self,
        message: str,
        *,
        run_id: str,
        experiment_id: str,
        executor_name: str,
        stage: ExecutionStage,
        cause_type: str,
        cause_message: str,
        traceback_text: str = "",
        resource_usage: ResourceUsage | None = None,
        stdout_ref: str | None = None,
        stderr_ref: str | None = None,
        partial_artifact_ref: str | None = None,
        exit_code: int | None = None,
        runtime_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        for label, value in (
            ("run_id", run_id),
            ("experiment_id", experiment_id),
            ("executor_name", executor_name),
            ("cause_type", cause_type),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"execution failure {label} must be a non-empty string")
        if not isinstance(cause_message, str):
            raise ValueError("execution failure cause_message must be a string")
        if exit_code is not None and (
            not isinstance(exit_code, int) or isinstance(exit_code, bool)
        ):
            raise ValueError("execution failure exit_code must be an integer or None")

        self.run_id = run_id
        self.experiment_id = experiment_id
        self.executor_name = executor_name
        self.stage = stage
        self.cause_type = cause_type
        self.cause_message = cause_message
        self.traceback_text = traceback_text
        self.resource_usage = resource_usage
        self.stdout_ref = stdout_ref
        self.stderr_ref = stderr_ref
        self.partial_artifact_ref = partial_artifact_ref
        self.exit_code = exit_code
        self.runtime_metadata = dict(runtime_metadata or {})

    @property
    def gpu_hours_spent(self) -> float | None:
        if self.resource_usage is None:
            return None
        value = float(self.resource_usage.gpu_hours)
        if not math.isfinite(value) or value < 0:
            raise ValueError("execution failure resource usage is invalid")
        return value


def declared_active_accelerator_count(context: ExecutionContext) -> int:
    """Best available pre-worker accelerator count for conservative crash cost.

    Cloud runtimes such as Kaggle should declare the count in
    ``backend.runtime.active_accelerator_count`` (or the measured profile). For
    example a Kaggle T4x2 session declares ``2``. If no declaration exists,
    Chowder conservatively assumes one accelerator when the hardware profile has
    VRAM and zero for CPU-only execution.
    """

    config = context.resolved_config
    backend = config.get("backend", {}) if isinstance(config, Mapping) else {}
    if isinstance(backend, Mapping):
        profile = backend.get("profile", {})
        if isinstance(profile, Mapping) and profile.get("active_accelerator_count") is not None:
            count = int(profile["active_accelerator_count"])
            if count < 0:
                raise ValueError("active_accelerator_count cannot be negative")
            return count
        runtime = backend.get("runtime", {})
        if isinstance(runtime, Mapping) and runtime.get("active_accelerator_count") is not None:
            count = int(runtime["active_accelerator_count"])
            if count < 0:
                raise ValueError("active_accelerator_count cannot be negative")
            return count
    return 1 if context.hardware.vram_gb > 0 else 0


def normalize_execution_exception(
    exc: BaseException,
    *,
    experiment: Experiment,
    executor_name: str,
    context: ExecutionContext,
    wall_seconds: float,
) -> ExecutionFailure:
    """Convert a legacy/raw executor exception without discarding evidence."""

    if isinstance(exc, ExecutionFailure):
        return exc
    active = declared_active_accelerator_count(context)
    usage = ResourceUsage.from_wall_time(
        wall_seconds=max(0.0, float(wall_seconds)),
        active_accelerator_count=active,
        visible_accelerator_count=active,
    )
    return ExecutionFailure(
        f"{executor_name} failed during training: {type(exc).__qualname__}: {exc}",
        run_id=f"{experiment.experiment_id}-failed-{uuid4().hex[:12]}",
        experiment_id=experiment.experiment_id,
        executor_name=executor_name,
        stage=ExecutionStage.TRAIN,
        cause_type=type(exc).__qualname__,
        cause_message=str(exc),
        traceback_text="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        resource_usage=usage,
        runtime_metadata={"normalized_from_raw_exception": True},
    )
