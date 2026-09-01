from __future__ import annotations

import math
from enum import Enum
from typing import Any, Mapping

from .resources import ResourceUsage


class ExecutionStage(str, Enum):
    PREPARE = "prepare"
    LAUNCH = "launch"
    TRAIN = "train"
    FINALIZE = "finalize"
    EVALUATE = "evaluate"


class ExecutionFailure(RuntimeError):
    """Structured failure emitted by an execution backend.

    The exception remains usable as a normal ``RuntimeError`` while preserving
    the evidence an investigator needs after a worker process has disappeared.
    In particular, partial resource usage survives a crash so failed runs are not
    treated as free compute.
    """

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
