from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from .execution_failure import ExecutionFailure
from .executors import ExecutionContext
from .incident import EnvironmentSnapshot, FailureCapture, IncidentFingerprint, compute_fingerprint
from .investigation import Investigation, RemediationRecord, RemediationRegistry, route_failure


@dataclass(frozen=True)
class ExecutorFailureAnalysis:
    capture: FailureCapture
    fingerprint: IncidentFingerprint
    routed: RemediationRecord | Investigation


def _hardware_summary(context: ExecutionContext) -> str:
    hardware = context.hardware
    payload = {
        "vram_gb": hardware.vram_gb,
        "ram_gb": hardware.ram_gb,
        "nvme_gb": hardware.nvme_gb,
        "pcie_gbps": hardware.pcie_gbps,
        "ram_gbps": hardware.ram_gbps,
        "nvme_gbps": hardware.nvme_gbps,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def capture_execution_failure(
    failure: ExecutionFailure,
    *,
    context: ExecutionContext,
    occurred_at: str,
    attempt_number: int = 1,
) -> FailureCapture:
    usage = failure.resource_usage
    extra: dict[str, Any] = {
        "execution_stage": failure.stage.value,
        "stdout_ref": failure.stdout_ref,
        "stderr_ref": failure.stderr_ref,
        "exit_code": failure.exit_code,
        "runtime_metadata": dict(failure.runtime_metadata),
    }
    if usage is not None:
        extra["resource_usage"] = {
            "wall_seconds": usage.wall_seconds,
            "accelerator_seconds": usage.accelerator_seconds,
            "active_accelerator_count": usage.active_accelerator_count,
            "visible_accelerator_count": usage.visible_accelerator_count,
            "peak_vram_gb_by_accelerator": dict(usage.peak_vram_gb_by_accelerator),
        }

    installed = failure.runtime_metadata.get("installed_packages", {})
    if not isinstance(installed, Mapping):
        installed = {}
    environment = EnvironmentSnapshot(
        hardware_summary=_hardware_summary(context),
        accelerator_count=(usage.visible_accelerator_count if usage is not None else 0),
        installed_packages={str(k): str(v) for k, v in installed.items()},
        config_patch=dict(context.resolved_config),
        extra=extra,
    )
    return FailureCapture(
        incident_id=f"execution-{failure.run_id}-attempt-{attempt_number}",
        experiment_id=failure.experiment_id,
        executor_name=failure.executor_name,
        occurred_at=occurred_at,
        exception_type=failure.cause_type,
        exception_message=failure.cause_message,
        traceback_text=failure.traceback_text,
        environment=environment,
        attempt_number=attempt_number,
        gpu_hours_spent=failure.gpu_hours_spent or 0.0,
        run_id=failure.run_id,
        partial_artifact_ref=failure.partial_artifact_ref,
    )


def analyze_execution_failure(
    failure: ExecutionFailure,
    *,
    context: ExecutionContext,
    registry: RemediationRegistry,
    gpu_hour_budget: float,
    investigation_id: str,
    occurred_at: str,
) -> ExecutorFailureAnalysis:
    capture = capture_execution_failure(
        failure,
        context=context,
        occurred_at=occurred_at,
    )
    fingerprint = compute_fingerprint(capture)
    routed = route_failure(
        capture,
        fingerprint,
        registry,
        gpu_hour_budget=gpu_hour_budget,
        investigation_id=investigation_id,
    )
    return ExecutorFailureAnalysis(
        capture=capture,
        fingerprint=fingerprint,
        routed=routed,
    )
