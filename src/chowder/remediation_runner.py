from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from .incident import EnvironmentSnapshot, capture_from_exception, compute_fingerprint
from .investigation import (
    HypothesisTrial,
    Investigation,
    RemediationOutcome,
    RemediationRecord,
    config_patch_digest,
    remediation_context_digest,
)
from .replay import GroundTruthMissingError


@runtime_checkable
class RemediationExecutor(Protocol):
    def run(self, config_patch: Mapping[str, Any]) -> RemediationOutcome:
        ...


@dataclass(frozen=True)
class RemediationExperiment:
    """Run one remediation hypothesis under attempt and GPU-hour limits.

    Budget is checked before *every* attempt. The investigation is charged by
    its caller only after this record is accepted, so this method reserves
    locally against the investigation's current remaining budget and never
    returns a record whose measured/declared remediation spend exceeds it.
    """

    executor: RemediationExecutor
    max_attempts: int = 1
    gpu_hours_per_attempt: float = 0.0

    def run(
        self,
        investigation: Investigation,
        trial: HypothesisTrial,
        *,
        environment: EnvironmentSnapshot,
    ) -> RemediationRecord:
        if investigation.remaining_budget() <= 0:
            raise ValueError(
                f"investigation {investigation.investigation_id} has no remaining "
                "GPU-hour budget -- cannot run a remediation experiment"
            )
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.gpu_hours_per_attempt < 0:
            raise ValueError("gpu_hours_per_attempt cannot be negative")

        remediation_id = (
            f"{investigation.investigation_id}-{config_patch_digest(trial.config_patch)[:12]}"
        )
        available_budget = investigation.remaining_budget()
        context_sha = remediation_context_digest(investigation.capture)

        attempts_used = 0
        outcome: RemediationOutcome | None = None
        for attempt in range(1, self.max_attempts + 1):
            projected_spend = self.gpu_hours_per_attempt * attempt
            if projected_spend > available_budget + 1e-12:
                break
            attempts_used = attempt
            try:
                outcome = self.executor.run(trial.config_patch)
            except GroundTruthMissingError:
                raise
            except Exception as exc:  # noqa: BLE001 -- executor exceptions are incident evidence
                new_capture = capture_from_exception(
                    exc,
                    incident_id=f"{remediation_id}-attempt-{attempt}",
                    experiment_id=investigation.capture.experiment_id,
                    executor_name=investigation.capture.executor_name,
                    occurred_at=investigation.capture.occurred_at,
                    environment=environment,
                    attempt_number=attempt,
                    gpu_hours_spent=self.gpu_hours_per_attempt,
                )
                new_fingerprint = compute_fingerprint(new_capture)
                return RemediationRecord(
                    remediation_id=f"{remediation_id}-partial-{attempt}",
                    fingerprint_sha256=investigation.fingerprint.fingerprint_sha256,
                    signature_kind=investigation.fingerprint.signature_kind,
                    description=(
                        f"attempt {attempt} raised a new exception during "
                        f"remediation: {type(exc).__qualname__}: {exc}"
                    ),
                    config_patch=trial.config_patch,
                    outcome=RemediationOutcome.PARTIALLY_RESOLVED,
                    attempts_used=attempt,
                    gpu_hours_spent=projected_spend,
                    notes=f"spawned_signature_kind={new_fingerprint.signature_kind.value}",
                    context_sha256=context_sha,
                    spawned_incident=new_capture,
                )
            else:
                if outcome is RemediationOutcome.RESOLVED:
                    break

        if attempts_used == 0:
            raise ValueError(
                "insufficient remaining GPU-hour budget for one remediation attempt"
            )
        assert outcome is not None
        return RemediationRecord(
            remediation_id=remediation_id,
            fingerprint_sha256=investigation.fingerprint.fingerprint_sha256,
            signature_kind=investigation.fingerprint.signature_kind,
            description=trial.hypothesis.intervention,
            config_patch=trial.config_patch,
            outcome=outcome,
            attempts_used=attempts_used,
            gpu_hours_spent=self.gpu_hours_per_attempt * attempts_used,
            context_sha256=context_sha,
        )
