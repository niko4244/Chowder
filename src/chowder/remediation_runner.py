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
)
from .replay import GroundTruthMissingError


@runtime_checkable
class RemediationExecutor(Protocol):
    """What a remediation attempt needs from an executor.

    Matches ``ReplayExecutor``'s shape (``replay.py``) so tests run against
    replayed ground truth; a live implementation would apply
    ``config_patch`` to a real training invocation instead. Raising is a
    legitimate outcome of ``run`` -- it means the attempt itself caused a
    new problem, handled explicitly below, not something callers are
    expected to prevent by construction.
    """

    def run(self, config_patch: Mapping[str, Any]) -> RemediationOutcome:
        ...


@dataclass(frozen=True)
class RemediationExperiment:
    """Runs one hypothesis trial against an executor, bounded on two axes.

    ``max_attempts`` (per hypothesis) and the investigation's own
    ``remaining_budget()`` (GPU-hours) are independent limits -- neither
    alone is sufficient. A single very expensive attempt could exhaust the
    whole budget in one try regardless of an attempts cap; a cap on
    attempts alone doesn't stop one long attempt from blowing through
    budget before a second attempt is even considered.
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

        remediation_id = (
            f"{investigation.investigation_id}-{config_patch_digest(trial.config_patch)[:12]}"
        )

        attempts_used = 0
        outcome: RemediationOutcome | None = None
        for attempt in range(1, self.max_attempts + 1):
            attempts_used = attempt
            try:
                outcome = self.executor.run(trial.config_patch)
            except GroundTruthMissingError:
                # Not incident evidence -- a benchmark/replay fixture that
                # never defined an outcome for this patch is a test-data
                # bug and must fail loudly, never be reinterpreted as "the
                # remediation attempt caused a new problem."
                raise
            except Exception as exc:  # noqa: BLE001 -- deliberately broad: any
                # OTHER exception the executor raises here IS the new
                # incident's evidence, not a bug in this runner to narrow
                # away.
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
                # Deliberately not opening a new Investigation here -- that
                # decision (route_failure on new_capture) belongs to
                # whatever is driving the loop, keeping this runner's job
                # limited to "run one trial, report what happened."
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
                    gpu_hours_spent=self.gpu_hours_per_attempt * attempt,
                    notes=f"new_incident_fingerprint_sha256={new_fingerprint.fingerprint_sha256}",
                )
            else:
                if outcome is RemediationOutcome.RESOLVED:
                    break

        assert outcome is not None  # loop always runs at least once (max_attempts >= 1)
        return RemediationRecord(
            remediation_id=remediation_id,
            fingerprint_sha256=investigation.fingerprint.fingerprint_sha256,
            signature_kind=investigation.fingerprint.signature_kind,
            description=trial.hypothesis.intervention,
            config_patch=trial.config_patch,
            outcome=outcome,
            attempts_used=attempts_used,
            gpu_hours_spent=self.gpu_hours_per_attempt * attempts_used,
        )
