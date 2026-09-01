from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .incident import FailureCapture, IncidentFingerprint, SignatureKind
from .models import Hypothesis


class RemediationOutcome(str, Enum):
    RESOLVED = "resolved"
    DID_NOT_RESOLVE = "did_not_resolve"
    PARTIALLY_RESOLVED = "partially_resolved"


def config_patch_digest(config_patch: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(config_patch), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def remediation_context_digest(capture: FailureCapture) -> str:
    """Bind an auto-applicable remediation to the runtime context it fixed.

    ``IncidentFingerprint`` intentionally normalizes volatile error text and is
    useful for bucketing recurring incidents. It is not sufficient by itself to
    prove that a previously successful fix is safe to auto-apply: package pins,
    hardware, executor configuration, or other environment state may have
    changed while the normalized exception stayed the same.
    """

    environment = capture.environment
    payload = {
        "executor_name": capture.executor_name,
        "exception_type": capture.exception_type,
        "hardware_summary": environment.hardware_summary,
        "accelerator_count": environment.accelerator_count,
        "installed_packages": dict(environment.installed_packages),
        "config_patch": dict(environment.config_patch),
        "extra": dict(environment.extra),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RemediationRecord:
    """A remediation that was actually attempted against one incident.

    Automatic reuse requires both the normalized incident fingerprint and the
    exact captured runtime-context digest. ``signature_kind`` remains useful for
    class-level investigation history but is never enough to auto-apply a fix.
    ``spawned_incident`` preserves a full new failure capture when a remediation
    gets further and then crashes differently.
    """

    remediation_id: str
    fingerprint_sha256: str
    signature_kind: SignatureKind
    description: str
    config_patch: Mapping[str, Any]
    outcome: RemediationOutcome
    attempts_used: int
    gpu_hours_spent: float
    notes: str = ""
    context_sha256: str | None = None
    spawned_incident: FailureCapture | None = None


@dataclass(frozen=True)
class RemediationRegistry:
    """Known remediation history with conservative exact-context reuse."""

    records: tuple[RemediationRecord, ...] = ()

    def lookup(
        self,
        fingerprint: IncidentFingerprint,
        *,
        capture: FailureCapture | None = None,
    ) -> RemediationRecord | None:
        wanted_context = remediation_context_digest(capture) if capture is not None else None
        for record in self.records:
            if record.outcome is not RemediationOutcome.RESOLVED:
                continue
            if record.fingerprint_sha256 != fingerprint.fingerprint_sha256:
                continue
            # Legacy/context-free records remain inspectable history, but may not
            # be auto-applied to a live capture because their environment cannot
            # be proven equivalent.
            if capture is not None:
                if record.context_sha256 is None or record.context_sha256 != wanted_context:
                    continue
            elif record.context_sha256 is not None:
                # Callers asking only by fingerprint cannot prove environment
                # equivalence, so do not return context-bound fixes.
                continue
            return record
        return None

    def class_history(self, signature_kind: SignatureKind) -> tuple[RemediationRecord, ...]:
        return tuple(r for r in self.records if r.signature_kind is signature_kind)

    def already_failed_for_class(
        self, signature_kind: SignatureKind, config_patch: Mapping[str, Any]
    ) -> RemediationRecord | None:
        digest = config_patch_digest(config_patch)
        for record in self.class_history(signature_kind):
            if record.outcome is RemediationOutcome.DID_NOT_RESOLVE and (
                config_patch_digest(record.config_patch) == digest
            ):
                return record
        return None

    def with_record(self, record: RemediationRecord) -> "RemediationRegistry":
        return RemediationRegistry(records=self.records + (record,))


class InvestigationStatus(str, Enum):
    OPEN = "open"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"


@dataclass(frozen=True)
class DiagnosticProbeResult:
    probe_id: str
    description: str
    observation: Mapping[str, Any]


@dataclass
class HypothesisTrial:
    hypothesis: Hypothesis
    config_patch: Mapping[str, Any]
    estimated_gpu_hours: float = 0.0
    probe_results: tuple[DiagnosticProbeResult, ...] = ()
    remediation: RemediationRecord | None = None
    rank: float | None = None


@dataclass
class Investigation:
    investigation_id: str
    fingerprint: IncidentFingerprint
    capture: FailureCapture
    gpu_hour_budget: float
    trials: list[HypothesisTrial] = field(default_factory=list)
    status: InvestigationStatus = InvestigationStatus.OPEN
    gpu_hours_spent: float = 0.0

    def remaining_budget(self) -> float:
        return max(0.0, self.gpu_hour_budget - self.gpu_hours_spent)

    def add_hypothesis(
        self,
        hypothesis: Hypothesis,
        *,
        config_patch: Mapping[str, Any],
        estimated_gpu_hours: float = 0.0,
        registry: RemediationRegistry | None = None,
    ) -> HypothesisTrial:
        if self.remaining_budget() <= 0:
            raise ValueError(
                f"investigation {self.investigation_id} has no remaining GPU-hour budget"
            )
        if registry is not None:
            prior_failure = registry.already_failed_for_class(
                self.fingerprint.signature_kind, config_patch
            )
            if prior_failure is not None:
                raise ValueError(
                    f"intervention already failed for signature "
                    f"{self.fingerprint.signature_kind.value!r} "
                    f"(remediation_id={prior_failure.remediation_id!r}); "
                    "propose a genuinely different intervention instead"
                )
        trial = HypothesisTrial(
            hypothesis=hypothesis,
            config_patch=config_patch,
            estimated_gpu_hours=estimated_gpu_hours,
        )
        self.trials.append(trial)
        if self.status is InvestigationStatus.OPEN:
            self.status = InvestigationStatus.HYPOTHESIS_TESTING
        return trial

    def record_probe(self, trial: HypothesisTrial, probe_result: DiagnosticProbeResult) -> None:
        trial.probe_results = trial.probe_results + (probe_result,)

    def resolve(self, trial: HypothesisTrial, remediation: RemediationRecord) -> None:
        if remediation.outcome is not RemediationOutcome.RESOLVED:
            raise ValueError("cannot resolve an investigation with a non-resolving remediation")
        if remediation.gpu_hours_spent > self.remaining_budget() + 1e-12:
            raise ValueError("remediation exceeds investigation GPU-hour budget")
        trial.remediation = remediation
        self.gpu_hours_spent += remediation.gpu_hours_spent
        self.status = InvestigationStatus.RESOLVED

    def record_failed_trial(self, trial: HypothesisTrial, remediation: RemediationRecord) -> None:
        if remediation.gpu_hours_spent > self.remaining_budget() + 1e-12:
            raise ValueError("remediation exceeds investigation GPU-hour budget")
        trial.remediation = remediation
        self.gpu_hours_spent += remediation.gpu_hours_spent
        if self.remaining_budget() <= 0:
            self.status = InvestigationStatus.ABANDONED

    def abandon(self) -> None:
        if self.status is InvestigationStatus.RESOLVED:
            raise ValueError(f"investigation {self.investigation_id} is already resolved")
        self.status = InvestigationStatus.ABANDONED


def route_failure(
    capture: FailureCapture,
    fingerprint: IncidentFingerprint,
    registry: RemediationRegistry,
    *,
    gpu_hour_budget: float,
    investigation_id: str,
) -> RemediationRecord | Investigation:
    """Auto-apply only a fix proven on this exact incident *and* runtime context."""

    known = registry.lookup(fingerprint, capture=capture)
    if known is not None:
        return known
    return Investigation(
        investigation_id=investigation_id,
        fingerprint=fingerprint,
        capture=capture,
        gpu_hour_budget=gpu_hour_budget,
    )
