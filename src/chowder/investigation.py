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
    PARTIALLY_RESOLVED = "partially_resolved"  # got further, then hit a different incident


def config_patch_digest(config_patch: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(config_patch), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RemediationRecord:
    """A remediation that has actually been tried before, with its outcome.

    ``fingerprint_sha256`` ties this to the exact incident it was tried
    against. ``signature_kind`` is kept alongside for class-level lookups
    (checking history across similar-but-not-identical incidents), but
    class similarity alone is never sufficient to auto-apply a fix -- see
    ``RemediationRegistry.lookup``.
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


@dataclass(frozen=True)
class RemediationRegistry:
    """Known remediation history.

    Deliberately conservative: ``lookup`` only ever returns a "known
    remediation, apply it" answer for an *exact* fingerprint match that
    previously resolved cleanly. Same-class-but-not-identical incidents are
    a real, common trap -- two of today's real incidents were both CUDA
    RuntimeErrors in the same custom-op family (qwen3_5's conv1d "no
    engine" vs. its gated-delta-rule cuBLAS failure), and the fix for one
    did nothing for the other. Auto-applying a same-class fix would have
    cost a full wasted attempt before discovering that. Class history is
    still useful -- see ``class_history`` and ``already_failed_for_class``
    -- but only to inform a real investigation, never to bypass one.
    """

    records: tuple[RemediationRecord, ...] = ()

    def lookup(self, fingerprint: IncidentFingerprint) -> RemediationRecord | None:
        for record in self.records:
            if (
                record.fingerprint_sha256 == fingerprint.fingerprint_sha256
                and record.outcome is RemediationOutcome.RESOLVED
            ):
                return record
        return None

    def class_history(self, signature_kind: SignatureKind) -> tuple[RemediationRecord, ...]:
        return tuple(r for r in self.records if r.signature_kind is signature_kind)

    def already_failed_for_class(
        self, signature_kind: SignatureKind, config_patch: Mapping[str, Any]
    ) -> RemediationRecord | None:
        """Has this exact intervention already failed against this failure class?

        Identity is the config patch's content digest, not free-text
        description, so two hypotheses that describe the same underlying
        change in different words are still recognized as the same
        intervention.
        """
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
    ABANDONED = "abandoned"  # exhausted its GPU-hour budget without resolving


@dataclass(frozen=True)
class DiagnosticProbeResult:
    """Cheap evidence-gathering, distinct from a remediation attempt.

    A probe observes (checks installed versions, queries available hardware
    shapes, inspects a partial artifact) -- it must never change execution
    state. That distinction is what keeps "gather more evidence" cheap and
    safe to run liberally, unlike a remediation trial, which is a real
    bounded experiment with its own GPU-hour cost.
    """

    probe_id: str
    description: str
    observation: Mapping[str, Any]


@dataclass
class HypothesisTrial:
    """One hypothesis being tested against one incident's investigation."""

    hypothesis: Hypothesis
    config_patch: Mapping[str, Any]
    estimated_gpu_hours: float = 0.0
    probe_results: tuple[DiagnosticProbeResult, ...] = ()
    remediation: RemediationRecord | None = None
    rank: float | None = None


@dataclass
class Investigation:
    """The hypothesis graph for one incident with no known remediation.

    Only reached via ``route_failure`` when ``RemediationRegistry.lookup``
    finds nothing -- i.e. this exact incident has never resolved cleanly
    before. Chowder's own reasoning does not live here: this is a state
    container an external driver (a hypothesis generator, an agent, a rule
    engine) populates and advances. That boundary mirrors
    ``TrainingExecutor``: the control plane consumes results without caring
    how they were produced, and this graph collects hypotheses/evidence
    without caring how they were generated.
    """

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
        """Register a new hypothesis, with the concrete change it proposes to try.

        ``Hypothesis.intervention`` (from ``models.py``) is free text -- it
        describes an idea, not an executable change. ``config_patch`` is
        the actual patch that idea resolves to, and is what "have we tried
        this before" has to compare against. ``estimated_gpu_hours`` is the
        generator's own pre-run cost estimate for this candidate -- used by
        ``ranking.py`` to break ties between equally-corroborated trials
        (cheaper first), not a measured cost, which only exists once a
        trial has actually run. When ``registry`` is supplied and this
        exact patch already failed against this incident's signature
        class, this raises rather than silently accepting a repeat -- the
        structural guard behind "did it avoid repeating failed
        interventions," enforced here rather than left to callers to
        remember.
        """
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
        trial.remediation = remediation
        self.gpu_hours_spent += remediation.gpu_hours_spent
        self.status = InvestigationStatus.RESOLVED

    def record_failed_trial(self, trial: HypothesisTrial, remediation: RemediationRecord) -> None:
        trial.remediation = remediation
        self.gpu_hours_spent += remediation.gpu_hours_spent
        if self.remaining_budget() <= 0:
            self.status = InvestigationStatus.ABANDONED


def route_failure(
    capture: FailureCapture,
    fingerprint: IncidentFingerprint,
    registry: RemediationRegistry,
    *,
    gpu_hour_budget: float,
    investigation_id: str,
) -> RemediationRecord | Investigation:
    """The "known remediation?" fork.

    Exact-fingerprint match against a previously *resolved* remediation
    routes to a bounded, low-risk retry of that exact fix. Anything else --
    a genuinely new incident, or one that merely resembles a past class --
    opens a real investigation instead of guessing.
    """
    known = registry.lookup(fingerprint)
    if known is not None:
        return known
    return Investigation(
        investigation_id=investigation_id,
        fingerprint=fingerprint,
        capture=capture,
        gpu_hour_budget=gpu_hour_budget,
    )
