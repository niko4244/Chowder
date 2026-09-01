from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from .incident import FailureCapture, IncidentFingerprint
from .investigation import DiagnosticProbeResult, RemediationOutcome, RemediationRegistry


@dataclass(frozen=True)
class ProbeContext:
    """Everything a diagnostic probe needs, bundled once.

    Distinct from ``executors.py``'s ``ExecutionContext``, which is the
    training-config context passed to a ``TrainingExecutor`` -- conflating
    the two would blur what each Protocol boundary actually means. This is
    specifically the incident-investigation context.
    """

    capture: FailureCapture
    fingerprint: IncidentFingerprint
    registry: RemediationRegistry


@runtime_checkable
class DiagnosticProbe(Protocol):
    """A probe observes; it must never change execution state.

    Matches the ``TrainingExecutor``/``EvaluationExecutor`` Protocol
    pattern in ``executors.py`` -- callers depend only on this contract,
    never on a specific probe's internals.
    """

    name: str

    def run(self, context: ProbeContext) -> DiagnosticProbeResult:
        ...


# Hardware compute-capability floor. Frozen alongside this module's other
# rule tables before docs/HIDDEN_SET_FREEZE.md hashes it (see
# EXECUTOR_INVESTIGATOR_PLAN.md Task 7) -- do not edit casually once that
# freeze exists; a change here after the freeze should fail
# ``test_freeze_intact`` loudly, not drift silently.
_MINIMUM_SUPPORTED_SM = 70  # PyTorch's floor as observed in this project's
# own incidents: an sm_60 Tesla P100 was rejected outright by the installed
# PyTorch build (sm_70-sm_120 supported), the root cause of a real incident.

_SM_PATTERN = re.compile(r"sm_(\d+)")


def _extract_sm_capability(hardware_summary: str) -> int | None:
    """Best-effort extraction of an ``sm_NN`` compute-capability marker."""
    match = _SM_PATTERN.search(hardware_summary)
    return int(match.group(1)) if match else None


@dataclass(frozen=True)
class InstalledPackageProbe:
    name: str = "installed_package_probe"

    def run(self, context: ProbeContext) -> DiagnosticProbeResult:
        packages = context.capture.environment.installed_packages
        return DiagnosticProbeResult(
            probe_id=self.name,
            description="installed package versions captured with the failure",
            observation={"installed_packages": dict(packages)},
        )


@dataclass(frozen=True)
class HardwareCompatibilityProbe:
    name: str = "hardware_compatibility_probe"

    def run(self, context: ProbeContext) -> DiagnosticProbeResult:
        summary = context.capture.environment.hardware_summary
        sm = _extract_sm_capability(summary)
        compatible = sm is None or sm >= _MINIMUM_SUPPORTED_SM
        return DiagnosticProbeResult(
            probe_id=self.name,
            description=(
                "hardware compute-capability vs. this project's known PyTorch floor"
            ),
            observation={
                "hardware_summary": summary,
                "detected_sm": sm,
                "minimum_supported_sm": _MINIMUM_SUPPORTED_SM,
                "compatible": compatible,
            },
        )


@dataclass(frozen=True)
class KnownWorkingConfigProbe:
    """Surfaces prior resolved fixes for this incident's signature class as
    *evidence*, never as an auto-apply -- ``RemediationRegistry.lookup``
    already refuses to auto-apply cross-incident fixes for exactly the
    reason two of today's real incidents (conv1d "no engine" vs. the
    gated-delta-rule cuBLAS failure) shared a signature-adjacent family but
    needed different fixes. This probe makes that same history usable by an
    investigation without weakening that guard.
    """

    name: str = "known_working_config_probe"

    def run(self, context: ProbeContext) -> DiagnosticProbeResult:
        candidates = context.registry.class_history(context.fingerprint.signature_kind)
        resolved = [r for r in candidates if r.outcome is RemediationOutcome.RESOLVED]
        return DiagnosticProbeResult(
            probe_id=self.name,
            description=(
                "prior resolved remediations for this incident's signature class -- "
                "evidence only, never an auto-fix"
            ),
            observation={
                "resolved_remediation_ids": tuple(r.remediation_id for r in resolved),
                "resolved_config_patches": tuple(dict(r.config_patch) for r in resolved),
            },
        )


@dataclass(frozen=True)
class ArtifactIntegrityProbe:
    """Checks a partial/downloaded artifact's actual hash against an
    expected one.

    Covers the "corrupted download" incident class from the original
    Executor Chaos Benchmark proposal: distinguishing a genuinely missing
    artifact (``ARTIFACT_NOT_FOUND``) from one that exists but doesn't
    match what was expected -- a distinct failure needing a different
    remediation (redownload vs. re-check-the-source), which is exactly the
    kind of near-miss-bucket conflation this project's own fingerprinting
    discipline exists to avoid.
    """

    expected_sha256: str
    name: str = "artifact_integrity_probe"

    def run(self, context: ProbeContext) -> DiagnosticProbeResult:
        ref = context.capture.partial_artifact_ref
        if ref is None:
            return DiagnosticProbeResult(
                probe_id=self.name,
                description="no partial artifact reference to check",
                observation={
                    "checked": False,
                    "reason": "no partial_artifact_ref on this capture",
                },
            )
        path = Path(ref)
        if not path.is_file():
            return DiagnosticProbeResult(
                probe_id=self.name,
                description="artifact reference does not exist on disk",
                observation={
                    "checked": False,
                    "reason": "artifact_ref not found",
                    "path": ref,
                },
            )
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        return DiagnosticProbeResult(
            probe_id=self.name,
            description="artifact hash comparison",
            observation={
                "checked": True,
                "expected_sha256": self.expected_sha256,
                "actual_sha256": actual,
                "matches": actual == self.expected_sha256,
            },
        )
