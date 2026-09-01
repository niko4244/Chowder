from __future__ import annotations

import hashlib
import json
import re
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class SignatureKind(str, Enum):
    """Coarse, rule-based classification of an incident's failure class.

    This is deliberately a cheap pattern match, not a diagnosis. Two
    incidents sharing a signature_kind are candidates for the same
    remediation family -- confirming that is the investigation layer's job,
    not this one's.
    """

    CUDA_OOM = "cuda_oom"
    CUDA_KERNEL_UNAVAILABLE = "cuda_kernel_unavailable"
    CUDA_EXECUTION_FAILED = "cuda_execution_failed"
    CUDA_DEVICE_MISMATCH = "cuda_device_mismatch"
    HARDWARE_INCOMPATIBLE = "hardware_incompatible"
    NETWORK_TRANSIENT = "network_transient"
    DEPENDENCY_INCOMPATIBLE = "dependency_incompatible"
    CONFIG_INVALID = "config_invalid"
    ARTIFACT_NOT_FOUND = "artifact_not_found"
    ARTIFACT_CORRUPTED = "artifact_corrupted"
    UNKNOWN = "unknown"


# Ordered: first matching rule wins. Order matters where a message could
# match more than one pattern (e.g. a device-mismatch error is also a CUDA
# RuntimeError, but the more specific rule must win).
_SIGNATURE_RULES: tuple[tuple[SignatureKind, tuple[str, ...]], ...] = (
    (SignatureKind.CUDA_OOM, ("outofmemoryerror", "cuda out of memory")),
    (SignatureKind.CUDA_DEVICE_MISMATCH, ("expected all tensors to be on the same device",)),
    (
        SignatureKind.CUDA_KERNEL_UNAVAILABLE,
        ("unable to find an engine to execute", "no kernel image is available"),
    ),
    (
        SignatureKind.CUDA_EXECUTION_FAILED,
        ("cublas_status_execution_failed", "cudnn_status_execution_failed"),
    ),
    (
        SignatureKind.HARDWARE_INCOMPATIBLE,
        ("is not compatible with the current pytorch installation", "cuda capability"),
    ),
    (
        SignatureKind.NETWORK_TRANSIENT,
        (
            "remoteprotocolerror",
            "connectionerror",
            "readtimeout",
            "chunkedencodingerror",
            "peer closed connection",
            "service unavailable",
            "too many requests",
        ),
    ),
    (
        SignatureKind.DEPENDENCY_INCOMPATIBLE,
        (
            "modulenotfounderror",
            "importerror",
            "attributeerror",
            "does not recognize this architecture",
        ),
    ),
    (SignatureKind.CONFIG_INVALID, ("unrecognized arguments", "argumenterror")),
    (SignatureKind.ARTIFACT_NOT_FOUND, ("filenotfounderror", "no such file or directory")),
    (
        SignatureKind.ARTIFACT_CORRUPTED,
        ("checksum mismatch", "sha256 mismatch", "hash mismatch"),
    ),
)


def classify_signature(exception_type: str, exception_message: str, traceback_text: str) -> SignatureKind:
    """Deterministic, rule-based coarse classification.

    Intentionally not machine-learned and not LLM-driven: this bucket needs
    to be cheap, auditable, and stable across runs, so "have we seen this
    class of failure before" is a reliable lookup key rather than a
    probabilistic guess. Refining or disputing this classification is the
    investigation layer's job, built on top of this record -- not a reason
    to make this layer smarter.
    """
    haystack = f"{exception_type}\n{exception_message}\n{traceback_text}".casefold()
    for kind, needles in _SIGNATURE_RULES:
        if any(needle in haystack for needle in needles):
            return kind
    return SignatureKind.UNKNOWN


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """The executor's environment at the moment of failure.

    Captured immediately, not re-queried later -- a retry may install
    different package versions or land on different hardware, so this is
    the only trustworthy record of what was actually true when the
    incident happened.
    """

    hardware_summary: str
    accelerator_count: int
    installed_packages: Mapping[str, str] = field(default_factory=dict)
    config_patch: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FailureCapture:
    """Raw evidence captured the instant an executor call raises.

    This is evidence, not diagnosis. Every field must be derivable
    mechanically from the exception and the execution context, so capture
    can happen even when nothing downstream understands the failure yet.
    """

    incident_id: str
    experiment_id: str
    executor_name: str
    occurred_at: str
    exception_type: str
    exception_message: str
    traceback_text: str
    environment: EnvironmentSnapshot
    attempt_number: int = 1
    gpu_hours_spent: float = 0.0
    run_id: str | None = None
    partial_artifact_ref: str | None = None


@dataclass(frozen=True)
class IncidentFingerprint:
    """A stable identity for "this kind of failure".

    ``fingerprint_sha256`` is exact-match identity: same signature kind,
    same executor, same hardware, same *normalized* exception message. Two
    captures sharing a fingerprint are the same incident recurring, not
    merely similar. ``signature_kind`` alone is the coarser bucket used to
    look up whether any remediation exists for this failure *class*, even
    from a different exact incident.
    """

    fingerprint_sha256: str
    signature_kind: SignatureKind
    signature_components: Mapping[str, str]


def _normalize_exception_message(message: str) -> str:
    """Strip volatile numeric/pointer detail so near-identical failures match.

    Real tracebacks carry run-specific numbers (byte counts, addresses, line
    numbers) that must not fracture the fingerprint for what is otherwise
    the same underlying failure -- two CUDA OOMs asking to allocate
    different byte counts are still "the same" incident class.
    """
    normalized = re.sub(r"0x[0-9a-fA-F]+", "<hex>", message)
    normalized = re.sub(r"\d+(\.\d+)?", "<num>", normalized)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def capture_from_exception(
    exc: BaseException,
    *,
    incident_id: str,
    experiment_id: str,
    executor_name: str,
    occurred_at: str,
    environment: EnvironmentSnapshot,
    attempt_number: int = 1,
    gpu_hours_spent: float = 0.0,
    run_id: str | None = None,
    partial_artifact_ref: str | None = None,
) -> FailureCapture:
    """Build a ``FailureCapture`` from a live Python exception.

    The one place this project constructs evidence directly from a raised
    exception rather than transcribing historical incident text by hand
    (as the dev/hidden fixtures do). Used both for a training executor's
    first failure and for a remediation attempt that raises a genuinely
    new problem mid-investigation -- the same construction either way, so
    a remediation-attempt crash is captured with the same fidelity as an
    original one, not a lesser summary of it.
    """
    return FailureCapture(
        incident_id=incident_id,
        experiment_id=experiment_id,
        executor_name=executor_name,
        occurred_at=occurred_at,
        exception_type=type(exc).__qualname__,
        exception_message=str(exc),
        traceback_text="".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
        environment=environment,
        attempt_number=attempt_number,
        gpu_hours_spent=gpu_hours_spent,
        run_id=run_id,
        partial_artifact_ref=partial_artifact_ref,
    )


def compute_fingerprint(capture: FailureCapture) -> IncidentFingerprint:
    signature_kind = classify_signature(
        capture.exception_type, capture.exception_message, capture.traceback_text
    )
    components = {
        "signature_kind": signature_kind.value,
        "executor_name": capture.executor_name,
        "exception_type": capture.exception_type,
        "hardware_summary": capture.environment.hardware_summary,
        "normalized_message": _normalize_exception_message(capture.exception_message),
    }
    return IncidentFingerprint(
        fingerprint_sha256=_canonical_digest(components),
        signature_kind=signature_kind,
        signature_components=components,
    )
