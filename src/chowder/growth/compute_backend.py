"""First-class compute backends: a declared attempt, verified end to end.

The growth loop executes a recipe through a ``TrainingFn``. On a laptop that
is a subprocess; a remote accelerator needs the same scientific guarantees the
local path has -- the code identity, the inputs, the outputs, the cost and the
failure all provable after the fact -- or the run has produced no evidence.
This module owns the contract every execution surface shares:

* :class:`SourceBinding` names the repository, the exact commit, the chowder
  version and the campaign/recipe/attempt identity a job ran as. A backend
  refuses a returned job whose commit does not equal the declared one.
* :class:`DeclaredInput` / :func:`bind_declared_inputs` hash the inputs before
  dispatch, and :func:`verify_declared_inputs` re-verifies them before push:
  an attempt reads exactly what the campaign declared, and a file whose bytes
  moved refuses instead of shipping. This is the declared-input contract.
* :func:`verify_artifact_manifest` refuses a returned artifact that is
  missing, unlisted, or hash-mismatched before any consumer can reference it.
* :class:`AttemptRequest` carries the frozen cost envelope; the returned
  :class:`AttemptOutcome` carries the measured cost and the settlement
  verdict, so remote spend settles through the same
  :func:`chowder.growth.compute_cost.settle_cost` as local spend.
* :meth:`AttemptOutcome.to_evidence` renders the vocabulary ``cycle``,
  ``attempt_failure.classify_failure`` and ``compute_cost.settlement_refusal``
  already read, so classification and advancement need no remote-specific
  branch.

A concrete backend implements :class:`ComputeBackend` (the Kaggle
implementation lives in :mod:`chowder.growth.kaggle_compute`). Nothing here
touches a network, a GPU, or a third-party CLI.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .compute_cost import ComputeCost, SettlementVerdict

__all__ = [
    "ComputeBackend",
    "ComputeBackendRefusal",
    "SourceBinding",
    "DeclaredInput",
    "ArtifactEntry",
    "AttemptRequest",
    "AttemptOutcome",
    "bind_declared_inputs",
    "verify_declared_inputs",
    "verify_artifact_manifest",
    "file_digest",
    "OUTCOME_SUCCEEDED",
    "OUTCOME_REFUSED",
    "OUTCOME_FAILED",
]

#: Machine-readable refusal identifiers, first token of the exception message
#: and of ``refused_by`` in rendered evidence. A consumer branches on these,
#: never on prose.
SOURCE_BINDING_SCHEMA = "SOURCE_BINDING"
DECLARED_INPUT_UNREADABLE = "DECLARED_INPUT_UNREADABLE"
DECLARED_INPUT_DIGEST_MISMATCH = "DECLARED_INPUT_DIGEST_MISMATCH"
ARTIFACT_MANIFEST_INVALID = "ARTIFACT_MANIFEST_INVALID"
ARTIFACT_MISSING = "ARTIFACT_MISSING"
ARTIFACT_EXTRA = "ARTIFACT_EXTRA"
ARTIFACT_DIGEST_MISMATCH = "ARTIFACT_DIGEST_MISMATCH"
ARTIFACT_ADMISSION_REFUSED = "ARTIFACT_ADMISSION_REFUSED"
PAYLOAD_NOT_SERIALIZABLE = "PAYLOAD_NOT_SERIALIZABLE"
REQUEST_CONTRACT = "REQUEST_CONTRACT"

#: Outcome statuses, spelled exactly as ``growth.training_binding`` spells the
#: local executor's statuses so a remote attempt rides the same selection and
#: advancement rules with no translation layer.
OUTCOME_SUCCEEDED = "SUCCEEDED"
OUTCOME_REFUSED = "REFUSED"
OUTCOME_FAILED = "FAILED"
OUTCOME_STATUSES = frozenset({OUTCOME_SUCCEEDED, OUTCOME_REFUSED, OUTCOME_FAILED})

_SHA256_HEX = frozenset("0123456789abcdef")


class ComputeBackendRefusal(RuntimeError):
    """A backend contract is broken. The code comes first, the prose after."""

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(f"{code}: {reason}")
        self.code = code
        self.reason = reason


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ComputeBackendRefusal(
            REQUEST_CONTRACT, f"{label} must be a non-empty string, got {value!r}"
        )
    return value


def _non_negative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComputeBackendRefusal(
            REQUEST_CONTRACT, f"{label} must be a number, got {value!r}"
        )
    import math

    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ComputeBackendRefusal(
            REQUEST_CONTRACT, f"{label} must be finite and non-negative, got {value!r}"
        )
    return value


def _sha256_text(value: Any, label: str) -> str:
    text = _required_text(value, label).lower()
    if len(text) != 64 or any(character not in _SHA256_HEX for character in text):
        raise ComputeBackendRefusal(
            REQUEST_CONTRACT, f"{label} must be a 64-character hex sha256, got {value!r}"
        )
    return text


@dataclass(frozen=True)
class SourceBinding:
    """What code a job ran as: repo, exact commit, version, and attempt ids.

    This is the identity a remote result is bound to. A kernel that ran
    different code than the campaign declared has produced no evidence, and
    the backend refuses it after the fact by comparing this binding's commit
    with the commit the kernel echoed.
    """

    repository: str
    commit_sha: str
    chowder_version: str
    cycle_id: str
    recipe_id: str
    attempt_id: str

    def __post_init__(self) -> None:
        _required_text(self.repository, "repository")
        if not isinstance(self.commit_sha, str):
            raise ComputeBackendRefusal(
                SOURCE_BINDING_SCHEMA,
                f"commit_sha must be a 40-character hex string, got {self.commit_sha!r}",
            )
        commit = self.commit_sha.strip().lower()
        if len(commit) != 40 or any(character not in _SHA256_HEX for character in commit):
            raise ComputeBackendRefusal(
                SOURCE_BINDING_SCHEMA,
                f"commit_sha must be a full 40-character commit sha, got {self.commit_sha!r}",
            )
        object.__setattr__(self, "commit_sha", commit)
        for label in ("chowder_version", "cycle_id", "recipe_id", "attempt_id"):
            _required_text(getattr(self, label), label)

    def to_dict(self) -> dict[str, str]:
        return {
            "repository": self.repository,
            "commit_sha": self.commit_sha,
            "chowder_version": self.chowder_version,
            "cycle_id": self.cycle_id,
            "recipe_id": self.recipe_id,
            "attempt_id": self.attempt_id,
        }


@dataclass(frozen=True)
class DeclaredInput:
    """One declared input: a name, where it lives, and what its bytes are."""

    name: str
    path: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        _required_text(self.name, "name")
        _required_text(self.path, "path")
        _sha256_text(self.sha256, "sha256")
        _non_negative(self.bytes, "bytes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
        }


@dataclass(frozen=True)
class ArtifactEntry:
    """One returned artifact, relative to the attempt's output root."""

    path: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        _required_text(self.path, "path")
        parts = self.path.replace("\\", "/").split("/")
        if self.path.startswith(("/", "\\")) or ".." in parts:
            raise ComputeBackendRefusal(
                ARTIFACT_MANIFEST_INVALID,
                f"artifact path {self.path!r} must be relative to the output root",
            )
        _sha256_text(self.sha256, "sha256")
        _non_negative(self.bytes, "bytes")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


def file_digest(path: str | Path) -> tuple[str, int]:
    """The sha256 and byte count of a file, or a refusal it cannot be read."""
    digest = hashlib.sha256()
    size = 0
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise ComputeBackendRefusal(
            DECLARED_INPUT_UNREADABLE, f"cannot read {path}: {exc}"
        ) from exc
    return digest.hexdigest(), size


def bind_declared_inputs(
    inputs: Mapping[str, str | Path],
) -> tuple[DeclaredInput, ...]:
    """Hash a campaign's declared inputs into immutable entries.

    Names are sorted so the bound set is deterministic regardless of the
    mapping's insertion order. A path that does not resolve to a readable
    file refuses here, before any dispatch, rather than shipping an input the
    kernel could not verify.
    """
    if not isinstance(inputs, Mapping):
        raise ComputeBackendRefusal(
            REQUEST_CONTRACT, "declared inputs must be a mapping of name to path"
        )
    bound: list[DeclaredInput] = []
    for name, path in sorted(inputs.items(), key=lambda item: str(item[0])):
        location = Path(path)
        if not location.is_file():
            raise ComputeBackendRefusal(
                DECLARED_INPUT_UNREADABLE,
                f"declared input {name!r} at {location} is not a readable file",
            )
        sha256, size = file_digest(location)
        bound.append(
            DeclaredInput(name=str(name), path=str(location), sha256=sha256, bytes=size)
        )
    return tuple(bound)


def verify_declared_inputs(inputs: Sequence[DeclaredInput]) -> None:
    """Re-verify every declared input against its bound digest, before push."""
    for entry in inputs:
        if not isinstance(entry, DeclaredInput):
            raise ComputeBackendRefusal(
                REQUEST_CONTRACT,
                f"declared inputs must be DeclaredInput entries, got {entry!r}",
            )
        location = Path(entry.path)
        if not location.is_file():
            raise ComputeBackendRefusal(
                DECLARED_INPUT_DIGEST_MISMATCH,
                f"declared input {entry.name!r} at {location} no longer exists",
            )
        sha256, size = file_digest(location)
        if sha256 != entry.sha256 or size != entry.bytes:
            raise ComputeBackendRefusal(
                DECLARED_INPUT_DIGEST_MISMATCH,
                f"declared input {entry.name!r} at {location} moved: now "
                f"{sha256} ({size} bytes), declared {entry.sha256} "
                f"({entry.bytes} bytes)",
            )


def verify_artifact_manifest(
    manifest: Sequence[ArtifactEntry], root: str | Path
) -> tuple[Path, ...]:
    """Prove every returned artifact against the kernel's own manifest.

    The output root must contain exactly the files the manifest lists: a
    missing file, a file the manifest does not name, a duplicate manifest
    entry, or a hash/size mismatch all refuse. Only a manifest that verifies
    may be referenced by an attempt, so a downloaded artifact is not a
    second-class citizen with a second-class gate.
    """
    output_root = Path(root)
    if not output_root.is_dir():
        raise ComputeBackendRefusal(
            ARTIFACT_MANIFEST_INVALID, f"artifact root {output_root} is not a directory"
        )
    expected: dict[str, ArtifactEntry] = {}
    for entry in manifest:
        if not isinstance(entry, ArtifactEntry):
            raise ComputeBackendRefusal(
                ARTIFACT_MANIFEST_INVALID,
                f"artifact manifest entries must be ArtifactEntry, got {entry!r}",
            )
        if entry.path in expected:
            raise ComputeBackendRefusal(
                ARTIFACT_MANIFEST_INVALID,
                f"artifact manifest names {entry.path!r} twice",
            )
        expected[entry.path] = entry
    actual: dict[str, Path] = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file():
            actual[path.relative_to(output_root).as_posix()] = path
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise ComputeBackendRefusal(
            ARTIFACT_MISSING,
            f"the artifact manifest names files the job did not return: {missing}",
        )
    extra = sorted(set(actual) - set(expected))
    if extra:
        raise ComputeBackendRefusal(
            ARTIFACT_EXTRA,
            f"the job returned files its manifest does not name: {extra}; an "
            "undocumented artifact is not verifiable evidence",
        )
    verified: list[Path] = []
    for relative, entry in expected.items():
        path = actual[relative]
        sha256, size = file_digest(path)
        if sha256 != entry.sha256 or size != entry.bytes:
            raise ComputeBackendRefusal(
                ARTIFACT_DIGEST_MISMATCH,
                f"artifact {relative!r} hashes {sha256} ({size} bytes), but the "
                f"kernel's manifest declared {entry.sha256} ({entry.bytes} bytes)",
            )
        verified.append(path)
    return tuple(verified)


@dataclass(frozen=True)
class AttemptRequest:
    """Everything a backend needs to run one declared attempt."""

    source: SourceBinding
    entry_point: str
    inputs: tuple[DeclaredInput, ...] = ()
    payload: Mapping[str, Any] = field(default_factory=dict)
    projected_cost: ComputeCost = field(
        default_factory=lambda: ComputeCost.zero(source="unstated")
    )
    device_ceiling: float | None = None
    wall_ceiling: float | None = None
    project_budget_wall_gpu_hours: float | None = None
    projection_tolerance: float = 0.25
    mounts: tuple[str, ...] = ()
    resume_from: str | None = None
    timeout_seconds: float = 3600.0

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceBinding):
            raise ComputeBackendRefusal(
                REQUEST_CONTRACT,
                f"source must be a SourceBinding, got {self.source!r}",
            )
        _required_text(self.entry_point, "entry_point")
        if not isinstance(self.projected_cost, ComputeCost):
            raise ComputeBackendRefusal(
                REQUEST_CONTRACT,
                f"projected_cost must be a ComputeCost, got {self.projected_cost!r}",
            )
        if not isinstance(self.payload, Mapping):
            raise ComputeBackendRefusal(
                REQUEST_CONTRACT, "payload must be a mapping"
            )
        try:
            json.dumps(dict(self.payload), sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ComputeBackendRefusal(
                PAYLOAD_NOT_SERIALIZABLE, f"payload cannot be serialized: {exc}"
            ) from exc
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "mounts", tuple(self.mounts))
        for label in (
            "device_ceiling",
            "wall_ceiling",
            "project_budget_wall_gpu_hours",
        ):
            value = getattr(self, label)
            if value is not None:
                _non_negative(value, label)
        _non_negative(self.projection_tolerance, "projection_tolerance")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or self.timeout_seconds <= 0.0
        ):
            raise ComputeBackendRefusal(
                REQUEST_CONTRACT,
                f"timeout_seconds must be positive, got {self.timeout_seconds!r}",
            )
        for mount in self.mounts:
            _required_text(mount, "mount")
        if self.resume_from is not None:
            _required_text(self.resume_from, "resume_from")


@dataclass(frozen=True)
class AttemptOutcome:
    """What a backend observed, in the vocabulary the growth loop already reads."""

    source: SourceBinding
    status: str
    artifacts: tuple[ArtifactEntry, ...] = ()
    cost: ComputeCost | None = None
    settlement: SettlementVerdict | None = None
    failure_class: str | None = None
    refusal_code: str | None = None
    refusal_reason: str = ""
    environment: Mapping[str, Any] = field(default_factory=dict)
    resume_state: str = "not-applicable"
    mounts: tuple[str, ...] = ()
    job_id: str = ""
    log: str = ""
    #: The source commit the executor's own record echoed, lowercased, when a
    #: record identified the running code. Empty means no record did -- a
    #: transport failure, a refusal before dispatch, or a job that could not
    #: identify its source at all. The backend compares it to the declared
    #: commit; a consumer reads it to state whether the attempt verified its
    #: source, rather than inferring verification from the outcome's status.
    source_commit_sha: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceBinding):
            raise ComputeBackendRefusal(
                REQUEST_CONTRACT, "outcome source must be a SourceBinding"
            )
        if self.status not in OUTCOME_STATUSES:
            raise ComputeBackendRefusal(
                REQUEST_CONTRACT,
                f"outcome status must be one of {sorted(OUTCOME_STATUSES)}, got "
                f"{self.status!r}",
            )
        if not isinstance(self.source_commit_sha, str):
            raise ComputeBackendRefusal(
                REQUEST_CONTRACT,
                f"source_commit_sha must be a string, got {self.source_commit_sha!r}",
            )
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "mounts", tuple(self.mounts))
        object.__setattr__(self, "environment", dict(self.environment))

    @property
    def succeeded(self) -> bool:
        return self.status == OUTCOME_SUCCEEDED

    def to_evidence(self) -> dict[str, Any]:
        """The attempt's evidence mapping, in the growth loop's vocabulary.

        A settlement refusal is rendered exactly as ``settlement_refusal``
        reads it (``budget_settlement`` plus the ``refused_by`` marker), so an
        over-budget remote attempt is classified BUDGET_EXHAUSTED by the same
        code path as a local one, never re-queued silently.
        """
        evidence: dict[str, Any] = {
            "status": self.status,
            "candidate_succeeded": True if self.succeeded else None,
            "refused_by": None,
            "refusal_reason": None,
            "failure_class": self.failure_class,
            "resume_state": self.resume_state,
            "artifact_ref": self.artifacts[0].path if self.artifacts else None,
            "artifact_sha256": self.artifacts[0].sha256 if self.artifacts else None,
            "artifact_manifest": [entry.to_dict() for entry in self.artifacts],
            "job_id": self.job_id,
            "environment": dict(self.environment),
            "source_commit_sha": self.source_commit_sha.strip().lower() or None,
        }
        if self.cost is not None:
            evidence["measured_gpu_hours"] = self.cost.wall_gpu_hours
            evidence["compute_cost"] = self.cost.to_dict()
        if self.settlement is not None:
            evidence["budget_settlement"] = self.settlement.to_dict()
            if not self.settlement.compliant:
                reasons = self.settlement.failure_reasons
                evidence["refused_by"] = "budget_settlement"
                evidence["refusal_reason"] = reasons[0] if reasons else "budget_settlement"
        if not self.succeeded and evidence["refused_by"] is None:
            evidence["refused_by"] = self.refusal_code or "compute_backend"
            evidence["refusal_reason"] = (
                self.refusal_reason or evidence["refused_by"]
            )
        return evidence


@runtime_checkable
class ComputeBackend(Protocol):
    """The execution seam a campaign dispatches an attempt through."""

    name: str

    def preflight(self, request: AttemptRequest) -> tuple[str, str] | None:
        """``None`` when the attempt may dispatch, else ``(code, reason)``."""

    def dispatch(
        self, request: AttemptRequest, *, destination: str | Path
    ) -> AttemptOutcome:
        """Run the attempt, verify its output, and settle its cost."""
