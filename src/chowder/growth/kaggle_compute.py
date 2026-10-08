"""Kaggle compute backend: the production bar, enforced in code.

``docs/KAGGLE_BACKEND_REQUIREMENTS.md`` states what a 0.5 campaign must clear
before it may spend Kaggle quota on candidate training. This module is that
bar, with the Kaggle API behind an injectable :class:`KaggleTransport` so
every check is exercised offline -- no token, no network, no GPU:

R1 source binding
    the :class:`KaggleJobSpec` carries the repository, commit, chowder
    version and campaign/recipe/attempt identity; after the run the backend
    refuses a record whose echoed commit differs from the declared one. A
    kernel that ran different code than the campaign declared has produced
    no evidence. A failed record with no echo at all is not a mismatch: the
    install never completed, and the failure it reports is classified below.
R2 declared inputs and artifact manifest
    every declared input is re-hashed before push and must be echoed by the
    kernel with the same digest; every returned artifact must verify against
    the kernel's own manifest -- missing, unlisted and hash-mismatched files
    all refuse -- and then passes the injected artifact-admission check, so a
    downloaded artifact goes through the same gate as a local one.
R3 settlement through ``compute_cost``
    the quota reading before and after the job, the declared accelerator's
    device multiplier and the job's wall seconds become a measured
    :class:`~chowder.growth.compute_cost.ComputeCost`, settled with
    :func:`~chowder.growth.compute_cost.settle_cost` against the same
    projection and ceilings local spend settles against. A settlement refusal
    is classified BUDGET_EXHAUSTED, never silently re-queued.
R4 resume vocabulary
    a declared resume must come back with ``resume_from`` naming the bound
    checkpoint and ``resume_state`` in the search's vocabulary
    (``resumed`` / ``not-a-resume``). A loader that failed is reported as
    ``not-a-resume`` and the attempt fails; a job that claims a resume the
    request never declared is refused.
R5 classified failures
    a job that did not complete is classified through
    ``attempt_failure.classify_failure`` before the attempt is recorded;
    "failed" is a class, not a log line. Its spend still settles, and the
    completed-job evidence checks (input verification, artifact manifest,
    environment) do not apply to it -- a failed record has no evidence to
    verify, so an install failure surfaces as the install error itself
    instead of as a derived refusal.
R6 recorded environment
    the kernel's own environment record (python version, resolved packages,
    base-model commit) must ship in the job record and travels into the
    attempt's evidence; a completed record without one refuses, while a
    failed record is classified from its own error.
R7 one kernel, one attempt
    each dispatch owns a fresh destination directory, the job id may never
    be returned twice, and the mounts the job declared are recorded and
    compared -- no shared state, no cross-attempt reads.
R8 quota is a ceiling
    the weekly balance is read before every push and a dispatch that could
    exceed it (or the campaign's declared Kaggle budget) refuses before
    anything runs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from .attempt_failure import FailureClass, classify_failure
from .compute_backend import (
    ARTIFACT_ADMISSION_REFUSED,
    AttemptOutcome,
    AttemptRequest,
    ArtifactEntry,
    ComputeBackendRefusal,
    DeclaredInput,
    OUTCOME_FAILED,
    OUTCOME_REFUSED,
    OUTCOME_SUCCEEDED,
    SourceBinding,
    verify_artifact_manifest,
    verify_declared_inputs,
)
from .compute_cost import ComputeCost, SettlementVerdict, settle_cost

__all__ = [
    "KAGGLE_BACKEND_NAME",
    "BACKEND_JOB_RECORD_NAME",
    "KaggleQuota",
    "KaggleAccelerator",
    "KAGGLE_ACCELERATORS",
    "resolve_kaggle_accelerator",
    "KaggleJobSpec",
    "KaggleJobRecord",
    "KaggleTransport",
    "KaggleTransportError",
    "KaggleComputeBackend",
]


KAGGLE_BACKEND_NAME = "kaggle"
#: The backend's own record, written beside the verified artifacts. It names
#: the spec, the quota readings and the settlement, so a downloaded attempt's
#: cost can be audited without reading the ledger.
BACKEND_JOB_RECORD_NAME = "compute-job.json"

#: Machine-readable refusal identifiers.
QUOTA_INSUFFICIENT = "KAGGLE_QUOTA_INSUFFICIENT"
QUOTA_CEILING_EXCEEDED = "KAGGLE_QUOTA_CEILING_EXCEEDED"
QUOTA_READING_REGRESSED = "KAGGLE_QUOTA_READING_REGRESSED"
DESTINATION_NOT_FRESH = "KAGGLE_DESTINATION_NOT_FRESH"
ACCELERATOR_UNKNOWN = "KAGGLE_ACCELERATOR_UNKNOWN"
ACCELERATOR_MISMATCH = "KAGGLE_ACCELERATOR_MISMATCH"
SOURCE_COMMIT_MISMATCH = "KAGGLE_SOURCE_COMMIT_MISMATCH"
INPUT_VERIFICATION_INCOMPLETE = "KAGGLE_INPUT_VERIFICATION_INCOMPLETE"
ENVIRONMENT_UNRECORDED = "KAGGLE_ENVIRONMENT_UNRECORDED"
RESUME_VOCABULARY_INVALID = "KAGGLE_RESUME_VOCABULARY_INVALID"
RESUME_IDENTITY_MISMATCH = "KAGGLE_RESUME_IDENTITY_MISMATCH"
RESUME_UNDECLARED = "KAGGLE_RESUME_UNDECLARED"
RESUME_NOT_TAKEN = "KAGGLE_RESUME_NOT_TAKEN"
MOUNTS_MISMATCH = "KAGGLE_MOUNTS_MISMATCH"
JOB_ID_REUSED = "KAGGLE_JOB_ID_REUSED"
JOB_ID_MISSING = "KAGGLE_JOB_ID_MISSING"
JOB_FAILED = "KAGGLE_JOB_FAILED"
#: The transport could not produce a kernel record at all (CLI missing, push
#: rejected, poll timeout, pull failure). The backend classifies the attempt
#: as INFRASTRUCTURE with the cause attached -- never an unhandled exception,
#: never a fabricated record.
TRANSPORT_FAILED = "KAGGLE_TRANSPORT_FAILED"

#: The search's resume vocabulary (``candidate_search.run_search`` reads it).
RESUME_STATES = ("resumed", "not-a-resume")

#: What the kernel must record about its own environment (R6). Extra fields
#: are welcome; these are the ones the evidence contract cannot do without.
REQUIRED_ENVIRONMENT_FIELDS = ("python_version", "packages", "model_commit")


class KaggleTransportError(RuntimeError):
    """The transport could not produce a kernel record.

    Raised, not returned: operational failures have no record to classify
    from, so the backend catches this type and records the attempt as an
    INFRASTRUCTURE failure with the cause attached.
    """


@dataclass(frozen=True)
class KaggleQuota:
    """One reading of the weekly accelerator balance."""

    remaining_gpu_hours: float
    raw: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.remaining_gpu_hours, bool)
            or not isinstance(self.remaining_gpu_hours, (int, float))
            or not math.isfinite(float(self.remaining_gpu_hours))
            or float(self.remaining_gpu_hours) < 0.0
        ):
            raise ComputeBackendRefusal(
                QUOTA_INSUFFICIENT,
                f"remaining_gpu_hours must be finite and non-negative, got "
                f"{self.remaining_gpu_hours!r}",
            )


@dataclass(frozen=True)
class KaggleAccelerator:
    """A declared Kaggle shape and the device multiplier it implies.

    The quota balance is charged in session hours; one session hour on a
    two-device shape is two device-GPU-hours. The multiplier is declared
    here, once, rather than guessed at settlement time.
    """

    name: str
    device_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ComputeBackendRefusal(
                ACCELERATOR_UNKNOWN, f"accelerator name must be non-empty, got {self.name!r}"
            )
        if (
            isinstance(self.device_count, bool)
            or not isinstance(self.device_count, int)
            or self.device_count < 1
        ):
            raise ComputeBackendRefusal(
                ACCELERATOR_UNKNOWN,
                f"accelerator {self.name!r} must have at least one device",
            )

    @property
    def device_gpu_hours_multiplier(self) -> float:
        return float(self.device_count)


KAGGLE_ACCELERATORS: Mapping[str, KaggleAccelerator] = {
    "T4": KaggleAccelerator("T4", 1),
    "T4x2": KaggleAccelerator("T4x2", 2),
    "P100": KaggleAccelerator("P100", 1),
}


def resolve_kaggle_accelerator(name: str) -> KaggleAccelerator:
    """The declared shape, or a refusal -- an unmeasured shape is not assumed."""
    accelerator = KAGGLE_ACCELERATORS.get(str(name))
    if accelerator is None:
        raise ComputeBackendRefusal(
            ACCELERATOR_UNKNOWN,
            f"declared accelerator {name!r} is not registered; known shapes: "
            f"{sorted(KAGGLE_ACCELERATORS)}. An unregistered shape has no "
            "declared device multiplier, so its spend cannot be converted "
            "honestly",
        )
    return accelerator


@dataclass(frozen=True)
class KaggleJobSpec:
    """One kernel's marching orders: identity, code, inputs, limits."""

    spec_id: str
    source: SourceBinding
    entry_point: str
    payload: Mapping[str, Any]
    inputs: tuple[DeclaredInput, ...]
    mounts: tuple[str, ...]
    resume_from: str | None
    timeout_seconds: float
    accelerator: str

    def __post_init__(self) -> None:
        if not isinstance(self.spec_id, str) or not self.spec_id.strip():
            raise ComputeBackendRefusal(
                JOB_ID_MISSING, f"spec_id must be non-empty, got {self.spec_id!r}"
            )
        if not isinstance(self.source, SourceBinding):
            raise ComputeBackendRefusal(
                JOB_ID_MISSING,
                f"spec source must be a SourceBinding, got {self.source!r}",
            )
        if not isinstance(self.entry_point, str) or not self.entry_point.strip():
            raise ComputeBackendRefusal(
                JOB_ID_MISSING, "entry_point must be a non-empty string"
            )
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "mounts", tuple(self.mounts))
        object.__setattr__(self, "payload", dict(self.payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "source": self.source.to_dict(),
            "entry_point": self.entry_point,
            "payload": dict(self.payload),
            "inputs": [entry.to_dict() for entry in self.inputs],
            "mounts": list(self.mounts),
            "resume_from": self.resume_from,
            "timeout_seconds": self.timeout_seconds,
            "accelerator": self.accelerator,
        }


@dataclass(frozen=True)
class KaggleJobRecord:
    """What the kernel-side record says happened, before the backend trusts it.

    Everything here is the *claim*. The backend verifies the commit echo, the
    input digests, the mounts, the resume identity, the environment record and
    every artifact hash before any of it becomes evidence.
    """

    job_id: str
    state: str
    wall_seconds: float
    accelerator: str
    source_commit_sha: str
    input_verification: tuple[tuple[str, str], ...] = ()
    artifact_manifest: tuple[ArtifactEntry, ...] = ()
    environment: Mapping[str, Any] = field(default_factory=dict)
    resume_state: str | None = None
    resume_from: str | None = None
    mounts: tuple[str, ...] = ()
    error: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str):
            raise ComputeBackendRefusal(
                JOB_ID_MISSING, f"job_id must be a string, got {self.job_id!r}"
            )
        if not isinstance(self.state, str) or not self.state.strip():
            raise ComputeBackendRefusal(
                JOB_FAILED, f"state must be a non-empty string, got {self.state!r}"
            )
        if not isinstance(self.source_commit_sha, str):
            raise ComputeBackendRefusal(
                SOURCE_COMMIT_MISMATCH,
                f"source_commit_sha must be a string, got {self.source_commit_sha!r}",
            )
        if (
            isinstance(self.wall_seconds, bool)
            or not isinstance(self.wall_seconds, (int, float))
            or not math.isfinite(float(self.wall_seconds))
            or float(self.wall_seconds) < 0.0
        ):
            raise ComputeBackendRefusal(
                JOB_FAILED,
                f"wall_seconds must be finite and non-negative, got {self.wall_seconds!r}",
            )
        object.__setattr__(
            self,
            "input_verification",
            tuple(tuple(pair) for pair in self.input_verification),
        )
        object.__setattr__(self, "artifact_manifest", tuple(self.artifact_manifest))
        object.__setattr__(self, "environment", dict(self.environment))
        object.__setattr__(self, "mounts", tuple(self.mounts))


@runtime_checkable
class KaggleTransport(Protocol):
    """The Kaggle API seam: push/poll/pull and quota, fully injectable."""

    def quota(self) -> KaggleQuota:
        """Read the weekly accelerator balance."""

    def run(self, spec: KaggleJobSpec, destination: Path) -> KaggleJobRecord:
        """Push the kernel, poll it under ``timeout_seconds``, pull its output."""


class KaggleComputeBackend:
    """A :class:`chowder.growth.compute_backend.ComputeBackend` on Kaggle."""

    name = KAGGLE_BACKEND_NAME

    def __init__(
        self,
        transport: KaggleTransport,
        *,
        accelerator: str,
        declared_quota_ceiling_gpu_hours: float | None = None,
        projection_tolerance: float = 0.25,
        artifact_admission: Callable[[ArtifactEntry, Path], None] | None = None,
        require_environment_fields: tuple[str, ...] = REQUIRED_ENVIRONMENT_FIELDS,
    ) -> None:
        self.transport = transport
        self.accelerator = resolve_kaggle_accelerator(accelerator)
        if declared_quota_ceiling_gpu_hours is not None:
            if (
                isinstance(declared_quota_ceiling_gpu_hours, bool)
                or not isinstance(declared_quota_ceiling_gpu_hours, (int, float))
                or not math.isfinite(float(declared_quota_ceiling_gpu_hours))
                or float(declared_quota_ceiling_gpu_hours) < 0.0
            ):
                raise ComputeBackendRefusal(
                    QUOTA_CEILING_EXCEEDED,
                    "declared_quota_ceiling_gpu_hours must be finite and "
                    f"non-negative, got {declared_quota_ceiling_gpu_hours!r}",
                )
        if (
            isinstance(projection_tolerance, bool)
            or not isinstance(projection_tolerance, (int, float))
            or float(projection_tolerance) < 0.0
        ):
            raise ComputeBackendRefusal(
                QUOTA_CEILING_EXCEEDED,
                f"projection_tolerance must be non-negative, got {projection_tolerance!r}",
            )
        self.declared_quota_ceiling_gpu_hours = (
            float(declared_quota_ceiling_gpu_hours)
            if declared_quota_ceiling_gpu_hours is not None
            else None
        )
        self.projection_tolerance = float(projection_tolerance)
        self.artifact_admission = artifact_admission
        self.require_environment_fields = tuple(require_environment_fields)
        self._quota_consumed_gpu_hours = 0.0
        self._dispatched_job_ids: set[str] = set()

    # ------------------------------------------------------------------
    # preflight: everything checkable before a kernel exists
    # ------------------------------------------------------------------

    def preflight(self, request: AttemptRequest) -> tuple[str, str] | None:
        """R2/R4/R8 checks that need no compute, before any push."""
        try:
            verify_declared_inputs(request.inputs)
        except ComputeBackendRefusal as refusal:
            return refusal.code, refusal.reason
        if not request.inputs:
            return (
                INPUT_VERIFICATION_INCOMPLETE,
                "a remote attempt ships the inputs it reads; this request "
                "declares none, so the kernel has nothing to verify and the "
                "result would bind to nothing",
            )
        if request.resume_from is not None and not str(request.resume_from).strip():
            return (
                RESUME_VOCABULARY_INVALID,
                "resume_from must name the bound checkpoint identity",
            )
        required = self._required_quota_hours(request)
        try:
            quota = self.transport.quota()
        except KaggleTransportError as exc:
            return (
                TRANSPORT_FAILED,
                f"the transport could not read the quota balance: {exc}",
            )
        if required > quota.remaining_gpu_hours + 1e-12:
            return (
                QUOTA_INSUFFICIENT,
                f"the projection needs {required:.6f} accelerator-hour(s) of "
                f"quota, but only {quota.remaining_gpu_hours:.6f} remain; "
                "refusing before push rather than discovering it mid-session",
            )
        if self.declared_quota_ceiling_gpu_hours is not None:
            spent = self._quota_consumed_gpu_hours + required
            if spent > self.declared_quota_ceiling_gpu_hours + 1e-12:
                return (
                    QUOTA_CEILING_EXCEEDED,
                    f"this dispatch would bring session quota spend to "
                    f"{spent:.6f}, beyond the declared Kaggle ceiling "
                    f"{self.declared_quota_ceiling_gpu_hours:.6f}",
                )
        return None

    # ------------------------------------------------------------------
    # dispatch: run, verify, settle
    # ------------------------------------------------------------------

    def dispatch(
        self, request: AttemptRequest, *, destination: str | Path
    ) -> AttemptOutcome:
        output_root = Path(destination)
        refusal = self.preflight(request)
        if refusal is not None:
            code, reason = refusal
            return self._outcome(
                request,
                OUTCOME_REFUSED,
                refusal_code=code,
                refusal_reason=reason,
                failure_class=(
                    FailureClass.BUDGET_EXHAUSTED.value
                    if code in {QUOTA_INSUFFICIENT, QUOTA_CEILING_EXCEEDED}
                    else FailureClass.INFRASTRUCTURE.value
                ),
            )
        if output_root.exists() and any(output_root.iterdir()):
            return self._outcome(
                request,
                OUTCOME_REFUSED,
                refusal_code=DESTINATION_NOT_FRESH,
                refusal_reason=(
                    f"destination {output_root} is not fresh; an attempt's "
                    "output is never shared with or read by another attempt"
                ),
                failure_class=FailureClass.INFRASTRUCTURE.value,
            )
        output_root.mkdir(parents=True, exist_ok=True)
        try:
            quota_before = self.transport.quota()
            spec = self._build_spec(request)
            record = self.transport.run(spec, output_root)
            quota_after = self.transport.quota()
        except KaggleTransportError as exc:
            return self._outcome(
                request,
                OUTCOME_FAILED,
                refusal_code=TRANSPORT_FAILED,
                refusal_reason=(
                    f"the transport could not produce a kernel record: {exc}"
                ),
                failure_class=FailureClass.INFRASTRUCTURE.value,
            )

        # R7: a kernel is never reused, and its id is never returned twice.
        if not record.job_id.strip():
            return self._outcome(
                request,
                OUTCOME_REFUSED,
                refusal_code=JOB_ID_MISSING,
                refusal_reason="the job record names no kernel identity",
                failure_class=FailureClass.INFRASTRUCTURE.value,
            )
        if record.job_id in self._dispatched_job_ids:
            return self._outcome(
                request,
                OUTCOME_REFUSED,
                refusal_code=JOB_ID_REUSED,
                refusal_reason=(
                    f"kernel {record.job_id!r} already ran an attempt; one "
                    "kernel, one attempt, no shared state"
                ),
                failure_class=FailureClass.INFRASTRUCTURE.value,
                job_id=record.job_id,
                source_commit_sha=record.source_commit_sha,
            )
        self._dispatched_job_ids.add(record.job_id)

        # R1: the code identity the kernel echoed must equal the declaration.
        # A record that reports a failure without identifying its source at
        # all (empty echo: the install never completed) is not a source
        # mismatch -- it is the failure the record itself reports, classified
        # below. A non-empty echo that differs still refuses whichever state
        # the record claims: the job ran other code than the campaign declared.
        echoed_commit = record.source_commit_sha.strip().lower()
        source_unidentified = not echoed_commit and record.state != "complete"
        if echoed_commit != request.source.commit_sha and not source_unidentified:
            return self._outcome(
                request,
                OUTCOME_REFUSED,
                refusal_code=SOURCE_COMMIT_MISMATCH,
                refusal_reason=(
                    f"the job ran commit {record.source_commit_sha!r}, but the "
                    f"attempt declared {request.source.commit_sha!r}; a job "
                    "that ran different code has produced no evidence"
                ),
                failure_class=FailureClass.INFRASTRUCTURE.value,
                job_id=record.job_id,
                source_commit_sha=record.source_commit_sha,
            )
        if record.accelerator != self.accelerator.name:
            return self._outcome(
                request,
                OUTCOME_REFUSED,
                refusal_code=ACCELERATOR_MISMATCH,
                refusal_reason=(
                    f"the job ran on {record.accelerator!r}, but the campaign "
                    f"declared {self.accelerator.name!r}; the declared shape's "
                    "multiplier prices the spend, so a different shape cannot "
                    "be settled"
                ),
                failure_class=FailureClass.INFRASTRUCTURE.value,
                job_id=record.job_id,
                source_commit_sha=record.source_commit_sha,
            )
        if tuple(record.mounts) != tuple(request.mounts):
            return self._outcome(
                request,
                OUTCOME_REFUSED,
                refusal_code=MOUNTS_MISMATCH,
                refusal_reason=(
                    f"the job declares mounts {list(record.mounts)!r}, but the "
                    f"attempt declared {list(request.mounts)!r}; the mount "
                    "surface is part of the contamination contract"
                ),
                failure_class=FailureClass.INFRASTRUCTURE.value,
                job_id=record.job_id,
                source_commit_sha=record.source_commit_sha,
            )
        resume_refusal = self._resume_refusal(request, record)
        if resume_refusal is not None:
            code, reason = resume_refusal
            return self._outcome(
                request,
                OUTCOME_REFUSED,
                refusal_code=code,
                refusal_reason=reason,
                failure_class=FailureClass.INFRASTRUCTURE.value,
                job_id=record.job_id,
                source_commit_sha=record.source_commit_sha,
            )

        # R5: a job that did not complete is classified from its own record,
        # and the quota its session burned is still measured and settled. The
        # completed-job provenance checks below do not apply to it: a failed
        # attempt has no evidence to verify, and demanding it would replace
        # the failure's own error (an install failure can echo neither an
        # installed commit nor verified inputs) with a derived refusal.
        if record.state != "complete":
            return self._failed_record_outcome(
                request, record, quota_before, quota_after
            )

        # R2a: the kernel must have verified every declared input, exactly.
        expected_inputs = {(entry.name, entry.sha256) for entry in request.inputs}
        observed_inputs = set(record.input_verification)
        if observed_inputs != expected_inputs:
            unverified = sorted(expected_inputs - observed_inputs)
            undeclared = sorted(observed_inputs - expected_inputs)
            return self._outcome(
                request,
                OUTCOME_REFUSED,
                refusal_code=INPUT_VERIFICATION_INCOMPLETE,
                refusal_reason=(
                    "the kernel did not verify the declared inputs exactly: "
                    f"unverified {unverified}, undeclared {undeclared}"
                ),
                failure_class=FailureClass.INFRASTRUCTURE.value,
                job_id=record.job_id,
                source_commit_sha=record.source_commit_sha,
            )

        # R2b: artifacts verify before anything may reference them.
        try:
            verify_artifact_manifest(record.artifact_manifest, output_root)
        except ComputeBackendRefusal as artifact_refusal:
            return self._outcome(
                request,
                OUTCOME_REFUSED,
                refusal_code=artifact_refusal.code,
                refusal_reason=artifact_refusal.reason,
                failure_class=FailureClass.INFRASTRUCTURE.value,
                job_id=record.job_id,
                source_commit_sha=record.source_commit_sha,
            )
        if self.artifact_admission is not None:
            for entry in record.artifact_manifest:
                try:
                    self.artifact_admission(entry, output_root / entry.path)
                except Exception as exc:  # noqa: BLE001 - wrapped, not swallowed
                    return self._outcome(
                        request,
                        OUTCOME_REFUSED,
                        refusal_code=ARTIFACT_ADMISSION_REFUSED,
                        refusal_reason=(
                            f"artifact {entry.path!r} failed admission: {exc}"
                        ),
                        failure_class=FailureClass.INFRASTRUCTURE.value,
                        job_id=record.job_id,
                        source_commit_sha=record.source_commit_sha,
                    )

        # R6: the environment the kernel recorded travels with the evidence.
        missing_environment = [
            field_name
            for field_name in self.require_environment_fields
            if not record.environment.get(field_name)
        ]
        if missing_environment:
            return self._outcome(
                request,
                OUTCOME_REFUSED,
                refusal_code=ENVIRONMENT_UNRECORDED,
                refusal_reason=(
                    "the job did not record its environment: missing "
                    f"{missing_environment}; provenance that was not recorded "
                    "cannot be reconstructed"
                ),
                failure_class=FailureClass.INFRASTRUCTURE.value,
                job_id=record.job_id,
                source_commit_sha=record.source_commit_sha,
            )

        # R3: measured cost from the quota delta, settled like any other spend.
        cost, settlement, settled_refusal = self._settle_spend(
            request, record, quota_before, quota_after
        )
        if settled_refusal is not None:
            return settled_refusal

        # R4b: a declared resume that did not take is reported, never silent.
        if request.resume_from is not None and record.resume_state == "not-a-resume":
            return self._outcome(
                request,
                OUTCOME_FAILED,
                refusal_code=RESUME_NOT_TAKEN,
                refusal_reason=(
                    "the kernel reported not-a-resume: the declared checkpoint "
                    f"{request.resume_from!r} did not load, so the continuation "
                    "was never trained"
                ),
                failure_class=FailureClass.INFRASTRUCTURE.value,
                cost=cost,
                settlement=settlement,
                environment=record.environment,
                resume_state="not-a-resume",
                job_id=record.job_id,
                source_commit_sha=record.source_commit_sha,
                log=record.error,
            )

        outcome = self._outcome(
            request,
            OUTCOME_SUCCEEDED,
            artifacts=tuple(record.artifact_manifest),
            cost=cost,
            settlement=settlement,
            environment=record.environment,
            resume_state=record.resume_state or "not-applicable",
            mounts=tuple(record.mounts),
            job_id=record.job_id,
            source_commit_sha=record.source_commit_sha,
            log=record.error,
        )
        self._write_job_record(
            output_root, spec, record, cost, settlement, quota_before, quota_after
        )
        return outcome

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _required_quota_hours(self, request: AttemptRequest) -> float:
        projected = request.projected_cost
        quota_counted = (
            projected.device_gpu_hours / self.accelerator.device_gpu_hours_multiplier
        )
        return max(projected.wall_gpu_hours, quota_counted)

    def _build_spec(self, request: AttemptRequest) -> KaggleJobSpec:
        return KaggleJobSpec(
            spec_id=(
                f"{request.source.cycle_id}:{request.source.recipe_id}:"
                f"{request.source.attempt_id}"
            ),
            source=request.source,
            entry_point=request.entry_point,
            payload=dict(request.payload),
            inputs=tuple(request.inputs),
            mounts=tuple(request.mounts),
            resume_from=request.resume_from,
            timeout_seconds=float(request.timeout_seconds),
            accelerator=self.accelerator.name,
        )

    def _failed_record_outcome(
        self,
        request: AttemptRequest,
        record: KaggleJobRecord,
        quota_before: KaggleQuota,
        quota_after: KaggleQuota,
    ) -> AttemptOutcome:
        """R5: a failed job is classified from its own record, with its spend settled.

        The completed-job provenance checks do not apply: a failed attempt has
        no evidence to verify. An install failure, for one, can echo neither an
        installed commit nor verified inputs, so demanding a completed job's
        record from it would surface a derived refusal instead of the install
        error. The quota a failed session burned is measured the same way.
        """
        cost, settlement, settled_refusal = self._settle_spend(
            request, record, quota_before, quota_after
        )
        if settled_refusal is not None:
            return settled_refusal
        classification = classify_failure(
            {
                "resume_state": record.resume_state,
                "candidate_succeeded": False,
                "status": record.state,
                "failure_reason": record.error,
            }
        )
        detail = f" -- {classification.reason}" if classification.reason else ""
        return self._outcome(
            request,
            OUTCOME_FAILED,
            refusal_code=JOB_FAILED,
            refusal_reason=(
                f"job state {record.state!r}: {record.error or 'no error recorded'}"
                f"{detail}"
            ),
            failure_class=classification.failure_class.value,
            cost=cost,
            settlement=settlement,
            environment=record.environment,
            resume_state=record.resume_state or "not-applicable",
            job_id=record.job_id,
            source_commit_sha=record.source_commit_sha,
            log=record.error,
        )

    def _settle_spend(
        self,
        request: AttemptRequest,
        record: KaggleJobRecord,
        quota_before: KaggleQuota,
        quota_after: KaggleQuota,
    ) -> tuple[ComputeCost | None, SettlementVerdict | None, AttemptOutcome | None]:
        """R3: measured cost from the quota delta, settled like any other spend.

        Returns the cost, the verdict and ``None``, or the refusing outcome
        when the reading regressed or the settlement refuses -- a settlement
        refusal is classified BUDGET_EXHAUSTED, never silently re-queued. The
        consumed quota is charged whatever the attempt did, because the kernel
        ran and the platform charges it whether or not it produced anything.
        """
        if quota_after.remaining_gpu_hours > quota_before.remaining_gpu_hours + 1e-9:
            return (
                None,
                None,
                self._outcome(
                    request,
                    OUTCOME_REFUSED,
                    refusal_code=QUOTA_READING_REGRESSED,
                    refusal_reason=(
                        f"quota balance rose from {quota_before.remaining_gpu_hours:.6f} "
                        f"to {quota_after.remaining_gpu_hours:.6f}; a job cannot "
                        "consume negative quota, so its cost is unmeasurable"
                    ),
                    failure_class=FailureClass.INFRASTRUCTURE.value,
                    job_id=record.job_id,
                    source_commit_sha=record.source_commit_sha,
                ),
            )
        quota_delta = quota_before.remaining_gpu_hours - quota_after.remaining_gpu_hours
        cost = ComputeCost.measured(
            device_gpu_hours=quota_delta * self.accelerator.device_gpu_hours_multiplier,
            wall_gpu_hours=record.wall_seconds / 3600.0,
            source=f"kaggle:{record.job_id}",
            measurement_method=(
                f"quota delta {quota_delta:.6f} session hour(s) x "
                f"{self.accelerator.device_count} declared device(s); wall from "
                "the job record"
            ),
        )
        settlement = settle_cost(
            actual=cost,
            projected=request.projected_cost,
            device_ceiling=request.device_ceiling,
            wall_ceiling=request.wall_ceiling,
            project_budget_wall_gpu_hours=request.project_budget_wall_gpu_hours,
            projection_tolerance=request.projection_tolerance,
        )
        self._quota_consumed_gpu_hours += quota_delta
        if (
            self.declared_quota_ceiling_gpu_hours is not None
            and self._quota_consumed_gpu_hours
            > self.declared_quota_ceiling_gpu_hours + 1e-12
        ):
            settlement = SettlementVerdict(
                compliant=False,
                failure_reasons=(
                    *settlement.failure_reasons,
                    f"{QUOTA_CEILING_EXCEEDED}: session quota consumed "
                    f"{self._quota_consumed_gpu_hours:.6f} exceeds the declared "
                    f"Kaggle ceiling "
                    f"{self.declared_quota_ceiling_gpu_hours:.6f}",
                ),
            )
        if not settlement.compliant:
            first_reason = (
                settlement.failure_reasons[0]
                if settlement.failure_reasons
                else "budget_settlement"
            )
            return (
                cost,
                settlement,
                self._outcome(
                    request,
                    OUTCOME_REFUSED,
                    refusal_code=first_reason.split(":", 1)[0].strip(),
                    refusal_reason=first_reason,
                    failure_class=FailureClass.BUDGET_EXHAUSTED.value,
                    cost=cost,
                    settlement=settlement,
                    environment=record.environment,
                    resume_state=record.resume_state or "not-applicable",
                    job_id=record.job_id,
                    source_commit_sha=record.source_commit_sha,
                    log=record.error,
                ),
            )
        return cost, settlement, None

    def _resume_refusal(
        self, request: AttemptRequest, record: KaggleJobRecord
    ) -> tuple[str, str] | None:
        if request.resume_from is None:
            if record.resume_state == "resumed" or record.resume_from:
                return (
                    RESUME_UNDECLARED,
                    "the job claims a resume "
                    f"({record.resume_state!r} from {record.resume_from!r}), but "
                    "the attempt declared none: a resume the lineage never "
                    "declared is a silent restart",
                )
            return None
        if record.resume_from != request.resume_from:
            return (
                RESUME_IDENTITY_MISMATCH,
                f"the job resumed from {record.resume_from!r}, but the attempt "
                f"declared {request.resume_from!r}: the continuation must bind "
                "to the checkpoint the search chose",
            )
        if record.resume_state not in RESUME_STATES:
            return (
                RESUME_VOCABULARY_INVALID,
                f"resume_state {record.resume_state!r} is not one of "
                f"{RESUME_STATES}; the search cannot read it",
            )
        return None

    def _outcome(
        self,
        request: AttemptRequest,
        status: str,
        *,
        refusal_code: str | None = None,
        refusal_reason: str = "",
        failure_class: str | None = None,
        artifacts: tuple[ArtifactEntry, ...] = (),
        cost: ComputeCost | None = None,
        settlement: SettlementVerdict | None = None,
        environment: Mapping[str, Any] | None = None,
        resume_state: str = "not-applicable",
        mounts: tuple[str, ...] | None = None,
        job_id: str = "",
        log: str = "",
        source_commit_sha: str = "",
    ) -> AttemptOutcome:
        return AttemptOutcome(
            source=request.source,
            status=status,
            artifacts=tuple(artifacts),
            cost=cost,
            settlement=settlement,
            failure_class=failure_class,
            refusal_code=refusal_code,
            refusal_reason=refusal_reason,
            environment=dict(environment or {}),
            resume_state=resume_state,
            mounts=tuple(request.mounts if mounts is None else mounts),
            job_id=job_id,
            log=log,
            source_commit_sha=source_commit_sha,
        )

    def _write_job_record(
        self,
        output_root: Path,
        spec: KaggleJobSpec,
        record: KaggleJobRecord,
        cost: ComputeCost,
        settlement: SettlementVerdict,
        quota_before: KaggleQuota,
        quota_after: KaggleQuota,
    ) -> None:
        document = {
            "backend": self.name,
            "spec": spec.to_dict(),
            "job_id": record.job_id,
            "state": record.state,
            "accelerator": record.accelerator,
            "source_commit_sha": record.source_commit_sha,
            "quota_before_gpu_hours": quota_before.remaining_gpu_hours,
            "quota_after_gpu_hours": quota_after.remaining_gpu_hours,
            "wall_seconds": record.wall_seconds,
            "cost": cost.to_dict(),
            "settlement": settlement.to_dict(),
            "environment": dict(record.environment),
            "resume_state": record.resume_state,
            "resume_from": record.resume_from,
            "mounts": list(record.mounts),
            "input_verification": [list(pair) for pair in record.input_verification],
        }
        (output_root / BACKEND_JOB_RECORD_NAME).write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
