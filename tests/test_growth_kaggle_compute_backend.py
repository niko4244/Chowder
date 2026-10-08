"""The first-class Kaggle compute backend, exercised offline.

The transport is a fake: it consumes a declared quota delta, writes exactly
the artifacts a kernel would pull, and echoes -- or deliberately fails to
echo -- the identity the backend is supposed to verify. Every R1-R8 check in
``docs/KAGGLE_BACKEND_REQUIREMENTS.md`` has a test that makes it fail, so the
bar cannot regress into prose.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from chowder.growth.attempt_failure import FailureClass
from chowder.growth.compute_backend import (
    ARTIFACT_ADMISSION_REFUSED,
    ARTIFACT_DIGEST_MISMATCH,
    ARTIFACT_EXTRA,
    ARTIFACT_MISSING,
    DECLARED_INPUT_DIGEST_MISMATCH,
    OUTCOME_FAILED,
    OUTCOME_REFUSED,
    OUTCOME_SUCCEEDED,
    PAYLOAD_NOT_SERIALIZABLE,
    SOURCE_BINDING_SCHEMA,
    ArtifactEntry,
    AttemptRequest,
    ComputeBackend,
    ComputeBackendRefusal,
    SourceBinding,
    bind_declared_inputs,
    file_digest,
    verify_artifact_manifest,
    verify_declared_inputs,
)
from chowder.growth.compute_cost import ComputeCost, settlement_refusal
from chowder.growth.kaggle_compute import (
    TRANSPORT_FAILED,
    ACCELERATOR_MISMATCH,
    ACCELERATOR_UNKNOWN,
    DESTINATION_NOT_FRESH,
    ENVIRONMENT_UNRECORDED,
    INPUT_VERIFICATION_INCOMPLETE,
    JOB_FAILED,
    JOB_ID_REUSED,
    MOUNTS_MISMATCH,
    QUOTA_CEILING_EXCEEDED,
    QUOTA_INSUFFICIENT,
    QUOTA_READING_REGRESSED,
    RESUME_IDENTITY_MISMATCH,
    RESUME_NOT_TAKEN,
    RESUME_UNDECLARED,
    RESUME_VOCABULARY_INVALID,
    SOURCE_COMMIT_MISMATCH,
    KaggleComputeBackend,
    KaggleJobRecord,
    KaggleQuota,
    KaggleTransportError,
    resolve_kaggle_accelerator,
)

COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40
ARTIFACT_PATH = "adapter/adapter_model.safetensors"
ARTIFACT_BYTES = b"quantized-weights"


def _source(attempt_id: str = "attempt-01") -> SourceBinding:
    return SourceBinding(
        repository="https://github.com/niko4244/Chowder",
        commit_sha=COMMIT,
        chowder_version="0.5.0-dev",
        cycle_id="gen2",
        recipe_id="recipe-01",
        attempt_id=attempt_id,
    )


def _declared_inputs(tmp_path: Path):
    material = tmp_path / "training-material.json"
    material.write_text('{"examples": 4}\n', encoding="utf-8")
    base = tmp_path / "base-model.manifest.json"
    base.write_text('{"model": "qwen3.8"}\n', encoding="utf-8")
    return bind_declared_inputs(
        {"training_material": material, "base_model": base}
    )


def _request(tmp_path: Path, **overrides) -> AttemptRequest:
    fields: dict = {
        "source": _source(),
        "entry_point": "kaggle/kernel_entry.py",
        "inputs": _declared_inputs(tmp_path),
        "payload": {"recipe": "recipe-01", "max_steps": 3},
        "projected_cost": ComputeCost.measured(
            device_gpu_hours=0.20, wall_gpu_hours=0.50, source="planner-projection"
        ),
        "device_ceiling": 0.75,
        "wall_ceiling": 0.75,
        "project_budget_wall_gpu_hours": 1.5,
        "projection_tolerance": 0.25,
        "mounts": ("owner/dataset",),
        "resume_from": None,
        "timeout_seconds": 600.0,
    }
    fields.update(overrides)
    return AttemptRequest(**fields)


def _artifact_record(
    spec, destination: Path, *, input_verification=None
) -> KaggleJobRecord:
    manifest = []
    for path in sorted(Path(destination).rglob("*")):
        if path.is_file():
            sha256, size = file_digest(path)
            manifest.append(
                ArtifactEntry(path.relative_to(destination).as_posix(), sha256, size)
            )
    return KaggleJobRecord(
        job_id="kernel-20261005-0001",
        state="complete",
        wall_seconds=1800.0,
        accelerator=spec.accelerator,
        source_commit_sha=spec.source.commit_sha,
        input_verification=(
            tuple(input_verification)
            if input_verification is not None
            else tuple((entry.name, entry.sha256) for entry in spec.inputs)
        ),
        artifact_manifest=tuple(manifest),
        environment={
            "python_version": "3.12.4",
            "packages": {"torch": "2.4.0", "transformers": "5.18.0"},
            "model_commit": "c" * 40,
        },
        resume_state=None,
        resume_from=None,
        mounts=tuple(spec.mounts),
    )


class FakeTransport:
    """A Kaggle shaped like a fake: quota readings and one job that writes files."""

    def __init__(
        self,
        *,
        remaining: float = 10.0,
        consume: float = 0.25,
        files: dict[str, bytes] | None = None,
        record=None,
        record_overrides: dict | None = None,
        input_verification=None,
    ) -> None:
        self.remaining = remaining
        self.consume = consume
        self.files = dict(
            files if files is not None else {ARTIFACT_PATH: ARTIFACT_BYTES}
        )
        self._record = record
        self._record_overrides = dict(record_overrides or {})
        self._input_verification = input_verification
        self.specs: list = []

    def quota(self) -> KaggleQuota:
        return KaggleQuota(remaining_gpu_hours=self.remaining)

    def run(self, spec, destination):
        self.specs.append(spec)
        for relative, payload in self.files.items():
            target = Path(destination) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        record = self._record or _artifact_record(
            spec, Path(destination), input_verification=self._input_verification
        )
        if self._record_overrides:
            record = replace(record, **self._record_overrides)
        self.remaining = max(0.0, self.remaining - self.consume)
        return record


def _backend(transport, **overrides) -> KaggleComputeBackend:
    fields: dict = {"accelerator": "T4x2"}
    fields.update(overrides)
    return KaggleComputeBackend(transport, **fields)


# --------------------------------------------------------------------------
# the contract primitives
# --------------------------------------------------------------------------


def test_a_source_binding_requires_a_full_commit() -> None:
    with pytest.raises(ComputeBackendRefusal, match=SOURCE_BINDING_SCHEMA):
        SourceBinding(
            repository="repo",
            commit_sha="abc123",
            chowder_version="0.5",
            cycle_id="gen2",
            recipe_id="r",
            attempt_id="a",
        )


def test_bind_declared_inputs_hashes_and_sorts(tmp_path: Path) -> None:
    first = tmp_path / "z-last.json"
    first.write_text("z", encoding="utf-8")
    second = tmp_path / "a-first.json"
    second.write_text("a", encoding="utf-8")
    bound = bind_declared_inputs({"z": first, "a": second})
    assert [entry.name for entry in bound] == ["a", "z"]
    assert bound[0].sha256 == file_digest(second)[0]
    assert bound[0].bytes == 1


def test_a_declared_input_that_moved_refuses_reverification(tmp_path: Path) -> None:
    inputs = _declared_inputs(tmp_path)
    Path(inputs[0].path).write_text("tampered", encoding="utf-8")
    with pytest.raises(ComputeBackendRefusal, match=DECLARED_INPUT_DIGEST_MISMATCH):
        verify_declared_inputs(inputs)


def test_verify_artifact_manifest_refuses_missing_extra_and_mismatch(
    tmp_path: Path,
) -> None:
    returned = tmp_path / "returned.txt"
    returned.write_bytes(b"payload")
    sha256, size = file_digest(returned)
    with pytest.raises(ComputeBackendRefusal, match=ARTIFACT_MISSING):
        verify_artifact_manifest((ArtifactEntry("absent.txt", sha256, size),), tmp_path)
    with pytest.raises(ComputeBackendRefusal, match=ARTIFACT_EXTRA):
        verify_artifact_manifest((), tmp_path)
    with pytest.raises(ComputeBackendRefusal, match=ARTIFACT_DIGEST_MISMATCH):
        verify_artifact_manifest(
            (ArtifactEntry("returned.txt", "f" * 64, size),), tmp_path
        )
    verified = verify_artifact_manifest(
        (ArtifactEntry("returned.txt", sha256, size),), tmp_path
    )
    assert verified == (returned,)


def test_attempt_request_refuses_an_unserializable_payload(tmp_path: Path) -> None:
    with pytest.raises(ComputeBackendRefusal, match=PAYLOAD_NOT_SERIALIZABLE):
        _request(tmp_path, payload={"bad": object()})


# --------------------------------------------------------------------------
# the Kaggle backend: R1-R8
# --------------------------------------------------------------------------


def test_the_backend_satisfies_the_compute_backend_protocol() -> None:
    backend = _backend(FakeTransport())
    assert isinstance(backend, ComputeBackend)
    assert backend.name == "kaggle"
    assert resolve_kaggle_accelerator("T4x2").device_gpu_hours_multiplier == 2.0


def test_an_unknown_accelerator_refuses() -> None:
    with pytest.raises(ComputeBackendRefusal, match=ACCELERATOR_UNKNOWN):
        _backend(FakeTransport(), accelerator="TPU-v5")


def test_preflight_refuses_a_declared_input_that_moved_before_any_push(
    tmp_path: Path,
) -> None:
    transport = FakeTransport()
    backend = _backend(transport)
    request = _request(tmp_path)
    Path(request.inputs[0].path).write_text("tampered", encoding="utf-8")
    outcome = backend.dispatch(request, destination=tmp_path / "attempt")
    assert outcome.status == OUTCOME_REFUSED
    assert outcome.refusal_code == DECLARED_INPUT_DIGEST_MISMATCH
    assert transport.specs == [], "nothing may be pushed when an input moved"


def test_preflight_refuses_a_projection_the_quota_cannot_cover(
    tmp_path: Path,
) -> None:
    transport = FakeTransport(remaining=0.10)
    backend = _backend(transport)
    outcome = backend.dispatch(_request(tmp_path), destination=tmp_path / "attempt")
    assert outcome.status == OUTCOME_REFUSED
    assert outcome.refusal_code == QUOTA_INSUFFICIENT
    assert outcome.failure_class == FailureClass.BUDGET_EXHAUSTED.value
    assert transport.specs == []


def test_the_declared_kaggle_budget_bounds_the_session(tmp_path: Path) -> None:
    transport = FakeTransport(consume=0.25)
    backend = _backend(transport, declared_quota_ceiling_gpu_hours=0.6)
    first = backend.dispatch(_request(tmp_path), destination=tmp_path / "attempt-1")
    assert first.status == OUTCOME_SUCCEEDED
    second = backend.dispatch(_request(tmp_path), destination=tmp_path / "attempt-2")
    assert second.status == OUTCOME_REFUSED
    assert second.refusal_code == QUOTA_CEILING_EXCEEDED
    assert second.failure_class == FailureClass.BUDGET_EXHAUSTED.value


def test_the_spec_carries_the_source_binding_and_input_digests(
    tmp_path: Path,
) -> None:
    transport = FakeTransport()
    backend = _backend(transport)
    request = _request(tmp_path)
    outcome = backend.dispatch(request, destination=tmp_path / "attempt")
    assert outcome.status == OUTCOME_SUCCEEDED
    spec = transport.specs[0]
    assert spec.spec_id == "gen2:recipe-01:attempt-01"
    assert spec.source.commit_sha == COMMIT
    assert spec.entry_point == "kaggle/kernel_entry.py"
    assert spec.accelerator == "T4x2"
    assert spec.mounts == ("owner/dataset",)
    assert spec.resume_from is None
    assert spec.timeout_seconds == 600.0
    assert [
        (entry.name, entry.sha256, entry.bytes) for entry in spec.inputs
    ] == [(entry.name, entry.sha256, entry.bytes) for entry in request.inputs]


def test_a_returned_commit_that_differs_refuses(tmp_path: Path) -> None:
    transport = FakeTransport(record_overrides={"source_commit_sha": OTHER_COMMIT})
    outcome = _backend(transport).dispatch(
        _request(tmp_path), destination=tmp_path / "attempt"
    )
    assert outcome.status == OUTCOME_REFUSED
    assert outcome.refusal_code == SOURCE_COMMIT_MISMATCH
    assert outcome.failure_class == FailureClass.INFRASTRUCTURE.value


def test_an_input_the_kernel_did_not_verify_refuses(tmp_path: Path) -> None:
    transport = FakeTransport(input_verification=())
    outcome = _backend(transport).dispatch(
        _request(tmp_path), destination=tmp_path / "attempt"
    )
    assert outcome.status == OUTCOME_REFUSED
    assert outcome.refusal_code == INPUT_VERIFICATION_INCOMPLETE


def test_a_missing_artifact_refuses(tmp_path: Path) -> None:
    transport = FakeTransport(
        record_overrides={
            "artifact_manifest": (ArtifactEntry("absent.bin", "e" * 64, 3),)
        }
    )
    outcome = _backend(transport).dispatch(
        _request(tmp_path), destination=tmp_path / "attempt"
    )
    assert outcome.status == OUTCOME_REFUSED
    assert outcome.refusal_code == ARTIFACT_MISSING


def test_an_unlisted_artifact_refuses(tmp_path: Path) -> None:
    transport = FakeTransport(record_overrides={"artifact_manifest": ()})
    outcome = _backend(transport).dispatch(
        _request(tmp_path), destination=tmp_path / "attempt"
    )
    assert outcome.status == OUTCOME_REFUSED
    assert outcome.refusal_code == ARTIFACT_EXTRA


def test_a_hash_mismatched_artifact_refuses(tmp_path: Path) -> None:
    transport = FakeTransport(
        record_overrides={
            "artifact_manifest": (
                ArtifactEntry(ARTIFACT_PATH, "f" * 64, len(ARTIFACT_BYTES)),
            )
        }
    )
    outcome = _backend(transport).dispatch(
        _request(tmp_path), destination=tmp_path / "attempt"
    )
    assert outcome.status == OUTCOME_REFUSED
    assert outcome.refusal_code == ARTIFACT_DIGEST_MISMATCH


def test_an_artifact_the_admission_check_refuses_is_refused(tmp_path: Path) -> None:
    def refuse_admission(entry, path):
        raise ValueError(f"inert adapter at {path.name}")

    transport = FakeTransport()
    backend = _backend(transport, artifact_admission=refuse_admission)
    outcome = backend.dispatch(_request(tmp_path), destination=tmp_path / "attempt")
    assert outcome.status == OUTCOME_REFUSED
    assert outcome.refusal_code == ARTIFACT_ADMISSION_REFUSED
    assert "inert adapter" in outcome.refusal_reason


def test_a_job_without_a_recorded_environment_refuses(tmp_path: Path) -> None:
    transport = FakeTransport(record_overrides={"environment": {}})
    outcome = _backend(transport).dispatch(
        _request(tmp_path), destination=tmp_path / "attempt"
    )
    assert outcome.status == OUTCOME_REFUSED
    assert outcome.refusal_code == ENVIRONMENT_UNRECORDED


def test_cost_settles_from_the_quota_delta_and_the_declared_shape(
    tmp_path: Path,
) -> None:
    transport = FakeTransport(consume=0.25)
    outcome = _backend(transport).dispatch(
        _request(tmp_path), destination=tmp_path / "attempt"
    )
    assert outcome.status == OUTCOME_SUCCEEDED
    assert outcome.cost is not None
    assert outcome.cost.device_measured is True
    assert outcome.cost.device_gpu_hours == pytest.approx(0.50)  # 0.25 session h x 2 devices
    assert outcome.cost.wall_gpu_hours == pytest.approx(0.50)  # 1800 s
    assert outcome.settlement is not None and outcome.settlement.compliant
    evidence = outcome.to_evidence()
    assert evidence["status"] == OUTCOME_SUCCEEDED
    assert evidence["candidate_succeeded"] is True
    assert evidence["source_commit_sha"] == COMMIT
    assert evidence["refused_by"] is None
    assert evidence["measured_gpu_hours"] == pytest.approx(0.50)
    assert evidence["compute_cost"]["device_measured"] is True


def test_an_over_budget_attempt_is_refused_as_budget_exhausted(
    tmp_path: Path,
) -> None:
    transport = FakeTransport(consume=1.0)  # 2.0 device-hours vs a 0.30 ceiling
    outcome = _backend(transport).dispatch(
        _request(tmp_path), destination=tmp_path / "attempt"
    )
    assert outcome.status == OUTCOME_REFUSED
    assert outcome.failure_class == FailureClass.BUDGET_EXHAUSTED.value
    evidence = outcome.to_evidence()
    assert settlement_refusal(evidence) == "ACTUAL_DEVICE_GPU_HOURS_EXCEEDED"
    assert evidence["refused_by"] == "budget_settlement"


def test_a_job_that_did_not_complete_is_classified_not_just_logged(
    tmp_path: Path,
) -> None:
    transport = FakeTransport(
        record_overrides={"state": "error", "error": "CUDA out of memory"}
    )
    outcome = _backend(transport).dispatch(
        _request(tmp_path), destination=tmp_path / "attempt"
    )
    assert outcome.status == OUTCOME_FAILED
    assert outcome.refusal_code == JOB_FAILED
    assert outcome.failure_class == FailureClass.INFRASTRUCTURE.value
    assert "CUDA out of memory" in outcome.refusal_reason
    evidence = outcome.to_evidence()
    assert evidence["candidate_succeeded"] is None
    assert evidence["failure_class"] == FailureClass.INFRASTRUCTURE.value


def test_a_failed_record_is_classified_from_its_own_error_without_provenance(
    tmp_path: Path,
) -> None:
    """An install failure echoes neither a commit nor verified inputs; it must
    still surface as the failure it is, with the quota its session burned."""
    transport = FakeTransport(
        record_overrides={
            "state": "error",
            "error": "installing the pinned source failed: network unreachable",
            "source_commit_sha": "",
            "input_verification": (),
            "artifact_manifest": (),
            "environment": {},
        }
    )
    outcome = _backend(transport).dispatch(
        _request(tmp_path), destination=tmp_path / "attempt"
    )
    assert outcome.status == OUTCOME_FAILED
    assert outcome.refusal_code == JOB_FAILED
    assert outcome.failure_class == FailureClass.INFRASTRUCTURE.value
    assert "installing the pinned source failed" in outcome.refusal_reason
    assert outcome.cost is not None and outcome.cost.device_measured is True
    assert outcome.settlement is not None and outcome.settlement.compliant
    evidence = outcome.to_evidence()
    assert evidence["candidate_succeeded"] is None
    assert evidence["refused_by"] == JOB_FAILED


def test_a_failed_record_that_identified_no_source_echoes_nothing(
    tmp_path: Path,
) -> None:
    """An install failure cannot name a commit; the evidence must not repeat
    the declared one as if the kernel had verified it."""
    transport = FakeTransport(
        record_overrides={
            "state": "error",
            "error": "installing the pinned source failed: network unreachable",
            "source_commit_sha": "",
        }
    )
    outcome = _backend(transport).dispatch(
        _request(tmp_path), destination=tmp_path / "attempt"
    )
    assert outcome.status == OUTCOME_FAILED
    evidence = outcome.to_evidence()
    assert evidence["source_commit_sha"] is None
    assert "installing the pinned source failed" in evidence["refusal_reason"]


def test_a_failed_record_that_ran_other_code_is_still_a_source_mismatch(
    tmp_path: Path,
) -> None:
    transport = FakeTransport(
        record_overrides={"state": "error", "source_commit_sha": OTHER_COMMIT}
    )
    outcome = _backend(transport).dispatch(
        _request(tmp_path), destination=tmp_path / "attempt"
    )
    assert outcome.status == OUTCOME_REFUSED
    assert outcome.refusal_code == SOURCE_COMMIT_MISMATCH
    assert outcome.failure_class == FailureClass.INFRASTRUCTURE.value


def test_a_resume_must_bind_the_declared_checkpoint(tmp_path: Path) -> None:
    request = _request(tmp_path, resume_from="ckpt-0007")
    transport = FakeTransport(
        record_overrides={"resume_state": "resumed", "resume_from": "ckpt-9999"}
    )
    outcome = _backend(transport).dispatch(request, destination=tmp_path / "attempt")
    assert outcome.status == OUTCOME_REFUSED
    assert outcome.refusal_code == RESUME_IDENTITY_MISMATCH


def test_a_resume_must_speak_the_search_vocabulary(tmp_path: Path) -> None:
    request = _request(tmp_path, resume_from="ckpt-0007")
    transport = FakeTransport(
        record_overrides={"resume_state": "maybe", "resume_from": "ckpt-0007"}
    )
    outcome = _backend(transport).dispatch(request, destination=tmp_path / "attempt")
    assert outcome.status == OUTCOME_REFUSED
    assert outcome.refusal_code == RESUME_VOCABULARY_INVALID


def test_a_resume_that_did_not_take_is_reported_not_silent(tmp_path: Path) -> None:
    request = _request(tmp_path, resume_from="ckpt-0007")
    transport = FakeTransport(
        record_overrides={"resume_state": "not-a-resume", "resume_from": "ckpt-0007"}
    )
    outcome = _backend(transport).dispatch(request, destination=tmp_path / "attempt")
    assert outcome.status == OUTCOME_FAILED
    assert outcome.refusal_code == RESUME_NOT_TAKEN
    assert outcome.resume_state == "not-a-resume"
    assert outcome.failure_class == FailureClass.INFRASTRUCTURE.value


def test_the_declared_resume_reaches_the_spec_and_the_success_outcome(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, resume_from="ckpt-0007")
    transport = FakeTransport(
        record_overrides={"resume_state": "resumed", "resume_from": "ckpt-0007"}
    )
    outcome = _backend(transport).dispatch(request, destination=tmp_path / "attempt")
    assert transport.specs[0].resume_from == "ckpt-0007"
    assert outcome.status == OUTCOME_SUCCEEDED
    assert outcome.resume_state == "resumed"


def test_a_job_that_claims_an_undeclared_resume_refuses(tmp_path: Path) -> None:
    transport = FakeTransport(
        record_overrides={"resume_state": "resumed", "resume_from": "ckpt-0007"}
    )
    outcome = _backend(transport).dispatch(
        _request(tmp_path), destination=tmp_path / "attempt"
    )
    assert outcome.status == OUTCOME_REFUSED
    assert outcome.refusal_code == RESUME_UNDECLARED


def test_recorded_mounts_must_match_the_declared_mounts(tmp_path: Path) -> None:
    transport = FakeTransport(record_overrides={"mounts": ()})
    outcome = _backend(transport).dispatch(
        _request(tmp_path), destination=tmp_path / "attempt"
    )
    assert outcome.status == OUTCOME_REFUSED
    assert outcome.refusal_code == MOUNTS_MISMATCH


def test_the_accelerator_the_job_reports_must_match_the_declared_shape(
    tmp_path: Path,
) -> None:
    transport = FakeTransport(record_overrides={"accelerator": "T4"})
    outcome = _backend(transport).dispatch(
        _request(tmp_path), destination=tmp_path / "attempt"
    )
    assert outcome.status == OUTCOME_REFUSED
    assert outcome.refusal_code == ACCELERATOR_MISMATCH


def test_a_kernel_is_never_reused_for_two_attempts(tmp_path: Path) -> None:
    transport = FakeTransport()
    backend = _backend(transport)
    first = backend.dispatch(_request(tmp_path), destination=tmp_path / "attempt-1")
    assert first.status == OUTCOME_SUCCEEDED
    second = backend.dispatch(_request(tmp_path), destination=tmp_path / "attempt-2")
    assert second.status == OUTCOME_REFUSED
    assert second.refusal_code == JOB_ID_REUSED


def test_a_non_fresh_destination_refuses(tmp_path: Path) -> None:
    destination = tmp_path / "attempt"
    destination.mkdir()
    (destination / "leftover.bin").write_bytes(b"another attempt's output")
    transport = FakeTransport()
    outcome = _backend(transport).dispatch(_request(tmp_path), destination=destination)
    assert outcome.status == OUTCOME_REFUSED
    assert outcome.refusal_code == DESTINATION_NOT_FRESH
    assert transport.specs == []


def test_a_quota_reading_that_regresses_makes_cost_unmeasurable(
    tmp_path: Path,
) -> None:
    transport = FakeTransport(consume=-0.25)  # balance rises: impossible to price
    outcome = _backend(transport).dispatch(
        _request(tmp_path), destination=tmp_path / "attempt"
    )
    assert outcome.status == OUTCOME_REFUSED
    assert outcome.refusal_code == QUOTA_READING_REGRESSED


def test_the_job_record_is_written_beside_the_verified_artifacts(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "attempt"
    transport = FakeTransport(consume=0.25)
    outcome = _backend(transport).dispatch(_request(tmp_path), destination=destination)
    assert outcome.status == OUTCOME_SUCCEEDED
    document = json.loads((destination / "compute-job.json").read_text(encoding="utf-8"))
    assert document["backend"] == "kaggle"
    assert document["spec"]["source"]["commit_sha"] == COMMIT
    assert document["spec"]["inputs"][0]["sha256"]
    assert document["cost"]["device_gpu_hours"] == pytest.approx(0.50)
    assert document["settlement"]["budget_compliant"] is True
    assert document["source_commit_sha"] == COMMIT
    assert document["environment"]["python_version"] == "3.12.4"


def test_a_transport_that_cannot_read_quota_refuses_before_push(
    tmp_path: Path,
) -> None:
    class FailingQuotaTransport(FakeTransport):
        def quota(self):
            raise KaggleTransportError("the kaggle CLI was not found")

    transport = FailingQuotaTransport()
    outcome = _backend(transport).dispatch(
        _request(tmp_path), destination=tmp_path / "attempt"
    )
    assert outcome.status == OUTCOME_REFUSED
    assert outcome.refusal_code == TRANSPORT_FAILED
    assert outcome.failure_class == FailureClass.INFRASTRUCTURE.value
    assert transport.specs == []


def test_a_transport_that_cannot_run_is_classified_not_raised(
    tmp_path: Path,
) -> None:
    class FailingRunTransport(FakeTransport):
        def run(self, spec, destination):
            self.specs.append(spec)
            raise KaggleTransportError("the push was rejected by Kaggle")

    transport = FailingRunTransport()
    outcome = _backend(transport).dispatch(
        _request(tmp_path), destination=tmp_path / "attempt"
    )
    assert outcome.status == OUTCOME_FAILED
    assert outcome.refusal_code == TRANSPORT_FAILED
    assert outcome.failure_class == FailureClass.INFRASTRUCTURE.value
    assert "push was rejected" in outcome.refusal_reason
    assert outcome.to_evidence()["source_commit_sha"] is None, (
        "no record produced, so no commit was observed"
    )
