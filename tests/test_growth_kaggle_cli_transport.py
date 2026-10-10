"""The production Kaggle CLI transport, exercised offline.

A fake ``kaggle`` CLI drives the real transport: it answers quota, accepts a
push, walks a scripted status sequence, and writes a kernel output directory
(record + artifacts). No network, no token, no GPU. The end-to-end test wraps
this transport in the real ``KaggleComputeBackend`` so the whole chain --
stage, push, poll, pull, verify, settle -- is proven in one place.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import re
import subprocess
from pathlib import Path

import pytest

from chowder.growth.compute_backend import (
    OUTCOME_FAILED,
    OUTCOME_SUCCEEDED,
    AttemptRequest,
    SourceBinding,
    bind_declared_inputs,
)
from chowder.growth.compute_cost import ComputeCost
from chowder.growth.kaggle_cli_transport import (
    KERNEL_ENTRY_NAME,
    KaggleCliTransport,
    kernel_slug,
    machine_shape,
    parse_quota,
    parse_status,
)
from chowder.growth.kaggle_compute import (
    TRANSPORT_FAILED,
    JOB_FAILED,
    KaggleComputeBackend,
    KaggleJobSpec,
    KaggleTransport,
    KaggleTransportError,
)
from chowder.growth.kaggle_kernel import KERNEL_JOB_SPEC_NAME

COMMIT = "a" * 40
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{3,48}[a-z0-9]$")


def _completed(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(["kaggle"], returncode, stdout=stdout, stderr=stderr)


class FakeCli:
    """A scripted `kaggle` CLI: quota readings, statuses, and pulled output."""

    def __init__(
        self,
        *,
        quota_readings=(12.5,),
        states=("running", "complete"),
        record=None,
        artifacts=None,
        platform_files=None,
        push_error="",
    ) -> None:
        self.quota_readings = list(quota_readings)
        self.states = list(states)
        self.record = record
        self.artifacts = dict(artifacts or {})
        self.platform_files = dict(platform_files or {"kernel.log": b"platform log\n"})
        self.push_error = push_error
        self.calls: list[list[str]] = []
        self._quota_index = 0

    def __call__(self, args):
        argv = list(args)
        self.calls.append(argv)
        rest = argv[1:]
        if rest == ["quota"]:
            index = min(self._quota_index, len(self.quota_readings) - 1)
            self._quota_index += 1
            return _completed(
                0, f"GPU quota: {self.quota_readings[index]} hours remaining\n"
            )
        if rest[:2] == ["kernels", "push"]:
            if self.push_error:
                return _completed(1, "", self.push_error)
            return _completed(0, "Kernel pushed successfully\n")
        if rest[:2] == ["kernels", "status"]:
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            return _completed(0, f'Your kernel has status "{state}"\n')
        if rest[:2] == ["kernels", "output"]:
            target = Path(rest[rest.index("-p") + 1])
            target.mkdir(parents=True, exist_ok=True)
            if self.record is not None:
                (target / "job-record.json").write_text(
                    json.dumps(self.record), encoding="utf-8"
                )
            for relative, payload in {**self.platform_files, **self.artifacts}.items():
                path = target / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            return _completed(0, "output pulled\n")
        if rest[:2] == ["kernels", "logs"]:
            return _completed(0, "platform log tail\n")
        return _completed(1, "", f"unexpected command: {argv}")


def _spec(tmp_path: Path, **overrides) -> KaggleJobSpec:
    material = tmp_path / "training-material.json"
    material.write_text('{"examples": 3}\n', encoding="utf-8")
    fields: dict = {
        "spec_id": "gen2:recipe-01:attempt-01",
        "source": SourceBinding(
            repository="https://github.com/niko4244/Chowder",
            commit_sha=COMMIT,
            chowder_version="0.5.0-dev",
            cycle_id="gen2",
            recipe_id="recipe-01",
            attempt_id="attempt-01",
        ),
        "entry_point": KERNEL_ENTRY_NAME,
        "payload": {},
        "inputs": bind_declared_inputs({"training_material": material}),
        "mounts": ("owner/dataset",),
        "resume_from": None,
        "timeout_seconds": 600.0,
        "accelerator": "T4x2",
    }
    fields.update(overrides)
    return KaggleJobSpec(**fields)


def _manifest(files: dict[str, bytes]) -> list[dict]:
    return [
        {
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
        for relative, payload in sorted(files.items())
    ]


def _record_for(spec: KaggleJobSpec, files: dict[str, bytes], **overrides) -> dict:
    document = {
        "state": "complete",
        "error": "",
        "source_commit_sha": spec.source.commit_sha,
        "input_verification": [[entry.name, entry.sha256] for entry in spec.inputs],
        "artifact_manifest": _manifest(files),
        "environment": {
            "python_version": "3.12.4",
            "packages": {"torch": "2.4.0"},
            "model_commit": "c" * 40,
        },
        "resume_state": None,
        "resume_from": None,
        "mounts": list(spec.mounts),
    }
    document.update(overrides)
    return document


def _transport(cli: FakeCli, staging: Path, **overrides) -> KaggleCliTransport:
    fields: dict = {
        "owner": "nikma",
        "runner": cli,
        "poll_seconds": 5.0,
        "staging_root": staging,
        "sleep": lambda seconds: None,
    }
    fields.update(overrides)
    return KaggleCliTransport(**fields)


def _request(tmp_path: Path, **overrides) -> AttemptRequest:
    material = tmp_path / "training-material.json"
    material.write_text('{"examples": 3}\n', encoding="utf-8")
    inputs = bind_declared_inputs({"training_material": material})
    fields: dict = {
        "source": SourceBinding(
            repository="https://github.com/niko4244/Chowder",
            commit_sha=COMMIT,
            chowder_version="0.5.0-dev",
            cycle_id="gen2",
            recipe_id="recipe-01",
            attempt_id="attempt-01",
        ),
        "entry_point": KERNEL_ENTRY_NAME,
        "inputs": inputs,
        "payload": {
            "input_paths": {inputs[0].name: f"/kaggle/input/{inputs[0].name}"},
            "model_commit": "c" * 40,
        },
        "projected_cost": ComputeCost.measured(
            device_gpu_hours=0.4, wall_gpu_hours=0.5, source="planner-projection"
        ),
        "device_ceiling": 2.0,
        "wall_ceiling": 0.75,
        "project_budget_wall_gpu_hours": 1.5,
        "mounts": ("owner/dataset",),
        "timeout_seconds": 600.0,
    }
    fields.update(overrides)
    return AttemptRequest(**fields)


# --------------------------------------------------------------------------
# parsing and identity
# --------------------------------------------------------------------------


def test_the_transport_satisfies_the_protocol(tmp_path: Path) -> None:
    transport = _transport(FakeCli(), tmp_path / "staging")
    assert isinstance(transport, KaggleTransport)


def test_quota_parsing_accepts_the_documented_forms_and_refuses_junk() -> None:
    assert parse_quota("GPU quota: 17.5 hours remaining").remaining_gpu_hours == 17.5
    assert parse_quota("17h 30m remaining").remaining_gpu_hours == pytest.approx(17.5)
    assert parse_quota("3.25 GPU hours left").remaining_gpu_hours == 3.25
    with pytest.raises(KaggleTransportError, match="could not parse"):
        parse_quota("quota information unavailable")


def test_status_parsing_handles_both_cli_forms_and_refuses_junk() -> None:
    assert parse_status('Your kernel has status "complete"') == "complete"
    assert parse_status('has status "KernelWorkerStatus.RUNNING"') == "running"
    assert parse_status("KernelWorkerStatus.ERROR") == "error"
    with pytest.raises(KaggleTransportError, match="unrecognised"):
        parse_status("something entirely different")


def test_machine_shape_maps_the_declared_shape_and_refuses_unknown() -> None:
    assert machine_shape("T4x2", {"T4x2": "NvidiaTeslaT4"}) == "NvidiaTeslaT4"
    with pytest.raises(KaggleTransportError, match="no Kaggle machine_shape"):
        machine_shape("P100", {"T4x2": "NvidiaTeslaT4"})


def test_kernel_slug_is_valid_deterministic_and_unique() -> None:
    first = kernel_slug("gen2:recipe-01:attempt-01", COMMIT, "chowder")
    assert SLUG_RE.match(first), first
    assert kernel_slug("gen2:recipe-01:attempt-01", COMMIT, "chowder") == first
    assert kernel_slug("gen2:recipe-01:attempt-02", COMMIT, "chowder") != first


# --------------------------------------------------------------------------
# staging and orchestration
# --------------------------------------------------------------------------


def test_the_staged_kernel_carries_the_declared_identity(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    files = {"adapter/model.bin": b"weights"}
    spec = _spec(
        tmp_path,
        payload={
            "command": ["python", "-m", "recipe"],
            "input_paths": {"training_material": "/kaggle/input/training_material"},
        },
    )
    cli = FakeCli(record=_record_for(spec, files), artifacts=files)
    transport = _transport(cli, staging)
    transport.run(spec, tmp_path / "attempt")

    kernel_dirs = [path for path in staging.iterdir() if path.is_dir()]
    assert len(kernel_dirs) == 1
    kernel_dir = kernel_dirs[0]
    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == f"nikma/{kernel_dir.name}"
    assert metadata["code_file"] == KERNEL_ENTRY_NAME
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == ["owner/dataset"]
    assert metadata["is_private"] is True and metadata["enable_gpu"] is True

    document = json.loads(
        (kernel_dir / KERNEL_JOB_SPEC_NAME).read_text(encoding="utf-8")
    )
    assert document["spec"]["source"]["commit_sha"] == COMMIT
    assert document["spec"]["inputs"][0]["sha256"]
    assert document["spec"]["mounts"] == ["owner/dataset"]
    assert document["kernel"]["command"] == ["python", "-m", "recipe"]
    assert COMMIT in document["kernel"]["install_spec"]
    assert document["kernel"]["input_paths"] == {
        "training_material": "/kaggle/input/training_material"
    }

    entry = (kernel_dir / KERNEL_ENTRY_NAME).read_text(encoding="utf-8")
    ast.parse(entry)
    assert "run_kernel_job" in entry
    assert "chowder-job-spec.json" in entry


def test_the_generated_entry_records_an_install_failure_with_the_declared_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one code path that runs before the package is installed must still
    write a record the backend can classify, not invent a run that happened."""
    output_dir = tmp_path / "kernel-working"
    spec = _spec(
        tmp_path,
        payload={"output_dir": str(output_dir)},
        resume_from="ckpt-0007",
    )
    staging = tmp_path / "staging"
    slug = kernel_slug(spec.spec_id, spec.source.commit_sha, "chowder")
    kernel_dir = staging / slug
    _transport(FakeCli(), staging)._stage(spec, kernel_dir, slug, f"nikma/{slug}")
    entry = kernel_dir / KERNEL_ENTRY_NAME

    def fail_install(command, **kwargs):
        assert "-m" in command and "pip" in command
        return _completed(1, "", "ERROR: network unreachable")

    monkeypatch.setattr(subprocess, "run", fail_install)
    namespace = {"__name__": "generated_kernel_entry", "__file__": str(entry)}
    exec(compile(entry.read_text(encoding="utf-8"), str(entry), "exec"), namespace)
    assert namespace["main"]() == 1

    record = json.loads((output_dir / "job-record.json").read_text(encoding="utf-8"))
    assert record["state"] == "error"
    assert "installing the pinned source failed" in record["error"]
    assert "network unreachable" in record["error"]
    assert record["spec_id"] == spec.spec_id
    assert record["mounts"] == list(spec.mounts)
    assert record["resume_from"] == "ckpt-0007"
    assert record["resume_state"] == "not-a-resume"
    assert record["source_commit_sha"] == "", "no run happened; no commit ran"


def test_run_pulls_artifacts_and_returns_the_kernel_record(tmp_path: Path) -> None:
    files = {"adapter/model.bin": b"weights", "adapter/config.json": b"{}"}
    spec = _spec(tmp_path)
    cli = FakeCli(record=_record_for(spec, files), artifacts=files)
    destination = tmp_path / "attempt"
    record = _transport(cli, tmp_path / "staging").run(spec, destination)

    assert record.state == "complete"
    assert record.job_id.startswith("nikma/chowder-")
    assert record.source_commit_sha == COMMIT
    assert record.input_verification == ((spec.inputs[0].name, spec.inputs[0].sha256),)
    assert [entry.path for entry in record.artifact_manifest] == [
        "adapter/config.json",
        "adapter/model.bin",
    ]
    assert (destination / "adapter/model.bin").read_bytes() == b"weights"
    assert not (destination / "job-record.json").exists(), (
        "the record is transported, not an artifact of the attempt"
    )
    assert not (destination / "kernel.log").exists(), "platform logs are not artifacts"
    raw = tmp_path / "attempt-kaggle-output"
    assert (raw / "job-record.json").is_file(), "the record stays available for the audit"


def test_a_terminal_failure_without_a_record_is_a_transport_error(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    cli = FakeCli(states=("error",), record=None)
    with pytest.raises(KaggleTransportError, match="without writing"):
        _transport(cli, tmp_path / "staging").run(spec, tmp_path / "attempt")


def test_the_poll_loop_times_out_under_the_declared_budget(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    cli = FakeCli(states=("running", "running", "running"), record=None)
    ticks = itertools.count()
    slept: list[float] = []
    transport = _transport(
        cli,
        tmp_path / "staging",
        clock=lambda: float(next(ticks)) * 1000.0,
        sleep=lambda seconds: slept.append(seconds),
    )
    with pytest.raises(KaggleTransportError, match="past timeout"):
        transport.run(spec, tmp_path / "attempt")
    assert slept == [5.0]


# --------------------------------------------------------------------------
# the backend on top of the transport
# --------------------------------------------------------------------------


def test_a_push_failure_is_classified_by_the_backend_as_infrastructure(
    tmp_path: Path,
) -> None:
    cli = FakeCli(push_error="quota exceeded")
    backend = KaggleComputeBackend(_transport(cli, tmp_path / "staging"), accelerator="T4x2")
    outcome = backend.dispatch(_request(tmp_path), destination=tmp_path / "attempt")
    assert outcome.status == OUTCOME_FAILED
    assert outcome.refusal_code == TRANSPORT_FAILED
    assert outcome.failure_class == "infrastructure"
    assert "quota exceeded" in outcome.refusal_reason


def test_a_kernel_error_record_is_classified_by_the_backend(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    files = {"adapter/model.bin": b"weights"}
    cli = FakeCli(
        states=("running", "error"),
        record=_record_for(spec, files, state="error", error="CUDA out of memory"),
        artifacts=files,
    )
    backend = KaggleComputeBackend(_transport(cli, tmp_path / "staging"), accelerator="T4x2")
    outcome = backend.dispatch(_request(tmp_path), destination=tmp_path / "attempt")
    assert outcome.status == OUTCOME_FAILED
    assert outcome.failure_class == "infrastructure"
    assert "CUDA out of memory" in outcome.refusal_reason


def test_an_install_failure_surfaces_as_the_failure_it_is(tmp_path: Path) -> None:
    """The install error is the refusal reason, not a derived mismatch, and the
    resumed attempt reports the vocabulary the search reads."""
    spec = _spec(tmp_path, resume_from="ckpt-0007")
    fallback = {
        "spec_id": spec.spec_id,
        "state": "error",
        "source_commit_sha": "",
        "input_verification": [],
        "artifact_manifest": [],
        "environment": {},
        "mounts": list(spec.mounts),
        "resume_from": "ckpt-0007",
        "resume_state": "not-a-resume",
        "command": None,
        "error": "installing the pinned source failed: ERROR: network unreachable",
    }
    cli = FakeCli(states=("running", "error"), record=fallback)
    backend = KaggleComputeBackend(_transport(cli, tmp_path / "staging"), accelerator="T4x2")
    outcome = backend.dispatch(
        _request(tmp_path, resume_from="ckpt-0007"), destination=tmp_path / "attempt"
    )
    assert outcome.status == OUTCOME_FAILED
    assert outcome.refusal_code == JOB_FAILED
    assert outcome.failure_class == "infrastructure"
    assert "installing the pinned source failed" in outcome.refusal_reason
    assert outcome.resume_state == "not-a-resume"
    assert outcome.cost is not None


def test_end_to_end_dispatch_through_the_cli_transport(tmp_path: Path) -> None:
    files = {"adapter/model.bin": b"weights"}
    spec = _spec(tmp_path)
    cli = FakeCli(
        quota_readings=(12.5, 12.5, 12.0),
        states=("queued", "running", "complete"),
        record=_record_for(spec, files),
        artifacts=files,
    )
    ticks = itertools.count()
    transport = _transport(cli, tmp_path / "staging", clock=lambda: float(next(ticks)))
    backend = KaggleComputeBackend(transport, accelerator="T4x2")
    destination = tmp_path / "attempt"
    outcome = backend.dispatch(_request(tmp_path), destination=destination)

    assert outcome.status == OUTCOME_SUCCEEDED, outcome.refusal_reason
    assert outcome.cost is not None
    assert outcome.cost.device_gpu_hours == pytest.approx(1.0)  # 0.5 session h x 2 devices
    assert outcome.settlement is not None and outcome.settlement.compliant
    assert (destination / "adapter/model.bin").is_file()
    assert (destination / "compute-job.json").is_file()
    evidence = outcome.to_evidence()
    assert evidence["artifact_manifest"][0]["path"] == "adapter/model.bin"
    assert evidence["compute_cost"]["device_measured"] is True
