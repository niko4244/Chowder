"""The campaign wiring, exercised offline end to end.

The prepared campaign is real (files on disk, the declared field set), the
backend is the real :class:`KaggleComputeBackend`, and only the Kaggle API is a
fake -- so a recipe's attempt is proven to travel from the prepared inputs
through binding, dispatch, settlement and evidence in one place.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from chowder.growth.campaign_prepare import (
    CONTAMINATION_MANIFEST_FIELD,
    PREPARED_INPUT_FIELDS,
    PreparedCampaign,
)
from chowder.growth.compute_backend import (
    DECLARED_INPUT_UNREADABLE,
    OUTCOME_FAILED,
    OUTCOME_REFUSED,
    OUTCOME_SUCCEEDED,
    SOURCE_BINDING_SCHEMA,
    ArtifactEntry,
    ComputeBackendRefusal,
    SourceBinding,
    bind_declared_inputs,
    file_digest,
)
from chowder.growth.cycle import select_candidate
from chowder.growth.kaggle_campaign import (
    KAGGLE_CAMPAIGN_BINDING,
    KAGGLE_CAMPAIGN_COMMAND_MISSING,
    KAGGLE_CAMPAIGN_EVIDENCE_FILE,
    KAGGLE_CAMPAIGN_INPUT_PATHS_INCOMPLETE,
    KAGGLE_CAMPAIGN_INPUTS_INCOMPLETE,
    KAGGLE_CAMPAIGN_INPUTS_UNDECLARED,
    KAGGLE_CAMPAIGN_MOUNTS_MISSING,
    KAGGLE_CAMPAIGN_SCHEMA,
    KaggleTrainingFn,
    build_attempt_request,
    declared_input_paths,
)
from chowder.growth.kaggle_inputs import KaggleInputPublisher
from chowder.growth.kaggle_compute import (
    KaggleComputeBackend,
    KaggleJobRecord,
    KaggleQuota,
    KaggleTransportError,
)
from chowder.growth.recipe_planner import TrainingRecipe
from chowder.growth.training_binding import GrowthEnvelope, check_growth_envelope

COMMIT = "a" * 40
MODEL_COMMIT = "c" * 40
DATASET = "nikma/chowder-prepared-abc123"
DECLARED_FIELDS = (*PREPARED_INPUT_FIELDS, CONTAMINATION_MANIFEST_FIELD)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _prepared(tmp_path: Path, *, drop=None, extra=None) -> PreparedCampaign:
    root = tmp_path / "prepared"
    root.mkdir()
    inputs: dict = {}
    for field_name in DECLARED_FIELDS:
        path = root / f"{field_name.replace('_', '-')}.json"
        path.write_text(json.dumps({"field": field_name}) + "\n", encoding="utf-8")
        inputs[field_name] = str(path)
    if drop is not None:
        inputs.pop(drop)
    if extra:
        inputs.update(extra)
    return PreparedCampaign(
        cycle_id="gen2",
        directory=root,
        inputs=inputs,
        recipe_ids=("recipe-01",),
    )


def _recipe(
    recipe_id: str = "recipe-01",
    *,
    device: float = 0.2,
    wall: float = 0.5,
    resume_from: str | None = None,
) -> TrainingRecipe:
    return TrainingRecipe(
        recipe_id=recipe_id,
        curriculum_item_ids=("item-1",),
        mixture={"TARGET": 1.0},
        learning_rate=1e-4,
        scheduler="cosine",
        warmup_steps=2,
        lora_rank=16,
        lora_alpha=32,
        target_modules=("q_proj", "v_proj"),
        seq_len=2048,
        batch_size=2,
        gradient_accumulation=4,
        max_steps=20,
        objective="sft",
        replay_rate=0.1,
        dataset_manifest={},
        projected_device_gpu_hours=device,
        projected_wall_gpu_hours=wall,
        resume_from_checkpoint=resume_from,
    )


def _envelope() -> GrowthEnvelope:
    return GrowthEnvelope(
        device_gpu_hours_ceiling=0.3,
        wall_gpu_hours_ceiling=0.75,
        project_gpu_hour_budget=1.5,
    )


def _source() -> SourceBinding:
    return SourceBinding(
        repository="https://github.com/niko4244/Chowder",
        commit_sha=COMMIT,
        chowder_version="0.5.0-dev",
        cycle_id="gen2",
        recipe_id="recipe-01",
        attempt_id="attempt-01",
    )


def _paths(prepared: PreparedCampaign) -> dict[str, tuple[str, str]]:
    return {
        name: (
            f"/kaggle/input/prepared-abc123/{Path(path).name}",
            f"/kaggle/input/nikma/prepared-abc123/{Path(path).name}",
        )
        for name, path in declared_input_paths(prepared).items()
    }


class FakeKernelTransport:
    """A Kaggle shaped like a fake: quota readings and one recorded attempt."""

    def __init__(
        self,
        *,
        remaining: float = 10.0,
        consume: float = 0.1,
        commit_echo: str = COMMIT,
        state: str = "complete",
        error: str = "",
        empty_evidence: bool = False,
        resume: str = "resumed",
        fail_with: Exception | None = None,
    ) -> None:
        self.remaining = remaining
        self.consume = consume
        self.commit_echo = commit_echo
        self.state = state
        self.error = error
        self.empty_evidence = empty_evidence
        self.resume = resume
        self.fail_with = fail_with
        self.specs: list = []

    def quota(self) -> KaggleQuota:
        return KaggleQuota(remaining_gpu_hours=self.remaining)

    def run(self, spec, destination):
        self.specs.append(spec)
        if self.fail_with is not None:
            raise self.fail_with
        files = {"adapter/model.bin": b"weights"}
        for relative, payload in files.items():
            target = Path(destination) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        if self.empty_evidence:
            verification: tuple = ()
            manifest: tuple = ()
            environment: dict = {}
        else:
            verification = tuple((entry.name, entry.sha256) for entry in spec.inputs)
            manifest = tuple(
                ArtifactEntry(relative, *file_digest(Path(destination) / relative))
                for relative in sorted(files)
            )
            environment = {
                "python_version": "3.12.4",
                "packages": {"torch": "2.4.0"},
                "model_commit": spec.payload.get("model_commit", MODEL_COMMIT),
            }
        resume_state = None
        resume_from = None
        if spec.resume_from is not None:
            resume_state = self.resume
            resume_from = spec.resume_from
        record = KaggleJobRecord(
            job_id=f"nikma/kernel-{len(self.specs):04d}",
            state=self.state,
            wall_seconds=600.0,
            accelerator=spec.accelerator,
            source_commit_sha=self.commit_echo,
            input_verification=verification,
            artifact_manifest=manifest,
            environment=environment,
            resume_state=resume_state,
            resume_from=resume_from,
            mounts=tuple(spec.mounts),
            error=self.error,
        )
        self.remaining = max(0.0, self.remaining - self.consume)
        return record


def _adapter(
    transport: FakeKernelTransport, prepared: PreparedCampaign, **overrides
) -> KaggleTrainingFn:
    fields: dict = {
        "backend": KaggleComputeBackend(transport, accelerator="T4x2"),
        "prepared": prepared,
        "envelope": _envelope(),
        "repository": "https://github.com/niko4244/Chowder",
        "commit_sha": COMMIT,
        "chowder_version": "0.5.0-dev",
        "entry_point": "kaggle/kernel_entry.py",
        "command": ["python", "-m", "chowder", "train"],
        "input_paths": _paths(prepared),
        "mounts": (DATASET,),
        "attempts_root": prepared.directory.parent / "kaggle-attempts",
        "timeout_seconds": 7200.0,
        "model_commit": MODEL_COMMIT,
    }
    fields.update(overrides)
    return KaggleTrainingFn(**fields)


# --------------------------------------------------------------------------
# the prepared manifest -> AttemptRequest builder
# --------------------------------------------------------------------------


def test_the_builder_binds_every_declared_input_and_carries_the_declaration(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    request = build_attempt_request(
        prepared,
        source=_source(),
        recipe=_recipe(),
        envelope=_envelope(),
        entry_point="kaggle/kernel_entry.py",
        command=["python", "-m", "chowder", "train"],
        input_paths=_paths(prepared),
        mounts=(DATASET,),
        timeout_seconds=600.0,
        model_commit=MODEL_COMMIT,
    )

    assert [entry.name for entry in request.inputs] == sorted(DECLARED_FIELDS)
    assert request.payload["command"] == ["python", "-m", "chowder", "train"]
    assert request.payload["input_paths"]["training_material_path"] == list(
        _paths(prepared)["training_material_path"]
    )
    assert request.payload["pip_extras"] == ["train"]
    assert request.payload["model_commit"] == MODEL_COMMIT
    assert request.projected_cost.device_gpu_hours == pytest.approx(0.2)
    assert request.projected_cost.wall_gpu_hours == pytest.approx(0.5)
    assert request.device_ceiling == 0.3
    assert request.wall_ceiling == 0.75
    assert request.project_budget_wall_gpu_hours == 1.5
    assert request.mounts == (DATASET,)
    assert request.resume_from is None
    assert request.timeout_seconds == 600.0


def test_a_prepared_campaign_missing_a_declared_input_refuses(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path, drop=CONTAMINATION_MANIFEST_FIELD)
    with pytest.raises(ComputeBackendRefusal, match=KAGGLE_CAMPAIGN_INPUTS_INCOMPLETE):
        build_attempt_request(
            prepared,
            source=_source(),
            recipe=_recipe(),
            envelope=_envelope(),
            entry_point="kaggle/kernel_entry.py",
            command=["python", "-m", "chowder", "train"],
            input_paths=_paths(prepared) | {CONTAMINATION_MANIFEST_FIELD: ["/x"]},
            mounts=(DATASET,),
            timeout_seconds=600.0,
        )


def test_a_prepared_campaign_with_an_unknown_input_refuses(tmp_path: Path) -> None:
    prepared = _prepared(
        tmp_path, extra={"surprise_path": str(tmp_path / "surprise.json")}
    )
    with pytest.raises(ComputeBackendRefusal, match=KAGGLE_CAMPAIGN_INPUTS_UNDECLARED):
        build_attempt_request(
            prepared,
            source=_source(),
            recipe=_recipe(),
            envelope=_envelope(),
            entry_point="kaggle/kernel_entry.py",
            command=["python", "-m", "chowder", "train"],
            input_paths=_paths(prepared),
            mounts=(DATASET,),
            timeout_seconds=600.0,
        )


def test_a_builder_without_a_declared_command_refuses(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    for command in ([], ["python", "  "], "python -m chowder"):
        with pytest.raises(ComputeBackendRefusal, match=KAGGLE_CAMPAIGN_COMMAND_MISSING):
            build_attempt_request(
                prepared,
                source=_source(),
                recipe=_recipe(),
                envelope=_envelope(),
                entry_point="kaggle/kernel_entry.py",
                command=command,
                input_paths=_paths(prepared),
                mounts=(DATASET,),
                timeout_seconds=600.0,
            )


def test_a_builder_without_a_mount_refuses(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    for mounts in ((), ("",)):
        with pytest.raises(ComputeBackendRefusal, match=KAGGLE_CAMPAIGN_MOUNTS_MISSING):
            build_attempt_request(
                prepared,
                source=_source(),
                recipe=_recipe(),
                envelope=_envelope(),
                entry_point="kaggle/kernel_entry.py",
                command=["python", "-m", "chowder", "train"],
                input_paths=_paths(prepared),
                mounts=mounts,
                timeout_seconds=600.0,
            )


def test_input_paths_must_cover_the_declared_inputs_exactly(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    missing = dict(_paths(prepared))
    missing.pop("training_material_path")
    undeclared = dict(_paths(prepared))
    undeclared["training_material"] = ("/kaggle/input/prepared-abc123/x",)
    for paths in (missing, undeclared):
        with pytest.raises(
            ComputeBackendRefusal, match=KAGGLE_CAMPAIGN_INPUT_PATHS_INCOMPLETE
        ):
            build_attempt_request(
                prepared,
                source=_source(),
                recipe=_recipe(),
                envelope=_envelope(),
                entry_point="kaggle/kernel_entry.py",
                command=["python", "-m", "chowder", "train"],
                input_paths=paths,
                mounts=(DATASET,),
                timeout_seconds=600.0,
            )


def test_a_relative_kernel_path_refuses(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    paths = dict(_paths(prepared))
    paths["data_registry_path"] = ("relative/data-registry.json",)
    with pytest.raises(ComputeBackendRefusal, match=KAGGLE_CAMPAIGN_SCHEMA):
        build_attempt_request(
            prepared,
            source=_source(),
            recipe=_recipe(),
            envelope=_envelope(),
            entry_point="kaggle/kernel_entry.py",
            command=["python", "-m", "chowder", "train"],
            input_paths=paths,
            mounts=(DATASET,),
            timeout_seconds=600.0,
        )


def test_a_declared_input_that_cannot_be_read_refuses(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    Path(prepared.inputs["hardware_budget_path"]).unlink()
    with pytest.raises(ComputeBackendRefusal, match=DECLARED_INPUT_UNREADABLE):
        build_attempt_request(
            prepared,
            source=_source(),
            recipe=_recipe(),
            envelope=_envelope(),
            entry_point="kaggle/kernel_entry.py",
            command=["python", "-m", "chowder", "train"],
            input_paths=_paths(prepared),
            mounts=(DATASET,),
            timeout_seconds=600.0,
        )


# --------------------------------------------------------------------------
# the TrainingFn adapter
# --------------------------------------------------------------------------


def test_an_attempt_dispatches_and_returns_selectable_evidence(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    transport = FakeKernelTransport()
    adapter = _adapter(transport, prepared)

    evidence = adapter(_recipe(), ())

    assert evidence["status"] == OUTCOME_SUCCEEDED
    assert evidence["candidate_succeeded"] is True
    assert evidence["binding"] == KAGGLE_CAMPAIGN_BINDING
    assert evidence["backend"] == "kaggle"
    assert evidence["attempt"] == "attempt-01"
    assert evidence["artifact_ref"] == "adapter/model.bin"
    assert evidence["job_id"] == "nikma/kernel-0001"
    assert evidence["source_identity"]["verified"] is True
    assert evidence["source_identity"]["observed_commit_sha"] == COMMIT
    assert len(evidence["declared_inputs"]) == len(DECLARED_FIELDS)
    assert evidence["input_paths"]["training_material_path"] == list(
        _paths(prepared)["training_material_path"]
    )
    assert evidence["actual_cost"]["device_measured"] is True
    assert evidence["actual_cost"]["device_gpu_hours"] == pytest.approx(0.2)
    assert evidence["measured_gpu_hours"] == pytest.approx(600.0 / 3600.0)

    selected = select_candidate([evidence])
    assert selected is not None and selected["attempt"] == "attempt-01"

    evidence_path = Path(evidence["attempt_dir"]) / KAGGLE_CAMPAIGN_EVIDENCE_FILE
    assert json.loads(evidence_path.read_text(encoding="utf-8"))["attempt"] == "attempt-01"

    spec = transport.specs[0]
    assert spec.accelerator == "T4x2"
    assert spec.payload["command"] == ["python", "-m", "chowder", "train"]
    assert spec.payload["input_paths"]["training_material_path"] == list(
        _paths(prepared)["training_material_path"]
    )


def test_every_attempt_gets_a_fresh_never_reused_directory(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    transport = FakeKernelTransport()
    adapter = _adapter(transport, prepared)

    first = adapter(_recipe(), ())
    second = adapter(_recipe("recipe-02"), ())

    assert first["attempt"] == "attempt-01"
    assert second["attempt"] == "attempt-02"
    assert first["attempt_dir"] != second["attempt_dir"]
    for evidence in (first, second):
        path = Path(evidence["attempt_dir"]) / KAGGLE_CAMPAIGN_EVIDENCE_FILE
        assert path.is_file()


def test_a_resume_declared_by_the_search_reaches_the_attempt_and_its_evidence(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    transport = FakeKernelTransport(resume="resumed")
    evidence = _adapter(transport, prepared)(
        _recipe(resume_from="ckpt-0007"), ()
    )

    assert transport.specs[0].resume_from == "ckpt-0007"
    assert evidence["declared_resume_from"] == "ckpt-0007"
    assert evidence["resume_state"] == "resumed"
    assert evidence["status"] == OUTCOME_SUCCEEDED


def test_an_install_failure_does_not_claim_its_source_was_verified(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    transport = FakeKernelTransport(
        commit_echo="",
        state="error",
        error="installing the pinned source failed: network unreachable",
        empty_evidence=True,
    )
    evidence = _adapter(transport, prepared)(_recipe(), ())

    assert evidence["status"] == OUTCOME_FAILED
    assert evidence["source_identity"]["verified"] is False
    assert evidence["source_identity"]["observed_commit_sha"] is None
    assert "installing the pinned source failed" in evidence["refusal_reason"]
    assert evidence["failure_reason"] is not None


def test_a_transport_failure_is_a_failed_attempt_without_a_source_claim(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    transport = FakeKernelTransport(
        fail_with=KaggleTransportError("the push was rejected by Kaggle")
    )
    evidence = _adapter(transport, prepared)(_recipe(), ())

    assert evidence["status"] == OUTCOME_FAILED
    assert evidence["source_identity"]["verified"] is False
    assert evidence["source_identity"]["job_id"] == ""
    assert "no kernel record" in evidence["source_identity"]["reason"]
    assert "push was rejected" in evidence["refusal_reason"]


def test_a_builder_refusal_is_recorded_not_raised(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    transport = FakeKernelTransport()
    adapter = _adapter(transport, prepared)
    Path(prepared.inputs["training_material_path"]).unlink()

    evidence = adapter(_recipe(), ())

    assert evidence["status"] == OUTCOME_REFUSED
    assert evidence["refused_by"] == DECLARED_INPUT_UNREADABLE
    assert transport.specs == [], "nothing may be dispatched when an input vanished"
    path = Path(evidence["attempt_dir"]) / KAGGLE_CAMPAIGN_EVIDENCE_FILE
    assert json.loads(path.read_text(encoding="utf-8"))["refused_by"] == (
        DECLARED_INPUT_UNREADABLE
    )


def test_admission_is_the_shared_growth_envelope_rule(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    adapter = _adapter(FakeKernelTransport(), prepared)

    admitted = _recipe(device=0.3, wall=0.75)
    assert adapter.admit(admitted) is None
    over_device = _recipe(device=0.4, wall=0.5)
    over_wall = _recipe(device=0.2, wall=0.9)
    for recipe in (over_device, over_wall):
        refusal = adapter.admit(recipe)
        assert refusal is not None and refusal[0] == "growth-envelope"
        assert refusal == check_growth_envelope(recipe, _envelope()), (
            "the remote executor must admit exactly the recipes the local one does"
        )


def test_the_published_dataset_paths_carry_exactly_the_declared_bytes(
    tmp_path: Path,
) -> None:
    """The publisher stages the declared bytes; the builder's candidate paths
    name those staged files; the kernel resolves them by digest. This closes
    the loop across the two modules without a Kaggle account."""
    prepared = _prepared(tmp_path)
    calls: list[list[str]] = []

    def runner(argv):
        calls.append([str(part) for part in argv])
        return subprocess.CompletedProcess(list(argv), 0, stdout="created\n", stderr="")

    publisher = KaggleInputPublisher(
        owner="nikma", runner=runner, staging_root=tmp_path / "staging"
    )
    dataset = publisher.publish(
        bind_declared_inputs(declared_input_paths(prepared)),
        dataset="chowder-prepared",
        title="Chowder prepared inputs",
    )
    staged = Path(calls[0][calls[0].index("-p") + 1])

    class MountCheckingTransport(FakeKernelTransport):
        def run(self, spec, destination):
            for entry in spec.inputs:
                candidates = spec.payload["input_paths"][entry.name]
                filename = dataset.files[entry.name]
                assert candidates[0] == f"/kaggle/input/{dataset.slug}/{filename}"
                assert candidates[1] == (
                    f"/kaggle/input/nikma/{dataset.slug}/{filename}"
                ), "the mount surface names the published dataset"
                assert file_digest(staged / filename) == (
                    entry.sha256,
                    entry.bytes,
                ), "the published bytes are the declared bytes"
            return super().run(spec, destination)

    transport = MountCheckingTransport()
    adapter = _adapter(
        transport,
        prepared,
        input_paths=dataset.input_paths,
        mounts=(dataset.reference,),
    )
    evidence = adapter(_recipe(), ())
    assert evidence["status"] == OUTCOME_SUCCEEDED
    assert evidence["requested_mounts"] == [dataset.reference]
    assert evidence["input_paths"] == {
        name: list(candidates) for name, candidates in dataset.input_paths.items()
    }


def test_a_misconfigured_binding_refuses_before_any_phase(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    with pytest.raises(ComputeBackendRefusal, match=KAGGLE_CAMPAIGN_INPUT_PATHS_INCOMPLETE):
        _adapter(
            FakeKernelTransport(),
            prepared,
            input_paths={
                name: paths
                for name, paths in _paths(prepared).items()
                if name != "data_registry_path"
            },
        )
    with pytest.raises(ComputeBackendRefusal, match=KAGGLE_CAMPAIGN_MOUNTS_MISSING):
        _adapter(FakeKernelTransport(), prepared, mounts=())
    with pytest.raises(ComputeBackendRefusal, match=SOURCE_BINDING_SCHEMA):
        _adapter(FakeKernelTransport(), prepared, commit_sha="not-a-commit")
