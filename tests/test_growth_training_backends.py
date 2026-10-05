"""The declared training backends, tested with no GPU and no network.

The whole point of the unified surface is that admission, the panel, the
capability matrix and the uniform outcome are provable *before* any real
compute exists. Every device fact here is injected (a fake framework module,
injected memory/disk readers, an injected remote backend), so the contract is
tested on a laptop and a real CUDA/Kaggle golden path stays a separate
verification step.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from chowder.growth import training_backends as backends
from chowder.growth.attempt_failure import FailureClass
from chowder.growth.campaign import (
    CampaignBudget,
    CampaignManifest,
    CampaignManifestError,
)
from chowder.backends.unsloth_peft import UnslothConfigError, UnslothPeftRunSpec
from chowder.growth.campaign_runner import (
    FIELD_ENFORCEMENT,
    CampaignRunRefusal,
    _attempt_summary,
    assert_every_field_enforced,
    build_executor_with_selection,
    envelope_for,
)
from chowder.growth.compute_backend import OUTCOME_FAILED, OUTCOME_SUCCEEDED, SourceBinding
from chowder.growth.kaggle_campaign import DECLARED_PREPARED_FIELDS, KaggleTrainingFn
from chowder.growth.recipe_planner import TrainingRecipe
from chowder.growth.training_backends import (
    CAPABILITY_CAPABILITY_DEPENDENT,
    CAPABILITY_DATA_LAYER,
    CAPABILITY_MODEL_HARDWARE_DEPENDENT,
    CAPABILITY_REFUSED,
    CAPABILITY_SUPPORTED,
    KAGGLE_BACKEND_UNAVAILABLE,
    KAGGLE_CONFIG_INCOMPLETE,
    STRUCTURAL_PREFLIGHT_CODES,
    TRAINING_BACKEND_AUTO_UNRESOLVED,
    TRAINING_BACKEND_EVIDENCE_MISSING,
    TRAINING_BACKEND_EVIDENCE_STATUS_UNKNOWN,
    TRAINING_BACKEND_NO_DEVICE,
    TRAINING_BACKEND_TEMPLATE_UNDECLARED,
    TRAINING_BACKEND_TRAINER_MISMATCH,
    TRAINING_BACKEND_UNSUPPORTED_CAPABILITY,
    TRAINING_BACKEND_UNSUPPORTED_CONFIG,
    TRAINING_BACKEND_UNKNOWN_PROVIDER,
    CapabilityRow,
    KaggleTrainingBackend,
    PreflightPanel,
    PreflightResult,
    LocalTrainingBackend,
    OverheadReport,
    TrainingBackendDeclaration,
    preflight_report,
    TrainingBackendRefusal,
    UnslothTrainingBackend,
    choose_auto_backend,
    probe_local_panel,
    resolve_training_backend,
    uniform_outcome_from_evidence,
)

COMMIT = "a" * 40
GIB = 1024**3

MINIMAL_DOCUMENT: dict[str, Any] = {
    "cycle_id": "gen2",
    "parent_version": "gen1",
    "base_model_path": "model",
    "base_model_digest": "b" * 64,
    "state_root": "state",
    "target_benchmarks": ["small-bench@1"],
    "protected_benchmarks": [],
    "broad_benchmarks": [],
    "calibration_benchmarks": [],
    "reliability_benchmarks": [],
    "budget": {
        "device_gpu_hours_ceiling_per_recipe": 4.0,
        "wall_gpu_hours_ceiling_per_recipe": 12.0,
        "device_gpu_hours_ceiling_campaign": 20.0,
        "wall_gpu_hours_ceiling_campaign": 60.0,
    },
    "recipes": ["recipe-01"],
    "candidate_selection_policy": "first_successful",
}


# --------------------------------------------------------------------------
# injected device facts
# --------------------------------------------------------------------------


class _FakeDeviceProperties:
    def __init__(self, name: str, total_memory: int) -> None:
        self.name = name
        self.total_memory = total_memory
        self.major = 8
        self.minor = 9


class _FakeCuda:
    def __init__(self, devices: Sequence[_FakeDeviceProperties]) -> None:
        self._devices = list(devices)

    def is_available(self) -> bool:
        return bool(self._devices)

    def device_count(self) -> int:
        return len(self._devices)

    def get_device_properties(self, index: int) -> _FakeDeviceProperties:
        return self._devices[index]


class _FakeTorchVersion:
    cuda = "12.4"


class _FakeTorch:
    __version__ = "2.7.0+fake"
    version = _FakeTorchVersion()

    def __init__(self, devices: Sequence[_FakeDeviceProperties] = ()) -> None:
        self.cuda = _FakeCuda(devices)


def _a100() -> _FakeDeviceProperties:
    return _FakeDeviceProperties("Fake A100", 24 * GIB)


def _readers() -> dict[str, Any]:
    return {
        "memory_reader": lambda: (32 * GIB, 16 * GIB),
        "disk_reader": lambda path: (str(path or "."), 500 * GIB),
    }


def _panel_with_device(*, devices: Sequence[_FakeDeviceProperties] | None = None):
    return probe_local_panel(
        torch_module=_FakeTorch([_a100()] if devices is None else devices),
        **_readers(),
    )


def _panel_without_device():
    return probe_local_panel(torch_module=_FakeTorch([]), **_readers())


def _probe(panel):
    def probe(*, disk_path=None):
        return panel

    return probe


# --------------------------------------------------------------------------
# declared material
# --------------------------------------------------------------------------


def _recipe(**overrides: Any) -> TrainingRecipe:
    fields: dict[str, Any] = {
        "recipe_id": "recipe-01",
        "curriculum_item_ids": ("item-01",),
        "mixture": {"target": 1.0},
        "learning_rate": 1e-4,
        "scheduler": "cosine",
        "warmup_steps": 10,
        "lora_rank": 16,
        "lora_alpha": 32,
        "target_modules": ("q_proj", "v_proj"),
        "seq_len": 512,
        "batch_size": 2,
        "gradient_accumulation": 4,
        "max_steps": 100,
        "objective": "sft",
        "replay_rate": 0.1,
        "dataset_manifest": {},
        "projected_device_gpu_hours": 1.0,
        "projected_wall_gpu_hours": 3.5,
    }
    fields.update(overrides)
    return TrainingRecipe(**fields)


def _budget(**overrides: Any) -> CampaignBudget:
    fields: dict[str, Any] = {
        "device_gpu_hours_ceiling_per_recipe": 10.0,
        "wall_gpu_hours_ceiling_per_recipe": 10.0,
        "device_gpu_hours_ceiling_campaign": 20.0,
        "wall_gpu_hours_ceiling_campaign": 20.0,
    }
    fields.update(overrides)
    return CampaignBudget(**fields)


def _write_model(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 172,
                "vocab_size": 100,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    return directory


def _manifest(tmp_path: Path, **overrides: Any) -> CampaignManifest:
    fields: dict[str, Any] = {
        "cycle_id": "gen2",
        "parent_version": "gen1",
        "base_model_path": str(_write_model(tmp_path / "model")),
        "base_model_digest": "b" * 64,
        "parent_adapter_path": "",
        "parent_adapter_digest": "",
        "state_root": str(tmp_path / "state"),
        "target_benchmarks": ("small-bench@1",),
        "protected_benchmarks": (),
        "broad_benchmarks": (),
        "calibration_benchmarks": (),
        "reliability_benchmarks": (),
        "budget": _budget(),
        "recipe_ids": ("recipe-01",),
        "candidate_selection_policy": "first_successful",
    }
    fields.update(overrides)
    return CampaignManifest(**fields)


def _template(
    tmp_path: Path,
    *,
    backend_type: str = "peft",
    engine: str = "transformers",
    quantization: str | None = None,
    training: Mapping[str, Any] | None = None,
) -> str:
    backend: dict[str, Any] = {"type": backend_type, "engine": engine}
    if quantization:
        backend["quantization"] = quantization
    if training is not None:
        backend["training"] = dict(training)
    path = tmp_path / f"template-{backend_type}-{engine}.json"
    path.write_text(json.dumps({"backend": backend}), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------
# the declaration
# --------------------------------------------------------------------------


def test_an_undeclared_campaign_is_local_and_unknown_providers_refuse():
    assert TrainingBackendDeclaration().provider == backends.PROVIDER_LOCAL
    assert TrainingBackendDeclaration.from_mapping(None).provider == "local"
    with pytest.raises(TrainingBackendRefusal) as refusal:
        TrainingBackendDeclaration.from_mapping({"provider": "wandb"})
    assert refusal.value.code == TRAINING_BACKEND_UNKNOWN_PROVIDER


def test_a_provider_specific_config_key_nothing_reads_refuses():
    with pytest.raises(TrainingBackendRefusal) as refusal:
        TrainingBackendDeclaration.from_mapping(
            {"provider": "local", "config": {"gpu": "a100"}}
        )
    assert refusal.value.code == TRAINING_BACKEND_UNSUPPORTED_CONFIG
    with pytest.raises(TrainingBackendRefusal):
        TrainingBackendDeclaration.from_mapping(
            {"provider": "kaggle", "config": {"prepare": "x"}}
        )


def test_unknown_declaration_keys_refuse_beside_the_provider():
    with pytest.raises(TrainingBackendRefusal) as refusal:
        TrainingBackendDeclaration.from_mapping({"provider": "local", "device": "cpu"})
    assert refusal.value.code == backends.TRAINING_BACKEND_SCHEMA


def test_auto_records_its_candidate_order_and_refuses_an_empty_one():
    declaration = TrainingBackendDeclaration.from_mapping(
        {"provider": "auto", "config": {"candidates": ["kaggle", "local", "kaggle"]}}
    )
    assert declaration.candidate_order() == ("kaggle", "local")
    default = TrainingBackendDeclaration.from_mapping({"provider": "auto"})
    assert default.candidate_order() == backends.AUTO_DEFAULT_CANDIDATES
    with pytest.raises(TrainingBackendRefusal):
        TrainingBackendDeclaration.from_mapping(
            {"provider": "auto", "config": {"candidates": []}}
        )
    with pytest.raises(TrainingBackendRefusal) as refusal:
        TrainingBackendDeclaration.from_mapping(
            {"provider": "auto", "config": {"candidates": ["auto"]}}
        )
    assert refusal.value.code == TRAINING_BACKEND_UNKNOWN_PROVIDER


def test_the_manifest_declares_the_backend_and_the_enforcement_table_covers_it():
    assert_every_field_enforced()
    assert "training_backend" in FIELD_ENFORCEMENT
    assert "training_backend" in CampaignManifest.__dataclass_fields__
    manifest = CampaignManifest.from_mapping(
        {**MINIMAL_DOCUMENT, "training_backend": {"provider": "kaggle", "config": {"owner": "me"}}}
    )
    assert manifest.training_backend.provider == "kaggle"
    assert manifest.training_backend.config == {"owner": "me"}
    # A manifest predating the field runs exactly as it did before: local.
    assert CampaignManifest.from_mapping(dict(MINIMAL_DOCUMENT)).training_backend.provider == "local"


def test_a_provider_the_manifest_cannot_honor_refuses_at_load():
    with pytest.raises(CampaignManifestError) as refusal:
        CampaignManifest.from_mapping(
            {**MINIMAL_DOCUMENT, "training_backend": {"provider": "wandb"}}
        )
    assert TRAINING_BACKEND_UNKNOWN_PROVIDER in str(refusal.value)


# --------------------------------------------------------------------------
# the panel
# --------------------------------------------------------------------------


def test_the_panel_reports_devices_ram_disk_and_runtime_from_injected_probes():
    panel = _panel_with_device()
    assert panel.device_count == 1
    assert panel.devices[0].name == "Fake A100"
    assert panel.devices[0].compute_capability == "8.9"
    assert panel.devices[0].vram_gb == 24.0
    assert panel.total_vram_gb == 24.0
    assert panel.system_ram_gb == 32.0
    assert panel.available_ram_gb == 16.0
    assert panel.disk_free_gb == 500.0
    assert panel.cuda_available is True
    assert panel.cuda_runtime == "12.4"
    assert panel.framework_version == "2.7.0+fake"
    # It is a device probe, and it says so rather than implying step timings.
    assert "not a step-timing measurement" in panel.measurement_method
    assert panel.warnings == ()
    assert panel.to_dict()["device_count"] == 1


def test_a_machine_without_a_framework_reports_that_rather_than_guessing():
    panel = probe_local_panel(
        torch_module=None, memory_reader=lambda: (None, None), disk_reader=lambda path: (None, None)
    )
    assert panel.device_count == 0
    assert panel.cuda_available is False
    assert panel.system_ram_gb is None
    assert panel.disk_free_gb is None
    assert any("framework is not importable" in warning for warning in panel.warnings)


# --------------------------------------------------------------------------
# LOCAL
# --------------------------------------------------------------------------


def test_local_preflight_refuses_without_a_device_and_honors_a_cpu_override(tmp_path):
    manifest = _manifest(tmp_path)
    backend = LocalTrainingBackend(probe=_probe(_panel_without_device()))
    refused = backend.preflight(manifest)
    assert refused.admitted is False
    assert refused.code == TRAINING_BACKEND_NO_DEVICE
    assert "device=cpu" in refused.reason

    overridden = LocalTrainingBackend(probe=_probe(_panel_without_device()))
    declaration = TrainingBackendDeclaration.from_mapping(
        {"provider": "local", "config": {"device": "cpu"}}
    )
    result = overridden.preflight(
        _manifest(tmp_path, training_backend=declaration, project_template_path=_template(tmp_path))
    )
    assert result.admitted is True
    assert result.overrides == ("device=cpu",)
    assert result.refusal() is None


def test_local_preflight_admits_a_visible_device_and_reports_the_declared_strategy(tmp_path):
    template = _template(tmp_path, engine="transformers", quantization="4bit")
    manifest = _manifest(tmp_path, project_template_path=template)
    result = LocalTrainingBackend(probe=_probe(_panel_with_device())).preflight(manifest)
    assert result.admitted is True
    assert result.code is None
    assert result.strategy == "peft/transformers/4bit"
    assert result.panel.device_count == 1
    assert result.to_dict()["admitted"] is True


def test_local_estimate_derives_memory_from_the_declared_model_config(tmp_path):
    manifest = _manifest(tmp_path, project_template_path=_template(tmp_path))
    backend = LocalTrainingBackend(probe=_probe(_panel_with_device()))
    recipe = _recipe()
    estimate = backend.estimate(manifest, recipe)
    assert estimate.available is True
    # 100*64 vocab + 2 layers x (4*64^2 attention + 2*64*172 gated MLP) params
    parameters = 100 * 64 + 2 * (4 * 64 * 64 + 2 * 64 * 172)
    assert estimate.model_bytes == parameters * 2  # bf16
    assert estimate.adapter_bytes > 0
    assert estimate.optimizer_bytes > 0
    assert estimate.activation_bytes > 0
    assert estimate.total_bytes == (
        estimate.model_bytes
        + estimate.adapter_bytes
        + estimate.optimizer_bytes
        + estimate.activation_bytes
    )
    assert estimate.total_vram_bytes == 24 * GIB
    assert estimate.requires_offload is False
    assert estimate.safe_sequence_length in backends._SAFE_SEQUENCE_LENGTHS
    assert estimate.assumptions


def test_a_quantized_recipe_reports_qlora_and_a_smaller_base(tmp_path):
    manifest = _manifest(
        tmp_path, project_template_path=_template(tmp_path, quantization="4bit")
    )
    backend = LocalTrainingBackend(probe=_probe(_panel_with_device()))
    estimate = backend.estimate(manifest, _recipe())
    assert "qlora" in estimate.strategy
    parameters = 100 * 64 + 2 * (4 * 64 * 64 + 2 * 64 * 172)
    assert estimate.model_bytes == int(parameters * 0.5)  # 4-bit base weights


def test_local_estimate_refuses_to_invent_numbers_without_a_model_config(tmp_path):
    manifest = _manifest(tmp_path, base_model_path=str(tmp_path / "empty-model"))
    Path(manifest.base_model_path).mkdir(parents=True, exist_ok=True)
    estimate = LocalTrainingBackend(probe=_probe(_panel_with_device())).estimate(
        manifest, _recipe()
    )
    assert estimate.available is False
    assert "config.json" in estimate.reason
    assert estimate.model_bytes is None


def test_local_admission_is_the_growth_envelope(tmp_path):
    backend = LocalTrainingBackend(probe=_probe(_panel_with_device()))
    recipe = _recipe()
    generous = _manifest(tmp_path)
    assert backend.admit(generous, recipe) is None
    tight = _manifest(tmp_path, budget=_budget(
        device_gpu_hours_ceiling_per_recipe=0.5,
        wall_gpu_hours_ceiling_per_recipe=0.5,
        device_gpu_hours_ceiling_campaign=1.0,
        wall_gpu_hours_ceiling_campaign=1.0,
    ))
    refusal = backend.admit(tight, recipe)
    assert refusal is not None
    assert refusal[0] == "growth-envelope"
    assert envelope_for(tight).device_gpu_hours_ceiling == 0.5


def test_local_resume_requires_a_checkpoint_that_exists(tmp_path):
    backend = LocalTrainingBackend(probe=_probe(_panel_with_device()))
    manifest = _manifest(tmp_path)
    with pytest.raises(TrainingBackendRefusal) as refusal:
        backend.resume(manifest, _recipe(), tmp_path / "absent")
    assert refusal.value.code == backends.TRAINING_BACKEND_RESUME_CHECKPOINT_MISSING
    checkpoint = tmp_path / "checkpoint-50"
    checkpoint.mkdir()
    resumed = backend.resume(manifest, _recipe(), checkpoint)
    assert resumed.resume_from_checkpoint == str(checkpoint)
    assert resumed.recipe_id == "recipe-01"


def test_local_capabilities_declare_the_mechanisms_it_really_runs():
    capabilities = LocalTrainingBackend(probe=_probe(_panel_with_device())).capabilities()
    assert capabilities.status("qlora") == CAPABILITY_SUPPORTED
    assert capabilities.status("checkpoint_resume") == CAPABILITY_SUPPORTED
    assert capabilities.status("optimizer_tiering") == CAPABILITY_SUPPORTED
    assert capabilities.refused_rows() == ()
    assert capabilities.to_dict()["provider"] == "local"


# --------------------------------------------------------------------------
# UNSLOTH
# --------------------------------------------------------------------------


def test_unsloth_refuses_a_template_that_names_another_trainer(tmp_path):
    manifest = _manifest(
        tmp_path, project_template_path=_template(tmp_path, engine="transformers")
    )
    result = UnslothTrainingBackend(probe=_probe(_panel_with_device())).preflight(manifest)
    assert result.admitted is False
    assert result.code == TRAINING_BACKEND_TRAINER_MISMATCH


def test_unsloth_refuses_a_declaration_with_no_template_to_confirm(tmp_path):
    result = UnslothTrainingBackend(probe=_probe(_panel_with_device())).preflight(
        _manifest(tmp_path)
    )
    assert result.admitted is False
    assert result.code == TRAINING_BACKEND_TEMPLATE_UNDECLARED


def test_unsloth_refuses_a_knob_the_isolated_engine_does_not_support(tmp_path):
    template = _template(
        tmp_path, engine="unsloth", training={"activation_offload": "always"}
    )
    manifest = _manifest(tmp_path, project_template_path=template)
    result = UnslothTrainingBackend(probe=_probe(_panel_with_device())).preflight(manifest)
    assert result.admitted is False
    assert result.code == TRAINING_BACKEND_UNSUPPORTED_CAPABILITY
    assert "activation_offload" in result.reason


def test_unsloth_admits_the_declared_engine_with_a_device_and_refuses_cpu(tmp_path):
    template = _template(tmp_path, engine="unsloth", quantization="4bit")
    manifest = _manifest(tmp_path, project_template_path=template)
    backend = UnslothTrainingBackend(probe=_probe(_panel_with_device()))
    result = backend.preflight(manifest)
    assert result.admitted is True
    assert result.strategy == "peft/unsloth/4bit"

    cpu = TrainingBackendDeclaration.from_mapping(
        {"provider": "unsloth", "config": {"device": "cpu"}}
    )
    refused = backend.preflight(_manifest(tmp_path, project_template_path=template, training_backend=cpu))
    assert refused.admitted is False
    assert refused.code == TRAINING_BACKEND_NO_DEVICE


def test_the_unsloth_matrix_names_where_each_decision_lives():
    capabilities = UnslothTrainingBackend().capabilities()
    for knob in backends.UNSLOTH_REFUSED_KNOBS:
        assert capabilities.status(knob) == CAPABILITY_REFUSED
        refusal = capabilities.refusal(knob)
        assert refusal is not None
        assert refusal[0] == TRAINING_BACKEND_UNSUPPORTED_CAPABILITY
    assert capabilities.status("lora_rank") == CAPABILITY_SUPPORTED
    assert capabilities.status("replay_mix") == CAPABILITY_DATA_LAYER
    assert capabilities.status("full_finetune") == CAPABILITY_MODEL_HARDWARE_DEPENDENT
    assert capabilities.status("checkpoint_resume") == CAPABILITY_SUPPORTED
    assert capabilities.status("custom_objective") == CAPABILITY_CAPABILITY_DEPENDENT
    assert capabilities.status("not-a-capability") is None
    assert len(capabilities.refused_rows()) == len(backends.UNSLOTH_REFUSED_KNOBS)
    with pytest.raises(TrainingBackendRefusal):
        CapabilityRow("mystery", "probably")


def test_unsloth_estimates_are_reported_as_unmeasured_rather_than_guessed(tmp_path):
    manifest = _manifest(
        tmp_path, project_template_path=_template(tmp_path, engine="unsloth")
    )
    estimate = UnslothTrainingBackend().estimate(manifest, _recipe())
    assert estimate.available is False
    assert estimate.strategy == "peft/unsloth"
    assert estimate.total_bytes is None


# --------------------------------------------------------------------------
# KAGGLE
# --------------------------------------------------------------------------


class _RecordingComputeBackend:
    """A ComputeBackend that would refuse to run: nothing dispatches in tests."""

    name = "fake-kaggle"

    def preflight(self, request: Any):
        return None

    def dispatch(self, request: Any, *, destination: Any):
        raise AssertionError("no dispatch in this test")


def _kaggle_config(tmp_path: Path) -> dict[str, Any]:
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    inputs: dict[str, str] = {}
    for name in DECLARED_PREPARED_FIELDS:
        document = prepared_dir / f"{name}.json"
        document.write_text("{}\n", encoding="utf-8")
        inputs[name] = str(document)
    prepared_path = tmp_path / "prepared-campaign.json"
    prepared_path.write_text(
        json.dumps(
            {
                "cycle_id": "gen2",
                "directory": str(tmp_path / "state"),
                "inputs": inputs,
                "recipe_ids": ["recipe-01"],
                "notes": [],
            }
        ),
        encoding="utf-8",
    )
    return {
        "prepared_path": str(prepared_path),
        "repository": "https://github.com/niko4244/Chowder",
        "commit_sha": COMMIT,
        "entry_point": "chowder.growth.kaggle_payload",
        "mounts": ["owner/dataset"],
        "attempts_root": str(tmp_path / "attempts"),
        "timeout_seconds": 3600.0,                "accelerator": "nvidia-tesla-p100",
                "owner": "owner",
                # Kernel-side paths are absolute POSIX paths; the local files
                # above are what the attempt hashes before it pushes them.
                "input_paths": {
                    name: f"/kaggle/input/prepared/{name}.json" for name in inputs
                },
                "payload": {"kind": "corpus-training", "project_gpu_hour_budget": 1.0},
    }


def _kaggle_manifest(tmp_path: Path, **overrides: Any) -> CampaignManifest:
    declaration = TrainingBackendDeclaration.from_mapping(
        {"provider": "kaggle", "config": _kaggle_config(tmp_path)}
    )
    return _manifest(tmp_path, training_backend=declaration, **overrides)


def test_kaggle_preflight_requires_the_declared_wiring(tmp_path):
    declaration = TrainingBackendDeclaration.from_mapping({"provider": "kaggle"})
    manifest = _manifest(tmp_path, training_backend=declaration)
    result = KaggleTrainingBackend().preflight(manifest)
    assert result.admitted is False
    assert result.code == KAGGLE_CONFIG_INCOMPLETE
    assert result.code in STRUCTURAL_PREFLIGHT_CODES


def test_kaggle_reports_the_blocked_operator_credential_status(tmp_path):
    manifest = _kaggle_manifest(tmp_path)
    result = KaggleTrainingBackend(cli_probe=lambda name: None).preflight(manifest)
    assert result.admitted is False
    assert result.code == KAGGLE_BACKEND_UNAVAILABLE
    assert "BLOCKED_BY_OPERATOR_CREDENTIAL_OR_QUOTA" in result.reason
    assert result.code not in STRUCTURAL_PREFLIGHT_CODES

    ready = KaggleTrainingBackend(cli_probe=lambda name: "/usr/local/bin/kaggle").preflight(
        manifest
    )
    assert ready.admitted is True
    assert any("checked at dispatch" in note for note in ready.notes)


def test_kaggle_builds_the_kernel_binding_from_the_declared_wiring(tmp_path):
    manifest = _kaggle_manifest(tmp_path)
    backend = KaggleTrainingBackend(backend=_RecordingComputeBackend())
    assert backend.preflight(manifest).admitted is True
    training_fn = backend.build_training_fn(manifest)
    assert isinstance(training_fn, KaggleTrainingFn)
    # It is a TrainingFn with the shared admission seam.
    assert callable(training_fn)
    assert training_fn.admit(_recipe()) is None


def test_kaggle_admission_is_the_same_growth_envelope(tmp_path):
    manifest = _kaggle_manifest(
        tmp_path,
        budget=_budget(
            device_gpu_hours_ceiling_per_recipe=0.1,
            wall_gpu_hours_ceiling_per_recipe=0.1,
            device_gpu_hours_ceiling_campaign=0.2,
            wall_gpu_hours_ceiling_campaign=0.2,
        ),
    )
    refusal = KaggleTrainingBackend(backend=_RecordingComputeBackend()).admit(
        manifest, _recipe()
    )
    assert refusal is not None
    assert refusal[0] == "growth-envelope"


def test_kaggle_resume_declares_a_remote_checkpoint_without_a_local_path(tmp_path):
    manifest = _kaggle_manifest(tmp_path)
    backend = KaggleTrainingBackend(backend=_RecordingComputeBackend())
    resumed = backend.resume(manifest, _recipe(), "checkpoint-100")
    assert resumed.resume_from_checkpoint == "checkpoint-100"


# --------------------------------------------------------------------------
# AUTO
# --------------------------------------------------------------------------


def test_auto_chooses_a_preflight_passable_provider_and_records_every_refusal(tmp_path):
    declaration = TrainingBackendDeclaration.from_mapping({"provider": "auto"})
    manifest = _manifest(
        tmp_path, project_template_path=_template(tmp_path, engine="unsloth")
    )
    chosen, selection = choose_auto_backend(
        declaration,
        manifest,
        providers=[
            LocalTrainingBackend(probe=_probe(_panel_without_device())),
            UnslothTrainingBackend(probe=_probe(_panel_with_device())),
        ],
    )
    assert chosen is not None
    assert chosen.provider == "unsloth"
    assert selection.provider == "unsloth"
    assert "auto chose unsloth" in selection.reason
    refused = {entry[0]: entry[1] for entry in selection.refused}
    # local refused on a hardware fact; kaggle was declared but not supplied.
    assert refused == {
        "local": TRAINING_BACKEND_NO_DEVICE,
        "kaggle": TRAINING_BACKEND_UNKNOWN_PROVIDER,
    }
    assert selection.to_dict()["refused"][0]["provider"] == "local"
    assert selection.to_dict()["cost_basis"] == selection.cost_basis


def test_auto_prefers_the_cheapest_known_overhead_over_the_declared_order(tmp_path):
    declaration = TrainingBackendDeclaration.from_mapping(
        {"provider": "auto", "config": {"candidates": ["unsloth", "kaggle"]}}
    )
    manifest = _manifest(tmp_path, training_backend=declaration)
    expensive = _FakeProvider("unsloth", overhead_hours=5.0)
    cheap = _FakeProvider("kaggle", overhead_hours=0.5)
    chosen, selection = choose_auto_backend(
        declaration, manifest, providers=[expensive, cheap]
    )
    assert chosen is cheap
    assert selection.provider == "kaggle"
    assert "cheapest known attach overhead" in selection.reason
    assert [entry["provider"] for entry in selection.cost_comparison] == [
        "unsloth",
        "kaggle",
    ]
    assert selection.cost_comparison[1]["overhead_hours"] == 0.5
    # And the record is explicit that this is not a measured end-to-end cost.
    assert "not a measured end-to-end cost" in selection.cost_basis


def test_auto_keeps_the_declared_order_when_no_overhead_is_measured(tmp_path):
    declaration = TrainingBackendDeclaration.from_mapping(
        {"provider": "auto", "config": {"candidates": ["unsloth", "kaggle"]}}
    )
    manifest = _manifest(tmp_path, training_backend=declaration)
    first = _FakeProvider("unsloth", overhead_hours=None)
    second = _FakeProvider("kaggle", overhead_hours=None)
    chosen, selection = choose_auto_backend(
        declaration, manifest, providers=[first, second]
    )
    assert chosen is first
    assert selection.provider == "unsloth"
    assert "no passable candidate reports a measured attach overhead" in selection.reason
    assert all(
        entry["overhead_hours"] is None for entry in selection.cost_comparison
    )


def test_auto_refuses_rather_than_dispatching_anything_when_no_candidate_passes(tmp_path):
    declaration = TrainingBackendDeclaration.from_mapping(
        {"provider": "auto", "config": {"candidates": ["local"]}}
    )
    manifest = _manifest(tmp_path)
    with pytest.raises(TrainingBackendRefusal) as refusal:
        resolve_training_backend(
            declaration,
            manifest,
            providers=[LocalTrainingBackend(probe=_probe(_panel_without_device()))],
        )
    assert refusal.value.code == TRAINING_BACKEND_AUTO_UNRESOLVED
    assert "no provider" in refusal.value.reason


def test_a_manual_declaration_is_returned_as_declared_with_no_selection(tmp_path):
    declaration = TrainingBackendDeclaration.from_mapping(
        {"provider": "local", "config": {"device": "cpu"}}
    )
    provider, selection = resolve_training_backend(
        declaration,
        _manifest(tmp_path),
        probes={"local": _probe(_panel_without_device())},
    )
    assert isinstance(provider, LocalTrainingBackend)
    assert selection is None


class _FakeProvider:
    """A provider whose executor is a sentinel: nothing trains in this test."""

    def __init__(
        self,
        provider: str,
        *,
        admitted: bool = True,
        overhead_hours: float | None = 0.0,
    ) -> None:
        self.provider = provider
        self.trainer = f"fake-{provider}"
        self.admitted = admitted
        self.overhead_hours = overhead_hours
        self.executor = _RecordingExecutor()
        self.built = 0

    def capabilities(self):
        raise AssertionError("capabilities are not asked for on this path")

    def preflight(self, manifest: Any) -> PreflightResult:
        return PreflightResult(
            provider=self.provider,
            admitted=self.admitted,
            code=None if self.admitted else TRAINING_BACKEND_NO_DEVICE,
            reason="fake preflight",
            panel=PreflightPanel(provider=self.provider),
        )

    def projected_overhead(self, manifest: Any) -> OverheadReport:
        return OverheadReport(hours=self.overhead_hours, basis="fake basis")

    def estimate(self, manifest: Any, recipe: Any):
        raise AssertionError("estimate is not asked for on this path")

    def admit(self, manifest: Any, recipe: Any) -> None:
        return None

    def build_training_fn(self, manifest: Any, *, state_root: Any = None, runner: Any = None):
        self.built += 1
        return self.executor

    def resume(self, manifest: Any, recipe: Any, checkpoint: Any):
        raise AssertionError("resume is not asked for on this path")


class _RecordingExecutor:
    """A TrainingFn-shaped sentinel that returns the evidence it is given."""

    def __init__(self, evidence: Mapping[str, Any] | None = None) -> None:
        self.evidence = dict(evidence or {"status": "SUCCEEDED", "recipe_id": "recipe-01"})
        self.calls = 0

    def admit(self, recipe: Any) -> None:
        return None

    def __call__(self, recipe: Any, items: Any) -> Mapping[str, Any]:
        self.calls += 1
        return dict(self.evidence)


def test_the_runner_dispatches_through_the_declared_provider_and_records_auto(tmp_path):
    declaration = TrainingBackendDeclaration.from_mapping(
        {"provider": "auto", "config": {"candidates": ["unsloth"]}}
    )
    manifest = _manifest(tmp_path, training_backend=declaration)
    fake = _FakeProvider("unsloth")
    executor, record = build_executor_with_selection(manifest, providers=[fake])
    assert fake.built == 1
    assert record["provider"] == "unsloth"
    assert record["trainer"] == "fake-unsloth"
    assert record["admitted"] is True
    assert record["selection"]["provider"] == "unsloth"
    assert "auto chose unsloth" in record["selection"]["reason"]
    # The executor names its backend in the evidence it returns, so the attempt
    # record carries the fact rather than reconstructing it.
    evidence = executor(_recipe(), ())
    assert evidence["backend"]["provider"] == "unsloth"
    assert evidence["backend"]["trainer"] == "fake-unsloth"
    assert evidence["backend"]["version"]
    assert fake.executor.calls == 1
    assert _attempt_summary((evidence,))[0]["backend"] == evidence["backend"]


def test_the_runner_refuses_a_structural_declaration_error_before_compute(tmp_path):
    template = _template(tmp_path, engine="transformers")
    declaration = TrainingBackendDeclaration.from_mapping({"provider": "unsloth"})
    manifest = _manifest(
        tmp_path, project_template_path=template, training_backend=declaration
    )
    with pytest.raises(CampaignRunRefusal) as refusal:
        build_executor_with_selection(manifest)
    assert TRAINING_BACKEND_TRAINER_MISMATCH in str(refusal.value)


def test_the_runner_refuses_an_auto_declaration_no_candidate_can_honor(tmp_path):
    declaration = TrainingBackendDeclaration.from_mapping(
        {"provider": "auto", "config": {"candidates": ["local"]}}
    )
    manifest = _manifest(tmp_path, training_backend=declaration)
    with pytest.raises(CampaignRunRefusal) as refusal:
        build_executor_with_selection(
            manifest, providers=[_FakeProvider("local", admitted=False)]
        )
    assert TRAINING_BACKEND_AUTO_UNRESOLVED in str(refusal.value)


def test_an_unknown_provider_refuses_at_resolution(tmp_path):
    with pytest.raises(TrainingBackendRefusal) as refusal:
        backends.backend_for_provider("wandb")
    assert refusal.value.code == TRAINING_BACKEND_UNKNOWN_PROVIDER
    declaration = TrainingBackendDeclaration(provider="local", config={"device": "tpu"})
    manifest = _manifest(tmp_path, training_backend=declaration)
    provider, _selection = resolve_training_backend(
        declaration, manifest, probes={"local": _probe(_panel_with_device())}
    )
    with pytest.raises(TrainingBackendRefusal) as refused:
        provider.preflight(manifest)
    assert refused.value.code == TRAINING_BACKEND_UNSUPPORTED_CONFIG
    assert "device" in refused.value.reason


# --------------------------------------------------------------------------
# the uniform outcome
# --------------------------------------------------------------------------


def _source() -> SourceBinding:
    return SourceBinding(
        repository="https://github.com/niko4244/Chowder",
        commit_sha=COMMIT,
        chowder_version="0.3.0",
        cycle_id="gen2",
        recipe_id="recipe-01",
        attempt_id="attempt-01",
    )


def test_a_successful_attempt_maps_onto_one_outcome_vocabulary():
    evidence = {
        "status": "SUCCEEDED",
        "candidate_succeeded": True,
        "artifact_ref": "artifact/adapter.bin",
        "artifact_sha256": "c" * 64,
        "measured_gpu_hours": 2.5,
        "candidate_metrics": {"loss": 0.5},
        "environment": {"gpu": "fake"},
        "resume_state": "not-a-resume",
        "source_commit_sha": COMMIT,
    }
    outcome = uniform_outcome_from_evidence(
        evidence,
        source=_source(),
        backend="local",
        backend_version="0.3.0",
        candidate_id="gen2-candidate",
        intervention_id="lora-rank-16",
        parent_checkpoint="gen1",
    )
    assert outcome.status == OUTCOME_SUCCEEDED
    assert outcome.attempt_id == "attempt-01"
    assert outcome.failure_class is None
    assert outcome.outcome.artifacts[0].path == "artifact/adapter.bin"
    assert outcome.outcome.cost is not None
    assert outcome.outcome.cost.wall_gpu_hours == 2.5
    rendered = outcome.to_evidence()
    assert rendered["backend"]["provider"] == "local"
    assert rendered["backend"]["parent_checkpoint"] == "gen1"
    assert rendered["candidate_id"] == "gen2-candidate"
    assert rendered["evidence_refs"] == []
    assert rendered["artifact_sha256"] == "c" * 64


def test_a_settlement_refusal_is_classified_by_the_shared_classifier():
    evidence = {
        "status": "FAILED",
        "candidate_succeeded": None,
        "budget_settlement": {
            "budget_compliant": False,
            "budget_failure_reasons": ["ACTUAL_WALL_GPU_HOURS_EXCEEDED: 9 > 1"],
        },
        "measured_gpu_hours": 9.0,
    }
    outcome = uniform_outcome_from_evidence(
        evidence, source=_source(), backend="kaggle", backend_version="0.3.0"
    )
    assert outcome.status == OUTCOME_FAILED
    assert outcome.failure_class == FailureClass.BUDGET_EXHAUSTED.value
    assert outcome.to_evidence()["refused_by"] == "budget_settlement"


def test_an_evidence_document_with_no_status_cannot_become_an_outcome():
    with pytest.raises(TrainingBackendRefusal) as refusal:
        uniform_outcome_from_evidence(
            {"status": "probably-fine"},
            source=_source(),
            backend="local",
            backend_version="0.3.0",
        )
    assert refusal.value.code == TRAINING_BACKEND_EVIDENCE_STATUS_UNKNOWN


# --------------------------------------------------------------------------
# the preflight report (what the `campaign preflight` command prints)
# --------------------------------------------------------------------------


def test_the_preflight_report_names_the_backend_and_estimates_each_recipe(tmp_path):
    manifest = _manifest(
        tmp_path,
        project_template_path=_template(tmp_path),
        training_backend=TrainingBackendDeclaration.from_mapping(
            {"provider": "local", "config": {"device": "cuda"}}
        ),
    )
    recipe = _recipe()
    report = preflight_report(
        manifest,
        recipes=[recipe],
        probes={"local": _probe(_panel_with_device())},
    )
    assert report["status"] == "ADMITTED"
    assert report["provider"] == "local"
    assert report["declared"]["config"] == {"device": "cuda"}
    assert report["stops_the_run"] is False
    assert report["refused_by"] is None
    assert report["panel"]["device_count"] == 1
    assert [entry["recipe_id"] for entry in report["estimates"]] == ["recipe-01"]
    assert report["estimates"][0]["available"] is True
    assert report["capabilities"]["rows"]
    assert report["overhead"]["hours"] == 0.0
    assert report["recipes_unavailable"] is None


def test_the_preflight_report_separates_a_reported_fact_from_an_enforced_refusal(tmp_path):
    manifest = _manifest(tmp_path)
    reported = preflight_report(
        manifest, probes={"local": _probe(_panel_without_device())}
    )
    assert reported["status"] == "REFUSED"
    assert reported["refused_by"] == TRAINING_BACKEND_NO_DEVICE
    # A hardware fact is reported, not enforced: the runner still builds.
    assert reported["stops_the_run"] is False
    assert reported["panel"]["device_count"] == 0

    mismatched = preflight_report(
        _manifest(
            tmp_path,
            project_template_path=_template(tmp_path, engine="transformers"),
            training_backend=TrainingBackendDeclaration.from_mapping(
                {"provider": "unsloth"}
            ),
        )
    )
    assert mismatched["refused_by"] == TRAINING_BACKEND_TRAINER_MISMATCH
    assert mismatched["stops_the_run"] is True


def test_the_preflight_report_reports_an_unresolvable_auto_declaration(tmp_path):
    declaration = TrainingBackendDeclaration.from_mapping(
        {"provider": "auto", "config": {"candidates": ["local"]}}
    )
    manifest = _manifest(tmp_path, training_backend=declaration)
    report = preflight_report(
        manifest,
        recipes_unavailable="no parent profile declared",
        providers=[LocalTrainingBackend(probe=_probe(_panel_without_device()))],
    )
    assert report["status"] == "REFUSED"
    assert report["refused_by"] == TRAINING_BACKEND_AUTO_UNRESOLVED
    assert report["stops_the_run"] is True
    assert report["estimate" if False else "estimates"] == []
    assert report["recipes_unavailable"] == "no parent profile declared"


# --------------------------------------------------------------------------
# the Kaggle backend, from the declaration alone
# --------------------------------------------------------------------------


def _declared_input_paths(tmp_path: Path) -> dict[str, str]:
    """Write one file per prepared input field, named exactly as the manifest fields."""
    directory = tmp_path / "declared-inputs"
    directory.mkdir(exist_ok=True)
    paths: dict[str, str] = {}
    for name in DECLARED_PREPARED_FIELDS:
        document = directory / f"{name}.json"
        document.write_text("{}\n", encoding="utf-8")
        paths[name] = str(document)
    return paths


def _kaggle_config_without_prepared_document(
    inputs: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "repository": "https://github.com/niko4244/Chowder",
        "commit_sha": COMMIT,
        "entry_point": "chowder.growth.kaggle_payload",
        "mounts": ["owner/dataset"],
        "attempts_root": "attempts",
        "timeout_seconds": 3600.0,
        "accelerator": "nvidia-tesla-p100",
        # Kernel-side paths are absolute POSIX paths; the local files are what an
        # attempt hashes before it pushes them.
        "input_paths": {
            name: f"/kaggle/input/prepared/{name}.json" for name in inputs
        },
        "payload": {"kind": "corpus-training", "project_gpu_hour_budget": 1.0},
    }


def test_the_kaggle_backend_builds_its_prepared_campaign_from_the_manifest(tmp_path):
    inputs = _declared_input_paths(tmp_path)
    declaration = TrainingBackendDeclaration.from_mapping(
        {"provider": "kaggle", "config": _kaggle_config_without_prepared_document(inputs)}
    )
    manifest = _manifest(tmp_path, training_backend=declaration, **inputs)
    backend = KaggleTrainingBackend(backend=_RecordingComputeBackend())
    assert backend.preflight(manifest).admitted is True
    training_fn = backend.build_training_fn(manifest)
    # The prepared campaign was assembled from the manifest's own declared
    # inputs: no hand-written prepared document was needed.
    assert isinstance(training_fn, KaggleTrainingFn)
    assert training_fn.prepared.inputs == inputs
    assert training_fn.admit(_recipe()) is None


def test_the_kaggle_backend_refuses_a_manifest_that_declares_no_inputs(tmp_path):
    declaration = TrainingBackendDeclaration.from_mapping(
        {
            "provider": "kaggle",
            "config": _kaggle_config_without_prepared_document({}),
        }
    )
    manifest = _manifest(tmp_path, training_backend=declaration)
    result = KaggleTrainingBackend(backend=_RecordingComputeBackend()).preflight(manifest)
    assert result.admitted is False
    assert result.code == KAGGLE_CONFIG_INCOMPLETE
    assert "evaluation_material_path" in result.reason
    assert result.code in STRUCTURAL_PREFLIGHT_CODES


def test_the_kaggle_backend_refuses_input_paths_that_do_not_cover_the_inputs(tmp_path):
    inputs = _declared_input_paths(tmp_path)
    config = _kaggle_config_without_prepared_document(inputs)
    config["input_paths"] = {
        name: path for name, path in config["input_paths"].items() if name != "data_registry_path"
    }
    declaration = TrainingBackendDeclaration.from_mapping(
        {"provider": "kaggle", "config": config}
    )
    manifest = _manifest(tmp_path, training_backend=declaration, **inputs)
    result = KaggleTrainingBackend(backend=_RecordingComputeBackend()).preflight(manifest)
    assert result.admitted is False
    assert result.code == TRAINING_BACKEND_UNSUPPORTED_CONFIG
    assert "data_registry_path" in result.reason


# --------------------------------------------------------------------------
# capability drift: the matrix must match the engine it claims to describe
# --------------------------------------------------------------------------

#: Every Unsloth capability the matrix calls ``supported`` and the declared
#: value it must carry into the executor's own spec. If the executor renames a
#: knob, the value stops arriving and this table fails -- which is the point.
_UNSLOTH_SUPPORTED_VALUES: tuple[tuple[str, tuple[str, ...], str, Any], ...] = (
    ("learning_rate", ("training", "learning_rate"), "learning_rate", 3e-4),
    ("lora_rank", ("lora", "r"), "lora_r", 32),
    ("lora_alpha", ("lora", "alpha"), "lora_alpha", 64),
    (
        "gradient_accumulation",
        ("training", "gradient_accumulation_steps"),
        "gradient_accumulation_steps",
        7,
    ),
    ("max_steps", ("training", "max_steps"), "max_steps", 123),
    ("batch_size", ("training", "batch_size"), "batch_size", 3),
    (
        "lr_scheduler_type",
        ("training", "lr_scheduler_type"),
        "lr_scheduler_type",
        "cosine",
    ),
    ("warmup_steps", ("training", "warmup_steps"), "warmup_steps", 11),
    ("warmup_ratio", ("training", "warmup_ratio"), "warmup_ratio", 0.05),
    ("quantization", ("quantization",), "quantization", "4bit"),
    ("max_length", ("max_length",), "max_length", 1024),
)


def _unsloth_config(tmp_path: Path, **backend: Any) -> dict[str, Any]:
    dataset = tmp_path / "unsloth-train.jsonl"
    if not dataset.exists():
        dataset.write_text('{"text": "hello"}\n', encoding="utf-8")
    return {
        "backend": {
            "base_model": "Fake/Model",
            "dataset": str(dataset),
            "training": dict(backend.pop("training", {})),
            "lora": {"target_modules": ["q_proj", "v_proj"]},
            **backend,
        }
    }


def _unsloth_spec(tmp_path: Path, document: Mapping[str, Any]) -> UnslothPeftRunSpec:
    return UnslothPeftRunSpec.from_resolved_config(
        document, work_dir=tmp_path, output_dir=tmp_path / "out", seed=17
    )


def test_every_supported_unsloth_capability_reaches_the_executor_spec(tmp_path):
    capabilities = UnslothTrainingBackend().capabilities()
    document = _unsloth_config(tmp_path)
    backend = document["backend"]
    for _capability, path, _attribute, value in _UNSLOTH_SUPPORTED_VALUES:
        section: dict[str, Any] = backend
        for key in path[:-1]:
            section = section.setdefault(key, {})
        section[path[-1]] = value
    spec = _unsloth_spec(tmp_path, document)
    for capability, path, attribute, value in _UNSLOTH_SUPPORTED_VALUES:
        assert capabilities.status(capability) == CAPABILITY_SUPPORTED, capability
        assert getattr(spec, attribute) == value, (
            f"{capability} (config {'/'.join(path)}) did not reach spec.{attribute}"
        )
    # target_modules and seed travel in their own namespaces.
    assert capabilities.status("target_modules") == CAPABILITY_SUPPORTED
    assert spec.target_modules == ("q_proj", "v_proj")
    assert capabilities.status("seed") == CAPABILITY_SUPPORTED
    assert spec.seed == 17


def test_the_unsloth_matrix_refusals_match_what_the_engine_actually_refuses(tmp_path):
    capabilities = UnslothTrainingBackend().capabilities()
    # The vocabulary the guard covers: the knobs the matrix refuses, plus knobs
    # the executor must accept, so a *new* refusal in the executor also fires.
    accepted_knobs = ("gradient_checkpointing", "optimizer")
    for knob in backends.UNSLOTH_REFUSED_KNOBS:
        assert capabilities.status(knob) == CAPABILITY_REFUSED, knob
        with pytest.raises(UnslothConfigError):
            _unsloth_spec(tmp_path, _unsloth_config(tmp_path, training={knob: True}))
    for knob in accepted_knobs:
        assert capabilities.status(knob) != CAPABILITY_REFUSED
        # A declared-but-supported knob must not be silently refused here.
        _unsloth_spec(tmp_path, _unsloth_config(tmp_path, training={knob: True}))


def test_checkpoint_resume_is_supported_and_its_basis_is_named(tmp_path):
    capabilities = UnslothTrainingBackend().capabilities()
    assert capabilities.status("checkpoint_resume") == CAPABILITY_SUPPORTED
    reason = next(
        row.reason
        for row in capabilities.rows
        if row.capability == "checkpoint_resume"
    )
    assert "chowder-unsloth-checkpoint-manifest.json" in reason
    assert "tests/test_unsloth_peft.py" in reason
    assert "real-CUDA acceptance" in reason
    # And the declaration really does reach the executor's spec.
    checkpoint = tmp_path / "checkpoint-50"
    checkpoint.mkdir()
    spec = _unsloth_spec(
        tmp_path, _unsloth_config(tmp_path, resume_from_checkpoint=str(checkpoint))
    )
    assert spec.resume_from_checkpoint == str(checkpoint.resolve())


def test_collect_and_verify_prove_an_attempts_evidence(tmp_path):
    manifest = _manifest(tmp_path)
    backend = LocalTrainingBackend(probe=_probe(_panel_with_device()))
    attempt = tmp_path / "attempts" / "attempt-01"
    attempt.mkdir(parents=True)
    with pytest.raises(TrainingBackendRefusal) as refusal:
        backend.collect(manifest, attempt)
    assert refusal.value.code == TRAINING_BACKEND_EVIDENCE_MISSING
    assert backend.verify(manifest, attempt) == (
        TRAINING_BACKEND_EVIDENCE_MISSING,
        refusal.value.reason,
    )

    artifact = attempt / "adapter.bin"
    artifact.write_bytes(b"adapter-bytes")
    digest = hashlib.sha256(b"adapter-bytes").hexdigest()
    (attempt / "training-evidence.json").write_text(
        json.dumps(
            {
                "status": "SUCCEEDED",
                "artifact_ref": "adapter.bin",
                "artifact_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    assert backend.verify(manifest, attempt) is None
    assert backend.collect(manifest, attempt)["status"] == "SUCCEEDED"

    artifact.write_bytes(b"tampered")
    refusal = backend.verify(manifest, attempt)
    assert refusal is not None
    assert refusal[0] == backends.TRAINING_BACKEND_ARTIFACT_DIGEST_MISMATCH
