import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from chowder.backends.transformers_peft import (
    TransformersPeftExecutor,
    TransformersPeftRunSpec,
    _ALLOWED_ACTIVATION_OFFLOAD_MODES,
    _ALLOWED_FROZEN_LAYER_STREAMING_MODES,
    _ALLOWED_OPTIMIZER_TIERING_MODES,
    _default_gradient_checkpointing,
    _default_quantization,
    _min_device_vram_gb,
    _resolve_activation_offload_flag,
    _resolve_frozen_layer_streaming_flag,
    _resolve_optimizer_tiering_flag,
)
from chowder.backends.transformers_worker import (
    _build_chat_example,
    _replay_sample_count,
    _resolve_target_modules,
    _validate_chat_messages,
)
from chowder.executors import ExecutionContext
from chowder.memory import HardwareProfile
from chowder.models import Experiment, Hypothesis
from chowder.provenance import sha256_file


def _hardware():
    return HardwareProfile(16, 64, 500, 12, 40, 3)


def _experiment():
    return Experiment("e1", None, Hypothesis("obs", "cause", "fix"), {}, 2.0)


def _config(dataset: str):
    return {
        "seed": 17,
        "backend": {
            "type": "transformers-peft",
            "base_model": "example/model",
            "dataset": dataset,
            "max_length": 256,
            "quantization": "4bit",
            "training": {"learning_rate": 1e-4, "epochs": 2},
            "lora": {
                "r": 8,
                "alpha": 16,
                "target_modules": ["q_proj", "v_proj"],
            },
        },
    }


def test_spec_is_built_from_resolved_config_and_workdir(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    spec = TransformersPeftRunSpec.from_resolved_config(
        _config("train.jsonl"),
        work_dir=tmp_path,
        output_dir=tmp_path / "adapter",
        seed=1,
    )
    assert spec.dataset == str(data.resolve())
    assert spec.dataset_sha256 is None
    assert spec.seed == 17
    assert spec.lora_r == 8
    assert spec.quantization == "4bit"
    assert spec.target_modules == ("q_proj", "v_proj")
    # Omitting the new optimizer/schedule fields must reproduce the exact
    # values already baked into every prior recipe -- these are HF's own
    # TrainingArguments defaults, not new Chowder-chosen defaults.
    assert spec.lr_scheduler_type == "linear"
    assert spec.warmup_ratio == 0.0
    assert spec.warmup_steps == 0
    assert spec.weight_decay == 0.0
    assert spec.max_grad_norm == 1.0
    assert spec.max_steps == -1


def test_spec_resolves_bound_replay_config(tmp_path):
    data = tmp_path / "train.jsonl"
    replay = tmp_path / "replay.jsonl"
    data.write_text('{"text":"repair"}\n')
    replay.write_text('{"text":"rehearsal"}\n')
    config = _config("train.jsonl")
    config["backend"]["dataset_sha256"] = sha256_file(data)
    config["backend"]["replay"] = {
        "dataset": "replay.jsonl",
        "sha256": sha256_file(replay),
        "ratio": 0.5,
    }
    spec = TransformersPeftRunSpec.from_resolved_config(
        config,
        work_dir=tmp_path,
        output_dir=tmp_path / "adapter",
        seed=1,
    )
    assert spec.dataset_sha256 == sha256_file(data)
    assert spec.replay_dataset == str(replay.resolve())
    assert spec.replay_sha256 == sha256_file(replay)
    assert spec.replay_ratio == pytest.approx(0.5)


def test_replay_requires_hash_and_valid_ratio(tmp_path):
    data = tmp_path / "train.jsonl"
    replay = tmp_path / "replay.jsonl"
    data.write_text('{"text":"repair"}\n')
    replay.write_text('{"text":"rehearsal"}\n')
    config = _config(str(data))
    config["backend"]["replay"] = {"dataset": str(replay), "ratio": 1.0}
    with pytest.raises(ValueError, match="dataset and SHA"):
        TransformersPeftRunSpec.from_resolved_config(
            config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
        )

    config["backend"]["replay"] = {
        "dataset": str(replay),
        "sha256": sha256_file(replay),
        "ratio": float("inf"),
    }
    with pytest.raises(ValueError, match="ratio"):
        TransformersPeftRunSpec.from_resolved_config(
            config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
        )


def test_remote_code_is_rejected_for_autonomous_execution(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["trust_remote_code"] = True
    with pytest.raises(ValueError, match="trust_remote_code"):
        TransformersPeftRunSpec.from_resolved_config(
            config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
        )


def test_profile_uses_measured_step_profile_when_available(tmp_path):
    config = _config("train.jsonl")
    config["backend"]["profile"] = {
        "estimated_steps": 100,
        "seconds_per_step": 3.6,
        "peak_vram_gb": 9.5,
        "source": "measured",
    }
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    estimate = TransformersPeftExecutor().profile(_experiment(), context)
    assert estimate.gpu_hours == pytest.approx(0.1)
    assert estimate.peak_vram_gb == 9.5
    assert estimate.confidence == 0.75


def test_run_requires_resolved_config(tmp_path):
    context = ExecutionContext(_hardware(), str(tmp_path), 1)
    with pytest.raises(ValueError, match="resolved_config"):
        TransformersPeftExecutor().run(_experiment(), context)


def test_executor_runs_worker_as_isolated_process_and_returns_artifact(
    tmp_path, monkeypatch
):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    context = ExecutionContext(
        _hardware(), str(tmp_path), 1, resolved_config=_config("train.jsonl")
    )
    observed_spec = {}

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            spec_path = Path(command[command.index("--spec") + 1])
            result_path = Path(command[command.index("--result") + 1])
            spec = json.loads(spec_path.read_text())
            observed_spec.update(spec)
            output = Path(spec["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            (output / "adapter_model.safetensors").write_bytes(b"adapter")
            result_path.write_text(
                json.dumps(
                    {
                        "telemetry": {"train_loss": 0.25, "global_step": 3},
                        "versions": {"transformers": "5.test"},
                        "provenance": {"resolved_model_commit": "abc123"},
                        "data_provenance": {
                            "primary_rows": 1,
                            "replay_selected_rows": 0,
                        },
                    }
                )
            )

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen", FakeProcess
    )
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert artifact.experiment_id == "e1"
    assert artifact.telemetry["train_loss"] == 0.25
    assert Path(artifact.artifact_ref).is_dir()
    assert artifact.evidence["versions"]["transformers"] == "5.test"
    assert artifact.evidence["model_provenance"]["resolved_model_commit"] == "abc123"
    assert observed_spec["dataset_sha256"] == sha256_file(data)
    assert artifact.evidence["dataset_sha256"] == sha256_file(data)
    assert artifact.evidence["replay_dataset_sha256"] is None
    assert artifact.evidence["data_provenance"]["primary_rows"] == 1
    assert len(artifact.evidence["execution_spec_sha256"]) == 64
    assert len(artifact.evidence["recipe_sha256"]) == 64
    assert len(artifact.evidence["artifact_sha256"]) == 64


def test_executor_rejects_primary_dataset_changed_after_proposal(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"original"}\n')
    config = _config(str(data))
    config["backend"]["dataset_sha256"] = sha256_file(data)
    data.write_text('{"text":"tampered"}\n')
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)

    def should_not_launch(*args, **kwargs):
        raise AssertionError("worker must not launch for mutated training data")

    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen", should_not_launch
    )
    with pytest.raises(ValueError, match="training dataset content changed"):
        TransformersPeftExecutor().run(_experiment(), context)


def test_executor_rejects_replay_dataset_changed_after_proposal(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    replay = tmp_path / "replay.jsonl"
    data.write_text('{"text":"repair"}\n')
    replay.write_text('{"text":"parent"}\n')
    config = _config(str(data))
    config["backend"]["dataset_sha256"] = sha256_file(data)
    config["backend"]["replay"] = {
        "dataset": str(replay),
        "sha256": sha256_file(replay),
        "ratio": 1.0,
    }
    replay.write_text('{"text":"tampered"}\n')
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)

    def should_not_launch(*args, **kwargs):
        raise AssertionError("worker must not launch for mutated replay data")

    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen", should_not_launch
    )
    with pytest.raises(ValueError, match="replay dataset content changed"):
        TransformersPeftExecutor().run(_experiment(), context)


def test_recipe_digest_is_stable_across_output_paths(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["dataset_sha256"] = sha256_file(data)
    a = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "a", seed=1
    )
    b = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "b", seed=1
    )
    assert a.digest() != b.digest()
    assert a.recipe_digest() == b.recipe_digest()


def test_replay_sample_count_is_bounded_and_deterministic():
    assert _replay_sample_count(4, 100, 0.5) == 2
    assert _replay_sample_count(4, 3, 10.0) == 3
    assert _replay_sample_count(1, 4, 0.01) == 1
    assert _replay_sample_count(0, 4, 1.0) == 0


# --- optimizer/schedule recipe controls -------------------------------------


def test_spec_reads_optimizer_schedule_fields_from_resolved_config(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"].update(
        {
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "warmup_steps": 50,
            "weight_decay": 0.01,
            "max_grad_norm": 0.5,
            "max_steps": 200,
        }
    )
    spec = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    assert spec.lr_scheduler_type == "cosine"
    assert spec.warmup_ratio == pytest.approx(0.1)
    assert spec.warmup_steps == 50
    assert spec.weight_decay == pytest.approx(0.01)
    assert spec.max_grad_norm == pytest.approx(0.5)
    assert spec.max_steps == 200


def test_unsupported_lr_scheduler_type_is_rejected(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"]["lr_scheduler_type"] = "warp_speed"
    with pytest.raises(ValueError, match="unsupported lr_scheduler_type"):
        TransformersPeftRunSpec.from_resolved_config(
            config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
        )


@pytest.mark.parametrize("ratio", [-0.1, 1.0, 1.1, float("nan"), float("inf")])
def test_warmup_ratio_out_of_range_is_rejected(tmp_path, ratio):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"]["warmup_ratio"] = ratio
    with pytest.raises(ValueError, match="warmup_ratio"):
        TransformersPeftRunSpec.from_resolved_config(
            config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
        )


def test_negative_warmup_steps_is_rejected(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"]["warmup_steps"] = -1
    with pytest.raises(ValueError, match="warmup_steps cannot be negative"):
        TransformersPeftRunSpec.from_resolved_config(
            config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
        )


def test_negative_weight_decay_is_rejected(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"]["weight_decay"] = -0.01
    with pytest.raises(ValueError, match="weight_decay must be finite and non-negative"):
        TransformersPeftRunSpec.from_resolved_config(
            config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
        )


def test_negative_max_grad_norm_is_rejected(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"]["max_grad_norm"] = -1.0
    with pytest.raises(ValueError, match="max_grad_norm must be finite and non-negative"):
        TransformersPeftRunSpec.from_resolved_config(
            config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
        )


@pytest.mark.parametrize("steps", [0, -2])
def test_invalid_max_steps_is_rejected(tmp_path, steps):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"]["max_steps"] = steps
    with pytest.raises(ValueError, match="max_steps must be -1"):
        TransformersPeftRunSpec.from_resolved_config(
            config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
        )


def test_recipe_digest_changes_when_weight_decay_changes(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    a = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    config["backend"]["training"]["weight_decay"] = 0.05
    b = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    assert a.recipe_digest() != b.recipe_digest()


# --- chat/message datasets & completion-only masking ------------------------


def test_spec_defaults_to_flat_text_format(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    spec = TransformersPeftRunSpec.from_resolved_config(
        _config(str(data)), work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    assert spec.dataset_format == "text"
    assert spec.messages_field == "messages"


def test_spec_reads_chat_format_and_messages_field(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"conversation":[{"role":"user","content":"hi"}]}\n')
    config = _config(str(data))
    config["backend"]["dataset_format"] = "chat"
    config["backend"]["messages_field"] = "conversation"
    spec = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    assert spec.dataset_format == "chat"
    assert spec.messages_field == "conversation"


def test_unsupported_dataset_format_is_rejected(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["dataset_format"] = "parquet"
    with pytest.raises(ValueError, match="unsupported dataset_format"):
        TransformersPeftRunSpec.from_resolved_config(
            config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
        )


def test_empty_messages_field_is_rejected_for_chat_format(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"messages":[]}\n')
    config = _config(str(data))
    config["backend"]["dataset_format"] = "chat"
    config["backend"]["messages_field"] = "   "
    with pytest.raises(ValueError, match="messages_field cannot be empty"):
        TransformersPeftRunSpec.from_resolved_config(
            config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
        )


def test_recipe_digest_changes_with_dataset_format(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    a = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    config["backend"]["dataset_format"] = "chat"
    config["backend"]["messages_field"] = "text"  # reuse the same file's field name
    b = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    assert a.recipe_digest() != b.recipe_digest()


def test_resume_is_rejected_when_dataset_format_changed(tmp_path, monkeypatch):
    """dataset_format is a bound input, not excluded like epochs/max_steps --
    switching between text and chat between save and resume changes what the
    checkpoint's optimizer state was actually produced from."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    dataset_sha = sha256_file(data)

    checkpoint_trainer_dir = tmp_path / "prior" / "trainer"
    checkpoint_dir = checkpoint_trainer_dir / "checkpoint-50"
    checkpoint_dir.mkdir(parents=True)

    config = _config(str(data))
    config["backend"]["dataset_sha256"] = dataset_sha
    spec_for_manifest = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "prior", seed=1
    )
    (checkpoint_trainer_dir / "chowder-checkpoint-manifest.json").write_text(
        json.dumps(TransformersPeftExecutor._bound_inputs(spec_for_manifest))
    )

    config["backend"]["dataset_format"] = "chat"
    config["backend"]["messages_field"] = "text"
    config["backend"]["resume_from_checkpoint"] = str(checkpoint_dir)
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)

    def should_not_launch(*args, **kwargs):
        raise AssertionError("worker must not launch when bound inputs changed")

    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen", should_not_launch
    )
    with pytest.raises(ValueError, match="refusing to resume"):
        TransformersPeftExecutor().run(_experiment(), context)


def test_validate_chat_messages_normalizes_a_well_formed_row():
    normalized = _validate_chat_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        row_index=0,
    )
    assert normalized == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


@pytest.mark.parametrize("bad", [None, [], "not a list", {"role": "user"}])
def test_validate_chat_messages_rejects_empty_or_non_list_rows(bad):
    with pytest.raises(RuntimeError, match="empty or invalid messages list"):
        _validate_chat_messages(bad, row_index=0)


def test_validate_chat_messages_rejects_malformed_message():
    with pytest.raises(RuntimeError, match="missing role/content"):
        _validate_chat_messages([{"role": "user"}], row_index=0)


def test_validate_chat_messages_rejects_unsupported_role():
    with pytest.raises(RuntimeError, match="unsupported message role 'tool'"):
        _validate_chat_messages(
            [{"role": "tool", "content": "x"}, {"role": "assistant", "content": "y"}],
            row_index=0,
        )


def test_validate_chat_messages_rejects_row_without_assistant_turn():
    with pytest.raises(RuntimeError, match="no assistant turn"):
        _validate_chat_messages(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
            row_index=0,
        )


# --- LoRA target module auto-detection & presets -----------------------------


class _FakeModelConfig:
    def __init__(self, model_type):
        self.model_type = model_type


class _FakeModel:
    def __init__(self, model_type):
        self.config = _FakeModelConfig(model_type)


def test_resolve_target_modules_explicit_wins_regardless_of_preset():
    model = _FakeModel("llama")
    resolved = _resolve_target_modules(
        model, explicit=("q_proj", "v_proj"), preset="attention_and_mlp"
    )
    assert resolved == ["q_proj", "v_proj"]


def test_resolve_target_modules_auto_delegates_to_peft():
    model = _FakeModel("llama")
    assert _resolve_target_modules(model, explicit=(), preset="auto") is None


def test_resolve_target_modules_attention_and_mlp_uses_curated_list():
    model = _FakeModel("llama")
    resolved = _resolve_target_modules(model, explicit=(), preset="attention_and_mlp")
    assert resolved == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def test_resolve_target_modules_attention_and_mlp_rejects_uncurated_architecture():
    model = _FakeModel("falcon")
    with pytest.raises(RuntimeError, match="no curated module list for model_type 'falcon'"):
        _resolve_target_modules(model, explicit=(), preset="attention_and_mlp")


def test_spec_defaults_to_auto_target_module_detection(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    del config["backend"]["lora"]["target_modules"]
    spec = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    assert spec.target_modules == ()
    assert spec.target_preset == "auto"


def test_spec_reads_target_preset_from_resolved_config(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    del config["backend"]["lora"]["target_modules"]
    config["backend"]["lora"]["target_preset"] = "attention_and_mlp"
    spec = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    assert spec.target_preset == "attention_and_mlp"


def test_unsupported_target_preset_is_rejected(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["lora"]["target_preset"] = "everything"
    with pytest.raises(ValueError, match="unsupported lora.target_preset"):
        TransformersPeftRunSpec.from_resolved_config(
            config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
        )


def test_target_module_entries_must_be_non_empty_strings(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["lora"]["target_modules"] = ["q_proj", "   "]
    with pytest.raises(ValueError, match="non-empty strings"):
        TransformersPeftRunSpec.from_resolved_config(
            config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
        )


def test_recipe_digest_changes_with_target_preset(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    del config["backend"]["lora"]["target_modules"]
    a = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    config["backend"]["lora"]["target_preset"] = "attention_and_mlp"
    b = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    assert a.recipe_digest() != b.recipe_digest()


def test_resume_is_rejected_when_target_preset_changed(tmp_path, monkeypatch):
    """target_preset (and target_modules) are bound inputs, not excluded like
    epochs/max_steps -- changing the requested LoRA structure between save
    and resume would build a model whose state_dict shape no longer matches
    the checkpoint's."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    dataset_sha = sha256_file(data)

    checkpoint_trainer_dir = tmp_path / "prior" / "trainer"
    checkpoint_dir = checkpoint_trainer_dir / "checkpoint-50"
    checkpoint_dir.mkdir(parents=True)

    config = _config(str(data))
    config["backend"]["dataset_sha256"] = dataset_sha
    del config["backend"]["lora"]["target_modules"]
    spec_for_manifest = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "prior", seed=1
    )
    (checkpoint_trainer_dir / "chowder-checkpoint-manifest.json").write_text(
        json.dumps(TransformersPeftExecutor._bound_inputs(spec_for_manifest))
    )

    config["backend"]["lora"]["target_preset"] = "attention_and_mlp"
    config["backend"]["resume_from_checkpoint"] = str(checkpoint_dir)
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)

    def should_not_launch(*args, **kwargs):
        raise AssertionError("worker must not launch when bound inputs changed")

    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen", should_not_launch
    )
    with pytest.raises(ValueError, match="refusing to resume"):
        TransformersPeftExecutor().run(_experiment(), context)


# --- hardware-aware recipe defaults ------------------------------------------


def test_min_device_vram_gb_is_zero_when_hardware_is_unknown():
    assert _min_device_vram_gb(None) == 0.0


def test_min_device_vram_gb_uses_legacy_single_pool_when_topology_is_empty():
    assert _min_device_vram_gb(HardwareProfile(16, 64, 500, 12, 40, 3)) == 16


def test_min_device_vram_gb_uses_the_smallest_device_under_multi_gpu():
    hardware = HardwareProfile(16, 64, 500, 12, 40, 3, accelerator_vram_gb=(16.0, 6.0))
    assert _min_device_vram_gb(hardware) == 6.0


@pytest.mark.parametrize(
    "vram,expected",
    [(0.0, True), (8.0, True), (23.9, True), (24.0, False), (40.0, False)],
)
def test_default_gradient_checkpointing_flips_at_the_ample_vram_threshold(vram, expected):
    hardware = HardwareProfile(vram, 64, 500, 12, 40, 3)
    assert _default_gradient_checkpointing(hardware) is expected


def test_default_gradient_checkpointing_is_memory_safe_when_hardware_is_unknown():
    assert _default_gradient_checkpointing(None) is True


@pytest.mark.parametrize("vram", [0.0, 16.0, 40.0])
def test_default_quantization_is_none_outside_the_low_vram_band(vram, monkeypatch):
    monkeypatch.setattr(
        "chowder.backends.transformers_peft.importlib.util.find_spec", lambda name: object()
    )
    hardware = HardwareProfile(vram, 64, 500, 12, 40, 3)
    assert _default_quantization(hardware) == "none"


def test_default_quantization_is_4bit_in_the_low_vram_band_when_bitsandbytes_is_available(
    monkeypatch,
):
    monkeypatch.setattr(
        "chowder.backends.transformers_peft.importlib.util.find_spec", lambda name: object()
    )
    hardware = HardwareProfile(8.0, 64, 500, 12, 40, 3)
    assert _default_quantization(hardware) == "4bit"


def test_default_quantization_stays_none_in_the_low_vram_band_without_bitsandbytes(monkeypatch):
    monkeypatch.setattr(
        "chowder.backends.transformers_peft.importlib.util.find_spec", lambda name: None
    )
    hardware = HardwareProfile(8.0, 64, 500, 12, 40, 3)
    assert _default_quantization(hardware) == "none"


def test_default_quantization_is_none_when_hardware_is_unknown(monkeypatch):
    monkeypatch.setattr(
        "chowder.backends.transformers_peft.importlib.util.find_spec", lambda name: object()
    )
    assert _default_quantization(None) == "none"


def test_spec_uses_hardware_aware_defaults_when_fields_are_unset(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "chowder.backends.transformers_peft.importlib.util.find_spec", lambda name: object()
    )
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    del config["backend"]["quantization"]
    low_vram = HardwareProfile(8.0, 64, 500, 12, 40, 3)
    spec = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1, hardware=low_vram
    )
    assert spec.quantization == "4bit"
    assert spec.gradient_checkpointing is True

    ample_vram = HardwareProfile(40.0, 64, 500, 12, 40, 3)
    spec2 = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1, hardware=ample_vram
    )
    assert spec2.quantization == "none"
    assert spec2.gradient_checkpointing is False


def test_spec_explicit_config_overrides_hardware_aware_defaults(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["quantization"] = "none"
    config["backend"]["training"]["gradient_checkpointing"] = True
    low_vram = HardwareProfile(8.0, 64, 500, 12, 40, 3)
    spec = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1, hardware=low_vram
    )
    # Both would have defaulted the opposite way for 8GB VRAM -- the
    # explicit config values must win outright, not be second-guessed.
    assert spec.quantization == "none"
    assert spec.gradient_checkpointing is True


def test_spec_without_hardware_argument_preserves_prior_fixed_defaults(tmp_path):
    """Backward compatibility for callers that don't pass hardware at all
    (e.g. pre-existing direct TransformersPeftRunSpec construction elsewhere
    in the test suite) -- must behave exactly as it did before this feature:
    quantization "none", gradient_checkpointing True."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    del config["backend"]["quantization"]
    spec = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    assert spec.quantization == "none"
    assert spec.gradient_checkpointing is True


def test_recipe_digest_changes_with_hardware_aware_default_resolution(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    del config["backend"]["quantization"]
    a = TransformersPeftRunSpec.from_resolved_config(
        config,
        work_dir=tmp_path,
        output_dir=tmp_path / "adapter",
        seed=1,
        hardware=HardwareProfile(8.0, 64, 500, 12, 40, 3),
    )
    b = TransformersPeftRunSpec.from_resolved_config(
        config,
        work_dir=tmp_path,
        output_dir=tmp_path / "adapter",
        seed=1,
        hardware=HardwareProfile(40.0, 64, 500, 12, 40, 3),
    )
    assert a.recipe_digest() != b.recipe_digest()


def test_executor_threads_context_hardware_into_default_resolution(tmp_path, monkeypatch):
    """End-to-end within the executor (not just the spec builder directly):
    proves _spec_for actually passes context.hardware through, and that the
    resolved value reaches the recorded evidence."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    ample_vram = HardwareProfile(40.0, 64, 500, 12, 40, 3)
    context = ExecutionContext(ample_vram, str(tmp_path), 1, resolved_config=config)
    observed_spec: dict = {}
    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen",
        _fake_process_factory(observed_spec),
    )
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert observed_spec["gradient_checkpointing"] is False
    hardware_defaults = artifact.evidence["hardware_aware_defaults"]
    assert hardware_defaults["min_device_vram_gb"] == 40.0
    assert hardware_defaults["gradient_checkpointing_defaulted"] is True
    assert hardware_defaults["quantization_defaulted"] is False  # _config() sets it explicitly
    assert hardware_defaults["resolved_gradient_checkpointing"] is False


# --- offline / local-model mode ----------------------------------------------


def test_spec_defaults_offline_to_false(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    spec = TransformersPeftRunSpec.from_resolved_config(
        _config(str(data)), work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    assert spec.offline is False


def test_spec_reads_offline_from_resolved_config(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["offline"] = True
    spec = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    assert spec.offline is True


def test_recipe_digest_is_stable_across_offline_toggle(tmp_path):
    """offline is purely operational -- it changes how the model is
    fetched, never what training produces -- so it must not be part of the
    recipe digest, unlike quantization/target_preset/etc."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    a = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    config["backend"]["offline"] = True
    b = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    assert a.recipe_digest() == b.recipe_digest()


def test_resume_allows_a_different_offline_value(tmp_path, monkeypatch):
    """offline is excluded from the checkpoint-resume bound-inputs check
    the same way it's excluded from recipe_digest -- it was never a hazard
    to the loaded optimizer state to begin with."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    dataset_sha = sha256_file(data)

    checkpoint_trainer_dir = tmp_path / "prior" / "trainer"
    checkpoint_dir = checkpoint_trainer_dir / "checkpoint-50"
    checkpoint_dir.mkdir(parents=True)

    config = _config(str(data))
    config["backend"]["dataset_sha256"] = dataset_sha
    config["backend"]["offline"] = False
    spec_for_manifest = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "prior", seed=1
    )
    (checkpoint_trainer_dir / "chowder-checkpoint-manifest.json").write_text(
        json.dumps(TransformersPeftExecutor._bound_inputs(spec_for_manifest))
    )

    config["backend"]["offline"] = True
    config["backend"]["resume_from_checkpoint"] = str(checkpoint_dir)
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    observed_spec: dict = {}
    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen",
        _fake_process_factory(observed_spec),
    )
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert observed_spec["resume_from_checkpoint"] == str(checkpoint_dir.resolve())
    assert artifact is not None


# --- activation offload (production wiring) ---------------------------------


def test_resolve_activation_offload_flag_defaults_to_off_when_unset():
    """Absent entirely -- not "auto" -- must resolve to off: an existing
    project.json with no activation_offload key must train exactly as it
    always has, never suddenly pay for a real experiment subprocess."""
    assert _resolve_activation_offload_flag({}) is False


def test_resolve_activation_offload_flag_always_resolves_true():
    assert _resolve_activation_offload_flag({"activation_offload": "always"}) is True


def test_resolve_activation_offload_flag_off_resolves_false():
    assert _resolve_activation_offload_flag({"activation_offload": "off"}) is False


def test_resolve_activation_offload_flag_auto_resolves_false_here():
    """"auto" reached directly (not through
    TransformersPeftExecutor.resolved_activation_offload) must resolve to
    False, never trigger a real experiment -- this function is called from
    checkpoint discovery, memory_preflight's own spec construction, and
    the offload/tiering experiments' own spec construction, all of which
    must stay cheap and side-effect-free."""
    assert _resolve_activation_offload_flag({"activation_offload": "auto"}) is False


def test_resolve_activation_offload_flag_accepts_a_plain_boolean():
    assert _resolve_activation_offload_flag({"activation_offload": True}) is True
    assert _resolve_activation_offload_flag({"activation_offload": False}) is False


def test_resolve_activation_offload_flag_rejects_unknown_values():
    with pytest.raises(ValueError, match="activation_offload"):
        _resolve_activation_offload_flag({"activation_offload": "sometimes"})


def test_allowed_activation_offload_modes_is_the_documented_set():
    assert _ALLOWED_ACTIVATION_OFFLOAD_MODES == {"auto", "always", "off"}


def test_recipe_digest_is_stable_across_activation_offload_setting(tmp_path):
    """Value-transparent -- saved_tensors_hooks changes only where an
    intermediate tensor physically lives, never what gets computed -- so
    unlike gradient_checkpointing (which changes the actual computation),
    activation_offload must not be part of the recipe digest."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"]["activation_offload"] = "off"
    a = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    config["backend"]["training"]["activation_offload"] = "always"
    b = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    assert a.activation_offload != b.activation_offload
    assert a.recipe_digest() == b.recipe_digest()


def test_resume_allows_a_different_activation_offload_setting(tmp_path, monkeypatch):
    """Same exclusion, exercised through the actual checkpoint bound-inputs
    check TransformersPeftExecutor.run() performs before resuming -- a
    checkpoint saved under one offload setting must resume cleanly under a
    different one."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    dataset_sha = sha256_file(data)

    checkpoint_trainer_dir = tmp_path / "prior" / "trainer"
    checkpoint_dir = checkpoint_trainer_dir / "checkpoint-50"
    checkpoint_dir.mkdir(parents=True)

    config = _config(str(data))
    config["backend"]["dataset_sha256"] = dataset_sha
    config["backend"]["training"]["activation_offload"] = "always"
    spec_for_manifest = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "prior", seed=1
    )
    (checkpoint_trainer_dir / "chowder-checkpoint-manifest.json").write_text(
        json.dumps(TransformersPeftExecutor._bound_inputs(spec_for_manifest))
    )

    config["backend"]["training"]["activation_offload"] = "off"
    config["backend"]["resume_from_checkpoint"] = str(checkpoint_dir)
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    observed_spec: dict = {}
    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen",
        _fake_process_factory(observed_spec),
    )
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert artifact is not None
    assert observed_spec["activation_offload"] is False


def test_resolved_activation_offload_returns_true_for_always():
    config = {"backend": {"training": {"activation_offload": "always"}}}
    context = ExecutionContext(_hardware(), ".", 1, resolved_config=config)
    assert TransformersPeftExecutor.resolved_activation_offload(context) is True


def test_resolved_activation_offload_returns_false_for_off():
    config = {"backend": {"training": {"activation_offload": "off"}}}
    context = ExecutionContext(_hardware(), ".", 1, resolved_config=config)
    assert TransformersPeftExecutor.resolved_activation_offload(context) is False


def test_resolved_activation_offload_defaults_to_false_when_unset():
    config = {"backend": {"training": {}}}
    context = ExecutionContext(_hardware(), ".", 1, resolved_config=config)
    assert TransformersPeftExecutor.resolved_activation_offload(context) is False


def test_resolved_activation_offload_rejects_unknown_string():
    config = {"backend": {"training": {"activation_offload": "sometimes"}}}
    context = ExecutionContext(_hardware(), ".", 1, resolved_config=config)
    with pytest.raises(ValueError, match="activation_offload"):
        TransformersPeftExecutor.resolved_activation_offload(context)


def _fake_offload_experiment(recommended: bool, **overrides):
    from chowder.activation_offload import ActivationOffloadExperiment

    fields = dict(
        device="cuda", available=True, batch_size=2, max_length=64,
        baseline_peak_vram_gb=1.0, offload_peak_vram_gb=0.8, vram_saved_gb=0.2,
        baseline_wall_seconds=0.1, offload_wall_seconds=0.12,
        wall_time_penalty_ratio=1.2, per_rank_available_gb=16.0,
        required=False, recommended=recommended,
    )
    fields.update(overrides)
    return ActivationOffloadExperiment(**fields)


def test_resolved_activation_offload_auto_uses_the_real_experiments_recommendation():
    config = {"backend": {"training": {"activation_offload": "auto"}}}
    context = ExecutionContext(_hardware(), ".", 1, resolved_config=config)
    with patch(
        "chowder.activation_offload.run_activation_offload_experiment",
        return_value=_fake_offload_experiment(recommended=True),
    ) as mock_experiment:
        result = TransformersPeftExecutor.resolved_activation_offload(context)
    mock_experiment.assert_called_once()
    assert result is True


def test_resolved_activation_offload_auto_declines_when_not_recommended():
    config = {"backend": {"training": {"activation_offload": "auto"}}}
    context = ExecutionContext(_hardware(), ".", 1, resolved_config=config)
    with patch(
        "chowder.activation_offload.run_activation_offload_experiment",
        return_value=_fake_offload_experiment(recommended=False),
    ):
        result = TransformersPeftExecutor.resolved_activation_offload(context)
    assert result is False


def test_run_rejects_activation_offload_under_multi_gpu_ddp(tmp_path):
    """Explicit safe rejection, not silent best-effort: saved_tensors_hooks
    under DDP has not been proven safe on real multi-GPU hardware, so this
    combination must fail clearly and *before* any subprocess spawns --
    never train with an unverified interaction just because nothing
    crashed locally."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"]["activation_offload"] = "always"
    config["backend"]["runtime"] = {"active_accelerator_count": 2}
    hardware = HardwareProfile(16, 64, 500, 12, 40, 3, accelerator_vram_gb=(16.0, 6.0))
    context = ExecutionContext(hardware, str(tmp_path), 1, resolved_config=config)

    def should_not_spawn(*args, **kwargs):
        raise AssertionError("must reject before spawning any subprocess")

    with patch("chowder.backends.transformers_peft.subprocess.Popen", should_not_spawn):
        with pytest.raises(ValueError, match="multi-GPU DDP"):
            TransformersPeftExecutor().run(_experiment(), context)


def test_run_allows_activation_offload_off_under_multi_gpu_ddp(tmp_path, monkeypatch):
    """The rejection is specifically about offload being *active* under
    DDP -- off (the default) must never be rejected just because
    active_accelerator_count > 1."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"]["activation_offload"] = "off"
    config["backend"]["runtime"] = {"active_accelerator_count": 2}
    hardware = HardwareProfile(16, 64, 500, 12, 40, 3, accelerator_vram_gb=(16.0, 6.0))
    context = ExecutionContext(hardware, str(tmp_path), 1, resolved_config=config)
    observed_spec: dict = {}

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            spec_path = Path(command[command.index("--spec") + 1])
            result_path = Path(command[command.index("--result") + 1])
            spec = json.loads(spec_path.read_text())
            observed_spec.update(spec)
            output = Path(spec["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            (output / "adapter_model.safetensors").write_bytes(b"adapter")
            result_path.write_text(
                json.dumps(
                    {
                        "telemetry": {"train_loss": 0.25, "global_step": 3},
                        "versions": {"transformers": "5.test"},
                        "provenance": {"resolved_model_commit": "abc123"},
                        "data_provenance": {"primary_rows": 1, "replay_selected_rows": 0},
                        "resource_usage": {
                            "active_accelerator_count": 2,
                            "visible_accelerator_count": 2,
                            "peak_vram_gb_by_accelerator": {"cuda:0": 1.0, "cuda:1": 0.9},
                        },
                    }
                )
            )

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr("chowder.backends.transformers_peft.subprocess.Popen", FakeProcess)
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert artifact is not None
    assert observed_spec["activation_offload"] is False


def test_evidence_records_activation_offload_mode_and_resolution_for_always(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"]["activation_offload"] = "always"
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    observed_spec: dict = {}
    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen",
        _fake_process_factory(observed_spec),
    )
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert artifact.evidence["hardware_aware_defaults"]["resolved_activation_offload"] is True
    offload_evidence = artifact.evidence["activation_offload"]
    assert offload_evidence["mode"] == "always"
    assert offload_evidence["resolved"] is True
    # "always" is an explicit choice, not an experiment-driven decision --
    # there is no prediction to report.
    assert "predicted_recommended" not in offload_evidence


def test_evidence_records_predicted_vs_actual_for_auto(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"]["activation_offload"] = "auto"
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    observed_spec: dict = {}
    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen",
        _fake_process_factory(observed_spec),
    )
    with patch(
        "chowder.activation_offload.run_activation_offload_experiment",
        return_value=_fake_offload_experiment(recommended=True),
    ):
        artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert observed_spec["activation_offload"] is True
    offload_evidence = artifact.evidence["activation_offload"]
    assert offload_evidence["mode"] == "auto"
    assert offload_evidence["resolved"] is True
    assert offload_evidence["predicted_recommended"] is True
    assert offload_evidence["predicted_vram_saved_gb"] == pytest.approx(0.2)
    # The fake worker result in _fake_process_factory carries no
    # peak_vram_gb/train_runtime_seconds/activation_offload_bytes_
    # transferred -- confirms the "actual" fields degrade to None rather
    # than raising when that telemetry is absent.
    assert offload_evidence["actual_peak_vram_gb"] is None
    assert offload_evidence["actual_avg_step_seconds"] is None
    assert offload_evidence["actual_bytes_transferred"] is None


# --- optimizer tiering (production wiring) -----------------------------------


def test_resolve_optimizer_tiering_flag_defaults_to_off_when_unset():
    """Same "absent means off, not auto" default as activation_offload,
    for the same reason: an existing project.json with no
    optimizer_tiering key must train exactly as it always has."""
    assert _resolve_optimizer_tiering_flag({}) is False


def test_resolve_optimizer_tiering_flag_always_resolves_true():
    assert _resolve_optimizer_tiering_flag({"optimizer_tiering": "always"}) is True


def test_resolve_optimizer_tiering_flag_off_resolves_false():
    assert _resolve_optimizer_tiering_flag({"optimizer_tiering": "off"}) is False


def test_resolve_optimizer_tiering_flag_auto_resolves_false_here():
    assert _resolve_optimizer_tiering_flag({"optimizer_tiering": "auto"}) is False


def test_resolve_optimizer_tiering_flag_accepts_a_plain_boolean():
    assert _resolve_optimizer_tiering_flag({"optimizer_tiering": True}) is True
    assert _resolve_optimizer_tiering_flag({"optimizer_tiering": False}) is False


def test_resolve_optimizer_tiering_flag_rejects_unknown_values():
    with pytest.raises(ValueError, match="optimizer_tiering"):
        _resolve_optimizer_tiering_flag({"optimizer_tiering": "sometimes"})


def test_allowed_optimizer_tiering_modes_is_the_documented_set():
    assert _ALLOWED_OPTIMIZER_TIERING_MODES == {"auto", "always", "off"}


def test_recipe_digest_changes_with_optimizer_tiering_setting(tmp_path):
    """The opposite of activation_offload's exclusion: bitsandbytes' paged
    optimizers use their own internal state-dict keys (state1/state2),
    incompatible with torch.optim.AdamW's (exp_avg/exp_avg_sq) -- a real,
    confirmed KeyError on resume across implementations, not a
    value-transparent change. optimizer_tiering must stay part of the
    recipe digest."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"]["optimizer_tiering"] = "off"
    a = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    config["backend"]["training"]["optimizer_tiering"] = "always"
    b = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    assert a.optimizer_tiering != b.optimizer_tiering
    assert a.recipe_digest() != b.recipe_digest()


def test_resume_is_rejected_when_optimizer_tiering_setting_changed(tmp_path):
    """The opposite of activation_offload's resume-allowed test, exercised
    through the same real checkpoint bound-inputs check: a checkpoint
    saved under one optimizer_tiering setting must be REJECTED, not
    silently resumed, under a different one -- resuming into it would
    otherwise reach bitsandbytes' real KeyError deep inside a spawned
    worker instead of failing clearly at config-resolution time."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    dataset_sha = sha256_file(data)

    checkpoint_trainer_dir = tmp_path / "prior" / "trainer"
    checkpoint_dir = checkpoint_trainer_dir / "checkpoint-50"
    checkpoint_dir.mkdir(parents=True)

    config = _config(str(data))
    config["backend"]["dataset_sha256"] = dataset_sha
    config["backend"]["training"]["optimizer_tiering"] = "off"
    spec_for_manifest = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "prior", seed=1
    )
    (checkpoint_trainer_dir / "chowder-checkpoint-manifest.json").write_text(
        json.dumps(TransformersPeftExecutor._bound_inputs(spec_for_manifest))
    )

    config["backend"]["training"]["optimizer_tiering"] = "always"
    config["backend"]["resume_from_checkpoint"] = str(checkpoint_dir)
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)

    def should_not_spawn(*args, **kwargs):
        raise AssertionError("must reject before spawning any subprocess")

    with patch("chowder.backends.transformers_peft.subprocess.Popen", should_not_spawn):
        with pytest.raises(ValueError, match="bound training input"):
            TransformersPeftExecutor().run(_experiment(), context)


def test_resolved_optimizer_tiering_returns_true_for_always():
    config = {"backend": {"training": {"optimizer_tiering": "always"}}}
    context = ExecutionContext(_hardware(), ".", 1, resolved_config=config)
    assert TransformersPeftExecutor.resolved_optimizer_tiering(context) is True


def test_resolved_optimizer_tiering_returns_false_for_off():
    config = {"backend": {"training": {"optimizer_tiering": "off"}}}
    context = ExecutionContext(_hardware(), ".", 1, resolved_config=config)
    assert TransformersPeftExecutor.resolved_optimizer_tiering(context) is False


def test_resolved_optimizer_tiering_defaults_to_false_when_unset():
    config = {"backend": {"training": {}}}
    context = ExecutionContext(_hardware(), ".", 1, resolved_config=config)
    assert TransformersPeftExecutor.resolved_optimizer_tiering(context) is False


def test_resolved_optimizer_tiering_rejects_unknown_string():
    config = {"backend": {"training": {"optimizer_tiering": "sometimes"}}}
    context = ExecutionContext(_hardware(), ".", 1, resolved_config=config)
    with pytest.raises(ValueError, match="optimizer_tiering"):
        TransformersPeftExecutor.resolved_optimizer_tiering(context)


def _fake_tiering_experiment(recommended: bool, **overrides):
    from chowder.optimizer_tiering import OptimizerTieringExperiment, OptimizerVariantMeasurement

    fields = dict(
        device="cuda", available=True, batch_size=2, max_length=64,
        variants=(
            OptimizerVariantMeasurement(name="adamw", step_seconds=0.05, state_bytes=1000),
            OptimizerVariantMeasurement(name="paged_adamw", step_seconds=0.06, state_bytes=1000),
            OptimizerVariantMeasurement(name="paged_adamw_8bit", step_seconds=0.04, state_bytes=500),
        ),
        model_peak_vram_gb=1.0, per_rank_available_gb=16.0,
        wall_time_penalty_ratio=1.2, required=False, recommended=recommended,
    )
    fields.update(overrides)
    return OptimizerTieringExperiment(**fields)


def test_resolved_optimizer_tiering_auto_uses_the_real_experiments_recommendation():
    config = {"backend": {"training": {"optimizer_tiering": "auto"}}}
    context = ExecutionContext(_hardware(), ".", 1, resolved_config=config)
    with patch(
        "chowder.optimizer_tiering.run_optimizer_tiering_experiment",
        return_value=_fake_tiering_experiment(recommended=True),
    ) as mock_experiment:
        result = TransformersPeftExecutor.resolved_optimizer_tiering(context)
    mock_experiment.assert_called_once()
    assert result is True


def test_resolved_optimizer_tiering_auto_declines_when_not_recommended():
    config = {"backend": {"training": {"optimizer_tiering": "auto"}}}
    context = ExecutionContext(_hardware(), ".", 1, resolved_config=config)
    with patch(
        "chowder.optimizer_tiering.run_optimizer_tiering_experiment",
        return_value=_fake_tiering_experiment(recommended=False),
    ):
        result = TransformersPeftExecutor.resolved_optimizer_tiering(context)
    assert result is False


def test_run_rejects_optimizer_tiering_under_multi_gpu_ddp(tmp_path):
    """Same explicit-safe-rejection principle as activation_offload: this
    combination has not been proven safe on real multi-GPU hardware here,
    so it must fail clearly and before any subprocess spawns."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"]["optimizer_tiering"] = "always"
    config["backend"]["runtime"] = {"active_accelerator_count": 2}
    hardware = HardwareProfile(16, 64, 500, 12, 40, 3, accelerator_vram_gb=(16.0, 6.0))
    context = ExecutionContext(hardware, str(tmp_path), 1, resolved_config=config)

    def should_not_spawn(*args, **kwargs):
        raise AssertionError("must reject before spawning any subprocess")

    with patch("chowder.backends.transformers_peft.subprocess.Popen", should_not_spawn):
        with pytest.raises(ValueError, match="multi-GPU DDP"):
            TransformersPeftExecutor().run(_experiment(), context)


def test_run_allows_optimizer_tiering_off_under_multi_gpu_ddp(tmp_path, monkeypatch):
    """The rejection is specifically about tiering being *active* under
    DDP -- off (the default) must never be rejected just because
    active_accelerator_count > 1."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"]["optimizer_tiering"] = "off"
    config["backend"]["runtime"] = {"active_accelerator_count": 2}
    hardware = HardwareProfile(16, 64, 500, 12, 40, 3, accelerator_vram_gb=(16.0, 6.0))
    context = ExecutionContext(hardware, str(tmp_path), 1, resolved_config=config)
    observed_spec: dict = {}

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            spec_path = Path(command[command.index("--spec") + 1])
            result_path = Path(command[command.index("--result") + 1])
            spec = json.loads(spec_path.read_text())
            observed_spec.update(spec)
            output = Path(spec["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            (output / "adapter_model.safetensors").write_bytes(b"adapter")
            result_path.write_text(
                json.dumps(
                    {
                        "telemetry": {"train_loss": 0.25, "global_step": 3},
                        "versions": {"transformers": "5.test"},
                        "provenance": {"resolved_model_commit": "abc123"},
                        "data_provenance": {"primary_rows": 1, "replay_selected_rows": 0},
                        "resource_usage": {
                            "active_accelerator_count": 2,
                            "visible_accelerator_count": 2,
                            "peak_vram_gb_by_accelerator": {"cuda:0": 1.0, "cuda:1": 0.9},
                        },
                    }
                )
            )

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr("chowder.backends.transformers_peft.subprocess.Popen", FakeProcess)
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert artifact is not None
    assert observed_spec["optimizer_tiering"] is False


def test_evidence_records_optimizer_tiering_mode_and_resolution_for_always(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"]["optimizer_tiering"] = "always"
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    observed_spec: dict = {}
    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen",
        _fake_process_factory(observed_spec),
    )
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert artifact.evidence["hardware_aware_defaults"]["resolved_optimizer_tiering"] is True
    tiering_evidence = artifact.evidence["optimizer_tiering"]
    assert tiering_evidence["mode"] == "always"
    assert tiering_evidence["resolved"] is True
    # "always" is an explicit choice, not an experiment-driven decision --
    # there is no prediction to report.
    assert "predicted_recommended" not in tiering_evidence


def test_evidence_records_predicted_vs_actual_for_optimizer_tiering_auto(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"]["optimizer_tiering"] = "auto"
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    observed_spec: dict = {}
    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen",
        _fake_process_factory(observed_spec),
    )
    with patch(
        "chowder.optimizer_tiering.run_optimizer_tiering_experiment",
        return_value=_fake_tiering_experiment(recommended=True),
    ):
        artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert observed_spec["optimizer_tiering"] is True
    tiering_evidence = artifact.evidence["optimizer_tiering"]
    assert tiering_evidence["mode"] == "auto"
    assert tiering_evidence["resolved"] is True
    assert tiering_evidence["predicted_recommended"] is True
    assert tiering_evidence["predicted_baseline_state_bytes"] == 1000
    assert tiering_evidence["predicted_paged_state_bytes"] == 1000
    # The fake worker result in _fake_process_factory carries no
    # global_step/train_runtime_seconds/optimizer_state_bytes -- confirms
    # the "actual" fields degrade to None rather than raising when that
    # telemetry is absent.
    assert tiering_evidence["actual_avg_step_seconds"] is None
    assert tiering_evidence["actual_optimizer_state_bytes"] is None


# --- frozen layer streaming (production wiring) -------------------------------


def test_resolve_frozen_layer_streaming_flag_defaults_to_off_when_unset():
    """Same "absent means off, not auto" default as activation_offload/
    optimizer_tiering, for the same reason: an existing project.json
    with no frozen_layer_streaming key must train exactly as it always
    has."""
    assert _resolve_frozen_layer_streaming_flag({}) is False


def test_resolve_frozen_layer_streaming_flag_always_resolves_true():
    assert _resolve_frozen_layer_streaming_flag({"frozen_layer_streaming": "always"}) is True


def test_resolve_frozen_layer_streaming_flag_off_resolves_false():
    assert _resolve_frozen_layer_streaming_flag({"frozen_layer_streaming": "off"}) is False


def test_resolve_frozen_layer_streaming_flag_auto_resolves_false_here():
    assert _resolve_frozen_layer_streaming_flag({"frozen_layer_streaming": "auto"}) is False


def test_resolve_frozen_layer_streaming_flag_accepts_a_plain_boolean():
    assert _resolve_frozen_layer_streaming_flag({"frozen_layer_streaming": True}) is True
    assert _resolve_frozen_layer_streaming_flag({"frozen_layer_streaming": False}) is False


def test_resolve_frozen_layer_streaming_flag_rejects_unknown_values():
    with pytest.raises(ValueError, match="frozen_layer_streaming"):
        _resolve_frozen_layer_streaming_flag({"frozen_layer_streaming": "sometimes"})


def test_allowed_frozen_layer_streaming_modes_is_the_documented_set():
    assert _ALLOWED_FROZEN_LAYER_STREAMING_MODES == {"auto", "always", "off"}


def test_recipe_digest_is_stable_across_frozen_layer_streaming_setting(tmp_path):
    """Value-transparent -- the custom autograd.Function changes only
    where a frozen layer's weight physically lives during compute, never
    what gets computed, and the checkpoint never contains the frozen
    base weights either way -- so unlike optimizer_tiering,
    frozen_layer_streaming must not be part of the recipe digest."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"]["frozen_layer_streaming"] = "off"
    a = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    config["backend"]["training"]["frozen_layer_streaming"] = "always"
    b = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    assert a.frozen_layer_streaming != b.frozen_layer_streaming
    assert a.recipe_digest() == b.recipe_digest()


def test_resume_allows_a_different_frozen_layer_streaming_setting(tmp_path, monkeypatch):
    """Same exclusion as activation_offload, exercised through the actual
    checkpoint bound-inputs check TransformersPeftExecutor.run() performs
    before resuming -- a checkpoint saved under one streaming setting
    must resume cleanly under a different one."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    dataset_sha = sha256_file(data)

    checkpoint_trainer_dir = tmp_path / "prior" / "trainer"
    checkpoint_dir = checkpoint_trainer_dir / "checkpoint-50"
    checkpoint_dir.mkdir(parents=True)

    config = _config(str(data))
    config["backend"]["dataset_sha256"] = dataset_sha
    config["backend"]["training"]["frozen_layer_streaming"] = "always"
    spec_for_manifest = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "prior", seed=1
    )
    (checkpoint_trainer_dir / "chowder-checkpoint-manifest.json").write_text(
        json.dumps(TransformersPeftExecutor._bound_inputs(spec_for_manifest))
    )

    config["backend"]["training"]["frozen_layer_streaming"] = "off"
    config["backend"]["resume_from_checkpoint"] = str(checkpoint_dir)
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    observed_spec: dict = {}
    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen",
        _fake_process_factory(observed_spec),
    )
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert artifact is not None
    assert observed_spec["frozen_layer_streaming"] is False


def test_resolved_frozen_layer_streaming_returns_true_for_always():
    config = {"backend": {"training": {"frozen_layer_streaming": "always"}}}
    context = ExecutionContext(_hardware(), ".", 1, resolved_config=config)
    assert TransformersPeftExecutor.resolved_frozen_layer_streaming(context) is True


def test_resolved_frozen_layer_streaming_returns_false_for_off():
    config = {"backend": {"training": {"frozen_layer_streaming": "off"}}}
    context = ExecutionContext(_hardware(), ".", 1, resolved_config=config)
    assert TransformersPeftExecutor.resolved_frozen_layer_streaming(context) is False


def test_resolved_frozen_layer_streaming_defaults_to_false_when_unset():
    config = {"backend": {"training": {}}}
    context = ExecutionContext(_hardware(), ".", 1, resolved_config=config)
    assert TransformersPeftExecutor.resolved_frozen_layer_streaming(context) is False


def test_resolved_frozen_layer_streaming_rejects_unknown_string():
    config = {"backend": {"training": {"frozen_layer_streaming": "sometimes"}}}
    context = ExecutionContext(_hardware(), ".", 1, resolved_config=config)
    with pytest.raises(ValueError, match="frozen_layer_streaming"):
        TransformersPeftExecutor.resolved_frozen_layer_streaming(context)


def _fake_streaming_experiment(recommended: bool, **overrides):
    from chowder.frozen_layer_streaming import FrozenLayerStreamingExperiment

    fields = dict(
        device="cuda", available=True, batch_size=2, max_length=64,
        baseline_peak_vram_gb=1.0, streamed_peak_vram_gb=0.8, vram_saved_gb=0.2,
        baseline_wall_seconds=0.1, streamed_wall_seconds=0.12,
        wall_time_penalty_ratio=1.2, bytes_transferred_per_step=1000,
        per_rank_available_gb=16.0, required=False, recommended=recommended,
    )
    fields.update(overrides)
    return FrozenLayerStreamingExperiment(**fields)


def test_resolved_frozen_layer_streaming_auto_uses_the_real_experiments_recommendation():
    config = {"backend": {"training": {"frozen_layer_streaming": "auto"}}}
    context = ExecutionContext(_hardware(), ".", 1, resolved_config=config)
    with patch(
        "chowder.frozen_layer_streaming.run_frozen_layer_streaming_experiment",
        return_value=_fake_streaming_experiment(recommended=True),
    ) as mock_experiment:
        result = TransformersPeftExecutor.resolved_frozen_layer_streaming(context)
    mock_experiment.assert_called_once()
    assert result is True


def test_resolved_frozen_layer_streaming_auto_declines_when_not_recommended():
    config = {"backend": {"training": {"frozen_layer_streaming": "auto"}}}
    context = ExecutionContext(_hardware(), ".", 1, resolved_config=config)
    with patch(
        "chowder.frozen_layer_streaming.run_frozen_layer_streaming_experiment",
        return_value=_fake_streaming_experiment(recommended=False),
    ):
        result = TransformersPeftExecutor.resolved_frozen_layer_streaming(context)
    assert result is False


def test_run_rejects_frozen_layer_streaming_under_multi_gpu_ddp(tmp_path):
    """Explicit safe rejection, not silent best-effort: the custom
    autograd.Function and dedicated CUDA prefetch stream have not been
    proven safe on real multi-GPU hardware, so this combination must
    fail clearly and *before* any subprocess spawns."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"]["frozen_layer_streaming"] = "always"
    config["backend"]["runtime"] = {"active_accelerator_count": 2}
    hardware = HardwareProfile(16, 64, 500, 12, 40, 3, accelerator_vram_gb=(16.0, 6.0))
    context = ExecutionContext(hardware, str(tmp_path), 1, resolved_config=config)

    def should_not_spawn(*args, **kwargs):
        raise AssertionError("must reject before spawning any subprocess")

    with patch("chowder.backends.transformers_peft.subprocess.Popen", should_not_spawn):
        with pytest.raises(ValueError, match="multi-GPU DDP"):
            TransformersPeftExecutor().run(_experiment(), context)


def test_run_allows_frozen_layer_streaming_off_under_multi_gpu_ddp(tmp_path, monkeypatch):
    """The rejection is specifically about streaming being *active* under
    DDP -- off (the default) must never be rejected just because
    active_accelerator_count > 1."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"]["frozen_layer_streaming"] = "off"
    config["backend"]["runtime"] = {"active_accelerator_count": 2}
    hardware = HardwareProfile(16, 64, 500, 12, 40, 3, accelerator_vram_gb=(16.0, 6.0))
    context = ExecutionContext(hardware, str(tmp_path), 1, resolved_config=config)
    observed_spec: dict = {}

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            spec_path = Path(command[command.index("--spec") + 1])
            result_path = Path(command[command.index("--result") + 1])
            spec = json.loads(spec_path.read_text())
            observed_spec.update(spec)
            output = Path(spec["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            (output / "adapter_model.safetensors").write_bytes(b"adapter")
            result_path.write_text(
                json.dumps(
                    {
                        "telemetry": {"train_loss": 0.25, "global_step": 3},
                        "versions": {"transformers": "5.test"},
                        "provenance": {"resolved_model_commit": "abc123"},
                        "data_provenance": {"primary_rows": 1, "replay_selected_rows": 0},
                        "resource_usage": {
                            "active_accelerator_count": 2,
                            "visible_accelerator_count": 2,
                            "peak_vram_gb_by_accelerator": {"cuda:0": 1.0, "cuda:1": 0.9},
                        },
                    }
                )
            )

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr("chowder.backends.transformers_peft.subprocess.Popen", FakeProcess)
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert artifact is not None
    assert observed_spec["frozen_layer_streaming"] is False


def test_evidence_records_frozen_layer_streaming_mode_and_resolution_for_always(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"]["frozen_layer_streaming"] = "always"
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    observed_spec: dict = {}
    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen",
        _fake_process_factory(observed_spec),
    )
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert artifact.evidence["hardware_aware_defaults"]["resolved_frozen_layer_streaming"] is True
    streaming_evidence = artifact.evidence["frozen_layer_streaming"]
    assert streaming_evidence["mode"] == "always"
    assert streaming_evidence["resolved"] is True
    # "always" is an explicit choice, not an experiment-driven decision --
    # there is no prediction to report.
    assert "predicted_recommended" not in streaming_evidence


def test_evidence_records_predicted_vs_actual_for_frozen_layer_streaming_auto(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"]["frozen_layer_streaming"] = "auto"
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    observed_spec: dict = {}
    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen",
        _fake_process_factory(observed_spec),
    )
    with patch(
        "chowder.frozen_layer_streaming.run_frozen_layer_streaming_experiment",
        return_value=_fake_streaming_experiment(recommended=True),
    ):
        artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert observed_spec["frozen_layer_streaming"] is True
    streaming_evidence = artifact.evidence["frozen_layer_streaming"]
    assert streaming_evidence["mode"] == "auto"
    assert streaming_evidence["resolved"] is True
    assert streaming_evidence["predicted_recommended"] is True
    assert streaming_evidence["predicted_vram_saved_gb"] == pytest.approx(0.2)
    # The fake worker result in _fake_process_factory carries no
    # global_step/train_runtime_seconds/frozen_layer_streaming_bytes_
    # transferred -- confirms the "actual" fields degrade to None rather
    # than raising when that telemetry is absent.
    assert streaming_evidence["actual_avg_step_seconds"] is None
    assert streaming_evidence["actual_bytes_transferred"] is None


# --- checkpoint/restart -----------------------------------------------------


def test_save_strategy_and_resume_are_read_from_resolved_config(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    checkpoint_dir = tmp_path / "prior" / "trainer" / "checkpoint-50"
    checkpoint_dir.mkdir(parents=True)
    config = _config(str(data))
    config["backend"]["training"]["save_strategy"] = "steps"
    config["backend"]["training"]["save_steps"] = 25
    config["backend"]["training"]["save_total_limit"] = 3
    config["backend"]["resume_from_checkpoint"] = str(checkpoint_dir)
    spec = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    assert spec.save_strategy == "steps"
    assert spec.save_steps == 25
    assert spec.save_total_limit == 3
    assert spec.resume_from_checkpoint == str(checkpoint_dir.resolve())


def test_save_steps_required_when_strategy_is_steps(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"]["save_strategy"] = "steps"
    with pytest.raises(ValueError, match="save_steps must be positive"):
        TransformersPeftRunSpec.from_resolved_config(
            config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
        )


def test_unsupported_save_strategy_is_rejected(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"]["save_strategy"] = "every_full_moon"
    with pytest.raises(ValueError, match="unsupported save_strategy"):
        TransformersPeftRunSpec.from_resolved_config(
            config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
        )


def test_recipe_digest_is_stable_across_checkpoint_cadence(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["dataset_sha256"] = sha256_file(data)
    a = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "a", seed=1
    )
    config["backend"]["training"]["save_strategy"] = "steps"
    config["backend"]["training"]["save_steps"] = 10
    config["backend"]["training"]["save_total_limit"] = 2
    b = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "b", seed=1
    )
    assert a.recipe_digest() == b.recipe_digest()


def _fake_process_factory(observed_spec: dict):
    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            spec_path = Path(command[command.index("--spec") + 1])
            result_path = Path(command[command.index("--result") + 1])
            spec = json.loads(spec_path.read_text())
            observed_spec.update(spec)
            output = Path(spec["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            (output / "adapter_model.safetensors").write_bytes(b"adapter")
            if spec.get("save_strategy") != "no":
                (Path(output) / "trainer").mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps(
                    {
                        "telemetry": {"train_loss": 0.25, "global_step": 3},
                        "versions": {"transformers": "5.test"},
                        "provenance": {"resolved_model_commit": "abc123"},
                        "data_provenance": {"primary_rows": 1, "replay_selected_rows": 0},
                    }
                )
            )

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    return FakeProcess


def test_run_writes_a_checkpoint_manifest_when_save_strategy_enabled(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["training"]["save_strategy"] = "steps"
    config["backend"]["training"]["save_steps"] = 5
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    observed_spec: dict = {}
    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen",
        _fake_process_factory(observed_spec),
    )
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    manifest_path = Path(artifact.artifact_ref) / "trainer" / "chowder-checkpoint-manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    # Deliberately its own digest, not artifact.evidence["recipe_sha256"] --
    # see TransformersPeftExecutor._bound_inputs's docstring for why (it
    # excludes epochs, spec.recipe_digest() does not).
    assert len(manifest["checkpoint_recipe_sha256"]) == 64
    assert manifest["dataset_sha256"] == sha256_file(data)
    assert artifact.evidence["checkpoint"]["save_strategy"] == "steps"
    assert artifact.evidence["checkpoint"]["save_steps"] == 5


def test_run_writes_no_manifest_when_save_strategy_is_default_no(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    context = ExecutionContext(
        _hardware(), str(tmp_path), 1, resolved_config=_config(str(data))
    )
    observed_spec: dict = {}
    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen",
        _fake_process_factory(observed_spec),
    )
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert artifact.evidence["checkpoint"]["trainer_dir"] is None
    assert not (Path(artifact.artifact_ref) / "trainer").exists()


def test_resume_is_rejected_when_no_checkpoint_manifest_exists(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    checkpoint_dir = tmp_path / "prior" / "trainer" / "checkpoint-50"
    checkpoint_dir.mkdir(parents=True)  # no chowder-checkpoint-manifest.json alongside it
    config = _config(str(data))
    config["backend"]["resume_from_checkpoint"] = str(checkpoint_dir)
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)

    def should_not_launch(*args, **kwargs):
        raise AssertionError("worker must not launch for an unverifiable checkpoint")

    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen", should_not_launch
    )
    with pytest.raises(RuntimeError, match="no checkpoint manifest found"):
        TransformersPeftExecutor().run(_experiment(), context)


def test_resume_is_rejected_when_dataset_changed_since_checkpoint(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"original"}\n')
    original_sha = sha256_file(data)

    checkpoint_trainer_dir = tmp_path / "prior" / "trainer"
    checkpoint_dir = checkpoint_trainer_dir / "checkpoint-50"
    checkpoint_dir.mkdir(parents=True)

    config = _config(str(data))
    config["backend"]["dataset_sha256"] = original_sha
    spec_for_manifest = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "prior", seed=1
    )
    (checkpoint_trainer_dir / "chowder-checkpoint-manifest.json").write_text(
        json.dumps(TransformersPeftExecutor._bound_inputs(spec_for_manifest))
    )

    # A different dataset now bound to the same checkpoint path -- e.g. a
    # differently-configured retry pointed resume_from_checkpoint at someone
    # else's run directory, or the training data was edited in place.
    data2 = tmp_path / "train2.jsonl"
    data2.write_text('{"text":"different"}\n')
    config2 = _config(str(data2))
    config2["backend"]["dataset_sha256"] = sha256_file(data2)
    config2["backend"]["resume_from_checkpoint"] = str(checkpoint_dir)
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config2)

    def should_not_launch(*args, **kwargs):
        raise AssertionError("worker must not launch when bound inputs changed")

    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen", should_not_launch
    )
    with pytest.raises(ValueError, match="refusing to resume"):
        TransformersPeftExecutor().run(_experiment(), context)


def test_resume_succeeds_when_bound_inputs_match(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    dataset_sha = sha256_file(data)

    checkpoint_trainer_dir = tmp_path / "prior" / "trainer"
    checkpoint_dir = checkpoint_trainer_dir / "checkpoint-50"
    checkpoint_dir.mkdir(parents=True)

    config = _config(str(data))
    config["backend"]["dataset_sha256"] = dataset_sha
    spec_for_manifest = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "prior", seed=1
    )
    (checkpoint_trainer_dir / "chowder-checkpoint-manifest.json").write_text(
        json.dumps(TransformersPeftExecutor._bound_inputs(spec_for_manifest))
    )

    config["backend"]["resume_from_checkpoint"] = str(checkpoint_dir)
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    observed_spec: dict = {}
    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen",
        _fake_process_factory(observed_spec),
    )
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert observed_spec["resume_from_checkpoint"] == str(checkpoint_dir.resolve())
    assert artifact.evidence["checkpoint"]["resumed_from_checkpoint"] == str(
        checkpoint_dir.resolve()
    )


def test_resume_allows_a_different_total_epoch_count(tmp_path, monkeypatch):
    """Training for more total epochs than originally planned is the whole
    point of resuming -- it must not be treated as a bound-input change."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    dataset_sha = sha256_file(data)

    checkpoint_trainer_dir = tmp_path / "prior" / "trainer"
    checkpoint_dir = checkpoint_trainer_dir / "checkpoint-50"
    checkpoint_dir.mkdir(parents=True)

    config = _config(str(data))
    config["backend"]["dataset_sha256"] = dataset_sha
    config["backend"]["training"]["epochs"] = 1
    spec_for_manifest = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "prior", seed=1
    )
    (checkpoint_trainer_dir / "chowder-checkpoint-manifest.json").write_text(
        json.dumps(TransformersPeftExecutor._bound_inputs(spec_for_manifest))
    )

    config["backend"]["training"]["epochs"] = 5  # extend, not restart
    config["backend"]["resume_from_checkpoint"] = str(checkpoint_dir)
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    observed_spec: dict = {}
    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen",
        _fake_process_factory(observed_spec),
    )
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert observed_spec["resume_from_checkpoint"] == str(checkpoint_dir.resolve())
    assert artifact is not None


def test_resume_allows_a_different_max_steps(tmp_path, monkeypatch):
    """max_steps is excluded from the bound-inputs check for the same reason
    epochs is: extending the total training length is the point of
    resuming, not a hazard to the loaded optimizer state."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    dataset_sha = sha256_file(data)

    checkpoint_trainer_dir = tmp_path / "prior" / "trainer"
    checkpoint_dir = checkpoint_trainer_dir / "checkpoint-50"
    checkpoint_dir.mkdir(parents=True)

    config = _config(str(data))
    config["backend"]["dataset_sha256"] = dataset_sha
    config["backend"]["training"]["max_steps"] = 50
    spec_for_manifest = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "prior", seed=1
    )
    (checkpoint_trainer_dir / "chowder-checkpoint-manifest.json").write_text(
        json.dumps(TransformersPeftExecutor._bound_inputs(spec_for_manifest))
    )

    config["backend"]["training"]["max_steps"] = 200  # extend, not restart
    config["backend"]["resume_from_checkpoint"] = str(checkpoint_dir)
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    observed_spec: dict = {}
    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen",
        _fake_process_factory(observed_spec),
    )
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert observed_spec["resume_from_checkpoint"] == str(checkpoint_dir.resolve())
    assert artifact is not None


def test_resume_is_rejected_when_weight_decay_changed(tmp_path, monkeypatch):
    """Unlike epochs/max_steps, weight_decay shapes the optimizer trajectory
    itself -- changing it mid-resume must be rejected the same way a changed
    learning rate or batch size already is."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    dataset_sha = sha256_file(data)

    checkpoint_trainer_dir = tmp_path / "prior" / "trainer"
    checkpoint_dir = checkpoint_trainer_dir / "checkpoint-50"
    checkpoint_dir.mkdir(parents=True)

    config = _config(str(data))
    config["backend"]["dataset_sha256"] = dataset_sha
    config["backend"]["training"]["weight_decay"] = 0.0
    spec_for_manifest = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "prior", seed=1
    )
    (checkpoint_trainer_dir / "chowder-checkpoint-manifest.json").write_text(
        json.dumps(TransformersPeftExecutor._bound_inputs(spec_for_manifest))
    )

    config["backend"]["training"]["weight_decay"] = 0.05
    config["backend"]["resume_from_checkpoint"] = str(checkpoint_dir)
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)

    def should_not_launch(*args, **kwargs):
        raise AssertionError("worker must not launch when bound inputs changed")

    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen", should_not_launch
    )
    with pytest.raises(ValueError, match="refusing to resume"):
        TransformersPeftExecutor().run(_experiment(), context)


def test_worker_command_is_a_plain_single_process_for_one_accelerator(tmp_path):
    spec_path = tmp_path / "spec.json"
    result_path = tmp_path / "result.json"
    command = TransformersPeftExecutor._worker_command(
        spec_path, result_path, active_accelerator_count=1
    )
    assert command == [
        sys.executable,
        "-m",
        "chowder.backends.transformers_worker",
        "--spec",
        str(spec_path),
        "--result",
        str(result_path),
    ]


def test_worker_command_uses_accelerate_launch_for_multiple_accelerators(tmp_path):
    spec_path = tmp_path / "spec.json"
    result_path = tmp_path / "result.json"
    command = TransformersPeftExecutor._worker_command(
        spec_path, result_path, active_accelerator_count=2
    )
    assert command == [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--multi_gpu",
        "--num_processes=2",
        "--num_machines=1",
        "-m",
        "chowder.backends.transformers_worker",
        "--spec",
        str(spec_path),
        "--result",
        str(result_path),
    ]


def test_active_accelerator_count_defaults_to_one_when_unset(tmp_path):
    context = ExecutionContext(
        _hardware(), str(tmp_path), 1, resolved_config=_config("train.jsonl")
    )
    assert TransformersPeftExecutor._active_accelerator_count(context) == 1


def test_active_accelerator_count_reads_runtime_config_against_real_topology(tmp_path):
    config = _config("train.jsonl")
    config["backend"]["runtime"] = {"active_accelerator_count": 2}
    hardware = HardwareProfile(16, 64, 500, 12, 40, 3, accelerator_vram_gb=(16.0, 6.0))
    context = ExecutionContext(hardware, str(tmp_path), 1, resolved_config=config)
    assert TransformersPeftExecutor._active_accelerator_count(context) == 2


def test_active_accelerator_count_accepts_zero_as_no_accelerator_declared(tmp_path):
    # 0 is an established sentinel elsewhere (_profile_accelerator_count) for
    # "no accelerator used" -- e.g. CPU-only smoke configs -- distinct from
    # an unset value (which defaults to 1). It must still resolve to a plain
    # single-process launch, not be rejected.
    config = _config("train.jsonl")
    config["backend"]["runtime"] = {"active_accelerator_count": 0}
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    assert TransformersPeftExecutor._active_accelerator_count(context) == 0


def test_active_accelerator_count_rejects_negative_count(tmp_path):
    config = _config("train.jsonl")
    config["backend"]["runtime"] = {"active_accelerator_count": -1}
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    with pytest.raises(ValueError, match="cannot be negative"):
        TransformersPeftExecutor._active_accelerator_count(context)


def test_active_accelerator_count_rejects_count_above_legacy_single_pool_hardware(
    tmp_path,
):
    # _hardware() only sets the legacy vram_gb scalar (accelerator_vram_gb is
    # empty) -- exactly the shape most HardwareProfile construction still
    # uses. That must read as one visible device, not zero.
    config = _config("train.jsonl")
    config["backend"]["runtime"] = {"active_accelerator_count": 2}
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    with pytest.raises(ValueError, match=r"more accelerators than are visible \(1\)"):
        TransformersPeftExecutor._active_accelerator_count(context)


def test_active_accelerator_count_rejects_count_above_real_topology(tmp_path):
    config = _config("train.jsonl")
    config["backend"]["runtime"] = {"active_accelerator_count": 3}
    hardware = HardwareProfile(16, 64, 500, 12, 40, 3, accelerator_vram_gb=(16.0, 6.0))
    context = ExecutionContext(hardware, str(tmp_path), 1, resolved_config=config)
    with pytest.raises(ValueError, match=r"more accelerators than are visible \(2\)"):
        TransformersPeftExecutor._active_accelerator_count(context)


def test_executor_launches_accelerate_when_multiple_accelerators_requested(
    tmp_path, monkeypatch
):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["runtime"] = {"active_accelerator_count": 2}
    hardware = HardwareProfile(16, 64, 500, 12, 40, 3, accelerator_vram_gb=(16.0, 6.0))
    context = ExecutionContext(hardware, str(tmp_path), 1, resolved_config=config)

    observed: dict = {}

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            observed["command"] = command
            spec_path = Path(command[command.index("--spec") + 1])
            result_path = Path(command[command.index("--result") + 1])
            spec = json.loads(spec_path.read_text())
            output = Path(spec["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            (output / "adapter_model.safetensors").write_bytes(b"adapter")
            result_path.write_text(
                json.dumps(
                    {
                        "telemetry": {"train_loss": 0.25, "global_step": 3},
                        "versions": {"transformers": "5.test"},
                        "provenance": {"resolved_model_commit": "abc123"},
                        "data_provenance": {
                            "primary_rows": 1,
                            "replay_selected_rows": 0,
                        },
                        "resource_usage": {
                            "active_accelerator_count": 2,
                            "visible_accelerator_count": 2,
                            "peak_vram_gb_by_accelerator": {
                                "cuda:0": 1.0,
                                "cuda:1": 0.9,
                            },
                        },
                    }
                )
            )

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen", FakeProcess
    )
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    command = observed["command"]
    assert command[0] == sys.executable
    assert command[1:5] == [
        "-m",
        "accelerate.commands.launch",
        "--multi_gpu",
        "--num_processes=2",
    ]
    assert artifact.evidence["requested_active_accelerator_count"] == 2
    assert artifact.evidence["resource_usage"]["active_accelerator_count"] == 2


def test_executor_rejects_a_worker_that_did_not_engage_every_requested_accelerator(
    tmp_path, monkeypatch
):
    """A worker whose own resource snapshot reports fewer active accelerators
    than requested means accelerate launch silently failed to engage every
    device -- the run's GPU-hour accounting would be wrong if trusted."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["runtime"] = {"active_accelerator_count": 2}
    hardware = HardwareProfile(16, 64, 500, 12, 40, 3, accelerator_vram_gb=(16.0, 6.0))
    context = ExecutionContext(hardware, str(tmp_path), 1, resolved_config=config)
    observed_spec: dict = {}

    class UnderEngagedFakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            spec_path = Path(command[command.index("--spec") + 1])
            result_path = Path(command[command.index("--result") + 1])
            spec = json.loads(spec_path.read_text())
            observed_spec.update(spec)
            output = Path(spec["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            (output / "adapter_model.safetensors").write_bytes(b"adapter")
            result_path.write_text(
                json.dumps(
                    {
                        "telemetry": {"train_loss": 0.25, "global_step": 3},
                        "versions": {"transformers": "5.test"},
                        "provenance": {"resolved_model_commit": "abc123"},
                        "data_provenance": {
                            "primary_rows": 1,
                            "replay_selected_rows": 0,
                        },
                        "resource_usage": {
                            "active_accelerator_count": 1,
                            "visible_accelerator_count": 2,
                            "peak_vram_gb_by_accelerator": {"cuda:0": 1.0},
                        },
                    }
                )
            )

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen", UnderEngagedFakeProcess
    )
    with pytest.raises(RuntimeError, match="did not actually engage every requested device"):
        TransformersPeftExecutor().run(_experiment(), context)


@pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
def test_real_tiny_llama_resumes_training_from_a_real_checkpoint(tmp_path):
    """Prove resume actually works against real Transformers+PEFT+Trainer
    machinery, not just that the manifest plumbing is wired correctly --
    PEFT models resuming through Trainer's own checkpoint/optimizer-state
    mechanism is exactly the kind of interaction a mocked subprocess test
    cannot catch a real bug in."""
    data = tmp_path / "train.jsonl"
    rows = [
        {"text": "Question: What token comes after alpha? Answer: beta"},
        {"text": "Question: What token comes after red? Answer: blue"},
        {"text": "Question: What token comes after one? Answer: two"},
        {"text": "Question: What token comes after up? Answer: down"},
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    base_config = _config(str(data))
    base_config["backend"]["base_model"] = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
    base_config["backend"]["precision"] = "fp32"
    base_config["backend"]["quantization"] = "none"
    base_config["backend"]["lora"] = {
        "r": 4,
        "alpha": 8,
        "dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"],
    }
    base_config["backend"]["training"] = {
        "epochs": 1.0,
        "learning_rate": 0.001,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "logging_steps": 1,
        "gradient_checkpointing": False,
        "save_strategy": "steps",
        "save_steps": 1,
    }
    base_config["backend"]["runtime"] = {"timeout_seconds": 180.0}

    context = ExecutionContext(
        _hardware(), str(tmp_path), 1, resolved_config=base_config
    )
    first = TransformersPeftExecutor().run(_experiment(), context)
    trainer_dir = Path(first.artifact_ref) / "trainer"
    checkpoints = sorted(
        (p for p in trainer_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.rsplit("-", 1)[1]),
    )
    assert checkpoints, "no checkpoints were written by the first run"
    manifest_path = trainer_dir / "chowder-checkpoint-manifest.json"
    assert manifest_path.is_file()

    resume_config = json.loads(json.dumps(base_config))  # deep copy
    resume_config["backend"]["resume_from_checkpoint"] = str(checkpoints[-1])
    resume_config["backend"]["training"]["epochs"] = 2.0
    resume_context = ExecutionContext(
        _hardware(), str(tmp_path), 1, resolved_config=resume_config
    )

    # A mismatched resume must be rejected before any real training happens.
    tampered_config = json.loads(json.dumps(resume_config))
    tampered_config["backend"]["training"]["learning_rate"] = 0.5
    with pytest.raises(ValueError, match="refusing to resume"):
        TransformersPeftExecutor().run(
            _experiment(),
            ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=tampered_config),
        )

    # The matching resume must actually run and continue past where the
    # first run stopped.
    second = TransformersPeftExecutor().run(_experiment(), resume_context)
    assert second.telemetry["global_step"] >= first.telemetry["global_step"]
    assert second.evidence["checkpoint"]["resumed_from_checkpoint"] == str(
        checkpoints[-1].resolve()
    )


@pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
def test_real_tiny_llama_trains_with_activation_offload_always(tmp_path):
    """Prove activation_offload actually wraps a real Trainer.train() call
    in torch.autograd.graph.saved_tensors_hooks without breaking training
    -- a mocked subprocess test cannot catch a real interaction between
    the hooks and PEFT/Trainer's own autograd usage."""
    data = tmp_path / "train.jsonl"
    rows = [
        {"text": "Question: What token comes after alpha? Answer: beta"},
        {"text": "Question: What token comes after red? Answer: blue"},
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    config = _config(str(data))
    config["backend"]["base_model"] = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
    config["backend"]["precision"] = "fp32"
    config["backend"]["quantization"] = "none"
    config["backend"]["lora"] = {
        "r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"],
    }
    config["backend"]["training"] = {
        "epochs": 1.0, "learning_rate": 0.001, "batch_size": 1,
        "gradient_accumulation_steps": 1, "logging_steps": 1,
        "gradient_checkpointing": False, "activation_offload": "always",
    }
    config["backend"]["runtime"] = {"timeout_seconds": 180.0}

    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert artifact.telemetry["global_step"] > 0
    assert artifact.evidence["hardware_aware_defaults"]["resolved_activation_offload"] is True
    offload_evidence = artifact.evidence["activation_offload"]
    assert offload_evidence["mode"] == "always"
    assert offload_evidence["resolved"] is True
    assert offload_evidence["actual_avg_step_seconds"] is not None
    # Real transfer pressure: the pack/unpack hooks genuinely moved bytes
    # between device and host during this real training run -- but only
    # on CUDA. On CPU-only hardware every tensor is already host-resident,
    # so the hooks correctly no-op (nothing to offload) and the counter
    # stays 0; that is real, expected behavior, not a bug.
    import torch

    if torch.cuda.is_available():
        assert offload_evidence["actual_bytes_transferred"] > 0
    else:
        assert offload_evidence["actual_bytes_transferred"] == 0


@pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
def test_real_tiny_llama_activation_offload_auto_declines_for_negligible_pressure(tmp_path):
    """A real "auto" run against a tiny model must run the real
    activation-offload experiment and correctly decline it -- matching
    the experiment's own established finding for this exact model
    (negligible VRAM savings, a real measured wall-time penalty). This is
    the "only enable automatically when required or empirically
    worthwhile" requirement proven end to end, not just at the
    experiment-module level."""
    data = tmp_path / "train.jsonl"
    rows = [
        {"text": "Question: What token comes after alpha? Answer: beta"},
        {"text": "Question: What token comes after red? Answer: blue"},
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    config = _config(str(data))
    config["backend"]["base_model"] = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
    config["backend"]["precision"] = "fp32"
    config["backend"]["quantization"] = "none"
    config["backend"]["lora"] = {
        "r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"],
    }
    config["backend"]["training"] = {
        "epochs": 1.0, "learning_rate": 0.001, "batch_size": 1,
        "gradient_accumulation_steps": 1, "logging_steps": 1,
        "gradient_checkpointing": False, "activation_offload": "auto",
    }
    config["backend"]["runtime"] = {"timeout_seconds": 180.0}

    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert artifact.telemetry["global_step"] > 0
    assert artifact.evidence["hardware_aware_defaults"]["resolved_activation_offload"] is False
    offload_evidence = artifact.evidence["activation_offload"]
    assert offload_evidence["mode"] == "auto"
    assert offload_evidence["resolved"] is False
    # The experiment itself requires CUDA (there is nothing to offload
    # activations "off of" when everything is already host-resident) --
    # on CPU-only hardware it correctly reports unavailable rather than
    # measuring a penalty ratio, and auto declines for that honest reason
    # instead of the "negligible pressure" one this test is named for.
    import torch

    if torch.cuda.is_available():
        assert offload_evidence["predicted_available"] is True
        assert offload_evidence["predicted_recommended"] is False
    else:
        assert offload_evidence["predicted_available"] is False


@pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
def test_real_tiny_llama_resumes_across_a_different_activation_offload_setting(tmp_path):
    """The recipe-digest/bound-inputs exclusion proven at the unit level
    against a real checkpoint: training under activation_offload=always,
    then resuming the same checkpoint under activation_offload=off, must
    succeed against real Trainer/optimizer-state machinery."""
    data = tmp_path / "train.jsonl"
    rows = [
        {"text": "Question: What token comes after alpha? Answer: beta"},
        {"text": "Question: What token comes after red? Answer: blue"},
        {"text": "Question: What token comes after one? Answer: two"},
        {"text": "Question: What token comes after up? Answer: down"},
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    base_config = _config(str(data))
    base_config["backend"]["base_model"] = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
    base_config["backend"]["precision"] = "fp32"
    base_config["backend"]["quantization"] = "none"
    base_config["backend"]["lora"] = {
        "r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"],
    }
    base_config["backend"]["training"] = {
        "epochs": 1.0, "learning_rate": 0.001, "batch_size": 1,
        "gradient_accumulation_steps": 1, "logging_steps": 1,
        "gradient_checkpointing": False, "activation_offload": "always",
        "save_strategy": "steps", "save_steps": 1,
    }
    base_config["backend"]["runtime"] = {"timeout_seconds": 180.0}

    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=base_config)
    first = TransformersPeftExecutor().run(_experiment(), context)
    trainer_dir = Path(first.artifact_ref) / "trainer"
    checkpoints = sorted(
        (p for p in trainer_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.rsplit("-", 1)[1]),
    )
    assert checkpoints, "no checkpoints were written by the first run"

    resume_config = json.loads(json.dumps(base_config))
    resume_config["backend"]["resume_from_checkpoint"] = str(checkpoints[-1])
    resume_config["backend"]["training"]["activation_offload"] = "off"
    resume_context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=resume_config)

    second = TransformersPeftExecutor().run(_experiment(), resume_context)
    assert second.telemetry["global_step"] >= first.telemetry["global_step"]
    assert second.evidence["hardware_aware_defaults"]["resolved_activation_offload"] is False


# optimizer_tiering="always" bypasses run_optimizer_tiering_experiment's own
# graceful degradation entirely (that only applies to "auto") and
# unconditionally constructs a real bitsandbytes paged optimizer -- so unlike
# the activation_offload real tests, these need bitsandbytes actually
# installed (chowder-ai[qlora]), not just CHOWDER_REAL_ML_SMOKE=1 and [train].
# CI's "real transformers peft cpu smoke" job installs only [train,dev], so
# these are expected to skip there, not fail; they run for real locally
# wherever qlora is also installed.
_OPTIMIZER_TIERING_REAL_SMOKE = pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1"
    or importlib.util.find_spec("bitsandbytes") is None,
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1, train dependencies, "
    "and bitsandbytes (chowder-ai[qlora])",
)


@_OPTIMIZER_TIERING_REAL_SMOKE
def test_real_tiny_llama_trains_with_optimizer_tiering_always(tmp_path):
    """Prove optimizer_tiering actually sets optim='paged_adamw_32bit' on a
    real Trainer and trains successfully -- a mocked subprocess test cannot
    catch a real interaction between bitsandbytes' paged optimizer and
    PEFT/Trainer's own optimizer construction."""
    data = tmp_path / "train.jsonl"
    rows = [
        {"text": "Question: What token comes after alpha? Answer: beta"},
        {"text": "Question: What token comes after red? Answer: blue"},
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    config = _config(str(data))
    config["backend"]["base_model"] = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
    config["backend"]["precision"] = "fp32"
    config["backend"]["quantization"] = "none"
    config["backend"]["lora"] = {
        "r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"],
    }
    config["backend"]["training"] = {
        "epochs": 1.0, "learning_rate": 0.001, "batch_size": 1,
        "gradient_accumulation_steps": 1, "logging_steps": 1,
        "gradient_checkpointing": False, "optimizer_tiering": "always",
    }
    config["backend"]["runtime"] = {"timeout_seconds": 180.0}

    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert artifact.telemetry["global_step"] > 0
    assert artifact.evidence["hardware_aware_defaults"]["resolved_optimizer_tiering"] is True
    tiering_evidence = artifact.evidence["optimizer_tiering"]
    assert tiering_evidence["mode"] == "always"
    assert tiering_evidence["resolved"] is True
    assert tiering_evidence["actual_avg_step_seconds"] is not None
    # Real optimizer-state tensor introspection: a real optimizer genuinely
    # holds real state after stepping.
    assert tiering_evidence["actual_optimizer_state_bytes"] > 0


@_OPTIMIZER_TIERING_REAL_SMOKE
def test_real_tiny_llama_optimizer_tiering_auto_runs_the_real_experiment(tmp_path):
    """A real "auto" run must run the real optimizer-tiering experiment and
    resolve according to its verdict -- proven end to end, not just at the
    experiment-module level. Unlike activation_offload's analogous test,
    this doesn't assert a specific recommended/declined outcome: bitsandbytes'
    paged optimizer has no "zero benefit, never recommend" case by state size
    (see OptimizerTieringExperiment's docstring), so whether this tiny
    model's measured penalty ratio clears the acceptance threshold on any
    given machine is not something this test should hard-code -- it proves
    the real experiment ran and was honored, not a specific verdict."""
    data = tmp_path / "train.jsonl"
    rows = [
        {"text": "Question: What token comes after alpha? Answer: beta"},
        {"text": "Question: What token comes after red? Answer: blue"},
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    config = _config(str(data))
    config["backend"]["base_model"] = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
    config["backend"]["precision"] = "fp32"
    config["backend"]["quantization"] = "none"
    config["backend"]["lora"] = {
        "r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"],
    }
    config["backend"]["training"] = {
        "epochs": 1.0, "learning_rate": 0.001, "batch_size": 1,
        "gradient_accumulation_steps": 1, "logging_steps": 1,
        "gradient_checkpointing": False, "optimizer_tiering": "auto",
    }
    config["backend"]["runtime"] = {"timeout_seconds": 180.0}

    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert artifact.telemetry["global_step"] > 0
    tiering_evidence = artifact.evidence["optimizer_tiering"]
    assert tiering_evidence["mode"] == "auto"
    assert tiering_evidence["predicted_available"] is True
    resolved = artifact.evidence["hardware_aware_defaults"]["resolved_optimizer_tiering"]
    assert resolved == tiering_evidence["predicted_recommended"]
    assert resolved == tiering_evidence["resolved"]


@_OPTIMIZER_TIERING_REAL_SMOKE
def test_real_tiny_llama_resume_is_rejected_across_a_different_optimizer_tiering_setting(tmp_path):
    """The opposite of activation_offload's resume-allowed real test: a
    checkpoint trained under optimizer_tiering=off, then resumed under
    optimizer_tiering=always, must be REJECTED at config-resolution time --
    proving for real (not just via the unit-level bound-inputs check) that
    this never reaches bitsandbytes' real KeyError deep inside a resumed
    Trainer.train() call, which a manual reproduction confirmed happens
    when torch.optim.AdamW state is fed to a PagedAdamW32bit resume."""
    data = tmp_path / "train.jsonl"
    rows = [
        {"text": "Question: What token comes after alpha? Answer: beta"},
        {"text": "Question: What token comes after red? Answer: blue"},
        {"text": "Question: What token comes after one? Answer: two"},
        {"text": "Question: What token comes after up? Answer: down"},
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    base_config = _config(str(data))
    base_config["backend"]["base_model"] = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
    base_config["backend"]["precision"] = "fp32"
    base_config["backend"]["quantization"] = "none"
    base_config["backend"]["lora"] = {
        "r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"],
    }
    base_config["backend"]["training"] = {
        "epochs": 1.0, "learning_rate": 0.001, "batch_size": 1,
        "gradient_accumulation_steps": 1, "logging_steps": 1,
        "gradient_checkpointing": False, "optimizer_tiering": "off",
        "save_strategy": "steps", "save_steps": 1,
    }
    base_config["backend"]["runtime"] = {"timeout_seconds": 180.0}

    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=base_config)
    first = TransformersPeftExecutor().run(_experiment(), context)
    trainer_dir = Path(first.artifact_ref) / "trainer"
    checkpoints = sorted(
        (p for p in trainer_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.rsplit("-", 1)[1]),
    )
    assert checkpoints, "no checkpoints were written by the first run"

    resume_config = json.loads(json.dumps(base_config))
    resume_config["backend"]["resume_from_checkpoint"] = str(checkpoints[-1])
    resume_config["backend"]["training"]["optimizer_tiering"] = "always"
    resume_context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=resume_config)

    with pytest.raises(ValueError, match="bound training input"):
        TransformersPeftExecutor().run(_experiment(), resume_context)


@pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
def test_real_tiny_llama_trains_with_frozen_layer_streaming_always(tmp_path):
    """Prove frozen_layer_streaming actually wraps the real Trainer.train()
    call via chowder.memory_fabric.stream_frozen_layers, including real
    checkpoint saving under it -- a mocked subprocess test cannot catch a
    real interaction between the custom autograd.Function/CUDA prefetch
    stream and PEFT/Trainer's own machinery."""
    data = tmp_path / "train.jsonl"
    rows = [
        {"text": "Question: What token comes after alpha? Answer: beta"},
        {"text": "Question: What token comes after red? Answer: blue"},
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    config = _config(str(data))
    config["backend"]["base_model"] = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
    config["backend"]["precision"] = "fp32"
    config["backend"]["quantization"] = "none"
    config["backend"]["lora"] = {
        "r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"],
    }
    config["backend"]["training"] = {
        "epochs": 1.0, "learning_rate": 0.001, "batch_size": 1,
        "gradient_accumulation_steps": 1, "logging_steps": 1,
        "gradient_checkpointing": False, "frozen_layer_streaming": "always",
        "save_strategy": "steps", "save_steps": 1,
    }
    config["backend"]["runtime"] = {"timeout_seconds": 180.0}

    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert artifact.telemetry["global_step"] > 0
    assert artifact.evidence["hardware_aware_defaults"]["resolved_frozen_layer_streaming"] is True
    streaming_evidence = artifact.evidence["frozen_layer_streaming"]
    assert streaming_evidence["mode"] == "always"
    assert streaming_evidence["resolved"] is True
    assert streaming_evidence["actual_avg_step_seconds"] is not None
    # Real transfer pressure: FrozenLayerPrefetchRuntime genuinely moved
    # bytes between host and device during this real training run.
    assert streaming_evidence["actual_bytes_transferred"] > 0
    # A real checkpoint was written while streaming was active.
    trainer_dir = Path(artifact.artifact_ref) / "trainer"
    assert any(trainer_dir.glob("checkpoint-*"))


@pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
def test_real_tiny_llama_frozen_layer_streaming_auto_runs_the_real_experiment(tmp_path):
    """A real "auto" run must run the real frozen-layer-streaming
    experiment and resolve according to its verdict -- proven end to end,
    not just at the experiment-module level. Like optimizer_tiering's
    analogous test, this doesn't hard-code a specific recommended/
    declined outcome: whether this tiny model's measured penalty ratio
    clears the acceptance threshold on any given machine is not something
    this test should assume -- it proves the real experiment ran and was
    honored, not a specific verdict."""
    data = tmp_path / "train.jsonl"
    rows = [
        {"text": "Question: What token comes after alpha? Answer: beta"},
        {"text": "Question: What token comes after red? Answer: blue"},
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    config = _config(str(data))
    config["backend"]["base_model"] = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
    config["backend"]["precision"] = "fp32"
    config["backend"]["quantization"] = "none"
    config["backend"]["lora"] = {
        "r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"],
    }
    config["backend"]["training"] = {
        "epochs": 1.0, "learning_rate": 0.001, "batch_size": 1,
        "gradient_accumulation_steps": 1, "logging_steps": 1,
        "gradient_checkpointing": False, "frozen_layer_streaming": "auto",
    }
    config["backend"]["runtime"] = {"timeout_seconds": 180.0}

    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert artifact.telemetry["global_step"] > 0
    streaming_evidence = artifact.evidence["frozen_layer_streaming"]
    assert streaming_evidence["mode"] == "auto"
    assert streaming_evidence["predicted_available"] is True
    resolved = artifact.evidence["hardware_aware_defaults"]["resolved_frozen_layer_streaming"]
    assert resolved == streaming_evidence["predicted_recommended"]
    assert resolved == streaming_evidence["resolved"]


@pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
def test_real_tiny_llama_resumes_across_a_different_frozen_layer_streaming_setting(tmp_path):
    """The recipe-digest/bound-inputs exclusion proven at the unit level
    against a real checkpoint: training under frozen_layer_streaming=
    always, then resuming the same checkpoint under
    frozen_layer_streaming=off, must succeed against real Trainer/
    optimizer-state machinery -- the opposite of optimizer_tiering's
    resume-rejected test, matching activation_offload's own real
    resume-allowed test."""
    data = tmp_path / "train.jsonl"
    rows = [
        {"text": "Question: What token comes after alpha? Answer: beta"},
        {"text": "Question: What token comes after red? Answer: blue"},
        {"text": "Question: What token comes after one? Answer: two"},
        {"text": "Question: What token comes after up? Answer: down"},
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    base_config = _config(str(data))
    base_config["backend"]["base_model"] = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
    base_config["backend"]["precision"] = "fp32"
    base_config["backend"]["quantization"] = "none"
    base_config["backend"]["lora"] = {
        "r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"],
    }
    base_config["backend"]["training"] = {
        "epochs": 1.0, "learning_rate": 0.001, "batch_size": 1,
        "gradient_accumulation_steps": 1, "logging_steps": 1,
        "gradient_checkpointing": False, "frozen_layer_streaming": "always",
        "save_strategy": "steps", "save_steps": 1,
    }
    base_config["backend"]["runtime"] = {"timeout_seconds": 180.0}

    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=base_config)
    first = TransformersPeftExecutor().run(_experiment(), context)
    trainer_dir = Path(first.artifact_ref) / "trainer"
    checkpoints = sorted(
        (p for p in trainer_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.rsplit("-", 1)[1]),
    )
    assert checkpoints, "no checkpoints were written by the first run"

    resume_config = json.loads(json.dumps(base_config))
    resume_config["backend"]["resume_from_checkpoint"] = str(checkpoints[-1])
    resume_config["backend"]["training"]["frozen_layer_streaming"] = "off"
    resume_context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=resume_config)

    second = TransformersPeftExecutor().run(_experiment(), resume_context)
    assert second.telemetry["global_step"] >= first.telemetry["global_step"]
    assert second.evidence["hardware_aware_defaults"]["resolved_frozen_layer_streaming"] is False


@pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
def test_real_tiny_llama_max_steps_caps_training_and_schedule_fields_apply(tmp_path):
    """Prove max_steps really overrides epoch-based length against a real
    Trainer (not just that the value round-trips through the spec), and
    that the LR-scheduler/warmup/weight-decay/grad-clip fields are accepted
    by real TrainingArguments construction, not just Chowder's own schema."""
    data = tmp_path / "train.jsonl"
    rows = [
        {"text": f"Question: What token comes after {word}? Answer: next"}
        for word in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    config = _config(str(data))
    config["backend"]["base_model"] = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
    config["backend"]["precision"] = "fp32"
    config["backend"]["quantization"] = "none"
    config["backend"]["lora"] = {
        "r": 4,
        "alpha": 8,
        "dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"],
    }
    config["backend"]["training"] = {
        # 8 rows / batch_size 1 / grad_accum 1 * 5 epochs = 40 steps if
        # max_steps did not override it -- max_steps=2 must stop it early.
        "epochs": 5.0,
        "max_steps": 2,
        "learning_rate": 0.001,
        "lr_scheduler_type": "cosine",
        # Ratio-based warmup specifically, not warmup_steps -- this is the
        # path that broke against real transformers>=5.2 (warmup_ratio was
        # removed as its own TrainingArguments kwarg in favor of an
        # overloaded warmup_steps; see transformers_worker.py).
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "max_grad_norm": 0.5,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "logging_steps": 1,
        "gradient_checkpointing": False,
    }
    config["backend"]["runtime"] = {"timeout_seconds": 180.0}

    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert artifact.telemetry["global_step"] == 2


@pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
def test_real_chat_completion_only_masking_matches_assistant_turns():
    """Prove the boundary-diff masking approach against the real tokenizer
    and its real (generation-tag-free) Llama 3.1 chat template -- the exact
    template shape that made return_assistant_tokens_mask=True silently
    produce an all-zero mask when tried directly against it."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-LlamaForCausalLM-3.2")
    messages = _validate_chat_messages(
        [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "And of Germany?"},
            {"role": "assistant", "content": "Berlin."},
        ],
        row_index=0,
    )
    example = _build_chat_example(tokenizer, messages, max_length=512, row_index=0)
    input_ids = example["input_ids"]
    labels = example["labels"]
    assert len(labels) == len(input_ids) == len(example["attention_mask"])

    unmasked_text = tokenizer.decode(
        [token_id for token_id, label in zip(input_ids, labels) if label != -100]
    )
    assert "Paris" in unmasked_text
    assert "Berlin" in unmasked_text
    # The user's own turns must not have leaked into the trained labels --
    # phrasing unique to the question, not the answer (both answers happen
    # to contain "capital"/"Germany"-adjacent words, so those alone aren't
    # a safe negative check).
    assert "What is the capital" not in unmasked_text
    assert "And of Germany" not in unmasked_text
    assert 0 < sum(1 for label in labels if label != -100) < len(labels)


@pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
def test_real_chat_example_rejects_a_conversation_with_no_room_for_the_response():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-LlamaForCausalLM-3.2")
    messages = _validate_chat_messages(
        [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ],
        row_index=0,
    )
    with pytest.raises(RuntimeError, match="no assistant tokens remain after truncating"):
        _build_chat_example(tokenizer, messages, max_length=5, row_index=0)


@pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
def test_real_tiny_llama_trains_on_a_chat_dataset_with_completion_only_masking(tmp_path):
    """End-to-end: real LoRA training on chat/message rows, proving the
    whole pipeline (chat dataset loading, per-row masking, the
    DataCollatorForSeq2Seq label-padding path) works together against a
    real Trainer, not just that each piece works in isolation."""
    data = tmp_path / "train.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": "What token comes after alpha?"},
                {"role": "assistant", "content": "beta"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What token comes after red?"},
                {"role": "assistant", "content": "blue"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What token comes after one?"},
                {"role": "assistant", "content": "two"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What token comes after up?"},
                {"role": "assistant", "content": "down"},
            ]
        },
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    config = _config(str(data))
    config["backend"]["base_model"] = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
    config["backend"]["precision"] = "fp32"
    config["backend"]["quantization"] = "none"
    config["backend"]["dataset_format"] = "chat"
    config["backend"]["max_length"] = 128
    config["backend"]["lora"] = {
        "r": 4,
        "alpha": 8,
        "dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"],
    }
    config["backend"]["training"] = {
        "epochs": 1.0,
        "learning_rate": 0.001,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "logging_steps": 1,
        "gradient_checkpointing": False,
    }
    config["backend"]["runtime"] = {"timeout_seconds": 180.0}

    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert Path(artifact.artifact_ref).is_dir()

    provenance = artifact.evidence["data_provenance"]
    assert provenance["dataset_format"] == "chat"
    assert provenance["total_token_count"] > 0
    assert 0 < provenance["assistant_token_count"] < provenance["total_token_count"]


@pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
def test_real_tiny_llama_auto_detects_target_modules_when_unset(tmp_path):
    """No backend.lora.target_modules and no target_preset -- proves the
    default ("auto") actually reaches real PEFT's own per-architecture
    mapping and trains successfully, not just that the None sentinel is
    threaded through correctly against mocks."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"Question: What token comes after alpha? Answer: beta"}\n')

    config = _config(str(data))
    config["backend"]["base_model"] = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
    config["backend"]["precision"] = "fp32"
    config["backend"]["quantization"] = "none"
    config["backend"]["lora"] = {"r": 4, "alpha": 8, "dropout": 0.0}
    config["backend"]["training"] = {
        "epochs": 1.0,
        "learning_rate": 0.001,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "logging_steps": 1,
        "gradient_checkpointing": False,
    }
    config["backend"]["runtime"] = {"timeout_seconds": 180.0}

    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    provenance = artifact.evidence["model_provenance"]
    assert provenance["model_type"] == "llama"
    # PEFT's own actively-maintained default for llama -- not a value Chowder
    # invented, so this pins to whatever PEFT itself ships.
    assert provenance["resolved_target_modules"] == ["q_proj", "v_proj"]


@pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
def test_real_tiny_llama_attention_and_mlp_preset_targets_all_seven_modules(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"Question: What token comes after alpha? Answer: beta"}\n')

    config = _config(str(data))
    config["backend"]["base_model"] = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
    config["backend"]["precision"] = "fp32"
    config["backend"]["quantization"] = "none"
    config["backend"]["lora"] = {
        "r": 4,
        "alpha": 8,
        "dropout": 0.0,
        "target_preset": "attention_and_mlp",
    }
    config["backend"]["training"] = {
        "epochs": 1.0,
        "learning_rate": 0.001,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "logging_steps": 1,
        "gradient_checkpointing": False,
    }
    config["backend"]["runtime"] = {"timeout_seconds": 180.0}

    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    provenance = artifact.evidence["model_provenance"]
    assert provenance["resolved_target_modules"] == sorted(
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


@pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
def test_real_tiny_llama_trains_with_hardware_defaulted_gradient_checkpointing_disabled(
    tmp_path,
):
    """No backend.training.gradient_checkpointing set, real ample-VRAM
    hardware context -- proves the "off by default with real headroom"
    default reaches all the way through a real Trainer run, not just that
    the resolution function itself picks the right boolean in isolation.
    (The complementary "on by default when VRAM is small/unknown" default
    is not new worker-side behavior -- gradient_checkpointing=True was
    already this backend's fixed default and is already exercised by every
    other real test in this file.) The low-VRAM quantization default
    ("4bit") cannot be verified end-to-end here: it requires an actual CUDA
    device, and this CI job is CPU-only -- covered instead by the unit
    tests above, which do not depend on real hardware being present."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"Question: What token comes after alpha? Answer: beta"}\n')

    config = _config(str(data))
    config["backend"]["base_model"] = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
    config["backend"]["precision"] = "fp32"
    config["backend"]["quantization"] = "none"
    config["backend"]["lora"] = {
        "r": 4,
        "alpha": 8,
        "dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"],
    }
    config["backend"]["training"] = {
        "epochs": 1.0,
        "learning_rate": 0.001,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "logging_steps": 1,
        # gradient_checkpointing deliberately omitted.
    }
    config["backend"]["runtime"] = {"timeout_seconds": 180.0}

    ample_vram = HardwareProfile(40.0, 64, 500, 12, 40, 3)
    context = ExecutionContext(ample_vram, str(tmp_path), 1, resolved_config=config)
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    hardware_defaults = artifact.evidence["hardware_aware_defaults"]
    assert hardware_defaults["gradient_checkpointing_defaulted"] is True
    assert hardware_defaults["resolved_gradient_checkpointing"] is False
    assert artifact.telemetry["global_step"] > 0


@pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
def test_real_tiny_llama_training_retries_a_flaky_tokenizer_download(tmp_path, monkeypatch):
    """Proves with_hub_retries is actually wired into the worker's real
    tokenizer-loading call, not just correct in isolation against a
    directly-constructed function. Calls transformers_worker.train()
    in-process (not through TransformersPeftExecutor's usual real
    subprocess) because a subprocess's imports can't be monkeypatched from
    the parent test process -- this is the one place that deviates from
    this file's usual real-subprocess pattern, and only for that reason."""
    from chowder.backends import transformers_worker

    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"Question: What token comes after alpha? Answer: beta"}\n')

    config = _config(str(data))
    config["backend"]["base_model"] = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
    config["backend"]["precision"] = "fp32"
    config["backend"]["quantization"] = "none"
    config["backend"]["lora"] = {
        "r": 4,
        "alpha": 8,
        "dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"],
    }
    config["backend"]["training"] = {
        "epochs": 1.0,
        "learning_rate": 0.001,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "logging_steps": 1,
        "gradient_checkpointing": False,
    }
    config["backend"]["runtime"] = {"timeout_seconds": 180.0}
    spec = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )

    import httpx
    import transformers as real_transformers

    real_from_pretrained = real_transformers.AutoTokenizer.from_pretrained
    attempts = {"n": 0}

    def flaky_from_pretrained(*args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            try:
                raise httpx.ConnectTimeout("simulated timeout")
            except httpx.ConnectTimeout as exc:
                raise OSError("simulated wrapped transient failure") from exc
        return real_from_pretrained(*args, **kwargs)

    monkeypatch.setattr(
        real_transformers.AutoTokenizer, "from_pretrained", flaky_from_pretrained
    )
    # with_hub_retries' sleep parameter defaults to time.sleep, bound once at
    # function-definition time -- monkeypatching time.sleep afterward can't
    # intercept it (the default already holds a direct reference to the
    # original function object, not a name that later patching re-resolves).
    # Production code never passes sleep= explicitly, so this test accepts
    # the real ~3.5s of default backoff delay (1.0s + 2.0s, plus jitter)
    # rather than mocking internals the real code path doesn't expose.
    result = transformers_worker.train(spec)
    assert attempts["n"] == 3
    assert result is not None
    assert result["telemetry"]["global_step"] > 0


@pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
def test_real_tiny_llama_trains_with_offline_mode_against_a_cached_model(tmp_path):
    """backend.offline=True against the real, already-cached test model --
    proves local_files_only=True is actually threaded through to both the
    tokenizer and model from_pretrained calls and that a real cache hit
    trains successfully without touching the network at all."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"Question: What token comes after alpha? Answer: beta"}\n')

    config = _config(str(data))
    config["backend"]["base_model"] = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
    config["backend"]["precision"] = "fp32"
    config["backend"]["quantization"] = "none"
    config["backend"]["offline"] = True
    config["backend"]["lora"] = {
        "r": 4,
        "alpha": 8,
        "dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"],
    }
    config["backend"]["training"] = {
        "epochs": 1.0,
        "learning_rate": 0.001,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "logging_steps": 1,
        "gradient_checkpointing": False,
    }
    config["backend"]["runtime"] = {"timeout_seconds": 180.0}

    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    artifact = TransformersPeftExecutor().run(_experiment(), context)
    assert Path(artifact.artifact_ref).is_dir()
    assert artifact.telemetry["global_step"] > 0


@pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
def test_real_offline_mode_fails_fast_with_zero_retries_for_an_uncached_model(tmp_path):
    """offline=True against a model that is definitely not cached must fail
    immediately (LocalEntryNotFoundError, classified as permanent) rather
    than retrying into ~30s of pointless backoff for a fetch it was told
    never to attempt."""
    # Pre-import the same stack transformers_worker.train() imports lazily,
    # so its own internal `import torch`/etc. are sys.modules cache hits
    # instead of a genuine ~5s cold import -- this test measures whether
    # the offline lookup itself retries, not one-time interpreter warmup.
    import datasets  # noqa: F401
    import peft  # noqa: F401
    import torch  # noqa: F401
    import transformers  # noqa: F401

    from chowder.backends import transformers_worker

    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    config["backend"]["base_model"] = "chowder-test-org/definitely-not-cached-offline-xyz"
    config["backend"]["offline"] = True
    spec = TransformersPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )

    started = time.perf_counter()
    with pytest.raises(Exception):
        transformers_worker.train(spec)
    elapsed = time.perf_counter() - started
    assert elapsed < 5.0  # would be 1.0s+ into backoff alone if it retried even once
