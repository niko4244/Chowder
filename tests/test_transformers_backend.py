import json
import os
import sys
from pathlib import Path

import pytest

from chowder.backends.transformers_peft import (
    TransformersPeftExecutor,
    TransformersPeftRunSpec,
)
from chowder.backends.transformers_worker import (
    _build_chat_example,
    _replay_sample_count,
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
