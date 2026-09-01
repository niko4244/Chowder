import json
from pathlib import Path

import pytest

from chowder.backends.transformers_peft import (
    TransformersPeftExecutor,
    TransformersPeftRunSpec,
)
from chowder.backends.transformers_worker import _replay_sample_count
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
