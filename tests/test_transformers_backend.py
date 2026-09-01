import json
from pathlib import Path

import pytest

from chowder.backends.transformers_peft import TransformersPeftExecutor, TransformersPeftRunSpec
from chowder.executors import ExecutionContext
from chowder.memory import HardwareProfile
from chowder.models import Experiment, Hypothesis


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
            "lora": {"r": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]},
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
    assert spec.seed == 17
    assert spec.lora_r == 8
    assert spec.quantization == "4bit"
    assert spec.target_modules == ("q_proj", "v_proj")


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


def test_executor_runs_worker_as_isolated_process_and_returns_artifact(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=_config("train.jsonl"))

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            spec_path = Path(command[command.index("--spec") + 1])
            result_path = Path(command[command.index("--result") + 1])
            spec = json.loads(spec_path.read_text())
            output = Path(spec["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            (output / "adapter_model.safetensors").write_bytes(b"adapter")
            result_path.write_text(json.dumps({
                "telemetry": {"train_loss": 0.25, "global_step": 3},
                "versions": {"transformers": "5.test"},
                "provenance": {"resolved_model_commit": "abc123"},
            }))

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
    assert artifact.experiment_id == "e1"
    assert artifact.telemetry["train_loss"] == 0.25
    assert Path(artifact.artifact_ref).is_dir()
    assert artifact.evidence["versions"]["transformers"] == "5.test"
    assert artifact.evidence["model_provenance"]["resolved_model_commit"] == "abc123"
    assert len(artifact.evidence["execution_spec_sha256"]) == 64
    assert len(artifact.evidence["recipe_sha256"]) == 64
    assert len(artifact.evidence["dataset_sha256"]) == 64
    assert len(artifact.evidence["artifact_sha256"]) == 64


def test_recipe_digest_is_stable_across_output_paths(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    a = TransformersPeftRunSpec.from_resolved_config(
        _config(str(data)), work_dir=tmp_path, output_dir=tmp_path / "a", seed=1
    )
    b = TransformersPeftRunSpec.from_resolved_config(
        _config(str(data)), work_dir=tmp_path, output_dir=tmp_path / "b", seed=1
    )
    assert a.digest() != b.digest()
    assert a.recipe_digest() == b.recipe_digest()
