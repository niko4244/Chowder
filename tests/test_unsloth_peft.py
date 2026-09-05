from __future__ import annotations

import json
from pathlib import Path

import pytest

from chowder.backends.unsloth_peft import (
    UnslothConfigError,
    UnslothPeftExecutor,
    UnslothPeftRunSpec,
)
from chowder.executors import ExecutionContext
from chowder.memory import HardwareProfile
from chowder.models import Experiment, Hypothesis
from chowder.provenance import sha256_file
from chowder.unsloth_env import unsloth_env_dir, unsloth_python


def _hardware():
    return HardwareProfile(16, 64, 500, 12, 40, 3)


def _experiment():
    return Experiment("e1", None, Hypothesis("obs", "cause", "fix"), {}, 2.0)


def _config(dataset: str, **backend_overrides):
    backend = {
        "type": "peft",
        "engine": "unsloth",
        "base_model": "org/model",
        "dataset": dataset,
        "max_length": 256,
        "lora": {"r": 8, "alpha": 16},
        "training": {"learning_rate": 1e-4, "epochs": 1.0},
    }
    backend.update(backend_overrides)
    return {"backend": backend}


def _fake_isolated_python(work_dir: Path) -> Path:
    """Create a stand-in isolated-environment python executable so
    UnslothPeftExecutor._isolated_python's existence check passes, without
    an actual Unsloth install -- the real subprocess launch is mocked in
    tests that need it."""
    env_dir = unsloth_env_dir(work_dir)
    python_path = unsloth_python(env_dir)
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_bytes(b"")
    return python_path


# --- UnslothPeftRunSpec ----------------------------------------------------


def test_spec_is_built_from_resolved_config_and_workdir(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    spec = UnslothPeftRunSpec.from_resolved_config(
        _config("train.jsonl"), work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=7
    )
    assert spec.dataset == str(data.resolve())
    assert spec.base_model == "org/model"
    assert spec.lora_r == 8
    assert spec.lora_alpha == 16
    assert spec.seed == 7
    assert spec.quantization == "none"


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"base_model": ""}, "base_model is required"),
        ({"quantization": "8bit"}, "unsupported quantization"),
    ],
)
def test_spec_rejects_invalid_fields(tmp_path, overrides, message):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config("train.jsonl", **overrides)
    with pytest.raises(ValueError, match=message):
        UnslothPeftRunSpec.from_resolved_config(
            config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
        )


@pytest.mark.parametrize("mechanism", ["activation_offload", "optimizer_tiering", "frozen_layer_streaming"])
def test_spec_construction_fails_early_when_an_unsupported_mechanism_is_requested(
    tmp_path, mechanism
):
    """These mechanisms have not been verified against Unsloth's own
    patched model/attention implementation -- must fail clearly at
    config-resolution time, not silently no-op or crash mid-training."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config("train.jsonl", training={"epochs": 1.0, mechanism: "always"})
    with pytest.raises(UnslothConfigError, match=mechanism):
        UnslothPeftRunSpec.from_resolved_config(
            config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
        )


def test_spec_construction_allows_mechanisms_explicitly_off(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(
        "train.jsonl",
        training={"epochs": 1.0, "activation_offload": "off", "optimizer_tiering": False},
    )
    spec = UnslothPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    assert spec.base_model == "org/model"


# --- profile ----------------------------------------------------------------


def test_profile_uses_experiment_declared_estimate(tmp_path):
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=_config("train.jsonl"))
    estimate = UnslothPeftExecutor().profile(_experiment(), context)
    assert estimate.gpu_hours == pytest.approx(2.0)
    assert estimate.confidence == 0.25


# --- run: isolated-environment preflight ------------------------------------


def test_run_fails_clearly_when_no_isolated_environment_exists(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    context = ExecutionContext(
        _hardware(), str(tmp_path), 1, resolved_config=_config("train.jsonl")
    )
    with pytest.raises(UnslothConfigError, match="chowder setup unsloth"):
        UnslothPeftExecutor().run(_experiment(), context)


# --- run: mocked isolated-subprocess launch (no real Unsloth/CUDA) ---------


class _FakeProcess:
    returncode = 0

    def __init__(self, command, **kwargs):
        self.command = command
        spec_path = Path(command[command.index("--spec") + 1])
        result_path = Path(command[command.index("--result") + 1])
        spec = json.loads(spec_path.read_text())
        self.observed_spec = spec
        output = Path(spec["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "adapter_model.safetensors").write_bytes(b"adapter")
        result_path.write_text(
            json.dumps(
                {
                    "telemetry": {"train_loss": 0.5, "global_step": 3},
                    "resolved_target_modules": ["q_proj", "v_proj"],
                    "resource_usage": {
                        "active_accelerator_count": 1,
                        "visible_accelerator_count": 1,
                        "peak_vram_gb_by_accelerator": {"cuda:0": 4.2},
                    },
                    "model_provenance": {"requested_base_model": spec["base_model"]},
                    "versions": {"unsloth": "test", "torch": "test"},
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


def test_executor_runs_worker_via_isolated_python_and_returns_artifact(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    isolated_python = _fake_isolated_python(tmp_path)
    context = ExecutionContext(
        _hardware(), str(tmp_path), 1, resolved_config=_config("train.jsonl")
    )

    captured = {}

    class RecordingFakeProcess(_FakeProcess):
        def __init__(self, command, **kwargs):
            super().__init__(command, **kwargs)
            captured["command"] = command

    monkeypatch.setattr("chowder.backends.unsloth_peft.subprocess.Popen", RecordingFakeProcess)

    artifact = UnslothPeftExecutor().run(_experiment(), context)

    assert captured["command"][0] == str(isolated_python)
    assert captured["command"][1].endswith("unsloth_worker.py")
    assert artifact.experiment_id == "e1"
    assert artifact.telemetry["train_loss"] == 0.5
    assert Path(artifact.artifact_ref).is_dir()
    assert artifact.evidence["backend"] == "unsloth-peft"
    assert artifact.evidence["engine"] == "unsloth"
    assert artifact.evidence["dataset_sha256"] == sha256_file(data)
    assert artifact.evidence["resolved_target_modules"] == ["q_proj", "v_proj"]
    assert artifact.resource_usage is not None
    assert artifact.resource_usage.active_accelerator_count == 1
    assert artifact.resource_usage.peak_vram_gb_by_accelerator == {"cuda:0": 4.2}
    assert len(artifact.evidence["execution_spec_sha256"]) == 64
    assert len(artifact.evidence["artifact_sha256"]) == 64


def test_executor_rejects_dataset_changed_after_proposal(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"original"}\n')
    config = _config(str(data))
    config["backend"]["dataset_sha256"] = sha256_file(data)
    data.write_text('{"text":"tampered"}\n')
    _fake_isolated_python(tmp_path)
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)

    def should_not_launch(*args, **kwargs):
        raise AssertionError("worker must not launch once the bound dataset digest changed")

    monkeypatch.setattr("chowder.backends.unsloth_peft.subprocess.Popen", should_not_launch)
    with pytest.raises(RuntimeError, match="digest changed before worker load"):
        UnslothPeftExecutor().run(_experiment(), context)


def test_executor_raises_on_nonzero_worker_exit(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    _fake_isolated_python(tmp_path)
    context = ExecutionContext(
        _hardware(), str(tmp_path), 1, resolved_config=_config("train.jsonl")
    )

    class FailingProcess(_FakeProcess):
        def __init__(self, command, **kwargs):
            # Deliberately skip writing a result -- a worker-side crash.
            self.command = command
            self.returncode = 1

        def wait(self, timeout=None):
            return 1

    monkeypatch.setattr("chowder.backends.unsloth_peft.subprocess.Popen", FailingProcess)
    with pytest.raises(RuntimeError, match="unsloth worker failed"):
        UnslothPeftExecutor().run(_experiment(), context)


# --- cancel ------------------------------------------------------------------


def test_cancel_is_a_no_op_for_an_unknown_run_id():
    UnslothPeftExecutor().cancel("does-not-exist")  # must not raise


def test_cancel_terminates_a_tracked_process(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    _fake_isolated_python(tmp_path)
    context = ExecutionContext(
        _hardware(), str(tmp_path), 1, resolved_config=_config("train.jsonl")
    )

    executor = UnslothPeftExecutor()
    captured_run_id = {}

    class SlowProcess(_FakeProcess):
        def wait(self, timeout=None):
            # Simulate cancellation happening mid-run: record the run_id
            # the executor tracked, then behave like a real process being
            # asked to stop.
            (run_id,) = executor._processes.keys()
            captured_run_id["run_id"] = run_id
            executor.cancel(run_id)
            return super().wait(timeout=timeout)

    monkeypatch.setattr("chowder.backends.unsloth_peft.subprocess.Popen", SlowProcess)
    executor.run(_experiment(), context)
    assert captured_run_id["run_id"] not in executor._processes
