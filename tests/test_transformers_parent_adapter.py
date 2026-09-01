import json
from pathlib import Path

import pytest

from chowder.backends.transformers_peft import TransformersPeftExecutor, TransformersPeftRunSpec
from chowder.backends.transformers_worker import _verify_bound_adapter
from chowder.executors import ExecutionContext
from chowder.memory import HardwareProfile
from chowder.models import Experiment, Hypothesis
from chowder.provenance import sha256_directory, sha256_file


def _hardware():
    return HardwareProfile(16, 64, 500, 12, 40, 3)


def _experiment():
    return Experiment("continue-1", None, Hypothesis("obs", "cause", "repair"), {}, 0.5)


def _adapter(tmp_path):
    path = tmp_path / "parent-adapter"
    path.mkdir()
    (path / "adapter_config.json").write_text('{"peft_type":"LORA"}\n', encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"parent")
    return path


def _config(data, adapter):
    return {
        "backend": {
            "type": "transformers-peft",
            "base_model": "example/model",
            "dataset": str(data),
            "dataset_sha256": sha256_file(data),
            "parent_adapter": {
                "path": str(adapter),
                "sha256": sha256_directory(adapter),
            },
            "training": {"learning_rate": 5e-5, "epochs": 1},
            "lora": {"r": 16, "alpha": 32},
        }
    }


def test_run_spec_resolves_parent_adapter_and_hash(tmp_path):
    data = tmp_path / "repair.jsonl"
    data.write_text('{"text":"repair"}\n', encoding="utf-8")
    adapter = _adapter(tmp_path)
    spec = TransformersPeftRunSpec.from_resolved_config(
        _config(data, adapter),
        work_dir=tmp_path,
        output_dir=tmp_path / "output",
        seed=3,
    )
    assert spec.parent_adapter == str(adapter.resolve())
    assert spec.parent_adapter_sha256 == sha256_directory(adapter)
    assert spec.recipe_digest()


def test_worker_adapter_verifier_rejects_mutated_directory(tmp_path):
    adapter = _adapter(tmp_path)
    digest = sha256_directory(adapter)
    assert _verify_bound_adapter(str(adapter), digest, label="parent") == digest
    (adapter / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="adapter digest changed"):
        _verify_bound_adapter(str(adapter), digest, label="parent")


def test_executor_rejects_parent_adapter_tamper_before_worker_launch(tmp_path, monkeypatch):
    data = tmp_path / "repair.jsonl"
    data.write_text('{"text":"repair"}\n', encoding="utf-8")
    adapter = _adapter(tmp_path)
    config = _config(data, adapter)
    (adapter / "adapter_model.safetensors").write_bytes(b"tampered")
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)

    def should_not_launch(*args, **kwargs):
        raise AssertionError("worker must not launch for a mutated parent adapter")

    monkeypatch.setattr("chowder.backends.transformers_peft.subprocess.Popen", should_not_launch)
    with pytest.raises(ValueError, match="parent adapter content changed"):
        TransformersPeftExecutor().run(_experiment(), context)


def test_executor_passes_bound_parent_adapter_to_worker_and_records_lineage(tmp_path, monkeypatch):
    data = tmp_path / "repair.jsonl"
    data.write_text('{"text":"repair"}\n', encoding="utf-8")
    adapter = _adapter(tmp_path)
    adapter_sha = sha256_directory(adapter)
    context = ExecutionContext(
        _hardware(), str(tmp_path), 1, resolved_config=_config(data, adapter)
    )
    observed = {}

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            spec_path = Path(command[command.index("--spec") + 1])
            result_path = Path(command[command.index("--result") + 1])
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            observed.update(spec)
            output = Path(spec["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            (output / "adapter_config.json").write_text('{"peft_type":"LORA"}\n', encoding="utf-8")
            (output / "adapter_model.safetensors").write_bytes(b"continued")
            result_path.write_text(
                json.dumps(
                    {
                        "telemetry": {"train_loss": 0.1, "global_step": 2},
                        "versions": {"peft": "test"},
                        "provenance": {
                            "continued_from_parent_adapter": True,
                            "parent_adapter_sha256": adapter_sha,
                        },
                        "data_provenance": {"primary_rows": 1},
                    }
                ),
                encoding="utf-8",
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
    assert observed["parent_adapter"] == str(adapter.resolve())
    assert observed["parent_adapter_sha256"] == adapter_sha
    assert artifact.evidence["parent_adapter_sha256"] == adapter_sha
    assert artifact.evidence["continued_from_parent_adapter"] is True
    assert artifact.evidence["model_provenance"]["parent_adapter_sha256"] == adapter_sha
