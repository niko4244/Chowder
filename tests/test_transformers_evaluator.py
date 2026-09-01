import json
from pathlib import Path

import pytest

from chowder.evaluators.transformers_text import TransformersTextEvaluator, TransformersTextEvalSpec
from chowder.evaluators.transformers_text_worker import _score
from chowder.executors import ExecutionContext, TrainingArtifact
from chowder.memory import HardwareProfile
from chowder.models import Experiment, Hypothesis
from chowder.provenance import sha256_directory


def _hardware():
    return HardwareProfile(16, 64, 500, 12, 40, 3)


def _experiment():
    return Experiment("e1", None, Hypothesis("obs", "cause", "fix"), {}, 1.0)


def _config(dataset: str):
    return {
        "seed": 11,
        "backend": {
            "type": "transformers-peft",
            "base_model": "example/model",
            "revision": "requested-rev",
            "quantization": "4bit",
            "precision": "bf16",
        },
        "evaluation": {
            "type": "transformers-text",
            "quantization": "inherit",
            "precision": "inherit",
            "suites": [
                {
                    "name": "quality",
                    "dataset": dataset,
                    "scoring": "normalized_exact_match",
                    "max_new_tokens": 16,
                }
            ],
        },
    }


def _artifact(tmp_path, *, resolved_commit="resolved123"):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    return TrainingArtifact(
        run_id="run-1",
        experiment_id="e1",
        artifact_ref=str(adapter),
        gpu_hours=0.25,
        evidence={
            "artifact_sha256": sha256_directory(adapter),
            "model_provenance": {"resolved_model_commit": resolved_commit},
        },
    )


def test_eval_spec_pins_resolved_training_commit_and_inherits_runtime(tmp_path):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"2+2?","expected":"4"}\n')
    artifact = _artifact(tmp_path)
    spec = TransformersTextEvalSpec.from_context(
        config=_config("eval.jsonl"),
        artifact=artifact,
        work_dir=tmp_path,
        output_dir=tmp_path / "out",
        seed=1,
    )
    assert spec.revision == "resolved123"
    assert spec.quantization == "4bit"
    assert spec.precision == "bf16"
    assert spec.seed == 11
    assert spec.suites[0].name == "quality"
    assert spec.suites[0].dataset == str(data.resolve())


def test_evaluator_refuses_mutated_adapter(tmp_path):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"2+2?","expected":"4"}\n')
    artifact = _artifact(tmp_path)
    Path(artifact.artifact_ref, "adapter_model.safetensors").write_bytes(b"tampered")
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=_config("eval.jsonl"))
    with pytest.raises(ValueError, match="content digest changed"):
        TransformersTextEvaluator().evaluate(
            experiment=_experiment(), artifact=artifact, context=context
        )


def test_evaluator_returns_named_metrics_from_isolated_worker(tmp_path, monkeypatch):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"2+2?","expected":"4"}\n')
    artifact = _artifact(tmp_path)
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=_config("eval.jsonl"))

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            result_path = Path(command[command.index("--result") + 1])
            result_path.write_text(json.dumps({
                "metrics": {"quality": 0.75},
                "suites": {"quality": {"rows": 4, "scoring": "normalized_exact_match"}},
                "versions": {"transformers": "5.test"},
            }))

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr("chowder.evaluators.transformers_text.subprocess.Popen", FakeProcess)
    result = TransformersTextEvaluator().evaluate(
        experiment=_experiment(), artifact=artifact, context=context
    )
    assert result.metrics == {"quality": 0.75}
    assert result.artifact_ref == artifact.artifact_ref
    assert result.gpu_hours >= artifact.gpu_hours
    assert result.evidence["artifact_sha256"] == artifact.evidence["artifact_sha256"]
    assert len(result.evidence["evaluation_dataset_sha256"]["quality"]) == 64


def test_worker_scoring_is_deterministic_and_explicit():
    assert _score(" Answer  42 ", "answer 42", "normalized_exact_match") == 1.0
    assert _score("Answer", "answer", "exact_match") == 0.0
