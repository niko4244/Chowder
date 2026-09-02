import json
from pathlib import Path

import pytest

from chowder.contamination import write_holdout_fingerprint_index
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


def _fake_payload(result_path: Path, *, metric: float, device: str, gpu_count: int):
    fingerprint_path = result_path.parent / "holdout-fingerprints-quality.jsonl"
    fingerprint_digest = write_holdout_fingerprint_index([("2+2?", "4")], fingerprint_path)
    return {
        "metrics": {"quality": metric},
        "suites": {
            "quality": {
                "rows": 1,
                "scoring": "normalized_exact_match",
                "holdout_fingerprints_file": str(fingerprint_path),
                "holdout_fingerprints_sha256": fingerprint_digest,
            }
        },
        "runtime": {"device": device, "gpu_count": gpu_count},
        "versions": {"transformers": "5.test"},
    }


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
    assert spec.device == "auto"
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


def test_evaluator_returns_named_metrics_and_verified_holdout_index(tmp_path, monkeypatch):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"2+2?","expected":"4"}\n')
    artifact = _artifact(tmp_path)
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=_config("eval.jsonl"))

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            result_path = Path(command[command.index("--result") + 1])
            result_path.write_text(json.dumps(_fake_payload(
                result_path, metric=0.75, device="cuda:0", gpu_count=1
            )))

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
    assert result.source_artifact_ref == artifact.artifact_ref
    assert result.gpu_hours >= 0
    assert result.gpu_hours < artifact.gpu_hours
    assert result.evidence["artifact_sha256"] == artifact.evidence["artifact_sha256"]
    assert len(result.evidence["evaluation_dataset_sha256"]["quality"]) == 64
    assert len(result.evidence["holdout_fingerprint_sha256"]["quality"]) == 64


def test_evaluator_rejects_tampered_holdout_fingerprint_index(tmp_path, monkeypatch):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"2+2?","expected":"4"}\n')
    artifact = _artifact(tmp_path)
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=_config("eval.jsonl"))

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            result_path = Path(command[command.index("--result") + 1])
            payload = _fake_payload(result_path, metric=1.0, device="cpu", gpu_count=0)
            fingerprint_path = Path(payload["suites"]["quality"]["holdout_fingerprints_file"])
            fingerprint_path.write_text("tampered\n", encoding="utf-8")
            result_path.write_text(json.dumps(payload))

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr("chowder.evaluators.transformers_text.subprocess.Popen", FakeProcess)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        TransformersTextEvaluator().evaluate(
            experiment=_experiment(), artifact=artifact, context=context
        )


def test_worker_scoring_is_deterministic_and_explicit():
    assert _score(" Answer  42 ", "answer 42", "normalized_exact_match") == 1.0
    assert _score("Answer", "answer", "exact_match") == 0.0


def test_cpu_evaluation_does_not_consume_gpu_hour_budget(tmp_path, monkeypatch):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"2+2?","expected":"4"}\n')
    artifact = _artifact(tmp_path)
    config = _config("eval.jsonl")
    config["evaluation"]["device"] = "cpu"
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            result_path = Path(command[command.index("--result") + 1])
            result_path.write_text(json.dumps(_fake_payload(
                result_path, metric=1.0, device="cpu", gpu_count=0
            )))

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
    assert result.gpu_hours == 0.0
    assert result.evidence["runtime"]["device"] == "cpu"


def test_evaluator_profile_reads_declared_estimated_gpu_hours(tmp_path):
    context = ExecutionContext(
        _hardware(),
        str(tmp_path),
        1,
        resolved_config={"evaluation": {"estimated_gpu_hours": 0.3}},
    )
    estimate = TransformersTextEvaluator().profile(_experiment(), context)
    assert estimate.gpu_hours == pytest.approx(0.3)
    assert estimate.confidence == 0.25


def test_evaluator_profile_defaults_to_zero_when_unset(tmp_path):
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config={})
    estimate = TransformersTextEvaluator().profile(_experiment(), context)
    assert estimate.gpu_hours == 0.0


def test_evaluator_cancel_terminates_a_tracked_running_process():
    class FakeRunningProcess:
        def __init__(self):
            self.terminated = False
            self.waited = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.waited = True

    evaluator = TransformersTextEvaluator()
    process = FakeRunningProcess()
    evaluator._processes["run-1"] = process
    evaluator.cancel("run-1")
    assert process.terminated
    assert process.waited


def test_evaluator_cancel_is_a_no_op_for_unknown_or_finished_run():
    evaluator = TransformersTextEvaluator()
    evaluator.cancel("never-started")

    class FinishedProcess:
        def poll(self):
            return 0

        def terminate(self):
            raise AssertionError("must not terminate an already-finished process")

    evaluator._processes["run-2"] = FinishedProcess()
    evaluator.cancel("run-2")
