import json
from pathlib import Path

import pytest

from chowder.evaluators.lm_eval import LmEvalEvaluator, LmEvalSpec
from chowder.evaluators.lm_eval_worker import _extract_metrics, _model_args
from chowder.executors import ExecutionContext, TrainingArtifact
from chowder.memory import HardwareProfile
from chowder.models import Experiment, Hypothesis
from chowder.provenance import sha256_directory


def _artifact(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    return TrainingArtifact(
        "run-1",
        "e1",
        str(adapter),
        0.2,
        evidence={
            "artifact_sha256": sha256_directory(adapter),
            "model_provenance": {"resolved_model_commit": "resolved-commit"},
        },
    )


def _config():
    return {
        "seed": 23,
        "backend": {
            "type": "transformers-peft",
            "base_model": "example/model",
            "quantization": "4bit",
            "precision": "bf16",
        },
        "evaluation": {
            "type": "lm-eval",
            "tasks": ["hellaswag", "arc_easy"],
            "metric_map": {
                "hellaswag": "hellaswag:acc_norm,none",
                "arc_easy": "arc_easy:acc_norm,none",
            },
            "batch_size": "auto",
            "quantization": "inherit",
            "precision": "inherit",
        },
    }


def _experiment():
    return Experiment("e1", None, Hypothesis("obs", "cause", "fix"), {}, 1.0)


def _context(tmp_path):
    return ExecutionContext(
        HardwareProfile(16, 64, 500, 12, 40, 3),
        str(tmp_path),
        1,
        resolved_config=_config(),
    )


def test_lm_eval_spec_uses_training_commit_and_explicit_metric_map(tmp_path):
    spec = LmEvalSpec.from_context(
        config=_config(),
        artifact=_artifact(tmp_path),
        work_dir=tmp_path,
        output_dir=tmp_path / "eval",
        seed=1,
    )
    assert spec.revision == "resolved-commit"
    assert spec.tasks == ("hellaswag", "arc_easy")
    assert spec.metric_map["hellaswag"] == "hellaswag:acc_norm,none"
    assert spec.quantization == "4bit"
    assert spec.precision == "bf16"
    assert spec.seed == 23


def test_model_args_loads_base_plus_peft_without_remote_code(tmp_path):
    spec = LmEvalSpec.from_context(
        config=_config(),
        artifact=_artifact(tmp_path),
        work_dir=tmp_path,
        output_dir=tmp_path / "eval",
        seed=1,
    )
    args = _model_args(spec)
    assert args["pretrained"] == "example/model"
    assert args["peft"].endswith("adapter")
    assert args["revision"] == "resolved-commit"
    assert args["load_in_4bit"] is True
    assert args["trust_remote_code"] is False


def test_extract_metrics_requires_explicit_task_metric_keys():
    raw = {
        "results": {
            "hellaswag": {"acc_norm,none": 0.81, "acc_norm_stderr,none": 0.01},
            "arc_easy": {"acc_norm,none": 0.75},
        }
    }
    mapped = _extract_metrics(raw, {
        "reasoning": "hellaswag:acc_norm,none",
        "knowledge": "arc_easy:acc_norm,none",
    })
    assert mapped == {"reasoning": 0.81, "knowledge": 0.75}
    with pytest.raises(RuntimeError, match="missing"):
        _extract_metrics(raw, {"bad": "hellaswag:does_not_exist"})


def test_lm_eval_evaluator_returns_evaluation_only_gpu_cost_and_protocol(tmp_path, monkeypatch):
    artifact = _artifact(tmp_path)

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            result_path = Path(command[command.index("--result") + 1])
            result_path.write_text(json.dumps({
                "metrics": {"hellaswag": 0.81, "arc_easy": 0.75},
                "raw_results_sha256": "a" * 64,
                "task_configs_sha256": "b" * 64,
                "runtime": {"device": "cuda:0", "gpu_count": 1},
                "versions": {
                    "lm-eval": "0.4.13",
                    "torch": "2.test",
                    "transformers": "5.test",
                    "peft": "0.test",
                },
            }))

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr("chowder.evaluators.lm_eval.subprocess.Popen", FakeProcess)
    outcome = LmEvalEvaluator().evaluate(
        experiment=_experiment(), artifact=artifact, context=_context(tmp_path)
    )
    assert outcome.source_artifact_ref == artifact.artifact_ref
    assert outcome.metrics == {"hellaswag": 0.81, "arc_easy": 0.75}
    assert outcome.gpu_hours >= 0
    assert outcome.gpu_hours < artifact.gpu_hours
    assert outcome.evidence["raw_results_sha256"] == "a" * 64
    assert outcome.evidence["task_configs_sha256"] == "b" * 64
    assert len(outcome.evidence["protocol_sha256"]) == 64
    assert outcome.evidence["protocol"]["task_configs_sha256"] == "b" * 64
