from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from chowder.checkpoint_bisect import evaluate_all_checkpoints
from chowder.executors import EvaluationOutcome, ExecutionContext
from chowder.memory import HardwareProfile
from chowder.models import Experiment, ExperimentResult, Goal, Hypothesis, MetricTarget

_REAL_ML_SMOKE = pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
_TINY_MODEL = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"


class FakeEvaluator:
    """Returns a controllable score per checkpoint step. Never fakes the
    checkpoint directory itself or the sha256 digest computed over it --
    those are real filesystem content hashed by the real provenance helper,
    only the evaluation score is scripted."""

    name = "fake-eval"

    def __init__(self, scores_by_step: dict[int, float]):
        self.scores_by_step = scores_by_step
        self.evaluated_artifact_refs: list[str] = []

    def profile(self, experiment, context):
        raise NotImplementedError

    def evaluate(self, *, experiment, artifact, context):
        self.evaluated_artifact_refs.append(artifact.artifact_ref)
        step = int(Path(artifact.artifact_ref).name.rsplit("-", 1)[1])
        return EvaluationOutcome(
            f"eval-{artifact.run_id}",
            experiment.experiment_id,
            artifact.artifact_ref,
            {"quality": self.scores_by_step[step]},
            0.001,
            {"suite": "fake"},
        )

    def cancel(self, run_id):
        pass


def _hardware() -> HardwareProfile:
    return HardwareProfile(16, 64, 500, 12, 40, 3)


def _write_checkpoints(adapter_dir: Path, steps: list[int]) -> None:
    trainer_dir = adapter_dir / "trainer"
    for step in steps:
        checkpoint_dir = trainer_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "adapter_model.safetensors").write_bytes(f"weights-{step}".encode())


@pytest.fixture
def goal() -> Goal:
    return Goal(metrics=(MetricTarget(name="quality", weight=1.0, minimum=0.0),), gpu_hour_budget=100.0)


@pytest.fixture
def baseline() -> ExperimentResult:
    return ExperimentResult("baseline", {"quality": 0.80}, 1.0)


@pytest.fixture
def experiment() -> Experiment:
    return Experiment("rejected", None, Hypothesis("o", "c", "i"), {}, 1.0)


def test_finds_first_regressing_checkpoint_by_step(tmp_path, goal, baseline, experiment):
    adapter_dir = tmp_path / "adapter"
    _write_checkpoints(adapter_dir, [2, 4, 6, 8])
    rejected_result = ExperimentResult("rejected", {"quality": 0.55}, 4.0, artifact_ref=str(adapter_dir))
    evaluator = FakeEvaluator({2: 0.82, 4: 0.90, 6: 0.60, 8: 0.55})
    context = ExecutionContext(work_dir=str(tmp_path), seed=0, hardware=_hardware(), resolved_config={})

    outcome = evaluate_all_checkpoints(
        experiment=experiment,
        rejected_result=rejected_result,
        evaluator=evaluator,
        context=context,
        goal=goal,
        baseline=baseline,
    )

    assert [v.step for v in outcome.verdicts] == [2, 4, 6, 8]
    assert [v.decision.accepted for v in outcome.verdicts] == [True, True, False, False]
    assert outcome.first_regressing is not None
    assert outcome.first_regressing.step == 6
    assert len(evaluator.evaluated_artifact_refs) == 4


def test_first_regressing_is_none_when_every_checkpoint_passes_the_gate(
    tmp_path, goal, baseline, experiment
):
    adapter_dir = tmp_path / "adapter"
    _write_checkpoints(adapter_dir, [2, 4])
    rejected_result = ExperimentResult("rejected", {"quality": 0.90}, 2.0, artifact_ref=str(adapter_dir))
    evaluator = FakeEvaluator({2: 0.82, 4: 0.90})
    context = ExecutionContext(work_dir=str(tmp_path), seed=0, hardware=_hardware(), resolved_config={})

    outcome = evaluate_all_checkpoints(
        experiment=experiment,
        rejected_result=rejected_result,
        evaluator=evaluator,
        context=context,
        goal=goal,
        baseline=baseline,
    )

    assert outcome.first_regressing is None


def test_verdicts_carry_real_checkpoint_provenance(tmp_path, goal, baseline, experiment):
    adapter_dir = tmp_path / "adapter"
    _write_checkpoints(adapter_dir, [2])
    rejected_result = ExperimentResult("rejected", {"quality": 0.82}, 1.0, artifact_ref=str(adapter_dir))
    evaluator = FakeEvaluator({2: 0.82})
    context = ExecutionContext(work_dir=str(tmp_path), seed=0, hardware=_hardware(), resolved_config={})

    outcome = evaluate_all_checkpoints(
        experiment=experiment,
        rejected_result=rejected_result,
        evaluator=evaluator,
        context=context,
        goal=goal,
        baseline=baseline,
    )

    verdict = outcome.verdicts[0]
    assert verdict.result.evidence["checkpoint_step"] == 2
    digest = verdict.result.evidence["checkpoint_sha256"]
    assert isinstance(digest, str) and len(digest) == 64
    assert verdict.result.evidence["evaluation"] == {"suite": "fake"}
    # changing the checkpoint's real on-disk content changes the recorded digest
    (adapter_dir / "trainer" / "checkpoint-2" / "adapter_model.safetensors").write_bytes(b"different")
    outcome2 = evaluate_all_checkpoints(
        experiment=experiment,
        rejected_result=rejected_result,
        evaluator=evaluator,
        context=context,
        goal=goal,
        baseline=baseline,
    )
    assert outcome2.verdicts[0].result.evidence["checkpoint_sha256"] != digest


def test_no_checkpoints_returns_empty_outcome(tmp_path, goal, baseline, experiment):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    rejected_result = ExperimentResult("rejected", {"quality": 0.50}, 1.0, artifact_ref=str(adapter_dir))
    evaluator = FakeEvaluator({})
    context = ExecutionContext(work_dir=str(tmp_path), seed=0, hardware=_hardware(), resolved_config={})

    outcome = evaluate_all_checkpoints(
        experiment=experiment,
        rejected_result=rejected_result,
        evaluator=evaluator,
        context=context,
        goal=goal,
        baseline=baseline,
    )

    assert outcome.verdicts == ()
    assert outcome.first_regressing is None


def test_non_checkpoint_directories_under_trainer_are_ignored(tmp_path, goal, baseline, experiment):
    adapter_dir = tmp_path / "adapter"
    _write_checkpoints(adapter_dir, [2])
    (adapter_dir / "trainer" / "not-a-checkpoint").mkdir(parents=True)
    rejected_result = ExperimentResult("rejected", {"quality": 0.82}, 1.0, artifact_ref=str(adapter_dir))
    evaluator = FakeEvaluator({2: 0.82})
    context = ExecutionContext(work_dir=str(tmp_path), seed=0, hardware=_hardware(), resolved_config={})

    outcome = evaluate_all_checkpoints(
        experiment=experiment,
        rejected_result=rejected_result,
        evaluator=evaluator,
        context=context,
        goal=goal,
        baseline=baseline,
    )

    assert [v.step for v in outcome.verdicts] == [2]


@_REAL_ML_SMOKE
def test_real_bisect_locates_regression_across_real_checkpoints(tmp_path):
    """End-to-end with a REAL tiny model training run producing real
    step-numbered checkpoints and a REAL independent evaluator -- not a
    training-loss proxy. Uses a strict goal (a quality floor no real
    checkpoint of an undertrained tiny model can reach) so every real
    checkpoint is genuinely gate-rejected, proving the bisect logic against
    real evaluation subprocess output, not scripted scores."""
    import json

    from chowder.backends.transformers_peft import TransformersPeftExecutor
    from chowder.evaluators.transformers_text import TransformersTextEvaluator

    train_path = tmp_path / "train.jsonl"
    rows = [{"text": f"Question: what is {i}? Answer: {i * 2}"} for i in range(20)]
    train_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"prompt": "Question: what is 1? Answer:", "expected": "2"}) + "\n",
        encoding="utf-8",
    )

    resolved_config = {
        "backend": {
            "type": "transformers-peft",
            "base_model": _TINY_MODEL,
            "dataset": str(train_path),
            "max_length": 64,
            "quantization": "none",
            "precision": "fp32",
            "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
            "training": {
                "max_steps": 4,
                "save_strategy": "steps",
                "save_steps": 2,
                "learning_rate": 0.001,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "logging_steps": 1,
                "gradient_checkpointing": False,
            },
            "runtime": {"timeout_seconds": 180.0},
        },
        "evaluation": {
            "type": "transformers-text",
            "estimated_gpu_hours": 0.05,
            "precision": "fp32",
            "quantization": "none",
            "device": "cpu",
            "trust_remote_code": False,
            "runtime": {"timeout_seconds": 180.0},
            "suites": [
                {
                    "name": "quality",
                    "dataset": str(eval_path),
                    "prompt_field": "prompt",
                    "expected_field": "expected",
                    "scoring": "normalized_exact_match",
                    "max_new_tokens": 2,
                    "use_chat_template": False,
                }
            ],
        },
    }
    context = ExecutionContext(
        hardware=_hardware(), work_dir=str(tmp_path), seed=1, resolved_config=resolved_config
    )
    experiment = Experiment("rejected-real", None, Hypothesis("o", "c", "i"), {}, 1.0)

    trainer = TransformersPeftExecutor()
    artifact = trainer.run(experiment, context)
    checkpoints = sorted((Path(artifact.artifact_ref) / "trainer").glob("checkpoint-*"))
    assert len(checkpoints) >= 2, "test setup must produce multiple real checkpoints to bisect"

    rejected_result = ExperimentResult(
        experiment.experiment_id, {"quality": 0.0}, artifact.gpu_hours, artifact_ref=artifact.artifact_ref
    )
    strict_goal = Goal(
        metrics=(MetricTarget(name="quality", weight=1.0, minimum=1.1),), gpu_hour_budget=100.0
    )
    baseline = ExperimentResult("baseline", {"quality": 1.0}, 1.0)
    evaluator = TransformersTextEvaluator()

    outcome = evaluate_all_checkpoints(
        experiment=experiment,
        rejected_result=rejected_result,
        evaluator=evaluator,
        context=context,
        goal=strict_goal,
        baseline=baseline,
    )

    assert len(outcome.verdicts) == len(checkpoints)
    assert all(not v.decision.accepted for v in outcome.verdicts), (
        "an unreachable quality floor must gate-reject every real checkpoint"
    )
    assert outcome.first_regressing is not None
    assert outcome.first_regressing.step == outcome.verdicts[0].step
    for verdict in outcome.verdicts:
        assert len(verdict.result.evidence["checkpoint_sha256"]) == 64
