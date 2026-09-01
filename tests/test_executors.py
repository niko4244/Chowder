import pytest

from chowder.engine import EvolutionEngine
from chowder.executors import (
    CostEstimate,
    EvaluationExecutor,
    ExecutionContext,
    TrainingArtifact,
    TrainingExecutor,
)
from chowder.memory import HardwareProfile
from chowder.models import Experiment, ExperimentResult, Goal, Hypothesis, MetricTarget


class StubTrainer:
    name = "stub-trainer"

    def profile(self, experiment, context):
        return CostEstimate(gpu_hours=0.1)

    def run(self, experiment, context):
        return TrainingArtifact(
            run_id="run-1",
            experiment_id=experiment.experiment_id,
            artifact_ref="./adapter",
            gpu_hours=0.1,
            telemetry={"train_loss": 0.4},
        )

    def cancel(self, run_id):
        return None


class StubEvaluator:
    name = "stub-evaluator"

    def evaluate(self, *, experiment, artifact, context):
        return ExperimentResult(
            experiment_id=experiment.experiment_id,
            metrics={"quality": 0.8},
            gpu_hours=artifact.gpu_hours,
            artifact_ref=artifact.artifact_ref,
        )


def _experiment():
    return Experiment(
        experiment_id="e1",
        parent_id=None,
        hypothesis=Hypothesis("obs", "cause", "change"),
        config_patch={},
        estimated_gpu_hours=0.1,
    )


def _context(tmp_path):
    return ExecutionContext(
        hardware=HardwareProfile(8, 32, 100, 12, 40, 3),
        work_dir=str(tmp_path),
        seed=1,
    )


def test_training_and_evaluation_are_separate_contracts(tmp_path):
    trainer = StubTrainer()
    evaluator = StubEvaluator()
    assert isinstance(trainer, TrainingExecutor)
    assert isinstance(evaluator, EvaluationExecutor)

    experiment = _experiment()
    artifact = trainer.run(experiment, _context(tmp_path))
    assert artifact.telemetry["train_loss"] == 0.4
    assert not hasattr(artifact, "metrics")

    result = evaluator.evaluate(experiment=experiment, artifact=artifact, context=_context(tmp_path))
    assert result.metrics["quality"] == 0.8


def test_engine_rejects_unevaluated_training_artifact():
    goal = Goal((MetricTarget("quality", minimum=0.8),), gpu_hour_budget=1)
    engine = EvolutionEngine(goal, ExperimentResult("base", {"quality": 0.7}, 0))
    artifact = TrainingArtifact("r", "e1", "./adapter", 0.1)
    with pytest.raises(TypeError, match="evaluated ExperimentResult"):
        engine.adjudicate([artifact])  # type: ignore[list-item]
