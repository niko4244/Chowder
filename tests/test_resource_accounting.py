import math

import pytest

from chowder.cycle import ExperimentCycleRunner
from chowder.engine import EvolutionEngine
from chowder.executors import (
    CostEstimate,
    EvaluationOutcome,
    ExecutionContext,
    TrainingArtifact,
)
from chowder.memory import HardwareProfile
from chowder.models import Experiment, ExperimentResult, Goal, Hypothesis, MetricTarget
from chowder.resources import ResourceUsage


def _hardware() -> HardwareProfile:
    return HardwareProfile(
        vram_gb=16,
        ram_gb=32,
        nvme_gb=100,
        pcie_gbps=12,
        ram_gbps=40,
        nvme_gbps=3,
    )


def _experiment(experiment_id: str = "candidate", hours: float = 0.5) -> Experiment:
    return Experiment(
        experiment_id=experiment_id,
        parent_id=None,
        hypothesis=Hypothesis("obs", "cause", "fix"),
        config_patch={},
        estimated_gpu_hours=hours,
    )


def _engine(budget: float = 2.0) -> EvolutionEngine:
    goal = Goal((MetricTarget("score", minimum=1),), gpu_hour_budget=budget)
    baseline = ExperimentResult("base", {"score": 1.0}, 0.0)
    return EvolutionEngine(goal, baseline)


def test_kaggle_two_t4s_charge_two_gpu_hours_for_one_wall_hour():
    usage = ResourceUsage.from_wall_time(
        wall_seconds=3600,
        active_accelerator_count=2,
        visible_accelerator_count=2,
        peak_vram_gb_by_accelerator={"cuda:0": 15.2, "cuda:1": 15.1},
    )
    assert usage.gpu_hours == pytest.approx(2.0)
    assert usage.wall_seconds == pytest.approx(3600)
    assert usage.active_accelerator_count == 2
    assert usage.peak_vram_gb_by_accelerator == {"cuda:0": 15.2, "cuda:1": 15.1}


def test_resource_usage_does_not_treat_two_gpu_vram_pools_as_one():
    usage = ResourceUsage.from_wall_time(
        wall_seconds=10,
        active_accelerator_count=2,
        visible_accelerator_count=2,
        peak_vram_gb_by_accelerator={"t4-0": 16.0, "t4-1": 16.0},
    )
    assert max(usage.peak_vram_gb_by_accelerator.values()) == 16.0
    assert 32.0 not in usage.peak_vram_gb_by_accelerator.values()


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_resource_usage_rejects_nonfinite_values(bad):
    with pytest.raises(ValueError):
        ResourceUsage(wall_seconds=bad, accelerator_seconds=0)


def test_metric_target_rejects_invalid_bounds_and_tolerance():
    with pytest.raises(ValueError, match="minimum cannot exceed maximum"):
        MetricTarget("score", minimum=2, maximum=1)
    with pytest.raises(ValueError, match="regression_tolerance cannot be negative"):
        MetricTarget("score", regression_tolerance=-0.1)


def test_goal_rejects_duplicate_metric_names():
    with pytest.raises(ValueError, match="must be unique"):
        Goal(
            (MetricTarget("score"), MetricTarget("score")),
            gpu_hour_budget=1,
        )


def test_experiment_and_result_reject_nan_compute():
    with pytest.raises(ValueError, match="must be finite"):
        _experiment(hours=math.nan)
    with pytest.raises(ValueError, match="must be finite"):
        ExperimentResult("x", {"score": 1}, math.nan)


def test_engine_can_resize_profiled_reservation_up_and_down():
    engine = _engine(3.0)
    experiment = _experiment(hours=0.5)
    assert engine.propose((experiment,)) == (experiment,)
    engine.resize_reservation("candidate", 1.25)
    assert engine.reservation_for("candidate") == pytest.approx(1.25)
    assert engine.reserved_gpu_hours == pytest.approx(1.25)
    engine.resize_reservation("candidate", 0.75)
    assert engine.reserved_gpu_hours == pytest.approx(0.75)


def test_engine_refuses_profiled_reservation_that_cannot_fit():
    engine = _engine(1.0)
    experiment = _experiment(hours=0.5)
    engine.propose((experiment,))
    with pytest.raises(ValueError, match="does not fit remaining budget"):
        engine.resize_reservation("candidate", 1.5)
    assert engine.reservation_for("candidate") == pytest.approx(0.5)
    assert engine.spent_gpu_hours == 0


class _ProfileTooLargeTrainer:
    name = "profile-too-large"
    ran = False

    def profile(self, experiment, context):
        return CostEstimate(gpu_hours=2.0)

    def run(self, experiment, context):
        self.ran = True
        raise AssertionError("training must not run when preflight cannot fit")

    def cancel(self, run_id):
        return None


class _UnusedEvaluator:
    name = "unused"

    def profile(self, experiment, context):
        raise NotImplementedError

    def evaluate(self, *, experiment, artifact, context):
        raise AssertionError("evaluation must not run")

    def cancel(self, run_id):
        return None


def test_cycle_preflight_releases_budget_without_charging_when_profile_cannot_fit(tmp_path):
    engine = _engine(1.0)
    experiment = _experiment(hours=0.5)
    engine.propose((experiment,))
    trainer = _ProfileTooLargeTrainer()
    runner = ExperimentCycleRunner(
        engine=engine,
        trainer=trainer,
        evaluator=_UnusedEvaluator(),
        context=ExecutionContext(_hardware(), str(tmp_path), seed=1),
    )
    outcome = runner.run_generation((experiment,))
    assert not trainer.ran
    assert outcome.candidates[0].error.startswith("preflight ValueError")
    assert engine.spent_gpu_hours == 0
    assert engine.reserved_gpu_hours == 0
    assert not engine.has_reservation(experiment.experiment_id)


class _MeasuredTrainer:
    name = "measured"

    def profile(self, experiment, context):
        return CostEstimate(gpu_hours=0.8)

    def run(self, experiment, context):
        return TrainingArtifact(
            run_id="train-1",
            experiment_id=experiment.experiment_id,
            artifact_ref="adapter",
            gpu_hours=0.5,
        )

    def cancel(self, run_id):
        return None


class _NoProfileTrainer(_MeasuredTrainer):
    def profile(self, experiment, context):
        raise NotImplementedError


class _MeasuredEvaluator:
    name = "measured-eval"

    def profile(self, experiment, context):
        raise NotImplementedError

    def evaluate(self, *, experiment, artifact, context):
        return EvaluationOutcome(
            run_id="eval-1",
            experiment_id=experiment.experiment_id,
            source_artifact_ref=artifact.artifact_ref,
            metrics={"score": 2.0},
            gpu_hours=0.1,
        )

    def cancel(self, run_id):
        return None


class _ProfiledEvaluator(_MeasuredEvaluator):
    """Unlike _MeasuredEvaluator, implements profile() -- this is the branch
    that should take priority over the declared evaluation.estimated_gpu_hours
    fallback, mirroring how a trainer's own profile() already outranks the
    engine's existing reservation."""

    name = "profiled-eval"

    def profile(self, experiment, context):
        return CostEstimate(gpu_hours=0.15)


def test_cycle_reserves_training_profile_plus_declared_evaluation_cost(tmp_path):
    engine = _engine(2.0)
    experiment = _experiment(hours=0.25)
    engine.propose((experiment,))
    runner = ExperimentCycleRunner(
        engine=engine,
        trainer=_MeasuredTrainer(),
        evaluator=_MeasuredEvaluator(),
        context=ExecutionContext(_hardware(), str(tmp_path), seed=1),
        base_config={"evaluation": {"estimated_gpu_hours": 0.2}},
    )
    outcome = runner.run_generation((experiment,))
    assert outcome.promoted is not None
    assert outcome.promoted.gpu_hours == pytest.approx(0.6)
    compute = outcome.promoted.evidence["compute"]
    assert compute["reserved_lifecycle_gpu_hours"] == pytest.approx(1.0)
    assert engine.spent_gpu_hours == pytest.approx(0.6)


def test_cycle_uses_existing_reservation_when_profile_is_not_implemented(tmp_path):
    engine = _engine(2.0)
    experiment = _experiment(hours=0.75)
    engine.propose((experiment,))
    runner = ExperimentCycleRunner(
        engine=engine,
        trainer=_NoProfileTrainer(),
        evaluator=_MeasuredEvaluator(),
        context=ExecutionContext(_hardware(), str(tmp_path), seed=1),
        base_config={"evaluation": {"estimated_gpu_hours": 0.1}},
    )
    outcome = runner.run_generation((experiment,))
    assert outcome.promoted is not None
    assert outcome.promoted.evidence["compute"]["reserved_lifecycle_gpu_hours"] == pytest.approx(0.85)
    assert engine.spent_gpu_hours == pytest.approx(0.6)


def test_cycle_reserves_evaluator_profile_over_declared_reserve_when_implemented(tmp_path):
    engine = _engine(2.0)
    experiment = _experiment(hours=0.25)
    engine.propose((experiment,))
    runner = ExperimentCycleRunner(
        engine=engine,
        trainer=_MeasuredTrainer(),
        evaluator=_ProfiledEvaluator(),
        context=ExecutionContext(_hardware(), str(tmp_path), seed=1),
        # A declared estimate is present too, but the evaluator's own
        # profile() (0.15) must win over it (0.9), the same way a trainer's
        # profile() already outranks the engine's existing reservation.
        base_config={"evaluation": {"estimated_gpu_hours": 0.9}},
    )
    outcome = runner.run_generation((experiment,))
    assert outcome.promoted is not None
    compute = outcome.promoted.evidence["compute"]
    assert compute["reserved_lifecycle_gpu_hours"] == pytest.approx(0.95)
    assert engine.spent_gpu_hours == pytest.approx(0.6)
