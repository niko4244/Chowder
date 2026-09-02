from dataclasses import replace

from chowder.cancellation import CancellationToken
from chowder.cycle import ExperimentCycleRunner
from chowder.engine import EvolutionEngine
from chowder.executors import EvaluationOutcome, ExecutionContext, TrainingArtifact
from chowder.failures import FailureRecord, FailureSourceRole
from chowder.memory import HardwareProfile
from chowder.models import Experiment, ExperimentResult, ExperimentStatus, Goal, Hypothesis, MetricTarget
from chowder.registry import RunRegistry


def _experiment(name="e1", hours=1.0):
    return Experiment(name, None, Hypothesis("obs", "cause", "fix"), {}, hours)


def _engine():
    return EvolutionEngine(
        Goal((MetricTarget("quality", minimum=0.8),), gpu_hour_budget=10),
        ExperimentResult("base", {"quality": 0.7}, 0),
    )


def _context(tmp_path):
    return ExecutionContext(HardwareProfile(16, 64, 500, 12, 40, 3), str(tmp_path), 7)


class Trainer:
    name = "fake-trainer"

    def profile(self, experiment, context):
        raise NotImplementedError

    def run(self, experiment, context):
        return TrainingArtifact("train-1", experiment.experiment_id, "/artifact", 0.4, evidence={"sha": "x"})

    def cancel(self, run_id):
        pass


class Evaluator:
    name = "fake-eval"

    def profile(self, experiment, context):
        raise NotImplementedError

    def evaluate(self, *, experiment, artifact, context):
        return EvaluationOutcome("eval-1", experiment.experiment_id, artifact.artifact_ref, {"quality": 0.85}, 0.1, {"suite": "q"})

    def cancel(self, run_id):
        pass


def test_generation_combines_training_and_evaluation_cost_before_adjudication(tmp_path):
    engine = _engine()
    exp = _experiment()
    assert engine.propose([exp]) == (exp,)
    runner = ExperimentCycleRunner(engine, Trainer(), Evaluator(), _context(tmp_path))
    outcome = runner.run_generation([exp])
    assert outcome.promoted is not None
    assert outcome.promoted.gpu_hours == 0.5
    assert engine.spent_gpu_hours == 0.5
    assert engine.baseline.metrics["quality"] == 0.85
    assert engine.graph.nodes["e1"].status is ExperimentStatus.PASSED
    assert outcome.promoted.evidence["compute"]["evaluation_gpu_hours"] == 0.1


def test_generation_rejects_nonfinite_metrics_and_conservatively_charges_failure(tmp_path):
    class BadEvaluator(Evaluator):
        def evaluate(self, *, experiment, artifact, context):
            return replace(super().evaluate(experiment=experiment, artifact=artifact, context=context), metrics={"quality": float("nan")})

    engine = _engine()
    exp = _experiment(hours=1.0)
    engine.propose([exp])
    outcome = ExperimentCycleRunner(engine, Trainer(), BadEvaluator(), _context(tmp_path)).run_generation([exp])
    assert outcome.promoted is None
    assert outcome.failures[0].error.startswith("ValueError:")
    assert engine.graph.nodes["e1"].status is ExperimentStatus.FAILED
    assert engine.spent_gpu_hours == 1.0
    assert engine.reserved_gpu_hours == 0


def test_generation_requires_a_proposed_reserved_experiment(tmp_path):
    engine = _engine()
    exp = _experiment()
    runner = ExperimentCycleRunner(engine, Trainer(), Evaluator(), _context(tmp_path))
    try:
        runner.run_generation([exp])
    except ValueError as exc:
        assert "proposed" in str(exc)
    else:
        raise AssertionError("unproposed experiment was allowed to execute")


def test_generation_persists_training_evaluation_result_and_status(tmp_path):
    engine = _engine()
    exp = _experiment()
    engine.propose([exp])
    with RunRegistry(tmp_path / "runs.db") as registry:
        registry.record_experiment(exp)
        runner = ExperimentCycleRunner(engine, Trainer(), Evaluator(), _context(tmp_path), registry=registry)
        outcome = runner.run_generation([exp])
        assert outcome.promoted is not None
        assert len(tuple(registry.list_training_artifacts())) == 1
        evaluations = tuple(registry.list_evaluation_outcomes())
        assert len(evaluations) == 1
        assert evaluations[0].gpu_hours == 0.1
        results = tuple(registry.list_results())
        assert results[0].gpu_hours == 0.5


def test_gate_rejection_marks_candidate_rejected_and_still_accounts_total_cost(tmp_path):
    class RegressingEvaluator(Evaluator):
        def evaluate(self, *, experiment, artifact, context):
            return EvaluationOutcome(
                "eval-bad",
                experiment.experiment_id,
                artifact.artifact_ref,
                {"quality": 0.65},
                0.1,
                {},
            )

    engine = _engine()
    exp = _experiment()
    engine.propose([exp])
    outcome = ExperimentCycleRunner(engine, Trainer(), RegressingEvaluator(), _context(tmp_path)).run_generation([exp])
    assert outcome.promoted is None
    assert engine.graph.nodes["e1"].status is ExperimentStatus.REJECTED
    assert engine.spent_gpu_hours == 0.5
    assert outcome.ranking[0].decision.accepted is False


def test_generation_harvests_and_persists_failure_diagnostics(tmp_path):
    class ProtocolEvaluator(Evaluator):
        def evaluate(self, *, experiment, artifact, context):
            return EvaluationOutcome(
                "eval-1",
                experiment.experiment_id,
                artifact.artifact_ref,
                {"quality": 0.85},
                0.1,
                {"protocol_sha256": "a" * 64},
            )

    def harvester(evaluation):
        return (
            FailureRecord(
                failure_id="f" * 64,
                experiment_id=evaluation.experiment_id,
                evaluation_run_id=evaluation.run_id,
                evaluator="fake-eval",
                suite="quality",
                row_index=0,
                protocol_sha256="a" * 64,
                artifact_sha256="b" * 64,
                source_role=FailureSourceRole.GATE_HOLDOUT,
                prompt="hard prompt",
                expected="right",
                prediction="wrong",
                score=0.0,
                failure_kind="answer_mismatch",
            ),
        )

    engine = _engine()
    exp = _experiment()
    engine.propose([exp])
    with RunRegistry(tmp_path / "runs.db") as registry:
        registry.record_experiment(exp)
        runner = ExperimentCycleRunner(
            engine,
            Trainer(),
            ProtocolEvaluator(),
            _context(tmp_path),
            registry=registry,
            failure_harvester=harvester,
        )
        outcome = runner.run_generation([exp])
        assert outcome.promoted is not None
        assert len(outcome.harvested_failures) == 1
        assert len(outcome.repair_plans) == 1
        assert len(tuple(registry.list_failures())) == 1
        assert len(tuple(registry.list_repair_plans())) == 1
        assert outcome.promoted.evidence["diagnostics"]["failure_count"] == 1


def test_diagnostic_failure_does_not_invalidate_valid_evaluation(tmp_path):
    def broken_harvester(evaluation):
        raise RuntimeError("diagnostic parser broke")

    engine = _engine()
    exp = _experiment()
    engine.propose([exp])
    runner = ExperimentCycleRunner(
        engine,
        Trainer(),
        Evaluator(),
        _context(tmp_path),
        failure_harvester=broken_harvester,
    )
    outcome = runner.run_generation([exp])
    assert outcome.promoted is not None
    assert outcome.candidates[0].diagnostic_error == "RuntimeError: diagnostic parser broke"


# --- cooperative cancellation -------------------------------------------------


def test_cancellation_requested_before_start_skips_the_candidate_cleanly(tmp_path):
    class MustNotRun(Trainer):
        def run(self, experiment, context):
            raise AssertionError("must not run once cancellation was requested")

    engine = _engine()
    exp = _experiment()
    engine.propose([exp])
    token = CancellationToken()
    token.request()
    runner = ExperimentCycleRunner(
        engine, MustNotRun(), Evaluator(), _context(tmp_path), cancellation=token
    )
    outcome = runner.run_generation([exp])
    candidate = outcome.candidates[0]
    assert candidate.error == "cancelled before start"
    assert candidate.artifact is None
    assert engine.spent_gpu_hours == 0
    assert engine.reserved_gpu_hours == 0
    assert not engine.has_reservation(exp.experiment_id)


def test_cancellation_during_training_is_reported_cleanly_without_investigation(tmp_path):
    token = CancellationToken()

    class CancelsMidTraining(Trainer):
        def run(self, experiment, context):
            # Simulates request() successfully terminating an in-flight
            # subprocess: the token becomes requested, and the interrupted
            # call raises rather than returning an artifact.
            token.request()
            raise RuntimeError("worker terminated")

    engine = _engine()
    exp = _experiment()
    engine.propose([exp])
    runner = ExperimentCycleRunner(
        engine, CancelsMidTraining(), Evaluator(), _context(tmp_path), cancellation=token
    )
    outcome = runner.run_generation([exp])
    candidate = outcome.candidates[0]
    assert candidate.error is not None
    assert candidate.error.startswith("cancelled: ")
    assert candidate.executor_analysis is None


def test_cancellation_during_evaluation_is_reported_cleanly_without_investigation(tmp_path):
    token = CancellationToken()

    class CancelsMidEvaluation(Evaluator):
        def evaluate(self, *, experiment, artifact, context):
            token.request()
            raise RuntimeError("worker terminated")

    engine = _engine()
    exp = _experiment()
    engine.propose([exp])
    runner = ExperimentCycleRunner(
        engine, Trainer(), CancelsMidEvaluation(), _context(tmp_path), cancellation=token
    )
    outcome = runner.run_generation([exp])
    candidate = outcome.candidates[0]
    assert candidate.error is not None
    assert candidate.error.startswith("cancelled: ")
    assert candidate.executor_analysis is None
    assert candidate.artifact is not None  # training itself completed normally


def test_bind_cancellation_is_used_when_the_executor_supports_it(tmp_path):
    seen = []

    class BindAwareTrainer(Trainer):
        def bind_cancellation(self, token):
            seen.append(token)

    engine = _engine()
    exp = _experiment()
    engine.propose([exp])
    token = CancellationToken()
    runner = ExperimentCycleRunner(
        engine, BindAwareTrainer(), Evaluator(), _context(tmp_path), cancellation=token
    )
    runner.run_generation([exp])
    assert seen == [token, None]  # bound before run(), cleared afterward


def test_a_trainer_without_bind_cancellation_support_is_unaffected(tmp_path):
    """Trainer/Evaluator (used throughout this file) never define
    bind_cancellation -- proves passing a token doesn't break a plain
    executor that doesn't opt into the capability."""
    engine = _engine()
    exp = _experiment()
    engine.propose([exp])
    runner = ExperimentCycleRunner(
        engine, Trainer(), Evaluator(), _context(tmp_path), cancellation=CancellationToken()
    )
    outcome = runner.run_generation([exp])
    assert outcome.promoted is not None
    assert outcome.candidates[0].error is None


def test_bind_progress_is_used_when_the_trainer_supports_it(tmp_path):
    seen = []

    class ProgressAwareTrainer(Trainer):
        def bind_progress_callback(self, callback):
            seen.append(callback)

    engine = _engine()
    exp = _experiment()
    engine.propose([exp])
    callback = lambda event: None  # noqa: E731
    runner = ExperimentCycleRunner(
        engine, ProgressAwareTrainer(), Evaluator(), _context(tmp_path), progress_callback=callback
    )
    runner.run_generation([exp])
    assert seen == [callback, None]  # bound before run(), cleared afterward


def test_a_trainer_without_bind_progress_support_is_unaffected(tmp_path):
    """Trainer/Evaluator (used throughout this file) never define
    bind_progress_callback -- proves passing one doesn't break a plain
    executor that doesn't opt into the capability."""
    engine = _engine()
    exp = _experiment()
    engine.propose([exp])
    runner = ExperimentCycleRunner(
        engine, Trainer(), Evaluator(), _context(tmp_path), progress_callback=lambda event: None
    )
    outcome = runner.run_generation([exp])
    assert outcome.promoted is not None
    assert outcome.candidates[0].error is None
