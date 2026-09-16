from dataclasses import replace

import pytest

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


def test_run_round_with_promote_false_settles_reservation_but_does_not_promote(tmp_path):
    """run_generation's own real behavior (promote=True by default) is
    unchanged; run_round(promote=False) exists for successive halving,
    which must not let an early, cheap-budget round's winner become the
    new baseline just because it passed the real hard gate."""
    engine = _engine()
    exp = _experiment()
    assert engine.propose([exp]) == (exp,)
    runner = ExperimentCycleRunner(engine, Trainer(), Evaluator(), _context(tmp_path))
    outcome = runner.run_round([exp], promote=False)
    assert outcome.promoted is None
    assert engine.baseline.experiment_id == "base"
    # The real gate/settlement still happened for real -- reservation
    # settled, real GPU-hours charged, real gate decision applied.
    assert engine.spent_gpu_hours == 0.5
    assert not engine.has_reservation("e1")
    assert engine.graph.nodes["e1"].status is ExperimentStatus.PASSED
    assert len(outcome.ranking) == 1
    assert outcome.ranking[0].decision.accepted is True


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


class PairedEvaluator(Evaluator):
    """The paired router evaluator: the candidate outcome carries the
    untouched base's own score as ``base_holdout_loss`` evidence."""

    def evaluate(self, *, experiment, artifact, context):
        return EvaluationOutcome(
            "eval-1",
            experiment.experiment_id,
            artifact.artifact_ref,
            {"quality": 0.85},
            0.1,
            {"arm": "paired", "base_holdout_loss": 0.7},
        )


def _deferred_engine():
    return EvolutionEngine(
        Goal((MetricTarget("quality", minimum=0.8),), gpu_hour_budget=10),
        None,
        spent_gpu_hours=0.01,
        baseline_deferred=True,
    )


def test_a_deferred_baseline_provider_runs_between_evaluation_and_the_gate(tmp_path):
    """The amortized project path defers the baseline measurement to the
    candidate's own resident-pair evaluation; the runner must complete the
    engine's baseline BEFORE the gate adjudicates, or the gate would rank
    against a placeholder."""
    engine = _deferred_engine()
    exp = _experiment()
    assert engine.propose([exp]) == (exp,)
    seen = {}

    def provider(candidate_outcome):
        # The hook fires while the baseline is still deferred: the gate has
        # not adjudicated yet. (If it had, adjudicate itself would raise.)
        assert engine.baseline_deferred is True
        seen["base_loss"] = candidate_outcome.evaluation.evidence["base_holdout_loss"]
        return ExperimentResult("baseline", {"quality": 0.7}, 0.01)

    runner = ExperimentCycleRunner(
        engine, Trainer(), PairedEvaluator(), _context(tmp_path), deferred_baseline=provider
    )
    outcome = runner.run_generation([exp])

    assert seen == {"base_loss": 0.7}
    assert engine.baseline_deferred is False
    # Budget accounting: the provisional 0.01 was replaced by the measured
    # 0.01, and the candidate's real 0.5 was charged on top.
    assert engine.spent_gpu_hours == pytest.approx(0.51)
    # The gate adjudicated at all only because set_baseline ran first (a
    # deferred engine refuses adjudication), and it accepted because it
    # compared 0.85 against the completed 0.7 baseline. Post-promotion,
    # engine.baseline is legitimately the candidate's result.
    assert outcome.ranking[0].decision.accepted is True
    assert outcome.promoted is not None


def test_a_failing_deferred_baseline_provider_fails_the_generation_honestly(tmp_path):
    """A provider that cannot complete the measurement must fail the whole
    generation -- the candidate's numbers exist but no honest comparison
    does -- and its reservation must be settled, not stranded."""
    engine = _deferred_engine()
    exp = _experiment()
    assert engine.propose([exp]) == (exp,)

    def provider(candidate_outcome):
        raise RuntimeError("the resident pair measured no base score")

    runner = ExperimentCycleRunner(
        engine, Trainer(), Evaluator(), _context(tmp_path), deferred_baseline=provider
    )
    outcome = runner.run_generation([exp])

    assert outcome.promoted is None
    assert outcome.candidates[0].error == (
        "deferred baseline: RuntimeError: the resident pair measured no base score"
    )
    assert engine.graph.nodes["e1"].status is ExperimentStatus.FAILED
    assert not engine.has_reservation("e1"), (
        "the failed candidate's reservation must be settled, not stranded"
    )
    assert engine.spent_gpu_hours > 0


def test_the_deferred_baseline_provider_is_unused_when_the_baseline_is_present(tmp_path):
    """The hook is for the amortized path only; every existing caller keeps
    its constructor-time baseline and the provider never fires."""
    engine = _engine()
    exp = _experiment()
    assert engine.propose([exp]) == (exp,)
    calls = []

    runner = ExperimentCycleRunner(
        engine,
        Trainer(),
        Evaluator(),
        _context(tmp_path),
        deferred_baseline=lambda outcome: calls.append(outcome) or ExperimentResult("x", {"quality": 1.0}, 0.0),
    )
    outcome = runner.run_generation([exp])

    assert calls == []
    # The constructor-time baseline, not the provider's experiment, is what
    # the gate consumed (promote then legitimately replaced it).
    assert engine.baseline.experiment_id != "x"
    assert outcome.promoted is not None


def test_the_runner_completes_a_provider_that_persists_the_baseline_durably(tmp_path):
    """The provider is the single writer of the baseline row (the stranded-row
    discipline: whoever created the row settles it); the runner's job is only
    to hand it the measured evidence before the gate and complete the engine."""
    engine = _deferred_engine()
    exp = _experiment()
    assert engine.propose([exp]) == (exp,)
    with RunRegistry(tmp_path / "runs.db") as registry:
        registry.record_experiment(
            Experiment("baseline", None, Hypothesis("o", "c", "i"), {}, 0.01)
        )
        registry.record_experiment(exp)

        def provider(candidate_outcome):
            result = ExperimentResult(
                "baseline",
                {"quality": candidate_outcome.evaluation.evidence["base_holdout_loss"]},
                0.01,
            )
            registry.record_result(result)
            registry.update_experiment_status("baseline", ExperimentStatus.PASSED.value)
            return result

        runner = ExperimentCycleRunner(
            engine,
            Trainer(),
            PairedEvaluator(),
            _context(tmp_path),
            base_config={},
            registry=registry,
            deferred_baseline=provider,
        )
        outcome = runner.run_generation([exp])
        experiments = {e.experiment_id: e for e in registry.list_experiments()}
        results = {r.experiment_id: r for r in registry.list_results()}

    assert outcome.promoted is not None
    assert experiments["baseline"].status is ExperimentStatus.PASSED
    assert results["baseline"].metrics == {"quality": 0.7}
