"""Production persistence of executor-failure incidents (Priority 6).

The recording path (`RunRegistry.record_execution_incident`) and the
consumption path (`build_censored_outcomes`) each had their own tests;
these cover the seam between them: the cycle runner persisting real crash
analyses so FAILED experiments carry structured classifications.
"""

from chowder.cycle import ExperimentCycleRunner
from chowder.engine import EvolutionEngine
from chowder.censored_outcomes import build_censored_outcomes
from chowder.executors import EvaluationOutcome, ExecutionContext, TrainingArtifact
from chowder.memory import HardwareProfile
from chowder.models import Experiment, ExperimentStatus, Hypothesis, MetricTarget, Goal
from chowder.registry import RunRegistry


def _experiment(name="e1", hours=1.0):
    return Experiment(name, None, Hypothesis("obs", "cause", "fix"), {}, hours)


def _engine():
    return EvolutionEngine(
        Goal((MetricTarget("quality", minimum=0.8),), gpu_hour_budget=10),
        ExperimentResult_placeholder(),
    )


def ExperimentResult_placeholder():
    from chowder.models import ExperimentResult

    return ExperimentResult("base", {"quality": 0.7}, 0)


def _context(tmp_path):
    return ExecutionContext(HardwareProfile(16, 64, 500, 12, 40, 3), str(tmp_path), 7)


class Trainer:
    name = "fake-trainer"

    def profile(self, experiment, context):
        raise NotImplementedError

    def run(self, experiment, context):
        return TrainingArtifact(
            "train-1", experiment.experiment_id, "/artifact", 0.4, evidence={"sha": "x"}
        )

    def cancel(self, run_id):
        pass


class Evaluator:
    name = "fake-eval"

    def profile(self, experiment, context):
        raise NotImplementedError

    def evaluate(self, *, experiment, artifact, context):
        return EvaluationOutcome(
            "eval-1",
            experiment.experiment_id,
            artifact.artifact_ref,
            {"quality": 0.85},
            0.1,
            {"suite": "q"},
        )

    def cancel(self, run_id):
        pass


class CrashingTrainer(Trainer):
    def run(self, experiment, context):
        raise RuntimeError("CUDA out of memory")


def test_training_crash_persists_a_classified_incident(tmp_path):
    engine = _engine()
    exp = _experiment()
    engine.propose([exp])
    with RunRegistry(tmp_path / "runs.db") as registry:
        registry.record_experiment(exp)
        runner = ExperimentCycleRunner(
            engine, CrashingTrainer(), Evaluator(), _context(tmp_path), registry=registry
        )
        outcome = runner.run_generation([exp])

        assert outcome.failures[0].error.startswith("RuntimeError:")
        assert outcome.failures[0].diagnostic_error is None
        assert outcome.failures[0].executor_analysis is not None

        rows = build_censored_outcomes(registry)
        assert len(rows) == 1
        row = rows[0]
        assert row.status is ExperimentStatus.FAILED
        assert row.incident_id is not None
        assert row.signature_kind == "cuda_oom"
        assert row.fingerprint_sha256 is not None
        assert row.executor_name == "fake-trainer"
        # The capture measured wall-seconds on the declared accelerator
        # count; it is a real non-negative measurement, not a placeholder.
        assert row.gpu_hours is not None and row.gpu_hours >= 0.0


def test_evaluation_crash_persists_a_classified_incident(tmp_path):
    class CrashingEvaluator(Evaluator):
        def evaluate(self, *, experiment, artifact, context):
            raise RuntimeError("CUBLAS_STATUS_EXECUTION_FAILED")

    engine = _engine()
    exp = _experiment()
    engine.propose([exp])
    with RunRegistry(tmp_path / "runs.db") as registry:
        registry.record_experiment(exp)
        runner = ExperimentCycleRunner(
            engine, Trainer(), CrashingEvaluator(), _context(tmp_path), registry=registry
        )
        outcome = runner.run_generation([exp])

        assert outcome.failures[0].error.startswith("RuntimeError:")
        assert outcome.failures[0].executor_analysis is not None

        rows = build_censored_outcomes(registry)
        assert len(rows) == 1
        row = rows[0]
        assert row.status is ExperimentStatus.FAILED
        assert row.signature_kind == "cuda_execution_failed"
        assert row.executor_name == "fake-eval"


def test_incident_persistence_failure_becomes_a_diagnostic(tmp_path):
    class LedgerBrokenRegistry(RunRegistry):
        def record_execution_incident(self, analysis):
            raise RuntimeError("disk full")

    engine = _engine()
    exp = _experiment()
    engine.propose([exp])
    with RunRegistry(tmp_path / "runs.db") as registry:
        registry.record_experiment(exp)
        broken = LedgerBrokenRegistry.__new__(LedgerBrokenRegistry)
        broken.path = registry.path
        broken._conn = registry._conn
        runner = ExperimentCycleRunner(
            engine, CrashingTrainer(), Evaluator(), _context(tmp_path), registry=broken
        )
        outcome = runner.run_generation([exp])

        # The crash outcome itself is unchanged...
        assert outcome.failures[0].error.startswith("RuntimeError:")
        assert outcome.failures[0].executor_analysis is not None
        # ...and the persistence failure is visible as a diagnostic.
        assert outcome.failures[0].diagnostic_error is not None
        assert "incident persistence" in outcome.failures[0].diagnostic_error
        assert "disk full" in outcome.failures[0].diagnostic_error


def test_cancelled_run_persists_no_incident(tmp_path):
    from chowder.cancellation import CancellationToken

    token = CancellationToken()

    class CancelsMidTraining(Trainer):
        def run(self, experiment, context):
            token.request()
            raise RuntimeError("worker terminated")

    engine = _engine()
    exp = _experiment()
    engine.propose([exp])
    with RunRegistry(tmp_path / "runs.db") as registry:
        registry.record_experiment(exp)
        runner = ExperimentCycleRunner(
            engine,
            CancelsMidTraining(),
            Evaluator(),
            _context(tmp_path),
            registry=registry,
            cancellation=token,
        )
        outcome = runner.run_generation([exp])

        assert outcome.failures[0].error.startswith("cancelled:")
        assert outcome.failures[0].executor_analysis is None
        rows = build_censored_outcomes(registry)
        assert len(rows) == 1
        row = rows[0]
        assert row.status is ExperimentStatus.FAILED
        # A deliberate stop is not an anomaly: no incident row exists, and
        # the view honestly reports the absence.
        assert row.incident_id is None
        assert row.signature_kind is None


def test_successful_run_persists_no_incident(tmp_path):
    engine = _engine()
    exp = _experiment()
    engine.propose([exp])
    with RunRegistry(tmp_path / "runs.db") as registry:
        registry.record_experiment(exp)
        runner = ExperimentCycleRunner(
            engine, Trainer(), Evaluator(), _context(tmp_path), registry=registry
        )
        outcome = runner.run_generation([exp])
        assert outcome.promoted is not None
        rows = build_censored_outcomes(registry)
        assert rows == ()


def test_preexisting_incident_is_not_duplicated_by_replay(tmp_path):
    """The registry's immutability contract: an identical analysis replay
    (same run_id) is idempotent, so a retried persistence attempt can never
    fork the crash history."""
    engine = _engine()
    exp = _experiment()
    engine.propose([exp])
    with RunRegistry(tmp_path / "runs.db") as registry:
        registry.record_experiment(exp)
        runner = ExperimentCycleRunner(
            engine, CrashingTrainer(), Evaluator(), _context(tmp_path), registry=registry
        )
        outcome = runner.run_generation([exp])
        analysis = outcome.failures[0].executor_analysis
        assert analysis is not None
        # Identical replay: idempotent, no error.
        registry.record_execution_incident(analysis)
        rows = list(registry.list_execution_incidents())
        assert len(rows) == 1
