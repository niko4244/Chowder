import pytest

from chowder.censored_outcomes import (
    CensoredOutcome,
    build_censored_outcomes,
    censoring_rate_by_arm,
    filter_censored,
    group_censored_by_arm,
)
from chowder.cycle import ExperimentCycleRunner
from chowder.engine import EvolutionEngine
from chowder.executors import EvaluationOutcome, ExecutionContext, TrainingArtifact
from chowder.execution_failure import ExecutionFailure, ExecutionStage
from chowder.executor_investigator import analyze_execution_failure
from chowder.intervention_outcomes import build_intervention_outcomes
from chowder.investigation import RemediationRegistry
from chowder.memory import HardwareProfile
from chowder.models import (
    Experiment,
    ExperimentResult,
    ExperimentStatus,
    Goal,
    Hypothesis,
    MetricTarget,
)
from chowder.registry import RunRegistry
from chowder.resources import ResourceUsage


def _goal(minimum=0.0):
    return Goal((MetricTarget("quality", minimum=minimum),), gpu_hour_budget=10.0)


def _baseline(quality=0.70):
    return ExperimentResult("base", {"quality": quality}, 0.0)


def _experiment(experiment_id, config_patch, *, parent_id=None, status=None, hours=1.0):
    experiment = Experiment(
        experiment_id,
        parent_id,
        Hypothesis("obs", "cause", f"intervention for {experiment_id}"),
        config_patch,
        hours,
    )
    if status is not None:
        experiment.status = status
    return experiment


def _hardware():
    return HardwareProfile(16, 64, 500, 12, 40, 3)


class _Trainer:
    name = "fake-trainer"

    def profile(self, experiment, context):
        raise NotImplementedError

    def run(self, experiment, context):
        return TrainingArtifact(
            f"train-{experiment.experiment_id}",
            experiment.experiment_id,
            "/artifact",
            0.4,
            telemetry={"global_step": 3},
            evidence={"engine": "transformers"},
        )

    def cancel(self, run_id):
        pass


class _CrashingTrainer(_Trainer):
    name = "fake-trainer"

    def run(self, experiment, context):
        raise RuntimeError("CUBLAS_STATUS_EXECUTION_FAILED")


class _Evaluator:
    name = "fake-eval"

    def profile(self, experiment, context):
        raise NotImplementedError

    def evaluate(self, *, experiment, artifact, context):
        return EvaluationOutcome(
            f"eval-{experiment.experiment_id}",
            experiment.experiment_id,
            artifact.artifact_ref,
            {"quality": 0.85},
            0.1,
            {"suite": "q"},
        )

    def cancel(self, run_id):
        pass


def test_resultless_rejected_experiment_becomes_a_censored_row(tmp_path):
    """A preflight-rejected/cancelled-before-start candidate (REJECTED, no
    result row) is exactly the censored row this view exists to represent:
    the intervention is known, nothing was measured, and every
    not-on-record field is `None` rather than a guess."""
    with RunRegistry(tmp_path / "runs.db") as registry:
        registry.record_experiment(
            _experiment("e-rej", {"backend": {"training": {"learning_rate": 1e-3}}}, status=ExperimentStatus.REJECTED)
        )
        rows = build_censored_outcomes(registry)

    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row, CensoredOutcome)
    assert row.experiment_id == "e-rej"
    assert row.parent_id is None
    assert row.status is ExperimentStatus.REJECTED
    assert row.arm == frozenset({"backend.training.learning_rate"})
    assert row.intervention == "intervention for e-rej"
    assert row.estimated_gpu_hours == 1.0
    # No incident was recorded, so there is no measured spend on record.
    assert row.gpu_hours is None
    assert row.incident_id is None
    assert row.signature_kind is None
    assert row.fingerprint_sha256 is None
    assert row.executor_name is None
    # The registry stores no error text for censored experiments; the
    # row makes that absence explicit instead of hiding the field.
    assert row.error_message is None


def test_scored_experiment_is_never_censored_even_when_gate_rejected(tmp_path):
    """An experiment with a persisted ExperimentResult has an *observed*
    outcome -- even a gate-rejected one -- so it belongs to
    `intervention_outcomes`, never to the censored view. "Censored" means
    the outcome was never measured, not that it was bad."""
    registry = RunRegistry(tmp_path / "runs.db")
    engine = EvolutionEngine(_goal(minimum=0.8), _baseline())
    experiment = _experiment("e1", {"backend": {"training": {"learning_rate": 1e-3}}})
    engine.propose([experiment])
    registry.record_experiment(experiment)
    ExperimentCycleRunner(
        engine,
        _Trainer(),
        _Evaluator(),
        ExecutionContext(_hardware(), str(tmp_path), 7),
        registry=registry,
    ).run_generation([experiment])

    outcome_rows = build_intervention_outcomes(
        registry, goal=_goal(minimum=0.8), baseline=_baseline()
    )
    censored_rows = build_censored_outcomes(registry)
    registry.close()

    assert len(outcome_rows) == 1
    assert censored_rows == ()


def test_planned_and_running_experiments_are_not_outcomes_yet(tmp_path):
    """PLANNED/RUNNING experiments have no ending to represent; emitting a
    row would fabricate one."""
    with RunRegistry(tmp_path / "runs.db") as registry:
        registry.record_experiment(_experiment("e-planned", {}, status=ExperimentStatus.PLANNED))
        registry.record_experiment(_experiment("e-running", {}, status=ExperimentStatus.RUNNING))
        assert build_censored_outcomes(registry) == ()


def test_failed_experiment_without_persisted_incident_has_none_fields(tmp_path):
    """The production crash path: the cycle runner marks the experiment
    FAILED and builds the executor analysis, but today nothing persists
    the incident into the registry. The view must report FAILED with
    every incident-derived field honestly `None` -- a real absence, not
    a classification gap to paper over."""
    registry = RunRegistry(tmp_path / "runs.db")
    engine = EvolutionEngine(_goal(minimum=0.8), _baseline())
    experiment = _experiment("e-crash", {"backend": {"training": {"learning_rate": 1e-3}}})
    engine.propose([experiment])
    registry.record_experiment(experiment)
    outcome = ExperimentCycleRunner(
        engine,
        _CrashingTrainer(),
        _Evaluator(),
        ExecutionContext(_hardware(), str(tmp_path), 7),
        registry=registry,
    ).run_generation([experiment])
    assert outcome.promoted is None
    assert outcome.candidates[0].error is not None

    rows = build_censored_outcomes(registry)
    registry.close()

    assert len(rows) == 1
    row = rows[0]
    assert row.status is ExperimentStatus.FAILED
    assert row.incident_id is None
    assert row.signature_kind is None
    assert row.fingerprint_sha256 is None
    assert row.executor_name is None
    assert row.gpu_hours is None


def test_failed_experiment_joins_recorded_incident_evidence(tmp_path):
    """When an incident analysis HAS been persisted for the experiment
    (however it got there -- the recording path exists and is tested in
    test_database_migrations), the view joins its structured evidence:
    classification, fingerprint, executor, and the capture-time measured
    GPU-hours."""
    context = ExecutionContext(
        _hardware(),
        str(tmp_path),
        seed=1,
        resolved_config={"backend": {"runtime": {"active_accelerator_count": 2}}},
    )
    failure = ExecutionFailure(
        "worker crash",
        run_id="run-crash",
        experiment_id="e1",
        executor_name="fake-trainer",
        stage=ExecutionStage.TRAIN,
        cause_type="RuntimeError",
        cause_message="CUBLAS_STATUS_EXECUTION_FAILED",
        resource_usage=ResourceUsage.from_wall_time(
            wall_seconds=36,
            active_accelerator_count=2,
            visible_accelerator_count=2,
        ),
    )
    analysis = analyze_execution_failure(
        failure,
        context=context,
        registry=RemediationRegistry(),
        gpu_hour_budget=0.25,
        investigation_id="inv-crash",
        occurred_at="2026-09-01T00:00:00+00:00",
    )

    with RunRegistry(tmp_path / "runs.db") as registry:
        registry.record_experiment(_experiment("e1", {"backend": {"training": {"learning_rate": 1e-3}}}))
        registry.update_experiment_status("e1", ExperimentStatus.FAILED.value)
        registry.record_execution_incident(analysis)

        rows = build_censored_outcomes(registry)

    assert len(rows) == 1
    row = rows[0]
    assert row.status is ExperimentStatus.FAILED
    assert row.incident_id == "execution-run-crash-attempt-1"
    assert row.signature_kind == "cuda_execution_failed"
    assert row.fingerprint_sha256 == analysis.fingerprint.fingerprint_sha256
    assert row.executor_name == "fake-trainer"
    # accelerator_seconds = 36 * 2, in GPU-hours.
    assert row.gpu_hours == pytest.approx((36 * 2) / 3600.0)


def test_first_recorded_incident_is_joined_when_several_exist(tmp_path):
    """An experiment could in principle carry more than one recorded
    incident; the view keeps the earliest-recorded classification rather
    than merging or overriding -- the deterministic stored-evidence-only
    choice."""
    context = ExecutionContext(_hardware(), str(tmp_path), seed=1)
    rows_out = []
    for index, cause in enumerate(("CUBLAS_STATUS_EXECUTION_FAILED", "CUDA out of memory")):
        failure = ExecutionFailure(
            f"crash {index}",
            run_id=f"run-{index}",
            experiment_id="e1",
            executor_name="fake-trainer",
            stage=ExecutionStage.TRAIN,
            cause_type="RuntimeError",
            cause_message=cause,
            resource_usage=ResourceUsage.from_wall_time(
                wall_seconds=10, active_accelerator_count=1
            ),
        )
        rows_out.append(
            analyze_execution_failure(
                failure,
                context=context,
                registry=RemediationRegistry(),
                gpu_hour_budget=0.25,
                investigation_id=f"inv-{index}",
                occurred_at="2026-09-01T00:00:00+00:00",
            )
        )

    with RunRegistry(tmp_path / "runs.db") as registry:
        registry.record_experiment(_experiment("e1", {}))
        registry.update_experiment_status("e1", ExperimentStatus.FAILED.value)
        for analysis in rows_out:
            registry.record_execution_incident(analysis)
        rows = build_censored_outcomes(registry)

    assert len(rows) == 1
    # rowid order: the first recorded incident wins.
    assert rows[0].signature_kind == "cuda_execution_failed"


def test_censoring_rate_by_arm_and_grouping(tmp_path):
    """Per-arm censoring shape: REJECTED-vs-FAILED split within censored
    rows, arms with no censored rows absent, and the arm keys identical
    to what the scored-row view groups by."""
    with RunRegistry(tmp_path / "runs.db") as registry:
        registry.record_experiment(_experiment("r1", {"a": 1}, status=ExperimentStatus.REJECTED))
        registry.record_experiment(_experiment("r2", {"a": 1}, status=ExperimentStatus.FAILED))
        registry.record_experiment(_experiment("r3", {"a": 1, "b": 2}, status=ExperimentStatus.REJECTED))
        registry.record_experiment(_experiment("r4", {"c": 3}))  # PLANNED: no row at all

        rows = build_censored_outcomes(registry)
        rates = censoring_rate_by_arm(rows)
        grouped = group_censored_by_arm(rows)

    arm_a = frozenset({"a"})
    arm_ab = frozenset({"a", "b"})
    arm_c = frozenset({"c"})

    assert set(grouped) == {arm_a, arm_ab}
    assert len(grouped[arm_a]) == 2
    # One of the arm-a rows was REJECTED of two censored rows.
    assert rates == {arm_a: pytest.approx(0.5), arm_ab: pytest.approx(1.0)}
    # Same arm identity the scored-row view uses.
    assert arm_a in {row.arm for row in rows}


def test_filter_censored_and_status_signature_filters(tmp_path):
    with RunRegistry(tmp_path / "runs.db") as registry:
        registry.record_experiment(_experiment("r1", {"a": 1}, status=ExperimentStatus.REJECTED))
        registry.record_experiment(_experiment("r2", {"a": 1}, status=ExperimentStatus.FAILED))
        registry.record_experiment(_experiment("r3", {"b": 2}, status=ExperimentStatus.FAILED))
        rows = build_censored_outcomes(registry)

    # Status filter.
    assert [row.experiment_id for row in filter_censored(rows, status=ExperimentStatus.REJECTED)] == ["r1"]

    # Arm filter.
    assert [row.experiment_id for row in filter_censored(rows, touches_key_path="b")] == ["r3"]

    # Signature filter: no incident was recorded for any row, so every
    # value excludes all of them -- "not on record" is not evidence of
    # any particular failure kind, and `None` rows never match.
    assert filter_censored(rows, signature_kind="cuda_oom") == ()
