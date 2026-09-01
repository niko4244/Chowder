from __future__ import annotations

import sqlite3

import pytest

import chowder.recursive_repair as recursive
from chowder.autonomous_repair import AutonomousRepairOutcome, _repairable_target
from chowder.cycle import CandidateCycleOutcome, ExperimentCycleRunner, GenerationOutcome
from chowder.engine import EvolutionEngine
from chowder.executors import EvaluationOutcome, ExecutionContext
from chowder.failures import FailureRecord, FailureSourceRole, RepairPlan, cluster_failures
from chowder.memory import HardwareProfile
from chowder.models import ExperimentResult, GateDecision, Goal, MetricTarget
from chowder.registry import RunRegistry
from chowder.repair_candidates import RepairVariant
from chowder.recursive_trace import RecursiveRepairTraceStore
from chowder.tournament import RankedCandidate


def _candidate(experiment_id: str, *, score: float, prompt: str):
    failure = FailureRecord(
        failure_id=(experiment_id + "f" * 64)[:64],
        experiment_id=experiment_id,
        evaluation_run_id=f"eval-{experiment_id}",
        evaluator="transformers-text",
        suite="reasoning",
        row_index=0,
        protocol_sha256="p" * 64,
        artifact_sha256="a" * 64,
        source_role=FailureSourceRole.GATE_HOLDOUT,
        prompt=prompt,
        expected="hidden answer",
        prediction="wrong",
        score=0.0,
        failure_kind="answer_mismatch",
    )
    cluster = cluster_failures((failure,))[0]
    plan = RepairPlan(
        plan_id=(experiment_id + "r" * 64)[:64],
        cluster_id=cluster.cluster_id,
        observation="failure",
        suspected_cause="weakness",
        intervention="independent repair",
        source_failure_ids=(failure.failure_id,),
        direct_training_allowed=False,
        requires_independent_source=True,
    )
    evaluation = EvaluationOutcome(
        run_id=f"eval-{experiment_id}",
        experiment_id=experiment_id,
        source_artifact_ref=f"artifact-{experiment_id}",
        metrics={"quality": 0.5 + score},
        gpu_hours=0.01,
        evidence={"protocol_sha256": "p" * 64},
    )
    result = ExperimentResult(
        experiment_id,
        {"quality": 0.5 + score},
        0.1,
        artifact_ref=f"artifact-{experiment_id}",
    )
    candidate = CandidateCycleOutcome(
        experiment_id=experiment_id,
        evaluation=evaluation,
        result=result,
        harvested_failures=(failure,),
        repair_plans=(plan,),
    )
    ranked = RankedCandidate(
        result=result,
        decision=GateDecision(
            False,
            score,
            {},
            ("quality",),
            (),
            False,
            "rejected",
        ),
        efficiency=score / 0.1,
    )
    return candidate, ranked


def _generation(experiment_id: str, *, score: float, prompt: str):
    candidate, ranked = _candidate(experiment_id, score=score, prompt=prompt)
    return GenerationOutcome((candidate,), (ranked,), None)


def _runner(tmp_path):
    registry = RunRegistry(tmp_path / "runs.db")
    engine = EvolutionEngine(
        Goal((MetricTarget("quality", minimum=0.8),), gpu_hour_budget=5.0),
        ExperimentResult("baseline", {"quality": 0.5}, 0.0),
    )
    runner = ExperimentCycleRunner(
        engine=engine,
        trainer=object(),
        evaluator=object(),
        context=ExecutionContext(
            HardwareProfile(16, 64, 500, 12, 40, 3), str(tmp_path), 7
        ),
        registry=registry,
    )
    return runner, registry


def test_trace_store_commits_hop_and_checkpoint_together(tmp_path):
    path = tmp_path / "trace.db"
    with RecursiveRepairTraceStore(path) as store:
        store.begin(
            session_id="session",
            policy={"max_depth": 3},
            metadata={"provider": "test"},
            initial_candidate_ids=("source",),
            state={"depth_completed": 0},
        )
        store.record_hop(
            session_id="session",
            depth=1,
            target_experiment_id="source",
            failure_signature="f" * 64,
            target_score=-0.1,
            score_improvement=None,
            remaining_budget_after=4.0,
            produced_candidate_ids=("repair-1",),
            promoted_experiment_id=None,
            state={"depth_completed": 1, "current_candidate_ids": ["repair-1"]},
        )
        session = store.get_session("session")
        assert session is not None
        assert session["status"] == "running"
        assert session["state"]["depth_completed"] == 1
        assert store.list_hops("session")[0]["produced_candidate_ids"] == (
            "repair-1",
        )

        store.finish(
            session_id="session",
            stop_reason="max_depth",
            stop_detail="done",
            state={"depth_completed": 1},
        )
        completed = store.get_session("session")
        assert completed is not None
        assert completed["status"] == "completed"
        assert completed["stop_reason"] == "max_depth"


def test_duplicate_hop_rolls_back_checkpoint_update(tmp_path):
    with RecursiveRepairTraceStore(tmp_path / "trace.db") as store:
        store.begin(
            session_id="session",
            policy={},
            metadata={},
            initial_candidate_ids=("source",),
            state={"checkpoint": 0},
        )
        store.record_hop(
            session_id="session",
            depth=1,
            target_experiment_id="source",
            failure_signature="a" * 64,
            target_score=0.0,
            score_improvement=None,
            remaining_budget_after=1.0,
            produced_candidate_ids=("repair-a",),
            promoted_experiment_id=None,
            state={"checkpoint": 1},
        )
        with pytest.raises(sqlite3.IntegrityError):
            store.record_hop(
                session_id="session",
                depth=1,
                target_experiment_id="other",
                failure_signature="b" * 64,
                target_score=0.1,
                score_improvement=0.1,
                remaining_budget_after=0.8,
                produced_candidate_ids=("repair-b",),
                promoted_experiment_id=None,
                state={"checkpoint": 999},
            )
        session = store.get_session("session")
        assert session is not None
        assert session["state"] == {"checkpoint": 1}
        assert len(store.list_hops("session")) == 1


def test_trace_store_marks_failed_session_and_rejects_more_hops(tmp_path):
    with RecursiveRepairTraceStore(tmp_path / "trace.db") as store:
        store.begin(
            session_id="session",
            policy={},
            metadata={},
            initial_candidate_ids=(),
            state={"depth_completed": 0},
        )
        store.fail(
            session_id="session",
            error_detail="RuntimeError: boom",
            state={"depth_completed": 0},
        )
        session = store.get_session("session")
        assert session is not None
        assert session["status"] == "failed"
        assert session["stop_reason"] == "error"
        with pytest.raises(ValueError, match="completed recursive repair session"):
            store.record_hop(
                session_id="session",
                depth=1,
                target_experiment_id="source",
                failure_signature="f" * 64,
                target_score=0.0,
                score_improvement=None,
                remaining_budget_after=1.0,
                produced_candidate_ids=("repair",),
                promoted_experiment_id=None,
                state={},
            )


def test_trace_store_rejects_nonfinite_json_checkpoint(tmp_path):
    with RecursiveRepairTraceStore(tmp_path / "trace.db") as store:
        with pytest.raises(ValueError):
            store.begin(
                session_id="bad",
                policy={},
                metadata={},
                initial_candidate_ids=(),
                state={"bad": float("inf")},
            )


def test_recursive_controller_persists_completed_hop_and_stop(tmp_path, monkeypatch):
    runner, registry = _runner(tmp_path)
    source = _generation("source", score=-0.2, prompt="failure one")
    second = _generation("repair-1", score=-0.1, prompt="failure two")

    def fake_hop(*, runner, source_generation, provider, variants, candidate_id=None, replay_ratio=1.0):
        target = _repairable_target(source_generation, candidate_id=candidate_id)
        return AutonomousRepairOutcome(
            source_generation=source_generation,
            target=target,
            population=None,
            repair_generation=second,
        )

    monkeypatch.setattr(recursive, "run_single_hop_autonomous_repair", fake_hop)
    outcome = recursive.run_bounded_autonomous_repair(
        runner=runner,
        source_generation=source,
        provider=type("Provider", (), {"name": "test", "version": "1"})(),
        variants=(RepairVariant("default", 0.1),),
        policy=recursive.RecursiveRepairPolicy(max_depth=1),
    )
    assert outcome.session_id is not None
    with RecursiveRepairTraceStore(registry.path) as store:
        session = store.get_session(outcome.session_id)
        assert session is not None
        assert session["status"] == "completed"
        assert session["stop_reason"] == "max_depth"
        assert session["state"]["depth_completed"] == 1
        assert session["state"]["current_candidate_ids"] == ["repair-1"]
        hops = store.list_hops(outcome.session_id)
        assert len(hops) == 1
        assert hops[0]["target_experiment_id"] == "source"
        assert hops[0]["produced_candidate_ids"] == ("repair-1",)
    registry.close()


def test_recursive_controller_marks_trace_failed_on_unexpected_error(tmp_path, monkeypatch):
    runner, registry = _runner(tmp_path)
    source = _generation("source", score=-0.2, prompt="failure one")

    def explode(**kwargs):
        raise RuntimeError("synthetic crash")

    monkeypatch.setattr(recursive, "run_single_hop_autonomous_repair", explode)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        recursive.run_bounded_autonomous_repair(
            runner=runner,
            source_generation=source,
            provider=type("Provider", (), {"name": "test", "version": "1"})(),
            variants=(RepairVariant("default", 0.1),),
        )

    connection = sqlite3.connect(registry.path)
    row = connection.execute(
        "SELECT session_id, status, stop_reason, stop_detail FROM recursive_repair_sessions"
    ).fetchone()
    connection.close()
    assert row is not None
    _, status, stop_reason, stop_detail = row
    assert status == "failed"
    assert stop_reason == "error"
    assert "synthetic crash" in stop_detail
    registry.close()
