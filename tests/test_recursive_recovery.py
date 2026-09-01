from __future__ import annotations

import json

from chowder.executors import EvaluationOutcome, TrainingArtifact
from chowder.failures import FailureRecord, FailureSourceRole, RepairPlan
from chowder.models import Experiment, ExperimentResult, Hypothesis
from chowder.recursive_recovery import (
    RecoveryDisposition,
    analyze_recursive_repair_session,
    list_interrupted_recursive_sessions,
)
from chowder.recursive_trace import RecursiveRepairTraceStore
from chowder.registry import RunRegistry


def _engine_snapshot():
    return {
        "goal": {
            "metrics": [
                {
                    "name": "quality",
                    "minimum": 0.8,
                    "maximum": None,
                    "weight": 1.0,
                    "regression_tolerance": 0.0,
                    "direction": "maximize",
                }
            ],
            "gpu_hour_budget": 5.0,
            "max_parallel_candidates": 2,
            "minimum_promotion_gain": 0.0,
            "require_protocol_match": False,
        },
        "baseline": {
            "experiment_id": "baseline",
            "metrics": {"quality": 0.5},
            "gpu_hours": 0.0,
            "artifact_ref": None,
            "evidence": {},
        },
    }


def _experiment(experiment_id: str, parent_id: str | None = None):
    return Experiment(
        experiment_id=experiment_id,
        parent_id=parent_id,
        hypothesis=Hypothesis("obs", "cause", "intervention"),
        config_patch={"backend": {"dataset": "train.jsonl"}},
        estimated_gpu_hours=0.2,
    )


def _record_complete_candidate(registry: RunRegistry, experiment_id: str):
    registry.record_training_artifact(
        TrainingArtifact(
            run_id=f"train-{experiment_id}",
            experiment_id=experiment_id,
            artifact_ref=f"adapter-{experiment_id}",
            gpu_hours=0.1,
            evidence={"dataset_sha256": "d" * 64, "artifact_sha256": "a" * 64},
        )
    )
    registry.record_evaluation_outcome(
        EvaluationOutcome(
            run_id=f"eval-{experiment_id}",
            experiment_id=experiment_id,
            source_artifact_ref=f"adapter-{experiment_id}",
            metrics={"quality": 0.6},
            gpu_hours=0.01,
            evidence={"protocol_sha256": "p" * 64},
        )
    )
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
        prompt="hidden prompt",
        expected="hidden answer",
        prediction="wrong",
        score=0.0,
        failure_kind="answer_mismatch",
    )
    registry.record_failure(failure)
    registry.record_repair_plan(
        RepairPlan(
            plan_id=(experiment_id + "r" * 64)[:64],
            cluster_id="c" * 64,
            observation="failure",
            suspected_cause="weakness",
            intervention="independent repair",
            source_failure_ids=(failure.failure_id,),
            direct_training_allowed=False,
            requires_independent_source=True,
        )
    )
    registry.record_result(
        ExperimentResult(
            experiment_id=experiment_id,
            metrics={"quality": 0.6},
            gpu_hours=0.11,
            artifact_ref=f"adapter-{experiment_id}",
            evidence={
                "diagnostics": {
                    "failure_count": 1,
                    "repair_plan_count": 1,
                    "error": None,
                }
            },
        )
    )


def _begin_running(
    registry: RunRegistry,
    *,
    session_id: str = "session",
    current: tuple[str, ...] = ("source",),
    metadata=None,
    depth: int = 0,
    signature_counts=None,
    previous_target_score=None,
    remaining_budget: float = 4.0,
    promoted=None,
):
    with RecursiveRepairTraceStore(registry.path) as store:
        store.begin(
            session_id=session_id,
            policy={
                "max_depth": 3,
                "min_score_improvement": 1e-4,
                "max_failure_signature_occurrences": 1,
                "replay_ratio": 1.0,
            },
            metadata=(
                {
                    "provider_name": "test",
                    "provider_version": "1",
                    "engine_snapshot": _engine_snapshot(),
                }
                if metadata is None
                else metadata
            ),
            initial_candidate_ids=current,
            state={
                "depth_completed": depth,
                "current_candidate_ids": list(current),
                "signature_counts": signature_counts or {},
                "previous_target_score": previous_target_score,
                "remaining_budget": remaining_budget,
                "promoted_experiment_id": promoted,
            },
        )


def test_missing_session_is_reported(tmp_path):
    with RunRegistry(tmp_path / "runs.db") as registry:
        report = analyze_recursive_repair_session(registry, "missing")
        assert report.disposition is RecoveryDisposition.SESSION_NOT_FOUND


def test_completed_session_is_not_resumable(tmp_path):
    with RunRegistry(tmp_path / "runs.db") as registry:
        _begin_running(registry)
        with RecursiveRepairTraceStore(registry.path) as store:
            store.finish(
                session_id="session",
                stop_reason="max_depth",
                stop_detail="done",
                state={
                    "depth_completed": 0,
                    "current_candidate_ids": ["source"],
                    "signature_counts": {},
                    "previous_target_score": None,
                    "remaining_budget": 4.0,
                    "promoted_experiment_id": None,
                },
            )
        report = analyze_recursive_repair_session(registry, "session")
        assert report.disposition is RecoveryDisposition.ALREADY_TERMINAL


def test_legacy_running_session_without_engine_snapshot_is_blocked(tmp_path):
    with RunRegistry(tmp_path / "runs.db") as registry:
        _begin_running(
            registry,
            metadata={"provider_name": "legacy", "baseline_experiment_id": "baseline"},
        )
        report = analyze_recursive_repair_session(registry, "session")
        assert report.disposition is RecoveryDisposition.MISSING_ENGINE_SNAPSHOT


def test_checkpoint_depth_mismatch_is_blocked(tmp_path):
    with RunRegistry(tmp_path / "runs.db") as registry:
        _begin_running(registry, depth=1)
        report = analyze_recursive_repair_session(registry, "session")
        assert report.disposition is RecoveryDisposition.CHECKPOINT_INCONSISTENT
        assert "depth" in report.detail


def test_promoted_running_checkpoint_is_terminal_pending(tmp_path):
    with RunRegistry(tmp_path / "runs.db") as registry:
        _begin_running(registry, promoted="winner")
        report = analyze_recursive_repair_session(registry, "session")
        assert report.disposition is RecoveryDisposition.TERMINAL_PENDING


def test_missing_current_experiment_is_blocked(tmp_path):
    with RunRegistry(tmp_path / "runs.db") as registry:
        _begin_running(registry)
        report = analyze_recursive_repair_session(registry, "session")
        assert report.disposition is RecoveryDisposition.MISSING_REGISTRY_EVIDENCE
        assert report.missing_evidence == ("experiment:source",)


def test_missing_training_evidence_is_blocked(tmp_path):
    with RunRegistry(tmp_path / "runs.db") as registry:
        registry.record_experiment(_experiment("source"))
        _begin_running(registry)
        report = analyze_recursive_repair_session(registry, "session")
        assert report.disposition is RecoveryDisposition.MISSING_REGISTRY_EVIDENCE
        assert "training:source" in report.missing_evidence
        assert "evaluation:source" in report.missing_evidence
        assert "result:source" in report.missing_evidence


def test_multiple_training_runs_are_ambiguous(tmp_path):
    with RunRegistry(tmp_path / "runs.db") as registry:
        registry.record_experiment(_experiment("source"))
        _record_complete_candidate(registry, "source")
        registry.record_training_artifact(
            TrainingArtifact(
                run_id="train-source-retry",
                experiment_id="source",
                artifact_ref="adapter-source-retry",
                gpu_hours=0.1,
            )
        )
        _begin_running(registry)
        report = analyze_recursive_repair_session(registry, "session")
        assert report.disposition is RecoveryDisposition.AMBIGUOUS_REGISTRY_EVIDENCE
        assert any(value.startswith("training:source:2") for value in report.missing_evidence)


def test_orphaned_child_progress_is_detected_before_retraining(tmp_path):
    with RunRegistry(tmp_path / "runs.db") as registry:
        registry.record_experiment(_experiment("source"))
        _record_complete_candidate(registry, "source")
        registry.record_experiment(_experiment("repair-orphan", parent_id="source"))
        _begin_running(registry)
        report = analyze_recursive_repair_session(registry, "session")
        assert report.disposition is RecoveryDisposition.ORPHANED_PROGRESS
        assert report.orphaned_child_ids == ("repair-orphan",)


def test_clean_checkpoint_is_resumable(tmp_path):
    with RunRegistry(tmp_path / "runs.db") as registry:
        registry.record_experiment(_experiment("source"))
        _record_complete_candidate(registry, "source")
        _begin_running(registry)
        report = analyze_recursive_repair_session(registry, "session")
        assert report.disposition is RecoveryDisposition.RESUMABLE
        assert report.resumable
        assert report.current_candidate_ids == ("source",)
        assert report.engine_snapshot == _engine_snapshot()


def test_interrupted_session_listing_returns_only_running(tmp_path):
    with RunRegistry(tmp_path / "runs.db") as registry:
        _begin_running(registry, session_id="running")
        _begin_running(registry, session_id="finished")
        with RecursiveRepairTraceStore(registry.path) as store:
            store.finish(
                session_id="finished",
                stop_reason="max_depth",
                stop_detail="done",
                state={
                    "depth_completed": 0,
                    "current_candidate_ids": ["source"],
                    "signature_counts": {},
                    "previous_target_score": None,
                    "remaining_budget": 4.0,
                    "promoted_experiment_id": None,
                },
            )
        assert list_interrupted_recursive_sessions(registry.path) == ("running",)


def test_new_controller_session_persists_goal_and_baseline_snapshot(tmp_path, monkeypatch):
    import chowder.recursive_repair as recursive
    from chowder.autonomous_repair import AutonomousRepairOutcome, _repairable_target
    from chowder.cycle import CandidateCycleOutcome, ExperimentCycleRunner, GenerationOutcome
    from chowder.engine import EvolutionEngine
    from chowder.memory import HardwareProfile
    from chowder.models import GateDecision, Goal, MetricTarget
    from chowder.tournament import RankedCandidate

    registry = RunRegistry(tmp_path / "controller.db")
    engine = EvolutionEngine(
        Goal((MetricTarget("quality", minimum=0.8),), gpu_hour_budget=5.0),
        ExperimentResult("baseline", {"quality": 0.5}, 0.0),
    )
    runner = ExperimentCycleRunner(
        engine=engine,
        trainer=object(),
        evaluator=object(),
        context=__import__("chowder.executors", fromlist=["ExecutionContext"]).ExecutionContext(
            HardwareProfile(16, 64, 500, 12, 40, 3), str(tmp_path), 7
        ),
        registry=registry,
    )

    failure = FailureRecord(
        failure_id="f" * 64,
        experiment_id="source",
        evaluation_run_id="eval-source",
        evaluator="transformers-text",
        suite="reasoning",
        row_index=0,
        protocol_sha256="p" * 64,
        artifact_sha256="a" * 64,
        source_role=FailureSourceRole.GATE_HOLDOUT,
        prompt="hidden",
        expected="answer",
        prediction="wrong",
        score=0.0,
        failure_kind="answer_mismatch",
    )
    plan = RepairPlan(
        plan_id="r" * 64,
        cluster_id=__import__("chowder.failures", fromlist=["cluster_failures"]).cluster_failures((failure,))[0].cluster_id,
        observation="failure",
        suspected_cause="weakness",
        intervention="repair",
        source_failure_ids=(failure.failure_id,),
        direct_training_allowed=False,
        requires_independent_source=True,
    )
    result = ExperimentResult("source", {"quality": 0.4}, 0.1)
    evaluation = EvaluationOutcome(
        "eval-source", "source", "adapter-source", {"quality": 0.4}, 0.01, {"protocol_sha256": "p" * 64}
    )
    candidate = CandidateCycleOutcome(
        "source", evaluation=evaluation, result=result, harvested_failures=(failure,), repair_plans=(plan,)
    )
    ranked = RankedCandidate(
        result,
        GateDecision(False, -0.1, {}, ("quality",), (), False, "rejected"),
        -1.0,
    )
    source_generation = GenerationOutcome((candidate,), (ranked,), None)
    next_generation = GenerationOutcome((), (), ExperimentResult("winner", {"quality": 0.9}, 0.1))

    def fake_hop(*, runner, source_generation, provider, variants, candidate_id=None, replay_ratio=1.0):
        return AutonomousRepairOutcome(
            source_generation,
            _repairable_target(source_generation, candidate_id=candidate_id),
            None,
            next_generation,
        )

    monkeypatch.setattr(recursive, "run_single_hop_autonomous_repair", fake_hop)
    outcome = recursive.run_bounded_autonomous_repair(
        runner=runner,
        source_generation=source_generation,
        provider=type("Provider", (), {"name": "test", "version": "1"})(),
        variants=(__import__("chowder.repair_candidates", fromlist=["RepairVariant"]).RepairVariant("default", 0.1),),
    )
    with RecursiveRepairTraceStore(registry.path) as store:
        session = store.get_session(outcome.session_id)
        assert session is not None
        snapshot = session["metadata"]["engine_snapshot"]
        assert snapshot["baseline"]["metrics"] == {"quality": 0.5}
        assert snapshot["goal"]["gpu_hour_budget"] == 5.0
        assert snapshot["goal"]["metrics"][0]["direction"] == "maximize"
    registry.close()
