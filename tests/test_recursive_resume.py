from __future__ import annotations

import pytest

import chowder.recursive_repair as recursive
from chowder.autonomous_repair import AutonomousRepairOutcome, _repairable_target
from chowder.cycle import CandidateCycleOutcome, ExperimentCycleRunner, GenerationOutcome
from chowder.engine import EvolutionEngine
from chowder.executors import EvaluationOutcome, ExecutionContext, TrainingArtifact
from chowder.failures import FailureRecord, FailureSourceRole, RepairPlan
from chowder.memory import HardwareProfile
from chowder.models import (
    Experiment,
    ExperimentResult,
    GateDecision,
    Goal,
    Hypothesis,
    MetricTarget,
)
from chowder.recursive_repair import (
    RecursiveRepairPolicy,
    _engine_snapshot,
    _execution_snapshot,
    _policy_payload,
    _variant_metadata,
)
from chowder.recursive_resume import RecursiveResumeError, resume_recursive_repair_session
from chowder.recursive_trace import RecursiveRepairTraceStore
from chowder.registry import RunRegistry
from chowder.repair_candidates import RepairVariant
from chowder.tournament import RankedCandidate


class _NamedExecutor:
    def __init__(self, name: str):
        self.name = name


class _Provider:
    name = "provider"
    version = "1"


POLICY = RecursiveRepairPolicy(max_depth=3)
VARIANTS = (RepairVariant("default", 0.1),)


def _runner(tmp_path, registry: RunRegistry, *, seed: int = 7):
    engine = EvolutionEngine(
        Goal((MetricTarget("quality", minimum=0.8),), gpu_hour_budget=5.0),
        ExperimentResult("baseline", {"quality": 0.5}, 0.0),
    )
    return ExperimentCycleRunner(
        engine=engine,
        trainer=_NamedExecutor("trainer"),
        evaluator=_NamedExecutor("evaluator"),
        context=ExecutionContext(
            HardwareProfile(16, 64, 500, 12, 40, 3), str(tmp_path), seed
        ),
        base_config={"research": {"tag": "resume-test"}},
        registry=registry,
    )


def _source_experiment():
    return Experiment(
        experiment_id="source",
        parent_id=None,
        hypothesis=Hypothesis("obs", "cause", "intervention"),
        config_patch={},
        estimated_gpu_hours=0.2,
    )


def _record_source(registry: RunRegistry):
    experiment = _source_experiment()
    registry.record_experiment(experiment)
    artifact = TrainingArtifact(
        run_id="train-source",
        experiment_id="source",
        artifact_ref="adapter-source",
        gpu_hours=0.1,
        evidence={"dataset_sha256": "d" * 64, "artifact_sha256": "a" * 64},
    )
    evaluation = EvaluationOutcome(
        run_id="eval-source",
        experiment_id="source",
        source_artifact_ref=artifact.artifact_ref,
        metrics={"quality": 0.4},
        gpu_hours=0.01,
        evidence={"protocol_sha256": "p" * 64},
    )
    failure = FailureRecord(
        failure_id="f" * 64,
        experiment_id="source",
        evaluation_run_id=evaluation.run_id,
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
    plan = RepairPlan(
        plan_id="r" * 64,
        cluster_id="c" * 64,
        observation="failure",
        suspected_cause="weakness",
        intervention="independent repair",
        source_failure_ids=(failure.failure_id,),
        direct_training_allowed=False,
        requires_independent_source=True,
    )
    result = ExperimentResult(
        "source",
        {"quality": 0.4},
        0.11,
        artifact_ref=artifact.artifact_ref,
        evidence={
            "training_run_id": artifact.run_id,
            "evaluation_run_id": evaluation.run_id,
            "evaluation_protocol_sha256": "p" * 64,
            "compute": {
                "training_gpu_hours": 0.1,
                "evaluation_gpu_hours": 0.01,
                "total_gpu_hours": 0.11,
            },
            "diagnostics": {
                "failure_count": 1,
                "repair_plan_count": 1,
                "error": None,
            },
        },
    )
    registry.record_training_artifact(artifact)
    registry.record_evaluation_outcome(evaluation)
    registry.record_failure(failure)
    registry.record_repair_plan(plan)
    registry.record_result(result)
    registry.update_experiment_status("source", "rejected")


def _begin_interrupted(runner: ExperimentCycleRunner, *, session_id="resume-session"):
    assert runner.registry is not None
    with RecursiveRepairTraceStore(runner.registry.path) as store:
        store.begin(
            session_id=session_id,
            policy=_policy_payload(POLICY),
            metadata={
                "provider_name": _Provider.name,
                "provider_version": _Provider.version,
                "variants": _variant_metadata(VARIANTS),
                "baseline_experiment_id": runner.engine.baseline.experiment_id,
                "engine_snapshot": _engine_snapshot(runner),
                "execution_snapshot": _execution_snapshot(runner),
            },
            initial_candidate_ids=("source",),
            state={
                "depth_completed": 0,
                "current_candidate_ids": ["source"],
                "signature_counts": {},
                "previous_target_score": None,
                "remaining_budget": 4.89,
                "promoted_experiment_id": None,
            },
        )
    return session_id


def test_recovery_claim_is_single_owner_and_fences_stale_writer(tmp_path):
    path = tmp_path / "runs.db"
    with RunRegistry(path) as registry:
        runner = _runner(tmp_path, registry)
        _record_source(registry)
        session_id = _begin_interrupted(runner)

        with RecursiveRepairTraceStore(path) as owner, RecursiveRepairTraceStore(path) as stale:
            owner.claim_recovery(session_id=session_id, claim_token="owner")
            with pytest.raises(ValueError, match="already has a recovery claim"):
                stale.claim_recovery(session_id=session_id, claim_token="other")
            with pytest.raises(ValueError, match="fenced"):
                stale.record_hop(
                    session_id=session_id,
                    depth=1,
                    target_experiment_id="source",
                    failure_signature="s" * 64,
                    target_score=-0.1,
                    score_improvement=None,
                    remaining_budget_after=4.0,
                    produced_candidate_ids=("repair",),
                    promoted_experiment_id=None,
                    state={},
                )
            owner.release_recovery_claim(session_id=session_id, claim_token="owner")


def test_resume_rejects_execution_environment_drift_without_leaking_claim(tmp_path):
    path = tmp_path / "runs.db"
    with RunRegistry(path) as registry:
        original = _runner(tmp_path, registry, seed=7)
        _record_source(registry)
        session_id = _begin_interrupted(original)

        changed = _runner(tmp_path, registry, seed=8)
        with pytest.raises(RecursiveResumeError, match="executor/seed/base-config/hardware"):
            resume_recursive_repair_session(
                runner=changed,
                session_id=session_id,
                provider=_Provider(),
                variants=VARIANTS,
                policy=POLICY,
            )
        with RecursiveRepairTraceStore(path) as store:
            assert store.get_recovery_claim(session_id) is None
            assert store.get_session(session_id)["status"] == "running"


def test_clean_resume_restores_budget_and_continues_from_checkpoint(monkeypatch, tmp_path):
    path = tmp_path / "runs.db"
    with RunRegistry(path) as registry:
        setup_runner = _runner(tmp_path, registry)
        _record_source(registry)
        session_id = _begin_interrupted(setup_runner)

        runner = _runner(tmp_path, registry)
        calls: list[str] = []

        winner = ExperimentResult("repair-winner", {"quality": 0.9}, 0.1)
        winner_candidate = CandidateCycleOutcome("repair-winner", result=winner)
        winner_ranked = RankedCandidate(
            result=winner,
            decision=GateDecision(
                accepted=True,
                score=0.4,
                regressions={},
                unmet_targets=(),
                missing_metrics=(),
                goal_met=True,
                reason="accepted",
            ),
            efficiency=4.0,
        )
        promoted_generation = GenerationOutcome(
            candidates=(winner_candidate,),
            ranking=(winner_ranked,),
            promoted=winner,
        )

        def fake_hop(
            *, runner, source_generation, provider, variants,
            candidate_id=None, replay_ratio=1.0,
        ):
            calls.append(candidate_id)
            return AutonomousRepairOutcome(
                source_generation=source_generation,
                target=_repairable_target(
                    source_generation, candidate_id=candidate_id
                ),
                population=None,
                repair_generation=promoted_generation,
            )

        monkeypatch.setattr(recursive, "run_single_hop_autonomous_repair", fake_hop)
        outcome = resume_recursive_repair_session(
            runner=runner,
            session_id=session_id,
            provider=_Provider(),
            variants=VARIANTS,
            policy=POLICY,
        )

        assert calls == ["source"]
        assert outcome.starting_depth == 0
        assert outcome.depth == 1
        assert outcome.promoted is winner
        assert runner.engine.spent_gpu_hours == pytest.approx(0.11)
        with RecursiveRepairTraceStore(path) as store:
            assert store.get_recovery_claim(session_id) is None
            session = store.get_session(session_id)
            assert session is not None
            assert session["status"] == "completed"
            assert len(store.list_hops(session_id)) == 1
