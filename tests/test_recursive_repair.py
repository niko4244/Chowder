from __future__ import annotations

from dataclasses import replace

import pytest

import chowder.recursive_repair as recursive
from chowder.autonomous_repair import AutonomousRepairOutcome, _repairable_target
from chowder.cancellation import CancellationToken
from chowder.cycle import CandidateCycleOutcome, ExperimentCycleRunner, GenerationOutcome
from chowder.engine import EvolutionEngine
from chowder.executors import EvaluationOutcome, ExecutionContext
from chowder.failures import (
    FailureRecord,
    FailureSourceRole,
    RepairPlan,
    cluster_failures,
)
from chowder.memory import HardwareProfile
from chowder.models import (
    ExperimentResult,
    GateDecision,
    Goal,
    MetricTarget,
)
from chowder.repair_candidates import RepairVariant
from chowder.tournament import RankedCandidate


def _runner(*, budget=5.0):
    engine = EvolutionEngine(
        Goal((MetricTarget("quality", minimum=0.8),), gpu_hour_budget=budget),
        ExperimentResult("baseline", {"quality": 0.5}, 0.0),
    )
    return ExperimentCycleRunner(
        engine=engine,
        trainer=object(),
        evaluator=object(),
        context=ExecutionContext(
            HardwareProfile(16, 64, 500, 12, 40, 3), ".", 7
        ),
    )


def _candidate(
    experiment_id: str,
    *,
    score: float,
    prompt: str = "hidden prompt",
    expected: str = "hidden answer",
    failure_kind: str = "answer_mismatch",
    row_index: int = 0,
):
    failure_id = (experiment_id.replace("-", "") + "f" * 64)[:64]
    failure = FailureRecord(
        failure_id=failure_id,
        experiment_id=experiment_id,
        evaluation_run_id=f"eval-{experiment_id}",
        evaluator="transformers-text",
        suite="reasoning",
        row_index=row_index,
        protocol_sha256="p" * 64,
        artifact_sha256="a" * 64,
        source_role=FailureSourceRole.GATE_HOLDOUT,
        prompt=prompt,
        expected=expected,
        prediction="wrong",
        score=0.0,
        failure_kind=failure_kind,
    )
    cluster = cluster_failures((failure,))[0]
    plan = RepairPlan(
        plan_id=(experiment_id.replace("-", "") + "r" * 64)[:64],
        cluster_id=cluster.cluster_id,
        observation="one failure",
        suspected_cause="remaining weakness",
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
    outcome = CandidateCycleOutcome(
        experiment_id=experiment_id,
        evaluation=evaluation,
        result=result,
        harvested_failures=(failure,),
        repair_plans=(plan,),
    )
    ranked = RankedCandidate(
        result=result,
        decision=GateDecision(
            accepted=False,
            score=score,
            regressions={},
            unmet_targets=("quality",),
            missing_metrics=(),
            goal_met=False,
            reason="rejected",
        ),
        efficiency=score / 0.1,
    )
    return outcome, ranked


def _generation(*rows, promoted=None):
    candidates = tuple(row[0] for row in rows)
    ranking = tuple(row[1] for row in rows)
    return GenerationOutcome(candidates, ranking, promoted)


def _fake_hop_for(next_generations, calls):
    queue = list(next_generations)

    def fake_hop(*, runner, source_generation, provider, variants, candidate_id=None, replay_ratio=1.0):
        calls.append(candidate_id)
        target = _repairable_target(source_generation, candidate_id=candidate_id)
        next_generation = queue.pop(0)
        return AutonomousRepairOutcome(
            source_generation=source_generation,
            target=target,
            population=None,
            repair_generation=next_generation,
        )

    return fake_hop


def test_failure_signature_is_stable_across_experiment_and_run_identity():
    a = _generation(_candidate("a", score=-0.1))
    b = _generation(_candidate("b", score=-0.05))
    target_a = _repairable_target(a)
    target_b = _repairable_target(b)
    assert recursive.failure_signature(target_a) == recursive.failure_signature(target_b)


def test_failure_signature_changes_when_hidden_failure_row_changes():
    a = _generation(_candidate("a", score=-0.1, prompt="prompt a"))
    b = _generation(_candidate("b", score=-0.05, prompt="prompt b"))
    assert recursive.failure_signature(_repairable_target(a)) != recursive.failure_signature(
        _repairable_target(b)
    )


def test_recursive_policy_validates_bounds():
    with pytest.raises(ValueError, match="max_depth"):
        recursive.RecursiveRepairPolicy(max_depth=0)
    with pytest.raises(ValueError, match="min_score_improvement"):
        recursive.RecursiveRepairPolicy(min_score_improvement=-1)
    with pytest.raises(ValueError, match="occurrences"):
        recursive.RecursiveRepairPolicy(max_failure_signature_occurrences=0)
    with pytest.raises(ValueError, match="replay_ratio"):
        recursive.RecursiveRepairPolicy(replay_ratio=0)


def test_already_promoted_source_stops_without_repair(monkeypatch):
    promoted = ExperimentResult("winner", {"quality": 0.9}, 0.1)
    generation = _generation(promoted=promoted)

    def should_not_run(**kwargs):
        raise AssertionError("repair must not run for a promoted source generation")

    monkeypatch.setattr(recursive, "run_single_hop_autonomous_repair", should_not_run)
    outcome = recursive.run_bounded_autonomous_repair(
        runner=_runner(),
        source_generation=generation,
        provider=object(),
        variants=(RepairVariant("default", 0.1),),
    )
    assert outcome.stop_reason is recursive.RecursiveRepairStopReason.PROMOTED
    assert outcome.depth == 0
    assert outcome.promoted is promoted


def test_no_repairable_diagnostic_stops_cleanly():
    result = ExperimentResult("x", {"quality": 0.4}, 0.1)
    ranked = RankedCandidate(
        result,
        GateDecision(False, -0.1, {}, ("quality",), (), False, "rejected"),
        -1.0,
    )
    generation = GenerationOutcome(
        (CandidateCycleOutcome("x", result=result),), (ranked,), None
    )
    outcome = recursive.run_bounded_autonomous_repair(
        runner=_runner(),
        source_generation=generation,
        provider=object(),
        variants=(RepairVariant("default", 0.1),),
    )
    assert outcome.stop_reason is recursive.RecursiveRepairStopReason.NO_REPAIRABLE_DIAGNOSTIC
    assert outcome.depth == 0


def test_repeated_failure_signature_stops_after_one_hop(monkeypatch):
    first = _generation(_candidate("source", score=-0.1))
    same_failure = _generation(_candidate("repair-1", score=-0.05))
    calls = []
    monkeypatch.setattr(
        recursive,
        "run_single_hop_autonomous_repair",
        _fake_hop_for((same_failure,), calls),
    )
    outcome = recursive.run_bounded_autonomous_repair(
        runner=_runner(),
        source_generation=first,
        provider=object(),
        variants=(RepairVariant("default", 0.1),),
        policy=recursive.RecursiveRepairPolicy(max_depth=5),
    )
    assert calls == ["source"]
    assert outcome.depth == 1
    assert outcome.stop_reason is recursive.RecursiveRepairStopReason.REPEATED_FAILURE


def test_no_progress_rejects_score_noise_before_second_hop(monkeypatch):
    first = _generation(_candidate("source", score=-0.1, prompt="failure one"))
    second = _generation(_candidate("repair-1", score=-0.09995, prompt="failure two"))
    calls = []
    monkeypatch.setattr(
        recursive,
        "run_single_hop_autonomous_repair",
        _fake_hop_for((second,), calls),
    )
    outcome = recursive.run_bounded_autonomous_repair(
        runner=_runner(),
        source_generation=first,
        provider=object(),
        variants=(RepairVariant("default", 0.1),),
        policy=recursive.RecursiveRepairPolicy(
            max_depth=5, min_score_improvement=1e-3
        ),
    )
    assert calls == ["source"]
    assert outcome.depth == 1
    assert outcome.stop_reason is recursive.RecursiveRepairStopReason.NO_PROGRESS


def test_promoted_repair_stops_immediately(monkeypatch):
    first = _generation(_candidate("source", score=-0.1, prompt="failure one"))
    winner = ExperimentResult("repair-winner", {"quality": 0.85}, 0.1)
    promoted_generation = _generation(promoted=winner)
    calls = []
    monkeypatch.setattr(
        recursive,
        "run_single_hop_autonomous_repair",
        _fake_hop_for((promoted_generation,), calls),
    )
    outcome = recursive.run_bounded_autonomous_repair(
        runner=_runner(),
        source_generation=first,
        provider=object(),
        variants=(RepairVariant("default", 0.1),),
    )
    assert outcome.stop_reason is recursive.RecursiveRepairStopReason.PROMOTED
    assert outcome.depth == 1
    assert outcome.promoted is winner


def test_max_depth_is_hard_limit(monkeypatch):
    first = _generation(_candidate("source", score=-0.2, prompt="failure zero"))
    second = _generation(_candidate("repair-1", score=-0.1, prompt="failure one"))
    third = _generation(_candidate("repair-2", score=0.0, prompt="failure two"))
    calls = []
    monkeypatch.setattr(
        recursive,
        "run_single_hop_autonomous_repair",
        _fake_hop_for((second, third), calls),
    )
    outcome = recursive.run_bounded_autonomous_repair(
        runner=_runner(),
        source_generation=first,
        provider=object(),
        variants=(RepairVariant("default", 0.1),),
        policy=recursive.RecursiveRepairPolicy(max_depth=2),
    )
    assert calls == ["source", "repair-1"]
    assert outcome.depth == 2
    assert outcome.stop_reason is recursive.RecursiveRepairStopReason.MAX_DEPTH


def test_budget_exhaustion_stops_before_hop(monkeypatch):
    generation = _generation(_candidate("source", score=-0.1))

    def should_not_run(**kwargs):
        raise AssertionError("repair must not run with no budget")

    monkeypatch.setattr(recursive, "run_single_hop_autonomous_repair", should_not_run)
    outcome = recursive.run_bounded_autonomous_repair(
        runner=_runner(budget=0.0),
        source_generation=generation,
        provider=object(),
        variants=(RepairVariant("default", 0.1),),
    )
    assert outcome.stop_reason is recursive.RecursiveRepairStopReason.BUDGET_EXHAUSTED
    assert outcome.depth == 0


def test_cancellation_requested_before_any_hop_stops_immediately(monkeypatch):
    generation = _generation(_candidate("source", score=-0.1))

    def should_not_run(**kwargs):
        raise AssertionError("repair must not run once cancellation was requested")

    monkeypatch.setattr(recursive, "run_single_hop_autonomous_repair", should_not_run)
    runner = _runner()
    token = CancellationToken()
    token.request()
    runner.cancellation = token
    outcome = recursive.run_bounded_autonomous_repair(
        runner=runner,
        source_generation=generation,
        provider=object(),
        variants=(RepairVariant("default", 0.1),),
    )
    assert outcome.stop_reason is recursive.RecursiveRepairStopReason.CANCELLED
    assert outcome.depth == 0


def test_cancellation_requested_between_hops_stops_with_cancelled_not_a_dead_end(monkeypatch):
    """A cancellation that fires during/after a hop must be reported as
    CANCELLED on the next iteration, not misread as an ordinary dead end
    (NO_REPAIRABLE_DIAGNOSTIC/REPEATED_FAILURE) just because the loop
    happens to check for novel targets right after."""
    token = CancellationToken()
    first = _generation(_candidate("source", score=-0.1, prompt="failure one"))
    second = _generation(_candidate("repair-1", score=0.5, prompt="failure two"))
    calls = []

    def fake_hop(*, runner, source_generation, provider, variants, candidate_id=None, replay_ratio=1.0):
        calls.append(candidate_id)
        token.request()
        target = _repairable_target(source_generation, candidate_id=candidate_id)
        return AutonomousRepairOutcome(
            source_generation=source_generation,
            target=target,
            population=None,
            repair_generation=second,
        )

    monkeypatch.setattr(recursive, "run_single_hop_autonomous_repair", fake_hop)
    runner = _runner()
    runner.cancellation = token
    outcome = recursive.run_bounded_autonomous_repair(
        runner=runner,
        source_generation=first,
        provider=object(),
        variants=(RepairVariant("default", 0.1),),
        policy=recursive.RecursiveRepairPolicy(max_depth=5),
    )
    assert calls == ["source"]  # exactly one hop attempted, not a second
    assert outcome.depth == 1
    assert outcome.stop_reason is recursive.RecursiveRepairStopReason.CANCELLED
    # GPU-hour accounting for the one hop that did run is preserved --
    # cancelling only prevents starting the next hop, it doesn't discard
    # what the completed one already recorded.
    assert outcome.hops[0].target_experiment_id == "source"
    assert outcome.hops[0].outcome.repair_generation is second


def test_cancellation_during_a_repair_hops_own_training_is_reported_as_cancelled(monkeypatch):
    """Distinct from the between-hops case: here the hop function returns
    normally with a rejected, cancelled candidate -- matching what
    _run_candidate actually produces when cancellation interrupts training
    mid-hop (it never raises, just returns error="cancelled: ..." with no
    result) -- rather than a hop that completed cleanly before the token
    was set. Without the fix, _repairable_target already excludes a
    result=None candidate from being repairable, so the next iteration
    would misreport this as a dead end (NO_REPAIRABLE_DIAGNOSTIC) instead
    of the cancellation that actually caused it."""
    token = CancellationToken()
    first = _generation(_candidate("source", score=-0.1, prompt="failure one"))

    cancelled_ranked = RankedCandidate(
        ExperimentResult("repair-1", {"quality": 0.0}, 0.05),
        GateDecision(False, -1.0, {}, ("quality",), (), False, "rejected"),
        -1.0,
    )
    cancelled_candidate = CandidateCycleOutcome(
        "repair-1", error="cancelled: RuntimeError: worker terminated"
    )
    cancelled_generation = GenerationOutcome((cancelled_candidate,), (cancelled_ranked,), None)

    def fake_hop(*, runner, source_generation, provider, variants, candidate_id=None, replay_ratio=1.0):
        token.request()  # the hop's own training was interrupted
        target = _repairable_target(source_generation, candidate_id=candidate_id)
        return AutonomousRepairOutcome(
            source_generation=source_generation,
            target=target,
            population=None,
            repair_generation=cancelled_generation,
        )

    monkeypatch.setattr(recursive, "run_single_hop_autonomous_repair", fake_hop)
    runner = _runner()
    runner.cancellation = token
    outcome = recursive.run_bounded_autonomous_repair(
        runner=runner,
        source_generation=first,
        provider=object(),
        variants=(RepairVariant("default", 0.1),),
        policy=recursive.RecursiveRepairPolicy(max_depth=5),
    )
    assert outcome.depth == 1
    assert outcome.stop_reason is recursive.RecursiveRepairStopReason.CANCELLED


def test_positive_budget_but_unfittable_population_is_no_admissible_candidate(monkeypatch):
    generation = _generation(_candidate("source", score=-0.1))

    def reject_for_budget(**kwargs):
        raise ValueError(
            "no replay-adjusted repair variant fits the remaining GPU-hour budget"
        )

    monkeypatch.setattr(
        recursive, "run_single_hop_autonomous_repair", reject_for_budget
    )
    outcome = recursive.run_bounded_autonomous_repair(
        runner=_runner(budget=0.05),
        source_generation=generation,
        provider=object(),
        variants=(RepairVariant("default", 0.1),),
    )
    assert outcome.stop_reason is recursive.RecursiveRepairStopReason.NO_ADMISSIBLE_CANDIDATE
    assert outcome.depth == 0


def test_repeated_top_candidate_does_not_hide_lower_ranked_novel_target(monkeypatch):
    first = _generation(_candidate("source", score=-0.2, prompt="repeat"))
    repeated = _candidate("repair-repeat", score=-0.1, prompt="repeat")
    novel = _candidate("repair-novel", score=-0.11, prompt="novel")
    second = _generation(repeated, novel)
    third = _generation(_candidate("repair-final", score=0.0, prompt="third"))
    calls = []
    monkeypatch.setattr(
        recursive,
        "run_single_hop_autonomous_repair",
        _fake_hop_for((second, third), calls),
    )
    outcome = recursive.run_bounded_autonomous_repair(
        runner=_runner(),
        source_generation=first,
        provider=object(),
        variants=(RepairVariant("default", 0.1),),
        policy=recursive.RecursiveRepairPolicy(max_depth=2),
    )
    assert calls == ["source", "repair-novel"]
    assert outcome.depth == 2
    assert outcome.hops[1].target_experiment_id == "repair-novel"
    assert outcome.stop_reason is recursive.RecursiveRepairStopReason.MAX_DEPTH
