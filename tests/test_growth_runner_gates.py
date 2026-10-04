"""The runner's advance/promote gates: settlement refusals end lineages, and
promotion consults the preregistered retention profile and the tier wall.

Two invariants most likely to be bypassed by convenience, made structural:

- a settlement-refused attempt (``candidate_succeeded=True`` but the run
  settled over budget) advances nothing and promotes nothing: the flag is set
  before settlement runs, so "did it train" is not "may it win";
- the promotion path consults the declared retention profile before anything
  promotes -- a target win over a constrained regression is a REJECTED with
  the violation named, and a constraint measured on search-readable evidence
  refuses outright.

The settlement shapes here are the ones the training binding actually writes
(the ``budget_settlement`` verdict plus the ``refused_by`` stamp), so these
tests pin the production contract, not a test-only vocabulary.
"""

from __future__ import annotations

import pytest

from chowder.evals.result import (
    CARRIED_REFERENCE,
    MEASURED_PARENT,
    MEASURED_THIS_GENERATION,
    SUPPORTED,
    BenchmarkRun,
)
from chowder.growth.benchmark_registry import BenchmarkRegistry
from chowder.growth.attempt_failure import FailureClass, classify_failure
from chowder.growth.candidate_search import advanced, run_search
from chowder.growth.compute_cost import settlement_refusal
from chowder.growth.contamination import ContaminationFirewall
from chowder.growth.curriculum import CurriculumEngine
from chowder.growth.cycle import CycleConfig, GrowthCycle, select_candidate
from chowder.growth.eval_isolation import SearchIsolationRefusal, classify_benchmarks
from chowder.growth.failure_bank import FailureBank
from chowder.growth.frontier_reference import SnapshotStore
from chowder.growth.lineage import GenerationLedger, RegressionMemory
from chowder.growth.metric_binding import (
    BindingRefusal,
    BindingReport,
    MetricBinder,
    PromotionAssembly,
)
from chowder.growth.promotion import BenchmarkResult, PromotionInput, evaluate_promotion
from chowder.growth.retention import RetentionConstraint, RetentionProfile

from test_growth_candidate_search import (
    _attempt,
    _checkpoint_artifact,
    _declaration,
    _plan,
    _project_cost,
    _recipe,
)

TARGET = "target@2026-10"
REASONING = "reasoning@heldout"
TOOL = "tool@heldout"
BROAD = "broad@heldout"

_PROJECTION_REASON = (
    "ACTUAL_EXCEEDS_PROJECTION: actual wall 0.500000 exceeds projection "
    "0.100000 by more than the declared tolerance 0.25"
)


def _settlement_refused(recipe_id: str, **extra: object) -> dict:
    """A production-shaped settlement refusal: trained fine, settled over.

    ``candidate_succeeded`` is True because the training binding sets it
    before settlement runs -- exactly the hole the gate exists to close.
    """
    row: dict = {
        "recipe_id": recipe_id,
        "status": "REFUSED",
        "candidate_succeeded": True,
        "artifact_ref": f"/attempts/{recipe_id}/adapter",
        "budget_settlement": {
            "budget_compliant": False,
            "budget_failure_reasons": [_PROJECTION_REASON],
        },
        "refused_by": "budget_settlement",
        "refusal_reason": _PROJECTION_REASON,
    }
    row.update(extra)
    return row


def _samples(mean: float, spread: float = 0.02, blocks: int = 5) -> tuple[float, ...]:
    """Per-sample scores centered on ``mean`` with honest spread."""
    pattern = (-1.5, -0.5, 0.0, 0.5, 1.5)
    return tuple(mean + spread * p for p in pattern * blocks)


def _retention_code_reasons(decision) -> list[str]:  # noqa: ANN001
    """The decision's reasons that carry a RETENTION_* machine identifier."""
    return [r for r in decision.reasons if str(r).split(":", 1)[0].startswith("RETENTION_")]


def _result(benchmark: str, score: float, *, origin: str) -> BenchmarkResult:
    return BenchmarkResult(
        benchmark_qualified_id=benchmark,
        score=score,
        samples=_samples(score),
        contamination="CLEAN",
        measurement_origin=origin,
    )


def _promotion_results(
    *,
    target: tuple[float, float],
    reasoning: tuple[float, float] | None,
) -> tuple[dict[str, BenchmarkResult], dict[str, BenchmarkResult]]:
    """(candidate_results, parent_results) for a promotion decision.

    ``target`` is (candidate, parent); it always improves, so the promotion
    rule alone says PROMOTED and only the gates can say otherwise.
    ``reasoning`` is the measured pair for the retention constraint's
    benchmark, or None to leave it out entirely. A flat measured broad
    benchmark keeps the promotion rule itself out of the way: an unmeasured
    battery is INCONCLUSIVE, which would mask what the gates do.
    """
    candidate = {
        TARGET: _result(TARGET, target[0], origin=MEASURED_THIS_GENERATION),
        BROAD: _result(BROAD, 0.52, origin=MEASURED_THIS_GENERATION),
    }
    parent = {
        TARGET: _result(TARGET, target[1], origin=MEASURED_PARENT),
        BROAD: _result(BROAD, 0.52, origin=MEASURED_PARENT),
    }
    if reasoning is not None:
        candidate[REASONING] = _result(
            REASONING, reasoning[0], origin=MEASURED_THIS_GENERATION
        )
        parent[REASONING] = _result(REASONING, reasoning[1], origin=MEASURED_PARENT)
    return candidate, parent


def _profile(*benchmarks: str) -> RetentionProfile:
    """One max-regression constraint per named benchmark, dimension = name."""
    return RetentionProfile(
        profile_id="gate-tests",
        constraints=tuple(
            RetentionConstraint(
                dimension=benchmark,
                kind="max-regression",
                value=0.0,
                benchmark=benchmark,
            )
            for benchmark in benchmarks
        ),
    )


class _NullPlanner:
    def propose(self, items, *, count: int = 4):  # noqa: ANN001, ARG002
        return ()


def _cycle(
    tmp_path,
    *,  # noqa: ANN001
    retention_profile: RetentionProfile | None = None,
    eval_tier_policy=None,  # noqa: ANN001
) -> GrowthCycle:
    return GrowthCycle(
        CycleConfig(
            cycle_id="cycle-gates",
            parent_version="gen0",
            candidate_version="gen1",
            device_gpu_hours_ceiling=1.0,
            target_benchmarks=(TARGET,),
            protected_benchmarks=(),
            broad_battery=(BROAD,),
            recipe_count=1,
            retention_profile=retention_profile,
            eval_tier_policy=eval_tier_policy,
        ),
        curriculum=CurriculumEngine(),
        planner=_NullPlanner(),
        failure_bank=FailureBank(),
        firewall=ContaminationFirewall(),
        ledger=GenerationLedger(tmp_path / "ledger"),
        regression_memory=RegressionMemory(tmp_path / "probes"),
        snapshots=SnapshotStore(tmp_path / "snapshots"),
        train_fn=lambda recipe, items: {},  # noqa: ARG005
    )


def _decide(
    cycle: GrowthCycle,
    *,
    candidate: dict[str, BenchmarkResult],
    parent: dict[str, BenchmarkResult],
):
    return cycle.decide_promotion(
        candidate_results=candidate,
        parent_results=parent,
        device_gpu_hours=0.01,
    )


# --------------------------------------------------------------------------
# the shared predicate: one owner of the settlement-refusal vocabulary
# --------------------------------------------------------------------------


def test_the_predicate_reads_the_production_settlement_verdict() -> None:
    evidence = _settlement_refused("a")
    assert settlement_refusal(evidence) == "ACTUAL_EXCEEDS_PROJECTION"


def test_the_predicate_reads_the_refusal_stamp_alone() -> None:
    assert settlement_refusal(
        {"refused_by": "budget_settlement", "refusal_reason": _PROJECTION_REASON}
    ) == "ACTUAL_EXCEEDS_PROJECTION"


def test_the_predicate_leaves_compliant_absent_and_other_refusal_records_alone() -> None:
    assert settlement_refusal(_attempt("a")) is None
    assert settlement_refusal(
        _attempt(
            "a",
            budget_settlement={"budget_compliant": True, "budget_failure_reasons": []},
        )
    ) is None
    # A different gate's refusal stamp is not settlement's to claim.
    assert settlement_refusal(
        _attempt("a", refused_by="readiness", refusal_reason="no evaluator wired")
    ) is None


def test_a_non_compliant_verdict_without_reasons_still_names_the_gate() -> None:
    assert settlement_refusal(
        {"budget_settlement": {"budget_compliant": False, "budget_failure_reasons": []}}
    ) == "budget_settlement"


# --------------------------------------------------------------------------
# gate 1: settlement-refused attempts stop advancing
# --------------------------------------------------------------------------


def test_a_settlement_refused_attempt_does_not_advance() -> None:
    """The hole: candidate_succeeded=True with an artifact, settled over."""
    results = [_settlement_refused("a"), _attempt("b")]
    assert advanced(results, survivor_count=2) == ("b",)


def test_a_settled_attempt_still_advances() -> None:
    """A compliant settlement changes nothing -- the gate only bites overruns."""
    settled = _attempt(
        "a",
        budget_settlement={"budget_compliant": True, "budget_failure_reasons": []},
    )
    assert advanced([settled], survivor_count=1) == ("a",)


def test_a_settlement_refused_lineage_ends_in_the_search(tmp_path) -> None:  # noqa: ANN001
    """An over-budget attempt cannot earn a larger budget, however well it trained.

    The attempt stays in the record with its spend -- the compute really
    happened -- but its lineage ends: it is not a survivor, the next round
    does not resume it, and the run's stops say why.
    """
    recipes = [_recipe("a"), _recipe("b")]
    plan = _plan(recipes)
    artifacts = {rid: _checkpoint_artifact(tmp_path, rid) for rid in ("a", "b")}

    def run_attempt(recipe):
        if recipe.recipe_id == "b":  # noqa: ANN001
            return _settlement_refused("b", artifact_ref=artifacts["b"])
        return _attempt("a", artifact_ref=artifacts["a"])

    run = run_search(
        plan,
        declaration=_declaration(),
        recipes=recipes,
        project_cost=_project_cost,
        run_attempt=run_attempt,
    )

    assert run.survivors == ("a",)
    assert "b" in run.lineage_stops
    assert "settlement refused the attempt (ACTUAL_EXCEEDS_PROJECTION)" in run.lineage_stops["b"]
    # The refused attempt is still in the accounting: real compute, real spend.
    b_attempts = [
        evidence
        for round_attempts in run.round_attempts
        for evidence in round_attempts
        if evidence.get("recipe_id") == "b"
    ]
    assert len(b_attempts) == 1
    assert run.candidate_cumulative["b"]["rounds"] == 1


def test_selection_skips_a_settlement_refused_attempt() -> None:
    """The promote-side screen: an unpriced attempt is never the candidate."""
    good = _attempt("b")
    assert select_candidate([_settlement_refused("a"), good]) is good
    assert select_candidate([_settlement_refused("a"), _settlement_refused("c")]) is None


def test_the_failure_classifier_reads_the_production_settlement_shape() -> None:
    """The classifier and the runner's advance rule agree on what was refused."""
    classification = classify_failure(_settlement_refused("a"))
    assert classification.failure_class is FailureClass.BUDGET_EXHAUSTED
    assert classification.evidence_state is None
    assert "ACTUAL_EXCEEDS_PROJECTION" in classification.reason


# --------------------------------------------------------------------------
# gate 2: promotion consults the preregistered retention profile
# --------------------------------------------------------------------------


def test_a_retention_regression_rejects_a_promotion(tmp_path) -> None:  # noqa: ANN001
    """The 0.60->0.78 with reasoning 0.71->0.51 shape is a refusal, not a footnote."""
    cycle = _cycle(tmp_path, retention_profile=_profile(REASONING))
    candidate, parent = _promotion_results(target=(0.45, 0.28), reasoning=(0.51, 0.71))

    decision = _decide(cycle, candidate=candidate, parent=parent)

    assert decision.verdict == "REJECTED"
    regression = [
        reason for reason in _retention_code_reasons(decision)
        if reason.startswith("RETENTION_REGRESSION") and "reasoning@heldout" in reason
    ]
    assert regression, decision.reasons
    # The reason string carries the domain's own RETENTION_* code: the
    # vocabulary is owned by RetentionViolation.code, not re-derived here.
    assert regression[0].startswith("RETENTION_REGRESSION: ")


def test_an_unmeasured_constraint_rejects_fail_closed(tmp_path) -> None:  # noqa: ANN001
    """A gate that cannot be measured is not a gate that was passed."""
    cycle = _cycle(tmp_path, retention_profile=_profile(TOOL))
    candidate, parent = _promotion_results(target=(0.45, 0.28), reasoning=None)

    decision = _decide(cycle, candidate=candidate, parent=parent)

    assert decision.verdict == "REJECTED"
    assert any(
        reason.startswith("RETENTION_UNMEASURED") for reason in decision.reasons
    )


def test_an_unmeasured_parent_side_refuses_the_delta(tmp_path) -> None:  # noqa: ANN001
    """max-regression needs both sides: a missing parent cannot certify."""
    cycle = _cycle(tmp_path, retention_profile=_profile(REASONING))
    candidate, parent = _promotion_results(target=(0.45, 0.28), reasoning=None)
    candidate[REASONING] = _result(REASONING, 0.72, origin=MEASURED_THIS_GENERATION)

    decision = _decide(cycle, candidate=candidate, parent=parent)

    assert decision.verdict == "REJECTED"
    assert any("no parent measurement" in reason for reason in decision.reasons)


def test_a_carried_parent_row_is_not_a_baseline(tmp_path) -> None:  # noqa: ANN001
    """A carried reference is a quotation from history, not a parent measurement.

    The parent side of a declared constraint demands earned provenance
    (``parent_measured``), symmetric with the candidate side's
    ``gate_eligible``: a carried row reads as unmeasured, so the gate
    refuses with RETENTION_UNMEASURED instead of silently anchoring the
    delta on a number nothing measured.
    """
    cycle = _cycle(tmp_path, retention_profile=_profile(REASONING))
    candidate, parent = _promotion_results(target=(0.45, 0.28), reasoning=None)
    # A measured candidate against a carried baseline with a plausible
    # number: trusting the carried row would anchor the declared delta on a
    # number nothing measured (here it would read as a regression); the
    # earned-provenance rule reads it as absent instead.
    candidate[REASONING] = _result(REASONING, 0.51, origin=MEASURED_THIS_GENERATION)
    parent[REASONING] = _result(REASONING, 0.71, origin=CARRIED_REFERENCE)

    decision = _decide(cycle, candidate=candidate, parent=parent)

    assert decision.verdict == "REJECTED"
    assert any(
        reason.startswith("RETENTION_UNMEASURED") and "no parent measurement" in reason
        for reason in decision.reasons
    )


def test_the_binder_path_never_hands_the_gate_a_carried_parent(tmp_path) -> None:  # noqa: ANN001
    """The wall is defense in depth: the provenance owner refuses first.

    ``metric_binding`` refuses a CARRIED_REFERENCE row on the parent role,
    so the production run path (``decide_promotion_from_runs``) never
    delivers one to the gate; the fail-closed rule in ``_retention_values``
    is the backstop for the caller-passed seam. This pins the no-op: the
    same carried row through the binder comes out as a refusal, not a
    parent result.
    """
    carried_parent_run = BenchmarkRun(
        benchmark_qualified_id=REASONING,
        adapter="native",
        generation_version="gen0",
        metric="accuracy",
        score=0.71,
        support=SUPPORTED,
        n_samples=25,
        measurement_origin=CARRIED_REFERENCE,
    )
    binder = MetricBinder(BenchmarkRegistry())
    outcome = binder.bind(carried_parent_run, generation_version="gen0", role="parent")

    assert isinstance(outcome, BindingRefusal)
    assert "carried reference" in outcome.reason


def test_an_absolute_floor_breach_is_named(tmp_path) -> None:  # noqa: ANN001
    cycle = _cycle(
        tmp_path,
        retention_profile=RetentionProfile(
            profile_id="floors",
            constraints=(
                RetentionConstraint(
                    dimension="tool", kind="absolute-floor", value=0.30, benchmark=TOOL
                ),
            ),
        ),
    )
    candidate, parent = _promotion_results(target=(0.45, 0.28), reasoning=None)
    candidate[TOOL] = _result(TOOL, 0.10, origin=MEASURED_THIS_GENERATION)
    parent[TOOL] = _result(TOOL, 0.40, origin=MEASURED_PARENT)

    decision = _decide(cycle, candidate=candidate, parent=parent)

    assert decision.verdict == "REJECTED"
    assert any(reason.startswith("RETENTION_FLOOR") for reason in decision.reasons)


def test_a_retention_compliant_candidate_promotes_unchanged(tmp_path) -> None:  # noqa: ANN001
    """The gate may only bite breaches: a clean candidate keeps its verdict."""
    cycle = _cycle(tmp_path, retention_profile=_profile(REASONING, TOOL))
    candidate, parent = _promotion_results(target=(0.45, 0.28), reasoning=(0.72, 0.71))
    candidate[TOOL] = _result(TOOL, 0.40, origin=MEASURED_THIS_GENERATION)
    parent[TOOL] = _result(TOOL, 0.40, origin=MEASURED_PARENT)

    decision = _decide(cycle, candidate=candidate, parent=parent)

    assert decision.verdict == "PROMOTED"
    assert not any(reason.startswith("RETENTION_") for reason in decision.reasons)


def test_without_a_profile_promotion_is_unchanged(tmp_path) -> None:  # noqa: ANN001
    """The profile binds because it was declared, not because a number moved."""
    cycle = _cycle(tmp_path)
    candidate, parent = _promotion_results(target=(0.45, 0.28), reasoning=(0.51, 0.71))

    decision = _decide(cycle, candidate=candidate, parent=parent)

    assert decision.verdict == "PROMOTED"


# --------------------------------------------------------------------------
# gate 2b: the tier wall reaches the promotion gate
# --------------------------------------------------------------------------


def test_a_constraint_on_search_readable_evidence_refuses(tmp_path) -> None:  # noqa: ANN001
    """A gate the search could see is a gate the search could shape.

    The measurements here would pass numerically; the classification alone
    refuses, because wiring -- not measurement -- is what is wrong.
    """
    policy = classify_benchmarks({REASONING: "search-evidence"})
    cycle = _cycle(
        tmp_path,
        retention_profile=_profile(REASONING),
        eval_tier_policy=policy,
    )
    candidate, parent = _promotion_results(target=(0.45, 0.28), reasoning=(0.72, 0.71))

    with pytest.raises(SearchIsolationRefusal) as error:
        _decide(cycle, candidate=candidate, parent=parent)

    assert "shape its own gate" in str(error.value)


def test_a_promotion_evidence_classification_allows_the_gate(tmp_path) -> None:  # noqa: ANN001
    policy = classify_benchmarks({REASONING: "promotion-evidence"})
    cycle = _cycle(
        tmp_path,
        retention_profile=_profile(REASONING),
        eval_tier_policy=policy,
    )
    candidate, parent = _promotion_results(target=(0.45, 0.28), reasoning=(0.72, 0.71))

    decision = _decide(cycle, candidate=candidate, parent=parent)

    assert decision.verdict == "PROMOTED"


# --------------------------------------------------------------------------
# gate 2 reaches the measured-runs path too
# --------------------------------------------------------------------------


class _StubBinder:
    """Returns a fixed PROMOTED assembly; the cycle's gates must still apply.

    The binder's own arithmetic is pinned elsewhere; this stub isolates the
    cycle's job: whatever the binder decided, the declared gates are applied
    before the assembly leaves the promotion path.
    """

    def promotion_input(self, **kwargs):  # noqa: ANN003, ARG002
        candidate, parent = _promotion_results(
            target=(0.45, 0.28), reasoning=(0.51, 0.71)
        )
        data = PromotionInput(
            candidate_version="gen1",
            parent_version="gen0",
            target_benchmarks=(TARGET,),
            candidate_results=candidate,
            parent_results=parent,
            protected_benchmarks=(),
            broad_battery_benchmarks=(BROAD,),
        )
        return PromotionAssembly(
            promotion_input=data,
            decision=evaluate_promotion(data),
            report=BindingReport("gen1", candidate),
            parent_report=BindingReport("gen0", parent),
        )


def test_the_from_runs_path_applies_the_gates(tmp_path) -> None:  # noqa: ANN001
    cycle = _cycle(tmp_path, retention_profile=_profile(REASONING))

    assembly = cycle.decide_promotion_from_runs(
        _StubBinder(), candidate_runs=(), parent_runs=()
    )

    assert assembly.decision.verdict == "REJECTED"
    assert any(
        reason.startswith("RETENTION_REGRESSION") for reason in assembly.decision.reasons
    )


def test_the_from_runs_path_leaves_a_ungated_decision_alone(tmp_path) -> None:  # noqa: ANN001
    cycle = _cycle(tmp_path)

    assembly = cycle.decide_promotion_from_runs(
        _StubBinder(), candidate_runs=(), parent_runs=()
    )

    assert assembly.decision.verdict == "PROMOTED"
