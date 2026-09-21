"""The autonomous growth control plane's core: profiling, memory, target choice.

The properties under test are the ones an autonomous loop depends on and the
old flat profile destroyed:

* a skill's estimate comes only from the benchmarks that measure it;
* a skill nobody measured is *unknown*, not zero;
* failures and interventions survive across processes, so generation N can
  teach generation N+1;
* the target selector is a reproducible function of durable evidence and cannot
  see a campaign's candidate results;
* a repeatedly failed intervention is penalized rather than repeated.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from chowder.evals.result import (
    MEASURED_PARENT,
    MEASURED_THIS_GENERATION,
    UNMEASURED,
    BenchmarkRun,
)
from chowder.growth.target_selection import (
    GrowthState,
    NextTargetSelector,
    build_skill_profile,
    classify_intervention,
)

MATH = "math500@2024-04"
MGSM = "mgsm@2022-11"
FORMAT = "generation-diagnostics@gen2-response-surface-v1"
CODING = "coding.generation"


def _run(qualified_id: str, score: float, *, origin: str = MEASURED_THIS_GENERATION,
         n_samples: int = 16) -> BenchmarkRun:
    return BenchmarkRun(
        benchmark_qualified_id=qualified_id,
        adapter="chowder_custom",
        generation_version="gen1",
        score=score,
        n_samples=n_samples,
        per_sample_scores=tuple([score] * n_samples),
        measurement_origin=origin,
    )


# --------------------------------------------------------------------------
# benchmark -> skill attribution
# --------------------------------------------------------------------------


def test_benchmark_specific_evidence_produces_different_skill_estimates() -> None:
    """math strong / formatting weak / coding unmeasured must not look alike."""
    profile = build_skill_profile(
        generation="gen1",
        runs=(_run(MATH, 0.90), _run(FORMAT, 0.20)),
    )
    math = profile.for_skill("math.competition")
    formatting = profile.for_skill("instruction.formatting")
    coding = profile.for_skill(CODING)

    assert math is not None and formatting is not None and coding is not None
    assert math.estimate == 0.90
    assert formatting.estimate == 0.20
    # The old flat mean would have given one number to every skill.
    assert math.estimate != formatting.estimate
    # Coding was never measured: unknown, not zero.
    assert coding.measured is False
    assert coding.estimate is None
    assert coding.confidence == 0.0
    assert coding.uncertainty == 1.0


def test_unmeasured_skill_is_unknown_not_zero() -> None:
    profile = build_skill_profile(generation="gen0", runs=())
    assert profile.measured == ()
    assert all(value.estimate is None for value in profile.estimates)
    assert all(value.uncertainty == 1.0 for value in profile.estimates)


def test_an_estimate_names_the_benchmarks_it_came_from() -> None:
    profile = build_skill_profile(generation="gen1", runs=(_run(MATH, 0.8),))
    math = profile.for_skill("math.competition")
    assert math is not None
    assert math.supporting_benchmarks == (MATH,)
    assert math.benchmark_measurements == {MATH: 0.8}
    assert math.provenance == MEASURED_THIS_GENERATION
    assert math.aggregation  # the method is stated, not implied


def test_carried_rows_do_not_produce_a_capability_estimate() -> None:
    """A reference row is not a measurement of this model."""
    profile = build_skill_profile(
        generation="gen1", runs=(_run(MATH, 0.9, origin=UNMEASURED),)
    )
    assert profile.for_skill("math.competition").measured is False


def test_a_parent_measurement_still_counts_but_weighs_less() -> None:
    candidate = build_skill_profile(
        generation="gen1", runs=(_run(MATH, 0.9, origin=MEASURED_THIS_GENERATION),)
    )
    parent = build_skill_profile(
        generation="gen1", runs=(_run(MATH, 0.9, origin=MEASURED_PARENT),)
    )
    assert parent.for_skill("math.competition").measured is True
    assert (
        parent.for_skill("math.competition").confidence
        <= candidate.for_skill("math.competition").confidence
    )


# --------------------------------------------------------------------------
# durable learning memory
# --------------------------------------------------------------------------


def _record_failure(state: GrowthState, *, generation: str, sample: str, category: str):
    from chowder.growth.failure_bank import FailureRecord

    return FailureRecord(
        failure_id=f"gen1|{FORMAT}|{sample}|abc",
        model_version="gen1",
        benchmark_qualified_id=FORMAT,
        sample_ref=sample,
        prompt_digest_hex="a" * 64,
        output_digest_hex="b" * 64,
        expected_behavior="emit a single response and stop",
        score=0.0,
        categories=(category,),
        verifier_evidence="digest",
        confidence=0.9,
        first_seen_generation=generation,
        last_seen_generation=generation,
        recurrence_count=1,
        repaired=False,
        repair_generation=None,
    )


def test_failures_survive_into_the_next_generation(tmp_path: Path) -> None:
    """A failure observed in gen2 is present when gen3 is planned."""
    state = GrowthState(tmp_path / "growth-state")
    state.persist_failures([_record_failure(state, generation="gen2", sample="s1", category="formatting")])
    del state  # a new process

    reopened = GrowthState(tmp_path / "growth-state")
    open_failures = reopened.reopen_failures()
    assert len(open_failures) == 1
    assert open_failures[0].first_seen_generation == "gen2"
    assert reopened.failure_bank().category_counts() == {"formatting": 1}


def test_repaired_failures_become_anti_forgetting_evidence(tmp_path: Path) -> None:
    state = GrowthState(tmp_path / "growth-state")
    record = _record_failure(state, generation="gen2", sample="s1", category="formatting")
    state.persist_failures([record])
    bank = state.failure_bank()
    bank.mark_repaired(record.failure_id, generation="gen3")
    assert bank.repaired_classes() == ("formatting",)


def test_intervention_and_target_history_persist(tmp_path: Path) -> None:
    state = GrowthState(tmp_path / "growth-state")
    state.record_intervention(
        target_skill="instruction.formatting",
        training_type="targeted_repair",
        cycle_id="gen2",
        generation="gen2",
        promotion_result="REJECTED",
        measured_effect=0.0,
    )
    reopened = GrowthState(tmp_path / "growth-state")
    assert reopened.attempts_on("instruction.formatting") == 1
    assert reopened.successful_effects("instruction.formatting") == (0.0,)


# --------------------------------------------------------------------------
# target selection
# --------------------------------------------------------------------------


def _profile():
    return build_skill_profile(
        generation="gen1", runs=(_run(MATH, 0.90), _run(MGSM, 0.85), _run(FORMAT, 0.20))
    )


def test_selector_is_reproducible_from_durable_evidence(tmp_path: Path) -> None:
    selector = NextTargetSelector()
    state = GrowthState(tmp_path / "growth-state")
    first = selector.propose(parent_version="gen1", profile=_profile(), state=state)
    second = selector.propose(parent_version="gen1", profile=_profile(), state=state)
    assert first.to_dict() == second.to_dict()
    # The weak, non-protected skill is chosen over the strong ones.
    assert first.target_skill == "instruction.formatting"
    assert first.suggested_training_type in {"targeted_repair", "sft"}


def test_selector_takes_no_candidate_evidence() -> None:
    """A target cannot be chosen by peeking at a candidate's results."""
    parameters = set(inspect.signature(NextTargetSelector.propose).parameters)
    assert "candidate_runs" not in parameters
    assert "candidate_results" not in parameters
    assert "runs" not in parameters
    assert parameters == {
        "self",
        "parent_version",
        "profile",
        "state",
        "category_counts",
        "known_skills",
        "max_targets_reported",
        # Not candidate evidence: the targets the running loop has already
        # exhausted, which the loop learns from its own durable record.
        "exclude_skills",
    }


def test_protected_skills_are_never_selection_targets(tmp_path: Path) -> None:
    selector = NextTargetSelector(protected_skills=("instruction.formatting",))
    state = GrowthState(tmp_path / "growth-state")
    proposal = selector.propose(parent_version="gen1", profile=_profile(), state=state)
    assert proposal.target_skill != "instruction.formatting"


def test_repeatedly_failed_intervention_is_penalized(tmp_path: Path) -> None:
    state = GrowthState(tmp_path / "growth-state")
    for _ in range(2):
        state.record_intervention(
            target_skill="instruction.formatting",
            training_type="targeted_repair",
            cycle_id="gen2",
            generation="gen2",
            measured_effect=0.0,
        )
    selector = NextTargetSelector(max_same_target_attempts=2)
    profile = _profile()
    estimate = profile.for_skill("instruction.formatting")
    repeated = selector.score_skill(estimate, state=state)
    fresh = selector.score_skill(estimate, state=GrowthState(tmp_path / "empty"))

    assert repeated.repeat_penalty > fresh.repeat_penalty
    assert repeated.total < fresh.total

    decision = classify_intervention(
        estimate=estimate, attempts=2, max_same_target_attempts=2
    )
    assert decision.treatment == "untrainable_with_current_path"
    assert decision.requires_review is True


def test_an_unmeasured_skill_is_never_selected_for_training(tmp_path: Path) -> None:
    profile = build_skill_profile(generation="gen1", runs=(_run(MATH, 0.9),))
    state = GrowthState(tmp_path / "growth-state")
    coding = profile.for_skill(CODING)
    decision = classify_intervention(estimate=coding, attempts=0)
    assert decision.treatment == "evaluation_needed"
    # ... and it is disclosed as the reason a skill was not chosen.
    proposal = NextTargetSelector().propose(
        parent_version="gen1", profile=profile, state=state
    )
    assert CODING in proposal.why_not_other_targets
    assert "insufficient evidence" in proposal.why_not_other_targets[CODING]


def test_a_structural_weakness_is_research_not_another_training_run() -> None:
    profile = _profile()
    decision = classify_intervention(
        estimate=profile.for_skill("instruction.formatting"), structural=True
    )
    assert decision.treatment == "architecture_research"
    assert decision.requires_review is True


def test_the_target_proposal_records_its_reasons(tmp_path: Path) -> None:
    proposal = NextTargetSelector().propose(
        parent_version="gen1",
        profile=_profile(),
        state=GrowthState(tmp_path / "growth-state"),
    )
    document = json.loads(json.dumps(proposal.to_dict()))
    assert document["weakness_evidence"]
    assert document["why_not_other_targets"]
    assert document["factors"]["total"] == document["priority"]
    assert document["expected_cost_gpu_hours"] > 0
