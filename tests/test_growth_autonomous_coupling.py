"""The couplings that make one generation teach the next, tested end to end.

Three invariants, each of which the autonomous path got wrong before:

* **one authoritative capability profile.** A skill is estimated from the
  benchmarks that measure *it*, and a skill nobody measured stays unknown.
  ``math strong / formatting weak / coding unmeasured`` must survive as three
  different states rather than collapsing into one mean;
* **protected skills are derived from the policy's protected benchmarks.** A
  capability measured only by a gated benchmark must not be proposed as an
  optimization target, and the selector must *say* it is a gate rather than
  silently omitting it;
* **the parent is a named lineage object.** A promotion advances the evidence
  pointer to the exact run it produced; a rejection leaves the previous parent
  authoritative.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chowder.evals.result import MEASURED_PARENT, BenchmarkRun
from chowder.growth.campaign_runner import PARENT_PROFILE_NOT_ATTRIBUTED, _load_profile
from chowder.growth.capability import CapabilityProfile
from chowder.growth.growth_loop import GrowthLoop
from chowder.growth.next_campaign import ParentEvidenceRef
from chowder.growth.simulator import skill_profile
from chowder.growth.target_selection import (
    GrowthState,
    NextTargetSelector,
    build_skill_profile,
    protected_skills_for,
)
from fixtures_growth_loop import (
    NO_RUN_ROOT,
    RecordingExecutor,
    outcome,
    parent_manifest,
    planned_recipes,
    policy_from,
)

#: A strong math capability, a weak formatting one, and a coding capability
#: nobody measured. The three states the mission asks to keep distinct.
MATH_ID = "math500@2024-04"
FORMATTING_ID = "generation-diagnostics@gen2-response-surface-v1"
CODING_ID = "humaneval@2023-05-31"

IDENTITY = ("adapters/promoted", "d" * 64)


def _row(qualified_id: str, score: float, *, generation: str = "gen1") -> BenchmarkRun:
    return BenchmarkRun(
        benchmark_qualified_id=qualified_id,
        adapter="chowder_custom",
        generation_version=generation,
        score=score,
        n_samples=16,
        measurement_origin=MEASURED_PARENT,
    )


def _three_state_profile() -> Any:
    return build_skill_profile(
        generation="gen1",
        runs=(_row(MATH_ID, 0.9), _row(FORMATTING_ID, 0.25)),
    )


# --------------------------------------------------------------------------
# Part 2 -- one authoritative, evidence-attributed capability profile
# --------------------------------------------------------------------------


def test_math_formatting_and_coding_remain_three_different_states() -> None:
    """The attribution invariant: measured strong, measured weak, unmeasured."""
    profile = _three_state_profile()

    math = profile.for_skill("math.competition")
    formatting = profile.for_skill("instruction.formatting")
    coding = profile.for_skill("coding.generation")

    assert math is not None and math.estimate == pytest.approx(0.9)
    assert formatting is not None and formatting.estimate == pytest.approx(0.25)
    assert math.estimate != formatting.estimate, "one mean for every skill is the defect"
    assert coding is not None and coding.estimate is None, "unmeasured is not zero"
    assert coding.confidence == 0.0
    assert coding.supporting_benchmarks == ()


def test_the_derived_capability_view_keeps_the_unmeasured_skills_named() -> None:
    """The flat shape the curriculum reads loses ``None``, so it must say so."""
    view = CapabilityProfile.from_skill_profile(
        _three_state_profile(), model_version="gen1"
    )

    assert view.skill("math.competition").estimate == pytest.approx(0.9)
    assert view.skill("instruction.formatting").estimate == pytest.approx(0.25)
    # Unknown is carried at confidence 0 -- which the curriculum's own
    # `confidence <= 0.05` filter reads as "do not prioritize" -- and is named,
    # so the view cannot be mistaken for a measured zero.
    unmeasured = view.skill("coding.generation")
    assert unmeasured.confidence == 0.0
    assert unmeasured.evidence == ()
    assert "coding.generation" in view.notes["unmeasured"]
    assert "math.competition" not in view.notes["unmeasured"]
    assert view.notes["authoritative_profile"] == "SkillProfile"


def test_a_flat_unattributed_parent_profile_is_refused_by_name(
    tmp_path: Path,
) -> None:
    """Silently accepting the old shape would put an un-attributed mean back."""
    document = tmp_path / "parent-profile.json"
    document.write_text(
        '{"model_version": "gen1", "raw_scores": {}, "skills": []}', encoding="utf-8"
    )

    class _Manifest:
        parent_profile_path = str(document)
        parent_version = "gen1"

    with pytest.raises(Exception, match=PARENT_PROFILE_NOT_ATTRIBUTED):
        _load_profile(_Manifest())


# --------------------------------------------------------------------------
# Part 5 -- protected skills derived from the policy's protected benchmarks
# --------------------------------------------------------------------------


def test_protected_skills_are_the_skills_of_the_protected_benchmarks() -> None:
    skills = protected_skills_for([MATH_ID, "mgsm@2022-11"])

    assert "math.competition" in skills
    assert "math.algebra" in skills
    assert "math.arithmetic" in skills
    assert "multilingual.comprehension" in skills
    # The target instrument's skill is not gated, so it stays targetable.
    assert "instruction.formatting" not in skills


def test_a_protected_benchmark_the_registry_does_not_know_refuses() -> None:
    """An unresolvable protected set protects nothing; that is refused, not empty."""
    with pytest.raises(ValueError, match="not in the benchmark registry"):
        protected_skills_for([MATH_ID, "not-a-benchmark@v0"])


def test_a_protected_skill_is_never_proposed_and_is_named_as_a_gate(
    tmp_path: Path,
) -> None:
    """A skill measured by a protected benchmark must not become a target.

    The stronger case is the mixed one: the same skill supported by both a
    protected and a targetable benchmark. The rule is stated once -- *any*
    protected support makes the skill a gate -- and the selector has to say so
    rather than leave the skill looking like something it forgot.
    """
    profile = _three_state_profile()
    selector = NextTargetSelector(protected_skills=protected_skills_for([MATH_ID]))

    proposal = selector.propose(
        parent_version="gen1",
        profile=profile,
        state=GrowthState(root=tmp_path / "state"),
    )

    assert proposal.target_skill == "instruction.formatting"
    assert "math.competition" in proposal.why_not_other_targets
    assert "protected" in proposal.why_not_other_targets["math.competition"]


def test_the_loop_derives_protected_skills_from_its_policy(tmp_path: Path) -> None:
    """The wiring, not just the helper: the loop passes the derived set down."""
    parent = parent_manifest(tmp_path)
    policy = policy_from(parent)
    loop = GrowthLoop(
        policy=policy,
        state=GrowthState(root=tmp_path / "state"),
        executor=RecordingExecutor([]),
        parent_declaration=parent,
        parent_profile=skill_profile({"instruction.formatting": 0.25}, generation="gen2"),
        prepare=planned_recipes,
        readiness=lambda frozen: True,
    )

    expected = protected_skills_for(policy.protected_benchmarks)
    assert loop.protected_skills == expected
    assert loop.selector.protected_skills == expected
    assert "math.competition" in expected


# --------------------------------------------------------------------------
# Part 3 -- the parent is a named lineage object, advanced only by a promotion
# --------------------------------------------------------------------------


def _loop(tmp_path: Path, executor: Any, **kwargs: Any) -> tuple[GrowthLoop, GrowthState]:
    parent = kwargs.pop("parent", None) or parent_manifest(tmp_path)
    policy = kwargs.pop("policy", None) or policy_from(parent)
    state = kwargs.pop("state", None) or GrowthState(root=tmp_path / "growth-state")
    profile = kwargs.pop("profile", {"instruction.formatting": 0.31})
    generation = kwargs.pop("generation", "gen2")
    loop = GrowthLoop(
        policy=policy,
        state=state,
        executor=executor,
        parent_declaration=parent,
        parent_profile=skill_profile(profile, generation=generation) if profile else None,
        prepare=planned_recipes,
        readiness=lambda frozen: True,
        **kwargs,
    )
    return loop, state


def test_the_initial_parent_evidence_is_projected_from_the_declaration(
    tmp_path: Path,
) -> None:
    """Stated, never discovered: every field is a declared field."""
    parent = parent_manifest(tmp_path)
    loop, _ = _loop(tmp_path, RecordingExecutor([]), parent=parent)

    ref = loop.parent_evidence
    assert ref.run_root == str(parent.state_root)
    assert ref.base_model_path == parent.base_model_path
    assert ref.base_model_digest == parent.base_model_digest
    assert ref.adapter_path == parent.parent_adapter_path
    assert ref.adapter_digest == parent.parent_adapter_digest
    assert ref.identity == loop.parent_identity
    assert ref.measured_arm_path == str(
        Path(parent.state_root) / "candidate_evaluation.json"
    )


def test_a_promotion_advances_parent_evidence_to_the_exact_run_it_produced(
    tmp_path: Path,
) -> None:
    executor = RecordingExecutor(
        [
            outcome(
                "PROMOTED",
                wall_gpu_hours=0.42,
                measured_target_effect=0.18,
                promoted_identity=IDENTITY,
                profile_scores={"instruction.formatting": 0.88},
            )
        ]
    )
    loop, state = _loop(
        tmp_path, executor, policy=policy_from(parent_manifest(tmp_path), maximum_generations=1)
    )
    report = loop.run()

    assert report.decision.action == "STOP_SUCCESS"
    # The durable record, not just the in-memory pointer: a resume restores from
    # here, so the ref has to be what was written.
    recorded = state.parent_evidence()
    assert recorded is not None
    assert recorded["generation"] == "gen3"
    assert recorded["run_root"] == str(tmp_path / "gen3-a1-instruction-formatting")
    assert recorded["adapter_path"] == IDENTITY[0]
    assert recorded["adapter_digest"] == IDENTITY[1]
    assert loop.parent_evidence.generation == "gen3"
    assert loop.parent_identity == IDENTITY


def test_a_rejection_leaves_the_previous_parent_evidence_authoritative(
    tmp_path: Path,
) -> None:
    executor = RecordingExecutor(
        [
            outcome(
                "REJECTED",
                wall_gpu_hours=0.42,
                measured_target_effect=0.0,
                profile_scores={"instruction.formatting": 0.31},
            )
        ]
    )
    parent = parent_manifest(tmp_path)
    initial = ParentEvidenceRef.from_declaration(parent)
    loop, state = _loop(
        tmp_path,
        executor,
        parent=parent,
        policy=policy_from(parent, maximum_generations=1),
    )
    loop.run()

    # Nothing promoted, so nothing in the durable record may claim a new parent.
    assert state.parent_evidence() is None
    assert loop.parent_evidence == initial
    assert loop.parent_evidence.generation == "gen2"


def test_a_promotion_without_a_run_root_does_not_advance_the_pointer(
    tmp_path: Path,
) -> None:
    """A promotion the loop cannot locate refuses instead of guessing a directory."""
    executor = RecordingExecutor(
        [
            outcome(
                "PROMOTED",
                wall_gpu_hours=0.42,
                promoted_identity=IDENTITY,
                profile_scores={"instruction.formatting": 0.88},
                run_root=NO_RUN_ROOT,
            )
        ]
    )
    parent = parent_manifest(tmp_path)
    loop, state = _loop(tmp_path, executor, parent=parent)
    report = loop.run()

    assert report.decision.action == "STOP_UNCERTAIN", report.decision.reason
    assert "PROMOTION_WITHOUT_EVIDENCE_ROOT" in report.decision.reason_codes
    assert state.parent_evidence() is None
    assert loop.parent_evidence == ParentEvidenceRef.from_declaration(parent)


def test_a_sibling_run_roots_evidence_is_never_substituted(tmp_path: Path) -> None:
    """The defect named by the audit: ``state_root.parent.parent`` as a locator.

    A ref that names a run root other than the one the declaration describes is
    refused before composition, so no directory search can stand in for it.
    """
    parent = parent_manifest(tmp_path)
    sibling = ParentEvidenceRef(
        generation=parent.resolved_candidate_version(),
        run_root=str(tmp_path / "some-other-run"),
        base_model_path=parent.base_model_path,
        base_model_digest=parent.base_model_digest,
        adapter_path=parent.parent_adapter_path,
        adapter_digest=parent.parent_adapter_digest,
    )
    executor = RecordingExecutor([])
    loop, state = _loop(tmp_path, executor, parent=parent)
    loop.parent_evidence = sibling

    assert sibling.run_root != str(parent.state_root)
    report = loop.run()

    # Composition refuses: the ref describes evidence this generation did not
    # produce, so no campaign is built and nothing is spent.
    assert report.decision.action == "REQUIRES_HUMAN_REVIEW"
    assert report.generations == ()
    assert executor.calls == []
    assert state.parent_evidence() is None
