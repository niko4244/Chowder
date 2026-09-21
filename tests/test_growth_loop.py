"""The outer controller: what it refuses, what it spends, what it resumes.

The loop is the only component that can authorise a *second* generation, so the
properties under test are its refusals rather than its happy path:

* no measured capability means no target, so no campaign;
* a campaign that ran without reporting a measured wall cost ends the session
  with a durable decision -- it is not charged an estimate and it is not free;
* a promotion the loop cannot name an adapter for, or cannot build a next target
  from, stops instead of advancing the parent;
* an envelope that cannot hold the next campaign launches nothing;
* session spend is the sum of measured campaign costs, exactly;
* resuming reads the durable record: a finished session returns its stored
  verdict and spends nothing, and a promotion recorded durably is restored
  rather than guessed at.

The last two exist because this pass found them broken: the unmeasured-cost rule
was documented but unreachable (``charge`` raised first, so the loop crashed
instead of deciding), and ``run`` never read the stopping state it wrote.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chowder.growth.growth_loop import (
    CAMPAIGN_SPENT_NO_MEASURED_COST,
    NO_MEASURED_CAPABILITY,
    PROFILE_GENERATION_MISMATCH,
    PROMOTION_WITHOUT_IDENTITY,
    PROMOTION_WITHOUT_PROFILE,
    REMAINING_ENVELOPE_TOO_SMALL,
    STOP_BUDGET,
    STOP_UNCERTAIN,
    GrowthLoop,
    PlateauCheck,
    detect_plateau,
)
from chowder.growth.simulator import skill_profile
from chowder.growth.target_selection import GrowthState
from fixtures_growth_loop import (
    RecordingExecutor,
    outcome,
    parent_manifest,
    planned_recipes,
    policy_from,
)

START = {
    "math.reasoning": 0.62,
    "instruction.formatting": 0.31,
    "termination.control": 0.44,
}
PROMOTED_AFTER_FORMATTING = {**START, "instruction.formatting": 0.88}
IDENTITY = ("adapters/gen3", "a" * 64)


def _loop(
    tmp_path: Path,
    executor,
    *,
    policy=None,
    parent=None,
    profile=START,
    generation="gen2",
    state=None,
):
    parent = parent or parent_manifest(tmp_path)
    policy = policy or policy_from(parent)
    state = state or GrowthState(root=tmp_path / "growth-state")
    loop = GrowthLoop(
        policy=policy,
        state=state,
        executor=executor,
        parent_declaration=parent,
        # The generation the profile was measured on is part of the evidence, not
        # a detail: the loop refuses to plan from a profile of another model.
        parent_profile=skill_profile(profile, generation=generation) if profile else None,
        prepare=planned_recipes,
        readiness=lambda frozen: True,
    )
    return loop, state


def test_a_parent_with_no_measured_profile_launches_nothing(tmp_path: Path) -> None:
    executor = RecordingExecutor([])
    loop, _ = _loop(tmp_path, executor, profile=None)

    report = loop.run()

    assert report.decision.action == STOP_UNCERTAIN
    assert NO_MEASURED_CAPABILITY in report.decision.reason_codes
    assert executor.calls == []
    assert report.budget["spent_wall_gpu_hours"] == 0.0


def test_a_campaign_that_reported_no_measured_cost_ends_the_session(
    tmp_path: Path,
) -> None:
    """The regression: this used to raise out of ``run`` instead of deciding.

    A campaign that executed and reported no measured wall cost must produce a
    terminal, durable refusal. Charging it zero would make unaccounted compute
    free, and raising ``LoopBudgetError`` through the caller would end the
    session with no decision, no stopping state and no accounting.
    """
    executor = RecordingExecutor(
        [outcome("REJECTED", wall_gpu_hours=None, measured_target_effect=0.0)]
    )
    loop, state = _loop(tmp_path, executor)

    report = loop.run()

    assert len(executor.calls) == 1, "the campaign did run; only its cost is missing"
    assert report.decision.action == STOP_UNCERTAIN
    assert CAMPAIGN_SPENT_NO_MEASURED_COST in report.decision.reason_codes
    assert report.decision.terminal
    assert report.budget["spent_wall_gpu_hours"] == 0.0, "an unmeasured cost is not zero"
    assert state.stopping_state()["action"] == STOP_UNCERTAIN
    assert len(report.generations) == 1


def test_a_promotion_without_a_named_adapter_cannot_advance_the_parent(
    tmp_path: Path,
) -> None:
    executor = RecordingExecutor(
        [
            outcome(
                "PROMOTED",
                wall_gpu_hours=0.42,
                measured_target_effect=0.18,
                promoted_identity=None,
                profile_scores=PROMOTED_AFTER_FORMATTING,
            )
        ]
    )
    loop, _ = _loop(tmp_path, executor)

    report = loop.run()

    assert report.decision.action == STOP_UNCERTAIN
    assert PROMOTION_WITHOUT_IDENTITY in report.decision.reason_codes
    assert report.parent_version == "gen2", "an unnamed promotion cannot move the parent"


def test_a_promotion_without_a_measured_profile_cannot_advance_the_parent(
    tmp_path: Path,
) -> None:
    executor = RecordingExecutor(
        [
            outcome(
                "PROMOTED",
                wall_gpu_hours=0.42,
                measured_target_effect=0.18,
                promoted_identity=IDENTITY,
                profile_scores=None,
            )
        ]
    )
    loop, _ = _loop(tmp_path, executor)

    report = loop.run()

    assert report.decision.action == STOP_UNCERTAIN
    assert PROMOTION_WITHOUT_PROFILE in report.decision.reason_codes
    assert report.parent_version == "gen2"


def test_an_envelope_too_small_for_the_campaign_launches_nothing(tmp_path: Path) -> None:
    parent = parent_manifest(tmp_path)
    policy = policy_from(parent, maximum_total_wall_gpu_hours=0.01)
    executor = RecordingExecutor([])
    loop, _ = _loop(tmp_path, executor, policy=policy, parent=parent)

    report = loop.run()

    assert report.decision.action == STOP_BUDGET
    assert REMAINING_ENVELOPE_TOO_SMALL in report.decision.reason_codes
    assert executor.calls == []


def test_session_spend_is_the_sum_of_the_measured_campaign_costs(tmp_path: Path) -> None:
    executor = RecordingExecutor(
        [
            outcome(
                "PROMOTED",
                wall_gpu_hours=0.42,
                measured_target_effect=0.18,
                promoted_identity=IDENTITY,
                profile_scores=PROMOTED_AFTER_FORMATTING,
            ),
            outcome("REJECTED", wall_gpu_hours=0.37, measured_target_effect=0.0),
        ]
    )
    loop, _ = _loop(tmp_path, executor, policy=policy_from(parent_manifest(tmp_path), maximum_generations=2))

    report = loop.run()

    assert len(executor.calls) == 2
    assert report.budget["spent_wall_gpu_hours"] == pytest.approx(0.42 + 0.37)
    assert report.budget["remaining_wall_gpu_hours"] == pytest.approx(6.0 - 0.79)


def test_the_loop_never_exceeds_the_policy_generation_limit(tmp_path: Path) -> None:
    parent = parent_manifest(tmp_path)
    executor = RecordingExecutor(
        [
            outcome("REJECTED", wall_gpu_hours=0.4, measured_target_effect=0.0),
            outcome("REJECTED", wall_gpu_hours=0.4, measured_target_effect=0.0),
            outcome("REJECTED", wall_gpu_hours=0.4, measured_target_effect=0.0),
        ]
    )
    loop, _ = _loop(
        tmp_path,
        executor,
        policy=policy_from(parent, maximum_generations=2),
        parent=parent,
    )

    loop.run()

    assert len(executor.calls) == 2, "the third declared outcome must never be reached"


def test_a_finished_session_resumes_from_its_durable_verdict_and_spends_nothing(
    tmp_path: Path,
) -> None:
    """A restart must adopt what the run already decided, not re-run it."""
    first_executor = RecordingExecutor(
        [
            outcome(
                "PROMOTED",
                wall_gpu_hours=0.42,
                measured_target_effect=0.18,
                promoted_identity=IDENTITY,
                profile_scores=PROMOTED_AFTER_FORMATTING,
            )
        ]
    )
    parent = parent_manifest(tmp_path)
    policy = policy_from(parent, maximum_generations=1)
    loop, state = _loop(tmp_path, first_executor, policy=policy, parent=parent)
    finished = loop.run()
    assert finished.decision.terminal

    # A fresh process, same durable state, and an executor that must not be used.
    second_executor = RecordingExecutor([])
    resumed, _ = _loop(
        tmp_path,
        second_executor,
        policy=policy,
        parent=parent_manifest(tmp_path),
        profile=None,
        state=state,
    )
    report = resumed.run(resume=True)

    assert second_executor.calls == [], "resuming a finished session must spend nothing"
    assert report.decision.action == finished.decision.action
    assert report.decision.reason_codes == finished.decision.reason_codes
    assert report.budget["spent_wall_gpu_hours"] == pytest.approx(0.42)
    assert len(report.generations) == 1
    assert report.generations[0].verdict == "PROMOTED"


def test_resuming_restores_the_adapter_and_declaration_a_promotion_produced(
    tmp_path: Path,
) -> None:
    """A crash after the verdict must continue from the *promoted* parent.

    The durable record names the adapter the generation promoted and the frozen
    declaration it ran under; without restoring both, a resumed loop would train
    the next generation from the parent it started with and re-spend the same
    generation.
    """
    first = RecordingExecutor(
        [
            outcome(
                "PROMOTED",
                wall_gpu_hours=0.42,
                measured_target_effect=0.18,
                promoted_identity=IDENTITY,
                profile_scores=PROMOTED_AFTER_FORMATTING,
            )
        ]
    )
    parent = parent_manifest(tmp_path)
    policy = policy_from(parent, maximum_generations=1)
    loop, state = _loop(tmp_path, first, policy=policy, parent=parent)
    loop.run()

    # The process died after the campaign's verdict was recorded but before the
    # loop decided what to do next: no stopping state, one durable generation.
    (state.root / "stopping-state.json").unlink()

    second = RecordingExecutor(
        [outcome("REJECTED", wall_gpu_hours=0.31, measured_target_effect=0.0)]
    )
    resumed_loop, _ = _loop(
        tmp_path,
        second,
        policy=policy,
        parent=parent_manifest(tmp_path),
        profile=PROMOTED_AFTER_FORMATTING,
        generation="gen3",
        state=state,
    )
    report = resumed_loop.run(resume=True)

    assert len(second.calls) == 1
    assert second.calls[0].startswith("gen4-"), (
        "the resumed generation must train from the promoted parent, not gen2"
    )
    assert resumed_loop.parent_identity == IDENTITY
    assert len(report.generations) == 2
    assert report.budget["spent_wall_gpu_hours"] == pytest.approx(0.42 + 0.31)


def test_a_resumed_loop_handed_the_pre_promotion_profile_refuses(tmp_path: Path) -> None:
    """The profile is evidence about *a* generation, and the pairing is checked.

    A resumed loop restores the declaration the promotion produced (gen3 here),
    so a profile measured on the generation before it describes a different
    model. Choosing the next target from it would attribute gen3's weaknesses to
    gen2 -- and would do so invisibly, because both objects are individually
    well-formed. The loop stops instead, having launched nothing.
    """
    first = RecordingExecutor(
        [
            outcome(
                "PROMOTED",
                wall_gpu_hours=0.42,
                measured_target_effect=0.18,
                promoted_identity=IDENTITY,
                profile_scores=PROMOTED_AFTER_FORMATTING,
            )
        ]
    )
    parent = parent_manifest(tmp_path)
    policy = policy_from(parent, maximum_generations=1)
    loop, state = _loop(tmp_path, first, policy=policy, parent=parent)
    loop.run()
    (state.root / "stopping-state.json").unlink()

    second = RecordingExecutor(
        [outcome("REJECTED", wall_gpu_hours=0.31, measured_target_effect=0.0)]
    )
    # Deliberately the *pre-promotion* profile while resuming from the promotion.
    resumed, _ = _loop(
        tmp_path,
        second,
        policy=policy,
        parent=parent_manifest(tmp_path),
        profile=START,
        generation="gen2",
        state=state,
    )
    report = resumed.run(resume=True)

    assert second.calls == [], "nothing may be trained from a mislabelled profile"
    assert report.decision.action == STOP_UNCERTAIN
    assert PROFILE_GENERATION_MISMATCH in report.decision.reason_codes
    assert "gen3" in report.decision.reason and "gen2" in report.decision.reason


def test_resuming_a_promotion_whose_adapter_was_never_recorded_is_refused(
    tmp_path: Path,
) -> None:
    """Rows written before this pass carry no promoted identity: refuse them.

    Guessing which adapter such a row promoted is exactly the class of defect
    the integrity pass exists to remove, so the resume path refuses rather than
    training from an unidentified parent.
    """
    executor = RecordingExecutor([])
    loop, state = _loop(tmp_path, executor)
    state.record_intervention(
        target_skill="instruction.formatting",
        training_type="targeted_repair",
        cycle_id="gen3-a1-instruction-formatting",
        generation="gen3",
        cost_gpu_hours=0.4,
        candidate_result="PROMOTED",
        promotion_result="promoted",
        measured_effect=0.2,
    )

    report = loop.run(resume=True)

    assert executor.calls == []
    assert report.decision.action == STOP_UNCERTAIN
    assert PROMOTION_WITHOUT_IDENTITY in report.decision.reason_codes


def _run(**overrides):
    """A completed CampaignRun document, as the loop reads it back."""
    from chowder.growth.campaign_runner import CampaignRun

    document = {
        "cycle_id": "gen3-a1-instruction-formatting",
        "parent_version": "gen2",
        "candidate_version": "gen3",
        "verdict": "PROMOTED",
        "phases": (),
        "admission": (),
        "cost": {"wall_gpu_hours": 0.42, "device_gpu_hours": 0.4},
        "settlement": {},
        "ceiling_enforcement": {},
        "certification": {"status": "PASS", "reasons": ("all requirements held",)},
        "selection": {"artifact_ref": "adapters/gen3", "artifact_sha256": "a" * 64},
        "promotion": {
            "target_deltas": {"generation-diagnostics@gen2": 0.2},
            "protected_deltas": {"math500@2024-04": -0.01},
        },
        "record_path": "",
    }
    document.update(overrides)
    return CampaignRun(**document)


def test_a_run_without_a_measured_cost_reports_no_cost_rather_than_zero() -> None:
    """The seam must not turn an unmeasured campaign into a free one."""
    from chowder.growth.growth_loop import campaign_outcome

    outcome_ = campaign_outcome(_run(cost={"wall_gpu_hours": None}))

    assert outcome_.wall_gpu_hours is None
    assert outcome_.device_gpu_hours is None


def test_a_promoted_run_reports_the_adapter_it_actually_selected() -> None:
    """The next generation trains from the bytes this run promoted."""
    from chowder.growth.growth_loop import campaign_outcome

    outcome_ = campaign_outcome(_run())

    assert outcome_.parent_identity == ("adapters/gen3", "a" * 64)
    assert outcome_.wall_gpu_hours == pytest.approx(0.42)
    assert outcome_.measured_target_effect == pytest.approx(0.2)
    assert outcome_.regressions == ("math500@2024-04",)
    assert outcome_.promoted


def test_a_promotion_without_a_bound_artifact_has_no_identity() -> None:
    """A digest the run did not record cannot be supplied by the seam."""
    from chowder.growth.growth_loop import campaign_outcome

    assert campaign_outcome(_run(selection={})).parent_identity is None
    assert campaign_outcome(_run(selection={"artifact_ref": "a", "artifact_sha256": "short"})).parent_identity is None


def test_a_run_that_measured_no_profile_produces_none() -> None:
    """Absent evidence stays absent, so the loop cannot plan from nothing."""
    from chowder.growth.growth_loop import campaign_outcome

    assert campaign_outcome(_run()).profile is None


def test_plateau_requires_every_allowed_treatment_to_have_been_tried(
    tmp_path: Path,
) -> None:
    """The rule that makes scenario F honest, asserted on its own.

    A target is exhausted when none of the treatments the policy allows remains
    untried -- ``targeted_repair`` tried once is not evidence that ``sft`` would
    not work.
    """
    state = GrowthState(root=tmp_path / "state")
    state.record_intervention(
        target_skill="instruction.formatting",
        training_type="targeted_repair",
        cycle_id="gen3-a1-instruction-formatting",
        generation="gen3",
        cost_gpu_hours=0.4,
        measured_effect=0.0,
    )
    check: PlateauCheck = detect_plateau(
        state=state,
        target_skill="instruction.formatting",
        maximum_same_target_attempts=1,
        epsilon=0.01,
        allowed_treatments=("targeted_repair", "sft"),
    )
    assert not check.plateaued
    assert check.untried_treatments == ("sft",)

    state.record_intervention(
        target_skill="instruction.formatting",
        training_type="sft",
        cycle_id="gen3-a2-instruction-formatting",
        generation="gen3",
        cost_gpu_hours=0.4,
        measured_effect=0.0,
    )
    exhausted: PlateauCheck = detect_plateau(
        state=state,
        target_skill="instruction.formatting",
        maximum_same_target_attempts=1,
        epsilon=0.01,
        allowed_treatments=("targeted_repair", "sft"),
    )
    assert exhausted.plateaued
    assert exhausted.reason_code == "TARGET_EXHAUSTED"
