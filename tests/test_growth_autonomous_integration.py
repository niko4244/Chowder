"""The autonomous path on production seams: does it actually reach READY?

The unit suites prove each component. This one asks the question the mission
asks -- *can the loop compose, plan, freeze and gate a real campaign without a
human, and does its preview match what a run would do* -- using the real pieces:

* the real :class:`NextCampaignBuilder` (draft, then freeze);
* the real :func:`prepare_campaign`, so the real curriculum planner, the real
  :class:`RecipePlanner`, the corpus, the registry, the template and the
  contamination manifest are all the production ones;
* the real :func:`check_campaign_readiness`, in full, with no check stubbed out.

Only two things are stood in for, and both are *measurements* rather than
compute: the device probe (a declared measured hardware budget) and the
benchmark slice source (deterministic 16-item slices). The training and
evaluation workers are never invoked, because nothing here runs a campaign.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chowder.evals.result import MEASURED_PARENT, BenchmarkRun, EvalReport
from chowder.growth.campaign import CampaignManifest
from chowder.growth.growth_loop import GrowthLoop, PreparedAttempt, production_prepare
from chowder.growth.next_campaign import LoopPolicy
from chowder.growth.service import AutonomousGrowthService, GrowthServiceRefusal
from chowder.growth.target_selection import GrowthState, build_skill_profile

from test_growth_campaign_prepare import (
    BROAD_ID,
    PROTECTED_ID,
    TARGET_ID,
    _manifest_document,
    _parent_evidence,
    _probe,
    _slice_source,
)
from fixtures_growth_loop import RecordingExecutor, planned_recipes, policy_from

def _device_can_be_measured() -> bool:
    """Whether preparation's hardware budget can be a *real* measurement here.

    The tests below drive preparation through the production service, which
    probes the local accelerator for its measured step timings and **refuses**
    when there is nothing to probe. Handing it a fabricated budget would be the
    same dishonesty this mission exists to remove, so on a machine (or CI runner)
    with no CUDA device these two tests skip rather than weaken: nothing is lost,
    because a campaign cannot run on such a machine either.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 - any import failure means no probe
        return False
    return bool(torch.cuda.is_available())


#: Preparation measures the device. No device, no measurement, no test.
_NEEDS_A_MEASURED_DEVICE = pytest.mark.skipif(
    not _device_can_be_measured(),
    reason=(
        "preparation's hardware budget is a real device probe; no CUDA device is "
        "available here, so the production service refuses by design"
    ),
)


#: The generation the parent declaration *produced*. A declaration states the
#: generation it advances from and the generation it produces; the loop stands on
#: the one it produced, so this fixture's parent is gen1 and the next is gen2.
PARENT_GENERATION = "gen1"


def _arm_row(qualified_id: str, score: float) -> BenchmarkRun:
    return BenchmarkRun(
        benchmark_qualified_id=qualified_id,
        adapter="chowder_custom",
        generation_version=PARENT_GENERATION,
        score=score,
        n_samples=16,
        per_sample_scores=(float(score),) * 16,
        measurement_origin=MEASURED_PARENT,
    )


def _ancestor_arm(tmp_path: Path) -> Path:
    """The trusted-ancestor (gen0) arm the declaration must name."""
    path = tmp_path / "ancestor" / "baseline-eval-report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    EvalReport(
        generation_version="gen0",
        runs=(_arm_row(PROTECTED_ID, 0.0), _arm_row(BROAD_ID, 0.0)),
    ).save(path)
    return path


def _production_loop(tmp_path: Path) -> tuple[GrowthLoop, GrowthState, Path]:
    """A loop standing on a real, readable parent generation, on real seams.

    The declaration is the parent's own -- ``parent_version`` names the
    generation before it and ``state_root`` is the run root that generation's
    evidence lives in -- so the lineage the loop starts from is stated rather
    than discovered, which is the property ``ParentEvidenceRef`` exists for.
    """
    parent_root = _parent_evidence(tmp_path / PARENT_GENERATION)
    document = _manifest_document(tmp_path)
    document["parent_version"] = "gen0"
    document["state_root"] = str(parent_root)
    document["baseline_eval_report_path"] = str(_ancestor_arm(tmp_path))
    manifest_path = tmp_path / "parent-declaration.json"
    manifest_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    parent = CampaignManifest.from_file(manifest_path)

    profile = build_skill_profile(
        generation=PARENT_GENERATION, runs=(_arm_row(TARGET_ID, 0.9375),)
    )
    # Written where the service reads it: the parent's attributed profile is a
    # document, and a service that cannot find it refuses rather than guessing.
    profile_path = tmp_path / "parent-skill-profile.json"
    profile_path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    state = GrowthState(root=tmp_path / "growth-state")
    loop = GrowthLoop(
        policy=policy_from(parent),
        state=state,
        executor=RecordingExecutor([]),
        parent_declaration=parent,
        parent_profile=profile,
        # The real preparation, with only its two measurement seams supplied.
        prepare=production_prepare(probe=_probe, slice_source=_slice_source()),
    )
    return loop, state, manifest_path, profile_path


def test_the_autonomous_loop_reaches_ready_on_production_seams(tmp_path: Path) -> None:
    """Compose, plan, freeze, gate -- with no check replaced by a lambda."""
    loop, _state, _path, _profile_path = _production_loop(tmp_path)

    prepared = loop.prepare_next()
    assert isinstance(prepared, PreparedAttempt), prepared

    # The frozen declaration names exactly the recipe set the production planner
    # proposed -- the deadlock this pass removed, proven on the real planner.
    declared = set(prepared.frozen.manifest.recipe_ids)
    assert declared == set(prepared.recipe_ids)
    assert prepared.recipe_ids, "the planner proposed nothing"
    assert all(
        rid.startswith("recipe-") for rid in prepared.recipe_ids
    ), "production planner ids, not the draft's placeholders"

    report = loop.readiness_of(prepared.frozen)
    refused = [c for c in report.checks if c.status == "refused"]
    skipped = [c for c in report.checks if c.status == "skipped"]
    assert report.ready, (
        f"readiness refused {[(c.check, c.detail) for c in refused]} and skipped "
        f"{[(c.check, c.detail) for c in skipped]}"
    )
    # No check may be silently skipped: a skipped prerequisite in a READY report
    # is a prerequisite nobody verified.
    assert not skipped

    # The freeze is write-once and the declaration is the one on disk.
    assert prepared.frozen.manifest_path.is_file()
    assert prepared.frozen.preregistration_path.is_file()


def test_the_prepared_profile_keeps_a_strong_skill_strong_and_an_unmeasured_one_unknown(
    tmp_path: Path,
) -> None:
    """The attribution invariant, through the production preparation path."""
    loop, _state, _path, _profile_path = _production_loop(tmp_path)
    profile = loop.profile

    assert profile is not None
    formatting = profile.for_skill("instruction.formatting")
    coding = profile.for_skill("coding.generation")
    assert formatting is not None and formatting.estimate is not None
    assert formatting.estimate > 0.5, "instruction.formatting was measured strong"
    assert coding is not None and coding.estimate is None
    assert coding.confidence == 0.0


@_NEEDS_A_MEASURED_DEVICE
def test_the_service_previews_exactly_what_the_loop_would_run(tmp_path: Path) -> None:
    """One decision engine: service, CLI and loop cannot disagree."""
    loop, state, manifest_path, profile_path = _production_loop(tmp_path)
    policy_path = tmp_path / "loop-policy.json"
    policy_path.write_text(
        json.dumps(loop.policy.to_dict(), indent=2), encoding="utf-8"
    )
    service = AutonomousGrowthService.open_from_paths(
        policy_path=policy_path,
        parent_declaration_path=manifest_path,
        state_root=state.root,
        parent_profile_path=profile_path,
        executor=RecordingExecutor([]),
    )

    planned = service.plan_next()
    assert not planned.decided
    assert planned.proposal is not None
    assert planned.proposal["target_skill"] == "instruction.formatting"

    # The lineage the UI shows is the ref's, field for field.
    lineage = service.inspect()
    assert lineage.generation == PARENT_GENERATION
    assert lineage.run_root == str(tmp_path / PARENT_GENERATION)
    assert lineage.trusted_ancestor == "gen0"
    assert "math.competition" in lineage.protected_skills
    assert "instruction.formatting" in lineage.profile_measured_skills

    view = service.prepare_next()
    assert view.attempt is not None
    assert not view.refused
    assert [c.name for c in view.checks], "readiness reported no checks"
    assert any(c.status == "ok" for c in view.checks)


def test_the_service_refuses_a_session_it_cannot_measure(tmp_path: Path) -> None:
    """Refused before any durable state exists: a refusal starts no session."""
    loop, _state, manifest_path, _profile_path = _production_loop(tmp_path)
    policy_path = tmp_path / "loop-policy.json"
    policy_path.write_text(json.dumps(loop.policy.to_dict()), encoding="utf-8")
    state_root = tmp_path / "refused-state"

    try:
        AutonomousGrowthService.open_from_paths(
            policy_path=policy_path,
            parent_declaration_path=manifest_path,
            state_root=state_root,
            require_profile=True,
        )
    except GrowthServiceRefusal as error:
        assert "SERVICE_PROFILE_ABSENT" in str(error)
    else:  # pragma: no cover - the refusal is the assertion
        raise AssertionError("a session with no measured parent must not open")

    assert not state_root.exists(), "a refused session must leave no state behind"


def test_the_operator_stop_is_durable_and_stops_before_the_next_generation(
    tmp_path: Path,
) -> None:
    """STOP AFTER CURRENT CAMPAIGN, honestly bounded by a generation boundary."""
    loop, state, _path, _profile_path = _production_loop(tmp_path)
    from fixtures_growth_loop import outcome

    first = RecordingExecutor(
        [
            outcome(
                "PROMOTED",
                wall_gpu_hours=0.1,
                promoted_identity=("adapters/gen2", "e" * 64),
                profile_scores={"instruction.formatting": 0.99},
            ),
            outcome("REJECTED", wall_gpu_hours=0.1),
        ]
    )
    loop.executor = first
    state.request_stop(reason="the operator closed the laptop")

    report = loop.run()

    assert report.decision.action == "STOP_OPERATOR"
    assert "OPERATOR_STOP_REQUESTED" in report.decision.reason_codes
    assert first.calls == [], "a stop requested before generation 1 starts nothing"
    assert state.operator_stop() is not None

    # Resume clears the request, so a stop cannot wedge the session forever.
    resumed_executor = RecordingExecutor(
        [outcome("REJECTED", wall_gpu_hours=0.1, profile_scores={"instruction.formatting": 0.31})]
    )
    loop.executor = resumed_executor
    resumed = loop.run(max_generations=1, resume=True)
    assert state.operator_stop() is None
    assert resumed.decision.action != "STOP_OPERATOR"


def test_history_does_not_report_a_rejected_generation_as_the_current_model(
    tmp_path: Path,
) -> None:
    """A rejection must not look like the lineage advanced.

    The reporting rule is asserted against a durable record directly, because
    the loop's *production* preparation refuses a promotion whose run root holds
    no measured arm -- which is right, and is why a fabricated two-generation
    history is not a shape a real run can produce.
    """
    from chowder.growth.service import history_from_state

    state = GrowthState(root=tmp_path / "history-state")
    state.record_intervention(
        target_skill="instruction.formatting",
        training_type="targeted_repair",
        cycle_id="gen2-a1-instruction-formatting",
        generation="gen2",
        parent_version="gen1",
        cost_gpu_hours=0.1,
        candidate_result="PROMOTED",
        promotion_result="promoted",
    )
    state.record_intervention(
        target_skill="instruction.formatting",
        training_type="targeted_repair",
        cycle_id="gen2-a2-instruction-formatting",
        generation="gen3",
        parent_version="gen2",
        cost_gpu_hours=0.2,
        candidate_result="REJECTED",
        promotion_result="not_promoted",
        measured_effect=-0.01,
    )

    rows = history_from_state(state)
    assert len(rows) == 2
    assert rows[0].promoted and rows[0].effective_generation == "gen2"
    assert not rows[1].promoted
    assert rows[1].generation == "gen3"
    assert rows[1].effective_generation == "gen2", (
        "a rejected generation leaves the previous generation current"
    )


def test_the_loop_refuses_to_plan_from_a_profile_of_another_generation(
    tmp_path: Path,
) -> None:
    """The defect from PR #196, still enforced after this pass's refactor."""
    loop, _state, _path, _profile_path = _production_loop(tmp_path)
    from chowder.growth.simulator import skill_profile

    loop._profile = skill_profile({"instruction.formatting": 0.31}, generation="gen0")
    planned = loop.plan_next()

    assert not isinstance(planned, PreparedAttempt)
    assert planned.action == "STOP_UNCERTAIN"
    assert "PROFILE_GENERATION_MISMATCH" in planned.reason_codes


@_NEEDS_A_MEASURED_DEVICE
def test_a_frozen_declaration_is_never_rewritten_through_the_service(
    tmp_path: Path,
) -> None:
    """Write-once, proven through the object a UI drives."""
    loop, state, manifest_path, profile_path = _production_loop(tmp_path)
    policy_path = tmp_path / "loop-policy.json"
    policy_path.write_text(json.dumps(loop.policy.to_dict()), encoding="utf-8")
    service = AutonomousGrowthService.open_from_paths(
        policy_path=policy_path,
        parent_declaration_path=manifest_path,
        state_root=state.root,
        parent_profile_path=profile_path,
        executor=RecordingExecutor([]),
    )

    first = service.prepare_next()
    assert first.attempt is not None
    frozen_bytes = first.attempt.frozen.manifest_path.read_bytes()

    second = service.prepare_next()
    # The same attempt collides with its own frozen declaration rather than
    # overwriting it: a preregistration that can be rewritten is not one.
    assert second.decision is not None or second.attempt is not None
    assert first.attempt.frozen.manifest_path.read_bytes() == frozen_bytes


def test_a_policy_that_does_not_cover_every_declared_benchmark_refuses(
    tmp_path: Path,
) -> None:
    """An unresolvable protected benchmark refuses rather than protecting nothing."""
    loop, _state, _path, _profile_path = _production_loop(tmp_path)
    policy_document = loop.policy.to_dict()
    policy_document["protected_benchmarks"] = ["not-a-real-benchmark@v0"]

    try:
        GrowthLoop(
            policy=LoopPolicy.from_mapping(policy_document),
            state=GrowthState(root=tmp_path / "s2"),
            executor=RecordingExecutor([]),
            parent_declaration=loop.parent_declaration,
            prepare=planned_recipes,
        )
    except ValueError as error:
        assert "not in the benchmark registry" in str(error)
    else:  # pragma: no cover - the refusal is the assertion
        raise AssertionError("an unresolvable protected set must refuse")
