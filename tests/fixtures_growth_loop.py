"""Shared fixtures for the autonomous growth loop's tests.

The fixtures are deliberately built from the *checked-in* Gen-2 declaration
rather than from literals. The loop's policy has to agree with the declaration a
real campaign will run under -- protected sets, protocol, execution throughput,
budget shape and trusted ancestor all come from one place -- so a fixture that
invented its own limits would prove the loop works on limits nobody will use.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from chowder.growth.campaign import CampaignManifest
from chowder.growth.growth_loop import CampaignOutcome
from chowder.growth.next_campaign import LoopPolicy
from chowder.growth.simulator import DEFAULT_PARENT_DECLARATION

#: The declaration Gen-2 will actually run under; the same file the simulator
#: takes its policy from, so the two cannot drift into proving different things.
GEN2_MANIFEST = DEFAULT_PARENT_DECLARATION

#: An outcome run root that says "this campaign named none". The recording
#: executor fills the real root in otherwise, because a real executor reports
#: where it wrote -- so a test that wants the *absent* case must say so rather
#: than rely on an empty string, which is also the default.
NO_RUN_ROOT = "<campaign-named-no-run-root>"


def parent_manifest(tmp_path: Path, **overrides: Any) -> CampaignManifest:
    """The real Gen-2 declaration, with its run root relocated into a tmp dir."""
    document = json.loads(GEN2_MANIFEST.read_text(encoding="utf-8"))
    document["state_root"] = str(Path(tmp_path) / "parent-run")
    document.update(overrides)
    return CampaignManifest.from_mapping(document, source="test-parent")


def policy_from(parent: CampaignManifest, **overrides: Any) -> LoopPolicy:
    """An immutable loop policy that agrees with ``parent`` unless overridden."""
    document: dict[str, Any] = {
        "maximum_generations": 3,
        "maximum_total_wall_gpu_hours": 6.0,
        "maximum_consecutive_non_promotions": 2,
        "maximum_same_target_attempts": 2,
        "maximum_candidates": len(parent.recipe_ids),
        "plateau_epsilon": 0.01,
        "allowed_training_types": ["targeted_repair", "sft"],
        "protected_benchmarks": list(parent.protected_benchmarks),
        "broad_benchmarks": list(parent.broad_benchmarks),
        "calibration_benchmarks": list(parent.calibration_benchmarks),
        "reliability_benchmarks": list(parent.reliability_benchmarks),
        "campaign_budget": {
            "device_gpu_hours_ceiling_per_recipe": parent.budget.device_gpu_hours_ceiling_per_recipe,
            "wall_gpu_hours_ceiling_per_recipe": parent.budget.wall_gpu_hours_ceiling_per_recipe,
            "device_gpu_hours_ceiling_campaign": parent.budget.device_gpu_hours_ceiling_campaign,
            "wall_gpu_hours_ceiling_campaign": parent.budget.wall_gpu_hours_ceiling_campaign,
            "device_time_measured": parent.budget.device_time_measured,
        },
        "protection": parent.protection.to_dict(),
        "evaluation_execution": parent.evaluation_execution.to_dict(),
        "candidate_selection_policy": parent.candidate_selection_policy,
        "stopping_rules": list(parent.stopping_rules),
        "promotion_policy_version": parent.promotion_policy_version,
        "human_review_triggers": ["architecture change"],
    }
    document.update(overrides)
    return LoopPolicy.from_mapping(document, source="test-policy")


def _measured_generation(frozen: Any, verdict: str) -> str:  # noqa: ANN401 - the seam
    """Which model a served outcome's capability profile is a measurement of.

    The same rule the production executor obeys by construction, because it
    derives the profile from the run root of the artifact the run selected: a
    promotion leaves the candidate it trained in the lineage, a rejection leaves
    the parent it started from. Stated once here so a test cannot accidentally
    attribute a measurement to whichever generation makes its assertion pass.
    """
    if str(verdict).upper() == "PROMOTED":
        return str(frozen.candidate_version)
    return str(frozen.manifest.parent_version)


class RecordingExecutor:
    """Serves declared outcomes in order and refuses to invent another one.

    Recording the cycle ids it was asked for is the point: "no training was
    launched" and "training was launched exactly once for this attempt" are
    properties of the loop, and they can only be checked by watching the seam
    the loop spends through.
    """

    def __init__(self, outcomes: Sequence[CampaignOutcome]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[str] = []

    def __call__(self, frozen: Any) -> CampaignOutcome:  # noqa: ANN401 - the seam
        if len(self.calls) >= len(self.outcomes):
            raise AssertionError(
                f"the loop asked for a {len(self.calls) + 1}th campaign and this "
                f"test declares {len(self.outcomes)}"
            )
        self.calls.append(frozen.cycle_id)
        served = self.outcomes[len(self.calls) - 1]
        # A real executor reports the run root it wrote, and the loop refuses a
        # promotion it cannot locate; a fixture that left it empty would be
        # modelling a campaign that cannot name its own evidence. An outcome
        # that names a root of its own keeps it, so a test can model the
        # promotion-without-a-locatable-run case deliberately.
        served = replace(
            served,
            run_root="" if served.run_root == NO_RUN_ROOT else str(frozen.directory),
        )
        if served.profile is None:
            return served
        generation = _measured_generation(frozen, served.verdict)
        return replace(served, profile={**dict(served.profile), "generation": generation})


def planned_recipes(draft: Any) -> Any:  # noqa: ANN401 - a CampaignDraft
    """A recording stand-in for preparation that *does* report its recipe set.

    Preparation is the phase that runs the production planner, so the loop
    freezes whatever recipe identities it reports. A no-op seam would leave the
    loop unable to freeze anything at all (which is the honest production
    behaviour), so a loop test that wants a campaign to run must say what the
    planner proposed.
    """
    from chowder.growth.growth_loop import PreparationResult

    return PreparationResult(
        recipe_ids=tuple(
            f"recipe-{index:02d}-lr0.0001-r16-replay0.1"
            for index in range(len(draft.placeholder_recipe_ids))
        ),
        detail="recording preparation",
    )


def outcome(
    verdict: str,
    *,
    wall_gpu_hours: float | None,
    measured_target_effect: float | None = None,
    promoted_identity: tuple[str, str] | None = None,
    profile_scores: dict[str, float] | None = None,
    run_root: str | None = None,
) -> CampaignOutcome:
    """One campaign's reported result, shaped like the production executor's."""
    from chowder.growth.simulator import skill_profile

    return CampaignOutcome(
        verdict=verdict,
        wall_gpu_hours=wall_gpu_hours,
        device_gpu_hours=None,
        measured_target_effect=measured_target_effect,
        parent_identity=promoted_identity,
        profile=skill_profile(profile_scores).to_dict() if profile_scores else None,
        failures=(),
        regressions=(),
        run_root=run_root if run_root is not None else "",
        reason=verdict,
    )
