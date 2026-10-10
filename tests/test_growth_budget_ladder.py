"""The adaptive budget ladder: escalation earned by evidence, reordering
inside the declared envelope, never a budget change or an envelope change.

The claims under test: the stage a campaign stands on is the *policy's* cap
tightened by *evidence* (never silently widened), a prior of zero excludes
nobody (that is the maturity gate's decision), the adaptive stage's reward is
training-side efficiency only, and every reorder provably keeps the plan's
per-round candidate sets and totals -- so the ceilings plan_search checked
still hold verbatim.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from chowder.growth.budget_ladder import (
    BudgetLadderRefusal,
    LadderStage,
    allocate,
    entry_order,
    ladder_stage,
    survivor_orderer,
    ucb1_order,
)
from chowder.growth.candidate_search import (
    CandidateSearchRefusal,
    plan_search,
    run_search,
)
from chowder.growth.recipe_planner import TrainingRecipe

from test_growth_candidate_search import (
    _checkpoint_artifact,
    _declaration,
    _project_cost,
    _recipe,
)


def _records(*family_ids: str) -> list[dict[str, object]]:
    return [{"family_id": family_id, "state": "failed"} for family_id in family_ids]


# --------------------------------------------------------------------------
# the stage: policy is the cap, evidence is the climb
# --------------------------------------------------------------------------


class TestLadderStage:
    def test_default_policy_is_deterministic(self):
        stage, reasons = ladder_stage(
            policy={}, evidence_records=[], family_ids=["training.sft-curriculum"]
        )
        assert stage is LadderStage.DETERMINISTIC
        assert any("deterministic" in reason for reason in reasons)

    def test_unknown_policy_refuses(self):
        with pytest.raises(BudgetLadderRefusal, match="unknown allocation policy"):
            ladder_stage(
                policy={"budget_ladder": "greedy-bandit"},
                evidence_records=[],
                family_ids=["f"],
            )

    def test_policy_is_a_cap_even_with_evidence(self):
        stage, _ = ladder_stage(
            policy={"budget_ladder": "deterministic"},
            evidence_records=_records("f"),
            family_ids=["f"],
            completed_searches=99,
        )
        assert stage is LadderStage.DETERMINISTIC

    def test_prior_weighted_needs_history_for_every_family(self):
        stage, reasons = ladder_stage(
            policy={"budget_ladder": "prior-weighted"},
            evidence_records=_records("training.sft-curriculum"),
            family_ids=["training.sft-curriculum", "architecture.hybrid-lm"],
        )
        assert stage is LadderStage.DETERMINISTIC
        assert any("architecture.hybrid-lm" in reason for reason in reasons)

    def test_prior_weighted_escalates_when_all_families_have_history(self):
        stage, reasons = ladder_stage(
            policy={"budget_ladder": "prior-weighted"},
            evidence_records=_records("training.sft-curriculum", "architecture.hybrid-lm"),
            family_ids=["training.sft-curriculum", "architecture.hybrid-lm"],
        )
        assert stage is LadderStage.PRIOR_WEIGHTED
        assert any("evidence history" in reason for reason in reasons)

    def test_adaptive_needs_completed_searches(self):
        records = _records("f")
        stage, reasons = ladder_stage(
            policy={"budget_ladder": "adaptive"},
            evidence_records=records,
            family_ids=["f"],
            completed_searches=1,
        )
        assert stage is LadderStage.PRIOR_WEIGHTED
        assert any("completed searches" in reason for reason in reasons)

        stage, reasons = ladder_stage(
            policy={"budget_ladder": "adaptive"},
            evidence_records=records,
            family_ids=["f"],
            completed_searches=2,
        )
        assert stage is LadderStage.ADAPTIVE


# --------------------------------------------------------------------------
# entry order: priors reorder, they never exclude
# --------------------------------------------------------------------------


class TestEntryOrder:
    def test_higher_prior_enters_first_ties_break_by_id(self):
        assert entry_order(
            ["b", "a", "c"], priors={"a": 1.5, "b": 0.35, "c": 1.5}
        ) == ("a", "c", "b")

    def test_missing_prior_refuses(self):
        with pytest.raises(BudgetLadderRefusal, match="has no prior"):
            entry_order(["a", "b"], priors={"a": 1.0})

    def test_zero_prior_refuses_rather_than_dropping_the_candidate(self):
        # Exclusion is the maturity gate's decision; the ladder refuses to
        # make it silently through a back door.
        with pytest.raises(BudgetLadderRefusal, match="maturity gate"):
            entry_order(["a", "b"], priors={"a": 1.0, "b": 0.0})

    def test_non_finite_prior_refuses(self):
        with pytest.raises(BudgetLadderRefusal):
            entry_order(["a"], priors={"a": math.inf})


# --------------------------------------------------------------------------
# ucb1 order: explore-first, deterministic
# --------------------------------------------------------------------------


class TestUcb1Order:
    def test_unobserved_candidates_rank_first(self):
        order = ucb1_order(
            ["observed-good", "unseen", "observed-bad"],
            efficiency={
                "observed-good": (10.0, 50),
                "observed-bad": (0.1, 50),
            },
        )
        assert order[0] == "unseen"

    def test_higher_mean_ranks_first_with_equal_counts(self):
        order = ucb1_order(
            ["low", "high"],
            efficiency={"low": (1.0, 10), "high": (5.0, 10)},
        )
        assert order == ("high", "low")

    def test_uncertainty_bonus_can_overcome_a_mean_deficit(self):
        # One observation of 2.0 gets a big exploration bonus; 100
        # observations of 2.5 get almost none.
        order = ucb1_order(
            ["rare", "rich"],
            efficiency={"rare": (2.0, 1), "rich": (2.5, 100)},
        )
        assert order == ("rare", "rich")

    def test_ties_break_by_id(self):
        order = ucb1_order(
            ["b", "a"], efficiency={"a": (1.0, 5), "b": (1.0, 5)}
        )
        assert order == ("a", "b")


class TestSurvivorOrderer:
    def test_non_adaptive_stages_get_the_identity_order(self):
        orderer = survivor_orderer(stage=LadderStage.PRIOR_WEIGHTED, efficiency={})
        assert orderer(["c", "a", "b"], round_row=None) == ["c", "a", "b"]

    def test_adaptive_stage_orders_by_ucb1(self):
        orderer = survivor_orderer(
            stage=LadderStage.ADAPTIVE,
            efficiency={"a": (0.1, 10), "b": (9.0, 10)},
        )
        assert orderer(["a", "b"], round_row=None) == ["b", "a"]


# --------------------------------------------------------------------------
# allocate: reorder inside the plan, never rewrite it
# --------------------------------------------------------------------------


def _plan(recipes, declaration=None, **overrides):
    options = {
        "per_recipe_device_ceiling": 0.30,
        "per_recipe_wall_ceiling": 0.10,
        "campaign_device_ceiling": 0.60,
        "campaign_wall_ceiling": 0.20,
    }
    options.update(overrides)
    return plan_search(
        declaration or _declaration(),
        recipes=recipes,
        project_cost=_project_cost,
        **options,
    )


class TestAllocate:
    def _plan_and_inputs(self):
        recipes = [_recipe(rid) for rid in ("r-a", "r-b", "r-c", "r-d")]
        plan = _plan(recipes)
        priors = {"r-a": 0.35, "r-b": 1.5, "r-c": 1.0, "r-d": 0.5}
        efficiency = {"r-a": (0.1, 5), "r-b": (9.0, 5), "r-c": (5.0, 5), "r-d": (2.0, 5)}
        return plan, priors, efficiency

    def test_deterministic_returns_the_plan_it_was_given(self):
        plan, _, _ = self._plan_and_inputs()
        reordered, receipt = allocate(plan, stage=LadderStage.DETERMINISTIC, priors={})
        assert [row.recipe_ids for row in reordered.rounds] == [
            row.recipe_ids for row in plan.rounds
        ]
        assert receipt.stage is LadderStage.DETERMINISTIC

    def test_prior_weighted_reorders_round0_only_and_keeps_everything_else(self):
        plan, priors, _ = self._plan_and_inputs()
        reordered, receipt = allocate(
            plan, stage=LadderStage.PRIOR_WEIGHTED, priors=priors
        )
        assert reordered.rounds[0].recipe_ids == ("r-b", "r-c", "r-d", "r-a")
        assert reordered.rounds[1].recipe_ids == plan.rounds[1].recipe_ids
        # The envelope is untouched: same sets, same totals, same schedule.
        for new, old in zip(reordered.rounds, plan.rounds):
            assert set(new.recipe_ids) == set(old.recipe_ids)
            assert new.projected_device_gpu_hours == old.projected_device_gpu_hours
        assert reordered.total_device_gpu_hours == plan.total_device_gpu_hours
        assert reordered.total_wall_gpu_hours == plan.total_wall_gpu_hours
        assert dict(reordered.schedule) == dict(plan.schedule)
        assert receipt.entry_recipe_ids == ("r-b", "r-c", "r-d", "r-a")

    def test_adaptive_reorders_later_rounds_by_efficiency(self):
        plan, priors, efficiency = self._plan_and_inputs()
        reordered, receipt = allocate(
            plan,
            stage=LadderStage.ADAPTIVE,
            priors=priors,
            efficiency=efficiency,
        )
        assert reordered.rounds[0].recipe_ids == ("r-b", "r-c", "r-d", "r-a")
        # Round 1 keeps its declared set but runs in UCB1 order.
        assert set(reordered.rounds[1].recipe_ids) == set(plan.rounds[1].recipe_ids)
        ids = list(reordered.rounds[1].recipe_ids)
        assert ucb1_order(ids, efficiency=efficiency) == tuple(ids)
        assert receipt.stage is LadderStage.ADAPTIVE

    def test_adaptive_without_observations_refuses(self):
        plan, priors, _ = self._plan_and_inputs()
        with pytest.raises(BudgetLadderRefusal, match="no observations were given"):
            allocate(plan, stage=LadderStage.ADAPTIVE, priors=priors)

    def test_undeclared_plan_refuses(self):
        from chowder.growth.candidate_search import CandidateSearchDeclaration

        empty = plan_search(
            CandidateSearchDeclaration(),
            recipes=[_recipe("r-a")],
            project_cost=_project_cost,
            per_recipe_device_ceiling=0.3,
            per_recipe_wall_ceiling=0.1,
            campaign_device_ceiling=0.6,
            campaign_wall_ceiling=0.2,
        )
        with pytest.raises(BudgetLadderRefusal, match="nothing to allocate"):
            allocate(
                empty, stage=LadderStage.PRIOR_WEIGHTED, priors={"r-a": 1.0}
            )

    def test_reallocate_the_reordered_plan_again_is_a_fixed_point(self):
        plan, priors, efficiency = self._plan_and_inputs()
        once, _ = allocate(
            plan, stage=LadderStage.ADAPTIVE, priors=priors, efficiency=efficiency
        )
        twice, _ = allocate(
            once, stage=LadderStage.ADAPTIVE, priors=priors, efficiency=efficiency
        )
        assert [row.recipe_ids for row in twice.rounds] == [
            row.recipe_ids for row in once.rounds
        ]


# --------------------------------------------------------------------------
# integration: the ordering hook decides who a binding cut keeps
# --------------------------------------------------------------------------


def _recording_run_attempt(root: Path, order_log: list[str]):
    def run_attempt(recipe: TrainingRecipe) -> dict[str, object]:
        order_log.append(recipe.recipe_id)
        return {
            "status": "SUCCEEDED",
            "candidate_succeeded": True,
            "artifact_ref": _checkpoint_artifact(root, recipe.recipe_id),
            "gpu_hours": 0.01,
        }

    return run_attempt


class TestRunSearchIntegration:
    def test_survivor_order_hook_decides_who_a_binding_cut_keeps(
        self, tmp_path: Path
    ) -> None:
        recipes = [_recipe(rid) for rid in ("r-a", "r-b", "r-c", "r-d")]
        declaration = _declaration()
        plan = _plan(recipes, declaration)
        order_log: list[str] = []

        # Default order: r-a and r-b run first and survive the round-0 cut;
        # the final round's own 2->1 cut keeps the first of them.
        default_run = run_search(
            plan,
            declaration=declaration,
            recipes=recipes,
            project_cost=_project_cost,
            run_attempt=_recording_run_attempt(tmp_path, order_log),
        )
        assert [row["recipe_id"] for row in default_run.round_attempts[-1]] == [
            "r-a",
            "r-b",
        ]
        assert default_run.survivors == ("r-a",)

        # The hook runs r-c and r-d first, so the same cut keeps *them*.
        adaptive_hook = survivor_orderer(
            stage=LadderStage.ADAPTIVE,
            efficiency={
                "r-a": (0.1, 5), "r-b": (0.2, 5),
                "r-c": (9.0, 5), "r-d": (8.0, 5),
            },
        )
        order_log.clear()
        adaptive_run = run_search(
            plan,
            declaration=declaration,
            recipes=recipes,
            project_cost=_project_cost,
            run_attempt=_recording_run_attempt(tmp_path, order_log),
            order_survivors=adaptive_hook,
        )
        assert adaptive_run.survivors == ("r-c",)
        # Round 1 ran the survivors the hook's order chose.
        assert [row["recipe_id"] for row in adaptive_run.round_attempts[-1]] == [
            "r-c",
            "r-d",
        ]

    def test_a_hook_that_drops_a_candidate_refuses(self, tmp_path: Path) -> None:
        recipes = [_recipe(rid) for rid in ("r-a", "r-b", "r-c", "r-d")]
        declaration = _declaration()
        plan = _plan(recipes, declaration)
        order_log: list[str] = []

        with pytest.raises(CandidateSearchRefusal, match="never rewrite who was admitted"):
            run_search(
                plan,
                declaration=declaration,
                recipes=recipes,
                project_cost=_project_cost,
                run_attempt=_recording_run_attempt(tmp_path, order_log),
                # Drop r-b: admission is the plan's decision, not the hook's.
                order_survivors=lambda ids, row: [rid for rid in ids if rid != "r-b"],
            )
