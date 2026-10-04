"""Bounded candidate search: declared, projected, screened, and refused.

The claims under test are the mission's: the schedule is one owner (the
successive-halving policy, not a second copy of its arithmetic), the screen is
training-side only (no benchmark score of any kind can advance a candidate), and
the cost is bounded *before* it is spent -- an over-envelope search refuses
rather than discovering the overrun when the actuals land.

The recipe ids here are shape-accurate stand-ins; the run-level tests in
``test_growth_campaign_runner`` exercise the same code through the real
manifest, executor and CLI surfaces.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from chowder.growth.candidate_search import (
    SEARCH_SCHEMA,
    CandidateSearchDeclaration,
    CandidateSearchRefusal,
    SearchProgress,
    advanced,
    plan_search,
    round_recipe,
    run_search,
)
from chowder.growth.recipe_planner import TrainingRecipe

#: A projection shaped like the planner's own: `load + steps x step_cost`.
STEP_COST = 0.004
LOAD_SECONDS = 5.0
WALL_MULTIPLIER = 3.5


def _project_cost(*, seq_len: int, max_steps: int) -> tuple[float, float]:
    device = (max_steps * STEP_COST + LOAD_SECONDS) / 3600.0
    return (device, device * WALL_MULTIPLIER)


def _recipe(recipe_id: str, *, max_steps: int = 20, lr: float = 1e-4) -> TrainingRecipe:
    device, wall = _project_cost(seq_len=2048, max_steps=max_steps)
    return TrainingRecipe(
        recipe_id=recipe_id,
        curriculum_item_ids=("item-1",),
        mixture={"TARGET": 1.0},
        learning_rate=lr,
        scheduler="cosine",
        warmup_steps=2,
        lora_rank=16,
        lora_alpha=32,
        target_modules=("q_proj", "v_proj"),
        seq_len=2048,
        batch_size=2,
        gradient_accumulation=4,
        max_steps=max_steps,
        objective="sft",
        replay_rate=0.1,
        dataset_manifest={},
        projected_device_gpu_hours=device,
        projected_wall_gpu_hours=wall,
    )


def _declaration(**overrides: object) -> CandidateSearchDeclaration:
    base = {
        "rounds": 2,
        "initial_max_steps": 12,
        "step_multiplier": 2.0,
        "survival_fraction": 0.5,
        "min_survivors": 1,
        "device_gpu_hours_ceiling": 0.30,
        "wall_gpu_hours_ceiling": 0.20,
    }
    base.update(overrides)
    return CandidateSearchDeclaration(**base)


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


# --------------------------------------------------------------------------
# the declaration: absent means no search, and a declared one is bounded
# --------------------------------------------------------------------------


def test_an_absent_search_is_no_search_at_all() -> None:
    declaration = CandidateSearchDeclaration()
    assert declaration.declared is False
    plan = _plan([_recipe("a")], declaration)
    assert plan.declared is False
    assert plan.rounds == ()
    assert plan.total_device_gpu_hours == 0.0


def test_a_declared_search_without_an_envelope_refuses() -> None:
    with pytest.raises(CandidateSearchRefusal) as error:
        _declaration(device_gpu_hours_ceiling=0.0)
    assert SEARCH_SCHEMA in str(error.value)
    assert "unbounded allocation" in str(error.value)


def test_an_unknown_search_key_refuses_rather_than_being_ignored() -> None:
    with pytest.raises(CandidateSearchRefusal) as error:
        CandidateSearchDeclaration.from_mapping({"rounds": 1, "scheduler": "cosine"})
    assert "unknown candidate_search fields" in str(error.value)
    assert "scheduler" in str(error.value)


@pytest.mark.parametrize(
    "overrides",
    [
        {"rounds": -1},
        {"initial_max_steps": 0},
        {"step_multiplier": 1.0},
        {"survival_fraction": 1.0},
        {"min_survivors": 0},
        {"wall_gpu_hours_ceiling": float("inf")},
    ],
)
def test_every_declared_bound_is_validated(overrides: dict[str, object]) -> None:
    with pytest.raises(CandidateSearchRefusal) as error:
        _declaration(**overrides)
    assert SEARCH_SCHEMA in str(error.value)


def test_the_schedule_owns_the_round_budget_and_the_survivor_rule() -> None:
    """One owner: the search asks the halving policy, it does not restate it."""
    declaration = _declaration(rounds=3, initial_max_steps=12, step_multiplier=2.0)
    schedule = declaration.schedule()
    assert [schedule.round_max_steps(r) for r in range(3)] == [12, 24, 48]
    assert [schedule.survivors(n) for n in (4, 2, 1)] == [2, 1, 1]


# --------------------------------------------------------------------------
# the plan: projected before it is spent, and refused when it does not fit
# --------------------------------------------------------------------------


def test_the_plan_projects_every_round_and_the_worst_case_total() -> None:
    plan = _plan([_recipe("a"), _recipe("b")])
    assert [row.round_index for row in plan.rounds] == [0, 1]
    assert [row.max_steps for row in plan.rounds] == [12, 24]
    # Worst case assumes nobody is screened out: 2 candidates, then 1.
    assert [row.recipe_ids for row in plan.rounds] == [("a", "b"), ("a",)]
    assert plan.total_device_gpu_hours == pytest.approx(
        sum(row.projected_device_gpu_hours for row in plan.rounds)
    )
    assert plan.total_wall_gpu_hours == pytest.approx(
        sum(row.projected_wall_gpu_hours for row in plan.rounds)
    )
    assert plan.schedule["rounds"] == 2


def test_a_round_the_executor_would_refuse_is_refused_before_compute() -> None:
    """A round over the per-recipe ceilings is not a round: the search refuses."""
    with pytest.raises(CandidateSearchRefusal) as error:
        _plan([_recipe("a")], per_recipe_device_ceiling=1e-9)
    assert SEARCH_SCHEMA in str(error.value)
    assert "per-recipe ceilings" in str(error.value)


def test_a_search_over_its_own_declared_envelope_refuses() -> None:
    with pytest.raises(CandidateSearchRefusal) as error:
        _plan([_recipe("a"), _recipe("b")], _declaration(device_gpu_hours_ceiling=1e-9))
    assert "its own declared ceiling" in str(error.value)


def test_a_search_over_the_campaign_ceiling_refuses() -> None:
    with pytest.raises(CandidateSearchRefusal) as error:
        _plan([_recipe("a")], campaign_wall_ceiling=1e-9)
    assert "the campaign ceiling" in str(error.value)


def test_a_declared_search_with_no_candidates_refuses() -> None:
    with pytest.raises(CandidateSearchRefusal) as error:
        _plan([])
    assert "no candidate to run" in str(error.value)


def test_round_recipe_changes_only_the_budget() -> None:
    recipe = _recipe("a", lr=2e-4)
    device, wall = _project_cost(seq_len=2048, max_steps=48)
    survivor = round_recipe(recipe, max_steps=48, device=device, wall=wall)
    assert survivor.recipe_id == recipe.recipe_id
    assert survivor.learning_rate == recipe.learning_rate
    assert survivor.lora_rank == recipe.lora_rank
    assert survivor.max_steps == 48
    assert survivor.projected_device_gpu_hours == pytest.approx(device)
    assert "search round budget 48 steps" in survivor.notes


# --------------------------------------------------------------------------
# the screen: training-side evidence only
# --------------------------------------------------------------------------


def _attempt(recipe_id: str, *, succeeded: bool = True, **extra: object) -> dict:
    row: dict = {
        "recipe_id": recipe_id,
        "status": "SUCCEEDED" if succeeded else "REFUSED",
        "candidate_succeeded": succeeded,
        "artifact_ref": f"/attempts/{recipe_id}/adapter" if succeeded else None,
    }
    row.update(extra)
    return row


def _checkpoint_artifact(root, recipe_id: str, *, step: int = 12) -> str:
    """A fake attempt artifact carrying the real trainer checkpoint layout
    (``trainer/checkpoint-N``), so the search's checkpoint resolution runs the
    same code path it runs against real training output."""
    artifact = root / f"attempts-{recipe_id}" / "adapter"
    (artifact / "trainer" / f"checkpoint-{step}").mkdir(parents=True, exist_ok=True)
    return str(artifact)


def test_only_attempts_that_trained_and_produced_an_artifact_advance() -> None:
    results = [
        _attempt("a", succeeded=False),
        _attempt("b"),
        _attempt("c"),
    ]
    assert advanced(results, survivor_count=2) == ("b", "c")


def test_a_benchmark_score_cannot_advance_a_candidate() -> None:
    """The screen is structurally blind to final-gate evidence.

    Even a row that *carries* a protected or target score is ordered only by
    whether it trained: the tempting field changes nothing about who advances.
    """
    results = [
        {
            "recipe_id": "a",
            "status": "SUCCEEDED",
            "candidate_succeeded": True,
            "artifact_ref": "/attempts/a/adapter",
            "target_scores": {"generation-diagnostics@v1": 0.99},
            "protected_scores": {"math500@2024-04": 1.0},
        },
        _attempt("b"),
    ]
    assert advanced(results, survivor_count=2) == ("a", "b")
    # And a row with no training evidence never advances, however it scores.
    scored_only = [
        {"recipe_id": "a", "target_scores": {"target@v1": 1.0}},
        _attempt("b"),
    ]
    assert advanced(scored_only, survivor_count=2) == ("b",)


# --------------------------------------------------------------------------
# the run: rounds, evidence, and a stop that still records its spend
# --------------------------------------------------------------------------


def test_the_run_walks_the_rounds_and_offers_only_the_last_round_to_selection(tmp_path) -> None:
    recipes = [_recipe("a"), _recipe("b")]
    plan = _plan(recipes)
    seen: list[tuple[str, int]] = []
    resumed: list[str | None] = []
    charged: list[str] = []
    checkpoints: dict[str, str] = {}

    def run_attempt(recipe: TrainingRecipe):
        seen.append((recipe.recipe_id, recipe.max_steps))
        resumed.append(recipe.resume_from_checkpoint)
        artifact = _checkpoint_artifact(tmp_path, recipe.recipe_id)
        checkpoints[recipe.recipe_id] = artifact
        return _attempt(recipe.recipe_id, artifact_ref=artifact)

    def on_attempt(evidence, row) -> None:
        charged.append(f"{evidence['recipe_id']}@{row.round_index}")

    run = run_search(
        plan,
        declaration=_declaration(),
        recipes=recipes,
        project_cost=_project_cost,
        run_attempt=run_attempt,
        on_attempt=on_attempt,
    )

    assert seen == [("a", 12), ("b", 12), ("a", 24)]
    assert charged == ["a@0", "b@0", "a@1"]
    assert run.survivors == ("a",)
    # Selection may only ever read the final round's attempts.
    assert [row["recipe_id"] for row in run.final_results] == ["a"]
    assert len(run.round_attempts) == 2
    assert sum(len(rows) for rows in run.round_attempts) == 3
    # Progressive allocation is continuation, not restart: round 0 starts
    # from the parent, and round 1 resumes the checkpoint the survivor's own
    # round-0 attempt produced (the highest real checkpoint-N under its
    # artifact, resolved the same way the EvolutionEngine controller resolves
    # its own continuations).
    round0_checkpoint = str(Path(checkpoints["a"]) / "trainer" / "checkpoint-12")
    assert resumed == [None, None, round0_checkpoint]
    round1 = run.round_attempts[1][0]
    assert round1["declared_resume_from"] == round0_checkpoint
    assert round1["checkpoint_dir"] == round0_checkpoint
    assert round1["search_incremental_steps"] == 12  # 24 - 12, not 24 again
    # A lineage's cumulative spend is one lookup, in addition to the
    # per-attempt incremental rows.
    assert run.candidate_cumulative["a"]["rounds"] == 2
    assert run.candidate_cumulative["b"]["rounds"] == 1


def test_a_candidate_that_fails_its_cheap_round_never_earns_a_larger_budget(tmp_path) -> None:
    recipes = [_recipe("a"), _recipe("b")]
    plan = _plan(recipes)
    seen: list[str] = []
    outcomes = {"a": False, "b": True}

    def run_attempt(recipe: TrainingRecipe):
        seen.append(recipe.recipe_id)
        return _attempt(
            recipe.recipe_id,
            succeeded=outcomes[recipe.recipe_id],
            artifact_ref=(
                _checkpoint_artifact(tmp_path, recipe.recipe_id)
                if outcomes[recipe.recipe_id]
                else None
            ),
        )

    run = run_search(
        plan,
        declaration=_declaration(),
        recipes=recipes,
        project_cost=_project_cost,
        run_attempt=run_attempt,
    )
    # Round 0 ran both; only the survivor ran round 1.
    assert seen == ["a", "b", "b"]
    assert run.survivors == ("b",)


def test_a_stopped_search_keeps_the_rounds_and_the_spend_it_did() -> None:
    recipes = [_recipe("a"), _recipe("b")]
    plan = _plan(recipes)

    run = run_search(
        plan,
        declaration=_declaration(),
        recipes=recipes,
        project_cost=_project_cost,
        run_attempt=lambda recipe: _attempt(recipe.recipe_id),
        should_stop=lambda device, wall: "campaign ceiling reached",
    )

    assert run.stopped_by == "campaign ceiling reached"
    assert len(run.rounds) == 1
    # The spend recorded is the spend actually made: the ceiling tripped on the
    # first attempt, so the round stopped there rather than running out its
    # full candidate list. A partial round's cost is one attempt, not the
    # round's worst-case projection.
    assert len(run.round_attempts[0]) == 1
    assert run.total_wall_gpu_hours == pytest.approx(
        _project_cost(seq_len=2048, max_steps=plan.rounds[0].max_steps)[1]
    )


def test_an_undeclared_plan_has_no_rounds_to_run() -> None:
    with pytest.raises(CandidateSearchRefusal) as error:
        run_search(
            _plan([_recipe("a")], CandidateSearchDeclaration()),
            declaration=CandidateSearchDeclaration(),
            recipes=[_recipe("a")],
            project_cost=_project_cost,
            run_attempt=lambda recipe: _attempt(recipe.recipe_id),
        )
    assert "not a search" in str(error.value)


def test_running_a_plan_under_a_different_declaration_refuses() -> None:
    """The admitted budget and the allocating budget must be one decision."""
    recipes = [_recipe("a")]
    plan = _plan(recipes)
    with pytest.raises(CandidateSearchRefusal) as error:
        run_search(
            plan,
            declaration=_declaration(rounds=3),
            recipes=recipes,
            project_cost=_project_cost,
            run_attempt=lambda recipe: _attempt(recipe.recipe_id),
        )
    assert "different search" in str(error.value)


def test_a_plan_naming_a_recipe_this_campaign_does_not_hold_refuses() -> None:
    recipes = [_recipe("a"), _recipe("b")]
    plan = _plan(recipes)
    with pytest.raises(CandidateSearchRefusal) as error:
        run_search(
            plan,
            declaration=_declaration(),
            recipes=[replace(recipes[0], recipe_id="other")],
            project_cost=_project_cost,
            run_attempt=lambda recipe: _attempt(recipe.recipe_id),
        )
    assert "does not hold" in str(error.value)
    assert "'a'" in str(error.value)


def test_stop_ends_the_round_it_fires_in() -> None:
    """A tripped ceiling stops the *next* attempt, not just the next round.

    The single-pass path in ``run_campaign`` breaks the moment the campaign
    ceiling trips. A declared search must not be the looser path: finishing the
    round it is already inside would spend up to ``round_size - 1`` further
    attempts past the ceiling that stopped it.
    """
    recipes = [_recipe("a"), _recipe("b"), _recipe("c")]
    plan = _plan(recipes)
    trained: list[str] = []

    run = run_search(
        plan,
        declaration=_declaration(),
        recipes=recipes,
        project_cost=_project_cost,
        run_attempt=lambda recipe: (
            trained.append(recipe.recipe_id) or _attempt(recipe.recipe_id)
        ),
        should_stop=lambda device, wall: "campaign ceiling reached",
    )

    # One attempt was made, the ceiling tripped, and nothing else was spent.
    assert trained == ["a"]
    assert run.stopped_by == "campaign ceiling reached"
    # The partial round is still in the record, with the attempt it did make:
    # a stopped round keeps its spend rather than vanishing from the evidence.
    assert len(run.round_attempts) == 1
    assert [row["recipe_id"] for row in run.round_attempts[0]] == ["a"]
    assert run.total_device_gpu_hours == pytest.approx(
        _project_cost(seq_len=2048, max_steps=plan.rounds[0].max_steps)[0]
    )


def test_a_run_may_not_train_a_recipe_the_plan_never_projected() -> None:
    """Who runs is the plan's decision, not the caller's recipe list.

    ``plan_search`` admits a budget for the candidates it projected. If the run
    seeds itself from whatever recipes it was handed, a caller passing a larger
    set spends round 0 on candidates no ceiling was ever checked against.
    """
    projected = [_recipe("a"), _recipe("b")]
    plan = _plan(projected)
    trained: list[str] = []

    run = run_search(
        plan,
        declaration=_declaration(),
        recipes=[*projected, _recipe("c"), _recipe("d")],
        project_cost=_project_cost,
        run_attempt=lambda recipe: (
            trained.append(recipe.recipe_id) or _attempt(recipe.recipe_id)
        ),
    )

    assert set(trained) <= set(plan.rounds[0].recipe_ids)
    assert "c" not in trained and "d" not in trained
    assert run.total_device_gpu_hours <= plan.total_device_gpu_hours


# --------------------------------------------------------------------------
# progressive allocation is continuation, not restart
# --------------------------------------------------------------------------


def test_a_round_after_the_first_is_priced_at_the_step_delta_not_a_full_retrain() -> None:
    """Continuation economics: round 1 trains 24-12=12 incremental steps.

    Pricing a continuation at the full round budget would double-charge the
    earlier round and make progressive allocation look unaffordable -- the
    projection must measure what the round actually adds.
    """
    recipes = [_recipe("a"), _recipe("b")]
    plan = _plan(recipes)
    assert plan.rounds[0].max_steps == 12
    assert plan.rounds[1].max_steps == 24
    # Round 0 is priced at its full budget (two candidates); round 1 at the
    # 12-step delta for the schedule's own worst-case survivor count (1).
    assert plan.rounds[0].projected_device_gpu_hours == pytest.approx(
        2 * _project_cost(seq_len=2048, max_steps=12)[0]
    )
    assert plan.rounds[1].projected_device_gpu_hours == pytest.approx(
        _project_cost(seq_len=2048, max_steps=12)[0]
    )
    # ...and a full-retrain pricing of round 1 would have been strictly larger.
    full_retrain = _project_cost(seq_len=2048, max_steps=24)[0]
    assert plan.rounds[1].projected_device_gpu_hours < 2 * full_retrain


def test_a_survivor_with_no_checkpoint_ends_its_lineage_instead_of_restarting(
    tmp_path,
) -> None:
    """No checkpoint, no continuation: the honest stop.

    A survivor whose round produced no resumable checkpoint cannot earn a
    larger budget -- continuing it would mean silently restarting the recipe
    from step 0 and paying for its earlier rounds a second time.
    """
    recipes = [_recipe("a")]
    plan = _plan(recipes)

    def run_attempt(recipe: TrainingRecipe):
        artifact = tmp_path / "a-artifact"
        artifact.mkdir(parents=True, exist_ok=True)
        return _attempt("a", artifact_ref=str(artifact))

    run = run_search(
        plan,
        declaration=_declaration(),
        recipes=recipes,
        project_cost=_project_cost,
        run_attempt=run_attempt,
    )

    assert run.lineage_stops == {
        "a": "no checkpoint from the previous round to continue from"
    }
    # The lineage ended: no continuation attempt was made or charged, and the
    # dead lineage is not reported as a survivor.
    assert len(run.round_attempts) == 1
    assert run.survivors == ()
    assert run.total_device_gpu_hours == pytest.approx(
        _project_cost(seq_len=2048, max_steps=12)[0]
    )


def test_an_executor_reported_restart_never_earns_a_larger_budget(tmp_path) -> None:
    """A silent restart is detected and disqualified, not rewarded.

    When the executor reports ``resume_state == "not-a-resume"`` for a
    declared continuation, the attempt stays in the accounting -- the compute
    really happened -- but its lineage cannot advance, so a restart can never
    win progressive allocation.
    """
    recipes = [_recipe("a"), _recipe("b")]
    plan = _plan(recipes)

    def run_attempt(recipe: TrainingRecipe):
        extra: dict = {"artifact_ref": _checkpoint_artifact(tmp_path, recipe.recipe_id)}
        if recipe.resume_from_checkpoint is not None:
            # The executor admits it restarted instead of resuming.
            extra["resume_state"] = "not-a-resume"
        return _attempt(recipe.recipe_id, **extra)

    run = run_search(
        plan,
        declaration=_declaration(),
        recipes=recipes,
        project_cost=_project_cost,
        run_attempt=run_attempt,
    )

    assert run.lineage_stops == {
        "a": (
            "the attempt reported it restarted instead of resuming the "
            "declared checkpoint"
        )
    }
    # Both rounds ran (the restart attempt still happened and was recorded),
    # but the round-1 lineage ends: nobody advances past a silent restart.
    assert len(run.round_attempts) == 2
    assert run.survivors == ()


def test_an_interrupted_search_resumes_from_its_recorded_progress(tmp_path) -> None:
    """A stopped search continues as the same search, not a new one.

    The completed rounds, their spend, and their survivors are carried over;
    only the remaining rounds run, and the survivor continues from the
    checkpoint its interrupted run recorded.
    """
    recipes = [_recipe("a"), _recipe("b")]
    plan = _plan(recipes)
    checkpoints: dict[tuple[str, int], str] = {}

    def run_attempt(recipe: TrainingRecipe):
        key = (recipe.recipe_id, recipe.max_steps)
        artifact = _checkpoint_artifact(tmp_path, f"{recipe.recipe_id}-{recipe.max_steps}")
        checkpoints[key] = artifact
        return _attempt(recipe.recipe_id, artifact_ref=artifact)

    # First pass: stop as soon as round 0 is over.
    def stop_after_round0(_device: float, _wall: float) -> str | None:
        return "operator interrupt"

    interrupted = run_search(
        plan,
        declaration=_declaration(),
        recipes=recipes,
        project_cost=_project_cost,
        run_attempt=run_attempt,
        should_stop=stop_after_round0,
    )
    assert interrupted.stopped_by == "operator interrupt"
    assert len(interrupted.rounds) == 1
    spent_before = interrupted.total_device_gpu_hours

    # Resume: the recorded progress is carried over and round 1 continues.
    progress = SearchProgress.from_run(interrupted)
    resumed = run_search(
        plan,
        declaration=_declaration(),
        recipes=recipes,
        project_cost=_project_cost,
        run_attempt=run_attempt,
        progress=progress,
    )

    assert resumed.stopped_by is None
    assert len(resumed.rounds) == 2
    # Round 0 was NOT re-run: only the continuation round trained.
    assert resumed.round_attempts[0] == interrupted.round_attempts[0]
    assert [row["recipe_id"] for row in resumed.round_attempts[1]] == ["a"]
    # The interrupted run's spend is carried, not double-counted.
    assert resumed.total_device_gpu_hours == pytest.approx(
        spent_before + _project_cost(seq_len=2048, max_steps=12)[0]
    )
    # And the continuation still resumes the survivor's own checkpoint.
    round0_checkpoint = str(
        Path(checkpoints[("a", 12)]) / "trainer" / "checkpoint-12"
    )
    assert resumed.round_attempts[1][0]["declared_resume_from"] == round0_checkpoint


def test_progress_from_a_concluded_search_refuses() -> None:
    """Only an interrupted search has something to resume."""
    declaration = _declaration(rounds=1)
    plan = _plan([_recipe("a")], declaration=declaration)
    run = run_search(
        plan,
        declaration=declaration,
        recipes=[_recipe("a")],
        project_cost=_project_cost,
        run_attempt=lambda recipe: _attempt(recipe.recipe_id),
    )
    try:
        SearchProgress.from_run(run)
    except CandidateSearchRefusal:
        pass
    else:
        raise AssertionError("a concluded search must not produce a resumable progress")


def test_progress_from_a_different_plan_refuses(tmp_path) -> None:
    """A search may only resume as the search it was declared as."""
    recipes = [_recipe("a"), _recipe("b")]
    plan = _plan(recipes)

    def run_attempt(recipe: TrainingRecipe):
        return _attempt(
            recipe.recipe_id,
            artifact_ref=_checkpoint_artifact(tmp_path, recipe.recipe_id),
        )

    interrupted = run_search(
        plan,
        declaration=_declaration(),
        recipes=recipes,
        project_cost=_project_cost,
        run_attempt=run_attempt,
        should_stop=lambda _d, _w: "interrupt",
    )
    progress = SearchProgress.from_run(interrupted)
    other_declaration = _declaration(initial_max_steps=20)
    other_plan = _plan(recipes, declaration=other_declaration)
    try:
        run_search(
            other_plan,
            declaration=other_declaration,
            recipes=recipes,
            project_cost=_project_cost,
            run_attempt=run_attempt,
            progress=progress,
        )
    except CandidateSearchRefusal as error:
        assert "prefix" in str(error)
    else:
        raise AssertionError("progress from a different plan must refuse")
