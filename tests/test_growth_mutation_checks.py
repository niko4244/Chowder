"""The mutation checks: for each way this architecture could be quietly
corrupted, run the corruption against its real enforcement point and prove
it is caught. These are not unit tests of happy paths -- each one *is* the
sabotage a future change might reintroduce, kept executable so the guard
cannot rot silently.

1. evidence tampering -> the hash chain audit refuses
2. a silent restart instead of a declared resume -> lineage stops
3. an inert search axis (a knob nothing reads) -> declaration refuses
4. a protected benchmark in a search-readable surface -> isolation refuses
5. removing an attempt from the accounting -> the recorded side no longer
   reconciles with the run's own totals
6. promoting despite a protected regression -> retention refuses
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chowder.growth.candidate_search import (
    CandidateSearchDeclaration,
    CandidateSearchRefusal,
    plan_search,
    run_search,
)
from chowder.growth.evidence import EvidenceRecord, EvidenceState, EvidenceStore
from chowder.growth.eval_isolation import (
    SearchIsolationRefusal,
    assert_search_isolation,
    classify_benchmarks,
)
from chowder.growth.recipe_planner import TrainingRecipe
from chowder.growth.retention import RetentionConstraint, RetentionProfile, evaluate_retention

from test_growth_candidate_search import (
    _checkpoint_artifact,
    _declaration,
    _project_cost,
    _recipe,
)


# --- 1. evidence tampering ----------------------------------------------------


def _store(tmp_path: Path) -> EvidenceStore:
    store = EvidenceStore(tmp_path / "evidence.jsonl")
    store.record(
        EvidenceRecord(
            record_id="rec-mutate-1",
            state=EvidenceState.FAILED,
            family_id="training.sft-curriculum",
            model_family="qwen3.8",
            checkpoint_identity="gen2@abc",
            architecture="dense",
            eval_suite="screening@v1",
            intervention_parameters={"learning_rate": 2e-4},
            software_runtime={"transformers": "5.18.0"},
            hardware_class="kaggle_2x_t4_16gb",
            sample_size=3,
            evidence_quality="paired-seeds",
            measured_effect={"target_metric": -0.05},
            date="2026-10-04",
            source_run="kernel-1",
        )
    )
    return store


def test_mutation_1_rewriting_a_failed_record_to_promising_is_caught(tmp_path):
    store = _store(tmp_path)
    rows = [json.loads(line) for line in store.path.read_text().splitlines()]
    rows[0]["state"] = "promising"  # the sabotage: upgrade the verdict
    store.path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="failed its hash chain"):
        store.audit()


# --- 2. silent restart --------------------------------------------------------


def test_mutation_2_an_executor_that_restarts_instead_of_resuming_loses_the_lineage(
    tmp_path: Path,
) -> None:
    recipes = [_recipe("r-a"), _recipe("r-b")]
    declaration = _declaration()
    plan = plan_search(
        declaration,
        recipes=recipes,
        project_cost=_project_cost,
        per_recipe_device_ceiling=0.3,
        per_recipe_wall_ceiling=0.1,
        campaign_device_ceiling=0.6,
        campaign_wall_ceiling=0.2,
    )

    def run_attempt(recipe: TrainingRecipe) -> dict[str, object]:
        return {
            "status": "SUCCEEDED",
            "candidate_succeeded": True,
            "artifact_ref": _checkpoint_artifact(tmp_path, recipe.recipe_id),
        }

    # Round 0 runs both from the parent. Round 1 declares a continuation;
    # the executor "succeeds" but admits it restarted.
    def restart_run_attempt(recipe: TrainingRecipe) -> dict[str, object]:
        evidence = dict(run_attempt(recipe))
        if recipe.resume_from_checkpoint is not None:
            evidence["resume_state"] = "not-a-resume"
        return evidence

    run = run_search(
        plan,
        declaration=declaration,
        recipes=recipes,
        project_cost=_project_cost,
        run_attempt=restart_run_attempt,
    )
    # The restarted lineage is named, and it is not a survivor: the spend
    # stays in the accounting, the lineage does not.
    assert run.lineage_stops
    assert all(
        recipe_id in run.lineage_stops for recipe_id in ("r-a", "r-b")
    ) or run.survivors == ()
    assert run.total_device_gpu_hours > 0.0
    for recipe_id in run.lineage_stops:
        assert recipe_id not in run.survivors


# --- 3. inert search axis -----------------------------------------------------


def test_mutation_3_a_search_axis_nothing_reads_refuses_the_declaration() -> None:
    with pytest.raises(CandidateSearchRefusal, match="unknown candidate_search fields"):
        CandidateSearchDeclaration.from_mapping(
            {"rounds": 2, "initial_max_steps": 10, "vibes": "immaculate"}
        )


# --- 4. protected benchmark in the search surface ------------------------------


def test_mutation_4_reading_a_protected_benchmark_from_the_search_refuses() -> None:
    policy = classify_benchmarks(
        {"dev-loss": "search-evidence", "protected-suite": "promotion-evidence"}
    )
    with pytest.raises(SearchIsolationRefusal, match="promotion evidence"):
        assert_search_isolation(
            policy=policy,
            search_readable_benchmarks=["dev-loss", "protected-suite"],
        )
    # And the lazy path: renaming it into the declaration's search section is
    # caught by the reserved-name rule even before isolation runs.
    with pytest.raises(SearchIsolationRefusal, match="reserved promotion"):
        classify_benchmarks({"protected-suite": "search-evidence"})


# --- 5. attempt removal from the accounting ------------------------------------


def test_mutation_5_dropping_an_attempt_breaks_the_recorded_reconciliation(
    tmp_path: Path,
) -> None:
    recipes = [_recipe(rid) for rid in ("r-a", "r-b", "r-c", "r-d")]
    declaration = _declaration()
    plan = plan_search(
        declaration,
        recipes=recipes,
        project_cost=_project_cost,
        per_recipe_device_ceiling=0.3,
        per_recipe_wall_ceiling=0.1,
        campaign_device_ceiling=0.6,
        campaign_wall_ceiling=0.2,
    )
    charged: list[tuple[str, float]] = []

    def on_attempt(evidence, round_row) -> None:
        charged.append((str(evidence["recipe_id"]), evidence["search_max_steps"]))

    run = run_search(
        plan,
        declaration=declaration,
        recipes=recipes,
        project_cost=_project_cost,
        run_attempt=lambda recipe: {
            "status": "SUCCEEDED",
            "candidate_succeeded": True,
            "artifact_ref": _checkpoint_artifact(tmp_path, recipe.recipe_id),
        },
        on_attempt=on_attempt,
    )

    # The identity a ledger reconciles against: every attempt the run made
    # was charged, the per-candidate cumulative sums to the run's total, and
    # the total matches the projection's arithmetic.
    assert {rid for rid, _ in charged} >= set(run.candidate_cumulative)
    assert set(run.candidate_cumulative) == {
        row["recipe_id"] for attempt_round in run.round_attempts for row in attempt_round
    }
    # (candidate_cumulative rounds each entry to 9 decimals, so allow a
    # few ulps of that rounding in the identity.)
    assert sum(r["device_gpu_hours"] for r in run.candidate_cumulative.values()) == pytest.approx(
        run.total_device_gpu_hours, abs=1e-8
    )
    # The sabotage: a record that "loses" one attempt no longer
    # reconciles with what was actually charged.
    tampered = dict(run.candidate_cumulative)
    dropped_id = sorted(tampered)[0]
    del tampered[dropped_id]
    assert round(sum(r["device_gpu_hours"] for r in tampered.values()), 9) != (
        run.total_device_gpu_hours
    )


# --- 6. promotion despite a protected regression --------------------------------


def test_mutation_6_promoting_a_capability_trade_refuses() -> None:
    profile = RetentionProfile(
        profile_id="gen3",
        constraints=(
            RetentionConstraint("reasoning", "max-regression", 0.0, "protected-suite"),
            RetentionConstraint("tool_validity", "absolute-floor", 0.85, "protected-suite"),
        ),
    )
    # The mandate's arithmetic: a big target gain bought with a reasoning
    # collapse must refuse -- no promotion with a footnote.
    violations = evaluate_retention(
        profile,
        parent_values={"reasoning": 0.71, "tool_validity": 0.90},
        candidate_values={"reasoning": 0.51, "tool_validity": 0.93},
    )
    assert [v.dimension for v in violations] == ["reasoning"]
