"""The production search controller: successive halving through run_project.

The library controller (`run_successive_halving`) and the UCB1 prioritizer
(`prioritize_candidates`) existed as tested library capabilities but were
never reachable from the interface a user runs. These tests pin the
project-level contract: opt-in `search` config, deterministic variant
enumeration, final-round-only promotion, exact lineage persistence, honest
elimination reasons, and real reservation settlement through the normal
project runner.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from chowder.project import ProjectValidationError, project_from_mapping
from chowder.registry import RunRegistry

from test_router_healing_end_to_end import _project_payload, _write_corpora


# --- the contract layer: schema ----------------------------------------------


def _search_payload(tmp_path, **overrides):
    base, train, holdout = _write_corpora(tmp_path)
    payload = _project_payload(tmp_path, base=base, train=train, holdout=holdout)
    search = {
        "initial_max_steps": 2,
        "step_multiplier": 2.0,
        "survival_fraction": 0.5,
        "min_survivors": 1,
        "max_rounds": 2,
        "variants": [
            {"backend": {"router_healing": {"learning_rate": 0.08}}},
            {"backend": {"router_healing": {"learning_rate": 0.12}}},
        ],
    }
    search.update(overrides)
    payload["search"] = search
    return payload


def test_search_config_round_trips_with_defaults(tmp_path):
    payload = _search_payload(tmp_path)
    del payload["search"]["max_rounds"]
    project = project_from_mapping(payload, source_dir=tmp_path)

    assert project.search is not None
    assert project.search.initial_max_steps == 2
    assert project.search.max_rounds is None  # unbounded by default
    # Deterministic enumeration: the declared experiment is variant 0;
    # each patch spawns a numbered sibling.
    population = project.search.variant_patches(project.experiment)
    assert [v.experiment_id for v in population] == [
        "router-pilot",
        "router-pilot-v1",
        "router-pilot-v2",
    ]
    assert [v.config_patch for v in population[1:]] == [
        {"backend": {"router_healing": {"learning_rate": 0.08}}},
        {"backend": {"router_healing": {"learning_rate": 0.12}}},
    ]


@pytest.mark.parametrize(
    ("bad_field", "bad_value", "message"),
    [
        ("step_multiplier", 1.0, "step_multiplier"),
        ("step_multiplier", 0.5, "step_multiplier"),
        ("survival_fraction", 1.0, "survival_fraction"),
        ("survival_fraction", 0.0, "survival_fraction"),
        ("min_survivors", 0, "min_survivors"),
    ],
)
def test_search_config_rejects_invalid_controller_parameters(
    tmp_path, bad_field, bad_value, message
):
    payload = _search_payload(tmp_path)
    payload["search"][bad_field] = bad_value

    with pytest.raises(ProjectValidationError, match=message):
        project_from_mapping(payload, source_dir=tmp_path)


def test_search_config_rejects_an_empty_variant_list(tmp_path):
    payload = _search_payload(tmp_path)
    payload["search"]["variants"] = []

    with pytest.raises(ProjectValidationError, match="variants"):
        project_from_mapping(payload, source_dir=tmp_path)


def test_search_and_repair_are_mutually_exclusive(tmp_path):
    """One candidate strategy per project: a search replaces the single
    candidate + repair ladder, and mixing both is a config error, not a
    silently-ignored knob."""
    payload = _search_payload(tmp_path)
    payload["repair"] = {
        "corpus_files": ["corpus.txt"],
        "variants": [{"config_patch": {"backend": {"router_healing": {"seed": 2}}}}],
    }

    with pytest.raises(ProjectValidationError, match="repair"):
        project_from_mapping(payload, source_dir=tmp_path)


# --- the real layer: the production controller -------------------------------


@pytest.fixture(scope="module")
def tiny_search_project(tmp_path_factory):
    """The same real tiny Qwen3 MoE fixture the paired tests use."""
    import test_router_healing_end_to_end as e2e

    e2e._require_real_model()

    root = tmp_path_factory.mktemp("router-search-project")
    # Rebuild the tiny model exactly like the e2e fixture (module-scoped,
    # built once for this file's real-layer tests).
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
    import torch

    lines = [
        f"the router selects expert {index % 4} for token number {index} in this sentence"
        for index in range(400)
    ]
    train = root / "corpus.txt"
    train.write_text("\n".join(lines), encoding="utf-8")
    holdout = root / "holdout.txt"
    holdout.write_text(
        "\n".join(
            f"held out sentence {index} asks which expert answers token {index * 3}"
            for index in range(400)
        ),
        encoding="utf-8",
    )
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.train_from_iterator(
        lines, trainers.WordLevelTrainer(vocab_size=64, special_tokens=["[UNK]", "[PAD]"])
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[UNK]"
    )
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=8,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
    )
    base = root / "base"
    Qwen3MoeForCausalLM(config).float().save_pretrained(base)
    fast.save_pretrained(base)
    return {"root": root, "base": base, "train": train, "holdout": holdout}


def _run_search(tmp_path, tiny_search_project, *, goal_overrides=None, **search_overrides):
    from chowder.project_runner import run_project

    base = tiny_search_project["base"]
    payload = _project_payload(
        tmp_path,
        base=base,
        train=tiny_search_project["train"],
        holdout=tiny_search_project["holdout"],
    )
    search = {
        "initial_max_steps": 2,
        "step_multiplier": 2.0,
        "survival_fraction": 0.5,
        "min_survivors": 1,
        "max_rounds": 2,
        "variants": [
            {"backend": {"router_healing": {"learning_rate": 0.08}}},
            {"backend": {"router_healing": {"learning_rate": 0.12}}},
        ],
    }
    search.update(search_overrides)
    payload["search"] = search
    if goal_overrides:
        payload["goal"].update(goal_overrides)
    payload["name"] = "router-healing-tiny-search"
    payload["work_dir"] = str(tmp_path / "work")
    payload["registry_path"] = str(tmp_path / "work" / "runs.db")
    project_path = tmp_path / "project-search.json"
    project_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    outcome = run_project(project_path)
    return outcome, json.loads(project_path.read_text(encoding="utf-8"))


def test_search_runs_halving_rounds_through_the_normal_runner(
    tmp_path, tiny_search_project
):
    """The user-facing contract: a project with `search` produces real
    successive-halving rounds through run_project, persists every round's
    exact lineage, eliminates candidates with honest reasons, and promotes
    only from the final round."""
    outcome, _payload = _run_search(tmp_path, tiny_search_project)

    search_outcome = outcome.search
    assert search_outcome is not None
    assert search_outcome.total_rounds == 2  # max_rounds=2 with 3 variants
    # Round 0 ran all three variants for real; round 1 ran only survivors.
    assert len(search_outcome.rounds[0].generation.candidates) == 3
    assert len(search_outcome.rounds[1].generation.candidates) <= 2
    # Every candidate reached a terminal outcome with real cost settled.
    for round_outcome in search_outcome.rounds:
        for candidate in round_outcome.generation.candidates:
            assert candidate.error is None, candidate.error
            assert candidate.result is not None
            assert candidate.result.gpu_hours >= 0.0

    with RunRegistry(outcome.project.registry_path) as registry:
        experiments = {
            e.experiment_id: e for e in registry.list_experiments()
        }
        rows = {
            (row["round_index"], row["experiment_id"]): row
            for row in registry.list_search_rounds()
        }

    # Round lineage: round-1 survivors are children of round-0 rows.
    round1_ids = {
        c.experiment_id for c in search_outcome.rounds[1].generation.candidates
    }
    for experiment_id in round1_ids:
        assert experiments[experiment_id].parent_id is not None
        assert experiments[experiment_id].parent_id in experiments
    # The registry's round rows match the outcome exactly: no drift between
    # the in-memory controller and the durable history.
    assert {
        (round_index, experiment_id)
        for round_index, experiment_id in rows
    } == {
        (round_outcome.round_index, candidate.experiment_id)
        for round_outcome in search_outcome.rounds
        for candidate in round_outcome.generation.candidates
    }

    # Honest elimination reasons, persisted with the round.
    eliminated_by_gate = {
        experiment_id
        for round_outcome in search_outcome.rounds
        for experiment_id in round_outcome.eliminated_by_gate_experiment_ids
    }
    eliminated_by_cutoff = {
        experiment_id
        for round_outcome in search_outcome.rounds
        for experiment_id in round_outcome.eliminated_by_cutoff_experiment_ids
    }
    assert not (eliminated_by_gate & eliminated_by_cutoff)
    for row in rows.values():
        if row["experiment_id"] in eliminated_by_gate:
            assert row["eliminated_by"] == "gate"
        elif row["experiment_id"] in eliminated_by_cutoff:
            assert row["eliminated_by"] == "cutoff"
        else:
            assert row["eliminated_by"] is None

    # Final-round-only promotion: whatever the outcome promoted came from the
    # last round's ranking, never an intermediate one.
    if outcome.search.promoted is not None:
        final_ids = {
            candidate.experiment_id
            for candidate in search_outcome.rounds[-1].generation.candidates
        }
        assert outcome.search.promoted.experiment_id in final_ids


def test_search_without_promotion_still_reports_a_complete_outcome(
    tmp_path, tiny_search_project
):
    """A search that ends with no promotion (the gate rejected every final
    survivor) is a COMPLETE run with a recorded negative result, not an
    error."""
    outcome, _payload = _run_search(
        tmp_path,
        tiny_search_project,
        goal_overrides={"minimum_promotion_gain": 100.0},
        variants=[
            {"backend": {"router_healing": {"learning_rate": 0.0001}}},
        ],
        min_survivors=1,
    )
    # A promotion gain no real 2-step run can deliver makes the gate reject
    # every final survivor: the search completes with a recorded negative
    # result -- a full evidence trail, not an error.
    assert outcome.search is not None
    assert outcome.search.promoted is None
    assert outcome.search.total_rounds >= 1
    assert outcome.search.rounds[-1].eliminated_by_gate_experiment_ids, (
        "a completed search with no promotion must record gate eliminations"
    )


def test_search_replay_on_a_used_registry_fails_loudly_without_double_charging(
    tmp_path, tiny_search_project
):
    """Re-running a completed search project on the same registry must fail
    loudly and must NOT silently re-execute terminal rows or double-charge
    the ledger. (The fine-grained PLANNED-replay semantics are pinned at the
    library level in test_search_controller_integration.py; this pins the
    user-facing consequence.)"""
    from chowder.project_runner import run_project

    outcome, _payload = _run_search(tmp_path, tiny_search_project)
    project_path = tmp_path / "project-search.json"

    with RunRegistry(outcome.project.registry_path) as registry:
        results_before = len(list(registry.list_results()))
        spent_before = sum(r.gpu_hours for r in registry.list_results())

    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - any loud failure
        run_project(project_path)
    assert "baseline" in str(excinfo.value) or "PLANNED" in str(excinfo.value)

    with RunRegistry(outcome.project.registry_path) as registry:
        results_after = len(list(registry.list_results()))
        spent_after = sum(r.gpu_hours for r in registry.list_results())
    assert results_after == results_before
    assert spent_after == pytest.approx(spent_before)
