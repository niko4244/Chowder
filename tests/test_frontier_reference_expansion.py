"""The Gen-0 frontier reference expansion: cited context, zero fake parity.

Lane D filled a deliberately empty database with real first-party numbers.
The point of these tests is that the expansion must not have made Chowder's
gap machinery any more credulous: every imported row stays context, and a
protocol mismatch can never quietly become a comparison.

The honesty pin is `test_seed_declares_no_high_confidence_rows`. A HIGH row
means "this protocol aligns with the frozen Gen-0 protocol"; no published
number found for math500/mgsm does. If someone later imports one, they have to
change this test deliberately -- which is the review gate, not an accident.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from chowder.growth.frontier_reference import (
    ALL_LEVELS,
    NOT_DIRECTLY_COMPARABLE,
    ChowderScore,
    FrontierDatabase,
    SnapshotStore,
    category_gaps,
    context_rows,
    gap_rows,
    protocol_divergence,
)

SEED_SCRIPT = Path(__file__).resolve().parents[1] / "docs" / "gen0" / "seed_frontier_references.py"


def _seed_module():
    spec = importlib.util.spec_from_file_location("gen0_seed_frontier_references", SEED_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _chowder_zeros():
    return (
        ChowderScore(
            generation_version="gen0",
            benchmark_qualified_id="math500@2024-04",
            score=0.0,
            tool_setting="none",
            reasoning_setting="chat_template",
        ),
        ChowderScore(
            generation_version="gen0",
            benchmark_qualified_id="mgsm@2022-11",
            score=0.0,
            tool_setting="none",
            reasoning_setting="chat_template",
        ),
    )


def _seeded_database(tmp_path: Path) -> FrontierDatabase:
    database = FrontierDatabase(tmp_path)
    for score in _seed_module().reference_scores():
        database.add(score)
    return database


def test_seed_rows_survive_the_production_provenance_gates(tmp_path):
    module = _seed_module()
    scores = module.reference_scores()
    assert len(scores) >= 12, "the expansion should cover both measured benchmarks"
    database = FrontierDatabase(tmp_path)
    for score in scores:
        database.add(score)  # raises if unpinned / out of range / unsourced
    assert len(database.scores()) == len(scores)


def test_every_seed_row_cites_its_source_and_names_the_mismatch(tmp_path):
    for score in _seed_module().reference_scores():
        assert score.source_url.startswith("https://")
        assert score.date, f"{score.model} has no publication date"
        assert "@" in score.benchmark_qualified_id
        assert 0.0 <= score.score <= 1.0
        assert score.level in ALL_LEVELS
        assert score.notes.strip(), f"{score.model} carries no protocol note"
        assert "Chowder" in score.notes, (
            f"{score.model} does not state Chowder's protocol, so a reader "
            "cannot judge the mismatch"
        )


def test_seed_declares_no_high_confidence_rows():
    """No published number found for these benchmarks matches the frozen protocol."""
    high = [
        f"{score.model} on {score.benchmark_qualified_id}"
        for score in _seed_module().reference_scores()
        if score.comparability_confidence == "HIGH"
    ]
    assert high == [], (
        "a HIGH-confidence import claims protocol alignment with "
        f"gen0-freeze-protocol-v1 and must be justified deliberately: {high}"
    )


def test_seeded_references_produce_no_comparable_gap_rows(tmp_path):
    database = _seeded_database(tmp_path)
    for chowder in _chowder_zeros():
        rows = gap_rows(database, chowder)
        assert [row for row in rows if row.comparability == "COMPARABLE"] == []
        # Lowering a row's confidence must not route it into a gap row either:
        # best_for_benchmark admits only HIGH, so none of these are decision
        # inputs at all.
        for level in ALL_LEVELS:
            assert database.best_for_benchmark(chowder.benchmark_qualified_id, level) is None


def test_context_rows_name_the_blocking_protocol_dimension(tmp_path):
    database = _seeded_database(tmp_path)
    chowder = _chowder_zeros()[0]
    rows = context_rows(database, chowder)
    assert rows, "the imported references must be visible somewhere"
    for row in rows:
        assert row.comparability == NOT_DIRECTLY_COMPARABLE
        assert row.divergence, "a context row must say why it is not comparable"
        assert row.source_url.startswith("https://")
    levels = {row.level for row in rows}
    assert levels == {
        "LEVEL_1_COMPARABLE_PEER",
        "LEVEL_2_OPEN_WEIGHT_FRONTIER",
        "LEVEL_3_STRETCH",
        "LEVEL_4_ABSOLUTE_FRONTIER",
    }


def test_context_prefers_the_most_protocol_adjacent_row(tmp_path):
    """A same-harness MEDIUM run beats a higher-scoring unrelated LOW row."""
    database = _seeded_database(tmp_path)
    row = next(
        row
        for row in context_rows(database, _chowder_zeros()[0])
        if row.level == "LEVEL_1_COMPARABLE_PEER"
    )
    assert row.comparability_confidence == "MEDIUM"
    assert row.model == "Meta-Llama-3.1-8B-Instruct"
    assert "minerva_math" in row.harness


def test_context_rows_never_aggregate_into_a_category_gap(tmp_path):
    database = _seeded_database(tmp_path)
    chowder = _chowder_zeros()[0]
    rows = gap_rows(database, chowder)
    assert rows == [], "context must not surface through the gap path"
    aggregate = category_gaps(
        {chowder.benchmark_qualified_id: rows},
        {chowder.benchmark_qualified_id: "math"},
    )
    assert aggregate.parity is None
    assert aggregate.mean_gap == 0.0
    assert aggregate.category == "math"


def test_mgsm_has_no_absolute_frontier_row():
    """Blanks stay blank: filling one to avoid an empty cell is the failure mode."""
    scores = [
        score
        for score in _seed_module().reference_scores()
        if score.benchmark_qualified_id == "mgsm@2022-11"
    ]
    assert scores, "mgsm rows should exist at the levels that had citable numbers"
    assert all(score.level != "LEVEL_4_ABSOLUTE_FRONTIER" for score in scores)


def test_protocol_divergence_is_the_single_source_of_the_verdict(tmp_path):
    database = _seeded_database(tmp_path)
    chowder = _chowder_zeros()[0]
    for row in context_rows(database, chowder):
        reference = next(
            score
            for score in database.scores()
            if score.model == row.model
            and score.level == row.level
            and score.benchmark_qualified_id == row.benchmark_qualified_id
        )
        assert protocol_divergence(
            reference,
            benchmark_qualified_id=row.benchmark_qualified_id,
            tool_setting="none",
            reasoning_setting="chat_template",
        )
        assert tuple(row.divergence) == protocol_divergence(
            reference,
            benchmark_qualified_id=row.benchmark_qualified_id,
            tool_setting="none",
            reasoning_setting=chowder.reasoning_setting,
        )


def test_context_snapshot_is_additive_and_the_gen0_snapshot_never_changes(tmp_path):
    store = SnapshotStore(tmp_path)
    store.freeze("gen0-frontier", "2026-09-17", ())
    before = (tmp_path / "frontier_snapshots.json").read_text(encoding="utf-8")

    scores = _seed_module().reference_scores()
    store.freeze("gen0-frontier-context-2026-09-17", "2026-09-17", scores)
    payload = json.loads((tmp_path / "frontier_snapshots.json").read_text(encoding="utf-8"))
    by_id = {item["snapshot_id"]: item for item in payload}

    assert by_id["gen0-frontier"]["scores"] == [], (
        "the generation-time snapshot must stay empty; enrichment is a new, "
        "later-dated snapshot"
    )
    assert len(by_id["gen0-frontier-context-2026-09-17"]["scores"]) == len(scores)
    assert json.loads(before) == [item for item in payload if item["snapshot_id"] == "gen0-frontier"]

    with pytest.raises(ValueError, match="never rewrite"):
        store.freeze("gen0-frontier-context-2026-09-17", "2026-09-18", scores)


def test_seed_is_deterministic():
    first = [score.to_dict() for score in _seed_module().reference_scores()]
    second = [score.to_dict() for score in _seed_module().reference_scores()]
    assert first == second


def test_duplicate_seed_row_is_refused(tmp_path):
    module = _seed_module()
    database = FrontierDatabase(tmp_path)
    score = module.reference_scores()[0]
    database.add(score)
    with pytest.raises(ValueError, match="duplicate reference"):
        database.add(score)


def test_seed_script_refuses_to_rerewrite_an_existing_snapshot(tmp_path, monkeypatch):
    module = _seed_module()
    SnapshotStore(tmp_path).freeze("gen0-frontier-context-2026-09-17", "2026-09-17", ())
    before = (tmp_path / "frontier_snapshots.json").read_text(encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "seed_frontier_references.py",
            "--root",
            str(tmp_path),
            "--snapshot-id",
            "gen0-frontier-context-2026-09-17",
        ],
    )
    assert module.main() == 2
    assert (tmp_path / "frontier_snapshots.json").read_text(encoding="utf-8") == before
