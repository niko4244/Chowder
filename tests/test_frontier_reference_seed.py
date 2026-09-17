"""The frontier reference seed's provenance gates.

The FrontierDatabase is the trust boundary between published numbers and
Chowder's promotion/gap machinery: every stored reference must be traceable
to its exact source. These tests pin the provenance gates the closeout adds
on top of the existing discipline (unknown levels, snapshot immutability,
and protocol comparability are already pinned in test_growth_decisions.py).

Provenance policy: a reference without an exact benchmark@version, a valid
normalized score, and a real source URL is not a reference -- it is
decoration. Duplicates are refused so a benchmark's "best" reference can
never be silently overwritten by a re-import.
"""

from __future__ import annotations

import pytest

from chowder.growth.frontier_reference import FrontierDatabase, ReferenceScore


def _reference(**overrides):
    base = dict(
        model="Qwen2.5-7B-Instruct",
        benchmark_qualified_id="math500@2024-04",
        score=0.755,
        level="LEVEL_1_COMPARABLE_PEER",
        date="2024-09-19",
        source_url="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
        harness="qwen-eval (Qwen team internal harness)",
        tool_setting="none",
        reasoning_setting="direct",
        first_party=True,
        comparability_confidence="MEDIUM",
    )
    base.update(overrides)
    return ReferenceScore(**base)


def test_duplicate_reference_entry_is_refused(tmp_path):
    db = FrontierDatabase(tmp_path)
    db.add(_reference())
    with pytest.raises(ValueError, match="duplicate reference"):
        db.add(_reference())


def test_unpinned_benchmark_name_is_refused(tmp_path):
    db = FrontierDatabase(tmp_path)
    with pytest.raises(ValueError, match="must be pinned"):
        db.add(_reference(benchmark_qualified_id="math500@latest"))
    with pytest.raises(ValueError, match="must be pinned"):
        db.add(_reference(benchmark_qualified_id="math500"))


def test_out_of_range_score_is_refused(tmp_path):
    db = FrontierDatabase(tmp_path)
    with pytest.raises(ValueError, match=r"normalized 0\.\.1"):
        db.add(_reference(score=1.5))
    with pytest.raises(ValueError, match=r"normalized 0\.\.1"):
        db.add(_reference(score=-0.1))


def test_missing_source_url_is_refused(tmp_path):
    db = FrontierDatabase(tmp_path)
    with pytest.raises(ValueError, match="source_url"):
        db.add(_reference(source_url=""))
    with pytest.raises(ValueError, match="source_url"):
        db.add(_reference(source_url="   "))


def test_medium_confidence_reference_is_recorded_but_never_decides(tmp_path):
    db = FrontierDatabase(tmp_path)
    db.add(_reference())
    reloaded = FrontierDatabase(tmp_path)
    # best_for_benchmark only surfaces HIGH-confidence references: a MEDIUM
    # reference is preserved for the record but is inert in gap machinery.
    assert reloaded.best_for_benchmark("math500@2024-04", "LEVEL_1_COMPARABLE_PEER") is None
    from chowder.growth.frontier_reference import compare_protocol

    best_effort = reloaded.scores()[0]
    assert (
        compare_protocol(
            best_effort,
            benchmark_qualified_id="math500@2024-04",
            tool_setting="none",
            reasoning_setting="direct",
        )
        != "COMPARABLE"
    )


def test_high_confidence_reference_round_trips_and_decides(tmp_path):
    db = FrontierDatabase(tmp_path)
    db.add(_reference(comparability_confidence="HIGH"))
    reloaded = FrontierDatabase(tmp_path)
    best = reloaded.best_for_benchmark("math500@2024-04", "LEVEL_1_COMPARABLE_PEER")
    assert best is not None and best.model == "Qwen2.5-7B-Instruct"
    from chowder.growth.frontier_reference import compare_protocol

    assert (
        compare_protocol(
            best,
            benchmark_qualified_id="math500@2024-04",
            tool_setting="none",
            reasoning_setting="direct",
        )
        == "COMPARABLE"
    )
