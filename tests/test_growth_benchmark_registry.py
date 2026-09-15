"""The frontier benchmark registry's rules are load-bearing.

Pinned versions, honest statuses, protected splits, adapter requirements,
and the training-use/split interplay are what keep the rest of the growth
system from drifting into fake comparisons. These tests pin those rules.
"""

from __future__ import annotations

import pytest

from chowder.growth.benchmark_registry import (
    BENCHMARK_STATUSES,
    LIFECYCLE_STATES,
    TIERS,
    BenchmarkEntry,
)
from chowder.growth.capability import ALL_SKILLS
from chowder.growth.catalog import default_registry


@pytest.fixture(scope="module")
def registry():
    return default_registry()


# ---------------- catalog integrity ----------------


def test_catalog_is_populated_and_version_pinned(registry):
    assert len(registry) >= 40
    for entry in registry:
        assert entry.version
        assert entry.version.strip().lower() not in {"latest", "current", ""}
        assert "@" not in entry.version  # qualified_id adds the @version suffix


def test_every_benchmark_maps_to_known_skills(registry):
    for entry in registry:
        assert entry.skills, entry.benchmark_id
        for skill in entry.skills:
            assert skill in ALL_SKILLS, f"{entry.benchmark_id}: unknown skill {skill}"


def test_categories_cover_the_mandated_dimensions(registry):
    categories = set(registry.categories())
    required = {
        "reasoning",
        "math",
        "coding",
        "agentic",
        "tools",
        "research",
        "knowledge",
        "instruction",
        "context",
        "multilingual",
        "professional",
        "science",
        "health",
        "self_improvement",
        "multimodal",
        "safety",
    }
    missing = required - categories
    assert not missing, f"catalog missing mandated categories: {sorted(missing)}"


def test_catalog_has_no_duplicate_qualified_ids():
    # default_registry() raises on duplicates during construction; reaching
    # here means every benchmark@version pair is unique.
    registry = default_registry()
    ids = [entry.qualified_id for entry in registry]
    assert len(ids) == len(set(ids))


# ---------------- entry-level rules ----------------


def _entry(**overrides):
    base = dict(
        benchmark_id="example_bench",
        version="2025-01",
        name="Example Bench",
        category="reasoning",
        subcategory="academic",
        status="RUNNABLE_PUBLIC",
        lifecycle="ACTIVE_FRONTIER",
        tier=3,
        scorer="exact_match",
        primary_metric="accuracy",
        skills=("reasoning.abstract",),
        dataset_source="hf://example/bench",
        implementation_source="github://example/bench",
        source="Example Org",
        license="MIT",
        release_date="2025-01-15",
        adapter="inspect",
    )
    base.update(overrides)
    return BenchmarkEntry(**base)


def test_entry_refuses_unpinned_version():
    with pytest.raises(ValueError, match="version must be pinned"):
        _entry(version="latest")


def test_entry_requires_adapter_when_runnable():
    with pytest.raises(ValueError, match="RUNNABLE_PUBLIC requires an adapter"):
        _entry(adapter="")


def test_entry_rejects_unknown_category_and_status():
    with pytest.raises(ValueError, match="unknown category"):
        _entry(category="vibes")
    with pytest.raises(ValueError, match="unknown status"):
        _entry(status="SORT_OF_RUNNABLE")
    with pytest.raises(ValueError, match="unknown lifecycle"):
        _entry(lifecycle="EXPERIMENTAL")


def test_entry_modality_unsupported_cannot_sit_in_local_tiers():
    with pytest.raises(ValueError, match="modality-unsupported"):
        _entry(
            status="UNSUPPORTED_BY_CURRENT_MODALITY",
            tier=2,
            adapter="",
        )


def test_training_use_on_protected_runnable_requires_named_split():
    with pytest.raises(ValueError, match="training_split"):
        _entry(training_use_permitted=True)
    allowed = _entry(
        training_use_permitted=True,
        extra={"training_split": "train"},
    )
    assert allowed.extra["training_split"] == "train"


def test_qualified_id_and_optimizability():
    entry = _entry()
    assert entry.qualified_id == "example_bench@2025-01"
    assert entry.optimizable()
    retired = _entry(lifecycle="RETIRED")
    assert not retired.optimizable()
    score_only = _entry(status="PUBLIC_SCORE_ONLY")
    assert not score_only.optimizable()
    assert score_only.may_score_against()


def test_registry_rejects_duplicate_qualified_ids():
    from chowder.growth.benchmark_registry import BenchmarkRegistry

    with pytest.raises(ValueError, match="duplicate benchmark entry"):
        BenchmarkRegistry((_entry(), _entry()))


def test_registry_lookups_by_tier_category_and_lifecycle(registry):
    runnable = registry.runnable()
    assert runnable
    for entry in registry.by_tier(0):
        assert entry.tier == 0 and entry.runnable()
    for entry in registry.by_category("math"):
        assert entry.category == "math"
    for entry in registry.active_frontier():
        assert entry.lifecycle == "ACTIVE_FRONTIER"
    assert set(BENCHMARK_STATUSES) >= {"RUNNABLE_PUBLIC", "PUBLIC_SCORE_ONLY"}
    assert "RETIRED" in LIFECYCLE_STATES and "SATURATED" in LIFECYCLE_STATES
    assert TIERS == frozenset({0, 1, 2, 3, 4})


def test_lookup_miss_is_explicit(registry):
    with pytest.raises(KeyError, match="not in registry"):
        registry.require("nonexistent_bench@9999")
