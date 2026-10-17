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
    SCORER_POSTURES,
    TIERS,
    BenchmarkEntry,
    Normalization,
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
    # direction and normalization have no defaults on BenchmarkEntry: an entry
    # that never decided how its metric is scored cannot be promoted on, so it
    # cannot even be constructed. The fixture below declares them explicitly.
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
        direction="higher_is_better",
        normalization=Normalization(kind="identity"),
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


def test_the_declared_scorer_is_a_posture_and_not_the_readout_that_runs(registry):
    """R2: 42 declarations, zero reads -- so the claim is closed and separated.

    ``BenchmarkEntry.scorer`` is a human's word for how a benchmark is meant to be
    judged ("exact_match", "judge"). The executable readout is the evaluation
    material's ``scoring``, a different vocabulary the worker validates. Nothing
    maps one onto the other on purpose -- one posture is served by several
    readouts -- so the two sets must not overlap: a reader who takes "the scorer
    is exact_match" for the readout that ran would be reading a posture as a
    measurement.
    """
    from chowder.evaluators.scoring import OBSERVED_SCORINGS
    from chowder.evaluators.transformers_text import _ALLOWED_SCORING

    postures = {entry.scorer for entry in registry}
    # Equality, not membership: the constant and the catalog have to move
    # together, so a new posture in the catalog cannot appear without being
    # declared here (and this test saying so).
    assert postures == SCORER_POSTURES, sorted(postures ^ SCORER_POSTURES)
    readouts = set(_ALLOWED_SCORING) | set(OBSERVED_SCORINGS)
    assert readouts, "an empty readout vocabulary would make this check meaningless"
    # One word is in both vocabularies -- ``exact_match`` is a posture a reader is
    # told *and* a readout the worker runs -- and it means a different thing in
    # each: "this benchmark is judged by exact match" vs "score these items with
    # the exact_match rule". It is pinned exactly, so neither vocabulary can grow
    # toward the other without this test failing and the overlap being restated.
    assert postures & readouts == {"exact_match"}, sorted(postures & readouts)
    # The audit measured 42 declarations across the six postures.
    assert len(registry) >= 40
