"""Frontier benchmark registry.

One entry per benchmark family member with a PINNED version -- results are
always recorded as ``benchmark_id@version`` and the registry refuses vague
"latest" versions. Entries carry the metadata the dashboard and promotion
rules need: category, scorer, splits (which are protected, which may
support development), contamination risk, and availability status.

Availability statuses:

- RUNNABLE_PUBLIC: Chowder can execute the benchmark end to end (data
  public, license permits, an adapter exists or can be written).
- PUBLIC_SCORE_ONLY: data is private (or compute/harness requirements are
  impractical locally), but published scores exist for reference display.
- PRIVATE: neither data nor protocol is public; reference display only.
- INTERNAL_REFERENCE_ONLY: kept for historical comparison of published
  numbers, never run by Chowder.
- UNSUPPORTED_BY_CURRENT_MODALITY: requires a modality Chowder's current
  target models do not have (recorded as N/A, never scored as zero).

Lifecycle states: ACTIVE_FRONTIER / ACTIVE_DIAGNOSTIC / LEGACY / SATURATED /
RETIRED. Optimization against RETIRED or SATURATED benchmarks is refused.

The seed catalog lives in :mod:`chowder.growth.catalog`; this module holds
the machinery and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

BENCHMARK_STATUSES = frozenset(
    {
        "RUNNABLE_PUBLIC",
        "PUBLIC_SCORE_ONLY",
        "PRIVATE",
        "INTERNAL_REFERENCE_ONLY",
        "UNSUPPORTED_BY_CURRENT_MODALITY",
    }
)

LIFECYCLE_STATES = frozenset(
    {
        "ACTIVE_FRONTIER",
        "ACTIVE_DIAGNOSTIC",
        "LEGACY",
        "SATURATED",
        "RETIRED",
    }
)

#: Split protection defaults. A "protected" split is firewalled from the
#: curriculum/data pipeline (never sampled, never read by training-side code).
#: A "dev" split may be inspected for diagnosis but must never be copied into
#: training data. "none" = no split distinction exists for this benchmark.
SPLIT_POLICY = frozenset({"protected", "dev", "none"})

#: Evaluation tiers (see evals/runner): which battery a benchmark belongs to.
TIERS = frozenset({0, 1, 2, 3, 4})

CATEGORIES = frozenset(
    {
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
)


@dataclass(frozen=True)
class BenchmarkEntry:
    """One pinned benchmark in the registry."""

    benchmark_id: str
    version: str  # pinned; "latest" is refused
    name: str
    category: str  # one of CATEGORIES
    subcategory: str
    status: str  # BENCHMARK_STATUSES
    lifecycle: str  # LIFECYCLE_STATES
    tier: int  # 0..4
    scorer: str  # exact_match | multiple_choice | unit_tests | judge | agent | ...
    primary_metric: str  # accuracy | pass@1 | success_rate | ...
    skills: tuple[str, ...]  # capability.ALL_SKILLS paths
    dataset_source: str
    implementation_source: str
    source: str  # who publishes/owns the benchmark
    license: str
    release_date: str  # ISO date of this version
    modality: str = "text"
    context_requirements: str = ""
    tool_requirements: str = ""
    environment_requirements: str = ""
    split_policy: str = "protected"  # SPLIT_POLICY
    random_baseline: float | None = None
    human_baseline: float | None = None
    frontier_reference: float | None = None
    frontier_reference_source: str = ""
    contamination_risk: str = "medium"  # high | medium | low
    training_use_permitted: bool = False  # may Chowder train on its data?
    notes: str = ""
    adapter: str = ""  # evals adapter that runs it (empty = none yet)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from .capability import assert_known_skills

        if self.version.strip().lower() in {"latest", "current", ""}:
            raise ValueError(
                f"{self.benchmark_id}: version must be pinned (got {self.version!r}); "
                "'latest' is never silently resolved"
            )
        if self.status not in BENCHMARK_STATUSES:
            raise ValueError(f"{self.benchmark_id}: unknown status {self.status!r}")
        if self.lifecycle not in LIFECYCLE_STATES:
            raise ValueError(f"{self.benchmark_id}: unknown lifecycle {self.lifecycle!r}")
        if self.tier not in TIERS:
            raise ValueError(f"{self.benchmark_id}: tier must be 0..4, got {self.tier}")
        if self.category not in CATEGORIES:
            raise ValueError(f"{self.benchmark_id}: unknown category {self.category!r}")
        if self.split_policy not in SPLIT_POLICY:
            raise ValueError(
                f"{self.benchmark_id}: split_policy must be one of {sorted(SPLIT_POLICY)}"
            )
        if not self.skills:
            raise ValueError(f"{self.benchmark_id}: must map to at least one skill")
        assert_known_skills(self.skills, context=f"benchmark {self.benchmark_id}")
        if self.status == "RUNNABLE_PUBLIC" and not self.adapter:
            raise ValueError(
                f"{self.benchmark_id}: RUNNABLE_PUBLIC requires an adapter name"
            )
        if (
            self.training_use_permitted
            and self.split_policy == "protected"
            and self.status == "RUNNABLE_PUBLIC"
        ):
            # A runnable benchmark whose test split is protected may still allow
            # training on an explicitly non-protected split (e.g. a train split);
            # that must be declared via extra['training_split'].
            if "training_split" not in self.extra:
                raise ValueError(
                    f"{self.benchmark_id}: training_use_permitted on a protected runnable "
                    "benchmark requires extra['training_split'] naming the usable split"
                )
        if self.status == "UNSUPPORTED_BY_CURRENT_MODALITY" and self.tier in {0, 1, 2}:
            raise ValueError(
                f"{self.benchmark_id}: modality-unsupported benchmarks cannot sit in "
                "tiers 0-2 (they are never run by local batteries)"
            )

    @property
    def qualified_id(self) -> str:
        """The version-qualified identity used in every score record."""
        return f"{self.benchmark_id}@{self.version}"

    def runnable(self) -> bool:
        return self.status == "RUNNABLE_PUBLIC"

    def may_score_against(self) -> bool:
        """May Chowder display/compare scores for this benchmark?"""
        return self.status in {
            "RUNNABLE_PUBLIC",
            "PUBLIC_SCORE_ONLY",
            "UNSUPPORTED_BY_CURRENT_MODALITY",
        }

    def optimizable(self) -> bool:
        """May curriculum selection target this benchmark?"""
        return self.lifecycle in {"ACTIVE_FRONTIER", "ACTIVE_DIAGNOSTIC"} and self.runnable()

    def to_dict(self) -> dict[str, Any]:
        data = {
            f: getattr(self, f)
            for f in self.__dataclass_fields__  # type: ignore[attr-defined]
        }
        data = {k: (list(v) if isinstance(v, tuple) else v) for k, v in data.items()}
        data["extra"] = dict(self.extra)
        return data


class BenchmarkRegistry:
    """The immutable set of benchmarks Chowder knows about."""

    def __init__(self, entries: tuple[BenchmarkEntry, ...] = ()):
        self._entries: dict[str, BenchmarkEntry] = {}
        for entry in entries:
            key = entry.qualified_id
            if key in self._entries:
                raise ValueError(f"duplicate benchmark entry: {key}")
            self._entries[key] = entry

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries.values())

    def get(self, qualified_id: str) -> BenchmarkEntry | None:
        return self._entries.get(qualified_id)

    def require(self, qualified_id: str) -> BenchmarkEntry:
        entry = self._entries.get(qualified_id)
        if entry is None:
            known = ", ".join(sorted(self._entries)[:12])
            raise KeyError(f"benchmark {qualified_id!r} not in registry; known include: {known}")
        return entry

    def by_id_any_version(self, benchmark_id: str) -> tuple[BenchmarkEntry, ...]:
        return tuple(e for e in self if e.benchmark_id == benchmark_id)

    def runnable(self) -> tuple[BenchmarkEntry, ...]:
        return tuple(e for e in self if e.runnable())

    def by_tier(self, tier: int) -> tuple[BenchmarkEntry, ...]:
        return tuple(e for e in self if e.tier == tier and e.runnable())

    def by_category(self, category: str) -> tuple[BenchmarkEntry, ...]:
        return tuple(e for e in self if e.category == category)

    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({e.category for e in self}))

    def active_frontier(self) -> tuple[BenchmarkEntry, ...]:
        return tuple(e for e in self if e.lifecycle == "ACTIVE_FRONTIER")

    def optimizable(self) -> tuple[BenchmarkEntry, ...]:
        return tuple(e for e in self if e.optimizable())

    def to_dicts(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self]


def _entry(row: Mapping[str, Any]) -> BenchmarkEntry:
    """Catalog row -> BenchmarkEntry (drops None-valued optional fields)."""
    row = dict(row)
    extra = dict(row.pop("extra", {}) or {})
    extra = {k: v for k, v in extra.items() if v is not None}
    optional = (
        "random_baseline",
        "human_baseline",
        "frontier_reference",
        "frontier_reference_source",
    )
    for key in optional:
        if row.get(key) is None:
            row.pop(key, None)
    for key in ("modality", "context_requirements", "tool_requirements",
                "environment_requirements", "notes", "adapter"):
        if row.get(key) is None:
            row.pop(key, None)
    return BenchmarkEntry(**row, extra=extra)
