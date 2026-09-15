"""The frontier comparison system: five ambition levels, honest comparability.

Reference scores for published models are stored per ``benchmark@version``
with their protocol (harness, tool setting, reasoning setting, first/third
party, comparability confidence). Comparisons are only emitted where
protocols are sufficiently aligned; anything else renders
``NOT DIRECTLY COMPARABLE``.

Ambition levels (the ladder Chowder always knows where it sits on):

- LEVEL_0_FLOOR: Chowder Generation 0 -- going materially below is unacceptable.
- LEVEL_1_COMPARABLE_PEER: best open model of similar size (7-9B class).
- LEVEL_2_OPEN_WEIGHT_FRONTIER: best open-weight model regardless of size.
- LEVEL_3_STRETCH: substantially larger open models -- a stretch reference,
  never a promotion requirement under the 7-9B/local-compute constraint.
- LEVEL_4_ABSOLUTE_FRONTIER: strongest current systems from major labs.

Long-term north star: shrink ``frontier_gap`` -- the normalized gap to the
absolute frontier -- generation over generation. Promotion NEVER requires
closing that gap; frontier context is direction, not judgment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

LEVEL_0_FLOOR = "LEVEL_0_FLOOR"
LEVEL_1_COMPARABLE_PEER = "LEVEL_1_COMPARABLE_PEER"
LEVEL_2_OPEN_WEIGHT_FRONTIER = "LEVEL_2_OPEN_WEIGHT_FRONTIER"
LEVEL_3_STRETCH = "LEVEL_3_STRETCH"
LEVEL_4_ABSOLUTE_FRONTIER = "LEVEL_4_ABSOLUTE_FRONTIER"

ALL_LEVELS = (
    LEVEL_0_FLOOR,
    LEVEL_1_COMPARABLE_PEER,
    LEVEL_2_OPEN_WEIGHT_FRONTIER,
    LEVEL_3_STRETCH,
    LEVEL_4_ABSOLUTE_FRONTIER,
)

NOT_DIRECTLY_COMPARABLE = "NOT DIRECTLY COMPARABLE"


@dataclass(frozen=True)
class ReferenceScore:
    """One published score under one exact protocol."""

    model: str
    benchmark_qualified_id: str  # benchmark_id@version
    score: float  # normalized 0..1 where the metric permits
    level: str  # one of ALL_LEVELS
    date: str
    source_url: str
    harness: str  # reported evaluation harness
    tool_setting: str  # e.g. "none", "agentic-tools"
    reasoning_setting: str  # e.g. "direct", "extended-thinking"
    first_party: bool
    comparability_confidence: str  # HIGH | MEDIUM | LOW
    metric_scale: str = "normalized_0_1"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "benchmark_qualified_id": self.benchmark_qualified_id,
            "score": self.score,
            "level": self.level,
            "date": self.date,
            "source_url": self.source_url,
            "harness": self.harness,
            "tool_setting": self.tool_setting,
            "reasoning_setting": self.reasoning_setting,
            "first_party": self.first_party,
            "comparability_confidence": self.comparability_confidence,
            "metric_scale": self.metric_scale,
            "notes": self.notes,
        }


def compare_protocol(
    reference: ReferenceScore,
    *,
    benchmark_qualified_id: str,
    tool_setting: str,
    reasoning_setting: str,
) -> str:
    """Protocol-compatibility verdict for one comparison.

    Returns "COMPARABLE" only when benchmark version, tool setting, and
    reasoning setting align AND the reference carries HIGH confidence.
    Everything else -- silently using "latest", mixing raw-model with
    agent-harness numbers, third-party reproductions of unknown fidelity --
    returns NOT_DIRECTLY_COMPARABLE rather than a fake comparison.
    """
    if reference.benchmark_qualified_id != benchmark_qualified_id:
        return NOT_DIRECTLY_COMPARABLE
    if reference.comparability_confidence != "HIGH":
        return NOT_DIRECTLY_COMPARABLE
    if reference.tool_setting != tool_setting:
        return NOT_DIRECTLY_COMPARABLE
    if reference.reasoning_setting != reasoning_setting:
        return NOT_DIRECTLY_COMPARABLE
    return "COMPARABLE"


@dataclass(frozen=True)
class ChowderScore:
    """Chowder's own measured score, with the protocol it was run under."""

    generation_version: str
    benchmark_qualified_id: str
    score: float
    tool_setting: str = "none"
    reasoning_setting: str = "direct"
    metric_scale: str = "normalized_0_1"


@dataclass(frozen=True)
class GapRow:
    benchmark_qualified_id: str
    level: str
    model: str
    chowder_score: float
    reference_score: float
    gap: float  # reference - chowder (positive = Chowder behind)
    parity_ratio: float | None  # only where zero/one anchors make it meaningful
    comparability: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_qualified_id": self.benchmark_qualified_id,
            "level": self.level,
            "model": self.model,
            "chowder_score": self.chowder_score,
            "reference_score": self.reference_score,
            "gap": self.gap,
            "parity_ratio": self.parity_ratio,
            "comparability": self.comparability,
        }


def frontier_parity(chowder_score: float, reference_score: float) -> float | None:
    """chowder/reference where the scale makes a ratio meaningful.

    Meaningful only on a true zero-anchored normalized 0..1 scale. On
    metrics without a real zero (perplexity, Elo) the caller must use the
    absolute gap instead -- the ratio of two shifted scales is a lie.
    """
    if reference_score <= 0.0:
        return None
    return round(chowder_score / reference_score, 4)


class FrontierDatabase:
    """Persisted registry of published reference scores."""

    def __init__(self, root: Path | str) -> None:
        self._path = Path(root) / "frontier_reference_scores.json"
        self._scores: list[ReferenceScore] = []
        if self._path.exists():
            self._scores = [
                ReferenceScore(**item)
                for item in json.loads(self._path.read_text(encoding="utf-8"))
            ]

    def add(self, score: ReferenceScore) -> None:
        if score.level not in ALL_LEVELS:
            raise ValueError(f"unknown frontier level: {score.level}")
        self._scores.append(score)
        self._flush()

    def best_for_benchmark(self, benchmark_qualified_id: str, level: str) -> ReferenceScore | None:
        """Highest comparable-protocol reference at the level, else None."""
        candidates = [
            score
            for score in self._scores
            if score.level == level
            and score.benchmark_qualified_id == benchmark_qualified_id
            and score.comparability_confidence == "HIGH"
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda score: score.score)

    def scores(self) -> tuple[ReferenceScore, ...]:
        return tuple(self._scores)

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [score.to_dict() for score in self._scores]
        self._path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )


def gap_rows(
    database: FrontierDatabase,
    chowder: ChowderScore,
) -> list[GapRow]:
    """All protocol-honest gap rows for one Chowder score.

    Benchmarks with no comparable reference are omitted here -- the report
    layer renders them as ``? unavailable`` rather than inventing a gap.
    """
    rows: list[GapRow] = []
    for level in ALL_LEVELS:
        reference = database.best_for_benchmark(chowder.benchmark_qualified_id, level)
        if reference is None:
            continue
        comparability = compare_protocol(
            reference,
            benchmark_qualified_id=chowder.benchmark_qualified_id,
            tool_setting=chowder.tool_setting,
            reasoning_setting=chowder.reasoning_setting,
        )
        if comparability != "COMPARABLE":
            rows.append(
                GapRow(
                    benchmark_qualified_id=chowder.benchmark_qualified_id,
                    level=level,
                    model=reference.model,
                    chowder_score=chowder.score,
                    reference_score=reference.score,
                    gap=reference.score - chowder.score,
                    parity_ratio=None,
                    comparability=NOT_DIRECTLY_COMPARABLE,
                )
            )
            continue
        rows.append(
            GapRow(
                benchmark_qualified_id=chowder.benchmark_qualified_id,
                level=level,
                model=reference.model,
                chowder_score=chowder.score,
                reference_score=reference.score,
                gap=reference.score - chowder.score,
                parity_ratio=(
                    frontier_parity(chowder.score, reference.score)
                    if chowder.metric_scale == "normalized_0_1"
                    and reference.metric_scale == "normalized_0_1"
                    else None
                ),
                comparability="COMPARABLE",
            )
        )
    return rows


@dataclass(frozen=True)
class FrontierSnapshot:
    """The frontier as it existed when a generation shipped."""

    snapshot_id: str
    date: str
    scores: tuple[ReferenceScore, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "date": self.date,
            "scores": [score.to_dict() for score in self.scores],
        }


class SnapshotStore:
    """Frontier snapshots are frozen at generation time, never rewritten:

    distinguishing "Chowder improved" from "the frontier moved faster"
    requires preserving what the frontier looked like at each release.
    """

    def __init__(self, root: Path | str) -> None:
        self._path = Path(root) / "frontier_snapshots.json"
        self._snapshots: dict[str, FrontierSnapshot] = {}
        if self._path.exists():
            for item in json.loads(self._path.read_text(encoding="utf-8")):
                snapshot = FrontierSnapshot(
                    snapshot_id=item["snapshot_id"],
                    date=item["date"],
                    scores=tuple(
                        ReferenceScore(**score_item) for score_item in item["scores"]
                    ),
                )
                self._snapshots[snapshot.snapshot_id] = snapshot

    def freeze(
        self, snapshot_id: str, date: str, scores: tuple[ReferenceScore, ...]
    ) -> FrontierSnapshot:
        if snapshot_id in self._snapshots:
            raise ValueError(f"snapshot {snapshot_id} already exists (never rewrite)")
        snapshot = FrontierSnapshot(snapshot_id=snapshot_id, date=date, scores=scores)
        self._snapshots[snapshot_id] = snapshot
        self._flush()
        return snapshot

    def get(self, snapshot_id: str) -> FrontierSnapshot:
        if snapshot_id not in self._snapshots:
            raise KeyError(f"unknown snapshot: {snapshot_id}")
        return self._snapshots[snapshot_id]

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [snapshot.to_dict() for snapshot in self._snapshots.values()]
        self._path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )


@dataclass(frozen=True)
class CategoryGap:
    category: str
    parity: float | None  # mean parity across comparable rows
    mean_gap: float
    comparabilities: Mapping[str, str] = field(default_factory=dict)


def category_gaps(rows: Mapping[str, list[GapRow]], categories: Mapping[str, str]) -> CategoryGap:
    """Aggregate per-benchmark gap rows into a category frontier gap.

    ``rows`` maps benchmark_qualified_id -> gap rows; ``categories`` maps
    benchmark_qualified_id -> category name. Only COMPARABLE rows enter the
    aggregate -- tainted or non-comparable scores are excluded, not zeroed.
    """
    from collections import defaultdict

    by_benchmark_parities: dict[str, list[float]] = defaultdict(list)
    gaps: list[float] = []
    comparabilities: dict[str, str] = {}
    for benchmark_id, benchmark_rows in rows.items():
        for row in benchmark_rows:
            comparabilities[f"{row.level}:{benchmark_id}"] = row.comparability
            if row.comparability != "COMPARABLE":
                continue
            gaps.append(row.gap)
            if row.parity_ratio is not None:
                by_benchmark_parities[benchmark_id].append(row.parity_ratio)
    parities = [p for plist in by_benchmark_parities.values() for p in plist]
    category = categories.get(next(iter(rows), ""), "unknown")
    return CategoryGap(
        category=category,
        parity=round(sum(parities) / len(parities), 4) if parities else None,
        mean_gap=round(sum(gaps) / len(gaps), 4) if gaps else 0.0,
        comparabilities=comparabilities,
    )


def render_gap_report(
    generation_version: str,
    frontier_snapshot: FrontierSnapshot,
    rows: Mapping[str, list[GapRow]],
    *,
    generation_history: Mapping[str, Mapping[str, float]] | None = None,
) -> str:
    """Markdown FRONTIER_GAP_REPORT for one generation.

    ``generation_history`` optionally maps earlier version -> benchmark ->
    score, so the report can show trend (improving vs stalling) alongside
    the gap.
    """
    lines: list[str] = [f"# {generation_version} Frontier Gap", ""]
    lines.append(f"Frontier snapshot: `{frontier_snapshot.snapshot_id}` ({frontier_snapshot.date})")
    lines.append("")
    for benchmark_id in sorted(rows):
        lines.append(f"## {benchmark_id}")
        lines.append("")
        lines.append("| Level | Model | Chowder | Reference | Gap | Parity | Comparability |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in rows[benchmark_id]:
            parity = f"{row.parity_ratio:.0%}" if row.parity_ratio is not None else "--"
            score_render = f"{row.reference_score:.3f}" if row.comparability == "COMPARABLE" else "--"
            lines.append(
                f"| {row.level} | {row.model} | {row.chowder_score:.3f} "
                f"| {score_render} | {row.gap:+.3f} | {parity} | {row.comparability} |"
            )
        if generation_history:
            history = [
                (version, scores[benchmark_id])
                for version, scores in sorted(generation_history.items())
                if benchmark_id in scores
            ]
            if len(history) >= 2:
                delta = history[-1][1] - history[-2][1]
                trend = "improving" if delta > 0 else ("stalling" if delta == 0 else "regressing")
                lines.append("")
                lines.append(f"Trend vs previous generation: {trend} ({delta:+.3f})")
        lines.append("")
    return "\n".join(lines)
