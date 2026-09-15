"""Evaluation runner and scoreboard for the growth system.

The runner executes an eval plan against the available adapters and produces
an ``EvalReport`` -- including honest non-measurements for benchmarks whose
harness or modality is unavailable. The scoreboard renders the report:
category tables, contamination markers, the raw-vs-harness distinction, and
the arrow language for generation deltas:

  ↑ improved   → statistically flat   ↓ regressed   ? unavailable
  N/A unsupported   ⚠ contaminated/non-comparable
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from chowder.growth.benchmark_registry import BenchmarkRegistry
from chowder.growth.frontier_reference import ChowderScore, FrontierDatabase, gap_rows
from chowder.growth.statistics import compare

from .result import (
    AGENT_HARNESS,
    NOT_APPLICABLE_MODALITY,
    SUPPORTED,
    EvalReport,
)

UP = "↑"
FLAT = "→"
DOWN = "↓"
UNKNOWN_MARK = "?"
NA_MARK = "N/A"
TAINTED_MARK = "⚠"


@dataclass(frozen=True)
class RunnerHooks:
    """Adapter dispatch supplied by the caller: benchmark id -> callable that
    executes it and returns a BenchmarkRun. Keeps the runner free of any
    concrete harness dependency."""

    executors: Mapping[str, Callable[[], object]]
    modality_support: Mapping[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.modality_support is None:
            object.__setattr__(self, "modality_support", {})


class EvalRunner:
    """Execute a benchmark list into a normalized EvalReport."""

    def __init__(self, hooks: RunnerHooks, registry: BenchmarkRegistry | None = None) -> None:
        self._hooks = hooks
        self._registry = registry

    def run(
        self,
        benchmark_ids: Sequence[str],
        generation_version: str,
        *,
        hardware: Mapping[str, object] | None = None,
        date: str = "",
    ) -> EvalReport:
        runs = []
        for benchmark_id in benchmark_ids:
            support = self._hooks.modality_support.get(benchmark_id, SUPPORTED)
            if support != SUPPORTED:
                runs.append(
                    _na_run(benchmark_id, generation_version, support)
                )
                continue
            executor = self._hooks.executors.get(benchmark_id)
            if executor is None:
                runs.append(
                    _unknown_run(benchmark_id, generation_version, "no executor registered")
                )
                continue
            runs.append(executor())  # type: ignore[arg-type]
        return EvalReport(
            generation_version=generation_version,
            runs=tuple(runs),
            hardware=dict(hardware or {}),
            date=date,
        )


def _na_run(benchmark_id: str, generation_version: str, support: str):
    from .result import BenchmarkRun  # noqa: PLC0415

    return BenchmarkRun(
        benchmark_qualified_id=benchmark_id,
        adapter="none",
        generation_version=generation_version,
        score=None,
        support=support,
        notes="benchmark not applicable to this model's modality/harness",
    )


def _unknown_run(benchmark_id: str, generation_version: str, reason: str):
    from .result import BenchmarkRun  # noqa: PLC0415

    return BenchmarkRun(
        benchmark_qualified_id=benchmark_id,
        adapter="none",
        generation_version=generation_version,
        score=None,
        support="UNKNOWN",
        notes=reason,
    )


class Scoreboard:
    """Render an EvalReport as markdown with honest marks."""

    def __init__(
        self,
        registry: BenchmarkRegistry,
        *,
        frontier: FrontierDatabase | None = None,
    ) -> None:
        self._registry = registry
        self._frontier = frontier

    def render(
        self,
        report: EvalReport,
        *,
        contamination: Mapping[str, object] | None = None,
        parent_report: EvalReport | None = None,
    ) -> str:
        """``contamination`` is the payload of the firewall's
        ``ContaminationFirewall.manifest()`` -- its ``benchmarks`` section maps
        benchmark id -> {"status": CLEAN|POSSIBLE|KNOWN_CONTAMINATION|UNKNOWN}."""
        lines: list[str] = [f"# {report.generation_version} — Scoreboard", ""]
        raw_table = self._render_table(report, contamination)
        lines.extend(raw_table)

        if parent_report is not None:
            lines.append("")
            lines.append("## vs previous generation")
            lines.append("")
            lines.extend(self._render_deltas(report, parent_report))

        if self._frontier is not None:
            lines.append("")
            lines.append("## vs frontier references")
            lines.append("")
            lines.extend(self._render_frontier(report))
        return "\n".join(lines) + "\n"

    def _render_table(
        self,
        report: EvalReport,
        contamination: Mapping[str, object] | None,
    ) -> list[str]:
        benchmarks_section: Mapping[str, Mapping[str, object]] = (
            (contamination or {}).get("benchmarks", {})  # type: ignore[assignment]
        )
        lines = [
            "| Benchmark | Category | Score | Kind | Status |",
            "| --- | --- | --- | --- | --- |",
        ]
        for run in report.runs:
            entry = self._registry.get(run.benchmark_qualified_id)
            category = entry.category if entry is not None else "?"
            if run.support == SUPPORTED and run.score is not None:
                score = f"{run.score:.3f}"
            elif run.support == NOT_APPLICABLE_MODALITY:
                score = NA_MARK
            else:
                score = UNKNOWN_MARK
            status = run.support
            assessment = benchmarks_section.get(run.benchmark_qualified_id)
            if assessment is not None and assessment.get("status") != "CLEAN":
                status = f"{TAINTED_MARK} {assessment.get('status')}"
            kind = run.measurement_kind
            lines.append(f"| {run.benchmark_qualified_id} | {category} | {score} | {kind} | {status} |")
        return lines

    def _render_deltas(
        self, report: EvalReport, parent_report: EvalReport
    ) -> list[str]:
        parent_by_id = {run.benchmark_qualified_id: run for run in parent_report.runs}
        lines = ["| Benchmark | Parent | Candidate | Delta | Verdict |", "| --- | --- | --- | --- | --- |"]
        for run in report.runs:
            if run.support != SUPPORTED or run.score is None:
                continue
            parent_run = parent_by_id.get(run.benchmark_qualified_id)
            if parent_run is None or parent_run.score is None:
                lines.append(
                    f"| {run.benchmark_qualified_id} | {UNKNOWN_MARK} | {run.score:.3f} "
                    f"| -- | {UNKNOWN_MARK} unavailable |"
                )
                continue
            delta = run.score - parent_run.score
            verdict = FLAT
            if run.per_sample_scores and parent_run.per_sample_scores:
                comparison = compare(parent_run.per_sample_scores, run.per_sample_scores)
                if comparison.verdict == "improved":
                    verdict = UP
                elif comparison.verdict == "regressed":
                    verdict = DOWN
            else:
                verdict = f"{FLAT} (no per-sample stats)"
            lines.append(
                f"| {run.benchmark_qualified_id} | {parent_run.score:.3f} | {run.score:.3f} "
                f"| {delta:+.3f} | {verdict} |"
            )
        return lines

    def _render_frontier(self, report: EvalReport) -> list[str]:
        assert self._frontier is not None
        lines: list[str] = []
        for run in report.runs:
            if run.support != SUPPORTED or run.score is None:
                continue
            rows = gap_rows(
                self._frontier,
                ChowderScore(
                    generation_version=report.generation_version,
                    benchmark_qualified_id=run.benchmark_qualified_id,
                    score=run.score,
                    tool_setting=run.tool_setting,
                    reasoning_setting=run.reasoning_setting,
                ),
            )
            if not rows:
                continue
            lines.append(f"### {run.benchmark_qualified_id}")
            lines.append("")
            lines.append("| Level | Model | Reference | Gap | Parity | Comparability |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for row in rows:
                reference = (
                    f"{row.reference_score:.3f}"
                    if row.comparability == "COMPARABLE"
                    else "--"
                )
                parity = f"{row.parity_ratio:.0%}" if row.parity_ratio is not None else "--"
                lines.append(
                    f"| {row.level} | {row.model} | {reference} | {row.gap:+.3f} "
                    f"| {parity} | {row.comparability} |"
                )
            lines.append("")
        return lines


def category_aggregates(
    report: EvalReport, categories: Mapping[str, str]
) -> dict[str, float]:
    """Mean supported score per category. Unsupported modalities are excluded,
    not zeroed -- a text-only model's computer-use row cannot drag down the
    coding aggregate it was never measuring."""
    totals: dict[str, list[float]] = {}
    for run in report.supported_runs():
        if run.score is None:
            continue
        category = categories.get(run.benchmark_qualified_id)
        if category is None:
            continue
        totals.setdefault(category, []).append(run.score)
    return {
        category: sum(scores) / len(scores)
        for category, scores in sorted(totals.items())
        if scores
    }


__all__ = [
    "AGENT_HARNESS",
    "DOWN",
    "EvalRunner",
    "FLAT",
    "NA_MARK",
    "RunnerHooks",
    "Scoreboard",
    "SUPPORTED",
    "TAINTED_MARK",
    "UNKNOWN_MARK",
    "UP",
    "category_aggregates",
]
