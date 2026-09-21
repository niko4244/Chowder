"""Render the mechanical frontier gap report for Generation 0.

Read-only: loads the frozen Generation-0 eval report and the immutable
``gen0-frontier`` snapshot, then renders each measured benchmark's gap rows
through ``gap_rows`` -- the same machinery the promotion layer uses. Where
no reference exists at a level the row renders as ``unavailable``; nothing
is invented, normalized, or averaged.

Usage:
    PYTHONPATH=<worktree>/src python render_gap_report.py \
        --eval-report <path>/freeze/eval-report.json \
        --snapshot-store <path>/freeze \
        --out FRONTIER_GAP_REPORT.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src").as_posix()))

from chowder.evals.result import SUPPORTED, EvalReport  # noqa: E402
from chowder.growth.frontier_reference import (  # noqa: E402
    ALL_LEVELS,
    ChowderScore,
    SnapshotStore,
    context_rows,
    gap_rows,
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-report", required=True)
    parser.add_argument("--snapshot-store", required=True)
    parser.add_argument("--snapshot-id", default="gen0-frontier")
    parser.add_argument("--out", default="FRONTIER_GAP_REPORT.md")
    args = parser.parse_args()

    report = EvalReport.load(args.eval_report)
    store = SnapshotStore(args.snapshot_store)
    try:
        snapshot = store.get(args.snapshot_id)
    except KeyError:
        print(f"snapshot {args.snapshot_id!r} does not exist; refusing to improvise", file=sys.stderr)
        return 2

    references_by_benchmark: dict[str, int] = {}
    for score in snapshot.scores:
        references_by_benchmark[score.benchmark_qualified_id] = (
            references_by_benchmark.get(score.benchmark_qualified_id, 0) + 1
        )

    from chowder.growth.frontier_reference import FrontierDatabase

    import tempfile

    database = FrontierDatabase(Path(tempfile.mkdtemp(prefix="gen0-gap-")))
    for score in snapshot.scores:
        try:
            database.add(score)
        except ValueError as error:
            print(f"snapshot score failed provenance validation: {error}", file=sys.stderr)
            return 2

    # Context comes from every frozen snapshot (they are additive and
    # immutable), so a later-dated enrichment is visible without any
    # reference row ever becoming gate-eligible: `gap_rows` still reads only
    # the generation-time snapshot built above.
    context_database = FrontierDatabase(Path(tempfile.mkdtemp(prefix="gen0-context-")))
    context_snapshots: list[str] = []
    for snapshot_id in _snapshot_ids(args.snapshot_store):
        try:
            extra = store.get(snapshot_id)
        except KeyError:
            continue
        added = 0
        for score in extra.scores:
            try:
                context_database.add(score)
                added += 1
            except ValueError:
                continue  # duplicates across snapshots are expected
        if added:
            context_snapshots.append(snapshot_id)

    lines = [
        "# Generation-0 Frontier Gap Report",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()} from the frozen",
        f"eval report `{Path(args.eval_report).resolve()}` and the immutable",
        f"snapshot `{args.snapshot_id}` ({len(snapshot.scores)} reference scores).",
        f"Context rows are drawn from every frozen snapshot: "
        f"{', '.join(context_snapshots) if context_snapshots else 'none'}",
        "",
        "Mechanical output of `gap_rows` (gate-eligible rows) followed by a",
        "cited reference-context table (`context_rows`). No reference at a level",
        "renders as `unavailable`; nothing is normalized into fake parity, and",
        "context rows never enter a gap, a parity ratio, or a promotion input.",
        "",
    ]
    measured = [run for run in report.runs if run.support == SUPPORTED and run.score is not None]
    if not measured:
        lines.append("_No measured scored runs in the eval report._")
    any_comparable = False
    for run in measured:
        lines.append(f"## {run.benchmark_qualified_id} — Chowder {run.score:.4f}")
        lines.append("")
        rows = gap_rows(
            database,
            ChowderScore(
                generation_version=report.generation_version,
                benchmark_qualified_id=run.benchmark_qualified_id,
                score=run.score,
                tool_setting=run.tool_setting,
                reasoning_setting=run.reasoning_setting,
            ),
        )
        if not rows:
            lines.append("| Level | Reference | Gap | Comparability |")
            lines.append("| --- | --- | --- | --- |")
            for level in ALL_LEVELS:
                lines.append(f"| {level} | unavailable | -- | unavailable |")
            lines.append("")
        else:
            lines.append("| Level | Model | Reference | Gap | Parity | Comparability |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for row in rows:
                reference = (
                    f"{row.reference_score:.3f}"
                    if row.comparability == "COMPARABLE"
                    else "--"
                )
                parity = f"{row.parity_ratio:.0%}" if row.parity_ratio is not None else "--"
                if row.comparability == "COMPARABLE":
                    any_comparable = True
                lines.append(
                    f"| {row.level} | {row.model} | {reference} | {row.gap:+.3f} "
                    f"| {parity} | {row.comparability} |"
                )
            lines.append("")

        context = context_rows(
            context_database,
            ChowderScore(
                generation_version=report.generation_version,
                benchmark_qualified_id=run.benchmark_qualified_id,
                score=run.score,
                tool_setting=run.tool_setting,
                reasoning_setting=run.reasoning_setting,
            ),
        )
        if context:
            lines.append(
                "Reference context — cited published numbers that are **not** "
                "gate-eligible, listed for orientation only:"
            )
            lines.append("")
            lines.append(
                "| Level | Model | Reference | Confidence | Harness | Blocked by |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for row in context:
                blocked = ", ".join(row.divergence) or "--"
                lines.append(
                    f"| {row.level} | {row.model} | {row.reference_score:.3f} "
                    f"| {row.comparability_confidence} | {row.harness} | {blocked} |"
                )
            lines.append("")
            lines.append(
                "These rows do not contribute a gap, a parity ratio, or a "
                "promotion input; protocol divergence is named in the last column."
            )
            lines.append("")

    if not any_comparable:
        lines.append(
            "**No protocol-comparable reference exists for any measured benchmark.**"
            " Every published number found for these benchmarks uses a different"
            " harness/shot/extraction protocol (see docs/FRONTIER_REFERENCE_SEED_2026-09-17.md);"
            " Chowder records that honestly instead of rendering fake gaps. The"
            " reference-context tables above name what exists and why it cannot"
            " be compared."
        )
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


def _snapshot_ids(root: str) -> tuple[str, ...]:
    """Snapshot ids recorded in the store, in file order (oldest first)."""
    path = Path(root) / "frontier_snapshots.json"
    if not path.exists():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(item["snapshot_id"] for item in payload)


if __name__ == "__main__":
    raise SystemExit(main())
