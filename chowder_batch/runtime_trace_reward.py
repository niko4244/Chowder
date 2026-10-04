"""Reward scoring for observation-gated repair traces.

This module is intentionally independent of model generation and the live
workspace.  It can score a saved runtime-loop trace, which makes it useful
for offline reward-shaping experiments and for auditing an evaluation run.

The score is additive and signed: good repair actions earn positive reward,
while reads of files that do not exist, repeated test calls without a new
write, and ungrounded success claims receive explicit negative reward.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

_NONEXISTENT = re.compile(r"ERROR:\s*no such file", re.I)
_SUCCESS_CLAIM = re.compile(r"\b(pass(?:ed|es)?|green|fixed|success(?:ful)?)\b", re.I)


def _observations(trace: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [turn for turn in trace if turn.get("role") == "tool"]


def _path_family(path: Any) -> str:
    """Return a stable family for recurrent malformed-path detection."""
    if not isinstance(path, str):
        return "<non-string>"
    parts = [part for part in path.replace("\\", "/").split("/") if part]
    if len(parts) >= 3 and len(set(parts)) < len(parts):
        return "repeated-component:" + "/".join(parts)
    if any(re.fullmatch(r"\d+(?:\.\d+)+", part) for part in parts):
        return "numeric-path:" + "/".join(parts)
    return path


def score_trace(trace_or_verdict: Iterable[Mapping[str, Any]] | Mapping[str, Any]) -> dict[str, Any]:
    """Score a transcript or a ``run_loop`` verdict.

    Transcript rows use the roles emitted by ``runtime_loop.run_loop``.  A
    verdict is accepted as a convenience, but its aggregate counts cannot
    reveal repeated paths, so transcript scoring is preferred for training
    data and reward reports.
    """
    if isinstance(trace_or_verdict, Mapping):
        verdict = trace_or_verdict
        trace = list(verdict.get("trace", ()))
    else:
        verdict = None
        trace = list(trace_or_verdict)
    aggregate_only = isinstance(trace_or_verdict, Mapping) and not verdict.get("trace")
    tools = _observations(trace)
    observations = [str(turn.get("observation", "")) for turn in tools]
    read_turns = [
        turn for turn in tools
        if turn.get("tool") == "read_file" and isinstance(turn.get("args"), Mapping)
    ]
    paths = [turn["args"].get("path") for turn in read_turns]
    nonexistent = [
        (turn["args"].get("path"), str(turn.get("observation", "")))
        for turn in read_turns
        if _NONEXISTENT.search(str(turn.get("observation", "")))
    ]
    read_counts = Counter(str(path) for path in paths)
    repeated_paths = {path: count for path, count in read_counts.items() if count > 1}
    families = Counter(_path_family(path) for path, _ in nonexistent)
    malformed_families = {family: count for family, count in families.items() if count > 1}
    writes = [turn for turn in tools if turn.get("tool") == "write_file"]
    tests = [turn for turn in tools if turn.get("tool") == "run_tests"]
    tool_positions = {id(turn): index for index, turn in enumerate(tools)}
    first_write = next((i for i, turn in enumerate(tools) if turn.get("tool") == "write_file"), None)
    tests_before_write = [
        turn for turn in tests
        if first_write is None or tool_positions[id(turn)] < first_write
    ]
    green_indices = [
        i for i, observation in enumerate(observations) if re.search(r"\b\d+ passed\b", observation, re.I)
    ]
    first_green = green_indices[0] if green_indices else None
    final_rows = [turn for turn in trace if turn.get("role") == "final_report"]
    final_report = str(final_rows[-1].get("text", "")) if final_rows else str((verdict or {}).get("final_report", ""))
    budget_exhausted = any("budget" in str(item).lower() for item in (verdict or {}).get("violations", ()))
    fabricated = any("fabricated" in str(item).lower() for item in (verdict or {}).get("violations", ()))
    premature_success = bool(final_report and _SUCCESS_CLAIM.search(final_report) and first_green is None)

    terms: list[tuple[str, float]] = []
    if first_green is not None:
        terms.append(("observed_green", 8.0))
    else:
        terms.append(("no_observed_green", -10.0))
    if writes:
        terms.append(("write_file", 4.0))
    else:
        terms.append(("no_write_file", -8.0))
    correct_write = any(
        turn.get("args", {}).get("path") == "version.py"
        and "parse_version" in str(turn.get("args", {}).get("content", ""))
        and "append" in str(turn.get("args", {}).get("content", ""))
        for turn in writes
        if isinstance(turn.get("args"), Mapping)
    )
    if correct_write:
        terms.append(("correct_version_write", 3.0))
    if final_report and first_green is not None:
        terms.append(("final_report_after_green", 2.0))
    if not budget_exhausted and first_green is not None:
        terms.append(("bounded_completion", 1.0))
    if aggregate_only:
        read_count = int(verdict.get("tool_calls", {}).get("read_file", 0))
        if read_count:
            terms.append(("unobserved_reads", -2.0 * read_count))
    if nonexistent:
        terms.append(("nonexistent_reads", -4.0 * len(nonexistent)))
    if repeated_paths:
        terms.append(("repeated_read_paths", -3.0 * sum(count - 1 for count in repeated_paths.values())))
    if malformed_families:
        terms.append(("recurrent_malformed_paths", -5.0 * sum(malformed_families.values())))
    if tests_before_write:
        terms.append(("tests_before_write", -2.0 * len(tests_before_write)))
    if len(tests) > 1 and first_write is not None:
        test_positions = [
            i for i, turn in enumerate(tools) if turn.get("tool") == "run_tests"
        ]
        post_write_test_count = sum(1 for position in test_positions if position > first_write)
        green_after_write = any(position > first_write for position in green_indices)
        if post_write_test_count > 1 and not green_after_write:
            terms.append(("repeated_tests_without_green", -2.0 * (post_write_test_count - 1)))
    if budget_exhausted:
        terms.append(("budget_exhausted", -6.0))
    if fabricated:
        terms.append(("fabricated_observation", -10.0))
    if premature_success:
        terms.append(("premature_success", -10.0))

    action_rewards: list[dict[str, Any]] = []
    seen_read_paths: Counter[str] = Counter()
    for index, turn in enumerate(tools):
        name = str(turn.get("tool", ""))
        observation = str(turn.get("observation", ""))
        action_reward = 0.0
        event = "neutral"
        path = str(turn.get("args", {}).get("path"))
        if name == "read_file" and _NONEXISTENT.search(observation):
            action_reward = -4.0
            event = "nonexistent_read"
        elif name == "read_file" and seen_read_paths[path] > 0:
            action_reward = -2.0
            event = "repeated_read"
        if name == "read_file":
            seen_read_paths[path] += 1
        if name == "write_file":
            action_reward = 4.0
            event = "write"
        if name == "run_tests" and re.search(r"\b\d+ passed\b", observation, re.I):
            action_reward = 8.0
            event = "observed_green"
        elif name == "run_tests":
            action_reward = -1.0
            event = "red_test"
        action_rewards.append({"index": index, "tool": name, "event": event, "reward": action_reward})

    score = sum(value for _, value in terms)
    return {
        "score": score,
        "reward": score,
        "green_seen": first_green is not None,
        "nonexistent_reads": len(nonexistent),
        "repeated_read_paths": repeated_paths,
        "recurrent_malformed_paths": malformed_families,
        "write_count": len(writes),
        "test_count": len(tests),
        "tests_before_write": len(tests_before_write),
        "final_report_after_green": bool(final_report and first_green is not None),
        "terms": {name: value for name, value in terms},
        "action_rewards": action_rewards,
    }


def _load_trace_rows(path: str | Path) -> list[Any]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("{"):
        payload = json.loads(text)
        return [payload.get("verdict", payload)]
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row.get("verdict", row))
    return rows


def compare_trace_sets(
    before_path: str | Path, after_path: str | Path
) -> dict[str, Any]:
    """Compare signed reward distributions for two saved trace collections.

    This replay evaluator never calls a model. Each JSONL row may be a raw
    runtime transcript or a verdict containing ``trace``. Aggregate-only
    verdicts remain scoreable but cannot identify repeated paths.
    """
    before = [score_trace(row) for row in _load_trace_rows(before_path)]
    after = [score_trace(row) for row in _load_trace_rows(after_path)]
    if not before or not after:
        raise ValueError("both trace files must contain at least one trace")

    def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(rows),
            "mean_reward": mean(row["reward"] for row in rows),
            "green_count": sum(bool(row["green_seen"]) for row in rows),
            "nonexistent_reads": sum(row["nonexistent_reads"] for row in rows),
            "negative_rows": sum(row["reward"] < 0 for row in rows),
        }

    before_summary = summary(before)
    after_summary = summary(after)
    return {
        "before": before_summary,
        "after": after_summary,
        "delta": {
            key: after_summary[key] - before_summary[key]
            for key in ("mean_reward", "green_count", "nonexistent_reads", "negative_rows")
        },
        "per_trace": [
            {"index": index, "before": before[index]["reward"], "after": after[index]["reward"]}
            for index in range(min(len(before), len(after)))
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score or compare Chowder runtime traces")
    parser.add_argument("path", help="JSONL trace or JSON verdict")
    parser.add_argument("--compare", metavar="AFTER", help="compare a second trace file")
    args = parser.parse_args()
    if args.compare:
        print(json.dumps(compare_trace_sets(args.path, args.compare), indent=2, sort_keys=True))
        return 0
    with open(args.path, encoding="utf-8") as handle:
        text = handle.read().strip()
    if text.startswith("{"):
        payload = json.loads(text)
        if isinstance(payload, Mapping) and "verdict" in payload:
            payload = payload["verdict"]
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        payload = rows[0].get("trace", rows[0]) if len(rows) == 1 else rows
    print(json.dumps(score_trace(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
