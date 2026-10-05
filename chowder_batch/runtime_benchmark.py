"""Deterministic multi-task runtime benchmark and promotion metrics.

This benchmark is independent of Spark: it exercises the same
read/write/test/final-report protocol on several small workspaces and exposes
runtime reward and nonexistent-read rate as numeric metrics.  The live Spark
loop uses the same trace and scoring schema.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable

try:
    from runtime_trace_reward import score_trace
except ModuleNotFoundError:  # package import in the test suite
    from .runtime_trace_reward import score_trace


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    goal: str
    files: dict[str, str]
    target: str
    fix: str
    test: Callable[[dict[str, str]], str]


def _version_task() -> BenchmarkTask:
    return BenchmarkTask(
        name="version_parser",
        goal="Repair version.py and observe the tests pass.",
        files={"version.py": "def parse_version(s):\n    return tuple(int(p) for p in s.split('.'))\n"},
        target="version.py",
        fix="def parse_version(s):\n    parts = s.split('.')\n    while len(parts) < 3: parts.append('0')\n    return tuple(int(p) for p in parts)\n",
        test=lambda files: "2 passed" if "while len(parts) < 3" in files.get("version.py", "") else "FAILED 2 - short version",
    )


def _sum_task() -> BenchmarkTask:
    return BenchmarkTask(
        name="sum_text",
        goal="Repair sum_text.py and observe the tests pass.",
        files={"sum_text.py": "def total(values):\n    return sum(values) - 1\n"},
        target="sum_text.py",
        fix="def total(values):\n    return sum(values)\n",
        test=lambda files: "3 passed" if "return sum(values)" in files.get("sum_text.py", "") else "FAILED 3 - total",
    )


def _slug_task() -> BenchmarkTask:
    return BenchmarkTask(
        name="slugify",
        goal="Repair slugify.py and observe the tests pass.",
        files={"slugify.py": "def slugify(value):\n    return value\n"},
        target="slugify.py",
        fix="def slugify(value):\n    return value.strip().lower().replace(' ', '-')\n",
        test=lambda files: "2 passed" if ".strip().lower()" in files.get("slugify.py", "") else "FAILED 2 - slug",
    )


TASKS = (_version_task(), _sum_task(), _slug_task())


def _turns(task: BenchmarkTask, *, bad_read: bool) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if bad_read:
        actions.append({"tool": "read_file", "args": {"path": "decoy/missing.py"}})
    actions.extend([
        {"tool": "read_file", "args": {"path": task.target}},
        {"tool": "write_file", "args": {"path": task.target, "content": task.fix}},
        {"tool": "run_tests", "args": {}},
    ])
    trace: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        name = action["tool"]
        args = action["args"]
        raw = "<tool_call>" + name + "".join(
            f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>" for key, value in args.items()
        ) + "</tool_call>"
        trace.append({"turn": index * 2, "role": "assistant", "text": raw})
        if name == "read_file":
            observation = task.files.get(args["path"], f"ERROR: no such file: {args['path']}")
        elif name == "write_file":
            task.files[args["path"]] = args["content"]
            observation = f"OK {args['path']} written"
        else:
            observation = task.test(task.files)
        trace.append({"turn": index * 2 + 1, "role": "tool", "tool": name, "args": args, "observation": observation})
    trace.extend([
        {"turn": len(trace), "role": "assistant", "text": "The observed test suite is green."},
        {"turn": len(trace) + 1, "role": "final_report", "text": "Fixed and verified green."},
    ])
    return trace


def run_benchmark(*, bad_read: bool = False) -> dict[str, Any]:
    """Run every task and return traces plus promotion metrics."""
    traces = []
    verdicts = []
    for task in TASKS:
        trace = _turns(task, bad_read=bad_read)
        scored = score_trace(trace)
        verdicts.append(scored)
        traces.append({"task": task.name, "trace": trace, "reward": scored})
    rewards = [row["reward"] for row in verdicts]
    nonexistent = [row["nonexistent_reads"] for row in verdicts]
    return {
        "traces": traces,
        "metrics": {
            "runtime_reward": mean(rewards),
            "runtime_green_rate": mean(bool(row["green_seen"]) for row in verdicts),
            "runtime_nonexistent_read_rate": mean(nonexistent),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bad-read", action="store_true")
    args = parser.parse_args()
    result = run_benchmark(bad_read=args.bad_read)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
