"""Build batch-008 action-only reward data from live failures and corrections.

Unlike batch-007, each row teaches one assistant decision after an explicit
observation.  The observation is a user message in the training transcript so
the existing chat labeler can mask it while supervising only the next action.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping

try:
    from runtime_eval import TASKS, RuntimeTask, TOOLS
except ModuleNotFoundError:
    from chowder.runtime_eval import TASKS, RuntimeTask, TOOLS

DEFAULT_TRACE = Path(r"F:\chowder-campaign\batch007-repair-only\.chowder\evals\baseline-3400d68d7020\eval-result.json")
DEFAULT_OUTPUT = Path(__file__).with_name("batch008_event_reward_train.jsonl")


def _assistant(text: str) -> dict[str, str]:
    return {"role": "assistant", "content": text}


def _user(text: str) -> dict[str, str]:
    return {"role": "user", "content": text}


def _call(name: str, args: Mapping[str, Any]) -> str:
    body = "".join(
        f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"
        for key, value in args.items()
    )
    return f"<tool_call>{name}{body}</tool_call>"


def _row(task: RuntimeTask, messages: list[dict[str, str]], reward: float, event: str, index: int) -> dict[str, Any]:
    return {
        "id": f"batch008-{task.name}-{index:03d}-{event}",
        "task": task.name,
        "event": event,
        "reward": float(reward),
        "messages": messages,
    }


def _messages_before(task: RuntimeTask, trace: list[Mapping[str, Any]], tool_index: int) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [_user(task.goal)]
    seen_tools = 0
    for turn in trace:
        if turn.get("role") == "tool":
            if seen_tools >= tool_index:
                break
            messages.append(_user(f"Observation: {turn.get('observation', '')}"))
            seen_tools += 1
        elif turn.get("role") == "assistant" and seen_tools < tool_index:
            messages.append(_assistant(str(turn.get("text", ""))))
    return messages


def _correct_trace(task: RuntimeTask) -> list[dict[str, Any]]:
    fix = {
        "version_parser": "def parse_version(s):\n    parts = s.split('.')\n    while len(parts) < 3: parts.append('0')\n    return tuple(int(p) for p in parts)\n",
        "sum_text": "def total(values):\n    return sum(values)\n",
        "slugify": "def slugify(value):\n    return value.strip().lower().replace(' ', '-')\n",
    }[task.name]
    actions = [
        ("read_file", {"path": task.target}),
        ("write_file", {"path": task.target, "content": fix}),
        ("run_tests", {}),
    ]
    trace: list[dict[str, Any]] = []
    for index, (name, args) in enumerate(actions):
        trace.append({"role": "assistant", "text": _call(name, args)})
        observation = task.initial.get(str(args.get("path", "")), "")
        if name == "write_file":
            observation = f"OK {task.target} written"
        elif name == "run_tests":
            observation = task.test_success
        trace.append({"role": "tool", "tool": name, "args": dict(args), "observation": observation})
    trace.extend([
        {"role": "assistant", "text": "The implementation is fixed and the observed suite is green."},
        {"role": "final_report", "text": "Fixed and verified green."},
    ])
    return trace


def _action_event(tool: Mapping[str, Any], task: RuntimeTask) -> tuple[str, float]:
    name = str(tool.get("tool", ""))
    observation = str(tool.get("observation", ""))
    args = tool.get("args", {})
    if name == "read_file" and "ERROR: no such file" in observation:
        return "nonexistent_read", -4.0
    if name == "read_file":
        return "existing_read", 1.0
    if name == "write_file":
        content = str(args.get("content", ""))
        return ("write", 4.0) if task.expected_fix in content else ("incorrect_write", -4.0)
    if name == "run_tests":
        return ("observed_green", 8.0) if "passed" in observation else ("red_test", -1.0)
    return "neutral", -1.0


def _rows_for_trace(task: RuntimeTask, trace: list[Mapping[str, Any]], prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tool_index = 0
    for turn in trace:
        if turn.get("role") != "tool":
            continue
        event, reward = _action_event(turn, task)
        action = _call(str(turn.get("tool", "")), turn.get("args", {}))
        messages = _messages_before(task, trace, tool_index) + [_assistant(action)]
        rows.append(_row(task, messages, reward, event, len(rows)))
        tool_index += 1
    if not rows or not any("passed" in str(turn.get("observation", "")) for turn in trace):
        messages = _messages_before(task, trace, tool_index) + [
            _assistant("The implementation is fixed and the observed suite is green.")
        ]
        rows.append(_row(task, messages, -4.0, "premature_or_unverified_completion", len(rows)))
    return rows


def build(trace_path: str | Path, output: str | Path, model_dir: str | Path = r"F:\Huihui-Spark-X2.5-4B-abliterated") -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True
    )
    payload = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    observed = payload.get("suites", {}).get("runtime_benchmark", [])
    observed_by_task = {row["task"]: row.get("trace", []) for row in observed}
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        rows.extend(_rows_for_trace(task, _correct_trace(task), "correct"))
        rows.extend(_rows_for_trace(task, observed_by_task.get(task.name, []), "observed"))
    for row in rows:
        row["text"] = tokenizer.apply_chat_template(
            row["messages"], tools=TOOLS, tokenize=False
        )
        row["completion"] = row["messages"][-1]["content"]
    random.Random(808).shuffle(rows)
    if not rows or any(row["reward"] == 0.0 for row in rows):
        raise RuntimeError("batch-008 event data contains no rows or a zero reward")
    Path(output).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default=str(DEFAULT_TRACE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model-dir", default=r"F:\Huihui-Spark-X2.5-4B-abliterated")
    args = parser.parse_args()
    count = build(args.trace, args.output, args.model_dir)
    print(f"wrote {count} action-only event rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
