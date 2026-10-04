"""Convert runtime transcripts into one signed training row per tool action.

The resulting rows keep the pre-action context, serialized action, observation,
event label, and scalar reward together.  This is deliberately separate from
trajectory-level outcome rewards: it lets a repair module learn not to issue a
specific bad read while still learning the successful completion policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from runtime_trace_reward import score_trace
except ModuleNotFoundError:  # package import in the test suite
    from .runtime_trace_reward import score_trace


def _action_text(turn: Mapping[str, Any]) -> str:
    return json.dumps(
        {"tool": turn.get("tool", ""), "args": turn.get("args", {})},
        sort_keys=True,
        ensure_ascii=False,
    )


def event_rows_from_trace(
    trace: Iterable[Mapping[str, Any]], *, trace_id: str = "trace"
) -> list[dict[str, Any]]:
    """Return one schema-stable row for each tool turn in ``trace``."""
    rows = list(trace)
    scored = score_trace(rows)
    action_rewards = scored["action_rewards"]
    result: list[dict[str, Any]] = []
    tool_index = 0
    for index, turn in enumerate(rows):
        if turn.get("role") != "tool":
            continue
        action_turn = rows[index - 1] if index > 0 and rows[index - 1].get("role") == "assistant" else None
        context_rows = rows[: index - 1] if action_turn is not None else rows[:index]
        context = json.dumps(context_rows, ensure_ascii=False, sort_keys=True)
        action = _action_text(turn)
        completion = str(action_turn.get("text", "")) if action_turn is not None else action
        observation = str(turn.get("observation", ""))
        event = action_rewards[tool_index]["event"]
        reward = float(action_rewards[tool_index]["reward"])
        result.append(
            {
                "id": f"{trace_id}-event-{tool_index:03d}",
                "trace_id": trace_id,
                "context": context,
                "action": action,
                "observation": observation,
                "completion": completion,
                "event": event,
                "reward": reward,
                # The completion suffix is the assistant's original action
                # message; post-action observations remain metadata, not targets.
                # ``text`` preserves the original event-reward training view.
                "text": f"Context:\n{context}\nAction:\n{action}\nObservation:\n{observation}",
                "sft_text": f"Context:\n{context}\nAction:\n{completion}",
            }
        )
        tool_index += 1
    return result


def _traces(path: Path) -> list[list[Mapping[str, Any]]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    payload = json.loads(text)
    if isinstance(payload, Mapping):
        payload = payload.get("verdict", payload)
        if isinstance(payload, Mapping):
            return [payload.get("trace", [])]
    if isinstance(payload, list) and payload and isinstance(payload[0], Mapping):
        if "trace" in payload[0]:
            return [row.get("trace", []) for row in payload]
        return [payload]
    return []


def build(source: str | Path, output: str | Path) -> int:
    traces = _traces(Path(source))
    rows = [row for index, trace in enumerate(traces) for row in event_rows_from_trace(trace, trace_id=f"trace-{index:03d}")]
    Path(output).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="runtime verdict JSON or JSONL trace set")
    parser.add_argument("output", help="event-reward JSONL output")
    args = parser.parse_args()
    count = build(args.source, args.output)
    print(f"wrote {count} event-level rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
