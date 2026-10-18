"""Build the ``tool-use@reap-moe-v1`` instrument from the Nemotron agentic set.

Each instrument item is one conversation cut at the first assistant turn that
makes exactly one parseable tool call: the context before that turn is the
prompt, that call is the expected answer.  What the build enforces, not by
convention:

* the held-out rule -- any source row whose uuid satisfies the public
  :func:`chowder.growth.tool_use_instrument.is_reserved_for_evaluation`
  predicate is dropped, so a row a protected evaluation measures can never be
  lifted into the instrument;
* a leak check against the REAP calibration corpus -- an item any of whose
  message texts appears verbatim in the calibration text is dropped, because an
  item that *shapes the pruned weights* cannot also measure them;
* a per-tool cap -- no single tool may be more than ~10% of the instrument, so
  the score is not dominated by one tool's call shape;
* determinism -- fixed seed, sorted traversal, and no timestamp in the manifest,
  so two runs on the same inputs produce byte-identical outputs.

Usage:
    python chowder_batch/build_tool_use_instrument.py \
        --source <interactive_agent.jsonl> \
        --calibration <calibration.jsonl> \
        --out-dir src/chowder/growth/instruments
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

#: The committed instrument filenames, shared with
#: :mod:`chowder.growth.tool_use_instrument`.
ITEMS_NAME = "tool_use_reap_moe_v1.jsonl"
MANIFEST_NAME = "tool_use_reap_moe_v1.manifest.json"

#: Default source and calibration paths (the owner's approved local copies).
DEFAULT_SOURCE = r"C:\llm-work\data\nvidia__Nemotron-Agentic-v1\data\interactive_agent.jsonl"
DEFAULT_CALIBRATION = r"C:\llm-work\reap\calib-v2\calibration.jsonl"

#: Frozen defaults.  A build is reproduced by naming these, never by "whatever
#: the tree held that day".
DEFAULT_COUNT = 240
DEFAULT_SEED = 20261018

#: One in ten of the instrument may share a tool: 240 // 10.
CAP_FRACTION = 10

#: A message shorter than this is not treated as a leak on its own -- a greeting
#: such as "Hi" is not the conversation, and matching it would drop half the set.
MIN_LEAK_CHARS = 64


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- reading the source -----------------------------------------------------


def _one_call(content: Mapping[str, Any]) -> dict[str, Any] | None:
    """The single parseable tool call of an assistant turn, or ``None``.

    A turn with zero or several calls is not a single-call item; a call whose
    ``arguments`` are not a JSON object is not a usable call (the scorer compares
    argument objects, and a bare string has no keys to compare).
    """
    calls = content.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        return None
    function = calls[0].get("function") if isinstance(calls[0], Mapping) else None
    if not isinstance(function, Mapping):
        return None
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return None
    if not isinstance(arguments, Mapping):
        return None
    return {"name": name, "arguments": dict(arguments)}


def _cut_item(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """One instrument item from a source row, or ``None`` when none qualifies."""
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    seen_user = False
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        if role == "user":
            seen_user = True
            continue
        if role != "assistant" or not seen_user:
            continue
        call = _one_call(message)
        if call is None:
            continue
        context = [dict(m) for m in messages[:index] if isinstance(m, Mapping)]
        return {
            "uuid": str(row.get("uuid", "")),
            "turn_index": index,
            "tools": list(row.get("tools") or []),
            "messages": context,
            "expected": call,
        }
    return None


def _leak_text(item: Mapping[str, Any]) -> Iterable[str]:
    for message in item.get("messages", []):
        content = message.get("content")
        if isinstance(content, str) and len(content.strip()) >= MIN_LEAK_CHARS:
            yield content.strip()


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_instrument(
    *,
    source: str | Path,
    calibration: str | Path,
    out_dir: str | Path,
    count: int = DEFAULT_COUNT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Build the items file and manifest; return the manifest mapping.

    ``seed`` is accepted and recorded so a build names its sampling seed, but the
    sample itself is deterministic without randomness (round-robin over sorted
    tool names); the seed is provenance, not a shuffle.
    """
    from chowder.growth.tool_use_instrument import (
        HELD_OUT_RULE,
        INSTRUMENT_ID,
        is_reserved_for_evaluation,
    )

    source_path = Path(source)
    calibration_path = Path(calibration)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    source_rows = _read_jsonl(source_path)
    calibration_rows = _read_jsonl(calibration_path)
    calibration_blob = "\n".join(
        str(row.get("text", "")) for row in calibration_rows
    )

    reserved = 0
    candidates: list[dict[str, Any]] = []
    leak_dropped = 0
    for row in source_rows:
        if is_reserved_for_evaluation(row.get("uuid", "")):
            reserved += 1
            continue
        item = _cut_item(row)
        if item is None:
            continue
        if any(text in calibration_blob for text in _leak_text(item)):
            leak_dropped += 1
            continue
        candidates.append(item)

    cap = max(1, count // CAP_FRACTION)
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sorted(candidates, key=lambda i: (i["expected"]["name"], i["uuid"], i["turn_index"])):
        by_name[item["expected"]["name"]].append(item)

    selected: list[dict[str, Any]] = []
    taken: Counter[str] = Counter()
    names = sorted(by_name)
    depth = 0
    while len(selected) < count and any(
        depth < min(len(by_name[name]), cap) for name in names
    ):
        for name in names:
            if len(selected) >= count:
                break
            pool = by_name[name]
            if depth >= len(pool) or taken[name] >= cap:
                continue
            selected.append(pool[depth])
            taken[name] += 1
        depth += 1

    selected.sort(key=lambda i: (i["expected"]["name"], i["uuid"], i["turn_index"]))
    items = [
        {
            "uuid": item["uuid"],
            "tools": item["tools"],
            "messages": item["messages"],
            "expected": item["expected"],
        }
        for item in selected
    ]

    items_bytes = "".join(
        json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n" for item in items
    ).encode("utf-8")
    (out_path / ITEMS_NAME).write_bytes(items_bytes)

    manifest: dict[str, Any] = {
        "instrument": INSTRUMENT_ID,
        "seed": seed,
        "count": len(items),
        "cap": cap,
        "source": {
            "path": str(source_path),
            "sha256": _sha256_file(source_path),
            "rows": len(source_rows),
        },
        "calibration": {
            "path": str(calibration_path),
            "sha256": _sha256_file(calibration_path),
            "rows": len(calibration_rows),
        },
        "held_out_rule": HELD_OUT_RULE,
        "held_out": {"reserved_uuids": reserved},
        "leak_check": {
            "min_chars": MIN_LEAK_CHARS,
            "dropped": leak_dropped,
        },
        "candidates": len(candidates),
        "per_tool_counts": dict(sorted(taken.items())),
        "items_sha256": hashlib.sha256(items_bytes).hexdigest(),
    }
    (out_path / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    manifest = build_instrument(
        source=args.source,
        calibration=args.calibration,
        out_dir=args.out_dir,
        count=args.count,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
