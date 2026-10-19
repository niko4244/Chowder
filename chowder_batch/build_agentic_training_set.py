"""Build the agentic-v1 SFT sets: the Fable arm and its mandatory Fable-free control.

Two files, one discipline:

``train-fable.jsonl``
    the three owner-approved Fable agentic sources (glint local, lordx64,
    swarm), cross-source deduped by ``FableAgenticProvider``, plus the
    Nemotron rows.
``train-control.jsonl``
    the *same* Nemotron rows and no Fable row at all -- the N2 control arm
    ``docs/REAP_MOE_LINEAGE.md`` requires until the Fable/Anthropic usage
    question is recorded.

The two arms differ by the Fable rows and nothing else, so a difference between
them measures the Fable exposure and not a sampling artefact.

Never training material, per the lineage charter: protected evaluation items
(GSM8K / MATH-500 / MGSM / HumanEval test splits and the
``tool-use@reap-moe-v1`` items), and every Nemotron row the evaluation split
reserves via ``is_reserved_for_evaluation``. Both are dropped *and counted*,
never silently.

Row shape is exactly what the worker's chat path reads (see the backend config
in ``docs/AGENTIC_TRAINING_SET_V1.md``)::

    {"messages": [...], "tools": "<json string>" or ""}

Tool-call ``function.arguments`` are JSON **strings**, not objects: ``datasets``
loads JSONL through Arrow, and Arrow unifies nested dicts across rows, which
would add every other row's argument keys to this one as nulls
(``training_data._parse_tool_calls``).

Usage:
    python chowder_batch/build_agentic_training_set.py --out C:\\llm-work\\train-sets\\agentic-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # a batch script, not an installed console entry
    sys.path.insert(0, str(ROOT / "src"))

from chowder.backends.training_data import (  # noqa: E402
    _build_chat_example,
    _validate_chat_messages,
    parse_tools,
)
from chowder.growth.agentic_providers import (  # noqa: E402
    FableAgenticProvider,
    _native_tool_calls,
    is_reserved_for_evaluation,
)
from chowder.growth.contamination import ContaminationFirewall  # noqa: E402
from chowder.growth.tool_use_instrument import INSTRUMENT_ID, load_items  # noqa: E402

#: Frozen defaults: the owner's approved local copies and the train-set root.
DEFAULT_OUT = r"C:\llm-work\train-sets\agentic-v1"
DEFAULT_TOKENIZER = r"G:\llm-cold\bf16\Qwen3.6-35B-A3B"
DEFAULT_GLINT = r"C:\Users\nikma\frontier-lowram-autoresearch\training\data\agentic_sft_train.jsonl"
DEFAULT_LORDX64 = r"C:\llm-work\data\lordx64__fable-tool-use-sft\data\train-00000-of-00001.parquet"
DEFAULT_SWARM = r"C:\llm-work\data\Swarm-AI-Research__fable5-traces-sft\data\train.jsonl"
DEFAULT_NEMOTRON = r"C:\llm-work\data\nvidia__Nemotron-Agentic-v1\data\interactive_agent.jsonl"

DEFAULT_SEED = 20261018
DEFAULT_FABLE_ROWS = 1500
DEFAULT_NEMOTRON_ROWS = 1500
DEFAULT_MAX_LENGTH = 4096

#: Recorded per source; the Fable sources are AGPL-3.0 (owner decision N1), and
#: the Anthropic usage question on Fable output is decision N2, still open --
#: which is exactly why the control arm exists.
LICENCES = {
    "glint_local": "AGPL-3.0 (Fable 5 output; N2 Anthropic usage question open)",
    "lordx64": "AGPL-3.0 (lordx64/fable-tool-use-sft, Fable 5 output)",
    "swarm": "AGPL-3.0 (Swarm-AI-Research/fable5-traces-sft, Fable 5 output)",
    "nemotron": "CC-BY-4.0 (nvidia/Nemotron-Agentic-v1, interactive_agent)",
}

#: The protected evaluation material every candidate row is checked against.
#: Fields mirror ``chowder_batch/reap_calibration.protected_ngrams``.
_PROTECTED_SPECS: tuple[tuple[str, str, str | None, str, tuple[str, ...]], ...] = (
    ("gsm8k@2021-09", "openai/gsm8k", "main", "test", ("question", "answer")),
    ("math500@2024-04", "HuggingFaceH4/MATH-500", None, "test", ("problem", "solution", "answer")),
    ("mgsm@2022-11", "juletxara/mgsm", "en", "test", ("question", "answer")),
    ("humaneval@2021-07", "openai/openai_humaneval", None, "test",
     ("prompt", "canonical_solution", "test")),
)

#: Reason keys every source's ``drops`` mapping carries, so a zero is explicit.
DROP_REASONS = (
    "reserved_uuid",
    "duplicate",
    "contaminated",
    "validation_error",
    "no_assistant_turn",
    "template_error",
    "empty_label_span",
    "over_cap",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _count_rows(path: Path) -> int:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


# --------------------------------------------------------------------------- #
# Protected material
# --------------------------------------------------------------------------- #


def _instrument_text(item: Mapping[str, Any]) -> str:
    """One instrument item as the text a leaking training row would contain."""
    return json.dumps(
        {key: item.get(key) for key in ("messages", "tools", "expected")},
        ensure_ascii=False,
        sort_keys=True,
    )


def load_protected_texts() -> dict[str, list[str]]:
    """The four benchmark test splits, offline, plus the frozen tool-use items.

    Reads the pinned HF caches with ``HF_DATASETS_OFFLINE=1``: preparation must
    not silently fetch a different revision of a benchmark than the one measured.
    """
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    texts: dict[str, list[str]] = {}
    for qualified_id, repo, config, split, fields in _PROTECTED_SPECS:
        dataset = load_dataset(repo, config, split=split)
        texts[qualified_id] = [
            str(row[field])
            for row in dataset
            for field in fields
            if isinstance(row.get(field), str)
        ]
    texts[INSTRUMENT_ID] = [_instrument_text(item) for item in load_items()]
    return texts


def build_firewall(protected: Mapping[str, Sequence[str]]) -> ContaminationFirewall:
    firewall = ContaminationFirewall()
    for qualified_id, texts in protected.items():
        if texts:
            firewall.register_protected(qualified_id, texts)
    return firewall


# --------------------------------------------------------------------------- #
# Row shape
# --------------------------------------------------------------------------- #


def _row_text(messages: Sequence[Mapping[str, Any]], tools: Any) -> str:
    """The text a row puts in front of the model -- what the firewall checks."""
    parts: list[str] = []
    for turn in messages:
        parts.append(str(turn.get("content") or ""))
        if turn.get("reasoning_content"):
            parts.append(str(turn["reasoning_content"]))
        for call in turn.get("tool_calls") or []:
            function = call.get("function") or {}
            parts.append(str(function.get("name") or ""))
            parts.append(json.dumps(function.get("arguments"), ensure_ascii=False, sort_keys=True))
    if tools:
        parts.append(json.dumps(tools, ensure_ascii=False, sort_keys=True))
    return "\n".join(parts)


def _arguments_as_json_strings(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Tool-call arguments back to JSON strings, the Arrow-safe row shape."""
    out: list[dict[str, Any]] = []
    for turn in messages:
        turn = dict(turn)
        if turn.get("tool_calls"):
            turn["tool_calls"] = [
                {
                    **call,
                    "function": {
                        **(call.get("function") or {}),
                        "arguments": json.dumps(
                            (call.get("function") or {}).get("arguments"),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    },
                }
                for call in turn["tool_calls"]
            ]
        out.append(turn)
    return out


def _has_trainable_assistant(messages: Sequence[Mapping[str, Any]]) -> bool:
    """An assistant turn with something to train on -- text or a call."""
    return any(
        turn["role"] == "assistant"
        and (str(turn.get("content") or "").strip() or turn.get("tool_calls"))
        for turn in messages
    )


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def _select(
    rows: Iterable[dict[str, Any]],
    *,
    label: str,
    cap: int,
    firewall: ContaminationFirewall,
    tokenizer: Any,
    max_length: int,
    token_counts: list[int],
    source_of: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Filter rows in order until ``cap`` survive; count every drop reason.

    Returns the kept rows and the number of candidates examined. Each drop is
    counted once, in the source entry ``source_of`` names -- the source entries
    are the only drop ledger, and ``drops_total`` sums them. The firewall is
    the one built by the caller; each row's text is checked exactly once.
    """
    kept: list[dict[str, Any]] = []
    examined = 0
    for index, row in enumerate(rows):
        if len(kept) >= cap:
            break
        examined += 1
        if examined % 200 == 0:
            print(f"[{label}] examined {examined}, kept {len(kept)}", file=sys.stderr, flush=True)
        entry = source_of(row)
        entry["examined"] += 1

        def drop(reason: str) -> None:
            entry["drops"][reason] += 1

        messages = row["messages"]
        tools = row["tools"] or None
        if not firewall.check_text(_row_text(messages, tools)).clean:
            drop("contaminated")
            continue
        try:
            normalized = _validate_chat_messages(messages, row_index=index)
        except Exception:
            drop("validation_error")
            continue
        if not _has_trainable_assistant(normalized):
            drop("no_assistant_turn")
            continue

        messages_out = _arguments_as_json_strings(normalized)
        tools_out = json.dumps(tools, ensure_ascii=False) if tools else ""
        if tokenizer is not None:
            try:
                # Render ``normalized`` (dict arguments), exactly as the worker's
                # chat path does: ``_validate_chat_messages`` parses the JSON-string
                # arguments back to mappings, and Qwen's template iterates them as
                # a mapping (a string raises "Can only get item pairs from a mapping").
                example = _build_chat_example(
                    tokenizer,
                    normalized,
                    max_length=max_length,
                    row_index=index,
                    tools=parse_tools(tools_out, row_index=index),
                )
            except Exception as error:  # a template refusal, or an empty label span
                drop("empty_label_span" if "no assistant tokens remain" in str(error) else "template_error")
                continue
            token_counts.append(len(example["input_ids"]))
        kept.append({"messages": messages_out, "tools": tools_out})
        entry["rows_out"] += 1
    return kept, examined


def _nemotron_rows(path: Path, *, drops: Counter[str]) -> Any:
    """Non-reserved Nemotron rows, yielded lazily.

    Streamed, not loaded through ``NemotronAgenticProvider``: the source is
    428 MB and the provider materialises every row before serving, which does not
    fit beside the tokenizer on this machine. The reserved-uuid predicate, the
    native-call normalisation and the row shape are the provider's, imported.
    """
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            uuid = row.get("uuid")
            if not isinstance(uuid, str) or not uuid.strip():
                raise SystemExit(f"{path}:{number} has no uuid; its reservation cannot be decided")
            if is_reserved_for_evaluation(uuid):
                drops["reserved_uuid"] += 1
                continue
            yield {"messages": _native_tool_calls(row["messages"]), "tools": row.get("tools") or None}


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _percentiles(values: Sequence[int]) -> dict[str, int]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(percent: int) -> int:
        return ordered[min(len(ordered) - 1, round(percent / 100 * (len(ordered) - 1)))]

    return {"min": ordered[0], "p50": at(50), "p90": at(90), "p99": at(99), "max": ordered[-1]}


def _load_tokenizer(tokenizer_dir: str) -> Any | None:
    if not (Path(tokenizer_dir) / "tokenizer.json").is_file():
        print(
            f"warning: no tokenizer at {tokenizer_dir}; template validation skipped",
            file=sys.stderr,
        )
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=False)


def _source_entry(name: str, path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "licence": LICENCES[name],
        "rows_in": _count_rows(path),
        "examined": 0,
        "rows_out": 0,
        "drops": Counter(),
    }


def build(
    *,
    out_dir: str | Path = DEFAULT_OUT,
    glint: str | None = DEFAULT_GLINT,
    lordx64: str | None = DEFAULT_LORDX64,
    swarm: str | None = DEFAULT_SWARM,
    nemotron: str = DEFAULT_NEMOTRON,
    fable_rows: int = DEFAULT_FABLE_ROWS,
    nemotron_rows: int = DEFAULT_NEMOTRON_ROWS,
    seed: int = DEFAULT_SEED,
    validate_template: bool = True,
    tokenizer_dir: str = DEFAULT_TOKENIZER,
    max_length: int = DEFAULT_MAX_LENGTH,
    protected: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Write both sets and their manifest; return the manifest."""
    fable_paths = {
        name: Path(path)
        for name, path in (("glint_local", glint), ("lordx64", lordx64), ("swarm", swarm))
        if path
    }
    if not fable_paths:
        raise SystemExit("at least one Fable source is required")
    nemotron_path = Path(nemotron)

    rng = random.Random(seed)
    print("[1/5] loading protected evaluation material", file=sys.stderr, flush=True)
    protected_texts = dict(protected) if protected is not None else load_protected_texts()
    firewall = build_firewall(protected_texts)
    print(f"[2/5] firewall built over {sum(map(len, protected_texts.values()))} protected texts",
          file=sys.stderr, flush=True)
    tokenizer = _load_tokenizer(tokenizer_dir) if validate_template else None
    if not validate_template:
        print("warning: --no-validate-template; rows are not rendered", file=sys.stderr)
    elif tokenizer is not None:
        print("[3/5] tokenizer loaded; rows are template-validated", file=sys.stderr, flush=True)

    sources: dict[str, dict[str, Any]] = {
        f"fable:{name}": _source_entry(name, path) for name, path in fable_paths.items()
    }
    sources["nemotron"] = _source_entry("nemotron", nemotron_path)

    def entry_for(row: Mapping[str, Any]) -> dict[str, Any]:
        return sources[f"fable:{row['provenance']['source']}"]

    # -- Nemotron, shared by both arms: selected once, reused verbatim ------- #
    print("[4/5] streaming Nemotron (reserved eval rows excluded)", file=sys.stderr, flush=True)
    nemotron_drops = sources["nemotron"]["drops"]
    nemotron_counts: list[int] = []
    nemotron_kept, nemotron_examined = _select(
        _nemotron_rows(nemotron_path, drops=nemotron_drops),
        label="nemotron",
        cap=nemotron_rows,
        firewall=firewall,
        tokenizer=tokenizer,
        max_length=max_length,
        token_counts=nemotron_counts,
        source_of=lambda _row: sources["nemotron"],
    )
    # Every line is reserved, examined, or never reached once the cap was met.
    nemotron_drops["over_cap"] += (
        sources["nemotron"]["rows_in"] - nemotron_drops["reserved_uuid"] - nemotron_examined
    )

    # -- Fable: the merged provider owns the cross-source dedupe ------------- #
    print("[5/5] loading Fable sources (glint, lordx64, swarm)", file=sys.stderr, flush=True)
    provider = FableAgenticProvider(
        {name: (str(path), sources[f"fable:{name}"]["sha256"]) for name, path in fable_paths.items()},
        skill="coding.agentic",
        allow_fable=True,
    )
    for name, dropped in provider.dropped_duplicates.items():
        sources[f"fable:{name}"]["drops"]["duplicate"] += dropped
    fable_pool = list(provider._rows)  # the provider's served rows, tools unfolded
    rng.shuffle(fable_pool)
    fable_counts: list[int] = []
    fable_kept, fable_examined = _select(
        fable_pool,
        label="fable",
        cap=fable_rows,
        firewall=firewall,
        tokenizer=tokenizer,
        max_length=max_length,
        token_counts=fable_counts,
        source_of=entry_for,
    )
    # The unexamined tail of the merged pool, attributed back to its sources so
    # every source's rows_in reconciles with rows_out + drops.
    for row in fable_pool[fable_examined:]:
        entry_for(row)["drops"]["over_cap"] += 1

    # -- write --------------------------------------------------------------- #
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    for filename, rows, counts, declared in (
        ("train-fable.jsonl", [*fable_kept, *nemotron_kept], [*fable_counts, *nemotron_counts], True),
        ("train-control.jsonl", nemotron_kept, nemotron_counts, False),
    ):
        entry: dict[str, Any] = {
            "rows": len(rows),
            "sha256": _write_jsonl(out / filename, rows),
            "fable_declared": declared,
        }
        if tokenizer is not None:
            entry["token_count_percentiles"] = _percentiles(counts)
        outputs[filename] = entry

    total_drops: Counter[str] = Counter()
    for entry in sources.values():
        entry["drops"] = {reason: entry["drops"][reason] for reason in DROP_REASONS}
        total_drops.update(entry["drops"])

    manifest = {
        "seed": seed,
        "max_length": max_length,
        "template_validated": tokenizer is not None,
        "fable_rows_requested": fable_rows,
        "nemotron_rows_requested": nemotron_rows,
        "sources": sources,
        "protected": {qualified_id: len(texts) for qualified_id, texts in protected_texts.items()},
        "drops_total": {reason: total_drops[reason] for reason in DROP_REASONS},
        "outputs": outputs,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--glint", default=DEFAULT_GLINT)
    parser.add_argument("--lordx64", default=DEFAULT_LORDX64)
    parser.add_argument("--swarm", default=DEFAULT_SWARM)
    parser.add_argument("--nemotron", default=DEFAULT_NEMOTRON)
    parser.add_argument("--fable-rows", type=int, default=DEFAULT_FABLE_ROWS)
    parser.add_argument("--nemotron-rows", type=int, default=DEFAULT_NEMOTRON_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--validate-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build(
        out_dir=args.out,
        glint=args.glint,
        lordx64=args.lordx64,
        swarm=args.swarm,
        nemotron=args.nemotron,
        fable_rows=args.fable_rows,
        nemotron_rows=args.nemotron_rows,
        seed=args.seed,
        validate_template=args.validate_template,
        tokenizer_dir=args.tokenizer,
        max_length=args.max_length,
    )
    print(json.dumps({
        "outputs": {name: entry["rows"] for name, entry in manifest["outputs"].items()},
        "template_validated": manifest["template_validated"],
        "drops_total": manifest["drops_total"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
