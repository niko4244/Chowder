"""Build the MoE Gen-1 training mix: the Fable arm and its Fable-free control.

The mix is the one approved in
``docs/PLAN-MOE-FINETUNE-BEAT-BASELINES-2026-10-19.md`` §3: six sources, 13-gram
excluded against every protected benchmark split, rendered into the worker's
chat row shape::

    {"messages": [...], "tools": "<json string>" or ""}

``train-control.jsonl`` is ``train-fable.jsonl`` minus exactly the Fable rows --
same Nemotron block, same order -- so a measured difference between the arms is
the Fable exposure and not a sampling artefact (decision N2 is still open).

Selection is a seeded permutation of each source's deduplicated pool, then the
first ``cap`` rows that survive the contamination firewall. For a pool where
more than ``cap`` rows are clean, the first ``cap`` clean rows of a uniform
permutation are a uniform sample without replacement; where the pool is smaller
than the cap, every row is examined and the shortfall is recorded. This keeps
the check off OT3's 8,000 half-megabyte rows beyond the ones actually needed.

Usage:
    python chowder_batch/build_moe_gen1_mix.py --out C:\\llm-work\\train-sets\\moe-gen1-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "chowder_batch"):  # a batch script, not an installed entry
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from chowder.backends.training_data import (  # noqa: E402
    _build_chat_example,
    _validate_chat_messages,
    parse_tools,
)

# The agentic-v1 builder owns the firewall wiring and the row shape; import,
# never copy, so both sets are checked by the same code.
from build_agentic_training_set import (  # noqa: E402
    _arguments_as_json_strings,
    _has_trainable_assistant,
    _load_tokenizer,
    _percentiles,
    _row_text,
    _sha256_file,
    _write_jsonl,
    build_firewall,
    load_protected_texts,
)
from chowder.growth.tool_use_instrument import INSTRUMENT_ID, items_path  # noqa: E402

DEFAULT_OUT = r"C:\llm-work\train-sets\moe-gen1-v1"
DEFAULT_TOKENIZER = r"G:\llm-cold\bf16\Qwen3.6-35B-A3B"
DEFAULT_OPENR1 = r"C:\Users\nikma\chowder_teacher_free\supply\openr1_math_sample500.jsonl"
DEFAULT_OT3 = r"C:\Users\nikma\chowder_teacher_free\ot3_sft.jsonl"
#: OT3 rows are long reasoning traces (almost none fit 2,048 tokens); this source is short, test-verified Python.
DEFAULT_SELFOSS = r"C:\llm-work\data\bigcode__self-oss-instruct-sc2-exec-filter-50k\selfoss.jsonl"
DEFAULT_ORNITH = r"C:\llm-work\ornith\curriculum-all\ornith_sft.jsonl"
DEFAULT_AGENTIC_FABLE = r"C:\llm-work\train-sets\agentic-v1\train-fable.jsonl"
DEFAULT_AGENTIC_CONTROL = r"C:\llm-work\train-sets\agentic-v1\train-control.jsonl"

#: The Ornith rows are refused if the bytes are not the reviewed curriculum.
ORNITH_SHA256 = "971475e0f44ba55cbeff3a693be221faf55e078bb67025292856c7e811b2fbb8"

DEFAULT_SEED = 20261019
DEFAULT_MAX_LENGTH = 2048  # the MoE Gen-1 training sequence (plan §4); longer rows are dropped, not cut
NGRAM = 13  # the plan's 13-gram rule, as in chowder_batch/reap_calibration.py

PLAN_FILE = "docs/PLAN-MOE-FINETUNE-BEAT-BASELINES-2026-10-19.md"

#: ``(source, cap)`` in manifest order. The two agentic sources are last so the
#: Fable pool can be derived against the already-indexed control file.
SOURCE_CAPS: tuple[tuple[str, int], ...] = (
    ("openr1_math", 400),
    ("gsm8k_train", 200),
    ("ornith", 39),
    ("ot3_code", 150),
    ("selfoss_code", 250),
    ("nemotron", 120),
    ("fable", 120),
)
FABLE_SOURCES = frozenset({"fable"})

LICENCES = {
    "openr1_math": "Apache-2.0 (open-r1/OpenR1-Math-220k verified solutions)",
    "gsm8k_train": "MIT (openai/gsm8k, train split)",
    "ornith": "not declared (Ornith curriculum; owner-supplied)",
    "ot3_code": "Apache-2.0 (open-thoughts/OpenThoughts-3, code reasoning)",
    "selfoss_code": "ODC-BY (bigcode/self-oss-instruct-sc2-exec-filter-50k, execution-filtered Python)",
    "nemotron": "CC-BY-4.0 (nvidia/Nemotron-Agentic-v1, interactive_agent)",
    "fable": "AGPL-3.0 (Fable 5 output; N2 Anthropic usage question open)",
}

#: Reason keys every source's ``drops`` mapping carries, so a zero is explicit.
DROP_REASONS = (
    "duplicate",
    "contaminated",
    "validation_error",
    "no_assistant_turn",
    "template_error",
    "empty_label_span",
    "over_length",
    "over_cap",
)


# --------------------------------------------------------------------------- #
# Text fingerprints
# --------------------------------------------------------------------------- #


def _words(text: str) -> list[str]:
    """The normalisation ``reap_calibration`` uses for its 13-gram check."""
    return re.findall(r"\w+", text.lower())


def _hash_gram(gram: str) -> int:
    """Stable 64-bit hash of one 13-gram (blake2b, not the salted builtin)."""
    return int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big")


def ngram_hashes(text: str) -> set[int]:
    words = _words(text)
    return {
        _hash_gram(" ".join(words[i : i + NGRAM]))
        for i in range(len(words) - NGRAM + 1)
    }


def _content_text(messages: Sequence[Mapping[str, Any]]) -> str:
    """The row's text without its ``system`` turns.

    The system turn is the shared environment contract (Nemotron's agent policy
    is the same paragraph in every row and in every ``tool-use@reap-moe-v1``
    item it was built from), so fingerprinting it marks the entire family. The
    user and assistant turns are where an item's own content lives. This is the
    same rationale the agentic builder applies to its tool schemas.
    """
    parts: list[str] = []
    for turn in messages:
        if turn.get("role") == "system":
            continue
        parts.append(str(turn.get("content") or ""))
        if turn.get("reasoning_content"):
            parts.append(str(turn["reasoning_content"]))
        for call in turn.get("tool_calls") or []:
            function = call.get("function") or {}
            parts.append(str(function.get("name") or ""))
            parts.append(json.dumps(function.get("arguments"), ensure_ascii=False, sort_keys=True))
    return "\n".join(parts)


def _protected_text(text: str) -> str:
    """A protected text with its shared system turn removed, when it has one.

    The instrument owns its items as ``{"messages": [...], "expected": ...}``
    (``build_agentic_training_set._instrument_text``); every other protected
    text is a benchmark field and is used verbatim.
    """
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        return text
    return _content_text(payload["messages"])


def protected_ngram_hashes(protected: Mapping[str, Sequence[str]]) -> set[int]:
    grams: set[int] = set()
    for texts in protected.values():
        for text in texts:
            grams |= ngram_hashes(_protected_text(text))
    return grams


def _detectors(text: str, firewall: Any, grams: set[int], content: str) -> list[str]:
    """Every detector that fired for one row, deduplicated.

    The firewall is the agentic-v1 one (exact, normalized, substring shingle
    overlap, MinHash) over the whole row; the plan's 13-gram rule runs over the
    non-system content only (:func:`_content_text`).
    """
    hits = {match.detector for match in firewall.check_text(text).matches}
    if grams and ngram_hashes(content) & grams:
        hits.add("ngram13")
    return sorted(hits)



# --------------------------------------------------------------------------- #
# Candidate rows
# --------------------------------------------------------------------------- #


def _canonical_hash(row: Mapping[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _openr1_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": str(row["problem"])},
            {"role": "assistant", "content": str(row["solution"])},
        ],
        "tools": "",
    }


def _gsm8k_row(row: Mapping[str, Any]) -> dict[str, Any]:
    # Worked steps and the closing ``#### n`` line stay verbatim.
    return {
        "messages": [
            {"role": "user", "content": str(row["question"])},
            {"role": "assistant", "content": str(row["answer"])},
        ],
        "tools": "",
    }


def _ot3_row(row: Mapping[str, Any]) -> dict[str, Any]:
    messages = []
    for turn in row.get("conversations") or []:
        origin = turn.get("from")
        role = {"human": "user", "gpt": "assistant", "system": "system"}.get(origin)
        if role is None:
            # An unknown speaker is not silently re-labelled; the row fails
            # validation below and is counted.
            role = "unknown"
        messages.append({"role": role, "content": turn.get("value") or ""})
    return {"messages": messages, "tools": ""}


def _selfoss_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": row["instruction"]},
                         {"role": "assistant", "content": row["response"]}], "tools": ""}


def _ornith_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"messages": row["messages"], "tools": ""}


def _agentic_row(row: Mapping[str, Any]) -> dict[str, Any]:
    tools = row.get("tools") or ""
    if not isinstance(tools, str):
        tools = json.dumps(tools, ensure_ascii=False)
    return {"messages": row["messages"], "tools": tools}


# --------------------------------------------------------------------------- #
# Pools: dedupe once, then a seeded permutation
# --------------------------------------------------------------------------- #


def _index_jsonl(
    path: Path,
    to_candidate: Callable[[Mapping[str, Any]], dict[str, Any]],
    *,
    exclude: frozenset[str] = frozenset(),
) -> tuple[list[int], int, int]:
    """Byte offsets of the unique rows, the rows included, and duplicates.

    ``exclude`` removes rows whose canonical hash is another source's (the
    Nemotron rows inside the Fable file); they are not counted here.
    """
    offsets: list[int] = []
    seen: set[str] = set()
    included = 0
    duplicates = 0
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            digest = _canonical_hash(to_candidate(json.loads(line)))
            if digest in exclude:
                continue
            included += 1
            if digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            offsets.append(offset)
    return offsets, included, duplicates


def _index_rows(
    rows: Iterable[Mapping[str, Any]],
    to_candidate: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    """In-memory counterpart of :func:`_index_jsonl` for the GSM8K split."""
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    included = 0
    duplicates = 0
    for row in rows:
        candidate = to_candidate(row)
        included += 1
        digest = _canonical_hash(candidate)
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        unique.append(candidate)
    return unique, included, duplicates


def _jsonl_rows(path: Path, offsets: Sequence[int], to_candidate) -> Iterator[dict[str, Any]]:
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            yield to_candidate(json.loads(handle.readline()))


def _shuffled(keys: list[Any], seed: int, label: str) -> list[Any]:
    """Seeded permutation; each source has its own stream so caps can move."""
    rng = random.Random(f"{seed}:{label}")
    keys = list(keys)
    rng.shuffle(keys)
    return keys


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def _entry(cap: int, path: str, sha256: str | None) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": sha256,
        "licence": "",
        "cap": cap,
        "rows_in": 0,
        "examined": 0,
        "kept": 0,
        "shortfall": 0,
        "drops": Counter({reason: 0 for reason in DROP_REASONS}),
        "contamination_detectors": Counter(),
    }


def _select(
    candidates: Iterable[dict[str, Any]],
    *,
    entry: dict[str, Any],
    cap: int,
    firewall: Any,
    grams: set[int],
    tokenizer: Any,
    max_length: int,
    token_counts: list[int],
    label: str,
) -> list[dict[str, Any]]:
    """Take the first ``cap`` clean rows of the (already shuffled) pool."""
    kept: list[dict[str, Any]] = []
    examined = 0
    for candidate in candidates:
        if len(kept) >= cap:
            break
        examined += 1
        if examined % 200 == 0:
            print(f"[{label}] examined {examined}, kept {len(kept)}", file=sys.stderr, flush=True)
        messages = candidate["messages"]
        tools = candidate["tools"]
        # The tool schema is the shared API surface every agentic row offers;
        # fingerprinting it flags the whole family (see ``_instrument_text``).
        # The conversation and the calls are what a leak reproduces.
        detectors = _detectors(
            _row_text(messages, None), firewall, grams, _content_text(messages)
        )
        if detectors:
            entry["drops"]["contaminated"] += 1
            entry["contamination_detectors"].update(detectors)
            continue
        try:
            normalized = _validate_chat_messages(messages, row_index=examined)
        except Exception:
            entry["drops"]["validation_error"] += 1
            continue
        if not _has_trainable_assistant(normalized):
            entry["drops"]["no_assistant_turn"] += 1
            continue
        messages_out = _arguments_as_json_strings(normalized)
        if tokenizer is not None:
            try:
                example = _build_chat_example(
                    tokenizer,
                    normalized,
                    max_length=1 << 30,  # measure the whole row; truncation would cut final answers
                    row_index=examined,
                    tools=parse_tools(tools, row_index=examined),
                )
            except Exception as error:
                reason = (
                    "empty_label_span"
                    if "no assistant tokens remain" in str(error)
                    else "template_error"
                )
                entry["drops"][reason] += 1
                continue
            if len(example["input_ids"]) > max_length:
                # A row longer than the training sequence would be cut before its
                # final answer; drop it whole and let the pool backfill.
                entry["drops"]["over_length"] += 1
                continue
            token_counts.append(len(example["input_ids"]))
        kept.append({"messages": messages_out, "tools": tools})
    entry["examined"] = examined
    return kept


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def build(
    *,
    out_dir: str | Path = DEFAULT_OUT,
    seed: int = DEFAULT_SEED,
    caps: Mapping[str, int] | None = None,
    openr1: str | Path = DEFAULT_OPENR1,
    ot3: str | Path = DEFAULT_OT3,
    selfoss: str | Path | None = None,
    ornith: str | Path = DEFAULT_ORNITH,
    agentic_fable: str | Path = DEFAULT_AGENTIC_FABLE,
    agentic_control: str | Path = DEFAULT_AGENTIC_CONTROL,
    gsm8k_rows: Iterable[Mapping[str, Any]] | None = None,
    protected: Mapping[str, Sequence[str]] | None = None,
    tokenizer: Any | None = None,
    validate_template: bool = True,
    tokenizer_dir: str = DEFAULT_TOKENIZER,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict[str, Any]:
    """Write both arms and the manifest; return the manifest."""
    limits = {**dict(SOURCE_CAPS), **dict(caps or {})}
    out = Path(out_dir)

    ornith_path = Path(ornith)
    ornith_sha = _sha256_file(ornith_path)
    if ornith_sha != ORNITH_SHA256:
        raise SystemExit(
            f"{ornith_path}: sha256 {ornith_sha} is not the reviewed curriculum "
            f"{ORNITH_SHA256}; refusing to build from different bytes"
        )

    print("[1/5] loading protected evaluation material", file=sys.stderr, flush=True)
    protected_texts = dict(protected) if protected is not None else load_protected_texts()
    firewall = build_firewall(protected_texts)
    grams = protected_ngram_hashes(protected_texts)
    print(
        f"[2/5] firewall over {sum(map(len, protected_texts.values()))} protected texts, "
        f"{len(grams)} distinct 13-grams",
        file=sys.stderr,
        flush=True,
    )

    if not validate_template:
        tokenizer = None
        print("warning: --no-validate-template; rows are not rendered", file=sys.stderr)
    elif tokenizer is None:
        tokenizer = _load_tokenizer(tokenizer_dir)

    sources: dict[str, dict[str, Any]] = {}
    rows_by_source: dict[str, list[dict[str, Any]]] = {}
    token_counts: dict[str, list[int]] = {name: [] for name, _ in SOURCE_CAPS}

    def run(
        name: str,
        *,
        path: str,
        sha256: str | None,
        candidates: Iterable[dict[str, Any]],
        pool_size: int,
        rows_in: int,
        duplicates: int,
    ) -> None:
        entry = _entry(limits[name], path, sha256)
        entry["licence"] = LICENCES[name]
        entry["rows_in"] = rows_in
        entry["drops"]["duplicate"] += duplicates
        kept = _select(
            candidates,
            entry=entry,
            cap=limits[name],
            firewall=firewall,
            grams=grams,
            tokenizer=tokenizer,
            max_length=max_length,
            token_counts=token_counts[name],
            label=name,
        )
        entry["drops"]["over_cap"] += pool_size - entry["examined"]
        entry["kept"] = len(kept)
        entry["shortfall"] = max(0, limits[name] - len(kept))
        sources[name] = entry
        rows_by_source[name] = kept
        print(
            f"[3/5] {name}: kept {len(kept)}/{limits[name]} "
            f"(examined {entry['examined']}, contaminated {entry['drops']['contaminated']})",
            file=sys.stderr,
            flush=True,
        )

    # -- the five plain sources -------------------------------------------- #
    plain = (
        ("openr1_math", Path(openr1), _openr1_row),
        ("ot3_code", Path(ot3), _ot3_row),
        ("ornith", ornith_path, _ornith_row),
        ("selfoss_code", Path(selfoss) if selfoss else None, _selfoss_row),
    )
    for name, path, to_candidate in plain:
        if path is None:
            # Not supplied: an empty pool, so the manifest records the full shortfall.
            run(name, path="not supplied", sha256="", candidates=iter(()), pool_size=0,
                rows_in=0, duplicates=0)
            continue
        offsets, rows_in, duplicates = _index_jsonl(path, to_candidate)
        run(
            name,
            path=str(path),
            sha256=_sha256_file(path),
            candidates=_jsonl_rows(path, _shuffled(offsets, seed, name), to_candidate),
            pool_size=len(offsets),
            rows_in=rows_in,
            duplicates=duplicates,
        )

    if gsm8k_rows is None:
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        from datasets import load_dataset

        gsm8k_rows = load_dataset("openai/gsm8k", "main", split="train")
    unique_rows, rows_in, duplicates = _index_rows(gsm8k_rows, _gsm8k_row)
    run(
        "gsm8k_train",
        path="openai/gsm8k:main:train (offline Hugging Face cache)",
        sha256=None,
        candidates=(unique_rows[i] for i in _shuffled(list(range(len(unique_rows))), seed, "gsm8k_train")),
        pool_size=len(unique_rows),
        rows_in=rows_in,
        duplicates=duplicates,
    )
    del unique_rows

    # -- Nemotron from the control file, then Fable as the fable-only rows -- #
    control_path = Path(agentic_control)
    control_offsets, control_rows_in, control_duplicates = _index_jsonl(
        control_path, _agentic_row
    )
    run(
        "nemotron",
        path=str(control_path),
        sha256=_sha256_file(control_path),
        candidates=_jsonl_rows(
            control_path, _shuffled(control_offsets, seed, "nemotron"), _agentic_row
        ),
        pool_size=len(control_offsets),
        rows_in=control_rows_in,
        duplicates=control_duplicates,
    )

    fable_path = Path(agentic_fable)
    control_hashes = frozenset(
        _canonical_hash(_agentic_row(json.loads(line)))
        for line in control_path.open("rb")
        if line.strip()
    )
    fable_offsets, fable_rows_in, fable_duplicates = _index_jsonl(
        fable_path, _agentic_row, exclude=control_hashes
    )
    run(
        "fable",
        path=str(fable_path),
        sha256=_sha256_file(fable_path),
        candidates=_jsonl_rows(
            fable_path, _shuffled(fable_offsets, seed, "fable"), _agentic_row
        ),
        pool_size=len(fable_offsets),
        rows_in=fable_rows_in,
        duplicates=fable_duplicates,
    )
    sources["fable"]["derived_from"] = {
        "arm": str(fable_path),
        "arm_sha256": _sha256_file(fable_path),
        "control": str(control_path),
        "control_sha256": _sha256_file(control_path),
        "rule": "rows present in the arm file and absent from the control file (canonical JSON hash)",
    }

    # -- write: one shuffle, then the control is a strict subsequence -------- #
    print("[4/5] shuffling and writing both arms", file=sys.stderr, flush=True)
    merged: list[tuple[str, dict[str, Any]]] = [
        (name, row) for name, _ in SOURCE_CAPS for row in rows_by_source[name]
    ]
    random.Random(seed).shuffle(merged)
    fable_arm = [row for _, row in merged]
    control_arm = [row for name, row in merged if name not in FABLE_SOURCES]

    out.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    for filename, rows, counts in (
        ("train-fable.jsonl", fable_arm, [c for name, _ in SOURCE_CAPS for c in token_counts[name]]),
        (
            "train-control.jsonl",
            control_arm,
            [c for name, _ in SOURCE_CAPS if name not in FABLE_SOURCES for c in token_counts[name]],
        ),
    ):
        entry: dict[str, Any] = {
            "rows": len(rows),
            "sha256": _write_jsonl(out / filename, rows),
        }
        if tokenizer is not None:
            entry["token_count_percentiles"] = _percentiles(counts)
        entry["sources"] = (
            [name for name, _ in SOURCE_CAPS]
            if filename == "train-fable.jsonl"
            else [name for name, _ in SOURCE_CAPS if name not in FABLE_SOURCES]
        )
        entry["fable_declared"] = filename == "train-fable.jsonl"
        outputs[filename] = entry

    drops_total: Counter[str] = Counter()
    for entry in sources.values():
        entry["drops"] = {reason: entry["drops"][reason] for reason in DROP_REASONS}
        entry["contamination_detectors"] = dict(sorted(entry["contamination_detectors"].items()))
        drops_total.update(entry["drops"])

    manifest = {
        "builder": "chowder_batch/build_moe_gen1_mix.py",
        "plan": PLAN_FILE,
        "seed": seed,
        "ngram": NGRAM,
        "max_length": max_length,
        "tokenizer": {
            "path": tokenizer_dir,
            "loaded": tokenizer is not None,
        },
        "template_validated": tokenizer is not None,
        "protected": {qualified_id: len(texts) for qualified_id, texts in protected_texts.items()},
        "protected_13grams": len(grams),
        "tool_use_instrument": (
            "present" if protected_texts.get(INSTRUMENT_ID) else "not present"
        ),
        "caps": {name: limits[name] for name, _ in SOURCE_CAPS},
        "sources": sources,
        "drops_total": {reason: drops_total[reason] for reason in DROP_REASONS},
        "outputs": outputs,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("[5/5] manifest written", file=sys.stderr, flush=True)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--openr1", default=DEFAULT_OPENR1)
    parser.add_argument("--ot3", default=DEFAULT_OT3)
    parser.add_argument("--selfoss", default=DEFAULT_SELFOSS, help="short execution-filtered code; empty to omit")
    parser.add_argument("--ornith", default=DEFAULT_ORNITH)
    parser.add_argument("--agentic-fable", default=DEFAULT_AGENTIC_FABLE)
    parser.add_argument("--agentic-control", default=DEFAULT_AGENTIC_CONTROL)
    parser.add_argument("--validate-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build(
        out_dir=args.out,
        seed=args.seed,
        openr1=args.openr1,
        ot3=args.ot3,
        selfoss=args.selfoss or None,
        ornith=args.ornith,
        agentic_fable=args.agentic_fable,
        agentic_control=args.agentic_control,
        validate_template=args.validate_template,
        tokenizer_dir=args.tokenizer,
        max_length=args.max_length,
    )
    print(
        json.dumps(
            {
                "outputs": {name: entry["rows"] for name, entry in manifest["outputs"].items()},
                "kept": {name: entry["kept"] for name, entry in manifest["sources"].items()},
                "shortfall": {
                    name: entry["shortfall"]
                    for name, entry in manifest["sources"].items()
                    if entry["shortfall"]
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
