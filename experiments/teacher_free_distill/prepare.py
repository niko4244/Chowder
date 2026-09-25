"""Fail-closed, teacher-free pilot dataset preparation for Chowder.

No teacher is loaded. Third-party data must be separately reviewed for their
actual redistribution and training terms before a source is approved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import zlib
from collections import Counter
from pathlib import Path
from typing import Iterator


def canonical(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(obj: object) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


LSH_BANDS = 8
LSH_SHINGLE = 5
# Fixed per-band hash parameters: deterministic across runs and processes
# (crc32 shingle digests, seeded multipliers — never Python's salted hash()).
_LSH_PARAMS = []
for _band in range(LSH_BANDS):
    _a = 1 + (zlib.crc32(f"band{_band}".encode()) * 2654435761) % (1 << 30)
    _b = zlib.crc32(f"offset{_band}".encode()) % (1 << 30)
    _LSH_PARAMS.append((_a, _b))


def _shingles(first_user: str) -> set[str]:
    words = normalized(first_user).split()
    if not words:
        return set()
    if len(words) < LSH_SHINGLE:
        return {" ".join(words)}
    return {" ".join(words[i:i + LSH_SHINGLE])
            for i in range(len(words) - LSH_SHINGLE + 1)}


def near_duplicate_key(first_user: str) -> tuple[str, ...]:
    """Banded minhash LSH signature over word 5-shingles (deterministic).

    Word-level edits that preserve most shingles collide with high
    probability, so "same question with a small suffix" is caught. This is a
    cheap screen, not exhaustive fuzzy dedup; recorded as a limitation.
    """
    shs = _shingles(first_user)
    if not shs:
        return ()
    digests = [zlib.crc32(s.encode("utf-8")) for s in shs]
    sig = []
    for a, b in _LSH_PARAMS:
        sig.append(min(((a * d + b) & 0xFFFFFFFF) for d in digests))
    return tuple(f"{i}:{v}" for i, v in enumerate(sig))


_tokenizer_cache: dict[str, object] = {}


def count_tokens(messages: list[dict[str, str]]) -> int | None:
    """Token count via a cached tokenizer (TFD_TOKENIZER env override).

    Returns None when no tokenizer is configured or loadable offline; char
    counts are always recorded so manifests stay complete without network.
    """
    name = os.environ.get("TFD_TOKENIZER", "")
    if not name:
        return None
    tok = _tokenizer_cache.get(name)
    if tok is None and name not in _tokenizer_cache:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(name)
        except Exception:
            tok = False
        _tokenizer_cache[name] = tok
    if not tok:
        return None
    try:
        encoded = tok.apply_chat_template(messages, tokenize=True)
        ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        return len(ids)
    except Exception:
        return None


def chat_messages(row: dict) -> list[dict[str, str]]:
    raw = row.get("messages") or row.get("conversations")
    if not isinstance(raw, list):
        raise ValueError("missing messages/conversations")
    roles = {"human": "user", "user": "user", "gpt": "assistant",
             "assistant": "assistant", "system": "system"}
    out: list[dict[str, str]] = []
    for part in raw:
        if not isinstance(part, dict):
            raise ValueError("invalid message")
        role = roles.get(part.get("role", part.get("from")))
        value = part.get("content", part.get("value"))
        if role is None or not isinstance(value, str) or not value.strip():
            raise ValueError("invalid role/content")
        out.append({"role": role, "content": value.strip()})
    if not any(x["role"] == "user" for x in out) or out[-1]["role"] != "assistant":
        raise ValueError("requires a user turn and final assistant answer")
    return out


def repair_examples(row: dict) -> list[dict]:
    """Only convert explicitly locally-replayed traces, never claimed success."""
    events = row.get("events")
    verification = row.get("verification")
    if not isinstance(events, list) or not events or not isinstance(verification, dict):
        raise ValueError("missing events or verification")
    if (verification.get("method") != "sandbox_replay"
        or verification.get("returncode") != 0
        or not isinstance(verification.get("tests_executed"), int)
        or isinstance(verification.get("tests_executed"), bool)
        or verification["tests_executed"] < 1
        or verification.get("trace_sha256") != digest(events)):
        raise ValueError("no matching successful local replay evidence")
    if not any(
        isinstance(e, dict) and e.get("kind") == "test"
        and e.get("returncode") == 0 and e.get("tests_executed", 0) > 0
        for e in events
    ):
        raise ValueError("no observed green test event")
    task = str(row.get("task", "")).strip()
    repository = str(row.get("repository", "")).strip()
    task_id = str(row.get("task_id", "")).strip()
    if not all((task, repository, task_id)):
        raise ValueError("missing task, repository or task_id")
    prefix = [{"role": "user", "content": f"Repository: {repository}\nTask: {task}"}]
    output = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("invalid event")
        action = event.get("action")
        observation = event.get("observation")
        # A successful trajectory can still contain unhelpful actions.
        if event.get("verdict") == "verified_good" and isinstance(action, dict):
            output.append({
                "messages": prefix + [{"role": "assistant", "content": canonical(action)}],
                "group": f"repair:{repository}:{task_id}"
            })
        if isinstance(action, dict):
            prefix.append({"role": "assistant", "content": canonical(action)})
        if isinstance(observation, str):
            prefix.append({"role": "user", "content": f"TOOL OBSERVATION: {observation}"})
    if not output:
        raise ValueError("no individually verified good actions")
    return output


def as_examples(source: dict, row: dict) -> list[dict]:
    if source["kind"] == "repair":
        return repair_examples(row)
    messages = chat_messages(row)
    first_user = next(x["content"] for x in messages if x["role"] == "user")
    return [{"messages": messages, "group": "prompt:" + digest(normalized(first_user))}]


def jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as stream:
        for line_num, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_num}: expected JSON object")
                yield row


def heldout_keys(path: Path | None) -> set[str]:
    if path is None:
        return set()
    keys = set()
    for row in jsonl(path):
        prompt = row.get("prompt")
        repository, task_id = row.get("repository"), row.get("task_id")
        if isinstance(prompt, str) and prompt.strip():
            keys.add("prompt:" + digest(normalized(prompt)))
        elif isinstance(repository, str) and isinstance(task_id, str) and repository and task_id:
            keys.add(f"repair:{repository}:{task_id}")
        else:
            raise ValueError("holdout rows require prompt or repository/task_id")
    return keys


def prepare(catalog_path: Path, inputs: dict[str, Path], out: Path, *,
            heldout: Path | None = None, dev_percent: int = 10,
            max_chars: int = 24000, max_rows: int = 1000) -> dict:
    if not 1 <= dev_percent <= 40 or max_rows < 1 or max_chars < 1:
        raise ValueError("invalid sampling settings")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    sources = catalog["sources"]
    for source_id in inputs:
        source = sources.get(source_id)
        if not isinstance(source, dict):
            raise ValueError(f"unregistered source: {source_id}")
        if source.get("approved") is not True or not source.get("license") or not source.get("review_reference"):
            raise PermissionError(f"license/provenance review required: {source_id}")
        if not source.get("revision"):
            raise ValueError(f"source revision must be pinned: {source_id}")
    protected = heldout_keys(heldout)
    seen = set()
    seen_chat_groups: set[str] = set()
    stats = Counter()
    outputs: dict[str, list[dict]] = {"train": [], "dev": []}
    records: list[dict] = []
    provenance: list[dict] = []
    quarantined: list[dict] = []
    near_bands: list[dict[str, str]] = [dict() for _ in range(LSH_BANDS)]
    for source_id, path in sorted(inputs.items()):
        source = sources[source_id]
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({"source": source_id, "revision": source["revision"],
                        "license": source["license"], "sha256": file_hash, "path": str(path)})
        accepted = 0
        for row_index, row in enumerate(jsonl(path), 1):
            if accepted >= max_rows:
                break
            try:
                examples = as_examples(source, row)
            except (ValueError, TypeError, KeyError):
                stats[f"{source_id}:unverified_or_malformed"] += 1
                continue
            for sample in examples:
                if accepted >= max_rows:
                    break
                messages = sample["messages"]
                first_user = next(x["content"] for x in messages if x["role"] == "user")
                key = "prompt:" + digest(normalized(first_user))
                if key in protected or sample["group"] in protected:
                    stats[f"{source_id}:holdout_collision"] += 1
                    continue
                signature = digest(messages)
                original_id = (sample["group"] if source["kind"] == "repair"
                               else f"{source_id}#row{row_index}")
                if signature in seen or (source["kind"] == "chat" and sample["group"] in seen_chat_groups):
                    stats[f"{source_id}:duplicate_or_conflict"] += 1
                    continue
                if sum(len(x["content"]) for x in messages) > max_chars:
                    stats[f"{source_id}:overlong"] += 1
                    continue
                near_key = near_duplicate_key(first_user)
                prior = None
                if near_key:
                    for band, value in enumerate(near_key):
                        prior = near_bands[band].get(value)
                        if prior is not None:
                            break
                if prior is not None:
                    stats[f"{source_id}:near_duplicate_quarantined"] += 1
                    quarantined.append({
                        "source": source_id, "original_id": original_id,
                        "reason": "near_duplicate",
                        "near_duplicate_of": prior, "content_sha256": signature,
                    })
                    continue
                seen.add(signature)
                if near_key:
                    for band, value in enumerate(near_key):
                        near_bands[band].setdefault(value, signature)
                if source["kind"] == "chat":
                    seen_chat_groups.add(sample["group"])
                split = "dev" if int(digest(sample["group"])[:8], 16) % 100 < dev_percent else "train"
                outputs[split].append({"messages": messages})  # Native Chowder chat contract
                provenance.append({
                    "source": source_id,
                    "revision": source["revision"],
                    "license": source["license"],
                    "original_id": original_id,
                    "source_example_row": row_index,
                    "task_category": source["kind"],
                    "verification": ("independent sandbox replay required (replay_smith.py)"
                                     if source["kind"] == "repair"
                                     else "published teacher traces; source-level review only"),
                    "content_sha256": signature,
                    "char_count": sum(len(x["content"]) for x in messages),
                    "token_count": count_tokens(messages),
                    "split": split,
                })
                stats[f"{source_id}:{split}"] += 1
                accepted += 1
    if not outputs["train"]:
        raise RuntimeError("no training examples passed provenance and quality gates")
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in outputs.items():
        (out / f"{split}.jsonl").write_text(
            "".join(canonical(x) + "\n" for x in rows), encoding="utf-8")
    if quarantined:
        (out / "quarantine.jsonl").write_text(
            "".join(canonical(x) + "\n" for x in quarantined), encoding="utf-8")
    per_source: dict[str, dict[str, int]] = {}
    for stat_key, count in sorted(stats.items()):
        source_name, _, reason = stat_key.partition(":")
        per_source.setdefault(source_name, {})[reason] = count
    rejected_total = sum(
        count for key, count in stats.items()
        if key.rsplit(":", 1)[-1] in
        ("unverified_or_malformed", "overlong", "holdout_collision", "duplicate_or_conflict")
    )
    quality_report = {
        "per_source": per_source,
        "totals": {
            "accepted": len(outputs["train"]) + len(outputs["dev"]),
            "rejected": rejected_total,
            "quarantined": len(quarantined),
        },
        "notes": [
            "near-duplicate screen is a leading-words+length-band check, not exhaustive fuzzy dedup",
            "quarantined rows are excluded from training and written to quarantine.jsonl",
            "token_count is recorded only when TFD_TOKENIZER is configured; char_count is always present",
            "no timestamps: the manifest stays byte-reproducible for fixed inputs",
        ],
    }
    manifest = {
        "format": "chowder-teacher-free-pilot-v1", "sources": records,
        "holdout_sha256": hashlib.sha256(heldout.read_bytes()).hexdigest() if heldout else None,
        "holdout_is_external": bool(heldout), "final_benchmark_generated": False,
        "counts": dict(sorted(stats.items())), "train_rows": len(outputs["train"]),
        "dev_rows": len(outputs["dev"]), "max_chars": max_chars,
        "max_rows_per_source": max_rows, "dev_percent": dev_percent,
        "provenance": provenance,
        "quality_report": quality_report,
        "note": "Sample-level QA is not proof of source correctness. Repair success requires independent sandbox replay."
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--input", action="append", required=True, metavar="SOURCE=PATH")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, help="External benchmark prompts; never copied to outputs")
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--max-chars", type=int, default=24000)
    args = parser.parse_args()
    inputs: dict[str, Path] = {}
    for item in args.input:
        source_id, sep, location = item.partition("=")
        if not sep or source_id in inputs:
            parser.error("--input must be unique SOURCE=PATH")
        inputs[source_id] = Path(location)
    print(json.dumps(prepare(args.catalog, inputs, args.out, heldout=args.holdout,
                             max_rows=args.max_rows, max_chars=args.max_chars), indent=2))


if __name__ == "__main__":
    main()
