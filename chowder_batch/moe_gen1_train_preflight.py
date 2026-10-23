"""CPU-only admission check for the existing MoE Gen-1 Fable/control chat sets.

The historical mix was screened at 4096 tokens but the frozen training recipe
uses 2048. The shared trainer otherwise truncates a response after it has
already kept some assistant labels. This check binds both files to their
manifest and uses the *same* tokenizer and masking code as the real workers;
it refuses any row that would be shortened by the training sequence ceiling.

Usage::

    python chowder_batch/moe_gen1_train_preflight.py \
        --mix C:\\llm-work\\train-sets\\moe-gen1-v1 \
        --tokenizer G:\\llm-cold\\bf16\\Qwen3.6-35B-A3B --max-length 2048

It reads only local files and performs no model-weight load or GPU operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from chowder.backends.training_data import (  # noqa: E402
    _build_chat_example,
    _validate_chat_messages,
    parse_tools,
)
from chowder.backends.tokenizer_assets import tokenizer_asset_digests  # noqa: E402

ARMS = ("train-fable.jsonl", "train-control.jsonl")


def _sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _canonical_hash(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def inspect_mix(
    mix: Path,
    tokenizer: Any,
    *,
    max_length: int = 2048,
    max_examples: int = 8,
) -> dict[str, Any]:
    """Return a fail-closed, independently reproducible training admission receipt.

    Both arms are checked because training just the Fable arm without its
    Fable-free control cannot establish the intended treatment contrast.
    This does *not* re-run the protected contamination firewall; the SHA-bound
    builder output and its manifest must already have that evidence.
    """
    if max_length <= 0 or max_examples < 0:
        raise ValueError("max_length must be positive and max_examples nonnegative")
    manifest = json.loads((mix / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("builder") != "chowder_batch/build_moe_gen1_mix.py":
        raise ValueError("unrecognized MoE Gen-1 mix builder")
    if manifest.get("template_validated") is not True:
        raise ValueError("mix was built without chat-template validation")
    declared_tokenizer = manifest.get("tokenizer")
    if (
        not isinstance(declared_tokenizer, dict)
        or declared_tokenizer.get("loaded") is not True
        or not isinstance(declared_tokenizer.get("path"), str)
        or not declared_tokenizer["path"]
    ):
        raise ValueError("mix manifest does not certify a loaded tokenizer")
    if not isinstance(manifest.get("outputs"), dict):
        raise ValueError("mix manifest is missing output identities")

    # SHA, length and the exact rows are bound before any tokenization.  A
    # missing / altered row is a refused admission, not silently re-indexed.
    arm_rows: dict[str, list[dict[str, Any]]] = {}
    for filename in ARMS:
        metadata = manifest["outputs"].get(filename)
        if not isinstance(metadata, dict) or not isinstance(metadata.get("sha256"), str):
            raise ValueError(f"manifest is missing {filename} digest")
        path = mix / filename
        if _sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"{filename} SHA-256 does not match manifest")
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if len(rows) != metadata.get("rows"):
            raise ValueError(f"{filename} row count does not match manifest")
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"{filename} contains a non-object row")
        arm_rows[filename] = rows

    # The control must be an order-preserving subset.  Hashes are canonical
    # because JSON whitespace and key ordering do not affect row identity.
    treatment = [_canonical_hash(row) for row in arm_rows[ARMS[0]]]
    control = [_canonical_hash(row) for row in arm_rows[ARMS[1]]]
    cursor = 0
    for digest in treatment:
        if cursor < len(control) and digest == control[cursor]:
            cursor += 1
    if cursor != len(control) or len(control) >= len(treatment):
        raise ValueError("Fable-free control is not a strict subsequence of the Fable arm")

    checked: dict[str, dict[str, Any]] = {}
    tokenized_cache: dict[str, tuple[int, int, int]] = {}
    for filename in ARMS:
        over_length = 0
        lost_answer = 0
        entirely_lost_answer = 0
        invalid = 0
        examples: list[dict[str, Any]] = []
        maximum = 0
        for index, row in enumerate(arm_rows[filename]):
            digest = _canonical_hash(row)
            try:
                if digest not in tokenized_cache:
                    messages = _validate_chat_messages(row.get("messages"), row_index=index)
                    # Effectively unbounded: only the real worker's final
                    # 2048 truncation must be under examination.
                    rendered = _build_chat_example(
                        tokenizer,
                        messages,
                        max_length=1 << 30,
                        row_index=index,
                        tools=parse_tools(row.get("tools"), row_index=index),
                    )
                    labels = rendered["labels"]
                    tokenized_cache[digest] = (
                        len(rendered["input_ids"]),
                        sum(label != -100 for label in labels),
                        sum(label != -100 for label in labels[max_length:]),
                    )
                length, assistants, lost = tokenized_cache[digest]
            except (RuntimeError, TypeError, ValueError, KeyError) as exc:
                invalid += 1
                if len(examples) < max_examples:
                    examples.append({"row": index + 1, "problem": "invalid_chat", "error": str(exc)})
                continue
            maximum = max(maximum, length)
            if length > max_length:
                over_length += 1
                if lost:
                    lost_answer += 1
                    if lost == assistants:
                        entirely_lost_answer += 1
                if len(examples) < max_examples:
                    examples.append(
                        {"row": index + 1, "problem": "over_length", "tokens": length,
                         "assistant_tokens_lost": lost}
                    )
        checked[filename] = {
            "rows": len(arm_rows[filename]),
            "max_rendered_tokens": maximum,
            "over_length": over_length,
            "answer_span_truncated": lost_answer,
            "answer_span_completely_removed": entirely_lost_answer,
            "invalid_chat": invalid,
            "examples": examples,
        }

    return {
        "scope": "bound Gen-1 chat mix / exact tokenizer / full 2048-token sequences only",
        "ready": all(
            info["over_length"] == 0 and info["invalid_chat"] == 0
            for info in checked.values()
        ),
        "max_length": max_length,
        "manifest_sha256": _sha256_file(mix / "manifest.json"),
        "tokenizer_manifest_path": declared_tokenizer["path"],
        "fable_only_rows": len(treatment) - len(control),
        "arms": checked,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--report", type=Path, help="optional path for a JSON receipt")
    args = parser.parse_args(argv)
    manifest = json.loads((args.mix / "manifest.json").read_text(encoding="utf-8"))
    declared = manifest.get("tokenizer")
    if not isinstance(declared, dict) or not isinstance(declared.get("path"), str):
        parser.error("mix manifest is missing its original tokenizer path")
    if args.tokenizer.resolve() != Path(declared["path"]).resolve():
        parser.error("tokenizer path differs from the mix-builder manifest")
    if not (args.tokenizer / "tokenizer.json").is_file():
        parser.error("local tokenizer.json is required to check complete answer spans")
    from transformers import AutoTokenizer

    assets_before = tokenizer_asset_digests(args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), trust_remote_code=False, local_files_only=True)
    result = inspect_mix(args.mix, tokenizer, max_length=args.max_length)
    assets_after = tokenizer_asset_digests(args.tokenizer)
    if assets_after != assets_before:
        parser.error("tokenizer assets changed during admission; rerun the 2048-token preflight")
    result["tokenizer_json_sha256"] = _sha256_file(args.tokenizer / "tokenizer.json")
    if (args.tokenizer / "tokenizer_config.json").is_file():
        result["tokenizer_config_sha256"] = _sha256_file(args.tokenizer / "tokenizer_config.json")
    result["tokenizer_assets_sha256"] = assets_after
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
