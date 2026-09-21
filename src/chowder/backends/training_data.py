from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ..provenance import sha256_directory

# Backend-neutral training-data contract: dataset digesting, replay sampling,
# bound-input verification, and chat-template tokenization/completion-only
# labeling. Extracted out of transformers_worker.py so a future training
# backend (e.g. the isolated Unsloth executor) can share the exact same
# dataset/parent-adapter binding and chat-tokenization behavior instead of
# duplicating it -- any divergence here would mean two backends training on
# "the same" dataset in subtly different ways.


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_bound_input(path: str, expected_sha: str | None, *, label: str) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} dataset not found: {resolved}")
    actual = _sha256_file(resolved)
    if expected_sha is not None and actual != expected_sha:
        raise RuntimeError(f"{label} dataset digest changed before worker load")
    return actual


def _verify_bound_adapter(path: str, expected_sha: str, *, label: str) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} adapter not found: {resolved}")
    actual = sha256_directory(resolved)
    if actual != expected_sha:
        raise RuntimeError(f"{label} adapter digest changed before worker load")
    return actual


def _replay_sample_count(primary_rows: int, replay_rows: int, ratio: float) -> int:
    if primary_rows < 0 or replay_rows < 0:
        raise ValueError("dataset row counts cannot be negative")
    if replay_rows == 0 or primary_rows == 0:
        return 0
    if not math.isfinite(float(ratio)) or ratio <= 0:
        raise ValueError("replay ratio must be finite and positive")
    return min(replay_rows, max(1, math.ceil(primary_rows * float(ratio))))


def _text_digest(dataset: Any, text_field: str) -> str:
    digest = hashlib.sha256()
    for index in range(len(dataset)):
        text = str(dataset[index][text_field]).encode("utf-8")
        digest.update(len(text).to_bytes(8, "big"))
        digest.update(text)
    return digest.hexdigest()


_CHAT_ROLES = {"system", "user", "assistant"}


def _validate_chat_messages(raw: Any, *, row_index: int) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(
            f"chat dataset row {row_index} has an empty or invalid messages list"
        )
    normalized: list[dict[str, str]] = []
    has_assistant = False
    for turn in raw:
        if not isinstance(turn, dict) or "role" not in turn or "content" not in turn:
            raise RuntimeError(
                f"chat dataset row {row_index} has a message missing role/content"
            )
        role = str(turn["role"])
        if role not in _CHAT_ROLES:
            raise RuntimeError(
                f"chat dataset row {row_index} has unsupported message role {role!r}; "
                f"supported roles are {sorted(_CHAT_ROLES)}"
            )
        if role == "assistant":
            has_assistant = True
        normalized.append({"role": role, "content": str(turn["content"])})
    if not has_assistant:
        raise RuntimeError(
            f"chat dataset row {row_index} has no assistant turn -- nothing to train on"
        )
    return normalized


def _render_chat_ids(
    tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool
) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    return list(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)


def _build_chat_example(
    tokenizer: Any, messages: list[dict[str, str]], *, max_length: int, row_index: int
) -> dict[str, list[int]]:
    """Tokenize one conversation with completion-only (assistant-turn) labels.

    Does not rely on the chat template defining a ``{% generation %}`` block
    -- most real templates (including the official Llama 3.1 template) don't.
    Instead, for each assistant turn, renders the conversation twice: once up
    to (not including) that turn with ``add_generation_prompt=True`` (the
    exact point the assistant's own tokens begin), and once through the end
    of that turn. The token-length difference is exactly the assistant's own
    generated span, verified to be a real prefix of the full sequence before
    being trusted -- a template that isn't prefix-consistent raises rather
    than silently mislabeling.
    """
    full_ids = _render_chat_ids(tokenizer, messages, add_generation_prompt=False)
    labels = [-100] * len(full_ids)
    for index, turn in enumerate(messages):
        if turn["role"] != "assistant":
            continue
        prefix_ids = _render_chat_ids(tokenizer, messages[:index], add_generation_prompt=True)
        through_ids = _render_chat_ids(
            tokenizer, messages[: index + 1], add_generation_prompt=False
        )
        if (
            len(prefix_ids) > len(full_ids)
            or len(through_ids) > len(full_ids)
            or full_ids[: len(prefix_ids)] != prefix_ids
            or full_ids[: len(through_ids)] != through_ids
        ):
            raise RuntimeError(
                f"chat dataset row {row_index}: chat template is not prefix-consistent "
                "across turns; cannot compute a reliable completion-only loss mask"
            )
        labels[len(prefix_ids) : len(through_ids)] = full_ids[len(prefix_ids) : len(through_ids)]

    full_ids = full_ids[:max_length]
    labels = labels[:max_length]
    if not any(label != -100 for label in labels):
        raise RuntimeError(
            f"chat dataset row {row_index}: no assistant tokens remain after truncating "
            f"to max_length={max_length} -- increase max_length or shorten this conversation"
        )
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def _chat_digest(dataset: Any, messages_field: str) -> str:
    digest = hashlib.sha256()
    for index in range(len(dataset)):
        payload = json.dumps(
            dataset[index][messages_field], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()
