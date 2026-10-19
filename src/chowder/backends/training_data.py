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


_CHAT_ROLES = {"system", "user", "assistant", "tool"}
#: Trailing generation-prompt tokens allowed to re-tokenise when the assistant
#: turn is appended (see _build_chat_example).
_MAX_BOUNDARY_RETOKENIZATION = 2
#: Agentic turn fields the chat template renders and the trainer must keep:
#: an assistant's tool calls and reasoning, a tool turn's call id and name.
_CHAT_EXTRA_FIELDS = ("tool_calls", "reasoning_content", "tool_call_id", "name")


def _validate_chat_messages(raw: Any, *, row_index: int) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(
            f"chat dataset row {row_index} has an empty or invalid messages list"
        )
    normalized: list[dict[str, Any]] = []
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
        content = turn["content"]
        message: dict[str, Any] = {"role": role, "content": "" if content is None else str(content)}
        for key in _CHAT_EXTRA_FIELDS:
            if turn.get(key) not in (None, "", []):
                message[key] = turn[key]
        if "tool_calls" in message:
            message["tool_calls"] = _parse_tool_calls(message["tool_calls"], row_index=row_index)
        normalized.append(message)
    if not has_assistant:
        raise RuntimeError(
            f"chat dataset row {row_index} has no assistant turn -- nothing to train on"
        )
    return normalized


def _parse_tool_calls(raw: Any, *, row_index: int) -> list[dict[str, Any]]:
    """Tool calls with dict arguments, the shape chat templates iterate.

    Arguments may arrive as JSON strings, and should: ``datasets`` loads JSONL
    through Arrow, which unifies nested dicts across rows and would add every
    other row's argument keys to this one as nulls.
    """
    if not isinstance(raw, list):
        raise RuntimeError(f"chat dataset row {row_index} has a non-list tool_calls")
    calls = []
    for call in raw:
        function = dict((call or {}).get("function") or {})
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                function["arguments"] = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"chat dataset row {row_index} has tool-call arguments that are not JSON"
                ) from exc
        calls.append({**call, "function": function})
    return calls


def parse_tools(raw: Any, *, row_index: int) -> list[Any] | None:
    """A row's tool schemas: a list, or a JSON string of one (Arrow-safe), or empty."""
    if raw in (None, "", []):
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"chat dataset row {row_index} has a tools value that is not JSON") from exc
    if not isinstance(raw, list):
        raise RuntimeError(f"chat dataset row {row_index} has tools that are not a list")
    return raw


def _render_chat_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    tools: list[Any] | None = None,
) -> list[int]:
    kwargs: dict[str, Any] = {"tools": tools} if tools else {}
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt, **kwargs
    )
    return list(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)


def _build_chat_example(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    max_length: int,
    row_index: int,
    tools: list[Any] | None = None,
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
    full_ids = _render_chat_ids(tokenizer, messages, add_generation_prompt=False, tools=tools)
    try:
        labels = _prefix_labels(tokenizer, messages, full_ids, row_index=row_index, tools=tools)
    except RuntimeError:
        # Qwen3.5/3.6 drop an earlier assistant turn's reasoning once a later user
        # turn follows, so turn-by-turn renders are never a prefix of the whole.
        # For a ChatML template the assistant turns are still delimited exactly in
        # the full render, so label them there -- as the model sees them in
        # history. A template without those markers keeps refusing.
        labels = _chatml_assistant_labels(tokenizer, messages, full_ids, row_index=row_index)
    return _truncate_example(full_ids, labels, max_length=max_length, row_index=row_index)


def _chatml_assistant_labels(
    tokenizer: Any, messages: list[dict[str, Any]], full_ids: list[int], *, row_index: int
) -> list[int]:
    """Label every ``<|im_start|>assistant\\n ... <|im_end|>`` span of the full render.

    Refuses unless the template is ChatML and the span count equals the number
    of assistant messages, so a mislabel cannot pass silently."""
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    start_id, end_id = vocab.get("<|im_start|>"), vocab.get("<|im_end|>")
    header = tokenizer.encode("assistant\n", add_special_tokens=False) if start_id is not None else []
    if start_id is None or end_id is None or not header:
        raise RuntimeError(
            f"chat dataset row {row_index}: chat template is not prefix-consistent "
            "across turns; cannot compute a reliable completion-only loss mask"
        )
    labels = [-100] * len(full_ids)
    spans = 0
    position = 0
    while position < len(full_ids):
        if full_ids[position] == start_id and full_ids[position + 1 : position + 1 + len(header)] == header:
            body = position + 1 + len(header)
            end = body
            while end < len(full_ids) and full_ids[end] != end_id:
                end += 1
            if end == len(full_ids):
                break
            labels[body : end + 1] = full_ids[body : end + 1]
            spans += 1
            position = end + 1
            continue
        position += 1
    expected = sum(1 for turn in messages if turn["role"] == "assistant")
    if spans != expected:
        raise RuntimeError(
            f"chat dataset row {row_index}: found {spans} ChatML assistant spans for "
            f"{expected} assistant turns; cannot compute a reliable completion-only loss mask"
        )
    return labels


def _prefix_labels(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    full_ids: list[int],
    *,
    row_index: int,
    tools: list[Any] | None,
) -> list[int]:
    labels = [-100] * len(full_ids)
    for index, turn in enumerate(messages):
        # Only assistant turns are trained on; tool results stay unlabelled.
        if turn["role"] != "assistant":
            continue
        prefix_ids = _render_chat_ids(
            tokenizer, messages[:index], add_generation_prompt=True, tools=tools
        )
        through_ids = _render_chat_ids(
            tokenizer, messages[: index + 1], add_generation_prompt=False, tools=tools
        )
        # The generation prompt may end mid-merge: Qwen3.5/3.6 opens the turn with
        # "<think>" + "\n" while the finished turn reads "<think>" + "\n\n" +
        # "</think>", and BPE merges the newlines differently. The span starts at
        # the longest common token prefix, tolerating at most this many trailing
        # boundary tokens; an exact-prefix template is unaffected.
        start = 0
        limit = min(len(prefix_ids), len(full_ids))
        while start < limit and full_ids[start] == prefix_ids[start]:
            start += 1
        if (
            len(prefix_ids) - start > _MAX_BOUNDARY_RETOKENIZATION
            or len(through_ids) > len(full_ids)
            or full_ids[: len(through_ids)] != through_ids
            or start > len(through_ids)
        ):
            raise RuntimeError(
                f"chat dataset row {row_index}: chat template is not prefix-consistent "
                "across turns; cannot compute a reliable completion-only loss mask"
            )
        labels[start : len(through_ids)] = full_ids[start : len(through_ids)]
    return labels


def _truncate_example(
    full_ids: list[int], labels: list[int], *, max_length: int, row_index: int
) -> dict[str, list[int]]:
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


def _chat_digest(dataset: Any, messages_field: str, tools_field: str = "") -> str:
    digest = hashlib.sha256()
    for index in range(len(dataset)):
        row = dataset[index][messages_field]
        # The tools render into the training text, so they are part of it; with no
        # tools field the digest is exactly what it always was.
        value = {"messages": row, "tools": dataset[index][tools_field]} if tools_field else row
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()
