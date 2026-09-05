"""Direct regression coverage for chowder.backends.training_data -- the
backend-neutral training-data contract extracted out of transformers_worker.py
(dataset digesting, replay sampling, bound-input verification, and
chat-template tokenization/completion-only labeling), so a future training
backend (e.g. the isolated Unsloth executor) can share it without silently
diverging from Transformers' own behavior.

tests/test_transformers_backend.py and tests/test_transformers_parent_adapter.py
already import several of these names via transformers_worker.py's re-export
and continue to pass unchanged -- this file exercises the module directly and
fills the specific gaps called out for this extraction: text digests, chat
digests, verify-bound-input, single-turn/multi-turn/system+user+assistant
chats, and a chat template that isn't prefix-consistent.
"""
from __future__ import annotations

import pytest

from chowder.backends.training_data import (
    _build_chat_example,
    _chat_digest,
    _render_chat_ids,
    _replay_sample_count,
    _text_digest,
    _validate_chat_messages,
    _verify_bound_adapter,
    _verify_bound_input,
)
from chowder.provenance import sha256_directory


class _FakeTokenizer:
    """Deterministic fake chat template, so _build_chat_example's
    prefix-consistency logic can be exercised without downloading a real
    tokenizer. Each turn renders as [role_marker, *content_char_codes,
    role_marker + 1]; add_generation_prompt appends a fixed marker. By
    construction this is prefix-consistent across turns (rendering messages
    [:k] is always a real prefix of rendering messages[:k+1]) unless
    `inconsistent=True`, which reverses multi-message renders to simulate a
    template _build_chat_example must reject rather than trust blindly.
    """

    _ROLE_MARKERS = {"system": 100, "user": 200, "assistant": 300}

    def __init__(self, inconsistent: bool = False) -> None:
        self.inconsistent = inconsistent

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        ids: list[int] = []
        for message in messages:
            marker = self._ROLE_MARKERS[message["role"]]
            ids.append(marker)
            ids.extend(ord(char) % 50 + 10 for char in message["content"])
            ids.append(marker + 1)
        if add_generation_prompt:
            # The generation prompt is exactly how an assistant turn begins
            # (the real-world shape: "<|assistant|>\n" *is* the prefix of a
            # real assistant turn's own rendering) -- this is what makes the
            # "normal" (non-inconsistent) case genuinely prefix-consistent.
            ids.append(self._ROLE_MARKERS["assistant"])
        if self.inconsistent and len(messages) > 1:
            ids = list(reversed(ids))
        return ids


# --- _text_digest / _chat_digest ----------------------------------------


class _ListDataset(list):
    """Minimal stand-in for a `datasets.Dataset`: __len__ + __getitem__."""


def test_text_digest_is_stable_for_identical_content():
    rows = _ListDataset([{"text": "hello"}, {"text": "world"}])
    assert _text_digest(rows, "text") == _text_digest(rows, "text")


def test_text_digest_changes_when_content_changes():
    a = _ListDataset([{"text": "hello"}])
    b = _ListDataset([{"text": "goodbye"}])
    assert _text_digest(a, "text") != _text_digest(b, "text")


def test_text_digest_distinguishes_concatenation_boundaries():
    """Length-prefixing each row's bytes matters: without it, ["ab","c"]
    and ["a","bc"] would hash identically."""
    a = _ListDataset([{"text": "ab"}, {"text": "c"}])
    b = _ListDataset([{"text": "a"}, {"text": "bc"}])
    assert _text_digest(a, "text") != _text_digest(b, "text")


def test_chat_digest_is_stable_for_identical_content():
    rows = _ListDataset(
        [{"messages": [{"role": "user", "content": "hi"}]}]
    )
    assert _chat_digest(rows, "messages") == _chat_digest(rows, "messages")


def test_chat_digest_changes_when_messages_change():
    a = _ListDataset([{"messages": [{"role": "user", "content": "hi"}]}])
    b = _ListDataset([{"messages": [{"role": "user", "content": "bye"}]}])
    assert _chat_digest(a, "messages") != _chat_digest(b, "messages")


def test_chat_digest_is_independent_of_key_order():
    """json.dumps(..., sort_keys=True) means dict key order in the source
    row must not change the digest."""
    a = _ListDataset([{"messages": [{"role": "user", "content": "hi"}]}])
    b = _ListDataset([{"messages": [{"content": "hi", "role": "user"}]}])
    assert _chat_digest(a, "messages") == _chat_digest(b, "messages")


# --- _replay_sample_count (additional edge cases) ------------------------


def test_replay_sample_count_rejects_negative_rows():
    with pytest.raises(ValueError, match="cannot be negative"):
        _replay_sample_count(-1, 4, 1.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        _replay_sample_count(4, -1, 1.0)


@pytest.mark.parametrize("ratio", [0.0, -1.0, float("inf"), float("nan")])
def test_replay_sample_count_rejects_non_finite_or_non_positive_ratio(ratio):
    with pytest.raises(ValueError, match="finite and positive"):
        _replay_sample_count(4, 4, ratio)


def test_replay_sample_count_never_exceeds_available_replay_rows():
    assert _replay_sample_count(1000, 3, 1.0) == 3


# --- _verify_bound_input ---------------------------------------------------


def test_verify_bound_input_returns_digest_when_no_expected_sha_given(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hi"}\n', encoding="utf-8")
    digest = _verify_bound_input(str(data), None, label="training")
    assert len(digest) == 64


def test_verify_bound_input_accepts_matching_expected_sha(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hi"}\n', encoding="utf-8")
    digest = _verify_bound_input(str(data), None, label="training")
    assert _verify_bound_input(str(data), digest, label="training") == digest


def test_verify_bound_input_rejects_a_changed_file(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hi"}\n', encoding="utf-8")
    digest = _verify_bound_input(str(data), None, label="training")
    data.write_text('{"text":"changed"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="digest changed before worker load"):
        _verify_bound_input(str(data), digest, label="training")


def test_verify_bound_input_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        _verify_bound_input(str(tmp_path / "missing.jsonl"), None, label="training")


# --- _verify_bound_adapter (complementary to test_transformers_parent_adapter.py) --


def test_verify_bound_adapter_rejects_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        _verify_bound_adapter(str(tmp_path / "missing-adapter"), "a" * 64, label="parent")


def test_verify_bound_adapter_accepts_matching_directory_digest(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    digest = sha256_directory(adapter)
    assert _verify_bound_adapter(str(adapter), digest, label="parent") == digest


# --- _validate_chat_messages: additional role-combination coverage --------


def test_validate_chat_messages_accepts_system_user_assistant():
    normalized = _validate_chat_messages(
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        row_index=0,
    )
    assert [m["role"] for m in normalized] == ["system", "user", "assistant"]


def test_validate_chat_messages_accepts_multi_turn_conversation():
    normalized = _validate_chat_messages(
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        ],
        row_index=0,
    )
    assert len(normalized) == 4


def test_validate_chat_messages_coerces_non_string_content_to_string():
    normalized = _validate_chat_messages(
        [{"role": "user", "content": 42}, {"role": "assistant", "content": "ok"}],
        row_index=0,
    )
    assert normalized[0]["content"] == "42"


# --- _render_chat_ids / _build_chat_example against the fake tokenizer ----


def test_render_chat_ids_appends_generation_marker_only_when_requested():
    tokenizer = _FakeTokenizer()
    messages = [{"role": "user", "content": "hi"}]
    without_prompt = _render_chat_ids(tokenizer, messages, add_generation_prompt=False)
    with_prompt = _render_chat_ids(tokenizer, messages, add_generation_prompt=True)
    assert with_prompt == without_prompt + [tokenizer._ROLE_MARKERS["assistant"]]


def test_build_chat_example_single_turn_masks_the_user_turn():
    tokenizer = _FakeTokenizer()
    messages = _validate_chat_messages(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        row_index=0,
    )
    example = _build_chat_example(tokenizer, messages, max_length=100, row_index=0)
    assert len(example["input_ids"]) == len(example["labels"]) == len(example["attention_mask"])
    assert all(mask == 1 for mask in example["attention_mask"])
    full_ids = _render_chat_ids(tokenizer, messages, add_generation_prompt=False)
    assert example["input_ids"] == full_ids
    # The masked-in span is exactly what follows the generation-prompt
    # prefix (the assistant role marker itself is prompt, not completion).
    prefix_ids = _render_chat_ids(tokenizer, messages[:1], add_generation_prompt=True)
    expected_unmasked = full_ids[len(prefix_ids):]
    unmasked = [tid for tid, label in zip(example["input_ids"], example["labels"]) if label != -100]
    assert unmasked == expected_unmasked


def test_build_chat_example_multi_turn_masks_every_assistant_span():
    tokenizer = _FakeTokenizer()
    messages = _validate_chat_messages(
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        ],
        row_index=0,
    )
    example = _build_chat_example(tokenizer, messages, max_length=100, row_index=0)
    unmasked_count = sum(1 for label in example["labels"] if label != -100)
    # Two assistant turns, each contributing 1 content char + 1 end marker
    # (the leading role marker is the generation-prompt prefix, not completion).
    assert unmasked_count == 4


def test_build_chat_example_system_user_assistant_masks_only_assistant():
    tokenizer = _FakeTokenizer()
    messages = _validate_chat_messages(
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ],
        row_index=0,
    )
    example = _build_chat_example(tokenizer, messages, max_length=100, row_index=0)
    unmasked_count = sum(1 for label in example["labels"] if label != -100)
    assert unmasked_count == 2  # assistant turn's content char + end marker


def test_build_chat_example_rejects_a_non_prefix_consistent_template():
    tokenizer = _FakeTokenizer(inconsistent=True)
    messages = _validate_chat_messages(
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        ],
        row_index=3,
    )
    with pytest.raises(RuntimeError, match="chat template is not prefix-consistent"):
        _build_chat_example(tokenizer, messages, max_length=100, row_index=3)


def test_build_chat_example_truncation_can_still_leave_assistant_tokens():
    tokenizer = _FakeTokenizer()
    messages = _validate_chat_messages(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        row_index=0,
    )
    full = _build_chat_example(tokenizer, messages, max_length=100, row_index=0)
    truncated = _build_chat_example(
        tokenizer, messages, max_length=len(full["input_ids"]) - 1, row_index=0
    )
    assert len(truncated["input_ids"]) == len(full["input_ids"]) - 1
    assert any(label != -100 for label in truncated["labels"])


def test_build_chat_example_raises_when_truncation_removes_all_assistant_tokens():
    tokenizer = _FakeTokenizer()
    messages = _validate_chat_messages(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        row_index=0,
    )
    with pytest.raises(RuntimeError, match="no assistant tokens remain after truncating"):
        _build_chat_example(tokenizer, messages, max_length=1, row_index=0)
