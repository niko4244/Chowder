"""P4: the *rendered* prompt bytes are protocol identity, not folklore.

Protocol v3 exists because two parents that tokenize identically can carry
different chat templates (the C/D finding), so a suite scored under one
rendering is not comparable with a suite scored under another. The suite spec
already carried the *request* (`use_chat_template`, `canonical_rendering`);
what was missing is what the worker actually rendered with.

These tests pin the shared renderer that both text workers must call --
sharing it is the point: `scoring.py` had to be extracted because the two
workers' copies drifted, and a rendering drift is worse, because it changes
the prompt bytes rather than only the score.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import chowder
from chowder.canonical_chat_template import canonical_template_sha256
from chowder.evaluators.rendering import (
    RENDERING_CANONICAL_TEMPLATE,
    RENDERING_RAW,
    RENDERING_TOKENIZER_TEMPLATE,
    render_prompt,
    tokenizer_template_sha256,
)


class _Tokenizer:
    """Enough tokenizer to exercise the three rendering paths."""

    def __init__(self, template=None):
        self.chat_template = template
        self.pad_token_id = 0
        self.eos_token_id = 1

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, chat_template=None
    ):
        template = self.chat_template if chat_template is None else chat_template
        if template is None:
            raise ValueError("tokenizer has no chat template")
        body = messages[0]["content"]
        return f"<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n"


def _render(tokenizer, *, use_chat_template, canonical_rendering, prompt="2+2?"):
    return render_prompt(
        tokenizer=tokenizer,
        prompt=prompt,
        suite_name="quality",
        use_chat_template=use_chat_template,
        canonical_rendering=canonical_rendering,
    )


def test_raw_rendering_is_recorded_and_carries_no_template_digest():
    text, evidence = _render(_Tokenizer(), use_chat_template=False, canonical_rendering=False)
    assert text == "2+2?"
    assert evidence == {"rendering": RENDERING_RAW}
    assert "chat_template_sha256" not in evidence


def test_tokenizer_template_digest_is_the_template_bytes_not_the_prompt():
    tokenizer = _Tokenizer("TEMPLATE-A")
    text, evidence = _render(tokenizer, use_chat_template=True, canonical_rendering=False)
    assert evidence["rendering"] == RENDERING_TOKENIZER_TEMPLATE
    assert evidence["chat_template_sha256"] == hashlib.sha256(b"TEMPLATE-A").hexdigest()
    assert text != "2+2?"  # rendered, not raw
    # a different prompt must not change the template identity
    _, other = _render(tokenizer, use_chat_template=True, canonical_rendering=False, prompt="9+9?")
    assert other["chat_template_sha256"] == evidence["chat_template_sha256"]


def test_a_different_tokenizer_template_is_a_different_identity():
    """The C/D finding at the identity layer: identical tokenizers carrying
    different templates must not share a rendering identity."""
    _, a = _render(_Tokenizer("TEMPLATE-A"), use_chat_template=True, canonical_rendering=False)
    _, b = _render(_Tokenizer("TEMPLATE-B"), use_chat_template=True, canonical_rendering=False)
    assert a["chat_template_sha256"] != b["chat_template_sha256"]


def test_named_template_mappings_hash_canonically():
    named = {"default": "TEMPLATE-A", "tool_use": "TEMPLATE-B"}
    _, evidence = _render(
        _Tokenizer(named), use_chat_template=True, canonical_rendering=False
    )
    expected = hashlib.sha256(
        json.dumps(named, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert evidence["chat_template_sha256"] == expected
    assert tokenizer_template_sha256(_Tokenizer(None)) is None


def test_canonical_rendering_reports_the_pinned_digest_not_the_tokenizer_template():
    tokenizer = _Tokenizer("SOME-CHECKPOINT-OWN-TEMPLATE")
    _, evidence = _render(tokenizer, use_chat_template=True, canonical_rendering=True)
    assert evidence["rendering"] == RENDERING_CANONICAL_TEMPLATE
    assert evidence["chat_template_sha256"] == canonical_template_sha256()
    assert evidence["chat_template_sha256"] != tokenizer_template_sha256(tokenizer)


def test_canonical_rendering_needs_no_tokenizer_template():
    """Canonical rendering supplies its own template, which is the whole
    point: a parent whose re-serialized tokenizer lost or changed the
    template still renders the same bytes as every other parent."""
    text, evidence = _render(_Tokenizer(None), use_chat_template=True, canonical_rendering=True)
    assert text and evidence["rendering"] == RENDERING_CANONICAL_TEMPLATE


def test_tokenizer_template_request_without_a_template_refuses():
    with pytest.raises(RuntimeError, match="no chat template"):
        _render(_Tokenizer(None), use_chat_template=True, canonical_rendering=False)


def test_canonical_rendering_without_use_chat_template_refuses():
    with pytest.raises(RuntimeError, match="canonical_rendering without use_chat_template"):
        _render(_Tokenizer("X"), use_chat_template=False, canonical_rendering=True)


# ---- the no-drift pin: one renderer, both workers ---------------------------

_WORKERS = (
    Path(chowder.__file__).parent / "evaluators" / "transformers_text_worker.py",
    Path(chowder.__file__).parent / "evaluators" / "base_text_worker.py",
)


@pytest.mark.parametrize("worker_path", _WORKERS)
def test_every_text_worker_renders_through_the_shared_helper(worker_path):
    text = worker_path.read_text(encoding="utf-8")
    assert "render_prompt" in text, f"{worker_path.name} does not use the shared renderer"
    assert "apply_chat_template" not in text, (
        f"{worker_path.name} carries its own rendering call -- that is the drift "
        "the shared helper exists to prevent"
    )


@pytest.mark.parametrize("worker_path", _WORKERS)
def test_every_text_worker_reports_the_rendering_in_suite_evidence(worker_path):
    text = worker_path.read_text(encoding="utf-8")
    # the evidence dict the worker writes per suite must spread the renderer's
    # evidence, not just use the rendered text
    assert "render_evidence" in text, (
        f"{worker_path.name} renders but never records what it rendered with"
    )
