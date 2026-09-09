"""Protocol v3 tests: canonical rendering, behavioral tokenizer gate, digest additivity.

The scientific core of this module: **v2 digests must reproduce
byte-for-byte** (retry7's recorded `c5e964df...` is banked evidence — any
drift would silently orphan the A/B result), while **v3 is a visibly
different fingerprint** (a new protocol generation must never masquerade
as v2). The behavioral gate is exercised over a synthetic corpus with a
planted divergence, so the fail-closed path is proven without any GPU or
model load.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chowder.parent_tournament as pt
from chowder.canonical_chat_template import (
    canonical_template_sha256,
    render_canonical,
    verify_canonical_template,
)
from chowder.parent_eval import ParentEvalSpec, ParentSuiteValidationError
from chowder.parent_suite_content import build_tournament_spec

# ---------------------------------------------------------------------------
# Canonical template module
# ---------------------------------------------------------------------------


def test_embedded_template_matches_pinned_digest():
    # The module must fail closed if its template bytes were altered.
    text = verify_canonical_template()  # raises on mismatch
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == canonical_template_sha256()
    assert len(text) > 1000  # the real Qwen3.8 template, not a stub


def test_render_canonical_uses_the_pinned_template_not_the_tokenizers_own():
    class FakeTokenizer:
        chat_template = "TOKENIZER'S OWN TEMPLATE — MUST NOT BE USED"

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, chat_template=None):
            assert chat_template is not None, "canonical template was not passed"
            assert chat_template == verify_canonical_template()
            assert messages == [{"role": "user", "content": "hi"}]
            assert add_generation_prompt is True
            return "RENDERED:" + chat_template[:10]

    out = render_canonical(FakeTokenizer(), "hi")
    assert out.startswith("RENDERED:")


# ---------------------------------------------------------------------------
# Digest additivity: v2 unchanged, v3 visibly different
# ---------------------------------------------------------------------------


def _suites_for_spec():
    # Minimal valid spec: one suite per dimension is required by coverage
    # validation; build them directly from the content module's names.
    from chowder.parent_suite_content import PROTECTED_SUITES

    suites = []
    for dimension, named in PROTECTED_SUITES.items():
        for suite_name in named:
            suites.append(
                {
                    "name": suite_name,
                    "dimension": dimension,
                    "dataset": f"/tmp/{suite_name}.jsonl",
                    "use_chat_template": True,
                }
            )
    return suites


def test_v2_digest_is_unchanged_by_v3_fields():
    """retry7's recorded digest must reproduce from v2-shaped data."""
    spec_v2 = ParentEvalSpec.from_dict(
        {"suites": _suites_for_spec(), "protocol_version": "v2"}
    )
    assert spec_v2.protocol_version == "v2"
    assert spec_v2.canonical_rendering is False
    assert spec_v2.canonical_template_sha256 is None
    # The v2 canonical JSON must not contain the v3 keys at all.
    assert "canonical_rendering" not in spec_v2.canonical_json()
    assert "canonical_template_sha256" not in spec_v2.canonical_json()


def test_from_dict_preserves_protocol_version():
    """Regression: from_dict used to drop protocol_version (re-default v2)."""
    spec = ParentEvalSpec.from_dict(
        {"suites": _suites_for_spec(), "protocol_version": "v3"}
    )
    assert spec.protocol_version == "v3"


def test_v3_spec_has_different_digest_and_carries_the_pin():
    data = {"suites": _suites_for_spec(), "protocol_version": "v2"}
    v2 = ParentEvalSpec.from_dict(data)
    v3 = ParentEvalSpec.from_dict(
        {
            **data,
            "protocol_version": "v3",
            "canonical_rendering": True,
            "canonical_template_sha256": canonical_template_sha256(),
        }
    )
    assert v2.digest() != v3.digest()
    assert v3.canonical_rendering is True
    assert v3.canonical_template_sha256 == canonical_template_sha256()
    assert "canonical_rendering" in v3.canonical_json()


def test_v3_spec_refuses_wrong_or_missing_template_pin():
    data = {"suites": _suites_for_spec(), "protocol_version": "v3"}
    with pytest.raises(ParentSuiteValidationError):
        ParentEvalSpec.from_dict({**data, "canonical_rendering": True})
    with pytest.raises(ParentSuiteValidationError):
        ParentEvalSpec.from_dict(
            {
                **data,
                "canonical_rendering": True,
                "canonical_template_sha256": "0" * 64,
            }
        )
    with pytest.raises(ParentSuiteValidationError):
        # canonical_rendering under v2 is meaningless — refuse.
        ParentEvalSpec.from_dict(
            {
                "suites": _suites_for_spec(),
                "protocol_version": "v2",
                "canonical_rendering": True,
                "canonical_template_sha256": canonical_template_sha256(),
            }
        )
    with pytest.raises(ParentSuiteValidationError):
        # pin without the flag is meaningless — refuse.
        ParentEvalSpec.from_dict(
            {"suites": _suites_for_spec(), "canonical_template_sha256": canonical_template_sha256()}
        )


# ---------------------------------------------------------------------------
# build_tournament_spec: v2 vs v3
# ---------------------------------------------------------------------------


def test_build_tournament_spec_v3_pins_canonical_rendering(tmp_path):
    from chowder.parent_suite_content import materialize_protected_suites

    root = tmp_path / "protected"
    materialize_protected_suites(root)
    v2 = build_tournament_spec(root)
    v3 = build_tournament_spec(root, protocol_version="v3")
    assert v2.protocol_version == "v2" and v2.canonical_rendering is False
    assert v3.protocol_version == "v3"
    assert v3.canonical_rendering is True
    assert v3.canonical_template_sha256 == canonical_template_sha256()
    assert v2.digest() != v3.digest()
    with pytest.raises(Exception):
        build_tournament_spec(root, protocol_version="v9")


# ---------------------------------------------------------------------------
# v3 behavioral tokenizer gate (synthetic corpus, no model load)
# ---------------------------------------------------------------------------


class _FakeParent:
    def __init__(self, label: str, local_path: str):
        self.label = label
        self.revision = "0" * 64
        self.local_path = local_path


@pytest.fixture
def probe_corpus(tmp_path, monkeypatch):
    """A tiny synthetic corpus standing in for the pinned probe, plus a
    patched constant so the gate's hash check passes against it."""
    rows = [
        {"text": "alpha beta gamma delta epsilon zeta eta theta"},
        {"text": "the quick brown fox jumps over the lazy dog " * 20},
        {"text": "to be or not to be that is the question " * 20},
    ]
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8", newline="\n"
    )
    sha = hashlib.sha256(corpus.read_bytes()).hexdigest()
    monkeypatch.setattr(pt, "_V3_PROBE_CORPUS_SHA256", sha)
    monkeypatch.setattr(pt, "_V3_PROBE_CORPUS_PATH", str(corpus))
    monkeypatch.setattr(pt, "_V3_PROBE_PASSAGES", (0, 1, 2))
    return corpus


@pytest.fixture
def fake_auto_tokenizer(monkeypatch):
    """Fake AutoTokenizer returning per-parent token ID sequences.

    The gate resolves ``from transformers import AutoTokenizer`` lazily at
    call time, so the fake is injected as a ``sys.modules`` entry: CI runs
    without transformers installed and these tests must still exercise the
    gate everywhere (no importorskip, no skip).
    """
    import types

    state = {}

    class FakeAuto:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert kwargs.get("local_files_only") is True
            return state[str(path)]

    fake_mod = types.ModuleType("transformers")
    fake_mod.AutoTokenizer = FakeAuto
    monkeypatch.setitem(sys.modules, "transformers", fake_mod)
    return state


def _tok_with_ids(ids):
    class T:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": ids}

    return T()


def test_v3_gate_passes_when_behavior_matches(probe_corpus, fake_auto_tokenizer, tmp_path):
    # One flat ID list per passage (the gate encodes passages one at a
    # time); long enough to clear the probe's coverage floor.
    ids = [1, 2, 3] * 400
    a = tmp_path / "a"
    b = tmp_path / "b"
    fake_auto_tokenizer[str(a)] = _tok_with_ids(ids)
    fake_auto_tokenizer[str(b)] = _tok_with_ids(ids)
    ref = _FakeParent("parent-a", str(a))
    cand = _FakeParent("parent-b", str(b))
    evidence = pt.ensure_parent_tokenizer_behavior_compatible(ref, cand)
    assert evidence["identical"] is True


def test_v3_gate_fails_closed_on_behavior_divergence(probe_corpus, fake_auto_tokenizer, tmp_path):
    ids_a = [1, 2, 3] * 400
    ids_b = [1, 2, 99] * 400  # same length, different IDs at position 2
    a = tmp_path / "a"
    b = tmp_path / "b"
    fake_auto_tokenizer[str(a)] = _tok_with_ids(ids_a)
    fake_auto_tokenizer[str(b)] = _tok_with_ids(ids_b)
    ref = _FakeParent("parent-a", str(a))
    cand = _FakeParent("parent-b", str(b))
    with pytest.raises(pt.ParentTokenizerMismatch):
        pt.ensure_parent_tokenizer_behavior_compatible(ref, cand)


def test_v3_gate_refuses_corrupted_probe_corpus(probe_corpus, tmp_path):
    # Corrupt the corpus after the fixture patched the constant: the gate
    # must refuse on the hash mismatch before touching any tokenizer.
    probe_corpus.write_text("tampered\n", encoding="utf-8")
    ref = _FakeParent("parent-a", str(tmp_path / "a"))
    cand = _FakeParent("parent-b", str(tmp_path / "b"))
    with pytest.raises(pt.ParentTournamentError, match="probe corpus hash mismatch"):
        pt.ensure_parent_tokenizer_behavior_compatible(ref, cand)
