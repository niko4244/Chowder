"""CPU-only adversarial tests for the source-bound original GSM8K scorer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chowder_batch import reap_original_gsm8k_frozen_scorer_20261029 as frozen


@pytest.mark.parametrize(
    ("text", "expected", "terminated", "correct"),
    [
        ("18", "18", True, True),
        ("There are 16 eggs and the answer is 18", "18", True, True),
        ("$70,000.0", "70000", True, True),
        ("<think>reason 18", "18", True, False),
        ("<think>wrong 12</think>final 18", "18", True, True),
        ("<think>wrong 18</think>final 12", "18", True, False),
        ("<think>18</think>final 18</think>", "18", True, False),
        ("18", "18", False, False),
        ("no answer", "18", True, False),
        ("-2", "-2.0", True, True),
        ("2.50", "2.5", True, False),
        ("1,234", "1234", True, True),
    ],
)
def test_original_semantics(text, expected, terminated, correct):
    assert frozen.score_original_gsm8k(
        text, expected, terminated=terminated,
    ) is correct


def test_source_bytes_are_pinned():
    assert frozen.require_original_scorer_source() == frozen.ORIGINAL_SCORER_SHA256


def test_modified_source_is_rejected(tmp_path):
    path = tmp_path / "scoring.py"
    path.write_text("def score(*args): return 1.0\n", encoding="utf-8")
    with pytest.raises(frozen.OriginalScorerSourceError, match="drift"):
        frozen.require_original_scorer_source(source_path=path)


def test_symlinked_source_is_rejected(tmp_path):
    link = tmp_path / "scoring.py"
    try:
        link.symlink_to(Path(frozen._ORIGINAL_SCORER_PATH))
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(frozen.OriginalScorerSourceError, match="missing"):
        frozen.require_original_scorer_source(source_path=link)


def test_imported_mutable_scorer_cannot_override(monkeypatch):
    import chowder.evaluators.scoring as ambient

    monkeypatch.setattr(ambient, "score", lambda *args: 1.0)
    assert frozen.score_original_gsm8k(
        "no answer", "18", terminated=True,
    ) is False


def test_original_source_digest_is_fixed():
    raw = frozen._ORIGINAL_SCORER_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == frozen.ORIGINAL_SCORER_SHA256


def test_all_1319_canonical_expected_answers_match_original_scorer():
    """CPU-only scorer parity on actual frozen answers, not generated outputs."""
    from chowder.evaluators.scoring import score

    path = (Path(__file__).resolve().parent.parent /
            "docs/quals/reap-v2-original-gsm8k-canonical-items.json")
    items = json.loads(path.read_text(encoding="utf-8"))
    encoded = json.dumps(items, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == (
        "913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89"
    )
    assert len(items) == 1319
    for i, item in enumerate(items):
        assert item["id"] == f"gsm8k-{i}"
        expected = item["expected"]
        for prediction in (
            expected,
            "<think>intermediate 999</think>" + expected,
            "<think>" + expected,
            "<think>" + expected + "</think>no answer",
            "no answer",
        ):
            for terminated in (True, False):
                original = bool(
                    terminated
                    and score(prediction, expected, "final_number_match") == 1.0
                )
                assert frozen.score_original_gsm8k(
                    prediction, expected, terminated=terminated,
                ) is original


@pytest.mark.parametrize("bad", [0, 1, None, "true"])
def test_non_boolean_termination_refused(bad):
    with pytest.raises(TypeError):
        frozen.score_original_gsm8k("18", "18", terminated=bad)


def test_nonstring_response_refused():
    with pytest.raises(TypeError):
        frozen.score_original_gsm8k(18, "18", terminated=True)
