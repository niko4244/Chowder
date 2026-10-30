"""Adversarial CPU-only tests; never starts native models or edits F: evidence."""

import hashlib
import json
from pathlib import Path

import pytest

from chowder_batch import reap_original_gsm8k_independent_grade_20261029 as grade
from chowder_batch import reap_original_linux_gsm8k_independent_pair_20261029 as pair


@pytest.fixture(scope="module")
def canonical():
    return grade.load_canonical()


@pytest.fixture(scope="module")
def synthetic_rows(canonical):
    session = "source-bound-session"
    rows = []
    for ident, _, expected in canonical:
        rows.append({"id": ident, "correct": True, "terminated": True,
                     "trigram_ratio": 1.0, "completion_tokens": 4,
                     "seconds": 1.0, "predicted_per_second": 4.0,
                     "text": f"Final answer {expected}", "eval_session_id": session})
    return rows


def _raw(rows):
    return ("\n".join(json.dumps(row, allow_nan=False) for row in rows) + "\n").encode()


def test_synthetic_complete_regrades_independently(canonical, synthetic_rows):
    result = pair.regrade_rows(_raw(synthetic_rows), canonical, "source-bound-session")
    assert (result["n"], result["correct"], result["naturally_terminated"],
            result["tokens"]) == (1319, 1319, 1319, 5276)
    assert result["release_approved"] is False
    assert result["native_request_and_response_lineage_authenticated"] is False


def test_rejects_valid_first200_prefix_in_full_linux_pair(canonical, synthetic_rows):
    """A canonical 200-prefix is valid for a different audit, not this one."""
    with pytest.raises(pair.IndependentPairError, match="full 1319"):
        pair.regrade_rows(_raw(synthetic_rows[:200]), canonical, "source-bound-session")


@pytest.mark.parametrize("mutation", [
    "wrong_session", "wrong_correct", "wrong_trigram", "wrong_id",
    "wrong_text", "wrong_termination", "bad_tokens", "bad_seconds",
    "extra_key", "missing_row", "duplicate_id", "missing_newline",
])
def test_rejects_adversarial_rows(canonical, synthetic_rows, mutation):
    rows = [row.copy() for row in synthetic_rows]
    first = rows[0]
    if mutation == "wrong_session":
        first["eval_session_id"] = "other-model-session"
    elif mutation == "wrong_correct":
        first["correct"] = False
    elif mutation == "wrong_trigram":
        first["trigram_ratio"] = 0.1
    elif mutation == "wrong_id":
        first["id"] = "gsm8k-99999"
    elif mutation == "wrong_text":
        first["text"] = "No numeric answer"
    elif mutation == "wrong_termination":
        first["terminated"] = False
    elif mutation == "bad_tokens":
        first["completion_tokens"] = True
    elif mutation == "bad_seconds":
        first["seconds"] = -1
    elif mutation == "extra_key":
        first["native_forgery"] = "not allowed"
    elif mutation == "missing_row":
        rows.pop()
    elif mutation == "duplicate_id":
        rows[-1]["id"] = first["id"]
    raw = _raw(rows)
    if mutation == "missing_newline":
        raw = raw.rstrip(b"\n")
    with pytest.raises(pair.IndependentPairError):
        pair.regrade_rows(raw, canonical, "source-bound-session")


def test_rejects_duplicate_json_object_key(canonical, synthetic_rows):
    raw = _raw(synthetic_rows).replace(b'"correct": true,',
                                       b'"correct": false, "correct": true,', 1)
    with pytest.raises(pair.IndependentPairError, match="JSON"):
        pair.regrade_rows(raw, canonical, "source-bound-session")


def test_rejects_nonfinite_json_number(canonical, synthetic_rows):
    raw = _raw(synthetic_rows).replace(b'"seconds": 1.0', b'"seconds": NaN', 1)
    with pytest.raises(pair.IndependentPairError, match="JSON"):
        pair.regrade_rows(raw, canonical, "source-bound-session")


def test_rejects_forged_repetition_metric(canonical, synthetic_rows):
    rows = [row.copy() for row in synthetic_rows]
    rows[0]["text"] = "x x x x x x x x x x Final answer " + canonical[0][2]
    with pytest.raises(pair.IndependentPairError, match="trigram"):
        pair.regrade_rows(_raw(rows), canonical, "source-bound-session")


def test_mutable_production_scorer_cannot_approve_forged_answer(
        canonical, synthetic_rows, monkeypatch):
    """The new archived regrader must not inherit the dirty bridge's scorer risk."""
    from chowder.evaluators import scoring

    monkeypatch.setattr(scoring, "score", lambda *_args, **_kwargs: 1.0)
    rows = [row.copy() for row in synthetic_rows]
    rows[0]["text"] = "No numeric answer"
    rows[0]["correct"] = True
    with pytest.raises(pair.IndependentPairError, match="canonical answer"):
        pair.regrade_rows(_raw(rows), canonical, "source-bound-session")


def test_pinned_reader_rejects_changed_bytes(tmp_path):
    path = tmp_path / "source.jsonl"
    path.write_bytes(b"source\n")
    sha = hashlib.sha256(b"source\n").hexdigest()
    assert pair._read_pinned(path, sha, 100) == b"source\n"
    path.write_bytes(b"forged\n")
    with pytest.raises(pair.IndependentPairError, match="SHA"):
        pair._read_pinned(path, sha, 100)


def test_pinned_reader_rejects_alias_and_symlink(tmp_path):
    path = tmp_path / "source.jsonl"
    path.write_bytes(b"source\n")
    sha = hashlib.sha256(b"source\n").hexdigest()
    hardlink = tmp_path / "hardlink.jsonl"
    try:
        hardlink.hardlink_to(path)
    except (OSError, NotImplementedError):
        pytest.skip("hardlinks not supported")
    with pytest.raises(pair.IndependentPairError, match="unsafe"):
        pair._read_pinned(path, sha, 100)
    hardlink.unlink()
    symlink = tmp_path / "symlink.jsonl"
    try:
        symlink.symlink_to(path)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported")
    with pytest.raises(pair.IndependentPairError, match="unsafe"):
        pair._read_pinned(symlink, sha, 100)


def test_live_original_linux_archives_if_mounted():
    if not all(Path(spec["rows"]).is_file() for spec in pair.ARMS.values()):
        pytest.skip("private original Linux archives not mounted")
    report = pair.verify_original_linux_pair()
    assert (report["q4_correct"], report["q8_correct"]) == (1229, 1220)
    assert report["paired"] == {"both_correct": 1198, "both_wrong": 68,
                                "candidate_only_correct": 22,
                                "reference_only_correct": 31}
    assert report["research_only"] is True
    assert report["frozen_windows_gate1_qualified"] is False
    assert report["frozen_dense_gate2_qualified"] is False
    assert report["status"] == "BLOCKED"
    assert report["chosen"] is None
    assert report["release_approved"] is False
