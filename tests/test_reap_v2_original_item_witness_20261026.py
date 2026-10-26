"""Adversarial CPU-only private row integrity tests: no generated-code execution."""

import json
from pathlib import Path

import pytest

from chowder_batch import reap_v2_original_item_witness_20261026 as witness


def _row(i: int, *, correct: bool = True, session: str = "locked-session") -> dict:
    return {"id": f"gsm8k-{i}", "correct": correct, "terminated": True,
            "trigram_ratio": 0.75, "completion_tokens": 10,
            "seconds": 1.25, "predicted_per_second": 8.0,
            "text": "PRIVATE MODEL RESPONSE THAT MUST NOT APPEAR IN WITNESS", "eval_session_id": session}


def _data(rows: list[dict]) -> bytes:
    return b"".join((json.dumps(row, allow_nan=True) + "\n").encode() for row in rows)


def _check(rows: list[dict], monkeypatch, *, digest: str | None = None):
    monkeypatch.setitem(witness.SIZES, "gsm8k", 2)
    # Test the exact-id contract with the normal 1319 GSM8K ID set?  Instead
    # exercise malformed-row checks first, before the full ID set check.
    raw = _data(rows)
    return witness.reverify_rows(raw, digest=digest or witness.file_sha(raw),
                                 session="locked-session", suite="gsm8k")


def test_changed_source_bytes_refused(monkeypatch):
    with pytest.raises(ValueError, match="hash"):
        _check([_row(0), _row(1)], monkeypatch, digest="0" * 64)


def test_wrong_session_refused(monkeypatch):
    with pytest.raises(ValueError, match="session"):
        _check([_row(0), _row(1, session="other")], monkeypatch)


def test_integer_correctness_not_accepted_as_boolean(monkeypatch):
    row = _row(1)
    row["correct"] = 1
    with pytest.raises(ValueError, match="booleans"):
        _check([_row(0), row], monkeypatch)


def test_duplicate_original_id_refused(monkeypatch):
    with pytest.raises(ValueError, match="unique"):
        _check([_row(0), _row(0)], monkeypatch)


def test_nan_and_infinite_fields_refused(monkeypatch):
    for value in (float("nan"), float("inf")):
        row = _row(1)
        row["seconds"] = value
        with pytest.raises(ValueError, match="nonfinite"):
            _check([_row(0), row], monkeypatch)


def test_extra_unregistered_field_refused(monkeypatch):
    row = _row(1)
    row["verified"] = True
    with pytest.raises(ValueError, match="fields"):
        _check([_row(0), row], monkeypatch)


def test_missing_terminal_newline_refused(monkeypatch):
    monkeypatch.setitem(witness.SIZES, "gsm8k", 2)
    raw = _data([_row(0), _row(1)]).rstrip(b"\n")
    with pytest.raises(ValueError, match="newline"):
        witness.reverify_rows(raw, digest=witness.file_sha(raw), session="locked-session", suite="gsm8k")


def test_private_result_exclusive_folder(tmp_path):
    with pytest.raises(ValueError, match="exclusive"):
        witness.run(tmp_path / "not-f-private")


def test_matched_paired_outcomes():
    left = [{"id": "item-1", "correct": True}, {"id": "item-2", "correct": False},
            {"id": "item-3", "correct": True}, {"id": "item-4", "correct": False}]
    right = [{"id": "item-1", "correct": True}, {"id": "item-2", "correct": True},
             {"id": "item-3", "correct": False}, {"id": "item-4", "correct": False}]
    assert witness.paired(left, right) == {"both_correct": 1, "both_wrong": 1,
                                          "q8_only": 1, "q4_only": 1}


def test_mismatched_paired_id_population_refused():
    with pytest.raises(ValueError, match="matched"):
        witness.paired([{"id": "x", "correct": True}], [{"id": "y", "correct": True}])


def test_private_output_contains_only_item_bits():
    p = Path("F:/reap-v2-automationbench-private-20261026/original-q8-q4-recount-wave-20261026-v1")
    if not p.exists():
        pytest.skip("this live host has no original private source-bound receipt")
    csv = (p / "original-q8-q4-item-level.private.csv").read_text(encoding="utf-8")
    assert "PRIVATE MODEL RESPONSE" not in csv
    assert "text" not in csv.splitlines()[0]
    assert len(csv.splitlines()) == sum(witness.SIZES.values()) + 1
