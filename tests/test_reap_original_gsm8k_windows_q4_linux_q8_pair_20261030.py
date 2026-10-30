"""CPU-only adversarial research pairing tests; no F: writes or inference."""

from __future__ import annotations

import hashlib
import json

import pytest

from chowder_batch import reap_original_gsm8k_frozen_scorer_20261029 as scorer
from chowder_batch import reap_original_gsm8k_independent_grade_20261029 as grade
from chowder_batch import reap_original_gsm8k_windows_q4_linux_q8_pair_20261030 as pair


@pytest.fixture
def synthetic():
    canonical = grade.load_canonical()
    q4 = {task_id: {"id": task_id, "text": expected, "correct": True,
                    "terminated": True} for task_id, _, expected in canonical}
    q8 = {key: dict(value) for key, value in q4.items()}
    for i in range(5):
        q8[f"gsm8k-{i}"].update({"text": "no answer", "correct": False})
    return q4, q8


def test_independent_paired_counts_and_exact_binomial_tail(synthetic):
    result = pair.paired_certificate(*synthetic)
    assert result["paired_counts"] == {
        "both_correct": 1314, "windows_q4_only": 5,
        "linux_q8_only": 0, "neither_correct": 0,
    }
    assert (result["q4_correct"], result["q8_correct"]) == (1319, 1314)
    assert (result["exact_two_sided_mcnemar_p_numerator"],
            result["exact_two_sided_mcnemar_p_denominator"]) == (1, 16)
    assert result["status"] == "BLOCKED" and result["chosen"] is None
    assert result["release_approved"] is False
    assert result["research_only"] is True
    assert result["native_http_lineage_authenticated"] is False
    assert result["same_host_frozen_comparison"] is False
    assert "q4_rows_sha256" not in result and "q8_rows_sha256" not in result


@pytest.mark.parametrize("forgery", [
    "wrong_correct", "wrong_terminated", "missing_id", "noncanonical_id",
])
def test_forged_per_item_outcome_or_population_refused(synthetic, forgery):
    q4, q8 = synthetic
    if forgery == "wrong_correct":
        q4["gsm8k-0"]["correct"] = False
    elif forgery == "wrong_terminated":
        q4["gsm8k-0"]["terminated"] = False
    elif forgery == "missing_id":
        del q4["gsm8k-0"]
    else:
        q4["fake-0"] = q4.pop("gsm8k-0")
    with pytest.raises(grade.IndependentGradeError):
        pair.paired_certificate(q4, q8)


def test_ambient_scorer_monkeypatch_cannot_change_research_pair(synthetic, monkeypatch):
    import chowder.evaluators.scoring as ambient

    monkeypatch.setattr(ambient, "score", lambda *_args: 1.0)
    assert pair.paired_certificate(*synthetic)["q8_correct"] == 1314


def test_original_scorer_source_drift_refused(synthetic, monkeypatch, tmp_path):
    fake = tmp_path / "scoring.py"
    fake.write_text("def score(*args): return 1.0\n", encoding="utf-8")
    monkeypatch.setattr(scorer, "_ORIGINAL_SCORER_PATH", fake)
    with pytest.raises(scorer.OriginalScorerSourceError, match="drift"):
        pair.paired_certificate(*synthetic)


def _write_rows(path, rows):
    path.write_bytes(b"".join(json.dumps(row, ensure_ascii=False).encode() + b"\n"
                              for row in rows))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_hash_drift_refused_before_any_grading(synthetic, tmp_path):
    path = tmp_path / "original.jsonl"
    _write_rows(path, list(synthetic[0].values()))
    with pytest.raises(pair.OriginalPairedEvidenceError, match="changed"):
        pair.load_pinned_rows(path, "0" * 64)


def test_test_only_pinned_source_requires_exact_canonical_inventory(synthetic, tmp_path):
    path = tmp_path / "original.jsonl"
    digest = _write_rows(path, list(synthetic[0].values()))
    assert len(pair.load_pinned_rows(path, digest)) == 1319
    modified = list(synthetic[0].values())
    modified[0] = dict(modified[0], id="gsm8k-1")
    digest = _write_rows(path, modified)
    with pytest.raises(pair.OriginalPairedEvidenceError, match="duplicated"):
        pair.load_pinned_rows(path, digest)


@pytest.mark.parametrize("bad_line", [
    b'{"id":"gsm8k-0","id":"gsm8k-0","correct":true}\n',
    b'{"id":"gsm8k-0","correct":NaN}\n',
])
def test_test_only_pinned_source_rejects_ambiguous_json(synthetic, tmp_path, bad_line):
    path = tmp_path / "original.jsonl"
    rows = list(synthetic[0].values())
    path.write_bytes(bad_line + b"".join(json.dumps(row).encode() + b"\n"
                                           for row in rows[1:]))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(pair.OriginalPairedEvidenceError):
        pair.load_pinned_rows(path, digest)
