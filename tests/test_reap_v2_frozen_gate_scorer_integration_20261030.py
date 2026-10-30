"""Adversarial CPU-only integration tests for original historical GSM8K grading.

These exercise the actual frozen-gate bridge helper, not a synthetic fixture
that bypasses the grader. No original F: evidence or native model is accessed.
"""

from __future__ import annotations

import json

import pytest

from chowder_batch import reap_original_gsm8k_frozen_scorer_20261029 as frozen
from chowder_batch import reap_original_gsm8k_independent_grade_20261029 as grade
from chowder_batch import reap_v2_frozen_gate_evidence_bridge_20261027 as bridge


def _rows(n: int) -> dict[str, dict]:
    canonical = grade.load_canonical()
    return {
        task_id: {
            "id": task_id,
            "text": expected,
            "correct": frozen.score_original_gsm8k(
                expected, expected, terminated=True,
            ),
            "terminated": True,
        }
        for task_id, _, expected in canonical[:n]
    }


@pytest.mark.parametrize("n", [200, 1319])
def test_actual_bridge_regrades_entire_canonical_population(tmp_path, n):
    bridge._verify_historical_gsm8k_correctness(tmp_path, _rows(n))


def test_ambient_mutable_scorer_cannot_forge_historical_pass(tmp_path, monkeypatch):
    import chowder.evaluators.scoring as ambient

    rows = _rows(200)
    rows["gsm8k-0"]["text"] = "no answer"
    rows["gsm8k-0"]["correct"] = True
    monkeypatch.setattr(ambient, "score", lambda *_args: 1.0)
    with pytest.raises(bridge.FrozenEvidenceError, match="gsm8k-0.*correctness"):
        bridge._verify_historical_gsm8k_correctness(tmp_path, rows)


def test_source_drift_blocks_even_valid_historical_rows(tmp_path, monkeypatch):
    rows = _rows(200)
    untrusted_source = tmp_path / "scoring.py"
    untrusted_source.write_text("def score(*args): return 1.0\n", encoding="utf-8")
    monkeypatch.setattr(frozen, "_ORIGINAL_SCORER_PATH", untrusted_source)
    with pytest.raises(frozen.OriginalScorerSourceError, match="drift"):
        bridge._verify_historical_gsm8k_correctness(tmp_path, rows)


@pytest.mark.parametrize("changed", [
    {"correct": True, "text": "no answer"},
    {"correct": True, "terminated": False},
    {"correct": 1},
    {"terminated": 1},
    {"text": None},
])
def test_forged_historical_fields_are_refused(tmp_path, changed):
    rows = _rows(200)
    rows["gsm8k-0"].update(changed)
    with pytest.raises(bridge.FrozenEvidenceError):
        bridge._verify_historical_gsm8k_correctness(tmp_path, rows)


def test_noncanonical_prefix_refused(tmp_path):
    rows = _rows(200)
    rows["gsm8k-200"] = rows.pop("gsm8k-199")
    with pytest.raises(bridge.FrozenEvidenceError, match="population"):
        bridge._verify_historical_gsm8k_correctness(tmp_path, rows)


def test_canonical_source_tampering_refused(tmp_path):
    rows = _rows(200)
    path = tmp_path / "canonical.json"
    items = json.loads(grade.CANONICAL_PATH.read_text(encoding="utf-8"))
    items[0]["expected"] = "999999"
    path.write_text(json.dumps(items), encoding="utf-8")
    with pytest.raises(bridge.FrozenEvidenceError, match="canonical source differs"):
        bridge._verify_historical_gsm8k_correctness(
            tmp_path, rows, canonical_path=path,
        )


def test_bridge_has_no_ambient_scorer_import():
    import inspect

    source = inspect.getsource(bridge._verify_historical_gsm8k_correctness)
    assert "from chowder.evaluators.scoring import score" not in source
    assert "frozen_gsm8k.score_original_gsm8k(" in source
