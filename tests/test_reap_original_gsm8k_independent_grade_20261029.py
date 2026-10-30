"""CPU-only source-lock and adversarial regrading tests; no model execution."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from chowder.evaluators.scoring import score as production_score
from chowder_batch import reap_original_gsm8k_independent_grade_20261029 as grade


@pytest.fixture(scope="module")
def canonical():
    return grade.load_canonical()


def _rows(canonical, n=200):
    return {
        item_id: {"id": item_id, "text": f"Final answer: {expected}",
                  "terminated": True, "correct": True}
        for item_id, _, expected in canonical[:n]
    }


def test_actual_original_canonical_population_and_digest(canonical):
    assert len(canonical) == 1319
    assert canonical[0][0] == "gsm8k-0"
    assert canonical[-1][0] == "gsm8k-1318"
    assert grade.FROZEN_GSM8K_SHA256 == (
        "913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89"
    )


def test_original_scorer_source_blob_immutably_identified():
    repo = Path(__file__).resolve().parent.parent
    env = {**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"}
    original = subprocess.check_output([
        "git", "show", grade.ORIGINAL_SCORER_GIT_COMMIT
        + ":src/chowder/evaluators/scoring.py",
    ], cwd=repo, env=env, timeout=20)
    assert hashlib.sha256(original).hexdigest() == grade.ORIGINAL_SCORER_GIT_SHA256


@pytest.mark.parametrize("prediction,expected", [
    ("The answer is 1,001", "1001"),
    ("<think>9</think>Final: -42.0", "-42"),
    ("<think>the answer is 7", "7"),
    ("<think>3</think>42</think>", "42"),
    ("No numeric answer", "9"),
    ("final answer 42", "42"),
    ("final answer 1,234.5", "1234.5"),
    ("final answer 3.00", "3.00"),
    ("final answer 3.00", "3"),
])
def test_independent_number_semantics_match_original_production(prediction, expected):
    assert grade.independent_number_match(prediction, expected) == (
        production_score(prediction, expected, "final_number_match") == 1.0
    )


def test_canonical_200_prefix_regrades_without_trusting_labels(canonical):
    rows = _rows(canonical)
    report = grade.verify_rows(rows, canonical)
    assert report["n"] == 200
    assert report["correct"] == 200
    assert report["naturally_terminated"] == 200
    assert report["research_only"] is True
    assert report["native_request_and_response_lineage_authenticated"] is False
    assert report["gen0_status"] == "BLOCKED"
    assert report["chosen"] is None
    assert report["release_approved"] is False


def test_complete_synthetic_population_not_an_official_score(canonical):
    rows = _rows(canonical, 1319)
    rows["gsm8k-5"]["terminated"] = False
    rows["gsm8k-5"]["correct"] = False
    report = grade.verify_rows(rows, canonical)
    assert (report["n"], report["correct"], report["naturally_terminated"]) == (
        1319, 1318, 1318
    )
    assert report["release_approved"] is False


@pytest.mark.parametrize("tamper", ["wrong_answer", "forged_failure", "wrong_id", "truncated"])
def test_reject_forged_or_incomplete_historical_rows(canonical, tamper):
    rows = _rows(canonical)
    if tamper == "wrong_answer":
        rows["gsm8k-0"]["text"] = "No answer"
    elif tamper == "forged_failure":
        rows["gsm8k-0"]["correct"] = False
    elif tamper == "wrong_id":
        rows["gsm8k-0"]["id"] = "gsm8k-1"
    else:
        rows["gsm8k-0"]["terminated"] = False
    with pytest.raises(grade.IndependentGradeError):
        grade.verify_rows(rows, canonical)


def test_cannot_reseal_altered_canonical_answers(canonical):
    altered = list(canonical)
    ident, prompt, _ = altered[0]
    altered[0] = (ident, prompt, "999999")
    with pytest.raises(grade.IndependentGradeError, match="source SHA"):
        grade.verify_rows(_rows(canonical), tuple(altered))


def test_source_copy_tampering_refused(tmp_path):
    source = grade.CANONICAL_PATH.read_bytes()
    path = tmp_path / "copy.json"
    path.write_bytes(source)
    assert grade.load_canonical(path)[-1][0] == "gsm8k-1318"
    doc = json.loads(source)
    doc[0]["expected"] = "999999"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(grade.IndependentGradeError, match="source SHA"):
        grade.load_canonical(path)


def test_duplicate_json_key_and_symlink_refused(tmp_path):
    raw = grade.CANONICAL_PATH.read_text(encoding="utf-8")
    path = tmp_path / "duplicate.json"
    path.write_text(raw.replace('"expected":', '"expected":"bad","expected":', 1),
                    encoding="utf-8")
    with pytest.raises(grade.IndependentGradeError, match="duplicate"):
        grade.load_canonical(path)
    link = tmp_path / "redirect.json"
    try:
        link.symlink_to(grade.CANONICAL_PATH)
    except (OSError, NotImplementedError):
        pytest.skip("filesystem does not permit test symlink")
    with pytest.raises(grade.IndependentGradeError, match="redirected"):
        grade.load_canonical(link)


def test_archived_original_windows_q4_1319_regrades_if_present(canonical):
    """Never reruns a model; regrades existing immutable F: source-bound rows."""
    path = Path(
        "F:/reap-v2-q4-windows-full1319-phase34-original-20261027-0152/"
        "reap128_v2_q4/gsm8k.jsonl"
    )
    if not path.is_file():
        pytest.skip("original archived F: Windows Q4 rows not mounted")
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "66d0f39545dd8769e837ed5772a265871cfe148bb9a529ea2188454d72c1243f"
    )
    rows = {}
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            item = json.loads(line, object_pairs_hook=grade._unique_pairs,
                              parse_constant=grade._reject_nonfinite)
            assert item["id"] not in rows
            rows[item["id"]] = item
    report = grade.verify_rows(rows, canonical)
    assert (report["n"], report["correct"], report["naturally_terminated"]) == (
        1319, 1224, 1281
    )
    assert report["release_approved"] is False
