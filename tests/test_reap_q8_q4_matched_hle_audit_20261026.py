"""Small deterministic RED/GREEN unit tests for paired, non-official HLE audit."""

import copy

import pytest

from chowder_batch import reap_q8_q4_matched_hle_audit_20261026 as audit


def _row(id, correct, session, kind="exactMatch"):
    return {"id": id, "correct": correct, "graded": True, "truncated": False,
            "completion_tokens": 17, "answer_type": kind, "eval_session_id": session}


def test_true_item_id_matched_paired_confusion_and_difference():
    q4 = [_row("a", True, "q4"), _row("b", False, "q4"),
          _row("c", True, "q4", "multipleChoice"), _row("d", False, "q4")]
    q8 = [_row("d", False, "q8"), _row("b", True, "q8"),
          _row("a", True, "q8"), _row("c", False, "q8", "multipleChoice")]
    result = audit.compare_rows(q4, q8, expected_ids={"a", "b", "c", "d"})
    assert result["population"] == 4
    assert result["q4_correct"] == result["q8_correct"] == 2
    assert result["partition"] == {"both_correct": 1, "q8_only_correct": 1,
                                    "q4_only_correct": 1, "both_incorrect": 1}
    assert result["q8_minus_q4_percentage_points"] == 0
    assert result["official_hle_judge_score"] == "UNAVAILABLE"
    assert result["gen0_status"] == "BLOCKED"
    assert result["release_approved"] is False


@pytest.mark.parametrize("fault", ["missing", "duplicate", "not_graded", "truncated",
                                    "different_type", "different_session", "invalid_bool",
                                    "invalid_tokens"])
def test_bad_pair_refuses_to_create_accuracy(fault):
    q4 = [_row("a", True, "q4"), _row("b", False, "q4")]
    q8 = [_row("a", False, "q8"), _row("b", True, "q8")]
    if fault == "missing":
        q8.pop()
    elif fault == "duplicate":
        q8[1]["id"] = "a"
    elif fault == "not_graded":
        q8[1]["graded"] = False
    elif fault == "truncated":
        q8[1]["truncated"] = True
    elif fault == "different_type":
        q8[0]["answer_type"] = "multipleChoice"
    elif fault == "different_session":
        q8[1]["eval_session_id"] = "another-q8"
    elif fault == "invalid_bool":
        q8[0]["correct"] = 0
    elif fault == "invalid_tokens":
        q8[0]["completion_tokens"] = "unknown"
    with pytest.raises(ValueError):
        audit.compare_rows(copy.deepcopy(q4), q8, expected_ids={"a", "b"})


def test_no_q8_output_yet_is_a_refusal(tmp_path):
    with pytest.raises(ValueError):
        audit.audit(tmp_path / "not_yet")


def test_live_q4_file_alone_cannot_become_false_paired_pass(tmp_path):
    with pytest.raises(ValueError, match="missing or oversized"):
        audit.audit(tmp_path, q4_dir=tmp_path)
