"""Adversarial, CPU-only tests: no candidate code or model server is used."""

import json

import pytest

from chowder_batch import reap_v2_lcb_v6_transport_integrity_20261028 as audit


SOURCE = [("a", "test.jsonl"), ("b", "test.jsonl"), ("c", "test2.jsonl")]


def row(qid, filename, *, error=None, code="print(1)"):
    return {
        "question_id": qid, "source_file": filename,
        "task": "LCB_generation", "prompt_style": "OpenAIChat",
        "max_tokens": 4096, "temperature": 0.2, "top_p": 0.95,
        "code": "" if error else code, "raw": "" if error else code,
        "finish_reason": None if error else "stop", "error": error,
    }


def test_transport_errors_are_missing_not_incorrect():
    rows = [row(*SOURCE[0]), row(*SOURCE[1], error="URLError: [WinError 10061] actively refused")]
    result = audit.classify(SOURCE, rows)
    assert result["successful_transport_rows"] == 1
    assert result["unresolved_source_cases"] == 2
    assert result["failed_transport_rows_not_scored"] == 1
    assert result["first_failed_question_id"] == "b"
    assert result["official_complete_score"] is None
    assert result["release_approved"] is False


def test_later_success_is_not_hidden():
    rows = [row(*SOURCE[0]), row(*SOURCE[1], error="timeout"), row(*SOURCE[2])]
    result = audit.classify(SOURCE, rows)
    assert result["contiguous_successful_prefix"] == 1
    assert result["successes_after_first_failure"] == 1
    assert result["unresolved_source_cases"] == 1


def test_empty_genuine_model_response_is_not_transport_failure():
    result = audit.classify(SOURCE, [row(*SOURCE[0], code="")])
    assert result["failed_transport_rows_not_scored"] == 0
    assert result["model_responses_with_empty_code"] == 1


@pytest.mark.parametrize("bad", [
    lambda r: r.update(question_id="wrong"),
    lambda r: r.update(source_file="test6.jsonl"),
    lambda r: r.update(max_tokens=1024),
    lambda r: r.update(temperature=0.7),
    lambda r: r.update(prompt_style="other"),
    lambda r: r.update(error=123),
    lambda r: r.update(raw=42),
    lambda r: r.update(code=None),
])
def test_rejects_source_and_protocol_drift(bad):
    record = row(*SOURCE[0])
    bad(record)
    with pytest.raises(ValueError):
        audit.classify(SOURCE, [record])


def test_rejects_error_with_candidate_text():
    record = row(*SOURCE[0], error="timeout")
    record["code"] = "print('fabricated')"
    with pytest.raises(ValueError, match="candidate text"):
        audit.classify(SOURCE, [record])


def test_rejects_torn_journal():
    with pytest.raises(ValueError, match="partial"):
        audit.parse_complete_journal(b'{"question_id":"a"')


def test_rejects_blank_or_malformed_journal():
    with pytest.raises(ValueError):
        audit.parse_complete_journal(b"\n")
    with pytest.raises(ValueError):
        audit.parse_complete_journal(b"{oops}\n")


def test_valid_journal_parsing():
    record = row(*SOURCE[0])
    assert audit.parse_complete_journal((json.dumps(record) + "\n").encode()) == [record]


@pytest.mark.parametrize("malformed", [
    b'{"question_id":"wrong","question_id":"a"}\n',
    b'{"question_id":"a","usage":{"tokens":1,"tokens":2}}\n',
    b'{"question_id":"a","elapsed_s":NaN}\n',
    b'{"question_id":"a","elapsed_s":Infinity}\n',
    b'{"question_id":"a","elapsed_s":-Infinity}\n',
    b'{"question_id":"\xff"}\n',
])
def test_rejects_ambiguous_or_nonstandard_json_receipts(malformed):
    with pytest.raises(ValueError):
        audit.parse_complete_journal(malformed)


def test_valid_unicode_and_nested_unique_receipt():
    record = row(*SOURCE[0])
    record["raw"] = "é\u2603"
    record["usage"] = {"prompt": 1, "completion": 2}
    raw = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    assert audit.parse_complete_journal(raw) == [record]


def test_rejects_duplicate_or_out_of_order_source_id():
    with pytest.raises(ValueError, match="canonical"):
        audit.classify(SOURCE, [row(*SOURCE[0]), row(*SOURCE[0])])


def test_rejects_more_than_source_population():
    with pytest.raises(ValueError, match="excessive"):
        audit.classify(SOURCE, [row(*SOURCE[0])] * 4)
