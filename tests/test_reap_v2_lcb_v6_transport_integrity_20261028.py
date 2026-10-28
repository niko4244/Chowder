"""Adversarial, CPU-only tests: no candidate code or model server is used."""

import json
import hashlib
from pathlib import Path

import pytest

from chowder_batch import reap_v2_lcb_v6_transport_integrity_20261028 as audit


SOURCE = [("a", "test.jsonl"), ("b", "test.jsonl"), ("c", "test2.jsonl")]


def row(qid, filename, *, error=None, code="print(1)"):
    result = {
        "question_id": qid, "source_file": filename,
        "task": "LCB_generation", "prompt_style": "OpenAIChat",
        "max_tokens": 4096, "temperature": 0.2, "top_p": 0.95,
        "code": "" if error else code, "raw": "" if error else code,
        "error": error, "elapsed_s": 1.0,
    }
    if error is None:
        result.update(finish_reason="stop", prompt_tokens=10, completion_tokens=20)
    return result


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
    lambda r: r.update(max_tokens=4096.0),
    lambda r: r.update(max_tokens=True),
    lambda r: r.update(temperature=0.7),
    lambda r: r.update(temperature="0.2"),
    lambda r: r.update(top_p="0.95"),
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


def test_success_cannot_claim_extracted_code_from_empty_model_content():
    record = row(*SOURCE[0])
    record["raw"] = ""
    with pytest.raises(ValueError, match="without model response"):
        audit.classify(SOURCE, [record])


@pytest.mark.parametrize("key", [
    "error", "finish_reason", "prompt_tokens", "completion_tokens", "elapsed_s",
])
def test_missing_mandatory_success_field_never_counts_as_model_response(key):
    record = row(*SOURCE[0])
    del record[key]
    with pytest.raises(ValueError):
        audit.classify(SOURCE, [record])


@pytest.mark.parametrize("key", ["error", "elapsed_s"])
def test_missing_mandatory_failure_field_never_counts_as_response(key):
    record = row(*SOURCE[0], error="timeout")
    del record[key]
    with pytest.raises(ValueError):
        audit.classify(SOURCE, [record])


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(prompt_tokens=True),
    lambda r: r.update(completion_tokens=-1),
    lambda r: r.update(elapsed_s=float("nan")),
    lambda r: r.update(elapsed_s=-1.0),
    lambda r: r.update(elapsed_s="1.0"),
    lambda r: r.update(unexpected_field="unattested"),
])
def test_nonoriginal_success_field_values_or_extras_refused(mutation):
    record = row(*SOURCE[0])
    mutation(record)
    with pytest.raises(ValueError):
        audit.classify(SOURCE, [record])


def test_failure_cannot_claim_success_only_fields():
    record = row(*SOURCE[0], error="timeout")
    record.update(finish_reason="stop", prompt_tokens=10, completion_tokens=20)
    with pytest.raises(ValueError, match="field set mismatch"):
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


def test_atomic_exclusive_transport_snapshot_preserves_existing_certificate(
    tmp_path: Path,
):
    journal = tmp_path / "journal.jsonl"
    raw = (json.dumps(row(*SOURCE[0])) + "\n").encode()
    journal.write_bytes(raw)
    result = {"generation_journal_sha256": hashlib.sha256(raw).hexdigest(),
              "official_complete_score": None}
    target = tmp_path / "snapshot.json"
    audit._seal_exclusive_snapshot(target, journal, result)
    assert json.loads(target.read_text()) == result
    assert not list(tmp_path.glob("*.pending-*"))
    with pytest.raises(FileExistsError):
        audit._seal_exclusive_snapshot(target, journal, result)
    assert json.loads(target.read_text()) == result


def test_atomic_snapshot_refuses_mutated_journal_without_publication(
    tmp_path: Path,
):
    journal = tmp_path / "journal.jsonl"
    original = (json.dumps(row(*SOURCE[0])) + "\n").encode()
    journal.write_bytes(original)
    result = {"generation_journal_sha256": hashlib.sha256(original).hexdigest()}
    journal.write_bytes(original + b"additional unverified record\n")
    target = tmp_path / "unsealed.json"
    with pytest.raises(ValueError, match="changed before evidence seal"):
        audit._seal_exclusive_snapshot(target, journal, result)
    assert not target.exists()


def test_atomic_snapshot_link_failure_leaves_no_torn_certificate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    journal = tmp_path / "journal.jsonl"
    raw = (json.dumps(row(*SOURCE[0])) + "\n").encode()
    journal.write_bytes(raw)
    result = {"generation_journal_sha256": hashlib.sha256(raw).hexdigest()}

    def reject_link(_source: Path, _target: Path) -> None:
        raise OSError("simulated disk full before atomic publication")

    monkeypatch.setattr(audit.os, "link", reject_link)
    target = tmp_path / "unsealed.json"
    with pytest.raises(OSError, match="disk full"):
        audit._seal_exclusive_snapshot(target, journal, result)
    assert not target.exists()
    assert not list(tmp_path.glob("*.pending-*"))
