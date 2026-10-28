"""CPU-only adversarial tests for original LCB transport recovery inventory."""

import hashlib
import json

import pytest

from chowder_batch import reap_v2_lcb_v6_recovery_manifest_20261028 as inventory


SOURCE = [("a", "test.jsonl"), ("b", "test.jsonl"),
          ("c", "test2.jsonl"), ("d", "test2.jsonl")]
SHA = "a" * 64


def row(qid, filename, *, error=None, code="print(1)"):
    return {
        "question_id": qid, "source_file": filename,
        "task": "LCB_generation", "prompt_style": "OpenAIChat",
        "max_tokens": 4096, "temperature": 0.2, "top_p": 0.95,
        "code": "" if error else code, "raw": "" if error else code,
        "finish_reason": None if error else "stop", "error": error,
    }


def test_partition_never_conflates_missing_with_incorrect():
    report = inventory.build_inventory(SOURCE, [row(*SOURCE[0]),
        row(*SOURCE[1], error="URLError: [WinError 10061] actively refused")],
        journal_sha256=SHA)
    assert [x["question_id"] for x in report["responded_original_session_unattested"]] == ["a"]
    assert [x["question_id"] for x in report["failed_transport_not_model_answers"]] == ["b"]
    assert [x["question_id"] for x in report["never_attempted"]] == ["c", "d"]
    assert report["unresolved_count"] == 3
    assert report["official_score"] is None
    assert report["cross_session_pooling_allowed"] is False
    assert report["frozen_gen0"]["release_approved"] is False


def test_later_success_is_preserved_separately():
    report = inventory.build_inventory(SOURCE, [row(*SOURCE[0]),
        row(*SOURCE[1], error="timeout"), row(*SOURCE[2])], journal_sha256=SHA)
    assert [x["index"] for x in report["responded_original_session_unattested"]] == [0, 2]
    assert [x["index"] for x in report["failed_transport_not_model_answers"]] == [1]
    assert report["unresolved_count"] == 2


def test_empty_extracted_code_is_still_model_response():
    report = inventory.build_inventory(SOURCE, [row(*SOURCE[0], code="")],
                                       journal_sha256=SHA)
    assert len(report["responded_original_session_unattested"]) == 1
    assert not report["failed_transport_not_model_answers"]


@pytest.mark.parametrize("source,rows,digest", [
    (SOURCE, [row("wrong", "test.jsonl")], SHA),
    (SOURCE, [row(*SOURCE[0], error="timeout"), row(*SOURCE[0])], SHA),
    (SOURCE, [row(*SOURCE[0])], "bad"),
    ([SOURCE[0], SOURCE[0]], [row(*SOURCE[0])], SHA),
    ([], [], SHA),
    (SOURCE, [row(*SOURCE[0])] * 5, SHA),
])
def test_rejects_bad_source_protocol_or_sha(source, rows, digest):
    with pytest.raises(ValueError):
        inventory.build_inventory(source, rows, journal_sha256=digest)


def test_inspect_refuses_writer_race_without_creating_recovery_artifact(
    tmp_path, monkeypatch,
):
    output = tmp_path / "gen_q4.jsonl"
    raw = (json.dumps(row(*SOURCE[0])) + "\n").encode()
    output.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(inventory.lcb, "OUT", tmp_path)
    monkeypatch.setattr(inventory.lcb, "FILES", ("test.jsonl",))
    monkeypatch.setattr(inventory.lcb, "read_jsonl", lambda _: [
        {"question_id": SOURCE[0][0]},
    ])
    monkeypatch.setattr(inventory.transport, "inspect", lambda: {
        "generation_journal_sha256": "0" * 64,
    })
    with pytest.raises(ValueError, match="changed between"):
        inventory.inspect()
    assert output.read_bytes() == raw
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    assert list(tmp_path.iterdir()) == [output]


def test_inspect_refuses_second_hash_drift(tmp_path, monkeypatch):
    output = tmp_path / "gen_q4.jsonl"
    raw = (json.dumps(row(*SOURCE[0])) + "\n").encode()
    output.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(inventory.lcb, "OUT", tmp_path)
    monkeypatch.setattr(inventory.lcb, "FILES", ("test.jsonl",))
    monkeypatch.setattr(inventory.lcb, "read_jsonl", lambda _: [
        {"question_id": SOURCE[0][0]},
    ])
    monkeypatch.setattr(inventory.lcb, "sha", lambda _: "f" * 64)
    monkeypatch.setattr(inventory.transport, "inspect", lambda: {
        "generation_journal_sha256": digest,
        "unresolved_source_cases": 0,
        "journal_rows": 1,
    })
    with pytest.raises(ValueError, match="changed while"):
        inventory.inspect()
