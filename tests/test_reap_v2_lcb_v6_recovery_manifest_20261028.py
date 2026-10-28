"""CPU-only adversarial tests for original LCB transport recovery inventory."""

import hashlib
import json

import pytest

from chowder_batch import reap_v2_lcb_v6_recovery_manifest_20261028 as inventory


SOURCE = [("a", "test.jsonl"), ("b", "test.jsonl"),
          ("c", "test2.jsonl"), ("d", "test2.jsonl")]
SHA = "a" * 64


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


def test_missing_error_key_cannot_diverge_from_transport_witness():
    item = row(*SOURCE[0])
    del item["error"]
    with pytest.raises(ValueError, match="missing original transport status"):
        inventory.build_inventory(SOURCE, [item], journal_sha256=SHA)


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


def _seal_fixture(tmp_path):
    journal = tmp_path / "original.jsonl"
    raw = (json.dumps(row(*SOURCE[0])) + "\n").encode("utf-8")
    journal.write_bytes(raw)
    result = inventory.build_inventory(
        SOURCE, [row(*SOURCE[0])], journal_sha256=hashlib.sha256(raw).hexdigest(),
    )
    return journal, raw, result


def test_exclusive_recovery_seal_is_atomic_and_never_overwrites(tmp_path):
    journal, raw, result = _seal_fixture(tmp_path)
    target = tmp_path / "sealed.json"
    inventory._seal_exclusive_inventory(target, journal, result)
    assert json.loads(target.read_text(encoding="utf-8")) == result
    assert journal.read_bytes() == raw
    assert not list(tmp_path.glob(".*.pending-*"))
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        inventory._seal_exclusive_inventory(target, journal, result)
    assert target.read_bytes() == original
    assert not list(tmp_path.glob(".*.pending-*"))


def test_exclusive_recovery_seal_refuses_changed_journal_before_publish(
    tmp_path, monkeypatch,
):
    journal, raw, result = _seal_fixture(tmp_path)
    original_verify = inventory._verify_complete_journal
    calls = 0

    def change_after_first_verification(path, expected_sha256):
        nonlocal calls
        identity = original_verify(path, expected_sha256)
        calls += 1
        if calls == 1:
            journal.write_bytes(raw + b"unauthorized append")
        return identity

    monkeypatch.setattr(inventory, "_verify_complete_journal",
                        change_after_first_verification)
    target = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError, match="changed before evidence seal"):
        inventory._seal_exclusive_inventory(target, journal, result)
    assert calls == 1
    assert not target.exists()
    assert not list(tmp_path.glob(".*.pending-*"))


def test_exclusive_recovery_seal_refuses_publication_error_without_partial_file(
    tmp_path, monkeypatch,
):
    journal, raw, result = _seal_fixture(tmp_path)

    def fail_link(_source, _target):
        raise OSError("publication refused")

    monkeypatch.setattr(inventory.os, "link", fail_link)
    target = tmp_path / "must-not-exist.json"
    with pytest.raises(OSError, match="publication refused"):
        inventory._seal_exclusive_inventory(target, journal, result)
    assert journal.read_bytes() == raw
    assert not target.exists()
    assert not list(tmp_path.glob(".*.pending-*"))


def test_exclusive_recovery_seal_rejects_nonblocked_certificate(tmp_path):
    journal, raw, result = _seal_fixture(tmp_path)
    result["frozen_gen0"]["release_approved"] = True
    target = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError, match="fail-closed"):
        inventory._seal_exclusive_inventory(target, journal, result)
    assert journal.read_bytes() == raw
    assert not target.exists()
