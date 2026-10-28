"""Adversarial CPU-only tests; never run generated LiveCodeBench code."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chowder_batch import reap_v2_lcb_v6_append_only_witness_20261028 as witness

CANONICAL = [("a", "test.jsonl"), ("b", "test2.jsonl"),
             ("c", "test3.jsonl")]
FAILURE = (
    "URLError: <urlopen error [WinError 10061] "
    "No connection could be made because the target machine actively refused it>"
)


def row(qid: str, source: str, *, error: str | None = None,
        code: str = "print(1)") -> dict:
    return {
        "question_id": qid,
        "source_file": source,
        "task": "LCB_generation",
        "prompt_style": "OpenAIChat",
        "max_tokens": 4096,
        "temperature": 0.2,
        "top_p": 0.95,
        "raw": "" if error else "candidate confidential text",
        "code": "" if error else code,
        "error": error,
        "finish_reason": None,
    }


def encode(*rows: dict) -> bytes:
    return b"".join((json.dumps(item) + "\n").encode() for item in rows)


def make(initial: bytes, later: bytes | None = None) -> dict:
    return witness.build_witness(
        CANONICAL, initial, initial if later is None else later,
        provenance={"source_git_commit": "pinned"},
    )


def test_successful_live_append_is_accepted_without_pooling() -> None:
    first = encode(row("a", "test.jsonl"), row("b", "test2.jsonl", error=FAILURE))
    later = first + encode(row("c", "test3.jsonl", error=FAILURE))
    result = make(first, later)
    assert result["prefix_rows"] == 2
    assert result["subsequent_bytes_beyond_prefix"] > 0
    assert result["transport_classification"]["connection_refused_rows"] == 1
    assert result["transport_classification"]["unresolved_source_cases"] == 2
    assert result["official_score"] is None
    assert result["frozen_gen0"] == {
        "status": "BLOCKED", "chosen": None, "release_approved": False,
    }
    assert not result["cross_session_pooling_allowed"]
    assert "candidate confidential text" not in json.dumps(result)


def test_inflight_partial_append_excluded_and_accounted() -> None:
    first = encode(row("a", "test.jsonl")) + b'{"question_id": "b"'
    result = make(first, first + b"}\n")
    assert result["prefix_rows"] == 1
    assert result["initial_inflight_tail_bytes_excluded"] == 19
    assert result["subsequent_bytes_beyond_prefix"] == 21


def test_empty_code_without_transport_error_is_still_response() -> None:
    result = make(encode(row("a", "test.jsonl", code="")))
    assert result["transport_classification"]["successful_transport_rows"] == 1
    assert result["transport_classification"]["model_responses_with_empty_code"] == 1
    assert result["transport_classification"]["failed_transport_rows_not_scored"] == 0


@pytest.mark.parametrize("initial,later", [
    (encode(row("a", "test.jsonl")), b""),
    (encode(row("a", "test.jsonl")), encode(row("b", "test2.jsonl"))),
    (encode(row("a", "test.jsonl")), encode(row("a", "test.jsonl"))[:-1]),
    (encode(row("a", "test.jsonl")), b"corrupt" + encode(row("a", "test.jsonl"))),
])
def test_truncation_or_rewrite_refused(initial: bytes, later: bytes) -> None:
    with pytest.raises(ValueError, match="truncated or rewritten"):
        make(initial, later)


@pytest.mark.parametrize("raw", [
    b"",
    b'{"question_id":"a"}',
    b"\n",
    b'{"question_id":"a"}\n\n',
    b'{"question_id":"a", "question_id":"b"}\n',
    b'{"question_id":"a", "nested":{"x":1,"x":2}}\n',
    b'{"question_id":"a", "v":NaN}\n',
    b"\xff\n",
])
def test_empty_malformed_ambiguous_or_nonfinite_refused(raw: bytes) -> None:
    with pytest.raises(ValueError):
        make(raw)


def test_overlarge_unterminated_tail_refused() -> None:
    with pytest.raises(ValueError, match="excessive"):
        make(encode(row("a", "test.jsonl")) + b"x" * (witness.MAX_INFLIGHT_TAIL + 1))


@pytest.mark.parametrize("change", [
    {"question_id": "b"},
    {"source_file": "test2.jsonl"},
    {"task": "LCB_other"},
    {"prompt_style": "Other"},
    {"temperature": 0.0},
    {"top_p": 1.0},
    {"max_tokens": 2048},
    {"code": None},
    {"raw": None},
    {"error": "failure", "raw": "candidate"},
])
def test_source_prompt_transport_tampering_refused(change: dict) -> None:
    item = row("a", "test.jsonl")
    item.update(change)
    with pytest.raises(ValueError):
        make(encode(item))


def test_completed_rows_cannot_exceed_canonical_population() -> None:
    raw = encode(
        row("a", "test.jsonl"),
        row("b", "test2.jsonl"),
        row("c", "test3.jsonl"),
        row("d", "test4.jsonl"),
    )
    with pytest.raises(ValueError, match="excessive"):
        make(raw)


def test_partial_tail_does_not_allow_invalid_completed_row() -> None:
    raw = encode(row("a", "test.jsonl")) + b'{"question_id": "a"}\npartial'
    with pytest.raises(ValueError):
        make(raw)


def test_verify_append_only_requires_newline_terminated_prefix() -> None:
    with pytest.raises(ValueError, match="complete original prefix"):
        witness.verify_append_only(b"partial", b"partial-more")


def test_final_seal_verifies_file_identity_and_prefix(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    prefix = encode(row("a", "test.jsonl"))
    path.write_bytes(prefix)
    info = path.stat()
    receipt = {
        "prefix_bytes": len(prefix),
        "prefix_sha256": witness._sha_bytes(prefix),
        "journal_file_identity": {"device": info.st_dev, "inode": info.st_ino},
    }
    witness._verify_prefix_still_present(path, receipt)
    with path.open("ab") as stream:
        stream.write(encode(row("b", "test2.jsonl")))
    witness._verify_prefix_still_present(path, receipt)
    receipt["journal_file_identity"]["inode"] += 1
    with pytest.raises(ValueError, match="identity changed"):
        witness._verify_prefix_still_present(path, receipt)


def test_final_seal_refuses_rewritten_prefix(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    prefix = encode(row("a", "test.jsonl"))
    path.write_bytes(prefix)
    info = path.stat()
    receipt = {
        "prefix_bytes": len(prefix),
        "prefix_sha256": witness._sha_bytes(prefix),
        "journal_file_identity": {"device": info.st_dev, "inode": info.st_ino},
    }
    path.write_bytes(prefix.replace(b"candidate", b"malicious"))
    with pytest.raises(ValueError, match="prefix changed"):
        witness._verify_prefix_still_present(path, receipt)
