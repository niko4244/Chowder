"""Adversarial CPU-only original guard.v3 native/guard accounting tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from chowder_batch import reap_original600_guardv3_native_journal_witness_20261029 as witness


def launch(task: int, *, child: int = 0, slot: int = 0) -> bytes:
    return (f"1.0 I slot launch_slot_: id  {slot} | task {task} | "
            f"processing task, is_child = {child}\n").encode()


def prompt(task: int, *, tokens: int = 25) -> bytes:
    return (f"1.1 I slot print_timing: id  0 | task {task} | "
            f"prompt eval time = 10.0 ms / {tokens} tokens\n").encode()


def decode(task: int, *, tokens: int = 12) -> bytes:
    return (f"1.2 I slot print_timing: id  0 | task {task} | "
            f"       eval time = 100.0 ms / {tokens} tokens\n").encode()


def release(task: int, *, truncated: int = 0) -> bytes:
    return (f"1.3 I slot      release: id  0 | task {task} | "
            f"stop processing: n_tokens = 37, truncated = {truncated}\n").encode()


def completed(task: int) -> bytes:
    return launch(task) + prompt(task) + decode(task) + release(task)


def journal(n: int, *, event: str = "forwarded") -> bytes:
    return b"".join((json.dumps({
        "request_index": i, "event": event, "policy": "guard.v3",
        "prompt_tokens_after": 1200, "max_tokens": 16384,
        "tool_arguments_unrepairable": int(event != "forwarded"),
    }, separators=(",", ":")) + "\n").encode() for i in range(1, n + 1))


def test_complete_native_and_guard_reconcile_without_release_authority():
    actual = witness.reconcile_pair(completed(0) + completed(7), journal(2))
    assert actual["launched"] == actual["released"] == 2
    assert actual["fully_timed_releases"] == 2
    assert actual["requests"] == 2
    assert actual["unmatched_native_launches"] == 0
    assert actual["per_request_http_bytes_authenticated"] is False


def test_one_extra_native_launch_is_explicitly_unmatched_not_a_scored_request():
    actual = witness.reconcile_pair(completed(0) + completed(7), journal(1))
    assert actual["unmatched_native_launches"] == 1
    assert actual["fully_timed_releases"] == 2
    with pytest.raises(witness.NativeJournalError, match="discrepancy"):
        witness.reconcile_pair(completed(0) + completed(7), journal(3))


def test_unfinished_terminal_request_stays_unknown():
    actual = witness.reconcile_pair(completed(0) + launch(7), journal(2))
    assert actual["unfinished_unknown"] == 1
    assert actual["fully_timed_releases"] == 1
    assert actual["released"] == 1


def test_terminal_release_without_final_timings_stays_unknown():
    actual = witness.reconcile_pair(completed(0) + launch(7) + release(7), journal(2))
    assert actual["released_without_final_timings_unknown"] == 1
    assert actual["fully_timed_releases"] == 1
    assert actual["released"] == 2
    with pytest.raises(witness.NativeJournalError, match="terminal tail"):
        witness.parse_native(launch(0) + release(0) + completed(7))


@pytest.mark.parametrize("raw", [
    launch(0) + launch(7) + release(0),
    completed(0) + completed(0),
    launch(0, child=1) + prompt(0) + decode(0) + release(0),
    launch(0, slot=1) + prompt(0) + decode(0) + release(0),
    launch(0) + release(0) + prompt(0) + decode(0),
    launch(0) + prompt(0) + decode(0) + release(0, truncated=1),
    launch(0) + prompt(0) + decode(0) + release(0) + release(0),
    launch(0) + prompt(0) + prompt(0) + decode(0) + release(0),
    launch(0) + decode(0) + decode(0) + release(0),
    launch(0) + decode(0) + prompt(0) + release(0),
    completed(0)[:-1],
])
def test_reject_bad_native_order_or_incomplete_source(raw: bytes):
    with pytest.raises(witness.NativeJournalError):
        witness.parse_native(raw)


def test_unserved_native_log_cannot_be_mistaken_for_model_execution():
    assert witness.parse_native(b"source-only startup\n", allow_empty=True)["launched"] == 0
    with pytest.raises(witness.NativeJournalError, match="no model requests"):
        witness.parse_native(b"source-only startup\n")


def test_bad_native_prompt_and_decode_timing_are_not_silently_ignored():
    for malformed in (
        prompt(0, tokens=0),
        prompt(0).replace(b"10.0 ms", b"0.0 ms"),
        decode(0, tokens=0),
        decode(0).replace(b"100.0 ms", b"0.0 ms"),
        decode(0).replace(b"100.0 ms", b"-10.0 ms"),
    ):
        raw = launch(0) + malformed + prompt(0) + decode(0) + release(0)
        with pytest.raises(witness.NativeJournalError):
            witness.parse_native(raw)


def test_unrepairable_event_still_counts_as_a_native_request():
    actual = witness.reconcile_pair(completed(0), journal(1, event="TOOL_ARGUMENTS_UNREPAIRABLE"))
    assert actual["events"]["TOOL_ARGUMENTS_UNREPAIRABLE"] == 1
    assert actual["requests"] == 1


@pytest.mark.parametrize("tamper", [
    lambda raw: raw.replace(b'"request_index":1', b'"request_index":2'),
    lambda raw: raw.replace(b'"policy":"guard.v3"', b'"policy":"guard.v4"'),
    lambda raw: raw.replace(b'"max_tokens":16384', b'"max_tokens":99999'),
    lambda raw: raw.replace(b'"prompt_tokens_after":1200', b'"prompt_tokens_after":0'),
    lambda raw: raw[:-1],
    lambda raw: raw.replace(b'"event":"forwarded"', b'"event":"forged"'),
    lambda raw: raw.replace(b'"request_index":1', b'"request_index":1,"request_index":1'),
])
def test_reject_tampered_guard_journal(tamper):
    with pytest.raises(witness.NativeJournalError):
        witness.parse_journal(tamper(journal(1)))


def test_refuse_original_file_hash_drift():
    with pytest.raises(witness.NativeJournalError, match="SHA drift"):
        witness._require_pinned(b"forged", "0" * 64, "original-native")


def test_reject_malformed_or_nonfinite_journal():
    with pytest.raises(witness.NativeJournalError):
        witness.parse_journal(b'{"request_index":NaN}\n')
    with pytest.raises(witness.NativeJournalError):
        witness.parse_journal(b"[]\n")


@pytest.mark.skipif(
    os.name != "nt" or not witness.historical.ROOT.is_dir(),
    reason="original Windows F: guard.v3 source evidence not mounted",
)
def test_original_source_bound_sessions_in_fresh_historical_interpreter():
    # Original historical Git-pinned modules must be imported in a fresh
    # interpreter; other tests may have imported modern modules already.
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-B", "-m",
         "chowder_batch.reap_original600_guardv3_native_journal_witness_20261029"],
        cwd=repo, capture_output=True, text=True, timeout=90, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["original_scored_tasks"] == 88
    assert report["original_strict_passes"] == 3
    assert report["native_launches"] == 2398
    assert report["native_releases"] == 2395
    assert report["native_fully_timed_releases"] == 2393
    assert report["native_untimed_releases_unknown"] == 2
    assert report["guard_requests"] == 2396
    assert report["unmatched_native_launches"] == 2
    assert report["unfinished_native_unknown"] == 3
    assert report["session_accounting_sha256"] == (
        "5105f7f928d5c10eb447d2cb89f8b3e34792aa58235c202e2366f6e97525847e"
    )
    assert report["native_http_request_response_authenticated"] is False
    assert report["gen0_status"] == "BLOCKED"
    assert report["chosen"] is None and report["release_approved"] is False
