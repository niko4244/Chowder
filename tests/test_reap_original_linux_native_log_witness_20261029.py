"""Adversarial CPU-only checks for the archived Linux native-log witness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import pytest

from chowder_batch import reap_original_linux_native_log_witness_20261029 as witness
from chowder_batch import reap_original_linux_gsm8k_independent_pair_20261029 as pair


def _events() -> bytes:
    """Three synthetic tasks; interleaved slots mimic genuine async serving."""
    lines = []
    for task, slot, prompt in ((1, 0, 108), (2, 1, 78), (3, 2, 100)):
        lines.append(
            f"1.0 I slot   operator(): id  {slot} | task {task} | "
            f"new prompt, n_ctx_slot = 4096, n_keep = 0, task.n_tokens = {prompt}"
        )
    for task, slot, prompt, completion in (
        (1, 0, 108, 170), (2, 1, 78, 210), (3, 2, 100, 170)
    ):
        lines.extend((
            f"2.0 I slot print_timing: id  {slot} | task {task} | "
            f"prompt eval time = 123.4 ms / {prompt} tokens",
            f"2.1 I slot print_timing: id  {slot} | task {task} | "
            f"       eval time = 456.7 ms / {completion} tokens",
        ))
    return ("\n".join(lines) + "\n").encode()


def _human_archive(*, session: str = "authentic-session") -> bytes:
    rows = []
    for index in range(164):
        rows.append({
            "id": f"HumanEval/{index}",
            "correct": False,
            "terminated": True,
            "trigram_ratio": 1.0,
            "completion_tokens": index + 1,
            "seconds": 0.1,
            "predicted_per_second": 10.0,
            "text": "pass",
            "eval_session_id": session,
        })
    return ("\n".join(json.dumps(row) for row in rows) + "\n").encode()


def test_interleaved_complete_lifecycles() -> None:
    result = witness.parse_native_lifecycles(_events(), expected_n=3)
    assert result["n"] == 3
    assert result["prompt_eval_tokens"] == 286
    assert result["new_prompt_tokens"][108] == 1
    assert result["completion_tokens"][170] == 2
    assert sum(result["completion_tokens"].elements()) == 550


def test_duplicate_native_event_refused() -> None:
    data = _events()
    duplicate = data.splitlines()[0] + b"\n"
    with pytest.raises(witness.NativeWitnessError, match="duplicate"):
        witness.parse_native_lifecycles(data + duplicate, expected_n=3)


def test_missing_native_completion_refused() -> None:
    lines = _events().splitlines(keepends=True)
    with pytest.raises(witness.NativeWitnessError, match="complete evaluation"):
        witness.parse_native_lifecycles(b"".join(lines[:-1]), expected_n=3)


def test_duplicate_native_task_refused() -> None:
    data = _events().replace(b"task 3", b"task 2")
    with pytest.raises(witness.NativeWitnessError):
        witness.parse_native_lifecycles(data, expected_n=3)


def test_native_slot_swap_refused() -> None:
    data = _events().replace(
        b"print_timing: id  1 | task 2",
        b"print_timing: id  0 | task 2",
    )
    with pytest.raises(witness.NativeWitnessError, match="changed GPU serving slot"):
        witness.parse_native_lifecycles(data, expected_n=3)


def test_native_context_drift_refused() -> None:
    with pytest.raises(witness.NativeWitnessError, match="frozen context"):
        witness.parse_native_lifecycles(
            _events().replace(b"n_ctx_slot = 4096", b"n_ctx_slot = 8192"),
            expected_n=3,
        )


def test_native_event_order_refused() -> None:
    lines = _events().splitlines(keepends=True)
    # A timing event for task 1 must not precede its new-prompt event.
    lines[0], lines[3] = lines[3], lines[0]
    with pytest.raises(witness.NativeWitnessError, match="out of order"):
        witness.parse_native_lifecycles(b"".join(lines), expected_n=3)


def test_native_missing_population_refused() -> None:
    with pytest.raises(witness.NativeWitnessError, match="complete cohort"):
        witness.parse_native_lifecycles(_events(), expected_n=4)


def test_native_invalid_timing_refused() -> None:
    with pytest.raises(witness.NativeWitnessError, match="malformed"):
        witness.parse_native_lifecycles(
            _events().replace(b"123.4 ms", b"NaN ms"), expected_n=3
        )


def test_native_token_zero_refused() -> None:
    with pytest.raises(witness.NativeWitnessError, match="no tokens"):
        witness.parse_native_lifecycles(
            _events().replace(b"210 tokens", b"0 tokens"), expected_n=3
        )


def test_human_population_exact_and_session_bound() -> None:
    tokens = witness.human_completion_tokens(_human_archive(), "authentic-session")
    assert sum(tokens.values()) == 164
    assert sum(tokens.elements()) == 13530


def test_human_duplicate_refused() -> None:
    raw = _human_archive().replace(b"HumanEval/163", b"HumanEval/162")
    with pytest.raises(witness.NativeWitnessError, match="row/session"):
        witness.human_completion_tokens(raw, "authentic-session")


def test_human_cross_session_refused() -> None:
    with pytest.raises(witness.NativeWitnessError, match="row/session"):
        witness.human_completion_tokens(_human_archive(), "different-session")


def test_human_truncated_refused() -> None:
    with pytest.raises(witness.NativeWitnessError, match="incomplete"):
        witness.human_completion_tokens(_human_archive().rstrip(b"\n"), "authentic-session")


def test_original_file_sha_pin_refuses_changed_bytes(tmp_path: Path) -> None:
    evidence = tmp_path / "native.log"
    evidence.write_bytes(_events())
    pinned = hashlib.sha256(evidence.read_bytes()).hexdigest()
    assert pair._read_pinned(evidence, pinned, 1024 * 1024) == _events()
    evidence.write_bytes(_events() + b"forged\n")
    with pytest.raises(pair.IndependentPairError, match="SHA differs"):
        pair._read_pinned(evidence, pinned, 1024 * 1024)


def test_live_pinned_native_logs_match_archived_rows_without_inference() -> None:
    if not all(spec["log"].is_file() and spec["native"].is_file()
               and spec["human"].is_file() and pair.ARMS[arm]["rows"].is_file()
               for arm, spec in witness.ARMS.items()):
        pytest.skip("Original F: archives not mounted on this test host")
    result = witness.verify_original_native_log_pair()
    assert {arm: result["arms"][arm]["native_requests"] for arm in witness.ARMS} == {
        "q4": 1483, "q8": 1483,
    }
    assert result["arms"]["q4"]["native_generation_tokens"] == 497474
    assert result["arms"]["q8"]["native_generation_tokens"] == 507841
    assert result["same_canonical_prompt_token_multiset"] is True
    assert result["per_item_native_response_text_authenticated"] is False
    assert result["independent_cryptographic_execution_witness"] is False
    assert result["frozen_windows_gate1_qualified"] is False
    assert result["frozen_dense_gate2_qualified"] is False
    assert result["status"] == "BLOCKED"
    assert result["chosen"] is None
    assert result["release_approved"] is False


def test_live_corrupted_native_generation_is_independently_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not witness.ARMS["q4"]["log"].is_file():
        pytest.skip("Original F: native log not mounted")
    original = pair._read_pinned

    def altered(path: Path, sha256: str, limit: int) -> bytes:
        raw = original(path, sha256, limit)
        if path == witness.ARMS["q4"]["log"]:
            raw, count = re.subn(
                rb"(\|\s+eval time =\s+[\d.]+ ms /\s+)(\d+)( tokens)",
                lambda m: m.group(1) + str(int(m.group(2)) + 1).encode() + m.group(3),
                raw,
                count=1,
            )
            assert count == 1
        return raw

    monkeypatch.setattr(pair, "_read_pinned", altered)
    with pytest.raises(witness.NativeWitnessError, match="token multiset"):
        witness.verify_original_native_log_pair()


def test_live_forged_native_session_receipt_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not witness.ARMS["q4"]["native"].is_file():
        pytest.skip("Original F: native receipt not mounted")
    original = pair._read_pinned

    def altered(path: Path, sha256: str, limit: int) -> bytes:
        raw = original(path, sha256, limit)
        if path == witness.ARMS["q4"]["native"]:
            forged = json.loads(raw)
            forged["session_id"] = "forged-different-native-session"
            return json.dumps(forged).encode()
        return raw

    monkeypatch.setattr(pair, "_read_pinned", altered)
    with pytest.raises(witness.NativeWitnessError, match="source/session"):
        witness.verify_original_native_log_pair()
