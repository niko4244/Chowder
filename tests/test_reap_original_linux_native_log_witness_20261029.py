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
            f"2.2 I slot      release: id  {slot} | task {task} | "
            f"stop processing: n_tokens = {prompt + completion - 1}, truncated = 0",
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
    assert result["cached_prompt_savings"] == {0: 3}
    assert result["cold_first_slot_requests"] is True
    assert result["max_simultaneously_active_slots"] == 3
    assert result["native_releases"] == 3
    assert result["per_task_declared_prompt_tokens"] == {1: 108, 2: 78, 3: 100}


def test_concurrent_requests_cannot_share_one_native_slot() -> None:
    # Each task's events remain locally ordered, so the older parser accepted
    # this physically impossible concurrent reuse of slot zero.
    data = _events().replace(
        b"id  1 | task 2", b"id  0 | task 2"
    )
    with pytest.raises(witness.NativeWitnessError, match="overlapping native requests"):
        witness.parse_native_lifecycles(data, expected_n=3)


def test_native_prompt_eval_cannot_exceed_declared_prompt() -> None:
    data = _events().replace(
        b"prompt eval time = 123.4 ms / 108 tokens",
        b"prompt eval time = 123.4 ms / 109 tokens",
    )
    with pytest.raises(witness.NativeWitnessError, match="exceeds declared"):
        witness.parse_native_lifecycles(data, expected_n=3)


def test_native_timing_requires_active_slot() -> None:
    lines = _events().splitlines(keepends=True)
    # Task one finishes, but an unknown task emits timing on the idle slot.
    lines.append(
        b"3.0 I slot print_timing: id  0 | task 4 | "
        b"prompt eval time = 1.0 ms / 108 tokens\n"
    )
    with pytest.raises(witness.NativeWitnessError, match="without active matching slot"):
        witness.parse_native_lifecycles(b"".join(lines), expected_n=3)


def _timing_rows() -> bytes:
    return (
        json.dumps({"id": "case/a", "completion_tokens": 178,
                    "predicted_per_second": 177000 / 5632.46})
        + "\n" + json.dumps({"id": "case/b", "completion_tokens": 210,
                              "predicted_per_second": 209000 / 7000.01})
        + "\n"
    ).encode()


def test_unique_per_item_native_timing_fingerprint() -> None:
    result = witness.bind_archived_rows_to_native_timings(
        (_timing_rows(),), {11: (178, "5632.46"), 12: (210, "7000.01")}
    )
    assert result["n"] == result["unique_native_tasks"] == 2
    assert len(result["mapping_sha256"]) == 64
    assert result["row_to_native_task"] == {"case/a": 11, "case/b": 12}


def _canonical_prompt_lengths() -> dict[str, int]:
    return {**{f"gsm8k-{i}": 50 + (i % 19) for i in range(1319)},
            **{f"HumanEval/{i}": 100 + (i % 13) for i in range(164)}}


def test_cross_arm_per_item_native_prompt_length_binding() -> None:
    q4 = _canonical_prompt_lengths()
    result = witness.bind_canonical_prompts_across_arms({"q4": q4, "q8": q4.copy()})
    assert result["n"] == 1483
    assert len(result["per_item_prompt_length_sha256"]) == 64


def test_cross_arm_equal_aggregate_prompt_multiset_but_swapped_items_refused() -> None:
    q4 = _canonical_prompt_lengths()
    q8 = q4.copy()
    q8["gsm8k-0"], q8["gsm8k-1"] = q8["gsm8k-1"], q8["gsm8k-0"]
    assert sorted(q4.values()) == sorted(q8.values())
    with pytest.raises(witness.NativeWitnessError, match="per-item.*disagree"):
        witness.bind_canonical_prompts_across_arms({"q4": q4, "q8": q8})


def test_cross_arm_missing_or_noncanonical_prompt_identity_refused() -> None:
    q4 = _canonical_prompt_lengths()
    q8 = q4.copy()
    q8["gsm8k-0"] = q8.pop("gsm8k-1")
    with pytest.raises(witness.NativeWitnessError, match="population mismatch"):
        witness.bind_canonical_prompts_across_arms({"q4": q4, "q8": q8})
    q8 = q4.copy()
    q8["forged-id"] = q8.pop("gsm8k-0")
    with pytest.raises(witness.NativeWitnessError, match="population mismatch"):
        witness.bind_canonical_prompts_across_arms({"q4": q4, "q8": q8})


def test_native_timing_collision_refused_even_with_matching_token_total() -> None:
    # Equal generation-token totals do not establish per-row provenance.
    with pytest.raises(witness.NativeWitnessError, match="unique unused"):
        witness.bind_archived_rows_to_native_timings(
            (_timing_rows(),), {11: (178, "5632.46"), 12: (178, "5632.46")}
        )


def test_native_timing_forged_rate_refused() -> None:
    forged = _timing_rows().replace(
        b'"predicted_per_second": 31.', b'"predicted_per_second": 41.'
    )
    assert forged != _timing_rows()
    with pytest.raises(witness.NativeWitnessError, match="unique unused"):
        witness.bind_archived_rows_to_native_timings(
            (forged,), {11: (178, "5632.46"), 12: (210, "7000.01")}
        )


def test_native_timing_duplicate_row_identity_refused() -> None:
    duplicate = _timing_rows().replace(b"case/b", b"case/a")
    with pytest.raises(witness.NativeWitnessError, match="timing fingerprint"):
        witness.bind_archived_rows_to_native_timings(
            (duplicate,), {11: (178, "5632.46"), 12: (210, "7000.01")}
        )


def test_native_missing_slot_release_refused() -> None:
    with pytest.raises(witness.NativeWitnessError, match="incomplete evaluation"):
        witness.parse_native_lifecycles(
            b"\n".join(line for line in _events().splitlines()
                       if b"release: id  2 | task 3" not in line) + b"\n",
            expected_n=3,
        )


def test_native_truncated_slot_release_refused() -> None:
    with pytest.raises(witness.NativeWitnessError, match="release malformed or truncated"):
        witness.parse_native_lifecycles(
            _events().replace(b"truncated = 0", b"truncated = 1", 1),
            expected_n=3,
        )


def test_native_release_token_accounting_refused() -> None:
    with pytest.raises(witness.NativeWitnessError, match="release token accounting"):
        witness.parse_native_lifecycles(
            _events().replace(b"n_tokens = 277", b"n_tokens = 278", 1),
            expected_n=3,
        )


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
    for arm in ("q4", "q8"):
        assert result["arms"][arm]["native_cold_prompt_requests"] == 4
        assert result["arms"][arm]["native_cached_prompt_requests"] == 1479
        assert result["arms"][arm]["native_cached_prompt_tokens_saved"] == 16269
        assert result["arms"][arm]["native_slot_overlap_refused"] is True
        assert result["arms"][arm]["native_nontruncated_slot_releases"] == 1483
        assert result["arms"][arm]["native_archived_rows_uniquely_timing_correlated"] == 1483
        assert len(result["arms"][arm]["native_archived_timing_mapping_sha256"]) == 64
    assert result["arms"]["q4"]["native_archived_timing_mapping_sha256"] == (
        "f6c4612fa2a61a23a2b83ebf5e78c2fe3daa256d844f22f496b53213a13b6040"
    )
    assert result["arms"]["q8"]["native_archived_timing_mapping_sha256"] == (
        "86afb604b4a5d3f6a95564e3042fb0700f9a644d1518dbec809340e79415c861"
    )
    assert result["per_item_native_timing_fingerprints_uniquely_correlated"] is True
    assert result["same_canonical_prompt_token_multiset"] is True
    assert result["same_per_item_native_prompt_lengths"] is True
    assert result["same_per_item_native_prompt_lengths_n"] == 1483
    assert result["same_per_item_native_prompt_lengths_sha256"] == (
        "c11f1973905352303f6890e4ec5781e3776a89aacbd46281a881f2f92811e8b0"
    )
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
    with pytest.raises(
        witness.NativeWitnessError,
        match="token multiset|timing fingerprint|release token accounting",
    ):
        witness.verify_original_native_log_pair()


def test_live_equal_total_prompt_cache_redistribution_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not witness.ARMS["q4"]["log"].is_file():
        pytest.skip("Original F: native log not mounted")
    original = pair._read_pinned

    def altered(path: Path, sha256: str, limit: int) -> bytes:
        raw = original(path, sha256, limit)
        if path == witness.ARMS["q4"]["log"]:
            index = 0

            def shift(match: re.Match[bytes]) -> bytes:
                nonlocal index
                # Four initial cold requests are preserved. Two warm requests
                # are perturbed oppositely: totals still match, but per-task
                # cache accounting no longer matches the source-bound log.
                delta = 1 if index == 4 else -1 if index == 5 else 0
                index += 1
                return (match.group(1)
                        + str(int(match.group(2)) + delta).encode()
                        + match.group(3))

            raw = re.sub(
                rb"(prompt eval time =\s+[\d.]+ ms /\s+)(\d+)( tokens)",
                shift,
                raw,
            )
            assert index == 1483
        return raw

    monkeypatch.setattr(pair, "_read_pinned", altered)
    with pytest.raises(witness.NativeWitnessError, match="prompt-cache accounting"):
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
