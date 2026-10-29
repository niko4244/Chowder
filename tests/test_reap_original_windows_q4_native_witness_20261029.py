"""CPU-only adversarial checks of original Windows Q4 native timing correlation."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json

import pytest

from chowder_batch import reap_original_gsm8k_independent_grade_20261029 as grade
from chowder_batch import reap_original_linux_native_log_witness_20261029 as native
from chowder_batch import reap_original_windows_q4_native_witness_20261029 as witness


@pytest.fixture(scope="module")
def archived():
    if not witness.ROOT.is_dir():
        pytest.skip("Original immutable F: Windows Q4 evidence not mounted")
    raw = {
        name.split("/")[-1]: witness.pair._read_pinned(
            witness.ROOT / name, sha, witness.LIMITS.get(name.split("/")[-1], 2 * 1024 * 1024)
        ) for name, sha in witness.PINS.items()
    }
    receipts = {name: witness.pair._json(raw[name]) for name in
                ("full1319-terminal.json", "serve.json", "gsm8k.inputs.json", "summary.json")}
    return raw, receipts, grade.load_canonical()


def _rows(raw):
    return [witness.pair._json(line) for line in raw.splitlines()]


def _jsonl(rows):
    return b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)


def test_authentic_original_windows_native_full1319_regrades_and_correlates(archived):
    result = witness.verify_original_windows_q4_native()
    assert result["independently_regraded_correct"] == 1224
    assert result["independently_regraded_n"] == 1319
    assert result["native_complete_task_lifecycles"] == 1319
    assert result["per_item_native_timing_matches"] == 1319
    assert result["per_item_native_timing_mapping_sha256"] == witness.EXPECTED_BINDING_SHA
    assert result["native_response_text_cryptographically_authenticated"] is False
    assert result["model_weights_rehashed_at_original_execution"] is False
    assert result["same_session_dense_reference_available"] is False
    assert result["frozen_gate1_qualified"] is False
    assert result["frozen_gate2_qualified"] is False
    assert (result["status"], result["chosen"], result["release_approved"]) == (
        "BLOCKED", None, False
    )


def test_independent_windows_regrade_does_not_trust_forged_correct(archived):
    raw, _, canonical = archived
    rows = _rows(raw["gsm8k.jsonl"])
    rows[0]["correct"] = not rows[0]["correct"]
    with pytest.raises(witness.WindowsQ4WitnessError, match="regrading"):
        witness.regrade_windows_rows(_jsonl(rows), canonical)


def test_independent_windows_regrade_rejects_duplicate_and_incomplete(archived):
    raw, _, canonical = archived
    rows = _rows(raw["gsm8k.jsonl"])
    with pytest.raises(witness.WindowsQ4WitnessError, match="duplicate"):
        witness.regrade_windows_rows(_jsonl([*rows, rows[0]]), canonical)
    with pytest.raises(witness.WindowsQ4WitnessError, match="regrading"):
        witness.regrade_windows_rows(_jsonl(rows[:200]), canonical)


def test_independent_windows_regrade_rejects_forged_trigram(archived):
    raw, _, canonical = archived
    rows = _rows(raw["gsm8k.jsonl"])
    rows[0]["trigram_ratio"] = 0.0
    with pytest.raises(witness.WindowsQ4WitnessError, match="trigram"):
        witness.regrade_windows_rows(_jsonl(rows), canonical)


def test_independent_windows_regrade_rejects_forged_termination(archived):
    raw, _, canonical = archived
    rows = _rows(raw["gsm8k.jsonl"])
    rows[0]["terminated"] = not rows[0]["terminated"]
    with pytest.raises(witness.WindowsQ4WitnessError):
        witness.regrade_windows_rows(_jsonl(rows), canonical)


@pytest.mark.parametrize("path,key,value", [
    ("serve.json", "flags", ["--fit", "off"]),
    ("serve.json", "measured_experts_offloaded", 1),
    ("serve.json", "eval_session_id", "forged"),
    ("gsm8k.inputs.json", "source_items_n", 200),
    ("full1319-terminal.json", "native_shutdown_verified", False),
    ("full1319-terminal.json", "release_approved", True),
    ("full1319-terminal.json", "status", "RELEASE_APPROVED"),
])
def test_source_receipt_tampering_fails_closed(archived, path, key, value):
    raw, receipts, _ = archived
    altered = deepcopy(receipts)
    altered[path][key] = value
    with pytest.raises(witness.WindowsQ4WitnessError):
        witness.validate_original_receipts(altered, raw["server.log"])


def test_native_per_item_timing_matches_real_original_rows(archived):
    raw, _, _ = archived
    parsed = native.parse_native_lifecycles(raw["server.log"], expected_n=1319)
    binding = native.bind_archived_rows_to_native_timings(
        (raw["gsm8k.jsonl"], b""), parsed["per_task_eval_timing"]
    )
    assert binding["n"] == 1319
    assert binding["mapping_sha256"] == witness.EXPECTED_BINDING_SHA
    assert parsed["native_releases"] == 1319
    assert parsed["cached_prompt_savings"] == Counter({0: 4, 11: 1315})
    assert parsed["max_simultaneously_active_slots"] == 4


def test_native_timing_fingerprint_rejects_forged_response_rate(archived):
    raw, _, _ = archived
    parsed = native.parse_native_lifecycles(raw["server.log"], expected_n=1319)
    rows = _rows(raw["gsm8k.jsonl"])
    rows[0]["predicted_per_second"] *= 2
    with pytest.raises(native.NativeWitnessError, match="timing fingerprint"):
        native.bind_archived_rows_to_native_timings(
            (_jsonl(rows), b""), parsed["per_task_eval_timing"]
        )


def test_native_identity_swap_changes_pinned_mapping_even_if_totals_preserved(archived):
    raw, _, _ = archived
    parsed = native.parse_native_lifecycles(raw["server.log"], expected_n=1319)
    rows = _rows(raw["gsm8k.jsonl"])
    first = rows[0]
    second = next(row for row in rows[1:]
                  if row["completion_tokens"] != first["completion_tokens"])
    for key in ("completion_tokens", "predicted_per_second"):
        first[key], second[key] = second[key], first[key]
    swapped = native.bind_archived_rows_to_native_timings(
        (_jsonl(rows), b""), parsed["per_task_eval_timing"]
    )
    assert swapped["n"] == 1319
    assert swapped["mapping_sha256"] != witness.EXPECTED_BINDING_SHA
    assert sum(row["completion_tokens"] for row in rows) == 459264


def test_native_log_refuses_missing_terminal_release(archived):
    raw, _, _ = archived
    lines = raw["server.log"].splitlines(keepends=True)
    for index in range(len(lines) - 1, -1, -1):
        if b"stop processing: n_tokens" in lines[index]:
            del lines[index]
            break
    else:
        pytest.fail("original log has no native releases")
    with pytest.raises(native.NativeWitnessError):
        native.parse_native_lifecycles(b"".join(lines), expected_n=1319)


def test_native_log_refuses_truncated_release(archived):
    raw, _, _ = archived
    changed = raw["server.log"].replace(b"truncated = 0", b"truncated = 1", 1)
    assert changed != raw["server.log"]
    with pytest.raises(native.NativeWitnessError, match="truncated"):
        native.parse_native_lifecycles(changed, expected_n=1319)


def test_original_f_evidence_pins_reject_tampered_bytes(archived, tmp_path):
    raw, _, _ = archived
    altered = tmp_path / "server.log"
    altered.write_bytes(raw["server.log"] + b"tampered\n")
    with pytest.raises(witness.pair.IndependentPairError, match="SHA"):
        witness.pair._read_pinned(altered, witness.PINS["reap128_v2_q4/server.log"],
                                 witness.LIMITS["server.log"])
