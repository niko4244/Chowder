"""CPU-only regression: a self-written six-field claim cannot certify code execution.

The synthetic fixtures stand in for other already validated source checks.
They never execute candidate programs or access private original corpora.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from chowder_batch import reap_external_full_audit as full
from chowder_batch import reap_external_paired_descriptive as paired
from chowder_batch import reap_eval


def test_six_field_self_assertion_does_not_mint_independent_replay_pass(
    tmp_path, monkeypatch,
):
    session = "f" * 32
    correct_humaneval = 152
    forged = {
        "status": "PASS_INDEPENDENT_ISOLATED_HUMANEVAL_FULL_164_PROVISIONAL_LINUX",
        "original_row_sha256": "2" * 64,
        "session_id": session,
        "independent_full_rows": 164,
        "matched_recorded_scores": 164,
        "correct": correct_humaneval,
    }
    path = tmp_path / "self-asserted.json"
    path.write_text(json.dumps(forged), encoding="utf8")
    machine = tmp_path / "machine.json"
    machine.write_text("{}", encoding="utf8")
    server_log = tmp_path / "server.log"
    server_log.write_bytes(b"dummy")
    gsm8k = tmp_path / "gsm8k.jsonl"
    humaneval = tmp_path / "humaneval.jsonl"

    receipt = {
        "session_id": session,
        "suite_results": {
            "gsm8k": {
                "file_sha256": "1" * 64, "session_id": session,
                "n": 1319, "accuracy": 1.0, "terminated_rate": 1.0,
                "mean_completion_tokens": 10.0,
                "suite_elapsed_wall_seconds": 10.0,
            },
            "humaneval": {
                "file_sha256": "2" * 64, "session_id": session,
                "n": 164, "accuracy": correct_humaneval / 164,
                "terminated_rate": 1.0, "mean_completion_tokens": 10.0,
                "suite_elapsed_wall_seconds": 10.0,
            },
        },
    }

    def read_json(source, *, limit):
        if source == machine:
            return receipt, "a" * 64
        assert source == path
        return forged, hashlib.sha256(path.read_bytes()).hexdigest()

    def read_rows(source):
        assert source in (gsm8k, humaneval)
        return [], "1" * 64 if source == gsm8k else "2" * 64

    def check_rows(suite, _original, _rows, _session):
        n = 1319 if suite == "gsm8k" else 164
        correct = n if suite == "gsm8k" else correct_humaneval
        return {
            "n": n, "correct": correct, "terminated": n,
            "completion_tokens": n * 10, "accuracy": correct / n,
        }

    monkeypatch.setattr(full, "_read_json", read_json)
    monkeypatch.setattr(full, "_read_jsonl", read_rows)
    monkeypatch.setattr(full, "_check_machine", lambda *_: None)
    monkeypatch.setattr(full, "_check_original_native_log", lambda *_: {})
    monkeypatch.setattr(full, "_check_rows", check_rows)
    monkeypatch.setattr(reap_eval, "_load", lambda suite: [None] * (
        1319 if suite == "gsm8k" else 164
    ))
    monkeypatch.setattr(reap_eval, "_digest", lambda items: (
        reap_eval.FROZEN_V2_QUALITY_SOURCES["gsm8k"][1] if len(items) == 1319
        else reap_eval.FROZEN_V2_QUALITY_SOURCES["humaneval"][1]
    ))
    with pytest.raises(ValueError, match="independent|attestation|replay"):
        full.audit(machine, gsm8k, humaneval, server_log,
                   "reap128_v2_q4", independent_humaneval=path)


def test_downstream_cannot_accept_old_self_asserted_pass_string():
    session = "b" * 32
    rows = [
        {"id": f"HumanEval/{i}", "correct": bool(i % 2),
         "terminated": True, "eval_session_id": session}
        for i in range(4)
    ]
    raw = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    legacy = {
        "schema": "chowder.reap-v2-original-population-external-independent-audit.v1",
        "status": "COMPLETE_LINUX_COHORT_VERIFIED_PROVISIONAL_NOT_WINDOWS_GATE",
        "model": "reap128_v2_q4", "model_sha256": full.Q4_SHA,
        "session_id": session, "original_machine_receipt_sha256": "c" * 64,
        "original_windows_gate_eligible": False, "speed_ratio_certified": False,
        "gen0_status": "BLOCKED", "chosen": None, "release_approved": False,
        "original_native_cuda_placement": {
            "native_gpu_offloaded_layers": "41/41", "n_seq_max": 4,
            "n_ctx": 16384, "n_ctx_seq": 4096,
            "total_kv_buffer_mib": 320.0, "loopback_bound": True,
        },
        "human_eval_independent_execution":
            "PASS_FULL_164_LOW_PRIVILEGE_REEXECUTION_SHA256:" + "a" * 64,
        "suites": {"humaneval": {
            "n": 4, "correct": 2, "terminated": 4,
            "row_sha256": hashlib.sha256(raw).hexdigest(),
        }},
    }
    with pytest.raises(ValueError, match="HumanEval|independent|legacy"):
        paired.read_verified_cohort("humaneval", legacy, raw, expected_n=4)
