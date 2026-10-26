"""CPU-only adversarial tests for complete external source-bound suite audit."""

import copy
import hashlib
import json

import pytest

from chowder_batch.reap_external_full_audit import (
    FROZEN_FLAGS, Q4_SHA, SERVER_SHA, _check_machine, _check_rows,
    _check_original_native_log,
)


def _receipt():
    return {
        "schema": "reap-v2-q4-kaggle-complete-provisional-eval.v1",
        "status": "COMPLETE_FULL_DATA_PROVISIONAL_NOT_FROZEN_WINDOWS_GATE",
        "not_official_windows_serving_attestation": True,
        "scientific_selection": "UNKNOWN",
        "flags": list(FROZEN_FLAGS),
        "session_id": "1" * 32,
        "data": {"q4_sha256": Q4_SHA, "server_binary_sha256": SERVER_SHA},
        "source_sha256": {
            "gsm8k_full": "913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89",
            "humaneval_full": "49fb688c6788b0531803dfd1cb85a69926ec5a7179e74cdfffc53ba52a92b420",
            "eval": "9736714f21834a45c23599a50fa92ef3bebc6cf70967aca7e2329d26b4d73ee1",
            "score": "d2e59aca5c0a621bac8d4979d522c9b6107f811610567d21c92a2b53960860eb",
        },
        "suite_results": {},
    }


def test_only_complete_frozen_model_and_source_are_accepted():
    assert Q4_SHA == "2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7"
    _check_machine(_receipt(), "reap128_v2_q4")


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(status="REFUSED_OR_ABORTED"),
    lambda r: r.update(flags=["--fit", "off", "-np", "4", "-c", "16384", "--jinja"]),
    lambda r: r.update(scientific_selection="PASS"),
    lambda r: r.update(not_official_windows_serving_attestation=False),
    lambda r: r.update(session_id="unexpected"),
    lambda r: r["data"].update(q4_sha256="0" * 64),
    lambda r: r["source_sha256"].update(gsm8k_full="0" * 64),
    lambda r: r["source_sha256"].update(score="0" * 64),
])
def test_host_provenance_or_scientific_promotion_mutations_refused(mutate):
    receipt = copy.deepcopy(_receipt())
    mutate(receipt)
    with pytest.raises(ValueError):
        _check_machine(receipt, "reap128_v2_q4")


def _rows():
    original = [
        {"id": "gsm8k-0", "expected": "2"},
        {"id": "gsm8k-1", "expected": "3"},
    ]
    rows = [
        {"id": "gsm8k-0", "correct": True, "terminated": True,
         "text": "The answer is 2", "trigram_ratio": 1.0,
         "completion_tokens": 12, "seconds": 0.5,
         "predicted_per_second": 25.0, "eval_session_id": "1" * 32},
        {"id": "gsm8k-1", "correct": False, "terminated": False,
         "text": "The answer is 3", "trigram_ratio": 1.0,
         "completion_tokens": 13, "seconds": 0.7,
         "predicted_per_second": None, "eval_session_id": "1" * 32},
    ]
    return original, rows


def test_original_row_score_and_session_recomputed_without_running_model():
    original, rows = _rows()
    result = _check_rows("gsm8k", original, rows, "1" * 32)
    assert result["n"] == 2
    assert result["correct"] == 1
    assert result["terminated"] == 1


@pytest.mark.parametrize("change", [
    lambda r: r.pop(),
    lambda r: r.append(dict(r[0])),
    lambda r: r[0].update(id="gsm8k-999"),
    lambda r: r[0].update(correct=False),
    lambda r: r[1].update(correct=True),
    lambda r: r[0].update(eval_session_id="2" * 32),
    lambda r: r[0].update(terminated=False),
    lambda r: r[0].update(trigram_ratio=-1),
    lambda r: r[0].update(completion_tokens=True),
    lambda r: r[0].update(seconds=float("inf")),
    lambda r: r[0].update(text=None),
])
def test_missing_duplicate_poisoned_stale_or_fabricated_rows_fail_closed(change):
    original, rows = _rows()
    change(rows)
    with pytest.raises(ValueError):
        _check_rows("gsm8k", original, rows, "1" * 32)


def test_complete_row_bytes_hash_exact_json_linewise(tmp_path):
    _, rows = _rows()
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b"\n".join(json.dumps(row).encode() for row in rows) + b"\n")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert expected != hashlib.sha256(path.read_bytes() + b" ").hexdigest()


def _native_log_bytes():
    return (b"I load_tensors: offloaded 41/41 layers to GPU\n"
            b"I llama_context: n_seq_max = 4\n"
            b"I llama_context: n_ctx = 16384\n"
            b"I llama_context: n_ctx_seq = 4096\n"
            b"I llama_kv_cache: size = 320.00 MiB\n"
            b"I llama_server: model loaded\n"
            b"I llama_server: listening on http://127.0.0.1:8790\n")


def test_exact_original_cuda_native_log_attests_gpu_and_four_way_kv_geometry():
    log = _native_log_bytes()
    claim = {"server_log_sha256": hashlib.sha256(log).hexdigest(),
             "server_log_bytes": len(log)}
    result = _check_original_native_log(claim, log)
    assert result["native_gpu_offloaded_layers"] == "41/41"
    assert result["n_ctx"] == 16384


@pytest.mark.parametrize("mutation", [
    lambda b: b.replace(b"41/41", b"40/41"),
    lambda b: b.replace(b"n_seq_max = 4", b"n_seq_max = 2"),
    lambda b: b.replace(b"n_ctx = 16384", b"n_ctx = 8192"),
    lambda b: b.replace(b"n_ctx_seq = 4096", b"n_ctx_seq = 2048"),
    lambda b: b.replace(b"320.00 MiB", b"80.00 MiB"),
    lambda b: b.replace(b"127.0.0.1:8790", b"127.0.0.1:9999"),
])
def test_original_gpu_fit_log_fail_closed_even_if_mutated_log_is_rehashed(mutation):
    log = mutation(_native_log_bytes())
    claim = {"server_log_sha256": hashlib.sha256(log).hexdigest(),
             "server_log_bytes": len(log)}
    with pytest.raises(ValueError):
        _check_original_native_log(claim, log)
