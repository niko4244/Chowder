"""CPU-only prospective sequence/concurrency consistency adversarial tests."""

from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json

import pytest

from chowder_batch import reap_v2_native_capture_temporal_contract_20261030 as temporal


def _event(i: int, *, start: int | None = None, end: int | None = None) -> dict:
    return {
        "schema": temporal.EVENT_SCHEMA, "id": f"gsm8k-{i}",
        "capture_seq": i, "request_id": f"{i + 1:032x}",
        "collector_instance_id": "a" * 32, "eval_session_id": "b" * 32,
        "collector_source_sha256": "c" * 64, "gguf_sha256": "d" * 64,
        "server_pid": 1234, "server_port": 8890,
        "start_monotonic_ns": i * 10 + 1 if start is None else start,
        "end_monotonic_ns": i * 10 + 6 if end is None else end,
    }


def _raw(events: list[dict]) -> bytes:
    return b"".join(json.dumps(e, separators=(",", ":")).encode() + b"\n"
                    for e in events)


def test_ordered_parallel_capture_passes_but_never_authenticates_origin():
    events = [_event(i, start=i + 1, end=i + 20) for i in range(4)]
    result = temporal.verify_temporal_stream(_raw(events), expected_n=4,
                                             approved_http_parallel=4)
    assert result["peak_simultaneous_requests"] == 4
    assert result["http_parallel_limit_checked"] is True
    assert result["observed_monotonic_span_ns"] == 22
    assert result["status"] == "BLOCKED"
    assert result["chosen"] is None and result["release_approved"] is False
    assert result["native_http_lineage_authenticated"] is False
    assert result["raw_http_bytes_independently_verified"] is False


def test_nonoverlapping_capture_has_one_slot():
    result = temporal.verify_temporal_stream(_raw([_event(i) for i in range(5)]),
                                             expected_n=5)
    assert result["peak_simultaneous_requests"] == 1
    assert result["http_parallel_limit_checked"] is False


def test_server_np4_does_not_imply_four_inflight_http_requests():
    events = [_event(i, start=i + 1, end=i + 100) for i in range(5)]
    result = temporal.verify_temporal_stream(_raw(events), expected_n=5,
                                             approved_http_parallel=5)
    assert result["peak_simultaneous_requests"] == 5
    assert result["declared_evaluator_http_parallel"] == 5


@pytest.mark.parametrize("mutation", [
    "swapped_lines", "permuted_seq", "decreasing_start", "bad_interval",
    "negative_start", "bool_start", "overflow_end", "too_many_inflight",
    "mixed_instance", "mixed_model", "mixed_server", "mixed_session",
    "duplicate_task", "duplicate_request", "malformed_request", "bad_schema",
    "wrong_n", "extra_event", "missing_event", "bool_sequence",
    "noncanonical_task", "out_of_range_task",
])
def test_adversarial_capture_sequence_refused(mutation):
    events = [_event(i) for i in range(5)]
    if mutation == "swapped_lines":
        events[0], events[1] = events[1], events[0]
    elif mutation == "permuted_seq":
        events[1]["capture_seq"] = 4
        events[4]["capture_seq"] = 1
    elif mutation == "decreasing_start":
        events[2]["start_monotonic_ns"] = 5
    elif mutation == "bad_interval":
        events[0]["end_monotonic_ns"] = 1
    elif mutation == "negative_start":
        events[0]["start_monotonic_ns"] = -1
    elif mutation == "bool_start":
        events[0]["start_monotonic_ns"] = True
    elif mutation == "overflow_end":
        events[0]["end_monotonic_ns"] = 1 << 63
    elif mutation == "too_many_inflight":
        events = [_event(i, start=i + 1, end=i + 100) for i in range(5)]
    elif mutation == "mixed_instance":
        events[1]["collector_instance_id"] = "f" * 32
    elif mutation == "mixed_model":
        events[1]["gguf_sha256"] = "f" * 64
    elif mutation == "mixed_server":
        events[1]["server_pid"] = 4567
    elif mutation == "mixed_session":
        events[1]["eval_session_id"] = "f" * 32
    elif mutation == "duplicate_task":
        events[1]["id"] = events[0]["id"]
    elif mutation == "noncanonical_task":
        events[1]["id"] = "gsm8k-01"
    elif mutation == "out_of_range_task":
        events[1]["id"] = "gsm8k-999"
    elif mutation == "duplicate_request":
        events[1]["request_id"] = events[0]["request_id"]
    elif mutation == "malformed_request":
        events[1]["request_id"] = "NOT-HEX"
    elif mutation == "bad_schema":
        events[1]["schema"] = "foreign"
    elif mutation == "wrong_n":
        with pytest.raises(temporal.TemporalCaptureRefused):
            temporal.verify_temporal_stream(_raw(events), expected_n=4)
        return
    elif mutation == "extra_event":
        events.append(deepcopy(events[-1]))
    elif mutation == "missing_event":
        events.pop()
    elif mutation == "bool_sequence":
        events[0]["capture_seq"] = False
    with pytest.raises(temporal.TemporalCaptureRefused):
        temporal.verify_temporal_stream(_raw(events), expected_n=5,
                                        approved_http_parallel=4)


@pytest.mark.parametrize("raw", [
    b"", b"[]\n", b"{}", b"\xef\xbb\xbf{}\n", b"{\"x\":NaN}\n",
    b'{"capture_seq":0,"capture_seq":0}\n', b"{}\r\n",
])
def test_malformed_json_or_framing_refused(raw):
    with pytest.raises(temporal.TemporalCaptureRefused):
        temporal.verify_temporal_stream(raw, expected_n=1)


@pytest.mark.parametrize("count", [0, -1, True, 1320, 1.5, "2"])
def test_invalid_expected_population_refused(count):
    with pytest.raises(temporal.TemporalCaptureRefused):
        temporal.verify_temporal_stream(_raw([_event(0)]), expected_n=count)


@pytest.mark.parametrize("parallel", [0, -1, True, 65, 1.5, "4"])
def test_invalid_independent_http_parallelism_refused(parallel):
    with pytest.raises(temporal.TemporalCaptureRefused):
        temporal.verify_temporal_stream(_raw([_event(0)]), expected_n=1,
                                        approved_http_parallel=parallel)


def _response_event(model="frozen-original-model.gguf"):
    event = _event(0)
    payload = {"model": model, "choices": [{"message": {"content": "42"},
                                               "finish_reason": "stop"}]}
    raw = json.dumps(payload).encode()
    event["response_b64"] = base64.b64encode(raw).decode()
    event["response_sha256"] = hashlib.sha256(raw).hexdigest()
    return event


def _model_check(event):
    return temporal.verify_response_model_claims(
        _raw([event]), expected_n=1,
        expected_alias="frozen-original-model.gguf",
        expected_gguf_sha256="d" * 64,
    )


def test_response_model_claim_matches_but_does_not_authenticate_model():
    result = _model_check(_response_event())
    assert result["response_model_claims_consistent_only"] is True
    assert result["native_http_lineage_authenticated"] is False
    assert result["status"] == "BLOCKED" and result["chosen"] is None


def test_response_model_claims_also_enforce_source_declared_http_parallelism():
    events = [_response_event() for _ in range(5)]
    for i, event in enumerate(events):
        event.update({"id": f"gsm8k-{i}", "capture_seq": i,
                      "request_id": f"{i + 1:032x}",
                      "start_monotonic_ns": i + 1,
                      "end_monotonic_ns": i + 100})
    raw = _raw(events)
    with pytest.raises(temporal.TemporalCaptureRefused, match="HTTP overlap"):
        temporal.verify_response_model_claims(
            raw, expected_n=5, expected_alias="frozen-original-model.gguf",
            expected_gguf_sha256="d" * 64, approved_http_parallel=4,
        )
    result = temporal.verify_response_model_claims(
        raw, expected_n=5, expected_alias="frozen-original-model.gguf",
        expected_gguf_sha256="d" * 64, approved_http_parallel=5,
    )
    assert result["http_parallel_limit_checked"] is True
    assert result["native_http_lineage_authenticated"] is False


@pytest.mark.parametrize("mutation", [
    "foreign_model", "missing_model", "nonstring_model", "different_event_hash",
    "altered_response_without_hash", "bad_base64", "bad_model_source_hash",
    "duplicate_model_json", "nonfinite_json", "oversized_response",
])
def test_native_response_model_claim_adversarial_mutations_refused(mutation):
    event = _response_event()
    if mutation in ("foreign_model", "missing_model", "nonstring_model"):
        payload = json.loads(base64.b64decode(event["response_b64"]))
        if mutation == "foreign_model":
            payload["model"] = "forged-foreign-model.gguf"
        elif mutation == "missing_model":
            del payload["model"]
        else:
            payload["model"] = 1
        response = json.dumps(payload).encode()
        event["response_b64"] = base64.b64encode(response).decode()
        event["response_sha256"] = hashlib.sha256(response).hexdigest()
    elif mutation == "different_event_hash":
        event["response_sha256"] = "0" * 64
    elif mutation == "altered_response_without_hash":
        event["response_b64"] = base64.b64encode(b"{}").decode()
    elif mutation == "bad_base64":
        event["response_b64"] = "not/base64!"
    elif mutation == "bad_model_source_hash":
        event["gguf_sha256"] = "0" * 64
    elif mutation in ("duplicate_model_json", "nonfinite_json"):
        response = (b'{"model":"frozen-original-model.gguf",'
                    b'"model":"frozen-original-model.gguf"}'
                    if mutation == "duplicate_model_json"
                    else b'{"model":"frozen-original-model.gguf","x":NaN}')
        event["response_b64"] = base64.b64encode(response).decode()
        event["response_sha256"] = hashlib.sha256(response).hexdigest()
    elif mutation == "oversized_response":
        event["response_b64"] = "A" * (temporal.MAX_RESPONSE_BYTES * 2)
    with pytest.raises(temporal.TemporalCaptureRefused):
        _model_check(event)


def test_production_response_alias_and_collector_approvals_are_absent():
    assert temporal.APPROVED_RESPONSE_MODEL_ALIASES == {}
    assert temporal.APPROVED_COLLECTOR_SOURCE_SHA256 is None
    assert temporal.APPROVED_EVALUATOR_HTTP_PARALLEL == {}
    with pytest.raises(temporal.TemporalCaptureRefused, match="approvals absent"):
        temporal.verify_production_response_model_claims(
            _raw([_response_event()]), arm="reap128_v2_q4",
        )
    with pytest.raises(temporal.TemporalCaptureRefused, match="unregistered"):
        temporal.verify_production_response_model_claims(
            _raw([_response_event()]), arm="unregistered-model",
        )
