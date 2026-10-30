"""CPU-only adversarial tests with ephemeral, in-memory test keys and captures."""

from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography")
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from chowder_batch import reap_v2_native_capture_external_signature_20261030 as sig
from chowder_batch import reap_v2_native_capture_temporal_contract_20261030 as temporal


ARM = "reap128_v2_q4"
N, MODEL = sig.ARMS[ARM]
SESSION = "a" * 32
INSTANCE = "b" * 32
COLLECTOR = "c" * 64
WITNESS = "d" * 64
ALIAS = "source-approved-test-only-original-q4"
ITEMS = json.loads((Path(__file__).resolve().parent.parent /
                    "docs/quals/reap-v2-original-gsm8k-canonical-items.json")
                   .read_text(encoding="utf-8"))[:N]
ROWS = {
    item["id"]: {
        "id": item["id"], "correct": True, "terminated": True,
        "trigram_ratio": sig._trigram(item["expected"]),
        "completion_tokens": 3, "seconds": 0.1,
        "predicted_per_second": 2.5, "text": item["expected"],
    }
    for item in ITEMS
}
ROWS_RAW = b"".join(sig._canonical(ROWS[item["id"]]) + b"\n" for item in ITEMS)


def _event(index: int) -> dict:
    item = ITEMS[index]
    request = sig._original_request(item["prompt"])
    response = json.dumps({
        "model": ALIAS,
        "choices": [{"message": {"content": item["expected"]},
                     "finish_reason": "stop"}],
        "usage": {"completion_tokens": 3},
        "timings": {"predicted_per_second": 2.5},
    }).encode()
    return {
        "schema": temporal.EVENT_SCHEMA, "id": f"gsm8k-{index}",
        "capture_seq": index, "request_id": f"{index + 1:032x}",
        "collector_instance_id": INSTANCE, "eval_session_id": SESSION,
        "collector_source_sha256": COLLECTOR, "gguf_sha256": MODEL,
        "server_pid": 12345, "server_port": 8890,
        "method": "POST", "url": "http://127.0.0.1:8890/v1/chat/completions",
        "status_code": 200,
        "start_monotonic_ns": 1_000_000 + index * 1000,
        "end_monotonic_ns": 1_000_100 + index * 1000,
        "request_b64": base64.b64encode(request).decode("ascii"),
        "request_sha256": hashlib.sha256(request).hexdigest(),
        "response_b64": base64.b64encode(response).decode("ascii"),
        "response_sha256": hashlib.sha256(response).hexdigest(),
        "scored_row_sha256": sig._sha(sig._canonical(ROWS[item["id"]])),
    }


def _raw(events: list[dict]) -> bytes:
    return b"".join(sig._canonical(event) + b"\n" for event in events)


def _checkpoint(private: Ed25519PrivateKey, capture: bytes, *,
                rows_raw: bytes = ROWS_RAW, changes: dict | None = None) -> bytes:
    unsigned = {
        "schema": sig.SCHEMA, "arm": ARM,
        "capture_sha256": sig._sha(capture), "capture_bytes": len(capture),
        "rows_sha256": sig._sha(rows_raw), "rows_bytes": len(rows_raw),
        "capture_events": N, "gguf_sha256": MODEL,
        "eval_session_id": SESSION, "collector_instance_id": INSTANCE,
        "collector_source_sha256": COLLECTOR,
        "witness_source_sha256": WITNESS,
        "witness_instance_id": "e" * 32,
        "witness_counter": 1,
        "previous_checkpoint_sha256": sig.ZERO_SHA256,
        "server_pid": 12345, "server_port": 8890,
    }
    unsigned.update(changes or {})
    signature = private.sign(sig.DOMAIN + sig._canonical(unsigned))
    return sig._canonical({**unsigned, "signature_b64": base64.b64encode(signature).decode()}) + b"\n"


@pytest.fixture(scope="module")
def specimen():
    private = Ed25519PrivateKey.generate()  # test-only, never persisted
    public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    events = [_event(i) for i in range(N)]
    capture = _raw(events)
    checkpoint = _checkpoint(private, capture)
    return private, public, events, capture, checkpoint


def _verify(capture: bytes, checkpoint: bytes, public: str,
            rows_raw: bytes = ROWS_RAW, **kwargs):
    args = dict(
        arm=ARM, public_key_hex=public,
        approved_rows_sha256=sig._sha(ROWS_RAW),
        approved_witness_source_sha256=WITNESS,
        approved_collector_source_sha256=COLLECTOR,
        expected_eval_session_id=SESSION,
        expected_collector_instance_id=INSTANCE,
        expected_previous_checkpoint_sha256=sig.ZERO_SHA256,
        expected_alias=ALIAS, approved_http_parallel=4,
        expected_server_pid=12345, expected_server_port=8890,
    )
    args.update(kwargs)
    return sig.verify_signed_capture(capture, checkpoint, rows_raw, **args)


def test_valid_signature_is_still_blocked_and_not_native_attestation(specimen):
    _, public, _, capture, checkpoint = specimen
    result = _verify(capture, checkpoint, public)
    assert result["events"] == 200
    assert result["signature_matches_supplied_key"] is True
    assert result["signer_independently_authenticated"] is False
    assert result["native_http_lineage_authenticated"] is False
    assert result["status"] == "BLOCKED"
    assert result["chosen"] is None and result["release_approved"] is False


def test_production_refuses_without_independent_source_approvals(specimen):
    _, _, _, capture, checkpoint = specimen
    assert sig.APPROVED_WITNESS_PUBLIC_KEY == {}
    assert sig.APPROVED_CAPTURE_SHA256 == {}
    with pytest.raises(sig.ExternalSignatureRefused, match="approvals absent"):
        sig.verify_production_signed_capture(capture, checkpoint, ROWS_RAW, arm=ARM)


@pytest.mark.parametrize("mutation", [
    "flip_capture", "truncate_capture", "alter_signature", "wrong_public_key",
    "wrong_witness_source", "wrong_collector_source", "wrong_session",
    "wrong_instance", "wrong_previous", "wrong_alias", "wrong_model",
    "bad_parallel", "wrong_arm", "empty_capture", "wrong_arm_type",
    "wrong_server_pid", "wrong_server_port", "missing_parallel", "bool_parallel",
    "wrong_rows_sha",
])
def test_mismatched_signature_or_external_authority_refused(specimen, mutation):
    private, public, _, capture, checkpoint = specimen
    args = {}
    if mutation == "flip_capture":
        capture = capture[:-2] + b"x\n"
    elif mutation == "truncate_capture":
        capture = capture[:-1]
    elif mutation == "alter_signature":
        obj = json.loads(checkpoint)
        obj["signature_b64"] = base64.b64encode(b"\0" * 64).decode()
        checkpoint = sig._canonical(obj) + b"\n"
    elif mutation == "wrong_public_key":
        public = Ed25519PrivateKey.generate().public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw).hex()
    elif mutation == "wrong_witness_source":
        args["approved_witness_source_sha256"] = "f" * 64
    elif mutation == "wrong_collector_source":
        args["approved_collector_source_sha256"] = "f" * 64
    elif mutation == "wrong_session":
        args["expected_eval_session_id"] = "f" * 32
    elif mutation == "wrong_instance":
        args["expected_collector_instance_id"] = "f" * 32
    elif mutation == "wrong_previous":
        args["expected_previous_checkpoint_sha256"] = "f" * 64
    elif mutation == "wrong_alias":
        args["expected_alias"] = "foreign-alias"
    elif mutation == "wrong_model":
        checkpoint = _checkpoint(private, capture, changes={"gguf_sha256": "f" * 64})
    elif mutation == "bad_parallel":
        args["approved_http_parallel"] = 0
    elif mutation == "missing_parallel":
        args["approved_http_parallel"] = None
    elif mutation == "bool_parallel":
        args["approved_http_parallel"] = True
    elif mutation == "wrong_arm":
        args["arm"] = "dense_v2_q4"
    elif mutation == "empty_capture":
        capture = b""
    elif mutation == "wrong_arm_type":
        args["arm"] = True
    elif mutation == "wrong_server_pid":
        args["expected_server_pid"] = 54321
    elif mutation == "wrong_server_port":
        args["expected_server_port"] = 8892
    elif mutation == "wrong_rows_sha":
        args["approved_rows_sha256"] = "f" * 64
    with pytest.raises(sig.ExternalSignatureRefused):
        _verify(capture, checkpoint, public, **args)


@pytest.mark.parametrize("change", [
    {"capture_bytes": 0}, {"capture_events": 199},
    {"witness_counter": 0}, {"witness_counter": True},
    {"witness_instance_id": "bad"},
    {"previous_checkpoint_sha256": "f" * 64},
    {"collector_instance_id": "f" * 32},
    {"collector_source_sha256": "f" * 64},
    {"eval_session_id": "f" * 32},
    {"server_pid": 54321}, {"server_port": 8892},
    {"rows_sha256": "f" * 64}, {"rows_bytes": 1},
])
def test_re_signed_forged_metadata_cannot_override_independent_identity(specimen, change):
    private, public, _, capture, _ = specimen
    with pytest.raises(sig.ExternalSignatureRefused):
        _verify(capture, _checkpoint(private, capture, changes=change), public)


def test_re_signed_reordered_capture_rejected_by_temporal_contract(specimen):
    private, public, events, _, _ = specimen
    reordered = deepcopy(events)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    capture = _raw(reordered)
    with pytest.raises(sig.ExternalSignatureRefused, match="temporal/model"):
        _verify(capture, _checkpoint(private, capture), public)


def test_re_signed_foreign_model_claim_rejected(specimen):
    private, public, events, _, _ = specimen
    forged = deepcopy(events)
    response = json.dumps({"model": "foreign-model", "choices": []}).encode()
    forged[0]["response_b64"] = base64.b64encode(response).decode()
    forged[0]["response_sha256"] = hashlib.sha256(response).hexdigest()
    capture = _raw(forged)
    with pytest.raises(sig.ExternalSignatureRefused, match="temporal/model"):
        _verify(capture, _checkpoint(private, capture), public)


def test_re_signed_excess_concurrency_rejected(specimen):
    private, public, events, _, _ = specimen
    forged = deepcopy(events)
    for i in range(5):
        forged[i]["start_monotonic_ns"] = 1_000_000 + i
        forged[i]["end_monotonic_ns"] = 1_100_000 + i
    capture = _raw(forged)
    with pytest.raises(sig.ExternalSignatureRefused, match="temporal/model"):
        _verify(capture, _checkpoint(private, capture), public)


def test_concurrent_out_of_canonical_task_order_is_permitted(specimen):
    private, public, events, _, _ = specimen
    reordered = deepcopy(events)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    # Capture order is REQUEST START order, not the canonical task index.
    for seq, event in enumerate(reordered):
        event["capture_seq"] = seq
        event["start_monotonic_ns"] = 1_000_000 + seq * 1000
        event["end_monotonic_ns"] = 1_000_100 + seq * 1000
    capture = _raw(reordered)
    assert _verify(capture, _checkpoint(private, capture), public)["events"] == N


def _replace_response(event: dict, response: bytes) -> None:
    event["response_b64"] = base64.b64encode(response).decode()
    event["response_sha256"] = sig._sha(response)


@pytest.mark.parametrize("mutation", [
    "missing_request", "bad_request_base64", "wrong_request_hash",
    "foreign_prompt", "foreign_url", "wrong_method", "http_error",
    "response_without_choices", "response_without_usage", "response_text_changed",
    "response_duplicate_json_key", "missing_row_binding",
])
def test_re_signed_capture_with_invalid_original_http_refused(specimen, mutation):
    private, public, events, _, _ = specimen
    forged = deepcopy(events)
    event = forged[0]
    if mutation == "missing_request":
        del event["request_b64"]
    elif mutation == "bad_request_base64":
        event["request_b64"] = "###"
    elif mutation == "wrong_request_hash":
        event["request_sha256"] = "f" * 64
    elif mutation == "foreign_prompt":
        raw = sig._original_request("forged foreign benchmark prompt")
        event["request_b64"] = base64.b64encode(raw).decode()
        event["request_sha256"] = sig._sha(raw)
    elif mutation == "foreign_url":
        event["url"] = "http://127.0.0.1:8892/v1/chat/completions"
    elif mutation == "wrong_method":
        event["method"] = "GET"
    elif mutation == "http_error":
        event["status_code"] = 500
    elif mutation == "response_without_choices":
        _replace_response(event, json.dumps({"model": ALIAS}).encode())
    elif mutation == "response_without_usage":
        payload = json.loads(base64.b64decode(event["response_b64"]))
        del payload["usage"]
        _replace_response(event, json.dumps(payload).encode())
    elif mutation == "response_text_changed":
        payload = json.loads(base64.b64decode(event["response_b64"]))
        payload["choices"][0]["message"]["content"] = "foreign response text"
        _replace_response(event, json.dumps(payload).encode())
    elif mutation == "response_duplicate_json_key":
        _replace_response(event, b'{"model":"' + ALIAS.encode()
                          + b'","model":"' + ALIAS.encode() + b'"}')
    elif mutation == "missing_row_binding":
        del event["scored_row_sha256"]
    capture = _raw(forged)
    with pytest.raises(sig.ExternalSignatureRefused):
        _verify(capture, _checkpoint(private, capture), public)


@pytest.mark.parametrize("mutation", ["forged_correct", "wrong_text", "duplicate_id", "wrong_row_hash"])
def test_re_signed_forged_scored_rows_refused_even_with_updated_rows_digest(
    specimen, mutation,
):
    private, public, events, _, _ = specimen
    forged = deepcopy(ROWS)
    if mutation == "forged_correct":
        forged["gsm8k-0"]["correct"] = False
    elif mutation == "wrong_text":
        forged["gsm8k-0"]["text"] = "foreign response text"
    elif mutation == "duplicate_id":
        forged["gsm8k-1"]["id"] = "gsm8k-0"
    elif mutation == "wrong_row_hash":
        forged["gsm8k-0"]["seconds"] = 999.99
    rows_raw = b"".join(sig._canonical(forged[item["id"]]) + b"\n" for item in ITEMS)
    capture = _raw(events)
    # Even a correctly re-signed forged row file with a newly declared hash
    # cannot override independent original HTTP and per-item grader checks.
    with pytest.raises(sig.ExternalSignatureRefused):
        _verify(capture, _checkpoint(private, capture, rows_raw=rows_raw), public,
                rows_raw=rows_raw, approved_rows_sha256=sig._sha(rows_raw))


def test_missing_original_canonical_source_refused(specimen, monkeypatch):
    _, public, _, capture, checkpoint = specimen
    with monkeypatch.context() as patcher:
        patcher.setattr(sig, "_CANONICAL", sig._CANONICAL.with_name("absent-original-source.json"))
        with pytest.raises(sig.ExternalSignatureRefused, match="source missing"):
            _verify(capture, checkpoint, public)


@pytest.mark.parametrize("field,foreign", [
    ("eval_session_id", "f" * 32),
    ("collector_instance_id", "f" * 32),
    ("collector_source_sha256", "f" * 64),
    ("server_pid", 54321),
    ("server_port", 8892),
])
def test_re_signed_foreign_event_owner_refused_even_when_capture_internally_consistent(
    specimen, field, foreign,
):
    private, public, events, _, _ = specimen
    forged = deepcopy(events)
    for event in forged:
        event[field] = foreign
    capture = _raw(forged)
    with pytest.raises(sig.ExternalSignatureRefused, match="owner contradicts"):
        _verify(capture, _checkpoint(private, capture), public)


@pytest.mark.parametrize("mutation", ["duplicate_key", "extra_field", "missing_field", "pretty", "no_lf", "crlf", "nonfinite", "infinite_number", "unpaired_surrogate"])
def test_ambiguous_or_noncanonical_checkpoint_refused(specimen, mutation):
    _, public, _, capture, checkpoint = specimen
    obj = json.loads(checkpoint)
    if mutation == "duplicate_key":
        checkpoint = checkpoint.replace(b'"arm":', b'"arm":"reap128_v2_q4","arm":', 1)
    elif mutation == "extra_field":
        obj["self_approved"] = True
        checkpoint = sig._canonical(obj) + b"\n"
    elif mutation == "missing_field":
        del obj["capture_sha256"]
        checkpoint = sig._canonical(obj) + b"\n"
    elif mutation == "pretty":
        checkpoint = json.dumps(obj, indent=2).encode() + b"\n"
    elif mutation == "no_lf":
        checkpoint = checkpoint.rstrip(b"\n")
    elif mutation == "crlf":
        checkpoint = checkpoint.replace(b"\n", b"\r\n")
    elif mutation == "nonfinite":
        checkpoint = checkpoint.replace(b'"capture_bytes":', b'"capture_bytes":NaN,"extra":', 1)
    elif mutation == "infinite_number":
        checkpoint = checkpoint.replace(b'"capture_bytes":', b'"capture_bytes":1e999,"extra":', 1)
    elif mutation == "unpaired_surrogate":
        checkpoint = checkpoint.replace(b'"arm":', b'"arm":"\\ud800","extra":', 1)
    with pytest.raises(sig.ExternalSignatureRefused):
        _verify(capture, checkpoint, public)


def test_replay_wrong_external_checkpoint_chain_refused(specimen):
    _, public, _, capture, checkpoint = specimen
    with pytest.raises(sig.ExternalSignatureRefused):
        _verify(capture, checkpoint, public,
                expected_previous_checkpoint_sha256="1" * 64)


def test_test_only_signing_key_does_not_create_production_approval(specimen):
    _, public, _, capture, checkpoint = specimen
    assert _verify(capture, checkpoint, public)["status"] == "BLOCKED"
    with pytest.raises(sig.ExternalSignatureRefused):
        sig.verify_production_signed_capture(capture, checkpoint, ROWS_RAW, arm=ARM)


def test_optional_ed25519_dependency_fails_closed(specimen, monkeypatch):
    import builtins

    _, public, _, capture, checkpoint = specimen
    original_import = builtins.__import__

    def deny_cryptography(name, *args, **kwargs):
        if name.startswith("cryptography"):
            raise ImportError("test-only unavailable cryptography")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(builtins, "__import__", deny_cryptography)
        with pytest.raises(sig.ExternalSignatureRefused, match="dependency unavailable"):
            _verify(capture, checkpoint, public)
