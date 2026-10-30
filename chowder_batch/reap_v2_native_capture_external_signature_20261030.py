"""Prospective external-signature consistency for original-v2 native captures.

An Ed25519 signature authenticates a *specific public key's* statement about
exact capture bytes. It does NOT establish who controlled that key, whether the
collector observed native HTTP, model weights, wall-clock time, or an original
frozen gate. Independently approved signer/process/source and raw-HTTP evidence
are separate prerequisites. No production approvals currently exist.

This module is CPU-only: no network, model execution, credential provisioning,
F: writes, or modifications to historical journals.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from chowder_batch import reap_original_gsm8k_frozen_scorer_20261029 as scorer
from chowder_batch import reap_v2_native_capture_temporal_contract_20261030 as temporal


SCHEMA = "chowder.reap-v2-native-capture-external-signature.v1"
DOMAIN = b"Chowder REAP128_v2 external capture checkpoint v1\x00"
MAX_CHECKPOINT_BYTES = 8192
MAX_CAPTURE_BYTES = temporal.MAX_BYTES
ZERO_SHA256 = "0" * 64
ARMS = temporal.FROZEN_ARM_SOURCE
_ROOT = Path(__file__).resolve().parent.parent
_CANONICAL = _ROOT / "docs/quals/reap-v2-original-gsm8k-canonical-items.json"
_EVAL_SOURCE = _ROOT / "chowder_batch/reap_eval.py"
_CANONICAL_RAW_SHA256 = "ca547069a7da83efb3d07d580046ccee2c1a78c8bc00c14b4e7a47885bec532b"
_EVAL_SOURCE_SHA256 = "9736714f21834a45c23599a50fa92ef3bebc6cf70967aca7e2329d26b4d73ee1"

# Source-owned independent authority is intentionally absent. In particular,
# do not derive these pins from a supplied checkpoint or self-reported PASS.
APPROVED_WITNESS_PUBLIC_KEY: dict[str, str] = {}
APPROVED_WITNESS_SOURCE_SHA256: dict[str, str] = {}
APPROVED_COLLECTOR_SOURCE_SHA256: dict[str, str] = {}
APPROVED_CAPTURE_SHA256: dict[str, str] = {}
APPROVED_ROWS_SHA256: dict[str, str] = {}
APPROVED_CHECKPOINT_SHA256: dict[str, str] = {}
APPROVED_EVAL_SESSION_ID: dict[str, str] = {}
APPROVED_COLLECTOR_INSTANCE_ID: dict[str, str] = {}
APPROVED_PREVIOUS_CHECKPOINT_SHA256: dict[str, str] = {}
APPROVED_SERVER_PID: dict[str, int] = {}
APPROVED_SERVER_PORT: dict[str, int] = {}

_SIGNED_FIELDS = frozenset({
    "schema", "arm", "capture_sha256", "capture_bytes", "capture_events",
    "rows_sha256", "rows_bytes",
    "gguf_sha256", "eval_session_id", "collector_instance_id",
    "collector_source_sha256", "witness_source_sha256", "witness_instance_id",
    "witness_counter", "previous_checkpoint_sha256", "server_pid", "server_port",
})


class ExternalSignatureRefused(ValueError):
    """Unapproved or inconsistent external capture checkpoint."""


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ExternalSignatureRefused(reason)


def _hex(value: object, n: int) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{" + str(n) + r"}", value) is not None


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ExternalSignatureRefused("duplicate checkpoint JSON key")
        obj[key] = value
    return obj


def _parse_checkpoint(raw: bytes) -> tuple[dict[str, Any], bytes]:
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_CHECKPOINT_BYTES,
             "checkpoint absent or oversized")
    try:
        obj = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
    except (UnicodeError, ValueError) as exc:
        raise ExternalSignatureRefused("checkpoint malformed") from exc
    _require(type(obj) is dict and set(obj) == _SIGNED_FIELDS | {"signature_b64"},
             "checkpoint schema fields missing or unexpected")
    try:
        _require(raw == _canonical(obj) + b"\n", "checkpoint bytes not canonical")
        unsigned = {key: obj[key] for key in _SIGNED_FIELDS}
        return obj, _canonical(unsigned)
    except (UnicodeError, ValueError) as exc:
        raise ExternalSignatureRefused("checkpoint contains invalid JSON scalar") from exc


def _original_request(prompt: str) -> bytes:
    """Exact original ``reap_eval._chat`` GSM8K request body serialization."""
    return json.dumps({
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0, "top_k": 1, "seed": 7, "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()


def _trigram(text: str) -> float:
    words = text.split()
    grams = {tuple(words[i:i + 3]) for i in range(len(words) - 2)}
    return round(len(grams) / (len(words) - 2), 4) if len(words) >= 3 else 1.0


def _decode_b64(value: object, *, limit: int) -> bytes:
    _require(type(value) is str and 0 < len(value) <= limit * 4 // 3 + 8,
             "native HTTP base64 absent or oversized")
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ExternalSignatureRefused("native HTTP base64 malformed") from exc
    _require(0 < len(raw) <= limit
             and base64.b64encode(raw).decode("ascii") == value,
             "native HTTP base64 noncanonical")
    return raw


def _verify_original_http_and_rows(capture: bytes, rows_raw: bytes, *, n: int,
                                   server_port: int) -> None:
    """Source-pinned independent per-item original HTTP and scored-row join."""
    _require(_CANONICAL.is_file() and not _CANONICAL.is_symlink()
             and _EVAL_SOURCE.is_file() and not _EVAL_SOURCE.is_symlink(),
             "original canonical prompt or frozen evaluator source missing")
    canonical_raw = _CANONICAL.read_bytes()
    eval_raw = _EVAL_SOURCE.read_bytes()
    _require(_sha(canonical_raw) == _CANONICAL_RAW_SHA256
             and _sha(eval_raw) == _EVAL_SOURCE_SHA256,
             "original canonical prompt or frozen evaluator source drift")
    scorer.require_original_scorer_source()
    try:
        items = json.loads(canonical_raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ExternalSignatureRefused("original canonical items malformed") from exc
    _require(type(items) is list and len(items) == 1319,
             "original canonical GSM8K population invalid")
    _require(type(rows_raw) is bytes and 0 < len(rows_raw) <= MAX_CAPTURE_BYTES,
             "original scored row bytes missing or oversized")
    row_lines = rows_raw.splitlines(keepends=True)
    _require(len(row_lines) == n, "original scored rows incomplete")
    rows: dict[str, dict] = {}
    for line in row_lines:
        _require(line.endswith(b"\n") and b"\r" not in line
                 and len(line) <= 1024 * 1024,
                 "original scored row framing invalid")
        try:
            row = temporal._decode(line)
        except temporal.TemporalCaptureRefused as exc:
            raise ExternalSignatureRefused("original scored row malformed") from exc
        task_id = row.get("id")
        _require(type(task_id) is str and task_id not in rows,
                 "original scored row duplicated or missing ID")
        rows[task_id] = row
    _require(set(rows) == {f"gsm8k-{i}" for i in range(n)},
             "original scored row task population mismatch")
    for line in capture.splitlines(keepends=True):
        try:
            event = temporal._decode(line)
        except temporal.TemporalCaptureRefused as exc:
            raise ExternalSignatureRefused("original capture event malformed") from exc
        task_id = event["id"]  # temporal already checked exact canonical ID set
        item = items[int(task_id.removeprefix("gsm8k-"))]
        _require(type(item) is dict and item.get("id") == task_id
                 and type(item.get("prompt")) is str
                 and type(item.get("expected")) is str
                 and event.get("id") == task_id,
                 "original canonical prompt/capture order mismatch")
        request = _decode_b64(event.get("request_b64"), limit=128 * 1024)
        response = _decode_b64(event.get("response_b64"), limit=512 * 1024)
        _require(event.get("method") == "POST"
                 and event.get("url") == f"http://127.0.0.1:{server_port}/v1/chat/completions"
                 and type(event.get("status_code")) is int
                 and event["status_code"] == 200
                 and event.get("request_sha256") == _sha(request)
                 and event.get("response_sha256") == _sha(response)
                 and request == _original_request(item["prompt"]),
                 "signed HTTP request or native response bytes not original")
        try:
            payload = temporal._decode(response)
        except temporal.TemporalCaptureRefused as exc:
            raise ExternalSignatureRefused("original native response malformed") from exc
        choices = payload.get("choices")
        _require(type(choices) is list and len(choices) >= 1
                 and type(choices[0]) is dict,
                 "original native completion choices absent")
        choice = choices[0]
        message = choice.get("message")
        _require(type(message) is dict
                 and (message.get("content") is None
                      or type(message.get("content")) is str)
                 and (choice.get("finish_reason") is None
                      or type(choice.get("finish_reason")) is str),
                 "original native completion content invalid")
        content = message.get("content") or ""
        usage = payload.get("usage", {})
        timings = payload.get("timings") or {}
        _require(type(usage) is dict and type(timings) is dict,
                 "original native usage/timings malformed")
        tokens = usage.get("completion_tokens")
        speed = timings.get("predicted_per_second")
        _require(type(tokens) is int and tokens >= 0
                 and (speed is None or (type(speed) in (float, int)
                                        and math.isfinite(speed) and speed >= 0)),
                 "original native token/timing data invalid")
        row = rows[task_id]
        terminated = choice.get("finish_reason") == "stop"
        _require(set(row) == {"id", "correct", "terminated", "trigram_ratio",
                                "completion_tokens", "seconds", "predicted_per_second", "text"}
                 and row["id"] == task_id
                 and type(row["text"]) is str and row["text"] == content
                 and type(row["terminated"]) is bool and row["terminated"] == terminated
                 and type(row["correct"]) is bool
                 and type(row["completion_tokens"]) is int
                 and row["completion_tokens"] == tokens
                 and (row["predicted_per_second"] is None
                      or type(row["predicted_per_second"]) in (float, int))
                 and row["predicted_per_second"] == speed
                 and type(row["trigram_ratio"]) in (float, int)
                 and row["trigram_ratio"] == _trigram(content)
                 and type(row["seconds"]) in (float, int)
                 and math.isfinite(row["seconds"]) and row["seconds"] >= 0
                 and row["seconds"] == round(row["seconds"], 2)
                 and row["correct"] is scorer.score_original_gsm8k(
                     content, item["expected"], terminated=terminated)
                 and event.get("scored_row_sha256") == _sha(_canonical(row)),
                 "original native completion differs from independently graded row")


def verify_signed_capture(
    capture: bytes, checkpoint: bytes, rows_raw: bytes, *, arm: str,
    public_key_hex: str, approved_rows_sha256: str,
    approved_witness_source_sha256: str, approved_collector_source_sha256: str,
    expected_eval_session_id: str, expected_collector_instance_id: str,
    expected_previous_checkpoint_sha256: str, expected_alias: str,
    approved_http_parallel: int, expected_server_pid: int,
    expected_server_port: int,
) -> dict[str, Any]:
    """Pure test/review helper; caller-supplied keys NEVER approve production."""
    _require(type(arm) is str and arm in ARMS, "unregistered original arm")
    _require(_hex(public_key_hex, 64)
             and _hex(approved_rows_sha256, 64)
             and _hex(approved_witness_source_sha256, 64)
             and _hex(approved_collector_source_sha256, 64)
             and _hex(expected_eval_session_id, 32)
             and _hex(expected_collector_instance_id, 32)
             and _hex(expected_previous_checkpoint_sha256, 64)
             and type(expected_server_pid) is int and expected_server_pid > 0
             and type(expected_server_port) is int
             and 0 < expected_server_port < 65536
             and type(approved_http_parallel) is int
             and 1 <= approved_http_parallel <= temporal.MAX_DECLARED_HTTP_PARALLEL,
             "missing independently specified source/owner identity")
    _require(type(capture) is bytes and 0 < len(capture) <= MAX_CAPTURE_BYTES,
             "invalid original capture bytes")
    obj, signed_bytes = _parse_checkpoint(checkpoint)
    n, model_sha = ARMS[arm]
    _require(obj["schema"] == SCHEMA and obj["arm"] == arm
             and obj["capture_sha256"] == _sha(capture)
             and type(rows_raw) is bytes and obj["rows_sha256"] == _sha(rows_raw)
             and obj["rows_sha256"] == approved_rows_sha256
             and type(obj["rows_bytes"]) is int
             and obj["rows_bytes"] == len(rows_raw)
             and type(obj["capture_bytes"]) is int
             and obj["capture_bytes"] == len(capture)
             and type(obj["capture_events"]) is int
             and obj["capture_events"] == n
             and obj["gguf_sha256"] == model_sha
             and obj["eval_session_id"] == expected_eval_session_id
             and obj["collector_instance_id"] == expected_collector_instance_id
             and obj["collector_source_sha256"] == approved_collector_source_sha256
             and obj["witness_source_sha256"] == approved_witness_source_sha256
             and type(obj["server_pid"]) is int
             and obj["server_pid"] == expected_server_pid
             and type(obj["server_port"]) is int
             and obj["server_port"] == expected_server_port
             and _hex(obj["witness_instance_id"], 32)
             and type(obj["witness_counter"]) is int
             and 0 < obj["witness_counter"] <= (1 << 63) - 1
             and obj["previous_checkpoint_sha256"] == expected_previous_checkpoint_sha256,
             "signed checkpoint contradicts independently specified capture/owner")
    signature_b64 = obj["signature_b64"]
    _require(type(signature_b64) is str and len(signature_b64) == 88,
             "missing or malformed Ed25519 signature")
    # Optional verification dependency: no new default runtime dependency or
    # silent fallback to an unauthenticated hash comparison.
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:
        raise ExternalSignatureRefused("Ed25519 verification dependency unavailable") from exc
    try:
        signature = base64.b64decode(signature_b64, validate=True)
        _require(len(signature) == 64
                 and base64.b64encode(signature).decode("ascii") == signature_b64,
                 "noncanonical Ed25519 signature")
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex)).verify(
            signature, DOMAIN + signed_bytes)
    except (binascii.Error, ValueError, InvalidSignature) as exc:
        raise ExternalSignatureRefused("external signature invalid") from exc
    # A signed blob may still be forged by its signer; separately check the
    # entire event population, ordering, model claim and declared concurrency.
    try:
        temporal.verify_response_model_claims(
            capture, expected_n=n, expected_alias=expected_alias,
            expected_gguf_sha256=model_sha,
            approved_http_parallel=approved_http_parallel,
        )
    except temporal.TemporalCaptureRefused as exc:
        raise ExternalSignatureRefused("signed native capture fails temporal/model consistency") from exc
    # The temporal validator requires consistency *within* the capture; it
    # cannot infer that its owner equals the independently expected owner.
    # Bind every event's owner to the signed and externally specified tuple.
    for line in capture.splitlines(keepends=True):
        try:
            event = temporal._decode(line)
        except temporal.TemporalCaptureRefused as exc:
            raise ExternalSignatureRefused("signed event JSON invalid") from exc
        _require(event.get("eval_session_id") == expected_eval_session_id
                 and event.get("collector_instance_id") == expected_collector_instance_id
                 and event.get("collector_source_sha256") == approved_collector_source_sha256
                 and type(event.get("server_pid")) is int
                 and event["server_pid"] == expected_server_pid
                 and type(event.get("server_port")) is int
                 and event["server_port"] == expected_server_port,
                 "signed capture owner contradicts independent serving owner")
    _verify_original_http_and_rows(capture, rows_raw, n=n,
                                   server_port=expected_server_port)
    return {
        "schema": SCHEMA, "arm": arm, "events": n,
        "capture_sha256": _sha(capture), "checkpoint_sha256": _sha(checkpoint),
        "signature_matches_supplied_key": True,
        "signer_independently_authenticated": False,
        "native_http_lineage_authenticated": False,
        "frozen_gate_approved": False, "status": "BLOCKED",
        "chosen": None, "release_approved": False,
    }


def verify_production_signed_capture(capture: bytes, checkpoint: bytes,
                                     rows_raw: bytes, *, arm: str) -> dict[str, Any]:
    """Production gate is deliberately closed pending external approvals."""
    _require(type(arm) is str and arm in ARMS, "unregistered original arm")
    pins = (
        APPROVED_WITNESS_PUBLIC_KEY, APPROVED_WITNESS_SOURCE_SHA256,
        APPROVED_COLLECTOR_SOURCE_SHA256, APPROVED_CAPTURE_SHA256,
        APPROVED_ROWS_SHA256,
        APPROVED_CHECKPOINT_SHA256, APPROVED_EVAL_SESSION_ID,
        APPROVED_COLLECTOR_INSTANCE_ID, APPROVED_PREVIOUS_CHECKPOINT_SHA256,
        APPROVED_SERVER_PID, APPROVED_SERVER_PORT,
        temporal.APPROVED_RESPONSE_MODEL_ALIASES,
        temporal.APPROVED_EVALUATOR_HTTP_PARALLEL,
    )
    _require(all(arm in mapping for mapping in pins),
             "independent witness/collector/capture/model approvals absent")
    _require(type(capture) is bytes and type(checkpoint) is bytes
             and type(rows_raw) is bytes
             and _sha(capture) == APPROVED_CAPTURE_SHA256[arm]
             and _sha(rows_raw) == APPROVED_ROWS_SHA256[arm]
             and _sha(checkpoint) == APPROVED_CHECKPOINT_SHA256[arm],
             "original capture/checkpoint differs from source-approved bytes")
    return verify_signed_capture(
        capture, checkpoint, rows_raw, arm=arm,
        approved_rows_sha256=APPROVED_ROWS_SHA256[arm],
        public_key_hex=APPROVED_WITNESS_PUBLIC_KEY[arm],
        approved_witness_source_sha256=APPROVED_WITNESS_SOURCE_SHA256[arm],
        approved_collector_source_sha256=APPROVED_COLLECTOR_SOURCE_SHA256[arm],
        expected_eval_session_id=APPROVED_EVAL_SESSION_ID[arm],
        expected_collector_instance_id=APPROVED_COLLECTOR_INSTANCE_ID[arm],
        expected_previous_checkpoint_sha256=APPROVED_PREVIOUS_CHECKPOINT_SHA256[arm],
        expected_alias=temporal.APPROVED_RESPONSE_MODEL_ALIASES[arm],
        approved_http_parallel=temporal.APPROVED_EVALUATOR_HTTP_PARALLEL[arm],
        expected_server_pid=APPROVED_SERVER_PID[arm],
        expected_server_port=APPROVED_SERVER_PORT[arm],
    )
