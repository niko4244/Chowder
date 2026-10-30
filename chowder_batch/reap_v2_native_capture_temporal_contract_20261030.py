"""Prospective original-v2 native-capture *temporal consistency* contract.

The original evaluator defaults to four HTTP workers, but its ``--parallel``
argument is not frozen by the server's ``-np 4`` decode-slot setting. A future
collector MUST bind the actual evaluator parallelism independently. For that
collector, ``capture_seq`` MUST be assigned when a
request starts, and records MUST be written in that sequence (buffering out-of-
order completions). Monotonic timestamps must come from one process clock.

This is deliberately NOT evidence of actual native inference: timestamps and
request IDs in a self-reported JSONL can be fabricated. The raw-byte witness,
independently authenticated collector/process origin, immutable external
attestation, and every frozen scientific gate remain separate requirements.
Historical archives without this prospective contract cannot be upgraded.
"""

from __future__ import annotations

import base64
import binascii
import heapq
import hashlib
import json
import re
from typing import Any


SCHEMA = "chowder.reap-v2-native-capture-temporal-consistency.v1"
EVENT_SCHEMA = "chowder.reap-v2-original-native-http-event.v1"
MAX_EVENTS = 1319
MAX_DECLARED_HTTP_PARALLEL = 64
MAX_NS = (1 << 63) - 1
MAX_BYTES = 128 * 1024 * 1024
MAX_LINE_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 512 * 1024
FROZEN_ARM_SOURCE: dict[str, tuple[int, str]] = {
    "reap128_v2": (1319, "89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6"),
    "dense_v2": (1319, "2ff57a74a36939c5baabf40a0d3eb06d4424ecc94cf7d4ef04b3ac02059097c2"),
    "random128_v2": (1319, "601871e00403658708eb4eebf3b05f9dc54a93b326e31f2f87b48cb59f92241b"),
    "reap128_v2_q4": (200, "2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7"),
    "dense_v2_q4": (200, "2c6d9466129e8802839566b35728e2506aaf1e8705cd137009e2f08feb9dd049"),
}

# No independently approved llama-server response model alias or collector
# source exists for the missing frozen sessions. Do not learn approval values
# from the capture itself, a supplied serve.json, or synthetic tests.
APPROVED_RESPONSE_MODEL_ALIASES: dict[str, str] = {}
APPROVED_COLLECTOR_SOURCE_SHA256: str | None = None
APPROVED_EVALUATOR_HTTP_PARALLEL: dict[str, int] = {}


class TemporalCaptureRefused(ValueError):
    """Capture violates the prospective sequence/clock contract."""


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise TemporalCaptureRefused(reason)


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise TemporalCaptureRefused("duplicate JSON key")
        obj[key] = value
    return obj


def _decode(line: bytes) -> dict[str, Any]:
    try:
        result = json.loads(
            line.decode("utf-8"), object_pairs_hook=_unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")),
        )
    except (ValueError, UnicodeError) as exc:
        raise TemporalCaptureRefused("malformed temporal event JSON") from exc
    _require(type(result) is dict, "temporal event must be an object")
    return result


def verify_temporal_stream(raw: bytes, *, expected_n: int,
                           approved_http_parallel: int | None = None) -> dict[str, Any]:
    """Check bounded ordering/concurrency only; never authenticate provenance.

    The caller MUST independently check canonical prompts, actual HTTP bytes,
    row grading, collector identity, and source-controlled approvals. In
    particular, passing this function cannot make a synthetic capture official.
    """
    _require(type(expected_n) is int and 1 <= expected_n <= MAX_EVENTS,
             "invalid prospective event population")
    _require(approved_http_parallel is None or (
        type(approved_http_parallel) is int
        and 1 <= approved_http_parallel <= MAX_DECLARED_HTTP_PARALLEL),
        "invalid independently declared evaluator HTTP parallelism")
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_BYTES,
             "invalid prospective capture bytes")
    lines = raw.splitlines(keepends=True)
    _require(len(lines) == expected_n, "capture population incomplete or extra")

    first_start: int | None = None
    last_start: int | None = None
    last_end = 0
    peak = 0
    active_ends: list[int] = []
    ids: set[str] = set()
    requests: set[str] = set()
    canonical_ids = {f"gsm8k-{i}" for i in range(expected_n)}
    owner: tuple[Any, ...] | None = None

    for sequence, line in enumerate(lines):
        _require(line.endswith(b"\n") and b"\r" not in line
                 and not line.startswith(b"\xef\xbb\xbf")
                 and len(line) <= MAX_LINE_BYTES,
                 "capture framing not canonical")
        event = _decode(line)
        _require(event.get("schema") == EVENT_SCHEMA
                 and type(event.get("capture_seq")) is int
                 and event["capture_seq"] == sequence,
                 "capture sequence must equal append order")
        task = event.get("id")
        request = event.get("request_id")
        _require(type(task) is str and task in canonical_ids
                 and type(request) is str and len(request) == 32
                 and all(char in "0123456789abcdef" for char in request)
                 and task not in ids and request not in requests,
                 "duplicate or malformed task/request identity")
        ids.add(task)
        requests.add(request)
        start = event.get("start_monotonic_ns")
        end = event.get("end_monotonic_ns")
        _require(type(start) is int and type(end) is int
                 and 0 <= start < end <= MAX_NS,
                 "invalid monotonic request interval")
        _require(last_start is None or start >= last_start,
                 "request starts contradict assigned capture sequence")
        identity = (
            event.get("collector_instance_id"), event.get("eval_session_id"),
            event.get("collector_source_sha256"), event.get("gguf_sha256"),
            event.get("server_pid"), event.get("server_port"),
        )
        _require(all(value is not None for value in identity)
                 and (owner is None or identity == owner),
                 "mixed collector/server/process identity")
        owner = identity
        while active_ends and active_ends[0] <= start:
            heapq.heappop(active_ends)
        heapq.heappush(active_ends, end)
        peak = max(peak, len(active_ends))
        if approved_http_parallel is not None:
            _require(peak <= approved_http_parallel,
                     "HTTP overlap exceeds independently declared evaluator parallelism")
        if first_start is None:
            first_start = start
        last_start = start
        last_end = max(last_end, end)

    return {
        "schema": SCHEMA,
        "event_count": expected_n,
        "peak_simultaneous_requests": peak,
        "declared_evaluator_http_parallel": approved_http_parallel,
        "http_parallel_limit_checked": approved_http_parallel is not None,
        "observed_monotonic_span_ns": last_end - first_start,
        "sequence_and_clock_consistent_only": True,
        "raw_http_bytes_independently_verified": False,
        "collector_origin_authenticated": False,
        "native_http_lineage_authenticated": False,
        "frozen_gate_approved": False,
        "status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }


def verify_response_model_claims(
    raw: bytes, *, expected_n: int, expected_alias: str,
    expected_gguf_sha256: str, approved_http_parallel: int | None = None,
) -> dict[str, Any]:
    """Check explicit native response-model claims, NOT their truth.

    ``expected_alias`` must eventually come from independently reviewed
    original llama-server invocation/version evidence, never a captured event.
    This pure function exists for adversarial CPU testing; the production
    entrypoint below has no authorized alias/collector and always refuses.
    """
    _require(type(expected_alias) is str and 0 < len(expected_alias) <= 1024
             and type(expected_gguf_sha256) is str
             and re.fullmatch(r"[0-9a-f]{64}", expected_gguf_sha256) is not None,
             "independent expected native model identity absent")
    # Validate the same sequence/concurrency envelope before reading claims.
    temporal = verify_temporal_stream(
        raw, expected_n=expected_n,
        approved_http_parallel=approved_http_parallel,
    )
    for line in raw.splitlines(keepends=True):
        event = _decode(line)
        _require(event.get("gguf_sha256") == expected_gguf_sha256,
                 "event model hash contradicts source-approved original model")
        encoded = event.get("response_b64")
        _require(type(encoded) is str
                 and 0 < len(encoded) <= MAX_RESPONSE_BYTES * 4 // 3 + 8,
                 "missing or oversized native response")
        try:
            response = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise TemporalCaptureRefused("invalid native response base64") from exc
        _require(0 < len(response) <= MAX_RESPONSE_BYTES
                 and base64.b64encode(response).decode("ascii") == encoded
                 and event.get("response_sha256") == hashlib.sha256(response).hexdigest(),
                 "native response bytes/hash inconsistent")
        payload = _decode(response)
        _require(type(payload.get("model")) is str
                 and payload["model"] == expected_alias,
                 "native response model claim missing or contradicts original serving model")
    return {
        "schema": SCHEMA,
        "event_count": temporal["event_count"],
        "http_parallel_limit_checked": temporal["http_parallel_limit_checked"],
        "response_model_claims_consistent_only": True,
        "response_alias_source_independently_approved": False,
        "collector_origin_authenticated": False,
        "native_http_lineage_authenticated": False,
        "frozen_gate_approved": False,
        "status": "BLOCKED", "chosen": None, "release_approved": False,
    }


def verify_production_response_model_claims(raw: bytes, *, arm: str) -> dict[str, Any]:
    """Never admit a self-approved model alias into the frozen evidence path."""
    _require(type(arm) is str and arm in FROZEN_ARM_SOURCE,
             "unregistered frozen original model arm")
    expected_n, expected_gguf_sha256 = FROZEN_ARM_SOURCE[arm]
    alias = APPROVED_RESPONSE_MODEL_ALIASES.get(arm)
    parallel = APPROVED_EVALUATOR_HTTP_PARALLEL.get(arm)
    _require(type(alias) is str and alias
             and type(APPROVED_COLLECTOR_SOURCE_SHA256) is str
             and re.fullmatch(r"[0-9a-f]{64}", APPROVED_COLLECTOR_SOURCE_SHA256)
             is not None and type(parallel) is int
             and 1 <= parallel <= MAX_DECLARED_HTTP_PARALLEL,
             "independent collector and response-model alias approvals absent")
    result = verify_response_model_claims(
        raw, expected_n=expected_n, expected_alias=alias,
        expected_gguf_sha256=expected_gguf_sha256,
        approved_http_parallel=parallel,
    )
    for line in raw.splitlines(keepends=True):
        _require(_decode(line).get("collector_source_sha256")
                 == APPROVED_COLLECTOR_SOURCE_SHA256,
                 "native collector source differs from independently approved identity")
    return result
