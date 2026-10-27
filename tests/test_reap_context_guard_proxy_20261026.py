"""Tests for the guard proxy: compaction, generation budget, runaway policy."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from chowder_batch.reap_context_guard_proxy_20261026 import (
    GEN_CEILING_TOKENS,
    GEN_FLOOR_TOKENS,
    PROMPT_BUDGET_TOKENS,
    RUNAWAY_REASON,
    compact_messages,
    estimate_tokens,
    generation_budget,
    is_runaway_error,
    prompt_tokens,
    runaway_completion,
    start_guard,
    stop_guard,
)


def convo(tool_len: int, turns: int = 6):
    messages = [
        {"role": "system", "content": "You are a workflow automation agent. " * 20},
        {"role": "user", "content": "Tag the Project Alpha emails for the PM, same as usual."},
    ]
    for i in range(turns):
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [json.dumps({"id": f"c{i}", "name": "api_fetch",
                                       "arguments": json.dumps({"url": "https://x/" + "a" * tool_len})})],
        })
        messages.append({"role": "tool", "tool_call_id": f"c{i}",
                         "content": json.dumps({"body": "b" * tool_len})})
    return messages


# --------------------------------------------------------------------------
# compaction
# --------------------------------------------------------------------------


def test_small_conversation_is_untouched():
    messages = convo(10, turns=1)
    out, stats = compact_messages(messages)
    assert stats["strategy"] == "unchanged"
    assert out == messages


def test_oversized_conversation_is_bounded():
    messages = convo(10000, turns=12)
    assert prompt_tokens(messages) > PROMPT_BUDGET_TOKENS
    out, stats = compact_messages(messages)
    assert prompt_tokens(out) <= PROMPT_BUDGET_TOKENS
    assert stats["prompt_tokens_after"] < stats["prompt_tokens_before"]
    assert stats["strategy"] in ("shrink_content", "compacted_history")


def test_compaction_keeps_system_first_user_and_recent_turns():
    messages = convo(10000, turns=12)
    out, _ = compact_messages(messages)
    assert out[0]["role"] == "system"
    assert out[1]["role"] == "user"
    # The newest tool result must survive verbatim, not be compacted away.
    assert "elided" not in (out[-1].get("content") or "")


def test_compaction_preserves_tool_call_ids():
    messages = convo(10000, turns=12)
    out, _ = compact_messages(messages)
    ids = {m["tool_call_id"] for m in out if m.get("tool_call_id")}
    # Structure stays valid: every surviving tool result still has its id.
    assert ids
    roles = [m["role"] for m in out]
    assert roles.index("tool") > 0


def test_marker_records_how_much_was_removed():
    messages = convo(10000, turns=12)
    out, _ = compact_messages(messages)
    joined = json.dumps(out)
    assert "elided" in joined or "compacted" in joined


# --------------------------------------------------------------------------
# generation budget
# --------------------------------------------------------------------------


def test_budget_never_exceeds_the_window():
    assert generation_budget(1_000, context=73728) == GEN_CEILING_TOKENS


def test_budget_shrinks_as_the_prompt_grows():
    small = generation_budget(1_000, context=73728)
    large = generation_budget(60_000, context=73728)
    assert large < small


def test_budget_refuses_when_no_room_left():
    assert generation_budget(73_700, context=73728) == 0


def test_budget_floor():
    # 73,728 - 65,000 - 512 = 8,216, above the floor but below the ceiling.
    assert generation_budget(65_000, context=73728) == 8_216
    assert generation_budget(70_000, context=73728) >= 0


def test_budget_below_floor_is_zero_not_negative():
    value = generation_budget(73_000, context=73728)
    assert value == 0


# --------------------------------------------------------------------------
# runaway policy
# --------------------------------------------------------------------------


def test_runaway_detection_matches_only_the_tool_json_error():
    body = b'{"error":{"message":"Failed to parse tool call arguments as JSON: parse error"}}'
    assert is_runaway_error(500, body) is True
    assert is_runaway_error(400, body) is False
    assert is_runaway_error(500, b'{"error":{"message":"other"}}') is False


def test_runaway_completion_is_a_valid_stop():
    payload = runaway_completion("model-x")
    assert payload["object"] == "chat.completion"
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]
    assert payload["model"] == "model-x"


# --------------------------------------------------------------------------
# live proxy round-trip
# --------------------------------------------------------------------------

UPSTREAM_BEHAVIOUR = {"mode": "ok"}


class _Upstream(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if UPSTREAM_BEHAVIOUR["mode"] == "runaway":
            payload = json.dumps({"error": {"message":
                "Failed to parse tool call arguments as JSON: missing closing quote"}}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        payload = json.dumps({
            "id": "x", "object": "chat.completion", "created": 0, "model": body.get("model", "m"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "max_tokens_seen": body.get("max_tokens"),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture()
def guard(tmp_path: Path):
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    receipt = tmp_path / "guard.jsonl"
    server, state, thread = start_guard(
        upstream_port=upstream.server_address[1],
        listen_port=0,
        context=73728,
        receipt_path=receipt,
    )
    port = server.server_address[1]
    try:
        yield port, receipt
    finally:
        stop_guard(server, thread)
        upstream.shutdown()
        upstream.server_close()


def _post(port: int, body: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, json.loads(response.read())


def test_proxy_clamps_max_tokens(guard):
    port, _ = guard
    UPSTREAM_BEHAVIOUR["mode"] = "ok"
    status, body = _post(port, {"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                "max_tokens": 400000})
    assert status == 200
    assert body["max_tokens_seen"] == GEN_CEILING_TOKENS


def test_proxy_converts_a_runaway_500_into_a_stop(guard):
    port, receipt = guard
    UPSTREAM_BEHAVIOUR["mode"] = "runaway"
    status, body = _post(port, {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert status == 200
    assert body["choices"][0]["finish_reason"] == "stop"
    events = [json.loads(line) for line in receipt.read_text("utf8").splitlines()]
    assert any(e["event"] == RUNAWAY_REASON for e in events)


def test_proxy_compacts_before_forwarding(guard):
    port, receipt = guard
    UPSTREAM_BEHAVIOUR["mode"] = "ok"
    _post(port, {"model": "m", "messages": convo(10000, turns=12)})
    events = [json.loads(line) for line in receipt.read_text("utf8").splitlines()]
    assert events[-1]["prompt_tokens_after"] <= PROMPT_BUDGET_TOKENS
    assert events[-1]["prompt_tokens_before"] > PROMPT_BUDGET_TOKENS
    assert events[-1]["event"] == "forwarded"
