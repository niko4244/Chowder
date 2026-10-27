"""Tests for the guard proxy: compaction, budget, runaway policy, arg repair."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from chowder_batch.reap_context_guard_proxy_20261026 import (
    ARGS_UNREPAIRABLE,
    RECORD_RESOLUTION_DIRECTIVE,
    RECORD_RESOLUTION_MARKER,
    apply_record_resolution,
    GEN_CEILING_TOKENS,
    GEN_FLOOR_TOKENS,
    PROMPT_BUDGET_TOKENS,
    RUNAWAY_REASON,
    compact_messages,
    estimate_tokens,
    generation_budget,
    is_runaway_error,
    prompt_tokens,
    repair_json_string_params,
    repair_tool_arguments,
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
        if UPSTREAM_BEHAVIOUR["mode"] == "toolcall_nested":
            arguments = json.dumps({
                "method": "POST", "url": "sales/notes",
                "params": "", "body": '{"text":"hello",}'})
            payload = json.dumps({
                "id": "x", "object": "chat.completion", "created": 0,
                "model": body.get("model", "m"),
                "choices": [{"index": 0, "finish_reason": "tool_calls",
                             "message": {"role": "assistant", "content": None,
                                         "tool_calls": [{"id": "call_1", "type": "function",
                                                          "function": {"name": "api_fetch",
                                                                       "arguments": arguments}}]}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if UPSTREAM_BEHAVIOUR["mode"] in ("toolcall_truncated", "toolcall_broken"):
            arguments = ("{\"query\":\"hello" if
                         UPSTREAM_BEHAVIOUR["mode"] == "toolcall_truncated" else "not json")
            payload = json.dumps({
                "id": "x", "object": "chat.completion", "created": 0,
                "model": body.get("model", "m"),
                "choices": [{"index": 0, "finish_reason": "tool_calls",
                             "message": {"role": "assistant", "content": None,
                                         "tool_calls": [{"id": "call_1", "type": "function",
                                                          "function": {"name": "search",
                                                                       "arguments": arguments}}]}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
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


def test_repair_leaves_valid_arguments_alone():
    repaired, reason = repair_tool_arguments('{"a": 1}')
    assert reason == "intact"
    assert json.loads(repaired) == {"a": 1}


def test_repair_turns_an_empty_argument_string_into_an_empty_object():
    # The measured phase-36 failure: the workspace API answers
    # `Invalid JSON: Expecting value: line 1 column 1 (char 0)` for "".
    assert repair_tool_arguments("") == ("{}", "empty_arguments")
    assert repair_tool_arguments("   ") == ("{}", "empty_arguments")


def test_repair_closes_a_truncated_argument_object():
    repaired, reason = repair_tool_arguments('{"query":"hello')
    assert reason == "closed_truncation"
    assert json.loads(repaired) == {"query": "hello"}


def test_repair_drops_a_trailing_comma():
    repaired, reason = repair_tool_arguments('{"a": 1,}')
    assert reason == "trailing_comma"
    assert json.loads(repaired) == {"a": 1}


def test_repair_unwraps_a_markdown_fence():
    repaired, reason = repair_tool_arguments('```json\n{"a":1}\n```')
    assert reason == "unfenced"
    assert json.loads(repaired) == {"a": 1}


def test_repair_refuses_non_objects():
    assert repair_tool_arguments("not json") == (None, "no_object")
    assert repair_tool_arguments("[1,2]")[0] is None
    assert repair_tool_arguments(None) == (None, "non_string")


def test_repair_fixes_malformed_nested_params_and_body():
    # The phase-36 shape: a valid outer object hiding malformed `params`/`body`.
    args = {"method": "POST", "url": "x", "params": "", "body": '{"text":"hi",}'}
    repaired, report = repair_json_string_params(args)
    assert json.loads(repaired["body"]) == {"text": "hi"}
    assert json.loads(repaired["params"]) == {}
    assert {r["key"] for r in report} == {"params", "body"}
    assert all(r["repaired"] for r in report)


def test_repair_ignores_plain_scalar_parameters():
    args = {"method": "GET", "url": "sales/notes", "body": "plain text"}
    repaired, report = repair_json_string_params(args)
    assert repaired == args
    assert report == []


def test_repair_reports_a_nested_body_it_cannot_fix():
    # Looks like JSON (so it is treated as an encoded document) but cannot be
    # made valid: it must be reported, never silently rewritten.
    args = {"body": '{"a": 1 2}'}
    repaired, report = repair_json_string_params(args)
    assert report == [{"key": "body", "reason": "unparsable", "repaired": False}]
    assert repaired["body"] == '{"a": 1 2}'


def test_repair_leaves_a_plain_text_body_alone():
    args = {"body": "not json at all"}
    repaired, report = repair_json_string_params(args)
    assert repaired == args
    assert report == []


def test_proxy_repairs_a_nested_body_before_dispatch(guard):
    port, receipt = guard
    UPSTREAM_BEHAVIOUR["mode"] = "toolcall_nested"
    status, body = _post(port, {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert status == 200
    sent = json.loads(body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert json.loads(sent["body"]) == {"text": "hello"}
    assert json.loads(sent["params"]) == {}
    events = [json.loads(line) for line in receipt.read_text("utf8").splitlines()]
    assert events[-1]["tool_arguments_repaired"] == 1


def test_proxy_repairs_a_truncated_tool_call_before_dispatch(guard):
    port, receipt = guard
    UPSTREAM_BEHAVIOUR["mode"] = "toolcall_truncated"
    status, body = _post(port, {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert status == 200
    call = body["choices"][0]["message"]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"query": "hello"}
    events = [json.loads(line) for line in receipt.read_text("utf8").splitlines()]
    assert events[-1]["tool_arguments_repaired"] == 1
    assert events[-1]["event"] == "forwarded"


def test_proxy_ends_the_turn_when_arguments_cannot_be_repaired(guard):
    port, receipt = guard
    UPSTREAM_BEHAVIOUR["mode"] = "toolcall_broken"
    status, body = _post(port, {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert status == 200
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "tool_calls" not in body["choices"][0]["message"]
    events = [json.loads(line) for line in receipt.read_text("utf8").splitlines()]
    assert events[-1]["event"] == ARGS_UNREPAIRABLE
    assert events[-1]["tool_arguments_unrepairable"] == 1


def test_record_resolution_is_injected_once_as_a_system_turn():
    prepared, injected = apply_record_resolution([{"role": "user", "content": "hi"}])
    assert injected is True
    assert prepared[0]["role"] == "system"
    assert RECORD_RESOLUTION_MARKER in prepared[0]["content"]
    assert prepared[1] == {"role": "user", "content": "hi"}


def test_record_resolution_is_idempotent():
    once, _ = apply_record_resolution([{"role": "user", "content": "hi"}])
    twice, injected = apply_record_resolution(once)
    assert injected is False
    assert twice == once
    assert sum(1 for m in twice
               if RECORD_RESOLUTION_MARKER in str(m.get("content"))) == 1


def test_record_resolution_demands_a_prior_read_and_a_named_id():
    assert "first perform a read" in RECORD_RESOLUTION_DIRECTIVE
    assert "Never write to an id" in RECORD_RESOLUTION_DIRECTIVE


def test_proxy_injects_record_resolution_only_when_armed(tmp_path: Path):
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    UPSTREAM_BEHAVIOUR["mode"] = "ok"
    try:
        for armed, expected in ((True, 1), (False, 0)):
            receipt = tmp_path / f"armed-{armed}.jsonl"
            server, _, thread = start_guard(
                upstream_port=upstream.server_address[1], listen_port=0,
                context=73728, receipt_path=receipt, record_resolution=armed)
            try:
                _post(server.server_address[1],
                      {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
            finally:
                stop_guard(server, thread)
            events = [json.loads(line) for line in receipt.read_text("utf8").splitlines()]
            assert events[-1]["record_resolution"] is armed
            assert events[-1]["record_resolution_injected"] == expected
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_proxy_compacts_before_forwarding(guard):
    port, receipt = guard
    UPSTREAM_BEHAVIOUR["mode"] = "ok"
    _post(port, {"model": "m", "messages": convo(10000, turns=12)})
    events = [json.loads(line) for line in receipt.read_text("utf8").splitlines()]
    assert events[-1]["prompt_tokens_after"] <= PROMPT_BUDGET_TOKENS
    assert events[-1]["prompt_tokens_before"] > PROMPT_BUDGET_TOKENS
    assert events[-1]["event"] == "forwarded"
