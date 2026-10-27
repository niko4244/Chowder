"""Local guard proxy between the pinned AutomationBench evaluator and llama-server.

Why this exists
---------------
Phase 30 measured two failures that the evaluator cannot be patched to avoid
(it is a pinned, git-clean public source and must stay that way):

1. **Prompt growth.** Agentic episodes append tool results every turn; a wide
   sample eventually outgrows any fixed window.
2. **Degenerate turns.** One public task reproducibly emits a 54k-58k token
   single turn. llama-server cuts it at the context edge, the tool-call JSON is
   incomplete, and the server answers HTTP 500. The evaluator then retries the
   same unchanged conversation up to 40 times with backoff, which aborts the
   whole episode rather than the one task.

Both are addressable *outside* the evaluator, at the HTTP boundary:

- **Compaction** rewrites oversized tool results before forwarding, so a prompt
  cannot grow without bound.
- **A computed generation budget** sets ``max_tokens`` per request from the
  remaining window, so one turn cannot consume the context.
- **A pre-registered runaway policy** converts an unparseable-tool-call 500 into
  a normal "assistant stopped" completion. The episode then scores that task as
  failed and continues to the next one, instead of retrying a conversation that
  can never recover.

Everything is recorded to a receipt so the policy is auditable rather than
silent. This lane is NONCOMPARABLE to unguarded runs: it changes what the model
sees and how a runaway is handled.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# --- pre-registered policy (fixed before the run, not tuned on results) -----

PROMPT_BUDGET_TOKENS = 48_000
GEN_CEILING_TOKENS = 16_384
GEN_FLOOR_TOKENS = 2_048
CONTEXT_MARGIN_TOKENS = 512
KEEP_RECENT_MESSAGES = 12
TOOL_CONTENT_HEAD = 600
TOOL_CONTENT_TAIL = 200
CHARS_PER_TOKEN = 3.5
ELISION_MARK = " ...[elided {n} chars]... "

POLICY_NAME = "guard.v1"
RUNAWAY_REASON = "RUNAWAY_TURN_TERMINATED"

# Errors the server raises when an OpenAI tool-call argument is not valid JSON,
# which is what a truncated generation always produces.
TOOL_JSON_ERROR = "Failed to parse tool call arguments as JSON"


# --------------------------------------------------------------------------
# pure helpers
# --------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Conservative character-based estimate; never under-counts JSON badly."""
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN) + 1)


def message_tokens(message: Mapping[str, Any]) -> int:
    total = 0
    content = message.get("content")
    if isinstance(content, str):
        total += estimate_tokens(content)
    elif isinstance(content, list):
        total += estimate_tokens(json.dumps(content))
    for call in message.get("tool_calls") or []:
        total += estimate_tokens(json.dumps(call))
    return total


def prompt_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    return sum(message_tokens(m) for m in messages)


def _elide(text: str, head: int, tail: int) -> str:
    if len(text) <= head + tail:
        return text
    return text[:head] + ELISION_MARK.format(n=len(text) - head - tail) + text[-tail:]


def _shrink_message(message: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    """Truncate bulky text in place while preserving tool_call/result pairing."""
    copy = dict(message)
    removed = 0
    content = copy.get("content")
    if isinstance(content, str) and len(content) > TOOL_CONTENT_HEAD + TOOL_CONTENT_TAIL:
        shrunk = _elide(content, TOOL_CONTENT_HEAD, TOOL_CONTENT_TAIL)
        removed += len(content) - len(shrunk)
        copy["content"] = shrunk
    calls = copy.get("tool_calls")
    if isinstance(calls, list):
        new_calls = []
        for call in calls:
            if isinstance(call, str) and len(call) > TOOL_CONTENT_HEAD + TOOL_CONTENT_TAIL:
                shrunk = _elide(call, TOOL_CONTENT_HEAD, TOOL_CONTENT_TAIL)
                removed += len(call) - len(shrunk)
                new_calls.append(shrunk)
            else:
                new_calls.append(call)
        copy["tool_calls"] = new_calls
    return copy, removed


def compact_messages(
    messages: Sequence[Mapping[str, Any]], *, budget: int = PROMPT_BUDGET_TOKENS
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bound the prompt while keeping the system turn, the task, and recent work.

    Only content is shrunk; roles, ordering and tool-call ids are preserved so
    the conversation stays structurally valid for the server.
    """
    original = list(messages)
    before = prompt_tokens(original)
    elided_chars = 0
    if before <= budget:
        return [dict(m) for m in original], {
            "prompt_tokens_before": before,
            "prompt_tokens_after": before,
            "elided_chars": 0,
            "strategy": "unchanged",
        }

    # Pass 1: shrink bulky bodies in the older turns only. The most recent
    # turns stay verbatim -- they are the evidence the next step reasons over,
    # and truncating them would change behaviour, not just cost.
    protected_from = max(0, len(original) - KEEP_RECENT_MESSAGES)
    shrunk = []
    for index, message in enumerate(original):
        if index >= protected_from:
            shrunk.append(dict(message))
            continue
        copy, removed = _shrink_message(message)
        elided_chars += removed
        shrunk.append(copy)
    after = prompt_tokens(shrunk)
    if after <= budget:
        return shrunk, {
            "prompt_tokens_before": before,
            "prompt_tokens_after": after,
            "elided_chars": elided_chars,
            "strategy": "shrink_content",
        }

    # Pass 2: keep the system turn, the first user turn and the most recent
    # messages; older middle turns collapse to a marker of the same role.
    keep_head = [i for i, m in enumerate(shrunk) if m.get("role") in ("system", "developer")][:1]
    if not keep_head and shrunk:
        keep_head = [0]
    first_user = next((i for i, m in enumerate(shrunk) if m.get("role") == "user"), None)
    pinned = set(keep_head)
    if first_user is not None:
        pinned.add(first_user)
    tail_start = max(0, len(shrunk) - KEEP_RECENT_MESSAGES)
    result: list[dict[str, Any]] = []
    for index, message in enumerate(shrunk):
        if index in pinned or index >= tail_start:
            result.append(message)
            continue
        placeholder = {
            "role": message.get("role", "user"),
            "content": f"[compacted {estimate_tokens(json.dumps(message))} tokens of earlier turn]",
        }
        if message.get("tool_call_id"):
            placeholder["tool_call_id"] = message["tool_call_id"]
        result.append(placeholder)

    after = prompt_tokens(result)
    return result, {
        "prompt_tokens_before": before,
        "prompt_tokens_after": after,
        "elided_chars": elided_chars,
        "strategy": "compacted_history",
        "messages_before": len(original),
        "messages_after": len(result),
    }


def generation_budget(prompt: int, *, context: int) -> int:
    """Tokens this turn may generate without reaching the context edge."""
    available = context - prompt - CONTEXT_MARGIN_TOKENS
    if available < GEN_FLOOR_TOKENS:
        return 0
    return min(available, GEN_CEILING_TOKENS)


def is_runaway_error(status: int, body: bytes) -> bool:
    if status < 500:
        return False
    return TOOL_JSON_ERROR.encode() in body


def runaway_completion(model: str, *, reason: str = RUNAWAY_REASON) -> dict[str, Any]:
    """A valid ChatCompletion that ends the turn without calling a tool."""
    return {
        "id": f"guard-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": (
                    "The previous turn could not be completed: the generation "
                    "degenerated and its tool call was cut. Ending this turn "
                    f"under policy {POLICY_NAME} ({reason})."
                ),
            },
            "logprobs": None,
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# --------------------------------------------------------------------------
# proxy
# --------------------------------------------------------------------------


class GuardState:
    def __init__(self, *, upstream_port: int, context: int, receipt_path: Path) -> None:
        self.upstream = f"http://127.0.0.1:{upstream_port}"
        self.context = context
        self.receipt_path = receipt_path
        self.lock = threading.Lock()
        self.index = 0
        self.counters = {
            "requests": 0,
            "compacted": 0,
            "budget_clamped": 0,
            "runaway_terminated": 0,
        }

    def record(self, entry: dict[str, Any]) -> None:
        with self.lock:
            self.index += 1
            entry["request_index"] = self.index
            with self.receipt_path.open("a", encoding="utf8") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
                handle.flush()


def _forward(state: GuardState, path: str, payload: bytes | None, method: str) -> tuple[int, bytes, str]:
    url = f"{state.upstream}{path}"
    request = urllib.request.Request(url, data=payload, method=method)
    if payload is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            return response.status, response.read(), response.headers.get("Content-Type", "application/json")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get("Content-Type", "application/json")
    except OSError as exc:
        return 502, json.dumps({"error": {"message": str(exc)}}).encode(), "application/json"


def _handle_chat(state: GuardState, body: dict[str, Any]) -> tuple[int, bytes, str, dict[str, Any]]:
    messages = body.get("messages") or []
    compacted, stats = compact_messages(messages)
    if stats["strategy"] != "unchanged":
        with state.lock:
            state.counters["compacted"] += 1
        body = {**body, "messages": compacted}

    budget = generation_budget(stats["prompt_tokens_after"], context=state.context)
    requested = body.get("max_tokens")
    if budget <= 0:
        # Not enough window left to generate anything: refuse before the server
        # can produce a truncated tool call.
        entry = {"event": "NO_GENERATION_BUDGET", "policy": POLICY_NAME, **stats}
        state.record(entry)
        return 200, json.dumps(runaway_completion(str(body.get("model")), reason="NO_GENERATION_BUDGET")).encode(), "application/json", entry
    if not isinstance(requested, int) or requested <= 0 or requested > budget:
        with state.lock:
            state.counters["budget_clamped"] += 1
        body = {**body, "max_tokens": budget}

    status, raw, content_type = _forward(state, "/v1/chat/completions", json.dumps(body).encode(), "POST")
    entry = {
        "event": "forwarded",
        "policy": POLICY_NAME,
        "max_tokens": body.get("max_tokens"),
        "requested_max_tokens": requested,
        **stats,
    }
    if is_runaway_error(status, raw):
        with state.lock:
            state.counters["runaway_terminated"] += 1
        entry["event"] = RUNAWAY_REASON
        state.record(entry)
        return 200, json.dumps(runaway_completion(str(body.get("model")))).encode(), "application/json", entry
    state.record(entry)
    return status, raw, content_type, entry


class _Handler(BaseHTTPRequestHandler):
    state: GuardState

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: A003 - silence
        return

    def _send(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        status, raw, ctype = _forward(self.state, self.path, None, "GET")
        self._send(status, raw, ctype)

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if self.path.rstrip("/") != "/v1/chat/completions":
            status, body, ctype = _forward(self.state, self.path, raw, "POST")
            self._send(status, body, ctype)
            return
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("request body must be an object")
        except (ValueError, UnicodeDecodeError):
            self._send(400, b'{"error":{"message":"malformed request"}}', "application/json")
            return
        with self.state.lock:
            self.state.counters["requests"] += 1
        status, payload, ctype, _ = _handle_chat(self.state, body)
        self._send(status, payload, ctype)


def start_guard(*, upstream_port: int, listen_port: int, context: int,
                receipt_path: Path) -> tuple[ThreadingHTTPServer, GuardState, threading.Thread]:
    state = GuardState(upstream_port=upstream_port, context=context, receipt_path=receipt_path)
    handler = type("GuardHandler", (_Handler,), {"state": state})
    server = ThreadingHTTPServer(("127.0.0.1", listen_port), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name="guard-proxy", daemon=True)
    thread.start()
    return server, state, thread


def stop_guard(server: ThreadingHTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=10)


def summarize(counters: Mapping[str, int]) -> dict[str, int]:
    return {key: int(value) for key, value in counters.items()}
