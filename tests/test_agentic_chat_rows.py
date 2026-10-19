"""Agentic chat rows train intact: tool turns, tool calls (dict or Arrow-safe JSON
string arguments), reasoning, and per-row tools passed to the chat template --
while plain role/content rows tokenize exactly as before."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chowder.backends.training_data import (
    _build_chat_example,
    _chat_digest,
    _validate_chat_messages,
    parse_tools,
)

CALL = {"type": "function", "function": {"name": "get_order", "arguments": '{"order_id": "A1"}'}}
TOOLS = [{"type": "function", "function": {"name": "get_order", "description": "Look up an order",
          "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                         "required": ["order_id"]}}}]
ROW = [
    {"role": "system", "content": "You are a support agent."},
    {"role": "user", "content": "Where is order A1?"},
    {"role": "assistant", "content": None, "tool_calls": [CALL], "reasoning_content": "Look it up."},
    {"role": "tool", "content": '{"status": "shipped"}', "tool_call_id": "c1", "name": "get_order"},
    {"role": "assistant", "content": "Order A1 has shipped."},
]


def test_agentic_turns_are_kept_with_dict_arguments():
    messages = _validate_chat_messages(ROW, row_index=0)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert messages[2]["content"] == ""  # None content is empty, never the string "None"
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == {"order_id": "A1"}
    assert messages[2]["reasoning_content"] == "Look it up."
    assert messages[3]["tool_call_id"] == "c1" and messages[3]["name"] == "get_order"
    assert ROW[2]["tool_calls"][0]["function"]["arguments"] == '{"order_id": "A1"}'  # input untouched


def test_plain_rows_are_unchanged():
    plain = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert _validate_chat_messages(plain, row_index=0) == plain


def test_bad_tool_payloads_refuse():
    bad = [{"role": "user", "content": "q"},
           {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": "{oops"}}]}]
    with pytest.raises(RuntimeError, match="not JSON"):
        _validate_chat_messages(bad, row_index=3)
    with pytest.raises(RuntimeError, match="non-list"):
        _validate_chat_messages([{"role": "assistant", "content": "", "tool_calls": "x"}], row_index=4)
    with pytest.raises(RuntimeError, match="not JSON"):
        parse_tools("[{oops", row_index=5)


def test_tools_accept_a_list_or_its_json_string():
    assert parse_tools(TOOLS, row_index=0) == TOOLS
    assert parse_tools(json.dumps(TOOLS), row_index=0) == TOOLS
    assert parse_tools("", row_index=0) is None


def test_chat_digest_covers_tools_only_when_declared():
    rows = [{"messages": ROW, "tools": json.dumps(TOOLS)}]
    assert _chat_digest(rows, "messages") == _chat_digest([{"messages": ROW}], "messages")
    assert _chat_digest(rows, "messages", "tools") != _chat_digest(rows, "messages")


QWEN = Path(r"G:\llm-cold\bf16\Qwen3.6-35B-A3B")


@pytest.mark.skipif(not QWEN.is_dir(), reason="needs the local Qwen3.6 tokenizer")
def test_real_qwen_template_renders_tools_and_labels_only_assistant_turns():
    tokenizer = pytest.importorskip("transformers").AutoTokenizer.from_pretrained(str(QWEN), local_files_only=True)
    messages = _validate_chat_messages(ROW, row_index=0)
    example = _build_chat_example(tokenizer, messages, max_length=4096, row_index=0, tools=TOOLS)
    text = tokenizer.decode(example["input_ids"])
    assert "get_order" in text and "Look up an order" in text  # the tool schema rendered
    labelled = tokenizer.decode([t for t, l in zip(example["input_ids"], example["labels"]) if l != -100])
    assert "order_id" in labelled and "Order A1 has shipped." in labelled  # the call and the answer are trained
    assert "status" not in labelled  # the tool result ({"status": ...}) is never trained on
