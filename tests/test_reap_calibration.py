"""reap_calibration v2: the Fable turn is the logged context plus its target, and
Ornith rows come only from a digest-pinned file's admitted, valid SFT rows."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("reap_calibration", ROOT / "chowder_batch" / "reap_calibration.py")
calib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(calib)


def test_fable_target_becomes_the_final_assistant_turn() -> None:
    row = {
        "messages": [{"role": "user", "content": "fix the bug"}],
        "target": {"reasoning": "look at app.py", "content": "",
                   "tool_calls": [{"type": "function", "function": {"name": "read", "arguments": "{}"}}]},
    }
    messages = calib._fable_messages(row)
    assert messages[0] == {"role": "user", "content": "fix the bug"}
    assert messages[-1] == {"role": "assistant", "content": "", "reasoning_content": "look at app.py",
                            "tool_calls": row["target"]["tool_calls"]}


def _admitted(tmp_path: Path, rows: list[dict]) -> tuple[str, str]:
    body = "".join(json.dumps(r) + "\n" for r in rows).encode("utf-8")
    path = tmp_path / "admitted.jsonl"
    path.write_bytes(body)
    return str(path), hashlib.sha256(body).hexdigest()


def _row(**extra: object) -> dict:
    row = {"admitted": True, "lane": "sft", "validity": {"valid": True}, "prompt": "2*21?", "expected_answer": "42"}
    row.update(extra)
    return row


def test_ornith_serves_only_admitted_valid_sft_rows(tmp_path: Path) -> None:
    path, sha = _admitted(tmp_path, [
        _row(),
        _row(admitted=False),
        _row(lane="rl"),
        _row(validity={"valid": False}),
        {"admitted": True, "lane": "sft", "validity": {"valid": True},
         "task": {"prompt": "nested?", "expected_answer": "yes"}},
    ])
    assert calib.ornith_rows(path, sha) == [
        {"prompt": "2*21?", "answer": "42"},
        {"prompt": "nested?", "answer": "yes"},
    ]


def test_ornith_refuses_a_file_that_is_not_the_pinned_one(tmp_path: Path) -> None:
    path, _ = _admitted(tmp_path, [_row()])
    with pytest.raises(SystemExit, match="pinned"):
        calib.ornith_rows(path, "0" * 64)


def test_a_fable_row_the_template_refuses_is_skipped_not_flattened() -> None:
    class Refusing:
        def apply_chat_template(self, *_args, **_kwargs):
            raise ValueError("No user query found in messages.")

    row = {"messages": [{"role": "system", "content": "s"}], "target": {"content": "ok"}}
    assert calib._render_fable(Refusing(), row) is None


def test_lordx64_xml_tool_use_becomes_native_tool_calls() -> None:
    text = ("<|im_start|>system\nbe an agent<|im_end|>\n<|im_start|>user\nlist files<|im_end|>\n"
            "<|im_start|>assistant\nLet me look.\n<tool_use name=\"Bash\" id=\"toolu_01\">\n"
            "{\"command\": \"ls\"}\n</tool_use><|im_end|>")
    messages = calib.lordx64_messages(text)
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert messages[2]["content"] == "Let me look."
    assert messages[2]["tool_calls"] == [{"type": "function", "function": {"name": "Bash", "arguments": {"command": "ls"}}}]


def test_string_tool_arguments_become_dicts() -> None:
    messages = [{"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "id": "c1", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]}]
    assert calib._native_tool_calls(messages)[0]["tool_calls"][0]["function"]["arguments"] == {"a": 1}
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == "{\"a\": 1}"  # input untouched


def test_dedupe_drops_rows_sharing_an_assistant_turn_and_grows_the_seen_set() -> None:
    long = "I will inspect the repository layout first and then run the test suite to see what fails. " * 2
    local = [{"role": "user", "content": "q"}, {"role": "assistant", "content": long}]
    seen = calib._fingerprints(local)
    fresh = [{"role": "user", "content": "q"}, {"role": "assistant", "content": long.upper()}]
    other = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "A different answer entirely. " * 5}]
    kept, dropped = calib.dedupe([(fresh, None), (other, None), (other, None)], seen)
    assert dropped == 2  # the case-only repackaging, and the second copy of `other`
    assert kept == [(other, None)]


def test_a_shared_stock_line_alone_does_not_make_a_duplicate() -> None:
    greeting = "Hello! I am here to help you with your order today. How can I assist you right now?"
    seen = calib._fingerprints([{"role": "assistant", "content": greeting}])
    convo = [{"role": "user", "content": "q"}, {"role": "assistant", "content": greeting}] + [
        {"role": "assistant", "content": f"Distinct substantive step number {i}: I checked the order record, confirmed the address and the refund policy."}
        for i in range(3)]
    kept, dropped = calib.dedupe([(convo, None)], seen)
    assert dropped == 0 and len(kept) == 1


def test_tool_call_arguments_identify_a_repackaged_turn() -> None:
    call = {"type": "function", "function": {"name": "Bash", "arguments": {
        "command": "cd /work/app && npm install three@0.160.0 && npm run build -- --mode production"}}}
    seen = calib._fingerprints([{"role": "assistant", "content": "", "tool_calls": [call]}])
    kept, dropped = calib.dedupe([([{"role": "assistant", "content": "ok", "tool_calls": [call]}], None)], seen)
    assert dropped == 1 and kept == []
