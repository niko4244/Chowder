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
