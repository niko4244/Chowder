"""Ornith rows in the REAP mix: bare stays the pre-registered text, worked adds the
verified solution, and the MoE SFT export writes that same worked turn once per task."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chowder_batch import ornith_sft_export, reap_calibration as calib  # noqa: E402

EXPORT = ROOT / "chowder_batch" / "ornith_sft_export.py"
FLOOR = 256  # the packing loop's drop threshold in reap_calibration.main
CODE = "import networkx as nx\nprint('max flow', 20)\n"
SHA_A = "a" * 64
SHA_B = "b" * 64


class FakeTokenizer:
    """Whitespace tokens: enough for ``_render``, ``decode`` and the packing floor."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": text.split()}

    def decode(self, ids: list[str]) -> str:
        return " ".join(ids)

    def apply_chat_template(self, messages: list[dict], tools=None, tokenize: bool = False) -> str:
        return "".join(f"<|{m['role']}|>{m['content']}\n" for m in messages)


def _row(**extra: object) -> dict:
    """An admitted-file row, as ``admitted.jsonl`` holds it."""
    row = {"admitted": True, "lane": "sft", "validity": {"valid": True},
           "prompt": "How many edges cross the minimum cut?", "expected_answer": "20",
           "verification_code": CODE, "proposal_sha256": SHA_A}
    row.update(extra)
    return row


def _out(**extra: object) -> dict:
    """A row as ``ornith_rows`` returns it, the shape the renderers consume."""
    row = {"prompt": "How many edges cross the minimum cut?", "answer": "20",
           "code": CODE, "proposal_sha256": SHA_A}
    row.update(extra)
    return row


def _admitted(tmp_path: Path, rows: list[dict]) -> tuple[str, str]:
    body = "".join(json.dumps(r) + "\n" for r in rows).encode("utf-8")
    path = tmp_path / "admitted.jsonl"
    path.write_bytes(body)
    return str(path), hashlib.sha256(body).hexdigest()


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(EXPORT), *args],
                          capture_output=True, text=True, cwd=str(ROOT))


def test_bare_is_the_default_and_renders_as_before_through_the_tokenizer() -> None:
    tokenizer = FakeTokenizer()
    row = _out()
    before = calib._render(tokenizer, [{"role": "user", "content": row["prompt"]},
                                       {"role": "assistant", "content": row["answer"]}])
    assert calib._parser().parse_args(["--tokenizer", "t", "--out", "o"]).ornith_render == "bare"
    assert calib.ornith_assistant(row) == row["answer"]
    assert calib.ornith_text(tokenizer, row) == before
    assert CODE not in calib.ornith_text(tokenizer, row, "bare")


def test_worked_render_carries_the_code_block_and_the_final_answer() -> None:
    tokenizer = FakeTokenizer()
    row = _out()
    text = calib.ornith_text(tokenizer, row, "worked")
    assert "Let me solve this step by step in Python." in text
    assert f"```python\n{CODE}\n```" in text
    assert f"Final answer: {row['answer']}" in text
    assert calib.ornith_assistant(row, "worked") == calib.worked_answer(CODE, row["answer"])


def test_worked_mode_lifts_a_short_row_over_the_packing_floor() -> None:
    """A bare Ornith row is ~100 tokens and is dropped; the ~200-token solution clears the floor."""
    tokenizer = FakeTokenizer()
    row = {"prompt": " ".join(["workload"] * 80), "answer": " ".join(["answer"] * 10),
           "code": "\n".join(["result = solve(problem, constraints)"] * 50)}
    bare = tokenizer(calib.ornith_text(tokenizer, row))["input_ids"]
    worked = tokenizer(calib.ornith_text(tokenizer, row, "worked"))["input_ids"]
    assert len(bare) < FLOOR <= len(worked)


def test_non_admitted_non_sft_and_invalid_rows_are_excluded_in_both_modes(tmp_path: Path) -> None:
    path, sha = _admitted(tmp_path, [
        _row(prompt="keep me"),
        _row(admitted=False, prompt="not admitted"),
        _row(lane="rl", prompt="not sft"),
        _row(validity={"valid": False}, prompt="not valid"),
    ])
    rows = calib.ornith_rows(path, sha)
    assert [r["prompt"] for r in rows] == ["keep me"]
    assert rows[0]["code"] == CODE
    tokenizer = FakeTokenizer()
    for render in ("bare", "worked"):
        rendered = "".join(calib.ornith_text(tokenizer, r, render) for r in rows)
        for junk in ("not admitted", "not sft", "not valid"):
            assert junk not in rendered
    assert [r["messages"][0]["content"] for r in ornith_sft_export.sft_rows(path, sha)] == ["keep me"]


def test_a_file_that_is_not_the_pinned_one_is_refused_in_both_paths(tmp_path: Path) -> None:
    path, _ = _admitted(tmp_path, [_row()])
    with pytest.raises(SystemExit, match="pinned"):
        calib.ornith_rows(path, "0" * 64)
    with pytest.raises(SystemExit, match="pinned"):
        ornith_sft_export.sft_rows(path, "0" * 64)


def test_the_export_cli_refuses_a_mismatched_sha256_and_writes_nothing(tmp_path: Path) -> None:
    path, _ = _admitted(tmp_path, [_row()])
    out = tmp_path / "sft.jsonl"
    done = _cli("--admitted", path, "--sha256", "0" * 64, "--out", str(out))
    assert done.returncode != 0
    assert "pinned" in done.stderr
    assert not out.exists()


def test_export_dedupes_proposal_sha256(tmp_path: Path) -> None:
    path, sha = _admitted(tmp_path, [
        _row(prompt="first task", proposal_sha256=SHA_A),
        _row(prompt="same proposal, listed again", proposal_sha256=SHA_A),
        _row(prompt="second task", proposal_sha256=SHA_B),
    ])
    rows = ornith_sft_export.sft_rows(path, sha)
    assert [r["proposal_sha256"] for r in rows] == [SHA_A, SHA_B]
    assert [r["messages"][0]["content"] for r in rows] == ["first task", "second task"]

    out = tmp_path / "sft.jsonl"
    done = _cli("--admitted", path, "--sha256", sha, "--out", str(out))
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "wrote 2 rows"
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert written == rows
    assert written[0] == {
        "messages": [{"role": "user", "content": "first task"},
                     {"role": "assistant", "content": calib.worked_answer(CODE, "20")}],
        "source": "ornith_curriculum",
        "proposal_sha256": SHA_A,
    }
