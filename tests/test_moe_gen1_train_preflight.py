"""Exact synthetic admission checks without model files, datasets or GPU."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chowder_batch.moe_gen1_train_preflight import inspect_mix, main


class Tokenizer:
    """Prefix-consistent deterministic template with one ID per character."""

    _ROLE = {"system": 71, "user": 72, "assistant": 73, "tool": 74}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, tools=None):
        ids = []
        if tools:
            ids.extend([91] * 7)
        for item in messages:
            ids.append(self._ROLE[item["role"]])
            ids.extend(ord(char) + 100 for char in item["content"])
        if add_generation_prompt:
            ids.append(self._ROLE["assistant"])
        return ids


def _row(prompt: str, response: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "tools": "",
    }


def _write_mix(path: Path, *, fable: list[dict], control: list[dict]) -> None:
    path.mkdir()
    outputs = {}
    for filename, rows in (("train-fable.jsonl", fable), ("train-control.jsonl", control)):
        data = "".join(json.dumps(row) + "\n" for row in rows).encode()
        (path / filename).write_bytes(data)
        outputs[filename] = {"sha256": hashlib.sha256(data).hexdigest(), "rows": len(rows)}
    (path / "manifest.json").write_text(
        json.dumps({
            "builder": "chowder_batch/build_moe_gen1_mix.py",
            "template_validated": True,
            "tokenizer": {"path": "fake-tokenizer", "loaded": True},
            "outputs": outputs,
        }),
        encoding="utf-8",
    )


def test_complete_answers_and_strict_fable_free_control_pass(tmp_path):
    control = _row("question", "short answer")
    treatment = _row("tool call", "do action")
    _write_mix(tmp_path / "mix", fable=[treatment, control], control=[control])
    result = inspect_mix(tmp_path / "mix", Tokenizer(), max_length=128)
    assert result["ready"] is True
    assert result["fable_only_rows"] == 1
    assert result["arms"]["train-control.jsonl"]["answer_span_truncated"] == 0


def test_partial_assistant_answer_must_fail_even_if_original_trainer_would_train(tmp_path):
    common = _row("q", "ok")
    long_answer = _row("q", "lengthy response goes past the limit")
    _write_mix(tmp_path / "mix", fable=[common, long_answer], control=[common])
    result = inspect_mix(tmp_path / "mix", Tokenizer(), max_length=18)
    assert result["ready"] is False
    assert result["arms"]["train-fable.jsonl"]["over_length"] == 1
    assert result["arms"]["train-fable.jsonl"]["answer_span_truncated"] == 1
    assert result["arms"]["train-fable.jsonl"]["answer_span_completely_removed"] == 0


def test_silent_loss_of_entire_assistant_reply_is_reported(tmp_path):
    common = _row("a", "ok")
    long_prompt = _row("x" * 200, "final reply")
    _write_mix(tmp_path / "mix", fable=[common, long_prompt], control=[common])
    result = inspect_mix(tmp_path / "mix", Tokenizer(), max_length=32)
    assert result["ready"] is False
    assert result["arms"]["train-fable.jsonl"]["answer_span_completely_removed"] == 1


def test_bound_digest_refuses_modified_training_row(tmp_path):
    common = _row("q", "yes")
    _write_mix(tmp_path / "mix", fable=[common, _row("r", "a")], control=[common])
    (tmp_path / "mix" / "train-fable.jsonl").write_text('changed\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        inspect_mix(tmp_path / "mix", Tokenizer())


def test_control_cannot_reorder_or_insert_rows(tmp_path):
    a, b, c = (_row(str(i), str(i + 10)) for i in range(3))
    _write_mix(tmp_path / "mix", fable=[a, b, c], control=[b, a])
    with pytest.raises(ValueError, match="strict subsequence"):
        inspect_mix(tmp_path / "mix", Tokenizer())


def test_unvalidated_manifest_refused(tmp_path):
    common = _row("q", "answer")
    _write_mix(tmp_path / "mix", fable=[common, _row("r", "a")], control=[common])
    manifest_path = tmp_path / "mix" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["template_validated"] = False
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="without chat-template validation"):
        inspect_mix(tmp_path / "mix", Tokenizer())


def test_cli_refuses_substituted_tokenizer_before_loading_model_files(tmp_path, capsys):
    common = _row("q", "answer")
    _write_mix(tmp_path / "mix", fable=[common, _row("r", "a")], control=[common])
    alternate = tmp_path / "other-tokenizer"
    alternate.mkdir()
    with pytest.raises(SystemExit) as exc:
        main(["--mix", str(tmp_path / "mix"), "--tokenizer", str(alternate)])
    assert exc.value.code == 2
    assert "tokenizer path differs" in capsys.readouterr().err
