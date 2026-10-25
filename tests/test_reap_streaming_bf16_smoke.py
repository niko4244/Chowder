"""Bounds on real CPU-only model loading, without opening production weights."""

from __future__ import annotations

from pathlib import Path

import pytest

from chowder_batch import reap_streaming_bf16_smoke as smoke


def test_empty_smoke_prompt_rejected_before_model_loading(tmp_path):
    with pytest.raises(ValueError, match="cannot be empty"):
        smoke.run(tmp_path / "missing-model", prompt="")


def test_non_cpu_visible_device_rejected_before_model_loading(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"):
        smoke.run(tmp_path / "missing-model", prompt="Hello")


def test_missing_model_does_not_publish_false_success(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    receipt = tmp_path / "never-publish.json"
    result = smoke.main([
        "--model", str(Path(tmp_path / "missing")),
        "--prompt", "The answer is",
        "--out", str(receipt),
    ])
    assert result != 0
    assert not receipt.exists()


def test_existing_receipt_cannot_be_overwritten(tmp_path):
    receipt = tmp_path / "owner.json"
    receipt.write_text("owner evidence")
    with pytest.raises(SystemExit):
        smoke.main(["--model", str(tmp_path), "--out", str(receipt)])
    assert receipt.read_text() == "owner evidence"
