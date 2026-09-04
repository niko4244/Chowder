from __future__ import annotations

import json

import pytest

from chowder.backends.dataset_influence_worker import _load_rows, _resolve_device, _resolve_dtype


class _FakeTorch:
    class float32:
        pass

    class bfloat16:
        pass

    class float16:
        pass

    class cuda:
        _available = True
        _bf16_supported = True

        @staticmethod
        def is_available():
            return _FakeTorch.cuda._available

        @staticmethod
        def is_bf16_supported():
            return _FakeTorch.cuda._bf16_supported


def test_resolve_dtype_fp32_is_explicit():
    assert _resolve_dtype(_FakeTorch, "fp32") is _FakeTorch.float32


def test_resolve_dtype_bf16_requires_support():
    _FakeTorch.cuda._available = True
    _FakeTorch.cuda._bf16_supported = False
    with pytest.raises(RuntimeError, match="bf16"):
        _resolve_dtype(_FakeTorch, "bf16")
    _FakeTorch.cuda._bf16_supported = True
    assert _resolve_dtype(_FakeTorch, "bf16") is _FakeTorch.bfloat16


def test_resolve_dtype_fp16_is_explicit():
    assert _resolve_dtype(_FakeTorch, "fp16") is _FakeTorch.float16


def test_resolve_dtype_auto_prefers_bf16_when_supported():
    _FakeTorch.cuda._available = True
    _FakeTorch.cuda._bf16_supported = True
    assert _resolve_dtype(_FakeTorch, "auto") is _FakeTorch.bfloat16


def test_resolve_dtype_auto_falls_back_to_fp16_without_bf16_support():
    _FakeTorch.cuda._available = True
    _FakeTorch.cuda._bf16_supported = False
    assert _resolve_dtype(_FakeTorch, "auto") is _FakeTorch.float16


def test_resolve_dtype_auto_uses_fp32_without_cuda():
    _FakeTorch.cuda._available = False
    assert _resolve_dtype(_FakeTorch, "auto") is _FakeTorch.float32
    _FakeTorch.cuda._available = True


def test_resolve_device_auto_uses_cuda_when_available():
    _FakeTorch.cuda._available = True
    assert _resolve_device(_FakeTorch, "auto") == "cuda:0"


def test_resolve_device_auto_uses_cpu_without_cuda():
    _FakeTorch.cuda._available = False
    assert _resolve_device(_FakeTorch, "auto") == "cpu"
    _FakeTorch.cuda._available = True


def test_resolve_device_rejects_cuda_when_unavailable():
    _FakeTorch.cuda._available = False
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        _resolve_device(_FakeTorch, "cuda:0")
    _FakeTorch.cuda._available = True


def test_load_rows_returns_text_field_values_in_order(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        "".join(json.dumps({"text": row}) + "\n" for row in ["first", "second", "third"]),
        encoding="utf-8",
    )
    assert _load_rows(str(path), "text") == ["first", "second", "third"]


def test_load_rows_skips_blank_lines(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"text": "a"}\n\n{"text": "b"}\n', encoding="utf-8")
    assert _load_rows(str(path), "text") == ["a", "b"]


def test_load_rows_rejects_missing_text_field(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"other": "a"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing"):
        _load_rows(str(path), "text")


def test_load_rows_rejects_empty_dataset(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="empty"):
        _load_rows(str(path), "text")
