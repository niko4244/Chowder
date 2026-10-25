"""Numerical GGUF Q8 reconstruction checks; no GPU or model loading."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gguf import GGMLQuantizationType as Q
from gguf import quants

from chowder_batch import reap_gguf_quant_sample as sample


def _bits_and_q8(n: int = 1024):
    # The BF16 source values are constructed in native on-disk little endian
    # representation; no approximate torch comparisons are involved.
    values = np.linspace(-0.9, 0.7, n, dtype=np.float32)
    values = (values.view("<u4") >> 16).astype("<u2")
    exact = (values.astype("<u4") << 16).view("<f4")
    bf16 = np.frombuffer(values.tobytes(), dtype="u1").copy()
    quantized = quants.quantize(exact.reshape(-1, 32), Q.Q8_0).reshape(-1)
    return bf16, quantized


def _stage_pair(tmp_path: Path):
    bf16 = tmp_path / "base.gguf"
    q8 = tmp_path / "quant.gguf"
    bf16.write_bytes(b"GGUF" + b"\x00" * 52)
    q8.write_bytes(b"GGUF" + b"\x01" * 52)
    for phase, file in enumerate((bf16, q8)):
        record = {
            "schema": "chowder.reap-gguf-stage.v1",
            "phase": phase, "output": str(file.resolve()),
            "sha256": ("a" if phase == 0 else "b") * 64,
            "spec_sha256": "c" * 64,
            "input_sha256": "f" * 64 if phase == 0 else "a" * 64,
            "status": "PUBLISHED_GGUF_NOT_QUALITY_QUALIFIED",
            "size_bytes": file.stat().st_size,
        }
        (tmp_path / (file.name + ".reap-stage.json")).write_text(json.dumps(record))
    return bf16, q8


def test_stage_identity_accepts_only_shared_source_bound_witnesses(tmp_path):
    bf16, q8 = _stage_pair(tmp_path)
    assert sample._stage_identity(bf16, q8)["q8_0"]["input_sha256"] == "a" * 64
    sidecar = tmp_path / (q8.name + ".reap-stage.json")
    record = json.loads(sidecar.read_text())
    record["input_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="does not descend"):
        sample._stage_identity(bf16, q8)


def test_q8_bf16_byte_window_measures_expected_finite_dequant_errors():
    reference, quantized = _bits_and_q8(1024)
    result = sample._compare_window(reference, quantized, element_offset=0)
    assert result["count"] == 256
    assert 0 < result["max_abs_error"] < 0.01
    assert result["sum_abs_error"] > 0
    assert result["sum_squared_error"] > 0
    altered = quantized.copy()
    altered[5] ^= 1
    damaged = sample._compare_window(reference, altered, element_offset=0)
    assert damaged["sum_abs_error"] != result["sum_abs_error"]


def test_quant_sampling_offsets_are_deterministic_and_block_aligned():
    assert sample._windows(1024) == [0, 256, 512, 768]
    assert sample._windows(256) == [0]
    with pytest.raises(ValueError, match="32-aligned"):
        sample._windows(1023)
    with pytest.raises(ValueError, match="unaligned"):
        sample._compare_window(*_bits_and_q8(), element_offset=1)


def test_real_40_layer_inventory_is_sampled_without_loading_model(tmp_path, monkeypatch):
    bf16, q8 = _stage_pair(tmp_path)
    raw, quant = _bits_and_q8()

    def tensors(is_quant: bool):
        answer = []
        for i in range(40):
            for suffix in sample.EXPERT_KEYS:
                answer.append(SimpleNamespace(
                    name=f"blk.{i}.{suffix}", shape=np.array((32, 32)),
                    tensor_type=Q.Q8_0 if is_quant else Q.BF16,
                    data=quant if is_quant else raw,
                ))
        return answer

    import gguf
    monkeypatch.setattr(
        gguf, "GGUFReader",
        lambda path, _mode: SimpleNamespace(tensors=tensors(Path(path) == q8)),
    )
    result = sample.measure(bf16, q8)
    assert result["status"] == "DESCRIPTIVE_QUANTIZATION_SAMPLE_NOT_QUALITY_GATE"
    assert result["compared_expert_tensors"] == 120
    assert result["total_compared_values"] == 120 * 4 * 256
    assert 0 < result["mean_abs_error"] < 0.01
    assert len(result["samples"]) == 120
    assert "Does not measure perplexity" in result["limitations"]
