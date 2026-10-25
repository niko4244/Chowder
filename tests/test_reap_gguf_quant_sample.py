"""Numerical GGUF Q8 reconstruction checks; no GPU or model loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gguf import GGMLQuantizationType as Q
from gguf import GGUFReader, GGUFWriter, quants

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


def _chunked_pair(tmp_path: Path, monkeypatch, *, target_file_type=7):
    """Actual tiny 733-tensor GGUF files and source-bound final chunk receipt."""
    bf16 = tmp_path / "source.gguf"
    q8 = tmp_path / "chunked.gguf"
    raw, quant = _bits_and_q8(256)
    names = [f"blk.{i}.{suffix}" for i in range(sample.LAYERS)
             for suffix in sample.EXPERT_KEYS]
    names.extend(f"extra.{i:03d}.weight" for i in range(
        sample.TENSOR_INVENTORY - len(names)))

    for file, file_type in ((bf16, 32), (q8, target_file_type)):
        writer = GGUFWriter(str(file), "qwen35moe")
        writer.add_file_type(file_type)
        writer.add_uint32("qwen35moe.block_count", sample.LAYERS)
        writer.add_uint32("qwen35moe.expert_count", 128)
        writer.add_uint32("qwen35moe.expert_used_count", 8)
        for index, name in enumerate(names):
            if index < 120:
                if file == bf16:
                    data, dtype = raw.reshape(8, 64), Q.BF16
                else:
                    data, dtype = quant.reshape(8, 34), Q.Q8_0
                writer.add_tensor(name, data, raw_dtype=dtype)
            elif name == "extra.000.weight":
                # The pinned llama-quantize production stream *actually*
                # canonicalizes copied F32 [2048,1] to [2048]. No FP32 byte
                # may change, and source/quantized rows remain BF16-bound.
                values = np.arange(2048, dtype="<f4")
                writer.add_tensor(name, values.reshape(1, 2048) if file == bf16
                                  else values, raw_dtype=Q.F32)
            else:
                writer.add_tensor(name, np.full((8, 32), index / 733,
                                                dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def file_hash(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    source_sha = file_hash(bf16)
    spec_sha = "c" * 64
    monkeypatch.setattr(sample, "FROZEN_SOURCE_SHA256", source_sha)
    monkeypatch.setattr(sample, "FROZEN_SPEC_SHA256", spec_sha)
    stage = {
        "schema": "chowder.reap-gguf-stage.v1",
        "phase": 0, "output": str(bf16.resolve()),
        "sha256": source_sha, "spec_sha256": spec_sha,
        "status": "PUBLISHED_GGUF_NOT_QUALITY_QUALIFIED",
        "size_bytes": bf16.stat().st_size,
    }
    (tmp_path / (bf16.name + ".reap-stage.json")).write_text(json.dumps(stage))

    reader = GGUFReader(str(q8), "r")
    receipts = {
        t.name: {
            "shape": [int(v) for v in t.shape],
            "tensor_type": t.tensor_type.name,
            "raw_sha256": hashlib.sha256(t.data.tobytes()).hexdigest(),
        }
        for t in reader.tensors
    }
    groups = [names[:250], names[250:500], names[500:]]
    record = {
        "schema": "chowder.reap128-v2-gguf-q8-chunked-final.v1",
        "status": "Q8_TENSOR_EXACT_CONSTRUCTED_NOT_QUALITY_QUALIFIED",
        "source_gguf_sha256": source_sha,
        "output": str(q8.resolve()),
        "output_sha256": file_hash(q8),
        "output_size_bytes": q8.stat().st_size,
        "frozen_stage_spec_sha256": spec_sha,
        "quantizer_binary_sha256": sample.FROZEN_QUANTIZER_SHA256,
        "chunk_count": len(groups),
        "chunked_reference_receipts": {
            "all_tensors": receipts, "source_groups": groups,
        },
        "quality_gate": "NOT_EVALUATED", "gpu_fit_gate": "NOT_EVALUATED",
    }
    receipt_path = tmp_path / (q8.name + ".reap-chunked.json")
    receipt_path.write_text(json.dumps(record))
    return bf16, q8, receipt_path


def test_real_chunked_q8_witness_all_733_tensors_accepted(tmp_path, monkeypatch):
    bf16, q8, _ = _chunked_pair(tmp_path, monkeypatch)
    identity = sample._stage_identity(bf16, q8)
    assert identity["q8_0"]["output_sha256"] == hashlib.sha256(q8.read_bytes()).hexdigest()
    result = sample.measure(bf16, q8)
    assert result["status"] == "DESCRIPTIVE_QUANTIZATION_SAMPLE_NOT_QUALITY_GATE"
    assert result["compared_expert_tensors"] == 120
    assert result["total_compared_values"] == 120 * 256
    assert 0 < result["mean_abs_error"] < 0.01


def test_chunked_cli_preflight_describes_hashes_without_quality_claim(
    tmp_path, monkeypatch, capsys,
):
    bf16, q8, _ = _chunked_pair(tmp_path, monkeypatch)
    out = tmp_path / "numeric-sample.json"
    assert sample.main(["--bf16", str(bf16), "--q8", str(q8),
                        "--out", str(out)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "status": "PREFLIGHT_ONLY_NUMERICS_NOT_MEASURED",
        "stage_sha256": [hashlib.sha256(bf16.read_bytes()).hexdigest(),
                         hashlib.sha256(q8.read_bytes()).hexdigest()],
    }
    assert not out.exists()


def test_chunked_final_refuses_non_q8_gguf_mode(tmp_path, monkeypatch):
    bf16, q8, _ = _chunked_pair(tmp_path, monkeypatch, target_file_type=32)
    with pytest.raises(ValueError, match="mode or frozen 40x128 geometry"):
        sample._stage_identity(bf16, q8)


@pytest.mark.parametrize("field,value,reason", [
    ("source_gguf_sha256", "0" * 64, "source, mode"),
    ("frozen_stage_spec_sha256", "0" * 64, "source, mode"),
    ("quantizer_binary_sha256", "0" * 64, "source, mode"),
    ("output_sha256", "0" * 64, "output SHA"),
    ("output_size_bytes", 1, "source, mode"),
    ("status", "QUALITY_GATE_PASS", "source, mode"),
    ("quality_gate", "PASS", "source, mode"),
    ("gpu_fit_gate", "PASS", "source, mode"),
    ("chunk_count", 4, "inventory or group count"),
])
def test_chunked_final_receipt_rejects_forged_ancestry_hash_or_gate(
    tmp_path, monkeypatch, field, value, reason,
):
    bf16, q8, sidecar = _chunked_pair(tmp_path, monkeypatch)
    record = json.loads(sidecar.read_text())
    record[field] = value
    sidecar.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=reason):
        sample._stage_identity(bf16, q8)


@pytest.mark.parametrize("alter,reason", [
    ("tensor_hash", "tensor shape/type/payload"),
    ("tensor_shape", "tensor shape/type/payload"),
    ("tensor_type", "tensor shape/type/payload"),
    ("missing_tensor", "inventory or group count"),
    ("duplicate_group_name", "cover each of 733"),
    ("unwitnessed_group_name", "cover each of 733"),
])
def test_chunked_final_receipt_rejects_tensor_forgery_and_group_tampering(
    tmp_path, monkeypatch, alter, reason,
):
    bf16, q8, sidecar = _chunked_pair(tmp_path, monkeypatch)
    record = json.loads(sidecar.read_text())
    refs = record["chunked_reference_receipts"]
    name = refs["source_groups"][0][0]
    if alter == "tensor_hash":
        refs["all_tensors"][name]["raw_sha256"] = "0" * 64
    elif alter == "tensor_shape":
        refs["all_tensors"][name]["shape"] = [999]
    elif alter == "tensor_type":
        refs["all_tensors"][name]["tensor_type"] = "F32"
    elif alter == "missing_tensor":
        del refs["all_tensors"][name]
    elif alter == "duplicate_group_name":
        refs["source_groups"][0][1] = name
    elif alter == "unwitnessed_group_name":
        refs["source_groups"][0][0] = "unknown"
    sidecar.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=reason):
        sample._stage_identity(bf16, q8)


def test_chunked_refuses_in_place_file_mutation_and_forged_source_stage(tmp_path, monkeypatch):
    bf16, q8, sidecar = _chunked_pair(tmp_path, monkeypatch)
    with q8.open("r+b") as handle:
        handle.seek(-8, 2)
        byte = handle.read(1)
        handle.seek(-1, 1)
        handle.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match="output SHA"):
        sample._stage_identity(bf16, q8)

    record = json.loads(sidecar.read_text())
    record["output_sha256"] = hashlib.sha256(q8.read_bytes()).hexdigest()
    sidecar.write_text(json.dumps(record))
    stage_path = bf16.with_name(bf16.name + ".reap-stage.json")
    stage = json.loads(stage_path.read_text())
    stage["sha256"] = "a" * 64
    stage_path.write_text(json.dumps(stage))
    record["source_gguf_sha256"] = stage["sha256"]
    sidecar.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="registered full BF16 stage SHA"):
        sample._stage_identity(bf16, q8)


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
