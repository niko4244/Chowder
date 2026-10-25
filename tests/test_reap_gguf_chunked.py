"""Regression for source-bound independently quantized MoE GGUF tensors."""

from __future__ import annotations

import io
import gc
import os
import subprocess
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter

from chowder_batch import reap_gguf_chunked as chunk


def _small_bf16(path: Path) -> None:
    rng = np.random.default_rng(20261025)
    writer = GGUFWriter(str(path), "qwen35moe")
    writer.add_file_type(32)
    writer.add_quantization_version(2)
    settings = {
        "block_count": 2, "context_length": 1024,
        "embedding_length": 256, "attention.head_count": 4,
        "attention.head_count_kv": 2, "expert_count": 16,
        "expert_used_count": 4, "attention.key_length": 64,
        "attention.value_length": 64, "expert_feed_forward_length": 256,
        "expert_shared_feed_forward_length": 256,
        "ssm.conv_kernel": 4, "ssm.state_size": 8,
        "ssm.group_count": 2, "ssm.time_step_rank": 4,
        "ssm.inner_size": 512, "full_attention_interval": 4,
        "rope.dimension_count": 64,
    }
    for key, value in settings.items():
        writer.add_uint32("qwen35moe." + key, value)
    writer.add_array("qwen35moe.rope.dimension_sections", [64, 0, 0, 0])
    writer.add_array("qwen35moe.attention.recurrent_layers", [1, 0])
    writer.add_float32("qwen35moe.rope.freq_base", 1_000_000)
    writer.add_float32("qwen35moe.attention.layer_norm_rms_epsilon", 1e-6)
    for layer in range(2):
        for type_ in ("ffn_gate_exps.weight", "ffn_up_exps.weight",
                      "ffn_down_exps.weight"):
            fp32 = rng.normal(0, 0.35, (16, 256, 256)).astype("<f4")
            packed = (fp32.view("<u4") >> 16).astype("<u2").view("u1")
            packed = packed.reshape((16, 256, 512))
            writer.add_tensor(
                f"blk.{layer}.{type_}", np.ascontiguousarray(packed),
                raw_dtype=GGMLQuantizationType.BF16,
            )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_deterministic_source_chunk_layout_and_rejects_overlarge_tensor():
    tensors = [
        SimpleNamespace(name="one", data=SimpleNamespace(nbytes=6 * chunk.MIB)),
        SimpleNamespace(name="two", data=SimpleNamespace(nbytes=6 * chunk.MIB)),
        SimpleNamespace(name="three", data=SimpleNamespace(nbytes=2 * chunk.MIB)),
    ]
    assert chunk.groups_for(SimpleNamespace(tensors=tensors), max_mib=8) == [
        ["one"], ["two", "three"],
    ]
    with pytest.raises(ValueError, match="larger"):
        chunk.groups_for(SimpleNamespace(tensors=[
            SimpleNamespace(name="oversize", data=SimpleNamespace(nbytes=9 * chunk.MIB)),
        ]), max_mib=8)
    with pytest.raises(ValueError, match="duplicate"):
        chunk.groups_for(SimpleNamespace(tensors=tensors + tensors[:1]), max_mib=8)


def test_lossless_f32_quantizer_singleton_dimension_normalization():
    original = SimpleNamespace(
        tensor_type=GGMLQuantizationType.F32, shape=np.array([2048, 1]),
        data=np.arange(2048, dtype="<f4").reshape(1, 2048),
    )
    normalized = SimpleNamespace(
        tensor_type=GGMLQuantizationType.F32, shape=np.array([2048]),
        data=np.arange(2048, dtype="<f4"),
    )
    assert chunk._quantized_shape_matches(original, normalized)
    assert not chunk._quantized_shape_matches(original, SimpleNamespace(
        tensor_type=GGMLQuantizationType.F32, shape=np.array([2048]),
        data=np.zeros(2048, dtype="<f4"),
    ))
    assert not chunk._quantized_shape_matches(original, SimpleNamespace(
        tensor_type=GGMLQuantizationType.F32, shape=np.array([1024, 2]),
        data=np.arange(2048, dtype="<f4"),
    ))
    assert not chunk._quantized_shape_matches(original, SimpleNamespace(
        tensor_type=GGMLQuantizationType.F16, shape=np.array([2048]),
        data=np.arange(2048, dtype="<f4"),
    ))


def test_chunk_ownership_lock_fails_closed_to_competing_owner(tmp_path):
    with chunk._exclusive_chunk_owner(tmp_path):
        with pytest.raises(ValueError, match="another owned"):
            with chunk._exclusive_chunk_owner(tmp_path):
                pytest.fail("duplicate construction incorrectly acquired lock")
    with chunk._exclusive_chunk_owner(tmp_path):
        pass
    assert (tmp_path / ".reap-gguf-chunked.lock").is_file()


def test_bf16_minis_preserve_every_raw_tensor_byte(tmp_path):
    source = tmp_path / "base.gguf"
    _small_bf16(source)
    names = [t.name for t in GGUFReader(str(source), "r").tensors]
    mini = tmp_path / "mini.gguf"
    proof = chunk.create_mini(source, mini, names[:2])
    assert proof["source_names"] == names[:2]
    assert set(proof["source_tensor_receipts"]) == set(names[:2])
    assert chunk.hash_file(mini) == proof["source_mini_sha256"]
    with pytest.raises(ValueError, match="fresh"):
        chunk.create_mini(source, mini, names[:2])
    with pytest.raises(ValueError, match="not found"):
        chunk.create_mini(source, tmp_path / "invalid.gguf", ["missing.tensor"])


def _synthetic_q8_parts(tmp_path):
    """Two independent source-ordered parts with raw Q8_0 and copied F32."""
    source = tmp_path / "tiny-source.gguf"
    writer = GGUFWriter(str(source), "qwen35moe")
    writer.add_file_type(32)
    writer.add_quantization_version(2)
    writer.add_uint32("qwen35moe.block_count", 2)
    names = ["blk.0.ffn_gate_exps.weight", "blk.1.ffn_up_exps.weight",
             "blk.0.ffn_gate_inp_shexp.weight"]
    for name in names[:2]:
        writer.add_tensor(name, np.full((2, 128), 4, dtype=np.uint8),
                          raw_dtype=GGMLQuantizationType.BF16)
    writer.add_tensor(names[2], np.arange(64, dtype="<f4").reshape(1, 64),
                      raw_dtype=GGMLQuantizationType.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    parts = []
    for index, group in enumerate((names[:2], names[2:])):
        target = tmp_path / f"part-{index}.gguf"
        writer = GGUFWriter(str(target), "qwen35moe")
        writer.add_file_type(7)
        writer.add_quantization_version(2)
        writer.add_uint32("qwen35moe.block_count", 2)
        for name in group:
            if name == names[2]:
                writer.add_tensor(name, np.arange(64, dtype="<f4"),
                                  raw_dtype=GGMLQuantizationType.F32)
            else:
                writer.add_tensor(name, np.full((2, 68), index + 3,
                                                dtype=np.uint8),
                                  raw_dtype=GGMLQuantizationType.Q8_0)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        parts.append(target)
    receipts = {
        t.name: {"shape": list(map(int, t.shape)), "tensor_type": t.tensor_type.name,
                 "raw_sha256": chunk.hash_tensor_data(t.data)}
        for part in parts for t in GGUFReader(str(part), "r").tensors
    }
    return source, parts, receipts


def test_q8_streaming_merge_byte_identical_to_historical_writer(tmp_path):
    source, parts, receipts = _synthetic_q8_parts(tmp_path)
    legacy = tmp_path / "old-writer.gguf"
    src = GGUFReader(str(source), "r")
    by_name = {t.name: t for part in parts
               for t in GGUFReader(str(part), "r").tensors}
    writer = GGUFWriter(str(legacy), "qwen35moe")
    chunk.copy_metadata(src, writer, file_type=7)
    for tensor in src.tensors:
        quantized = by_name[tensor.name]
        writer.add_tensor(tensor.name, quantized.data, raw_dtype=quantized.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    del src, by_name, writer

    streamed = tmp_path / "new-stream.gguf"
    result = chunk.merge_mini(source, parts, streamed,
                              expected_tensor_receipts=receipts, mode="Q8_0")
    assert result["output_sha256"] == chunk.hash_file(legacy)
    assert streamed.read_bytes() == legacy.read_bytes()
    assert list(GGUFReader(str(streamed), "r").fields) == list(
        GGUFReader(str(legacy), "r").fields)


def test_q8_streaming_merge_drops_old_mmaps_and_uses_bounded_reads(tmp_path, monkeypatch):
    import gguf

    source, parts, receipts = _synthetic_q8_parts(tmp_path)
    reader = gguf.GGUFReader
    alive = weakref.WeakSet()
    maximum = [0]

    def tracked(*args, **kwargs):
        gc.collect()
        maximum[0] = max(maximum[0], len(alive))
        assert len(alive) == 0, "prior GGUF reader retained into next part"
        instance = reader(*args, **kwargs)
        alive.add(instance)
        return instance

    monkeypatch.setattr(gguf, "GGUFReader", tracked)
    monkeypatch.setattr(chunk, "HASH_BLOCK", 31)
    target = tmp_path / "bounded.gguf"
    chunk.merge_mini(source, parts, target,
                     expected_tensor_receipts=receipts, mode="Q8_0")
    assert target.is_file()
    assert maximum[0] == 0


def test_q8_streaming_merge_rejects_part_payload_tamper_before_publication(tmp_path):
    source, parts, receipts = _synthetic_q8_parts(tmp_path)
    with parts[0].open("r+b") as fh:
        offset = GGUFReader(str(parts[0]), "r").tensors[0].data_offset
        fh.seek(offset)
        fh.write(b"\xff")
    target = tmp_path / "must-not-publish.gguf"
    with pytest.raises(ValueError, match="mismatch|payload|SHA|digest"):
        chunk.merge_mini(source, parts, target,
                         expected_tensor_receipts=receipts, mode="Q8_0")


def test_q8_abandoned_partial_merge_refused_without_touching_certified_parts(tmp_path):
    part = tmp_path / "chunk-001.q8_0.gguf"
    part.write_bytes(b"certified chunk fixture")
    witness = tmp_path / "merge-input-prior.json"
    witness.write_text("historical receipts")
    chunk._refuse_unsealed_q8_merge(tmp_path)
    orphan = tmp_path / "merged-q8-crashed.gguf"
    orphan.write_bytes(b"interrupted assembly")
    with pytest.raises(ValueError, match="unsealed Q8"):
        chunk._refuse_unsealed_q8_merge(tmp_path)
    assert orphan.read_bytes() == b"interrupted assembly"
    assert part.read_bytes() == b"certified chunk fixture"
    orphan.unlink()
    orphan_receipt = tmp_path / "merged-q8-crashed.gguf.receipt.json"
    orphan_receipt.write_text("incomplete worker receipt")
    with pytest.raises(ValueError, match="unsealed Q8"):
        chunk._refuse_unsealed_q8_merge(tmp_path)


@pytest.mark.skipif(
    os.name != "nt" or not Path("C:/llama.cpp/bin/llama-quantize.exe").is_file(),
    reason="pinned Windows llama-quantize executable unavailable",
)
def test_pinned_quantizer_independent_q8_minis_merge_bitwise_equal_to_monolith(
    tmp_path, monkeypatch,
):
    from chowder_batch import reap_gguf_supervised as supervise
    from chowder_batch.reap_gguf_q4_chunked import _require_no_competing_quantizer

    try:
        _require_no_competing_quantizer()
    except ValueError:
        pytest.skip("another registered conversion currently owns the CPU quantizer")

    source = tmp_path / "bf16.gguf"
    _small_bf16(source)
    quantizer = Path("C:/llama.cpp/bin/llama-quantize.exe")
    original_names = [t.name for t in GGUFReader(str(source), "r").tensors]
    groups = [original_names[:3], original_names[3:]]
    components = []
    identities = {}

    def run_local_owned(command, **kwargs):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
        result = subprocess.run(command, env=env, capture_output=True,
                                text=True, timeout=90, check=False)
        assert result.returncode == 0, result.stderr
        return result.returncode

    monkeypatch.setattr(supervise, "_owned_stage", run_local_owned)
    for index, group in enumerate(groups):
        mini = tmp_path / f"mini-{index}.gguf"
        output = tmp_path / f"q8-mini-{index}.gguf"
        chunk.create_mini(source, mini, group)
        record = chunk.quantize_mini(
            mini, output, quantizer=quantizer,
            mode="Q8_0", memory_mib=1, log=io.StringIO(),
        )
        assert record["quant_command"][1:3] == ["--max-buffer-size", "1"]
        assert record["output_sha256"] == chunk.hash_file(output)
        identities.update(record["quantized_tensors"])
        components.append(output)
    combined = tmp_path / "q8-combined.gguf"
    result = chunk.merge_mini(
        source, components, combined,
        expected_tensor_receipts=identities, mode="Q8_0",
    )
    assert result["status"] == "GGUF_CHUNKED_TENSOR_BITWISE_MERGE_NOT_QUALITY_QUALIFIED"
    assert result["tensors"] == 6
    canonical = tmp_path / "canonical.gguf"
    run_local_owned(
        [str(quantizer), str(source), str(canonical), "Q8_0", "2"],
    )
    expected = {t.name: t for t in GGUFReader(str(canonical), "r").tensors}
    actual = {t.name: t for t in GGUFReader(str(combined), "r").tensors}
    assert set(actual) == set(expected)
    assert all(
        a.tensor_type == expected[name].tensor_type
        and list(a.shape) == list(expected[name].shape)
        and chunk.hash_tensor_data(a.data) == chunk.hash_tensor_data(expected[name].data)
        for name, a in actual.items()
    )


@pytest.mark.skipif(
    os.name != "nt" or not Path("C:/llama.cpp/bin/llama-quantize.exe").is_file(),
    reason="pinned local Windows quantizer unavailable",
)
def test_owned_chunked_q8_pipeline_publishes_only_after_all_tensor_proofs(
    tmp_path, monkeypatch,
):
    import json
    import shutil
    from chowder_batch import reap_gguf_supervised as supervise
    from chowder_batch.reap_gguf_q4_chunked import _require_no_competing_quantizer

    try:
        _require_no_competing_quantizer()
    except ValueError:
        pytest.skip("another registered conversion currently owns the CPU quantizer")

    source = tmp_path / "source.gguf"
    _small_bf16(source)
    part_directory = tmp_path / "parts"
    scratch_directory = tmp_path / "scratch"
    part_directory.mkdir()
    scratch_directory.mkdir()
    output = tmp_path / "published.gguf"
    source_sha = chunk.hash_file(source)
    full_proof = tmp_path / "full-verification.json"
    full_proof.write_text(json.dumps({
        "schema": "chowder.reap128-v2-pruned-bf16-verification.v1",
        "status": "FULL_SOURCE_TENSOR_EQUIVALENCE_PASS",
        "candidate_shard_sha256": {str(i): "a" * 64 for i in range(26)},
    }))
    stage = {
        "schema": "chowder.reap-gguf-stage.v1", "phase": 0,
        "sha256": source_sha, "output": str(source),
        "size_bytes": source.stat().st_size,
        "input_sha256": chunk.hash_file(full_proof),
        "spec_sha256": "a" * 64,
        "status": "PUBLISHED_GGUF_NOT_QUALITY_QUALIFIED",
    }
    (tmp_path / (source.name + ".reap-stage.json")).write_text(json.dumps(stage))

    real_disk = shutil.disk_usage
    monkeypatch.setattr(
        shutil, "disk_usage",
        lambda path: SimpleNamespace(free=100 * chunk.GIB),
    )
    monkeypatch.setattr(supervise, "resource_problems", lambda *a, **k: [])

    def run_local_owned(command, **kwargs):
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=120,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
            cwd=Path(__file__).resolve().parents[1],
        )
        assert result.returncode == 0, result.stderr
        return result.returncode

    monkeypatch.setattr(supervise, "_owned_stage", run_local_owned)
    result = chunk.run_chunked(
        source=source, output=output,
        scratch=scratch_directory, chunks=part_directory,
        quantizer=Path("C:/llama.cpp/bin/llama-quantize.exe"),
        max_mib=8, slab_mib=1, parent_source_sha256=source_sha,
        full_verification=full_proof,
        expected_layers=2, expected_experts=16,
        log_path=tmp_path / "run.log",
    )
    assert result["status"] == "Q8_TENSOR_EXACT_CONSTRUCTED_NOT_QUALITY_QUALIFIED"
    assert result["chunk_count"] >= 2
    assert output.is_file()
    assert result["output_sha256"] == chunk.hash_file(output)
    assert output.with_name(output.name + ".reap-chunked.json").is_file()
    assert not list(part_directory.glob("*.gguf"))
    assert not list(scratch_directory.glob("*.bf16.gguf"))
    assert not list(scratch_directory.glob("merged-*.gguf"))
    assert list(part_directory.glob("*.receipt.json"))
    assert "CHUNKED_Q8_FINAL_PUBLISHED" in (tmp_path / "run.log").read_text()
    assert shutil.disk_usage is not real_disk  # patched guard, not a real disk purge
