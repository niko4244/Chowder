"""Regression for source-bound independently quantized MoE GGUF tensors."""

from __future__ import annotations

import io
import os
import subprocess
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
            fp32 = rng.normal(0, 0.35, (8, 256, 256)).astype("<f4")
            packed = (fp32.view("<u4") >> 16).astype("<u2").view("u1")
            packed = packed.reshape((8, 256, 512))
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


@pytest.mark.skipif(
    os.name != "nt" or not Path("C:/llama.cpp/bin/llama-quantize.exe").is_file(),
    reason="pinned Windows llama-quantize executable unavailable",
)
def test_pinned_quantizer_independent_q8_minis_merge_bitwise_equal_to_monolith(
    tmp_path, monkeypatch,
):
    from chowder_batch import reap_gguf_supervised as supervise

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
