"""Adversarial CPU-only source-bound mixed-K Q4 reconstruction tests."""

import hashlib
import json

import numpy as np
import pytest
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as Q

from chowder_batch import reap_gguf_q4_numeric_sample as numeric


def _fixture(tmp_path, monkeypatch):
    bf16 = tmp_path / "original.gguf"
    q4 = tmp_path / "model.gguf"
    names = [f"blk.{layer}.{suffix}" for layer in range(40)
             for suffix in ("ffn_gate_exps.weight", "ffn_up_exps.weight",
                            "ffn_down_exps.weight")]
    names += [f"other.{i:03d}.weight" for i in range(613)]
    for path, qfile in ((bf16, False), (q4, True)):
        writer = GGUFWriter(str(path), "qwen35moe")
        writer.add_file_type(15 if qfile else 32)
        writer.add_uint32("qwen35moe.block_count", 40)
        writer.add_uint32("qwen35moe.expert_count", 128)
        writer.add_uint32("qwen35moe.expert_used_count", 8)
        for index, name in enumerate(names):
            if index < 120:
                if qfile:
                    kind, size = ((Q.Q4_K, 144), (Q.Q5_K, 176),
                                  (Q.Q6_K, 210))[index % 3]
                    data = np.zeros((1, size), dtype="u1")
                else:
                    kind, data = Q.BF16, np.zeros((1, 512), dtype="u1")
            else:
                kind = Q.F32
                data = np.zeros((128,), dtype="<f4")
            writer.add_tensor(name, data, raw_dtype=kind)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
    hash_of = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(numeric.q8, "FROZEN_SOURCE_SHA256", hash_of(bf16))
    monkeypatch.setattr(numeric.q8, "FROZEN_SPEC_SHA256", "c" * 64)
    (tmp_path / (bf16.name + ".reap-stage.json")).write_text(json.dumps({
        "schema": "chowder.reap-gguf-stage.v1", "phase": 0,
        "output": str(bf16.resolve()), "sha256": hash_of(bf16),
        "size_bytes": bf16.stat().st_size,
        "spec_sha256": "c" * 64,
        "status": "PUBLISHED_GGUF_NOT_QUALITY_QUALIFIED",
    }))
    proof = {
        t.name: {"shape": list(map(int, t.shape)),
                 "tensor_type": t.tensor_type.name,
                 "raw_sha256": numeric.core.hash_tensor_span(
                     q4, int(t.data_offset), int(t.n_bytes))}
        for t in GGUFReader(str(q4), "r").tensors
    }
    (tmp_path / (q4.name + ".reap-chunked.json")).write_text(json.dumps({
        "schema": numeric.Q4_SCHEMA,
        "status": "Q4_K_M_TENSOR_EXACT_CONSTRUCTED_NOT_QUALITY_QUALIFIED",
        "source_gguf_sha256": hash_of(bf16), "output": str(q4.resolve()),
        "output_size_bytes": q4.stat().st_size, "output_sha256": hash_of(q4),
        "frozen_stage_spec_sha256": "c" * 64,
        "quantizer_binary_sha256": numeric.BINARY_SHA256,
        "chunk_count": 3,
        "chunked_reference_receipts": {
            "all_tensors": proof,
            "source_groups": [names[:250], names[250:500], names[500:]],
        },
        "quality_gate": "NOT_EVALUATED", "gpu_fit_gate": "NOT_EVALUATED",
    }))
    return bf16, q4


def test_full_mixed_k_q4_sample_is_descriptive_and_bounded(tmp_path, monkeypatch):
    bf16, q4 = _fixture(tmp_path, monkeypatch)
    res = numeric.measure(bf16, q4)
    assert res["status"] == "DESCRIPTIVE_Q4_QUANTIZATION_SAMPLE_NOT_QUALITY_GATE"
    assert res["verified_tensor_count"] == 733
    assert res["compared_expert_tensors"] == 120
    assert res["total_compared_values"] == 120 * 256
    assert res["max_abs_error"] == 0
    assert {x["type"] for x in res["samples"]} == {"Q4_K", "Q5_K", "Q6_K"}


@pytest.mark.parametrize("tamper", ["source", "output", "quantizer", "type", "hash",
                                     "groups", "gate", "mode"])
def test_q4_full_witness_refuses_forged_or_mutated_evidence(tmp_path, monkeypatch,
                                                              tamper):
    bf16, q4 = _fixture(tmp_path, monkeypatch)
    sidecar = tmp_path / (q4.name + ".reap-chunked.json")
    data = json.loads(sidecar.read_text())
    if tamper == "source":
        data["source_gguf_sha256"] = "a" * 64
    elif tamper == "output":
        data["output_sha256"] = "a" * 64
    elif tamper == "quantizer":
        data["quantizer_binary_sha256"] = "a" * 64
    elif tamper == "type":
        data["chunked_reference_receipts"]["all_tensors"]["blk.0.ffn_gate_exps.weight"]["tensor_type"] = "Q8_0"
    elif tamper == "hash":
        data["chunked_reference_receipts"]["all_tensors"]["blk.0.ffn_gate_exps.weight"]["raw_sha256"] = "a" * 64
    elif tamper == "groups":
        data["chunked_reference_receipts"]["source_groups"][0][0] = "extra.forged"
    elif tamper == "gate":
        data["quality_gate"] = "PASS"
    elif tamper == "mode":
        data["schema"] = "chowder.reap128-v2-gguf-q8-chunked-final.v1"
    sidecar.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="witness|SHA|groups|payload|geometry|mode"):
        numeric._verified_pair(bf16, q4)


def test_q4_refuses_mutated_published_tensor_bytes(tmp_path, monkeypatch):
    bf16, q4 = _fixture(tmp_path, monkeypatch)
    offset = int(GGUFReader(str(q4), "r").tensors[0].data_offset)
    with q4.open("r+b") as f:
        f.seek(offset)
        f.write(b"X")
    with pytest.raises(ValueError, match="SHA"):
        numeric._verified_pair(bf16, q4)


def test_q4_refuses_resealed_altered_copied_f32_source_payload(tmp_path, monkeypatch):
    bf16, q4 = _fixture(tmp_path, monkeypatch)
    name = "other.000.weight"
    tensor = next(t for t in GGUFReader(str(q4), "r").tensors if t.name == name)
    with q4.open("r+b") as stream:
        stream.seek(int(tensor.data_offset))
        stream.write(bytes([0x01]))
    sidecar = tmp_path / (q4.name + ".reap-chunked.json")
    receipt = json.loads(sidecar.read_text())
    receipt["output_sha256"] = hashlib.sha256(q4.read_bytes()).hexdigest()
    receipt["chunked_reference_receipts"]["all_tensors"][name]["raw_sha256"] = (
        numeric.core.hash_tensor_span(q4, int(tensor.data_offset), int(tensor.n_bytes))
    )
    sidecar.write_text(json.dumps(receipt))
    # The shared GGUF geometry preflight already performs a bytewise copied
    # F32 comparison; either it or our explicit source-span check may refuse.
    with pytest.raises(ValueError, match="tensor type/shape/payload|copied source tensor ancestry"):
        numeric._verified_pair(bf16, q4)
