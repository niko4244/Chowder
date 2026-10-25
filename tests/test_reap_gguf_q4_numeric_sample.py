"""Adversarial CPU-only source-bound mixed-K Q4 reconstruction tests."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as Q

from chowder_batch import reap_gguf_q4_numeric_sample as numeric
from chowder_batch import reap_gguf_q4_chunked as q4_chunked


def _sample_result():
    """Small, source-identified descriptive receipt for publication-only tests."""
    return {
        "status": "DESCRIPTIVE_Q4_QUANTIZATION_SAMPLE_NOT_QUALITY_GATE",
        "source_sha256": "a" * 64, "quantized_sha256": "b" * 64,
        "source": "C:/frozen/original.gguf", "model": "G:/frozen/q4.gguf",
        "verified_tensor_count": 733, "compared_expert_tensors": 120,
        "total_compared_values": 30720, "max_abs_error": 0.1,
        "mean_abs_error": 0.01, "root_mean_squared_error": 0.02,
    }


def _run_sample(tmp_path, monkeypatch, *, receipt_name="sample.json"):
    result = _sample_result()
    monkeypatch.setattr(numeric, "measure", lambda *_: result)
    out = tmp_path / receipt_name
    args = ["--bf16", str(tmp_path / "source.gguf"),
            "--q4", str(tmp_path / "quantized.gguf"), "--out", str(out)]
    return out, result, args


def test_q4_numeric_receipt_is_fsynced_before_exclusive_atomic_publish(
    tmp_path, monkeypatch, capsys,
):
    out, result, args = _run_sample(tmp_path, monkeypatch)
    real_fsync = q4_chunked.os.fsync
    observed = []

    def traced_fsync(fd):
        pending = list(tmp_path.glob(".sample.json.*.pending"))
        assert len(pending) == 1, "fsync must occur on the private staged receipt"
        assert not out.exists(), "the authoritative receipt must not exist before fsync"
        observed.append(json.loads(pending[0].read_text(encoding="utf-8")))
        return real_fsync(fd)

    monkeypatch.setattr(q4_chunked.os, "fsync", traced_fsync)
    assert numeric.main(args) == 0
    assert observed == [result]
    assert json.loads(out.read_text(encoding="utf-8")) == result
    assert not list(tmp_path.glob(".sample.json.*.pending"))
    assert "NOT_QUALITY_GATE" in capsys.readouterr().out


def test_q4_numeric_partial_write_failure_never_exposes_authoritative_receipt(
    tmp_path, monkeypatch, capsys,
):
    out, _result, args = _run_sample(tmp_path, monkeypatch)
    real_open = Path.open
    writes = []

    class InterruptedWrite:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *exc):
            return self.stream.__exit__(*exc)

        def write(self, content):
            self.stream.write(content[:20])
            writes.append(True)
            raise OSError("injected failure after writing an incomplete JSON prefix")

    def fail_open(self, mode="r", *args, **kwargs):
        stream = real_open(self, mode, *args, **kwargs)
        if self.parent == tmp_path and mode in ("x", "xb"):
            return InterruptedWrite(stream)
        return stream

    monkeypatch.setattr(Path, "open", fail_open)
    assert numeric.main(args) == 2
    assert writes == [True]
    assert not out.exists(), "partial JSON must never appear at the final path"
    assert not list(tmp_path.glob(".sample.json.*.pending"))
    assert "injected failure" in capsys.readouterr().err


def test_q4_numeric_publish_failure_cleans_fsync_staged_orphan(
    tmp_path, monkeypatch, capsys,
):
    out, _result, args = _run_sample(tmp_path, monkeypatch)
    real_rename = Path.rename
    real_link = q4_chunked.os.link

    def refuse_rename(self, target):
        if self.name.endswith(".pending") and Path(target) == out:
            raise OSError("injected atomic publication failure")
        return real_rename(self, target)

    def refuse_link(source, target, *args, **kwargs):
        if str(source).endswith(".pending") and Path(target) == out:
            raise OSError("injected atomic publication failure")
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(Path, "rename", refuse_rename)
    monkeypatch.setattr(q4_chunked.os, "link", refuse_link)
    assert numeric.main(args) == 2
    assert not out.exists()
    assert not list(tmp_path.glob(".sample.json.*.pending"))
    assert "injected atomic publication failure" in capsys.readouterr().err


def test_q4_numeric_receipt_never_overwrites_race_created_authoritative_path(
    tmp_path, monkeypatch,
):
    out, result, args = _run_sample(tmp_path, monkeypatch)
    original = "prior authoritative evidence\n"

    def race(_bf16, _q4):
        out.write_text(original, encoding="utf-8")
        return result

    monkeypatch.setattr(numeric, "measure", race)
    assert numeric.main(args) == 2
    assert out.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".sample.json.*.pending"))


def test_q4_numeric_stale_atomic_orphan_is_not_promoted_or_overwritten(
    tmp_path, monkeypatch,
):
    out, result, args = _run_sample(tmp_path, monkeypatch)
    orphan = tmp_path / ".sample.json.interrupted.pending"
    orphan.write_text('{"partial":', encoding="utf-8")
    assert numeric.main(args) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == result
    assert orphan.read_text(encoding="utf-8") == '{"partial":'
    assert list(tmp_path.glob(".sample.json.*.pending")) == [orphan]


def test_q4_numeric_main_publishes_source_bound_descriptive_evidence(
    tmp_path, monkeypatch, capsys,
):
    bf16, q4 = _fixture(tmp_path, monkeypatch)
    out = tmp_path / "numerical-sample.json"
    assert numeric.main(["--bf16", str(bf16), "--q4", str(q4),
                         "--out", str(out)]) == 0
    published = json.loads(out.read_text(encoding="utf-8"))
    assert published["status"] == "DESCRIPTIVE_Q4_QUANTIZATION_SAMPLE_NOT_QUALITY_GATE"
    assert published["source_sha256"] == hashlib.sha256(bf16.read_bytes()).hexdigest()
    assert published["quantized_sha256"] == hashlib.sha256(q4.read_bytes()).hexdigest()
    assert published["source"] == str(bf16.resolve())
    assert published["model"] == str(q4.resolve())
    assert published["verified_tensor_count"] == 733
    assert published["compared_expert_tensors"] == 120
    assert "NOT_QUALITY_GATE" in capsys.readouterr().out
    assert not list(tmp_path.glob(".numerical-sample.json.*.pending"))


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
