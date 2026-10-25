"""Real pinned quantizer tests for Q4_K_M mini-group byte preservation."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType

from chowder_batch import reap_gguf_chunked as core
from chowder_batch import reap_gguf_q4_chunked as q4
from test_reap_gguf_chunked import _small_bf16


def test_q4_merge_refuses_incomplete_provenance(tmp_path):
    model = tmp_path / "bf16.gguf"
    _small_bf16(model)
    with pytest.raises(ValueError, match="fresh output and complete chunk"):
        q4.merge_q4(model, [], tmp_path / "absent.gguf", {})


def test_q4_refuses_a_running_q8_or_other_quantizer(monkeypatch):
    import psutil

    fake_owner = SimpleNamespace(info={
        "pid": os.getpid() + 1, "name": "python.exe",
        "cmdline": ["python", "-m", "chowder_batch.reap_gguf_chunked", "--execute"],
    })
    monkeypatch.setattr(psutil, "process_iter", lambda fields: [fake_owner])
    with pytest.raises(ValueError, match="concurrent Q8/Q4"):
        q4._require_no_competing_quantizer()
    fake_owner.info = {"pid": os.getpid() + 2, "name": "llama-quantize.exe",
                       "cmdline": []}
    with pytest.raises(ValueError, match="concurrent Q8/Q4"):
        q4._require_no_competing_quantizer()
    fake_owner.info = {"pid": os.getpid() + 3, "name": "llama-server.exe",
                       "cmdline": ["ollama", "serve"]}
    q4._require_no_competing_quantizer()
    fake_owner.info = {"pid": os.getpid() + 4, "name": "python.exe",
                       "cmdline": ["python", "chowder_batch/reap_gguf_chunked.py",
                                   "--execute"]}
    with pytest.raises(ValueError, match="concurrent Q8/Q4"):
        q4._require_no_competing_quantizer()


def _synthetic_ready(tmp_path):
    """Tiny legitimate GGUF shapes with mixed Q4/Q5/Q6 and copied F32."""
    source = tmp_path / "original.gguf"
    writer = GGUFWriter(str(source), "qwen35moe")
    writer.add_file_type(32)
    originals = {}
    for number, name in enumerate(("blk.0.ffn_gate_exps.weight",
                                   "blk.0.ffn_up_exps.weight",
                                   "blk.0.ffn_down_exps.weight")):
        data = np.arange(512, dtype=np.uint8).reshape(1, 512).copy()
        writer.add_tensor(name, data, raw_dtype=GGMLQuantizationType.BF16)
        originals[name] = (data, number)
    writer.add_tensor("blk.0.ffn_gate_inp_shexp.weight",
                      np.arange(2048, dtype="<f4").reshape(1, 2048),
                      raw_dtype=GGMLQuantizationType.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    scratch = tmp_path / "scratch"
    chunks = tmp_path / "parts"
    scratch.mkdir()
    chunks.mkdir()
    assembly = scratch / ("merged-q4-" + "1" * 32 + ".gguf")
    target = GGUFWriter(str(assembly), "qwen35moe")
    target.add_file_type(15)
    type_bytes = [
        (GGMLQuantizationType.Q4_K, 144),
        (GGMLQuantizationType.Q5_K, 176),
        (GGMLQuantizationType.Q6_K, 210),
    ]
    for name, (_, idx) in originals.items():
        kind, size = type_bytes[idx]
        target.add_tensor(name, np.full((1, size), idx + 1, dtype=np.uint8),
                          raw_dtype=kind)
    target.add_tensor("blk.0.ffn_gate_inp_shexp.weight",
                      np.arange(2048, dtype="<f4"), raw_dtype=GGMLQuantizationType.F32)
    target.write_header_to_file()
    target.write_kv_data_to_file()
    target.write_tensors_to_file()
    target.close()
    actual = GGUFReader(str(assembly), "r")
    proofs = {t.name: {"shape": list(map(int, t.shape)),
                       "tensor_type": t.tensor_type.name,
                       "raw_sha256": core.hash_tensor_data(t.data)}
              for t in actual.tensors}
    return source, scratch, chunks, assembly, proofs


def _seal_synthetic(source, scratch, chunks, assembly, proofs, output,
                    groups, identity):
    """Write small per-group GGUF outputs and complete source-bound receipts."""
    part_tensors = {t.name: t for t in GGUFReader(str(assembly), "r").tensors}
    source_tensors = {t.name: t for t in GGUFReader(str(source), "r").tensors}
    parts = []
    for index, names in enumerate(groups):
        basename = f"chunk-{index+1:03d}-of-{len(groups):03d}"
        part = chunks / (basename + ".q4_k_m.gguf")
        writer = GGUFWriter(str(part), "qwen35moe")
        writer.add_file_type(15)
        for name in names:
            t = part_tensors[name]
            writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        record = {
            "schema": "chowder.reap128-v2-gguf-chunk.v1",
            "identity": {**identity, "index": index, "names": names},
            "output": str(part), "output_size_bytes": part.stat().st_size,
            "output_sha256": core.hash_file(part), "source_mini_sha256": "a" * 64,
            "input_tensor_provenance": {
                name: {"shape": list(map(int, source_tensors[name].shape)),
                       "tensor_type": source_tensors[name].tensor_type.name,
                       "raw_sha256": core.hash_tensor_data(source_tensors[name].data)}
                for name in names
            },
            "output_tensor_provenance": {name: proofs[name] for name in names},
            "effective_quant_command": [
                str(source.parent / "synthetic-quantizer.exe"),
                "--max-buffer-size", "1", str(scratch / (basename + ".bf16.gguf")),
                str(part), "Q4_K_M", "2",
            ],
            "status": "CHUNK_Q4_K_M_BITWISE_WITNESSED_NOT_QUALITY_GATE",
        }
        (chunks / (basename + ".receipt.json")).write_text(json.dumps(record))
        parts.append(part)
    q4._seal_finalization(
        scratch=scratch, source=source, output=output, assembly=assembly,
        combined={"mode": "Q4_K_M", "source": str(source),
                  "output": str(assembly), "tensors": len(proofs),
                  "output_sha256": core.hash_file(assembly)},
        identity=identity, groups=groups, tensor_proofs=proofs, parts=parts,
        quantizer=source.parent / "synthetic-quantizer.exe",
    )
    return parts


def test_q4_durable_finalization_survives_deleted_chunks_and_publishes(tmp_path, monkeypatch):
    source, scratch, chunks, assembly, proofs = _synthetic_ready(tmp_path)
    output = tmp_path / "target.gguf"
    identity = {"source_sha256": core.hash_file(source), "stage_spec_sha256": "a" * 64,
                "quantizer_sha256": q4.BINARY_SHA256, "mode": q4.MODE,
                "max_mib": 8, "slab_mib": 1, "chunk_count": 2}
    groups = [["blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"],
              ["blk.0.ffn_down_exps.weight", "blk.0.ffn_gate_inp_shexp.weight"]]
    stage = {"spec_sha256": "a" * 64}
    combined = {"output_sha256": core.hash_file(assembly)}
    parts = _seal_synthetic(source, scratch, chunks, assembly, proofs, output,
                            groups, identity)
    ledger = q4._finalization_path(scratch, output)
    assert ledger.is_file()
    parts[0].unlink()  # simulate crash after first owned part deletion
    monkeypatch.setattr(q4, "disk_usage", lambda path: SimpleNamespace(free=200 * q4.GIB))
    monkeypatch.setattr("chowder_batch.reap_gguf_supervised.resource_problems",
                        lambda *args, **kwargs: [])
    result = q4._recover_finalization(scratch=scratch, chunks=chunks, source=source,
                                      output=output, identity=identity, groups=groups,
                                      stage=stage, log=io.StringIO(),
                                      quantizer=source.parent / "synthetic-quantizer.exe")
    assert core.hash_file(output) == combined["output_sha256"]
    assert result["status"] == "Q4_K_M_TENSOR_EXACT_CONSTRUCTED_NOT_QUALITY_QUALIFIED"
    assert result["chunked_reference_receipts"]["all_tensors"] == proofs
    assert not assembly.exists()
    assert json.loads((output.with_name(output.name + ".reap-chunked.json")).read_text()) == result
    assert q4._recover_finalization(scratch=scratch, chunks=chunks, source=source,
                                     output=output, identity=identity, groups=groups,
                                     stage=stage, log=io.StringIO(),
                                     quantizer=source.parent / "synthetic-quantizer.exe") == result


def test_q4_finalization_refuses_tampered_assembly_even_after_chunks_removed(tmp_path):
    source, scratch, chunks, assembly, proofs = _synthetic_ready(tmp_path)
    output = tmp_path / "result.gguf"
    identity = {"source_sha256": core.hash_file(source), "stage_spec_sha256": "a" * 64,
                "quantizer_sha256": q4.BINARY_SHA256, "mode": q4.MODE,
                "max_mib": 8, "slab_mib": 1, "chunk_count": 1}
    groups = [list(proofs)]
    _seal_synthetic(source, scratch, chunks, assembly, proofs, output,
                    groups, identity)
    with assembly.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        stream.write(b"X")
    with pytest.raises(ValueError, match="assembled|assembly|digest|payload"):
        q4._recover_finalization(scratch=scratch, chunks=chunks, source=source,
                                  output=output, identity=identity, groups=groups,
                                  stage={"spec_sha256": "a" * 64}, log=io.StringIO(),
                                  quantizer=source.parent / "synthetic-quantizer.exe")
    assert not output.exists()


def test_q4_finalization_rejects_forged_output_tensor_shape_or_type(tmp_path):
    source, scratch, _chunks, assembly, proofs = _synthetic_ready(tmp_path)
    original = GGUFReader(str(source), "r")
    q4._validate_assembled_q4(original, assembly, proofs)
    f32_name = "blk.0.ffn_gate_inp_shexp.weight"
    src = next(t for t in original.tensors if t.name == f32_name)
    dst = next(t for t in GGUFReader(str(assembly), "r").tensors
               if t.name == f32_name)
    assert list(src.shape) == [2048, 1]
    assert list(dst.shape) == [2048]
    assert core.hash_tensor_data(src.data) == core.hash_tensor_data(dst.data)
    copied = {name: dict(value) for name, value in proofs.items()}
    copied["blk.0.ffn_gate_inp_shexp.weight"]["shape"] = [1024, 2]
    with pytest.raises(ValueError, match="Q4|tensor"):
        q4._validate_assembled_q4(original, assembly, copied)


@pytest.mark.parametrize("kind", ["input", "output"])
def test_q4_chunk_receipt_tamper_fails_before_finalization(tmp_path, kind):
    source, scratch, chunks, assembly, proofs = _synthetic_ready(tmp_path)
    identity = {"source_sha256": core.hash_file(source), "stage_spec_sha256": "a" * 64,
                "quantizer_sha256": q4.BINARY_SHA256, "mode": q4.MODE,
                "max_mib": 8, "slab_mib": 1, "chunk_count": 1}
    names = list(proofs)
    parts = _seal_synthetic(source, scratch, chunks, assembly, proofs,
                            tmp_path / "final.gguf", [names], identity)
    receipt = chunks / "chunk-001-of-001.receipt.json"
    record = json.loads(receipt.read_text())
    field = ("input_tensor_provenance" if kind == "input"
             else "output_tensor_provenance")
    record[field][names[0]]["raw_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="witness|provenance|payload"):
        q4._check_chunk_evidence(
            record=record, source_items={
                t.name: t for t in GGUFReader(str(source), "r").tensors
            },
            part=parts[0], group_identity={**identity, "index": 0, "names": names},
            quantizer=source.parent / "synthetic-quantizer.exe",
            mini=scratch / "chunk-001-of-001.bf16.gguf",
        )


@pytest.mark.parametrize("crash_point", ["partial_transfer", "before_final_receipt",
                                           "corrupt_prefix", "corrupt_witness"])
def test_q4_recovery_after_copy_and_publication_crashes(tmp_path, monkeypatch, crash_point):
    source, scratch, chunks, assembly, proofs = _synthetic_ready(tmp_path)
    output = tmp_path / "recover.gguf"
    identity = {"source_sha256": core.hash_file(source), "stage_spec_sha256": "a" * 64,
                "quantizer_sha256": q4.BINARY_SHA256, "mode": q4.MODE,
                "max_mib": 8, "slab_mib": 1, "chunk_count": 1}
    groups = [list(proofs)]
    parts = _seal_synthetic(source, scratch, chunks, assembly, proofs, output,
                            groups, identity)
    parts[0].unlink()
    if crash_point in ("partial_transfer", "corrupt_prefix"):
        transfer = q4._transfer_path(output, identity)
        partial = assembly.read_bytes()[: assembly.stat().st_size // 3]
        if crash_point == "corrupt_prefix":
            partial = b"X" + partial[1:]
        transfer.write_bytes(partial)
    elif crash_point == "before_final_receipt":
        shutil.copyfile(assembly, output)
    else:
        receipt = chunks / "chunk-001-of-001.receipt.json"
        record = json.loads(receipt.read_text())
        record["output_tensor_provenance"][groups[0][0]]["raw_sha256"] = "0" * 64
        receipt.write_text(json.dumps(record))
    monkeypatch.setattr(q4, "disk_usage", lambda path: SimpleNamespace(free=200 * q4.GIB))
    monkeypatch.setattr("chowder_batch.reap_gguf_supervised.resource_problems",
                        lambda *args, **kwargs: [])
    kwargs = {"scratch": scratch, "chunks": chunks, "source": source,
              "output": output, "identity": identity, "groups": groups,
              "stage": {"spec_sha256": "a" * 64}, "log": io.StringIO(),
              "quantizer": source.parent / "synthetic-quantizer.exe"}
    if crash_point in ("corrupt_prefix", "corrupt_witness"):
        with pytest.raises(ValueError, match="prefix|receipt|witness"):
            q4._recover_finalization(**kwargs)
        assert not output.exists()
    else:
        result = q4._recover_finalization(**kwargs)
        assert result["output_sha256"] == core.hash_file(output)
        assert not assembly.exists()


def test_q4_sealed_chunk_recovers_after_source_mini_cleanup_crash(tmp_path):
    source, scratch, chunks, assembly, proofs = _synthetic_ready(tmp_path)
    output = tmp_path / "mini-recovery.gguf"
    identity = {"source_sha256": core.hash_file(source), "stage_spec_sha256": "a" * 64,
                "quantizer_sha256": q4.BINARY_SHA256, "mode": q4.MODE,
                "max_mib": 8, "slab_mib": 1, "chunk_count": 1}
    groups = [list(proofs)]
    parts = _seal_synthetic(source, scratch, chunks, assembly, proofs,
                            output, groups, identity)
    part = parts[0]
    mini = scratch / "chunk-001-of-001.bf16.gguf"
    shutil.copyfile(source, mini)
    receipt = chunks / "chunk-001-of-001.receipt.json"
    record = json.loads(receipt.read_text())
    record["source_mini_sha256"] = core.hash_file(mini)
    receipt.write_text(json.dumps(record))
    q4._cleanup_sealed_mini(mini=mini, part=part, receipt=receipt,
                            group_identity={**identity, "index": 0, "names": groups[0]},
                            original=GGUFReader(str(source), "r"),
                            quantizer=Path(record["effective_quant_command"][0]))
    assert not mini.exists()


def test_q4_recovers_completed_merge_worker_without_requantization(tmp_path):
    source, scratch, _chunks, assembly, proofs = _synthetic_ready(tmp_path)
    witness = assembly.with_name(assembly.name + ".receipt.json")
    combined = {"mode": "Q4_K_M", "source": str(source), "output": str(assembly),
                "tensors": len(proofs), "output_sha256": core.hash_file(assembly)}
    witness.write_text(json.dumps(combined))
    assert q4._completed_merge_before_ledger(
        scratch=scratch, source=source, tensor_proofs=proofs,
    ) == (assembly, combined)
    witness.unlink()
    with pytest.raises(ValueError, match="ambiguous/incomplete"):
        q4._completed_merge_before_ledger(scratch=scratch, source=source,
                                          tensor_proofs=proofs)


def test_q4_uninspectable_python_quantizer_owner_refuses(monkeypatch):
    import psutil

    class Uninspectable:
        info = None

        def name(self):
            return "python.exe"

    def iter_processes(_fields):
        raise psutil.AccessDenied(pid=os.getpid() + 100)
        yield Uninspectable()

    # Process enumeration failures can hide the live owner: fail closed.
    monkeypatch.setattr(psutil, "process_iter", iter_processes)
    with pytest.raises(ValueError, match="uninspectable|concurrent"):
        q4._require_no_competing_quantizer()


@pytest.mark.skipif(
    os.name != "nt" or not Path("C:/llama.cpp/bin/llama-quantize.exe").is_file(),
    reason="real registered pinned Windows llama-quantize is unavailable",
)
def test_q4_minis_match_pinned_monolith_raw_tensors(tmp_path, monkeypatch):
    from chowder_batch import reap_gguf_supervised as supervisor

    try:
        q4._require_no_competing_quantizer()
    except ValueError:
        pytest.skip("another registered Q8/Q4 constructor currently owns the host")
    source = tmp_path / "source.gguf"
    _small_bf16(source)
    quantizer = Path("C:/llama.cpp/bin/llama-quantize.exe")
    groups = core.groups_for(GGUFReader(str(source), "r"), max_mib=8)
    assert len(groups) >= 2

    def run_small(command, **kwargs):
        p = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=120,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        )
        assert p.returncode == 0, p.stdout + p.stderr
        return p.returncode

    monkeypatch.setattr(supervisor, "_owned_stage", run_small)
    parts, receipts = [], {}
    for i, names in enumerate(groups):
        mini = tmp_path / f"mini-{i}.gguf"
        out = tmp_path / f"part-{i}.gguf"
        original = core.create_mini(source, mini, names)
        assert core.hash_file(mini) == original["source_mini_sha256"]
        quant = core.quantize_mini(
            mini, out, quantizer=quantizer, mode="Q4_K_M",
            memory_mib=1, log=io.StringIO(),
        )
        receipts.update(quant["quantized_tensors"])
        parts.append(out)
    assembled = tmp_path / "combined.gguf"
    result = q4.merge_q4(source, parts, assembled, receipts)
    assert result["tensors"] == 6
    assert result["mode"] == "Q4_K_M"
    assert GGUFReader(str(assembled), "r").get_field("general.file_type").contents() == 15
    canonical = tmp_path / "monolithic.gguf"
    run_small([str(quantizer), str(source), str(canonical), "Q4_K_M", "2"])
    expected = {t.name: t for t in GGUFReader(str(canonical), "r").tensors}
    actual = {t.name: t for t in GGUFReader(str(assembled), "r").tensors}
    assert set(actual) == set(expected)
    assert any(t.tensor_type == GGMLQuantizationType.Q4_K for t in actual.values())
    for name, tensor in actual.items():
        assert tensor.tensor_type == expected[name].tensor_type
        assert list(tensor.shape) == list(expected[name].shape)
        assert core.hash_tensor_data(tensor.data) == core.hash_tensor_data(expected[name].data)
