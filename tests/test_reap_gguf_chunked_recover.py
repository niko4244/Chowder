"""Recovery only from exact source-bound GGUF assembly and chunk audit bytes."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from gguf import GGUFReader

from chowder_batch import reap_gguf_chunked as core
from chowder_batch import reap_gguf_chunked_recover as recovery
from test_reap_gguf_quant_sample import _chunked_pair


def _write_json(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")


def _source_bound_setup(tmp_path: Path, monkeypatch):
    """Real on-disk 733-tensor synthetic BF16 and Q8, no quantizer process."""
    original_path, q8, final_receipt_path = _chunked_pair(tmp_path, monkeypatch)
    original = GGUFReader(str(original_path), "r")
    actual = GGUFReader(str(q8), "r")
    scratch, chunks = tmp_path / "private-scratch", tmp_path / "private-chunks"
    scratch.mkdir()
    chunks.mkdir()
    groups = core.groups_for(original, max_mib=8)
    assert len(groups) == 1
    spec = "c" * 64
    expected_binary = recovery.QUANTIZER_SHA
    signature = {
        "source_sha256": core.hash_file(original_path),
        "stage_spec_sha256": spec, "quantizer_sha256": expected_binary,
        "mode": "Q8_0", "slab_mib": 1, "max_mib": 8, "chunk_count": 1,
    }
    input_proofs = {
        t.name: {
            "shape": list(map(int, t.shape)),
            "tensor_type": t.tensor_type.name,
            "raw_sha256": core.hash_tensor_data(t.data),
        }
        for t in original.tensors
    }
    output_proofs = {
        t.name: {
            "shape": list(map(int, t.shape)),
            "tensor_type": t.tensor_type.name,
            "raw_sha256": core.hash_tensor_data(t.data),
        }
        for t in actual.tensors
    }
    part = chunks / "chunk-001-of-001.q8_0.gguf"
    shutil.copyfile(q8, part)
    part_record = {
        "schema": "chowder.reap128-v2-gguf-chunk.v1",
        "identity": {**signature, "index": 0, "names": groups[0]},
        "output": str(part), "output_size_bytes": part.stat().st_size,
        "output_sha256": core.hash_file(part),
        "input_tensor_provenance": input_proofs,
        "output_tensor_provenance": output_proofs,
        "effective_quant_command": [
            str(recovery.FROZEN_QUANTIZER_PATH.resolve()), "--max-buffer-size",
            "1", str(scratch / "chunk-001-of-001.bf16.gguf"), str(part), "Q8_0", "2",
        ],
        "status": "CHUNK_Q8_BITWISE_WITNESSED_NOT_QUALITY_GATE",
    }
    receipt_path = chunks / "chunk-001-of-001.receipt.json"
    _write_json(receipt_path, part_record)
    assembly = scratch / "merged-q8-synthetic.gguf"
    shutil.copyfile(q8, assembly)
    assembly_receipt = {
        "status": "GGUF_CHUNKED_TENSOR_BITWISE_MERGE_NOT_QUALITY_QUALIFIED",
        "mode": "Q8_0", "source": str(original_path),
        "output": str(assembly), "tensors": 733,
        "output_sha256": core.hash_file(assembly),
    }
    _write_json(scratch / (assembly.name + ".receipt.json"), assembly_receipt)
    _write_json(scratch / "merge-input-synthetic.json", output_proofs)
    log = tmp_path / "supervised.log"
    log.write_text(json.dumps({
        "event": "EXACT_SOURCE_BOUND_PART_CLEANED", "path": str(part),
        "sha256": core.hash_file(part),
    }) + "\n")
    part.unlink()
    final_receipt_path.unlink()  # only the synthetic fixture's claim, not a real published Q8
    return (original_path, original, scratch, chunks, assembly, log, part_record)


def test_recovery_exact_post_cleanup_733_tensor_merge_and_fp32_fold(tmp_path, monkeypatch):
    source, original, scratch, chunks, assembly, log, _ = _source_bound_setup(
        tmp_path, monkeypatch)
    stage = {"spec_sha256": "c" * 64}
    result = recovery._recheck_owned_assembly(
        source, scratch, chunks, original, stage, core.hash_file(source),
        log, max_mib=8, slab_mib=1)
    assembled, merged_receipt, groups, proofs = result
    assert assembled == assembly
    assert len(proofs) == 733
    assert len(groups) == 1
    assert merged_receipt["output_sha256"] == hashlib.sha256(assembly.read_bytes()).hexdigest()
    # A production-relevant FP32 [2048,1] -> [2048] fold exists in these GGUF bytes.
    after = {t.name: t for t in GGUFReader(str(assembly), "r").tensors}
    before = {t.name: t for t in original.tensors}
    assert list(before["extra.000.weight"].shape) == [2048, 1]
    assert list(after["extra.000.weight"].shape) == [2048]


def test_recovery_accepts_multiple_only_identical_old_merge_input_rosters(
    tmp_path, monkeypatch,
):
    """Aborted merge retries leave small old manifests, not extra GGUFs."""
    source, original, scratch, chunks, _assembly, log, _ = _source_bound_setup(
        tmp_path, monkeypatch)
    first = scratch / "merge-input-synthetic.json"
    shutil.copyfile(first, scratch / "merge-input-aborted-first-attempt.json")
    result = recovery._recheck_owned_assembly(
        source, scratch, chunks, original, {"spec_sha256": "c" * 64},
        core.hash_file(source), log, max_mib=8, slab_mib=1)
    assert len(result[3]) == 733
    forged = json.loads(first.read_text())
    forged[next(iter(forged))]["raw_sha256"] = "0" * 64
    (scratch / "merge-input-aborted-first-attempt.json").write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="merge tensor roster"):
        recovery._recheck_owned_assembly(
            source, scratch, chunks, original, {"spec_sha256": "c" * 64},
            core.hash_file(source), log, max_mib=8, slab_mib=1)


@pytest.mark.parametrize("tamper", [
    "no_cleanup_witness", "bad_cleanup_digest", "missing_chunk_receipt",
    "different_source_tensor", "different_quantized_tensor",
    "stale_merge_roster", "assembly_digest_corruption", "wrong_quantizer_command",
])
def test_recovery_rejects_unsealed_or_corrupted_evidence(tmp_path, monkeypatch, tamper):
    source, original, scratch, chunks, assembly, log, receipt = _source_bound_setup(
        tmp_path, monkeypatch)
    part_receipt = chunks / "chunk-001-of-001.receipt.json"
    if tamper in {"no_cleanup_witness", "bad_cleanup_digest"}:
        event = json.loads(log.read_text())
        if tamper == "no_cleanup_witness":
            log.write_text("")
        else:
            event["sha256"] = "0" * 64
            log.write_text(json.dumps(event) + "\n")
    elif tamper == "missing_chunk_receipt":
        part_receipt.unlink()
    elif tamper in {"different_source_tensor", "different_quantized_tensor"}:
        key = ("input_tensor_provenance" if tamper == "different_source_tensor"
               else "output_tensor_provenance")
        name = next(iter(receipt[key]))
        receipt[key][name]["raw_sha256"] = "0" * 64
        _write_json(part_receipt, receipt)
    elif tamper == "stale_merge_roster":
        roster = scratch / "merge-input-synthetic.json"
        payload = json.loads(roster.read_text())
        del payload[next(iter(payload))]
        _write_json(roster, payload)
    elif tamper == "assembly_digest_corruption":
        witness = scratch / (assembly.name + ".receipt.json")
        info = json.loads(witness.read_text())
        info["output_sha256"] = "0" * 64
        _write_json(witness, info)
    elif tamper == "wrong_quantizer_command":
        receipt["effective_quant_command"][-2] = "Q4_K_M"
        _write_json(part_receipt, receipt)
    with pytest.raises((ValueError, FileNotFoundError)):
        recovery._recheck_owned_assembly(
            source, scratch, chunks, original,
            {"spec_sha256": "c" * 64}, core.hash_file(source), log,
            max_mib=8, slab_mib=1)


def test_recovery_original_deletion_event_parser_denies_duplicates(tmp_path):
    log = tmp_path / "recovery.log"
    item = {"event": "EXACT_SOURCE_BOUND_PART_CLEANED",
            "path": str(tmp_path / "chunk-001.q8_0.gguf"),
            "sha256": "a" * 64}
    log.write_text(json.dumps(item) + "\n" + json.dumps(item) + "\n")
    with pytest.raises(ValueError, match="ambiguous"):
        recovery._cleanup_events(log)


def test_recovery_full_guarded_cross_volume_publication_from_verified_merge(
    tmp_path, monkeypatch,
):
    """Real tiny 733 GGUF header/payloads, no quantizer child or GPU work."""
    from chowder_batch import reap_gguf_supervised as supervisor

    source, _, scratch, chunks, assembly, log, _ = _source_bound_setup(
        tmp_path, monkeypatch)
    verification = tmp_path / "full-source-proof.json"
    _write_json(verification, {
        "schema": "chowder.reap128-v2-pruned-bf16-verification.v1",
        "status": "FULL_SOURCE_TENSOR_EQUIVALENCE_PASS",
        "candidate_shard_sha256": {str(i): "a" * 64 for i in range(26)},
    })
    stage_file = source.with_name(source.name + ".reap-stage.json")
    stage = json.loads(stage_file.read_text())
    stage["input_sha256"] = core.hash_file(verification)
    _write_json(stage_file, stage)
    monkeypatch.setattr(recovery, "FROZEN_BF16_SHA", core.hash_file(source))
    monkeypatch.setattr(recovery, "FROZEN_SPEC_SHA", "c" * 64)
    monkeypatch.setattr(recovery, "FROZEN_PROOF_SHA", core.hash_file(verification))
    # Local pytest path lies on one physical test volume; production strictly
    # requires F: source+scratch and I: final+chunks.
    monkeypatch.setattr(recovery, "_independent_volume_plan", lambda *a: True)
    monkeypatch.setattr(recovery, "disk_usage", lambda path: SimpleNamespace(
        free=100 * core.GIB))
    monkeypatch.setattr(supervisor, "resource_problems", lambda *a, **kw: [])
    target = tmp_path / "restored-q8.gguf"
    kwargs = {
        "source": source, "output": target, "scratch": scratch,
        "chunks": chunks, "full_verification": verification,
        "supervision_log": log,
        "quantizer": Path("C:/llama.cpp/bin/llama-quantize.exe"),
        "source_sha": core.hash_file(source), "max_mib": 8, "slab_mib": 1,
    }
    preflight = recovery.recover_final(**kwargs, execute=False)
    assert preflight["status"] == "RECOVERY_PREFLIGHT_PASS_NOT_PUBLISHED"
    assert not target.exists()
    actual = recovery.recover_final(**kwargs, execute=True)
    assert actual["status"] == "Q8_TENSOR_EXACT_CONSTRUCTED_NOT_QUALITY_QUALIFIED"
    assert core.hash_file(target) == core.hash_file(assembly)
    saved = json.loads((tmp_path / (target.name + ".reap-chunked.json")).read_text())
    assert saved["output"] == str(target)
    assert saved["output_sha256"] == core.hash_file(target)
    assert saved["chunked_reference_receipts"]["source_groups"][0]
    assert len(saved["chunked_reference_receipts"]["all_tensors"]) == 733
    with pytest.raises(ValueError, match="original exclusive"):
        recovery.recover_final(**kwargs, execute=True)
