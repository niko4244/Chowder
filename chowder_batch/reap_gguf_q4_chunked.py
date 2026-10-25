"""Frozen BF16-derived Q4_K_M GGUF construction in bounded, SHA-sealed chunks.

Never start alongside the Q8 constructor or any other quantizer. All outputs
remain unqualified until independent frozen inference and GPU gates complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from shutil import disk_usage

from chowder_batch import reap_gguf_chunked as core

GIB = core.GIB
MIB = core.MIB
MODE = "Q4_K_M"
Q4_FILE_TYPE = 15
BINARY_SHA256 = "eeb75877bd41d8d7b51d18129d873e6220014613baea349fa18e6b5d36c987de"
FINALIZATION_SCHEMA = "chowder.reap128-v2-gguf-q4-durable-finalization.v1"
MERGE_COPY_BLOCK = core.HASH_BLOCK


def _require_no_competing_quantizer() -> None:
    """Fail closed while another Q8/Q4 supervisor or quantizer owns the host."""
    import psutil

    try:
        processes = list(psutil.process_iter(["pid", "name", "cmdline"]))
    except (psutil.Error, OSError) as exc:
        raise ValueError("uninspectable process census cannot exclude Q8 owner") from exc
    for process in processes:
        try:
            info = process.info
            if info["pid"] == os.getpid():
                continue
            name = (info.get("name") or "").lower()
            argv = " ".join(info.get("cmdline") or []).lower()
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except psutil.AccessDenied:
            try:
                owner_name = process.name().lower()
            except (psutil.Error, OSError) as exc:
                raise ValueError("uninspectable process may be Q8 quantizer owner") from exc
            if owner_name == "llama-quantize.exe" or owner_name.startswith("python"):
                raise ValueError("another uninspectable Python/quantizer owner is active") from None
            continue
        if (name == "llama-quantize.exe"
                or (name.startswith("python") and any(marker in argv for marker in (
                    "chowder_batch.reap_gguf_chunked",
                    "chowder_batch.reap_gguf_q4_chunked",
                    "reap_gguf_chunked.py", "reap_gguf_q4_chunked.py",
                )))):
            raise ValueError("concurrent Q8/Q4 quantization owner detected; refusing Q4")


def _finalization_path(scratch: Path, output: Path) -> Path:
    return scratch / ("." + output.name + ".q4-finalization.json")


def _transfer_path(output: Path, identity: dict) -> Path:
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()[:16]
    return output.with_name("." + output.name + ".q4-owned-" + digest + ".gguf")


def _publish_exclusive(path: Path, value: dict) -> None:
    """Publish a fully flushed record atomically, without replacing evidence."""
    pending = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".pending")
    try:
        with pending.open("xb") as stream:
            stream.write((json.dumps(value, sort_keys=True, indent=2,
                                     allow_nan=False) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        # Windows rename fails if destination already exists, and works on
        # exFAT/USB where hard links are unsupported. POSIX needs hard link
        # because rename() would replace an existing destination there.
        if os.name == "nt":
            pending.rename(path)
        else:
            os.link(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _safe_assembled_path(scratch: Path, candidate: Path) -> None:
    if (candidate.is_symlink() or not candidate.is_file()
            or candidate.parent.resolve() != scratch.resolve()
            or not candidate.name.startswith("merged-q4-")
            or not candidate.name.endswith(".gguf")):
        raise ValueError("Q4 assembled checkpoint must be an owned regular scratch GGUF")


def _validate_assembled_q4(original, assembly: Path, proofs: dict) -> None:
    """Independently verify every preserved quantized payload against its proof.

    The original BF16/F32 tensor also establishes the legal output geometry;
    F32 singleton folding is permitted only if its bytes are unchanged.
    """
    from gguf import GGUFReader, GGMLQuantizationType

    if assembly.is_symlink() or not assembly.is_file():
        raise ValueError("Q4 assembled payload is absent or untrusted")
    merged = GGUFReader(str(assembly), "r")
    if merged.get_field("general.file_type").contents() != Q4_FILE_TYPE:
        raise ValueError("Q4 assembled GGUF file type changed")
    source_items = {t.name: t for t in original.tensors}
    output_items = {t.name: t for t in merged.tensors}
    if (not isinstance(proofs, dict) or len(source_items) != len(original.tensors)
            or len(output_items) != len(merged.tensors)
            or set(proofs) != set(source_items) or set(output_items) != set(source_items)):
        raise ValueError("Q4 assembled tensor inventory differs from original")
    allowed = {GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q5_K,
               GGMLQuantizationType.Q6_K, GGMLQuantizationType.Q8_0,
               GGMLQuantizationType.F32, GGMLQuantizationType.F16,
               GGMLQuantizationType.BF16}
    if not any(t.tensor_type == GGMLQuantizationType.Q4_K for t in output_items.values()):
        raise ValueError("Q4 assembled GGUF has no Q4_K payload")
    for name, target in output_items.items():
        src = source_items[name]
        proof = proofs[name]
        if (not isinstance(proof, dict) or target.tensor_type not in allowed
                or src.tensor_type not in (GGMLQuantizationType.BF16,
                                           GGMLQuantizationType.F32)
                or not core._quantized_shape_matches(src, target)
                or proof.get("shape") != list(map(int, target.shape))
                or proof.get("tensor_type") != target.tensor_type.name
                or proof.get("raw_sha256") != core.hash_tensor_data(target.data)):
            raise ValueError(f"Q4 assembled tensor shape/type/payload inconsistent: {name}")


def _check_chunk_evidence(
    *, record: dict, source_items: dict, part: Path,
    group_identity: dict, quantizer: Path, mini: Path,
) -> None:
    """Rebuild the two sides of a chunk witness from actual GGUF bytes."""
    from gguf import GGUFReader, GGMLQuantizationType

    _input_proofs, output_proofs = _check_receipt_binding(
        record=record, source_items=source_items,
        group_identity=group_identity, quantizer=quantizer, mini=mini, part=part,
    )
    names = group_identity["names"]
    tensors = GGUFReader(str(part), "r").tensors
    actual = {t.name: t for t in tensors}
    if len(actual) != len(tensors) or set(actual) != set(names):
        raise ValueError("Q4 chunk GGUF tensor inventory disagrees with trusted witness")
    allowed = {GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q5_K,
               GGMLQuantizationType.Q6_K, GGMLQuantizationType.Q8_0,
               GGMLQuantizationType.F32, GGMLQuantizationType.F16,
               GGMLQuantizationType.BF16}
    for name in names:
        src, quant = source_items[name], actual[name]
        output_expected = {"shape": list(map(int, quant.shape)),
                           "tensor_type": quant.tensor_type.name,
                           "raw_sha256": core.hash_tensor_data(quant.data)}
        if (output_proofs[name] != output_expected
                or quant.tensor_type not in allowed
                or not core._quantized_shape_matches(src, quant)):
            raise ValueError(f"Q4 chunk trusted tensor provenance forged or incompatible: {name}")


def _check_receipt_binding(*, record: dict, source_items: dict,
                           group_identity: dict, quantizer: Path,
                           mini: Path, part: Path) -> tuple[dict, dict]:
    """Validate persisted provenance even when an authorized chunk was deleted."""
    if (record.get("schema") != "chowder.reap128-v2-gguf-chunk.v1"
            or record.get("identity") != group_identity
            or record.get("output") != str(part)
            or not isinstance(record.get("output_sha256"), str)
            or len(record["output_sha256"]) != 64
            or type(record.get("output_size_bytes")) is not int
            or record["output_size_bytes"] <= 0
            or record.get("status") != "CHUNK_Q4_K_M_BITWISE_WITNESSED_NOT_QUALITY_GATE"):
        raise ValueError("Q4 trusted chunk schema/source/size/status invalid")
    command = [str(quantizer), "--max-buffer-size",
               str(group_identity["slab_mib"]), str(mini), str(part), MODE, "2"]
    if record.get("effective_quant_command") != command:
        raise ValueError("Q4 trusted chunk quantization command changed")
    input_proofs = record.get("input_tensor_provenance")
    output_proofs = record.get("output_tensor_provenance")
    names = group_identity["names"]
    if (not isinstance(input_proofs, dict) or set(input_proofs) != set(names)
            or not isinstance(output_proofs, dict) or set(output_proofs) != set(names)
            or not isinstance(record.get("source_mini_sha256"), str)
            or len(record["source_mini_sha256"]) != 64):
        raise ValueError("Q4 chunk tensor witness inventory incomplete")
    for name in names:
        src = source_items[name]
        input_expected = {"shape": list(map(int, src.shape)),
                          "tensor_type": src.tensor_type.name,
                          "raw_sha256": core.hash_tensor_data(src.data)}
        if (input_proofs[name] != input_expected
                or not isinstance(output_proofs[name], dict)
                or set(output_proofs[name]) != {"shape", "tensor_type", "raw_sha256"}):
            raise ValueError(f"Q4 trusted BF16 input or output witness changed: {name}")
    return input_proofs, output_proofs


def _cleanup_sealed_mini(*, mini: Path, part: Path, receipt: Path,
                         group_identity: dict, original, quantizer: Path) -> None:
    """Recover a crash after sealing the quantized chunk but before mini deletion."""
    from gguf import GGUFReader

    record = core._load_chunk_record(receipt, part, expected=group_identity)
    source_items = {t.name: t for t in original.tensors}
    _check_chunk_evidence(
        record=record, source_items=source_items, part=part,
        group_identity=group_identity, quantizer=quantizer, mini=mini,
    )
    if (mini.is_symlink() or not mini.is_file()
            or core.hash_file(mini) != record["source_mini_sha256"]):
        raise ValueError("Q4 retained source mini differs from sealed receipt")
    def verify_then_release_mapping() -> None:
        # Windows rejects unlink while any GGUF tensor mmap remains referenced.
        small = GGUFReader(str(mini), "r")
        actual = {t.name: t for t in small.tensors}
        if (len(actual) != len(small.tensors)
                or set(actual) != set(group_identity["names"])
                or small.get_field("general.file_type").contents() != 32):
            raise ValueError("Q4 retained source mini tensor inventory invalid")
        for name, tensor in actual.items():
            original_tensor = source_items[name]
            if (tensor.tensor_type != original_tensor.tensor_type
                    or list(tensor.shape) != list(original_tensor.shape)
                    or core.hash_tensor_data(tensor.data) !=
                    core.hash_tensor_data(original_tensor.data)):
                raise ValueError("Q4 retained source mini tensor differs from original")

    verify_then_release_mapping()
    mini.unlink()


def _completed_merge_before_ledger(*, scratch: Path, source: Path,
                                   tensor_proofs: dict) -> tuple[Path, dict] | None:
    """Recover a verified merge worker completed before the parent sealed it.

    Preserve incomplete/unidentified outputs for audit and fail closed instead
    of silently filling a scratch disk with repeated full-sized assemblies.
    """
    from gguf import GGUFReader

    merges = list(scratch.glob("merged-q4-*.gguf"))
    witnesses = list(scratch.glob("merged-q4-*.gguf.receipt.json"))
    if not merges and not witnesses:
        return None
    if (len(merges) != 1 or len(witnesses) != 1
            or witnesses[0] != merges[0].with_name(merges[0].name + ".receipt.json")
            or witnesses[0].is_symlink() or witnesses[0].stat().st_size > 4 * MIB):
        raise ValueError("Q4 unsealed merge worker artifacts ambiguous/incomplete")
    assembly = merges[0]
    _safe_assembled_path(scratch, assembly)
    combined = json.loads(witnesses[0].read_text(encoding="utf-8"))
    if (combined.get("source") != str(source)
            or combined.get("output") != str(assembly)
            or combined.get("mode") != MODE
            or combined.get("tensors") != len(tensor_proofs)
            or combined.get("output_sha256") != core.hash_file(assembly)):
        raise ValueError("Q4 unsealed merge worker receipt/source SHA mismatch")
    _validate_assembled_q4(GGUFReader(str(source), "r"), assembly, tensor_proofs)
    return assembly, combined


def _seal_finalization(*, scratch: Path, source: Path, output: Path,
                       assembly: Path, combined: dict, identity: dict,
                       groups: list[list[str]], tensor_proofs: dict,
                       parts: list[Path], quantizer: Path) -> None:
    """Durably seal the verified F: merge BEFORE deleting a single G: chunk."""
    from gguf import GGUFReader

    _safe_assembled_path(scratch, assembly)
    original = GGUFReader(str(source), "r")
    _validate_assembled_q4(original, assembly, tensor_proofs)
    if (combined.get("output") != str(assembly)
            or combined.get("source") != str(source)
            or combined.get("mode") != MODE
            or combined.get("tensors") != len(tensor_proofs)
            or combined.get("output_sha256") != core.hash_file(assembly)):
        raise ValueError("Q4 assembly worker receipt does not bind source/payload")
    if (not parts or len(parts) != len(groups)
            or identity.get("chunk_count") != len(groups)):
        raise ValueError("Q4 finalization requires every verified chunk receipt")
    source_items = {t.name: t for t in original.tensors}
    chunks = []
    for index, (names, part) in enumerate(zip(groups, parts)):
        receipt = part.with_name(part.stem.replace(".q4_k_m", "") + ".receipt.json")
        group_id = {**identity, "index": index, "names": names}
        record = core._load_chunk_record(receipt, part, expected=group_id)
        mini = scratch / (f"chunk-{index+1:03d}-of-{len(groups):03d}.bf16.gguf")
        _check_chunk_evidence(record=record, source_items=source_items,
                              part=part, group_identity=group_id,
                              quantizer=quantizer, mini=mini)
        chunks.append({"path": str(part), "receipt": str(receipt),
                       "sha256": record["output_sha256"],
                       "receipt_sha256": core.hash_file(receipt)})
    _publish_exclusive(_finalization_path(scratch, output), {
        "schema": FINALIZATION_SCHEMA,
        "identity": identity, "source": str(source), "output": str(output),
        "quantizer_path": str(quantizer),
        "assembly": str(assembly), "assembly_sha256": combined["output_sha256"],
        "assembly_size": assembly.stat().st_size,
        "groups": groups, "tensor_proofs": tensor_proofs,
        "chunks": chunks,
    })


def _final_result(*, source_sha: str, output: Path, assembly_sha: str,
                  stage: dict, identity: dict, groups: list[list[str]],
                  tensor_proofs: dict) -> dict:
    return {
        "schema": "chowder.reap128-v2-gguf-q4-chunked-final.v1",
        "status": "Q4_K_M_TENSOR_EXACT_CONSTRUCTED_NOT_QUALITY_QUALIFIED",
        "source_gguf_sha256": source_sha, "output": str(output),
        "output_sha256": assembly_sha, "output_size_bytes": output.stat().st_size,
        "frozen_stage_spec_sha256": stage["spec_sha256"],
        "quantizer_binary_sha256": BINARY_SHA256,
        "chunk_count": len(groups),
        "chunked_reference_receipts": {
            "all_tensors": tensor_proofs, "source_groups": groups,
        },
        "quality_gate": "NOT_EVALUATED", "gpu_fit_gate": "NOT_EVALUATED",
    }


def _recover_finalization(*, scratch: Path, chunks: Path, source: Path,
                          output: Path, identity: dict, groups: list[list[str]],
                          stage: dict, log, quantizer: Path) -> dict:
    """Resume from verified assembly, including crashes during deletion/copy/publish."""
    from gguf import GGUFReader
    from chowder_batch.reap_gguf_supervised import resource_problems

    ledger_file = _finalization_path(scratch, output)
    if (ledger_file.is_symlink() or not ledger_file.is_file()
            or ledger_file.stat().st_size > 4 * MIB):
        raise ValueError("Q4 recovered finalization ledger missing or untrusted")
    ledger = json.loads(ledger_file.read_text(encoding="utf-8"))
    if (ledger.get("schema") != FINALIZATION_SCHEMA
            or ledger.get("identity") != identity
            or ledger.get("source") != str(source)
            or ledger.get("output") != str(output)
            or ledger.get("quantizer_path") != str(quantizer)
            or ledger.get("groups") != groups
            or len(ledger.get("chunks", [])) != len(groups)):
        raise ValueError("Q4 finalization identity/source/chunk count changed")
    assembly = Path(ledger["assembly"])
    if (assembly.parent.resolve() != scratch.resolve()
            or not assembly.name.startswith("merged-q4-")
            or not assembly.name.endswith(".gguf")):
        raise ValueError("Q4 recovered assembled checkpoint path is untrusted")
    proof = ledger.get("tensor_proofs")
    original = GGUFReader(str(source), "r")
    source_items = {t.name: t for t in original.tensors}
    final_receipt = output.with_name(output.name + ".reap-chunked.json")
    if assembly.exists():
        _safe_assembled_path(scratch, assembly)
        if (assembly.stat().st_size != ledger["assembly_size"]
                or core.hash_file(assembly) != ledger["assembly_sha256"]):
            raise ValueError("Q4 assembled checkpoint size/digest mismatch")
        _validate_assembled_q4(original, assembly, proof)
    elif not (output.is_file() and final_receipt.is_file()):
        raise ValueError("Q4 verified assembly missing before final publication")
    else:
        if output.is_symlink() or core.hash_file(output) != ledger["assembly_sha256"]:
            raise ValueError("Q4 published output differs from checkpoint")
        _validate_assembled_q4(original, output, proof)

    for index, entry in enumerate(ledger["chunks"]):
        part = Path(entry["path"])
        receipt = Path(entry["receipt"])
        basename = f"chunk-{index+1:03d}-of-{len(groups):03d}"
        if (part != chunks / (basename + ".q4_k_m.gguf")
                or receipt != chunks / (basename + ".receipt.json")
                or receipt.is_symlink() or not receipt.is_file()
                or receipt.stat().st_size > 2 * MIB
                or core.hash_file(receipt) != entry["receipt_sha256"]):
            raise ValueError("Q4 deleted-chunk audit receipt changed")
        record = json.loads(receipt.read_text(encoding="utf-8"))
        group_identity = {**identity, "index": index, "names": groups[index]}
        _check_receipt_binding(
            record=record, source_items=source_items, group_identity=group_identity,
            part=part, quantizer=quantizer,
            mini=scratch / (basename + ".bf16.gguf"),
        )
        if (record["output_sha256"] != entry["sha256"]
                or record["output_tensor_provenance"] != {
                    name: proof[name] for name in groups[index]
                }):
            raise ValueError("Q4 deleted-chunk receipt/provenance differs from assembly")
        if part.exists() or part.is_symlink():
            checked = core._load_chunk_record(
                receipt, part, expected={**identity, "index": index,
                                         "names": groups[index]},
            )
            if checked["output_sha256"] != entry["sha256"]:
                raise ValueError("Q4 retained chunk SHA disagrees with finalization")
            mini = scratch / (basename + ".bf16.gguf")
            _check_chunk_evidence(
                record=record, source_items=source_items,
                part=part, group_identity={**identity, "index": index,
                                           "names": groups[index]},
                quantizer=quantizer, mini=mini,
            )
            if not assembly.exists():
                raise ValueError("Q4 cannot clean parts without verified assembly")
            part.unlink()
            log.write(json.dumps({"event": "Q4_VERIFIED_CHUNK_RECOVERY_CLEANUP",
                                  "index": index, "sha256": entry["sha256"]}) + "\n")
            log.flush()
    if output.exists() or output.is_symlink():
        if (output.is_symlink() or not output.is_file()
                or output.stat().st_size != ledger["assembly_size"]
                or core.hash_file(output) != ledger["assembly_sha256"]):
            raise ValueError("Q4 published output differs from durable sealed assembly")
    else:
        if final_receipt.exists() or final_receipt.is_symlink():
            raise ValueError("Q4 orphaned final receipt without output")
        # Deterministic private transfer permits safe byte-prefix verification
        # and append after interruption, rather than another whole-file copy.
        transfer = _transfer_path(output, identity)
        if transfer.is_symlink() or (transfer.exists() and not transfer.is_file()):
            raise ValueError("Q4 transfer checkpoint is not an owned regular file")
        offset = transfer.stat().st_size if transfer.exists() else 0
        if offset > ledger["assembly_size"]:
            raise ValueError("Q4 transfer checkpoint exceeds assembled payload")
        if disk_usage(output.parent).free < ledger["assembly_size"] - offset + 6 * GIB:
            raise ValueError("Q4 final transfer cannot preserve required disk reserve")
        with assembly.open("rb") as src, transfer.open("r+b" if offset else
                                                      ("r+b" if transfer.exists() else "xb")) as dst:
            for _ in range(0, offset, core.HASH_BLOCK):
                count = min(core.HASH_BLOCK, offset - src.tell())
                if src.read(count) != dst.read(count):
                    raise ValueError("Q4 transfer checkpoint prefix differs from assembly")
            dst.seek(offset)
            for block in iter(lambda: src.read(core.HASH_BLOCK), b""):
                if resource_problems([assembly, transfer]):
                    raise RuntimeError("Q4 final transfer aborted by host guard")
                dst.write(block)
            dst.flush()
            os.fsync(dst.fileno())
        if core.hash_file(transfer) != ledger["assembly_sha256"]:
            raise ValueError("Q4 owned transfer digest differs from assembly")
        transfer.rename(output)
    _validate_assembled_q4(original, output, proof)
    result = _final_result(source_sha=identity["source_sha256"], output=output,
                           assembly_sha=ledger["assembly_sha256"],
                           stage=stage, identity=identity, groups=groups,
                           tensor_proofs=proof)
    if final_receipt.exists() or final_receipt.is_symlink():
        if (final_receipt.is_symlink() or not final_receipt.is_file()
                or final_receipt.stat().st_size > 4 * MIB
                or json.loads(final_receipt.read_text(encoding="utf-8")) != result):
            raise ValueError("Q4 published final proof changed or is untrusted")
    else:
        _publish_exclusive(final_receipt, result)
    if assembly.exists():
        if core.hash_file(assembly) != ledger["assembly_sha256"]:
            raise ValueError("Q4 assembled checkpoint changed before cleanup")
        assembly.unlink()
    log.write(json.dumps({"event": "CHUNKED_Q4_FINAL_PUBLISHED",
                          "sha256": ledger["assembly_sha256"]}) + "\n")
    log.flush()
    return result


def merge_q4(source: Path, parts: list[Path], output: Path, expected: dict) -> dict:
    """Bitwise Q4 merge with no retained per-part reader or tensor mmaps.

    A part is opened only during its own inventory inspection. Only primitive
    offsets, type identifiers, dtype strings, sizes, and shapes survive. Payloads
    are subsequently read from one open part at a time in bounded byte blocks,
    without building a 42-part in-memory GGUF tokenizer/tensor collection.
    """
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
    import numpy as np

    if (not parts or output.exists() or output.is_symlink()
            or not output.parent.is_dir()):
        raise ValueError("Q4 merge requires fresh output and complete chunk list")

    writer = GGUFWriter(str(output), "qwen35moe")
    # Scope the original reader so its tokenizer metadata and ndarray views
    # cannot coexist with the subsequent 42 part readers.
    def source_metadata() -> tuple[list[str], dict[str, tuple]]:
        original = GGUFReader(str(source), "r")
        core.copy_metadata(original, writer, file_type=Q4_FILE_TYPE)
        order = [t.name for t in original.tensors]
        if len(order) != len(set(order)):
            raise ValueError("Q4 original source has duplicate tensor names")
        source_proofs = {}
        for t in original.tensors:
            if t.tensor_type not in (GGMLQuantizationType.BF16, GGMLQuantizationType.F32):
                raise ValueError("Q4 original source tensor is not BF16/F32")
            source_proofs[t.name] = (
                tuple(map(int, t.shape)), int(t.tensor_type),
                core.hash_tensor_data(t.data)
                if t.tensor_type == GGMLQuantizationType.F32 else None,
            )
        return order, source_proofs

    names, source_proofs = source_metadata()
    if (not isinstance(expected, dict) or len(expected) != len(names)
            or set(expected) != set(names)
            or len({str(part.resolve()) for part in parts}) != len(parts)):
        raise ValueError("Q4 chunk inventory differs from registered BF16")
    q4_types = {GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q5_K,
                GGMLQuantizationType.Q6_K, GGMLQuantizationType.Q8_0,
                GGMLQuantizationType.F32, GGMLQuantizationType.F16,
                GGMLQuantizationType.BF16}

    def inspect_one_part(path: Path) -> dict[str, tuple]:
        if path.is_symlink() or not path.is_file():
            raise ValueError("Q4 chunk is not a trusted regular file")
        reader = GGUFReader(str(path), "r")
        length = path.stat().st_size
        witness: dict[str, tuple] = {}
        for tensor in reader.tensors:
            name = tensor.name
            if name in witness or name not in source_proofs:
                raise ValueError(f"Q4 duplicate/unregistered chunk tensor: {name}")
            item = expected[name]
            shape = tuple(map(int, tensor.shape))
            dtype = tensor.tensor_type
            if (not isinstance(item, dict) or dtype not in q4_types
                    or item.get("shape") != list(shape)
                    or item.get("tensor_type") != dtype.name
                    or item.get("raw_sha256") != core.hash_tensor_data(tensor.data)):
                raise ValueError(f"Q4 chunk payload or type changed: {name}")
            original_shape, source_type, original_f32_sha = source_proofs[name]
            if source_type == int(GGMLQuantizationType.F32):
                normalized = original_shape
                while len(normalized) > len(shape) and normalized[-1] == 1:
                    normalized = normalized[:-1]
                if (dtype != GGMLQuantizationType.F32 or normalized != shape
                        or item["raw_sha256"] != original_f32_sha):
                    raise ValueError(f"Q4 copied F32 singleton bytes/shape changed: {name}")
            elif shape != original_shape:
                raise ValueError(f"Q4 quantized tensor shape differs: {name}")
            nbytes, offset = int(tensor.n_bytes), int(tensor.data_offset)
            if (nbytes <= 0 or offset < 0 or offset > length
                    or nbytes > length - offset or nbytes != tensor.data.nbytes):
                raise ValueError(f"Q4 chunk tensor byte bounds changed: {name}")
            witness[name] = (
                str(path.resolve()), offset, nbytes,
                tuple(map(int, tensor.data.shape)),
                tensor.data.dtype.str, int(dtype), shape,
            )
        return witness

    metadata: dict[str, tuple] = {}
    for part in parts:
        scanned = inspect_one_part(part)
        if set(metadata).intersection(scanned):
            raise ValueError("Q4 tensor duplicated across chunk files")
        metadata.update(scanned)
    if set(metadata) != set(names):
        raise ValueError("Q4 chunk tensor inventory differs from BF16")
    if not any(row[5] == int(GGMLQuantizationType.Q4_K) for row in metadata.values()):
        raise ValueError("Q4_K_M recipe produced no Q4_K tensors")

    try:
        for name in names:
            _path, _offset, nbytes, byte_shape, numpy_dtype, kind, _shape = metadata[name]
            writer.add_tensor_info(name, byte_shape, np.dtype(numpy_dtype),
                                   nbytes, raw_dtype=GGMLQuantizationType(kind))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        assert writer.fout is not None and len(writer.fout) == 1
        dest = writer.fout[0]
        writer.write_padding(dest, dest.tell())
        for name in names:
            part_path, offset, nbytes, _byteshape, _dtype, _kind, _shape = metadata[name]
            sha = hashlib.sha256()
            with Path(part_path).open("rb") as stream:
                stream.seek(offset)
                remaining = nbytes
                while remaining:
                    block = stream.read(min(remaining, MERGE_COPY_BLOCK))
                    if not block:
                        raise ValueError(f"Q4 part truncated during bounded stream: {name}")
                    dest.write(block)
                    sha.update(block)
                    remaining -= len(block)
            if sha.hexdigest() != expected[name]["raw_sha256"]:
                raise ValueError(f"Q4 payload changed during final raw copy: {name}")
            writer.write_padding(dest, nbytes)
        writer.flush()
    finally:
        writer.close()

    # This scope releases the complete final GGUFReader before returning.
    def validate_finished_file() -> None:
        merged = GGUFReader(str(output), "r")
        if (len(merged.tensors) != len(names)
                or merged.get_field("general.file_type").contents() != Q4_FILE_TYPE):
            raise ValueError("merged Q4 tensor inventory/file type changed")
        for name, tensor in zip(names, merged.tensors):
            proof = expected[name]
            if (tensor.name != name or tensor.tensor_type.name != proof["tensor_type"]
                    or list(map(int, tensor.shape)) != proof["shape"]
                    or core.hash_tensor_data(tensor.data) != proof["raw_sha256"]):
                raise ValueError(f"merged Q4 tensor identity changed: {name}")

    validate_finished_file()
    return {
        "status": "GGUF_CHUNKED_Q4_TENSOR_BITWISE_MERGE_NOT_QUALITY_QUALIFIED",
        "mode": MODE, "tensors": len(names), "output": str(output.resolve()),
        "output_sha256": core.hash_file(output), "source": str(source.resolve()),
    }


def _validate_parent(source: Path, quantizer: Path, full_verification: Path,
                     source_sha: str) -> dict:
    if (not source.is_file() or not quantizer.is_file()
            or not full_verification.is_file()
            or full_verification.stat().st_size > 4 * MIB
            or core.hash_file(quantizer) != BINARY_SHA256):
        raise ValueError("Q4 source, source proof, or pinned quantizer unavailable")
    stage_path = source.with_name(source.name + ".reap-stage.json")
    if stage_path.is_symlink() or not stage_path.is_file():
        raise ValueError("Q4 requires original BF16 published stage receipt")
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    if (stage.get("schema") != "chowder.reap-gguf-stage.v1"
            or stage.get("phase") != 0
            or stage.get("output") != str(source)
            or stage.get("sha256") != source_sha
            or stage.get("size_bytes") != source.stat().st_size
            or stage.get("input_sha256") != core.hash_file(full_verification)
            or stage.get("status") != "PUBLISHED_GGUF_NOT_QUALITY_QUALIFIED"):
        raise ValueError("Q4 original BF16 stage identity mismatch")
    proof = json.loads(full_verification.read_text(encoding="utf-8"))
    if (proof.get("schema") != "chowder.reap128-v2-pruned-bf16-verification.v1"
            or proof.get("status") != "FULL_SOURCE_TENSOR_EQUIVALENCE_PASS"
            or len(proof.get("candidate_shard_sha256", {})) != 26):
        raise ValueError("Q4 26-shard source equivalence not certified")
    if core.hash_file(source) != source_sha:
        raise ValueError("Q4 BF16 GGUF changed from registered full digest")
    return stage


def _build(*, source: Path, output: Path, scratch: Path, chunks: Path,
           quantizer: Path, full_verification: Path, log_path: Path,
           source_sha: str, max_mib: int = 1024, slab_mib: int = 1,
           expected_layers: int = 40, expected_experts: int = 128) -> dict:
    from gguf import GGUFReader
    from chowder_batch.reap_gguf_supervised import _owned_stage, resource_problems

    _require_no_competing_quantizer()
    source, output = source.resolve(), output.resolve()
    scratch, chunks = scratch.resolve(), chunks.resolve()
    quantizer, full_verification = quantizer.resolve(), full_verification.resolve()
    recovery = _finalization_path(scratch, output)
    if (source.drive.lower() == output.drive.lower()
            or source.drive.lower() != scratch.drive.lower()
            or output.drive.lower() != chunks.drive.lower()):
        raise ValueError("Q4 source/scratch and Q4 output/chunks need separate volumes")
    if (not scratch.is_dir() or scratch.is_symlink()
            or not chunks.is_dir() or chunks.is_symlink()
            or scratch == chunks or output.is_symlink()
            or (output.exists() and not recovery.is_file())
            or log_path.exists() or not log_path.parent.is_dir()):
        raise ValueError("Q4 requires clean distinct private workspaces and fresh audit")
    stage = _validate_parent(source, quantizer, full_verification, source_sha)
    reader = GGUFReader(str(source), "r")
    if (reader.get_field("general.file_type").contents() != 32
            or reader.get_field("qwen35moe.block_count").contents() != expected_layers
            or reader.get_field("qwen35moe.expert_count").contents() != expected_experts):
        raise ValueError("Q4 BF16 source not frozen 40x128 MoE geometry")
    groups = core.groups_for(reader, max_mib=max_mib)
    identity = {"source_sha256": source_sha,
                "stage_spec_sha256": stage["spec_sha256"],
                "quantizer_sha256": BINARY_SHA256, "mode": MODE,
                "slab_mib": slab_mib, "max_mib": max_mib,
                "chunk_count": len(groups)}
    if disk_usage(chunks).free < 8 * GIB or disk_usage(scratch).free < 8 * GIB:
        raise ValueError("Q4 physical scratch/output reserve insufficient")
    parts = []
    tensor_proofs: dict[str, dict] = {}
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        log.write(json.dumps({"event": "CHUNKED_Q4_START", "identity": identity}) + "\n")
        log.flush()
        if recovery.exists() or recovery.is_symlink():
            log.write(json.dumps({"event": "CHUNKED_Q4_RECOVER_SEALED_ASSEMBLY"}) + "\n")
            log.flush()
            return _recover_finalization(
                scratch=scratch, chunks=chunks, source=source,
                output=output, identity=identity, groups=groups,
                stage=stage, log=log, quantizer=quantizer,
            )
        source_items = {t.name: t for t in reader.tensors}
        for index, names in enumerate(groups):
            basename = f"chunk-{index+1:03d}-of-{len(groups):03d}"
            mini = scratch / (basename + ".bf16.gguf")
            part = chunks / (basename + ".q4_k_m.gguf")
            receipt = chunks / (basename + ".receipt.json")
            group_identity = {**identity, "index": index, "names": names}
            if part.exists() or receipt.exists():
                record = core._load_chunk_record(receipt, part, expected=group_identity)
                if mini.exists():
                    _cleanup_sealed_mini(
                        mini=mini, part=part, receipt=receipt,
                        group_identity=group_identity, original=reader,
                        quantizer=quantizer,
                    )
                    log.write(json.dumps({"event": "EXACT_Q4_SEALED_MINI_RECOVERED",
                                          "index": index}) + "\n")
                    log.flush()
                _check_chunk_evidence(
                    record=record, source_items=source_items, part=part,
                    group_identity=group_identity, quantizer=quantizer, mini=mini,
                )
                log.write(json.dumps({"event": "EXACT_Q4_CHUNK_REUSED", "index": index}) + "\n")
            else:
                _require_no_competing_quantizer()
                if mini.exists() or resource_problems([source, mini, part], initial=True):
                    raise ValueError("Q4 prelaunch guard or unsealed mini refused")
                prefix = core.create_mini(source, mini, names)
                quant = core.quantize_mini(mini, part, quantizer=quantizer,
                                           mode=MODE, memory_mib=slab_mib, log=log)
                record = {
                    "schema": "chowder.reap128-v2-gguf-chunk.v1",
                    "identity": group_identity, "output": str(part),
                    "output_size_bytes": part.stat().st_size,
                    "output_sha256": quant["output_sha256"],
                    "source_mini_sha256": prefix["source_mini_sha256"],
                    "input_tensor_provenance": prefix["source_tensor_receipts"],
                    "output_tensor_provenance": quant["quantized_tensors"],
                    "effective_quant_command": quant["quant_command"],
                    "status": "CHUNK_Q4_K_M_BITWISE_WITNESSED_NOT_QUALITY_GATE",
                }
                _check_chunk_evidence(
                    record=record, source_items=source_items, part=part,
                    group_identity=group_identity, quantizer=quantizer, mini=mini,
                )
                with receipt.open("x", encoding="utf-8", newline="\n") as witness:
                    witness.write(json.dumps(record, sort_keys=True, indent=2) + "\n")
                if core.hash_file(mini) != prefix["source_mini_sha256"]:
                    raise ValueError("Q4 original mini changed before owned cleanup")
                mini.unlink()
                log.write(json.dumps({"event": "EXACT_Q4_CHUNK_PUBLISHED",
                                      "index": index, "sha256": quant["output_sha256"]}) + "\n")
                log.flush()
            for tensor_name, value in record["output_tensor_provenance"].items():
                if tensor_name in tensor_proofs:
                    raise ValueError("duplicate quantized Q4 tensor in chunks")
                tensor_proofs[tensor_name] = value
            parts.append(part)
        if len(tensor_proofs) != len(reader.tensors):
            raise ValueError("Q4 source tensor inventory not complete")
        recovered_merge = _completed_merge_before_ledger(
            scratch=scratch, source=source, tensor_proofs=tensor_proofs,
        )
        if recovered_merge is None:
            assembly = scratch / ("merged-q4-" + uuid.uuid4().hex + ".gguf")
            proof_path = scratch / ("merge-input-q4-" + uuid.uuid4().hex + ".json")
            with proof_path.open("x", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(tensor_proofs, sort_keys=True) + "\n")
            command = [sys.executable, "-B", "-m", "chowder_batch.reap_gguf_q4_chunked",
                       "--merge-worker", "--source", str(source), "--out", str(assembly),
                       "--tensor-receipts", str(proof_path)]
            for part in parts:
                command.extend(["--part", str(part)])
            if _owned_stage(command, drives=[source, assembly, *parts], log=log):
                raise RuntimeError("Q4 guarded assembly aborted; sealed chunks retained")
            combined = json.loads((assembly.with_name(assembly.name + ".receipt.json")).read_text())
        else:
            assembly, combined = recovered_merge
            log.write(json.dumps({"event": "EXACT_Q4_COMPLETED_MERGE_RECOVERED",
                                  "sha256": combined["output_sha256"]}) + "\n")
            log.flush()
        if (combined.get("output_sha256") != core.hash_file(assembly)
                or combined.get("tensors") != len(tensor_proofs)):
            raise ValueError("Q4 assembly digest or inventory mismatch")
        # Seal source/chunk/payload lineage while every part still exists.
        # Interrupted cleanup/copy/receipt publication can then be resumed
        # directly from the independently revalidated assembled GGUF.
        _seal_finalization(
            scratch=scratch, source=source, output=output, assembly=assembly,
            combined=combined, identity=identity, groups=groups,
            tensor_proofs=tensor_proofs, parts=parts, quantizer=quantizer,
        )
        log.write(json.dumps({"event": "EXACT_Q4_MERGE_DURABLY_SEALED",
                              "sha256": combined["output_sha256"]}) + "\n")
        log.flush()
        return _recover_finalization(
            scratch=scratch, chunks=chunks, source=source, output=output,
            identity=identity, groups=groups, stage=stage, log=log,
            quantizer=quantizer,
        )


def run_q4(**kwargs) -> dict:
    directory = kwargs["chunks"].resolve()
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("Q4 chunk directory missing or is a symlink")
    with core._exclusive_chunk_owner(directory):
        return _build(**kwargs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--part", type=Path, action="append", default=[])
    parser.add_argument("--merge-worker", action="store_true")
    parser.add_argument("--tensor-receipts", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--chunks", type=Path)
    parser.add_argument("--quantizer", type=Path)
    parser.add_argument("--parent-source-sha256")
    parser.add_argument("--full-verification", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--max-mib", type=int, default=1024)
    parser.add_argument("--slab-mib", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        if args.merge_worker:
            if not args.tensor_receipts or not args.part:
                raise ValueError("Q4 worker needs sealed per-tensor proofs and parts")
            proofs = json.loads(args.tensor_receipts.read_text(encoding="utf-8"))
            result = merge_q4(args.source, args.part, args.out, proofs)
            with args.out.with_name(args.out.name + ".receipt.json").open(
                    "x", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
            print(json.dumps({"event": "Q4_CHUNK_MERGE_PASS",
                              "sha256": result["output_sha256"]}), flush=True)
            return 0
        if not args.execute:
            raise ValueError("Q4 execution requires --execute; no background side effects")
        if any(x is None for x in (args.scratch, args.chunks, args.quantizer,
                                    args.parent_source_sha256, args.full_verification,
                                    args.log)):
            raise ValueError("Q4 requires original source, stage and private workspaces")
        result = run_q4(
            source=args.source, output=args.out, scratch=args.scratch,
            chunks=args.chunks, quantizer=args.quantizer,
            full_verification=args.full_verification, log_path=args.log,
            source_sha=args.parent_source_sha256, max_mib=args.max_mib,
            slab_mib=args.slab_mib,
        )
        print(json.dumps({"status": result["status"],
                          "sha256": result["output_sha256"]}), flush=True)
        return 0
    except (ValueError, OSError, ImportError, RuntimeError, KeyError, TypeError) as exc:
        print(f"Q4 source-bound build refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
