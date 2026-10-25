"""Frozen BF16-derived Q4_K_M GGUF construction in bounded, SHA-sealed chunks.

Never start alongside the Q8 constructor or any other quantizer. All outputs
remain unqualified until independent frozen inference and GPU gates complete.
"""

from __future__ import annotations

import argparse
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


def _require_no_competing_quantizer() -> None:
    """Fail closed while another Q8/Q4 supervisor or quantizer owns the host."""
    import psutil

    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            info = process.info
            if info["pid"] == os.getpid():
                continue
            name = (info.get("name") or "").lower()
            argv = " ".join(info.get("cmdline") or []).lower()
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except psutil.AccessDenied:
            if process.name().lower() == "llama-quantize.exe":
                raise ValueError("another uninspectable quantizer is active") from None
            continue
        if (name == "llama-quantize.exe"
                or (name.startswith("python") and
                    ("chowder_batch.reap_gguf_chunked" in argv
                     or "chowder_batch.reap_gguf_q4_chunked" in argv))):
            raise ValueError("concurrent Q8/Q4 quantization owner detected; refusing Q4")


def merge_q4(source: Path, parts: list[Path], output: Path, expected: dict) -> dict:
    """Merge source-ordered Q4_K_M per-tensor payloads without requantizing."""
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType

    if not parts or output.exists() or not output.parent.is_dir():
        raise ValueError("Q4 merge requires fresh output and complete chunk list")
    original = GGUFReader(str(source), "r")
    all_tensors = []
    for path in parts:
        reader = GGUFReader(str(path), "r")
        all_tensors.extend(reader.tensors)
        del reader
    tensors = {t.name: t for t in all_tensors}
    names = [t.name for t in original.tensors]
    if (len(tensors) != len(all_tensors) or set(tensors) != set(names)
            or set(tensors) != set(expected)):
        raise ValueError("Q4 chunk inventory differs from registered BF16")
    q4_types = {GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q5_K,
                GGMLQuantizationType.Q6_K, GGMLQuantizationType.Q8_0,
                GGMLQuantizationType.F32, GGMLQuantizationType.F16,
                GGMLQuantizationType.BF16}
    for name, item in expected.items():
        tensor = tensors[name]
        if (tensor.tensor_type not in q4_types
                or list(map(int, tensor.shape)) != item["shape"]
                or tensor.tensor_type.name != item["tensor_type"]
                or core.hash_tensor_data(tensor.data) != item["raw_sha256"]):
            raise ValueError(f"Q4 chunk payload or type changed: {name}")
    if not any(t.tensor_type == GGMLQuantizationType.Q4_K for t in tensors.values()):
        raise ValueError("Q4_K_M recipe produced no Q4_K tensors")
    writer = GGUFWriter(str(output), "qwen35moe")
    core.copy_metadata(original, writer, file_type=Q4_FILE_TYPE)
    for name in names:
        tensor = tensors[name]
        writer.add_tensor(name, tensor.data, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    merged = GGUFReader(str(output), "r")
    after = {t.name: t for t in merged.tensors}
    if len(after) != len(names) or set(after) != set(names):
        raise ValueError("merged Q4 tensor inventory changed")
    if merged.get_field("general.file_type").contents() != Q4_FILE_TYPE:
        raise ValueError("merged Q4 metadata file type changed")
    for name in names:
        before, now = tensors[name], after[name]
        if (now.tensor_type != before.tensor_type or list(now.shape) != list(before.shape)
                or core.hash_tensor_data(now.data) != expected[name]["raw_sha256"]):
            raise ValueError(f"merged Q4 tensor identity changed: {name}")
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
    if (source.drive.lower() == output.drive.lower()
            or source.drive.lower() != scratch.drive.lower()
            or output.drive.lower() != chunks.drive.lower()):
        raise ValueError("Q4 source/scratch and Q4 output/chunks need separate volumes")
    if (not scratch.is_dir() or scratch.is_symlink()
            or not chunks.is_dir() or chunks.is_symlink()
            or scratch == chunks or output.exists() or output.is_symlink()
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
        for index, names in enumerate(groups):
            basename = f"chunk-{index+1:03d}-of-{len(groups):03d}"
            mini = scratch / (basename + ".bf16.gguf")
            part = chunks / (basename + ".q4_k_m.gguf")
            receipt = chunks / (basename + ".receipt.json")
            group_identity = {**identity, "index": index, "names": names}
            if part.exists() or receipt.exists():
                record = core._load_chunk_record(receipt, part, expected=group_identity)
                if mini.exists():
                    raise ValueError("Q4 sealed output has unexplained retained source")
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
        if (combined.get("output_sha256") != core.hash_file(assembly)
                or combined.get("tensors") != len(tensor_proofs)):
            raise ValueError("Q4 assembly digest or inventory mismatch")
        # Assembly lives on F:. All G: chunk contents remain separately
        # SHA-validated before removal to admit enough space for final G: copy.
        for index, part in enumerate(parts):
            receipt = part.with_name(part.stem.replace(".q4_k_m", "") + ".receipt.json")
            record = core._load_chunk_record(
                receipt, part, expected={**identity, "index": index, "names": groups[index]})
            if record["output_sha256"] != core.hash_file(part):
                raise ValueError("Q4 sealed chunk changed before cleanup")
            part.unlink()
            log.write(json.dumps({"event": "EXACT_Q4_CHUNK_CLEANED",
                                  "path": str(part), "sha256": record["output_sha256"]}) + "\n")
            log.flush()
        if disk_usage(output.parent).free < assembly.stat().st_size + 6 * GIB:
            raise ValueError("Q4 final volume cannot maintain disk reserve")
        transfer = output.with_name("." + output.stem + ".q4-owned-" +
                                    uuid.uuid4().hex + ".gguf")
        with assembly.open("rb") as src, transfer.open("xb") as target:
            for block in iter(lambda: src.read(core.HASH_BLOCK), b""):
                if resource_problems([assembly, transfer]):
                    raise RuntimeError("Q4 final transfer aborted by host guard")
                target.write(block)
        if core.hash_file(transfer) != combined["output_sha256"]:
            raise ValueError("Q4 transfer digest mismatch")
        transfer.rename(output)
        result = {
            "schema": "chowder.reap128-v2-gguf-q4-chunked-final.v1",
            "status": "Q4_K_M_TENSOR_EXACT_CONSTRUCTED_NOT_QUALITY_QUALIFIED",
            "source_gguf_sha256": source_sha,
            "output": str(output), "output_sha256": combined["output_sha256"],
            "output_size_bytes": output.stat().st_size,
            "frozen_stage_spec_sha256": stage["spec_sha256"],
            "quantizer_binary_sha256": BINARY_SHA256, "chunk_count": len(groups),
            "chunked_reference_receipts": {
                "all_tensors": tensor_proofs, "source_groups": groups,
            },
            "quality_gate": "NOT_EVALUATED", "gpu_fit_gate": "NOT_EVALUATED",
        }
        with output.with_name(output.name + ".reap-chunked.json").open(
                "x", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
        log.write(json.dumps({"event": "CHUNKED_Q4_FINAL_PUBLISHED",
                              "sha256": combined["output_sha256"]}) + "\n")
        log.flush()
        if core.hash_file(assembly) != combined["output_sha256"]:
            raise ValueError("Q4 assembled original changed before cleanup")
        assembly.unlink()
        log.write(json.dumps({"event": "EXACT_Q4_ASSEMBLY_CLEANED"}) + "\n")
        return result


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
    except (ValueError, OSError, ImportError) as exc:
        print(f"Q4 source-bound build refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
