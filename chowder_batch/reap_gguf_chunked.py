"""Bounded, restartable GGUF tensor-group quantization using frozen llama.cpp.

This is an alternate *construction* route for a host where monolithic Q8_0
fails the RAM guard. It does not change per-tensor quantization rules,
scientific gates, model weights, imatrix, or provider processes.

Only run against a BF16 GGUF already published with its full source-bound
stage receipt. Chunk outputs are separate GGUF files with exact SHA receipts.
Quantize one bounded chunk at a time, then merge its original tensor inventory
in BF16 order; validate the source-bound tensor bytes and final GGUF geometry.
No chunked output is deemed quality-qualified by these operations.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

GIB = 1024**3
MIB = 1024**2
MAX_CHUNK_MIB = 1536
TRANSIENT_RESERVE_GIB = 6
HASH_BLOCK = 8 * MIB


def hash_file(file: Path) -> str:
    h = hashlib.sha256()
    with file.open("rb") as f:
        for buf in iter(lambda: f.read(HASH_BLOCK), b""):
            h.update(buf)
    return h.hexdigest()


def hash_tensor_data(raw: Any) -> str:
    """Hash a GGUFReader tensor's memory-mapped bytes without copying it."""
    view = memoryview(raw)
    try:
        bytes_view = view.cast("B")
        h = hashlib.sha256()
        for off in range(0, len(bytes_view), HASH_BLOCK):
            h.update(bytes_view[off:off + HASH_BLOCK])
        return h.hexdigest()
    finally:
        view.release()


def _quantized_shape_matches(original: Any, quantized: Any) -> bool:
    """Permit only llama.cpp's lossless F32 trailing-unit-dimension fold.

    The pinned quantizer serializes an F32 [2048, 1] gate tensor as [2048]
    without changing a single data byte. Quantized shapes must otherwise
    match exactly, and *all* copied F32 payloads must retain original bytes.
    """
    from gguf import GGMLQuantizationType

    source_shape = tuple(map(int, original.shape))
    target_shape = tuple(map(int, quantized.shape))
    if original.tensor_type == GGMLQuantizationType.F32:
        if (quantized.tensor_type != GGMLQuantizationType.F32
                or hash_tensor_data(original.data) != hash_tensor_data(quantized.data)):
            return False
        while len(source_shape) > len(target_shape) and source_shape[-1] == 1:
            source_shape = source_shape[:-1]
    return source_shape == target_shape


@contextmanager
def _exclusive_chunk_owner(directory: Path):
    """Kernel-backed nonblocking lock, durable across interrupted owners.

    Never unlink this inode: a second process must lock the same file after a
    crash or restart. The OS releases the byte-range lock at process exit.
    """
    lock = directory / ".reap-gguf-chunked.lock"
    if lock.is_symlink():
        raise ValueError("untrusted GGUF chunk lock symlink")
    with lock.open("a+b") as handle:
        handle.seek(0)
        try:
            empty = handle.read(1) == b""
        except PermissionError as error:
            # Windows also prevents *reading* a locked byte range.
            raise ValueError("another owned GGUF chunk construction is running") from error
        if empty:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise ValueError("another owned GGUF chunk construction is running") from error
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise ValueError("another owned GGUF chunk construction is running") from error
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def groups_for(source_reader: Any, *, max_mib: int = 1024) -> list[list[str]]:
    if type(max_mib) is not int or not 8 <= max_mib <= MAX_CHUNK_MIB:
        raise ValueError("GGUF tensor chunk cap must be 8..1536 MiB")
    cap = max_mib * MIB
    groups: list[list[str]] = []
    current: list[str] = []
    size = 0
    names: set[str] = set()
    for t in source_reader.tensors:
        if t.name in names:
            raise ValueError("duplicate GGUF source tensor names")
        names.add(t.name)
        count = t.data.nbytes
        if count > cap:
            raise ValueError(f"tensor {t.name} is larger than configured bounded chunk cap")
        if current and size + count > cap:
            groups.append(current)
            current, size = [], 0
        current.append(t.name)
        size += count
    if current:
        groups.append(current)
    if not groups:
        raise ValueError("empty original BF16 GGUF")
    return groups


def copy_metadata(source_reader: Any, writer: Any, *, file_type: int) -> None:
    from gguf import GGUFValueType

    for name, field in source_reader.fields.items():
        if (name.startswith("GGUF.") or name == "general.architecture"
                or name.startswith("split.")):
            continue
        value = field.contents()
        if name == "general.file_type":
            value = file_type
        kind = field.types[0]
        writer.add_key_value(
            name, value, kind,
            field.types[1] if kind == GGUFValueType.ARRAY else None,
        )


def create_mini(source: Path, output: Path, names: list[str]) -> dict:
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType

    if not output.parent.is_dir() or output.exists() or output.is_symlink():
        raise ValueError("mini-GGUF must be a fresh external file")
    if not names or len(set(names)) != len(names):
        raise ValueError("missing/duplicated mini-GGUF tensor names")
    reader = GGUFReader(str(source), "r")
    by_name = {t.name: t for t in reader.tensors}
    if any(name not in by_name for name in names):
        raise ValueError("mini-GGUF tensor not found in original source")
    if any(by_name[name].tensor_type not in (
            GGMLQuantizationType.BF16, GGMLQuantizationType.F32)
           for name in names):
        raise ValueError("only original unquantized BF16/F32 tensor types allowed")
    writer = GGUFWriter(str(output), "qwen35moe")
    copy_metadata(reader, writer, file_type=32)
    expected = {}
    for name in names:
        original = by_name[name]
        writer.add_tensor(name, original.data, raw_dtype=original.tensor_type)
        expected[name] = {
            "shape": list(map(int, original.shape)),
            "tensor_type": original.tensor_type.name,
            "raw_sha256": hash_tensor_data(original.data),
        }
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    result = GGUFReader(str(output), "r")
    actual = {t.name: t for t in result.tensors}
    if set(actual) != set(expected):
        raise ValueError("mini-GGUF tensor inventory differs after publication")
    for name, info in expected.items():
        t = actual[name]
        if (list(map(int, t.shape)) != info["shape"]
                or t.tensor_type.name != info["tensor_type"]
                or hash_tensor_data(t.data) != info["raw_sha256"]):
            raise ValueError(f"mini-GGUF source tensor modified: {name}")
    return {"source_names": names, "source_tensor_receipts": expected,
            "source_mini_sha256": hash_file(output)}


def quantize_mini(
    source_mini: Path, target_mini: Path, *, quantizer: Path,
    mode: str, memory_mib: int, log,
) -> dict:
    """Call only the exact owned pinned executable; no CUDA or shared GPU."""
    from chowder_batch.reap_gguf_supervised import _owned_stage
    from gguf import GGUFReader, GGMLQuantizationType

    if mode not in ("Q8_0", "Q4_K_M"):
        raise ValueError("only frozen Q8_0 or Q4_K_M mode permitted")
    if type(memory_mib) is not int or not 1 <= memory_mib <= 256:
        raise ValueError("bounded quantizer slab must be 1..256 MiB")
    if not quantizer.is_file() or target_mini.exists():
        raise ValueError("missing quantizer or private target already exists")
    cmd = [str(quantizer.resolve()), "--max-buffer-size", str(memory_mib),
           str(source_mini.resolve()), str(target_mini.resolve()), mode, "2"]
    code = _owned_stage(cmd, drives=[source_mini, target_mini], log=log)
    if code:
        raise RuntimeError(f"owned mini-GGUF quantization aborted ({code}); partial preserved")
    target = GGUFReader(str(target_mini), "r")
    source = GGUFReader(str(source_mini), "r")
    a = {t.name: t for t in source.tensors}
    b = {t.name: t for t in target.tensors}
    if set(a) != set(b):
        raise ValueError("quantized chunk changed source tensor inventory")
    result = {}
    for name, t in b.items():
        src = a[name]
        if not _quantized_shape_matches(src, t):
            raise ValueError(f"quantized tensor shape differs: {name}")
        if t.tensor_type not in (
                GGMLQuantizationType.Q8_0, GGMLQuantizationType.Q4_K,
                GGMLQuantizationType.Q5_K, GGMLQuantizationType.Q6_K,
                GGMLQuantizationType.F32, GGMLQuantizationType.F16,
                GGMLQuantizationType.BF16):
            raise ValueError(f"unexpected output tensor type: {name}/{t.tensor_type}")
        result[name] = {
            "shape": list(map(int, t.shape)), "tensor_type": t.tensor_type.name,
            "raw_sha256": hash_tensor_data(t.data),
        }
    return {"output_sha256": hash_file(target_mini), "quantized_tensors": result,
            "quant_command": cmd, "quantizer_sha256": hash_file(quantizer)}


def merge_mini(
    source: Path, mini_outputs: list[Path], output: Path, *,
    expected_tensor_receipts: dict[str, dict], mode: str,
) -> dict:
    """Reassemble in original tensor order using mapped raw GGUF payloads."""
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType

    if mode != "Q8_0":
        raise ValueError("production chunk reassembly currently supports Q8_0 only")
    if output.exists() or not output.parent.is_dir() or not mini_outputs:
        raise ValueError("final merged output must be a fresh existing-directory path")
    bf16 = GGUFReader(str(source), "r")
    # Do not retain all chunk GGUFReader objects: they each parse the full
    # original tokenizer field arrays. Raw tensor mappings hold their own
    # mmap bases, so keeping 42 reader objects would amplify RAM needlessly.
    tensors = []
    for part in mini_outputs:
        reader = GGUFReader(str(part), "r")
        tensors.extend(reader.tensors)
        del reader
    lookup = {t.name: t for t in tensors}
    names = [t.name for t in bf16.tensors]
    if (len(lookup) != len(tensors) or set(lookup) != set(names)
            or set(lookup) != set(expected_tensor_receipts)):
        raise ValueError("chunk tensor inventory differs from source original")
    for name, item in expected_tensor_receipts.items():
        t = lookup[name]
        if (list(map(int, t.shape)) != item["shape"]
                or t.tensor_type.name != item["tensor_type"]
                or hash_tensor_data(t.data) != item["raw_sha256"]):
            raise ValueError(f"sample payload mismatch before merging: {name}")
    writer = GGUFWriter(str(output), "qwen35moe")
    copy_metadata(bf16, writer, file_type=7)
    for name in names:
        t = lookup[name]
        writer.add_tensor(name, t.data, raw_dtype=t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    reread = GGUFReader(str(output), "r")
    new = {t.name: t for t in reread.tensors}
    if len(new) != len(names):
        raise ValueError("merged GGUF inventory changed")
    for name in names:
        a, b = lookup[name], new[name]
        if (b.tensor_type != a.tensor_type or list(b.shape) != list(a.shape)
                or hash_tensor_data(b.data) != expected_tensor_receipts[name]["raw_sha256"]):
            raise ValueError(f"merged quantization payload mismatch: {name}")
    return {"status": "GGUF_CHUNKED_TENSOR_BITWISE_MERGE_NOT_QUALITY_QUALIFIED",
            "mode": mode, "tensors": len(names), "output": str(output.resolve()),
            "output_sha256": hash_file(output), "source": str(source.resolve()),
            "limitation": "Per-tensor quantization equivalence only; scientific quality and GPU tests still required"}


def _load_chunk_record(receipt: Path, output: Path, *,
                       expected: dict) -> dict:
    if (receipt.is_symlink() or output.is_symlink() or not receipt.is_file()
            or not output.is_file() or receipt.stat().st_size > 2 * MIB):
        raise ValueError("missing or untrusted partial quantization stage")
    saved = json.loads(receipt.read_text(encoding="utf-8"))
    if (saved.get("schema") != "chowder.reap128-v2-gguf-chunk.v1"
            or saved.get("identity") != expected
            or saved.get("output") != str(output.resolve())
            or saved.get("output_size_bytes") != output.stat().st_size
            or saved.get("output_sha256") != hash_file(output)):
        raise ValueError(f"resumable GGUF chunk identity/hash mismatch: {output}")
    return saved


def _run_chunked_locked(
    *,
    source: Path, output: Path, scratch: Path, chunks: Path,
    quantizer: Path, max_mib: int = 1024, slab_mib: int = 1,
    parent_source_sha256: str, full_verification: Path, log_path: Path,
    expected_layers: int = 40, expected_experts: int = 128,
) -> dict:
    """Execute serial bounded Q8 build with immutable chunk witnesses.

    The output is a NEW final file on the supplied output drive. It is never
    published unless the source, all quantized chunk payloads, the rebuilt
    full tensor inventory, and both physical-drive copies have matching
    receipts. A host stop leaves chunks intact for an admitted later run.
    """
    from gguf import GGUFReader
    from chowder_batch.reap_gguf_supervised import (
        _owned_stage, resource_problems,
    )

    source, output, quantizer = source.resolve(), output.resolve(), quantizer.resolve()
    full_verification = full_verification.resolve()
    scratch, chunks, log_path = scratch.resolve(), chunks.resolve(), log_path.resolve()
    paths = (scratch, chunks)
    if (not all(path.is_dir() and not path.is_symlink() for path in paths)
            or scratch == chunks or source.parent == chunks
            or source.parent == scratch or output.parent != chunks.parent
            or output.exists() or output.is_symlink()
            or not log_path.parent.is_dir() or log_path.exists()
            or not quantizer.is_file() or not full_verification.is_file()
            or full_verification.stat().st_size > 4 * MIB):
        raise ValueError("chunk workspaces, pinned binary, fresh audit or output unsafe")
    if len(parent_source_sha256) != 64:
        raise ValueError("source stage full SHA-256 required")
    from chowder_batch.reap_gguf_build_plan import FROZEN_LLAMA_COMMIT
    expected_binary_sha = "eeb75877bd41d8d7b51d18129d873e6220014613baea349fa18e6b5d36c987de"
    if hash_file(quantizer) != expected_binary_sha:
        raise ValueError(f"quantizer differs from registered {FROZEN_LLAMA_COMMIT} build")
    stage_path = source.with_name(source.name + ".reap-stage.json")
    if stage_path.is_symlink() or not stage_path.is_file():
        raise ValueError("source BF16 GGUF needs complete published stage witness")
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    if (stage.get("schema") != "chowder.reap-gguf-stage.v1"
            or stage.get("phase") != 0
            or stage.get("output") != str(source)
            or stage.get("sha256") != parent_source_sha256
            or stage.get("size_bytes") != source.stat().st_size
            or stage.get("input_sha256") != hash_file(full_verification)
            or stage.get("status") != "PUBLISHED_GGUF_NOT_QUALITY_QUALIFIED"):
        raise ValueError("source GGUF is not the registered published BF16 stage")
    verification = json.loads(full_verification.read_text(encoding="utf-8"))
    if (verification.get("schema") !=
            "chowder.reap128-v2-pruned-bf16-verification.v1"
            or verification.get("status") != "FULL_SOURCE_TENSOR_EQUIVALENCE_PASS"
            or len(verification.get("candidate_shard_sha256", {})) != 26):
        raise ValueError("source GGUF must descend from complete 26-shard BF16 proof")
    if hash_file(source) != parent_source_sha256:
        raise ValueError("published BF16 GGUF differs from full source-bound SHA witness")
    original = GGUFReader(str(source), "r")
    if original.get_field("general.file_type").contents() != 32:
        raise ValueError("chunked Q8 input must be the registered BF16 GGUF")
    if (original.get_field("qwen35moe.block_count").contents() != expected_layers
            or original.get_field("qwen35moe.expert_count").contents() != expected_experts):
        raise ValueError("source GGUF does not match the frozen expert/layer geometry")
    groups = groups_for(original, max_mib=max_mib)
    signature = {
        "source_sha256": parent_source_sha256,
        "stage_spec_sha256": stage["spec_sha256"],
        "quantizer_sha256": expected_binary_sha,
        "mode": "Q8_0", "slab_mib": slab_mib,
        "max_mib": max_mib, "chunk_count": len(groups),
    }
    from shutil import disk_usage

    if (disk_usage(chunks).free < 8 * GIB
            or disk_usage(scratch).free < 8 * GIB):
        raise ValueError("separate GGUF staging volume reserves not available")
    started = []
    by_name: dict[str, dict] = {}
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        log.write(json.dumps({"event": "CHUNKED_Q8_START", "identity": signature}) + "\n")
        for index, names in enumerate(groups):
            name = f"chunk-{index + 1:03d}-of-{len(groups):03d}"
            input_mini = scratch / (name + ".bf16.gguf")
            chunk_output = chunks / (name + ".q8_0.gguf")
            receipt_path = chunks / (name + ".receipt.json")
            identity = {**signature, "index": index, "names": names}
            if chunk_output.exists() or receipt_path.exists():
                report = _load_chunk_record(receipt_path, chunk_output,
                                            expected=identity)
                if input_mini.exists():
                    raise ValueError("unexpected original mini remains after sealed chunk")
                log.write(json.dumps({"event": "EXACT_CHUNK_REUSED", "index": index}) + "\n")
            else:
                if input_mini.exists() or resource_problems(
                    [source, input_mini, chunk_output], initial=True,
                ):
                    raise ValueError("prelaunch resources or abandoned mini input prevent next chunk")
                prefix = create_mini(source, input_mini, names)
                target = quantize_mini(
                    input_mini, chunk_output,
                    quantizer=quantizer, mode="Q8_0",
                    memory_mib=slab_mib, log=log,
                )
                report = {
                    "schema": "chowder.reap128-v2-gguf-chunk.v1",
                    "identity": identity,
                    "output": str(chunk_output),
                    "output_size_bytes": chunk_output.stat().st_size,
                    "output_sha256": target["output_sha256"],
                    "source_mini_sha256": prefix["source_mini_sha256"],
                    "input_tensor_provenance": prefix["source_tensor_receipts"],
                    "output_tensor_provenance": target["quantized_tensors"],
                    "effective_quant_command": target["quant_command"],
                    "status": "CHUNK_Q8_BITWISE_WITNESSED_NOT_QUALITY_GATE",
                }
                with receipt_path.open("x", encoding="utf-8", newline="\n") as witness:
                    witness.write(json.dumps(report, sort_keys=True,
                                             indent=2, allow_nan=False) + "\n")
                # Exact private mini was written by this process, and only
                # after its source and quantized child proofs are sealed.
                if hash_file(input_mini) != report["source_mini_sha256"]:
                    raise ValueError("source mini changed before audited cleanup")
                input_mini.unlink()
                log.write(json.dumps({"event": "EXACT_CHUNK_PUBLISHED", "index": index,
                                      "sha256": target["output_sha256"]}) + "\n")
                log.flush()
            for tensor_name, value in report["output_tensor_provenance"].items():
                if tensor_name in by_name:
                    raise ValueError("two Q8 chunks claim same tensor")
                by_name[tensor_name] = value
            started.append(chunk_output)
        # The merge is a separately guarded owned child, so even the
        # potentially ~20-GiB assembled output can be safely aborted.
        assembly = scratch / ("merged-q8-" + uuid.uuid4().hex + ".gguf")
        tensor_receipts = scratch / ("merge-input-" + uuid.uuid4().hex + ".json")
        with tensor_receipts.open("x", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(by_name, sort_keys=True, allow_nan=False) + "\n")
        command = [
            sys.executable, "-B", "-m", "chowder_batch.reap_gguf_chunked",
            "--merge-worker", "--source", str(source), "--out", str(assembly),
            "--tensor-receipts", str(tensor_receipts),
        ]
        for part in started:
            command.extend(["--part", str(part)])
        status = _owned_stage(command, drives=[source, assembly, *started], log=log)
        if status:
            raise RuntimeError(f"guarded GGUF merge aborted ({status}); parts retained")
        combined = json.loads((assembly.with_name(assembly.name + ".receipt.json")).read_text())
        if (combined.get("output_sha256") != hash_file(assembly)
                or combined.get("tensors") != len(by_name)):
            raise ValueError("merged GGUF content identity changed")
        # Avoid duplicating all chunks plus final model on I:. The assembled
        # source on F: is already byte-sealed before exact chunk cleanup.
        for index, part in enumerate(started):
            record = _load_chunk_record(
                part.with_name(part.stem.replace(".q8_0", "") + ".receipt.json"),
                part,
                expected={**signature,
                          "index": index,
                          "names": groups[index]},
            )
            if record["output_sha256"] != hash_file(part):
                raise ValueError("sealed chunk changed before cleanup")
            part.unlink()
            log.write(json.dumps({"event": "EXACT_SOURCE_BOUND_PART_CLEANED",
                                  "path": str(part),
                                  "sha256": record["output_sha256"]}) + "\n")
            log.flush()
        if disk_usage(output.parent).free < assembly.stat().st_size + 6 * GIB:
            raise ValueError("target volume cannot hold assembled Q8 plus protected reserve")
        transfer = output.with_name("." + output.stem + ".chunked-owned-" +
                                    uuid.uuid4().hex + ".gguf")
        with assembly.open("rb") as src, transfer.open("xb") as dest:
            for block in iter(lambda: src.read(HASH_BLOCK), b""):
                if resource_problems([assembly, transfer]):
                    raise RuntimeError("owned Q8 transfer aborted by physical RAM/commit/disk guard")
                dest.write(block)
        if hash_file(transfer) != combined["output_sha256"]:
            raise ValueError("cross-volume Q8 publication copy differs from sealed merge")
        transfer.rename(output)
        result = {
            "schema": "chowder.reap128-v2-gguf-q8-chunked-final.v1",
            "status": "Q8_TENSOR_EXACT_CONSTRUCTED_NOT_QUALITY_QUALIFIED",
            "source_gguf_sha256": parent_source_sha256,
            "output": str(output),
            "output_sha256": combined["output_sha256"],
            "output_size_bytes": output.stat().st_size,
            "frozen_stage_spec_sha256": stage["spec_sha256"],
            "quantizer_binary_sha256": expected_binary_sha,
            "chunk_count": len(groups),
            "chunked_reference_receipts": {
                "all_tensors": by_name, "source_groups": groups,
            },
            "quality_gate": "NOT_EVALUATED",
            "gpu_fit_gate": "NOT_EVALUATED",
        }
        with output.with_name(output.name + ".reap-chunked.json").open(
                "x", encoding="utf-8", newline="\n") as receipt:
            receipt.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
        log.write(json.dumps({"event": "CHUNKED_Q8_FINAL_PUBLISHED",
                              "sha256": combined["output_sha256"]}) + "\n")
        log.flush()
        # Remove only this run's source-identical assembled temporary GGUF.
        # The independently rehashed I: publication now has its own receipt.
        if hash_file(assembly) != combined["output_sha256"]:
            raise ValueError("assembled GGUF changed before owned cleanup")
        assembly.unlink()
        log.write(json.dumps({"event": "EXACT_ASSEMBLED_TEMP_CLEANED",
                              "sha256": combined["output_sha256"]}) + "\n")
        log.flush()
        return result


def run_chunked(**kwargs) -> dict:
    """Lock the chunk directory *before* source hashing or touching chunks."""
    directory = kwargs["chunks"].resolve()
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("chunk directory must exist and cannot be a symlink")
    with _exclusive_chunk_owner(directory):
        return _run_chunked_locked(**kwargs)


def main(argv=None) -> int:
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--max-mib", type=int, default=1024)
    parser.add_argument("--merge-worker", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--tensor-receipts", type=Path)
    parser.add_argument("--part", type=Path, action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--chunks", type=Path)
    parser.add_argument("--quantizer", type=Path)
    parser.add_argument("--parent-source-sha256")
    parser.add_argument("--full-verification", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--slab-mib", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        from gguf import GGUFReader

        if not args.source.is_file():
            raise ValueError("source BF16 GGUF missing")
        if args.merge_worker:
            if args.out is None or args.tensor_receipts is None or not args.part:
                raise ValueError("merge worker requires new output, part files and sealed tensors")
            tensor_map = json.loads(args.tensor_receipts.read_text(encoding="utf-8"))
            result = merge_mini(
                args.source, args.part, args.out,
                expected_tensor_receipts=tensor_map, mode="Q8_0",
            )
            receipt = args.out.with_name(args.out.name + ".receipt.json")
            with receipt.open("x", encoding="utf-8", newline="\n") as file:
                file.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
            print(json.dumps({"event": "GUARDED_CHUNK_MERGE_PASS",
                              "sha256": result["output_sha256"]}), flush=True)
            return 0
        if args.execute:
            if any(x is None for x in (
                    args.out, args.scratch, args.chunks, args.quantizer,
                    args.parent_source_sha256, args.full_verification, args.log)):
                raise ValueError("execute needs external scratch/chunks/output/log and source witness")
            if (args.source.resolve().drive.lower() == args.out.resolve().drive.lower()
                    or args.source.resolve().drive.lower() != args.scratch.resolve().drive.lower()
                    or args.out.resolve().drive.lower() != args.chunks.resolve().drive.lower()):
                raise ValueError("BF16 source/scratch and Q8 output/chunks need separate volumes")
            result = run_chunked(
                source=args.source, output=args.out, scratch=args.scratch,
                chunks=args.chunks, quantizer=args.quantizer,
                max_mib=args.max_mib, slab_mib=args.slab_mib,
                parent_source_sha256=args.parent_source_sha256,
                full_verification=args.full_verification,
                log_path=args.log,
            )
            print(json.dumps({"status": result["status"],
                              "output": result["output"],
                              "sha256": result["output_sha256"],
                              "tensors": len(result["chunked_reference_receipts"]["all_tensors"])}))
            return 0
        groups = groups_for(GGUFReader(str(args.source), "r"), max_mib=args.max_mib)
        print(json.dumps({
            "status": "READ_ONLY_CHUNK_PLAN_NOT_EXECUTED",
            "model": str(args.source.resolve()),
            "chunks": len(groups),
            "max_chunk_mib": args.max_mib,
            "tensor_counts": [len(group) for group in groups],
            "note": "Use source-bound orchestration with per-chunk witnesses before production",
        }))
        return 0
    except (ValueError, OSError, ImportError) as e:
        print(f"chunked GGUF read-only plan refused: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
