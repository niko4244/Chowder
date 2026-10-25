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
import subprocess
import sys
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
        if list(map(int, src.shape)) != list(map(int, t.shape)):
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
    quantized = [GGUFReader(str(path), "r") for path in mini_outputs]
    tensors = [t for reader in quantized for t in reader.tensors]
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


def main(argv=None) -> int:
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--max-mib", type=int, default=1024)
    args = parser.parse_args(argv)
    try:
        from gguf import GGUFReader

        if not args.source.is_file():
            raise ValueError("source BF16 GGUF missing")
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
