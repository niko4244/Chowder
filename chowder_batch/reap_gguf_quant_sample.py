"""Source-bound, bounded-memory descriptive Q8_0 versus BF16 GGUF spot check.

This is an independent CPU-only *weight reconstruction* measurement, not
GSM8K/HumanEval, a full GGUF-byte attestation, inference, or a registered gate.
Both GGUFs must already have been published by the frozen REAP construction
route with verified monolithic stage or complete source-bound chunked witnesses.

At four deterministic 256-value positions in each of the three routed
expert tensors of all 40 layers, compare Q8_0 GGUF block dequantization
to the original BF16 GGUF *raw bytes*. This never allocates whole tensors.
The receipt preserves all sample positions, quantized tensor inventory,
numeric error values, file stat identities and the stage SHA witnesses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

BLOCK_VALUES = 32
Q8_BLOCK_BYTES = 34
WINDOW_VALUES = 256
BLOCK_COUNT = 4
LAYERS = 40
EXPERT_KEYS = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
FROZEN_SOURCE_SHA256 = "80341aa776ad9b5f937c21b081b6d76de4a2ea277e5fb88c9ae2634a9da8f4c6"
FROZEN_SPEC_SHA256 = "7f9065d8036f7bd04c19a15f87540ad195658f46a80a1a09aa407b44c4836a94"
FROZEN_QUANTIZER_SHA256 = "eeb75877bd41d8d7b51d18129d873e6220014613baea349fa18e6b5d36c987de"
TENSOR_INVENTORY = 733
MAX_CHUNKED_RECEIPT_BYTES = 2 * 1024 * 1024


def _sha256_file(path: Path) -> str:
    """Hash full GGUF bytes using bounded reads; independent of the receipt."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_stage(path: Path, phase: int) -> dict:
    record_path = path.with_name(path.name + ".reap-stage.json")
    if (path.is_symlink() or record_path.is_symlink() or not path.is_file()
            or not record_path.is_file() or record_path.stat().st_size > 1024 * 1024):
        raise ValueError("only an existing published GGUF and bounded stage receipt are accepted")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (record.get("schema") != "chowder.reap-gguf-stage.v1"
            or record.get("phase") != phase
            or record.get("output") != str(path.resolve())
            or record.get("status") != "PUBLISHED_GGUF_NOT_QUALITY_QUALIFIED"
            or record.get("size_bytes") != path.stat().st_size
            or not isinstance(record.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
            or not isinstance(record.get("spec_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", record["spec_sha256"])):
        raise ValueError("GGUF stage witness does not match published model identity")
    return record


def _load_chunked(bf16: Path, q8: Path, source: dict) -> dict:
    """Verify the completed alternate Q8 construction against frozen BF16 ancestry.

    Full-file and per-tensor SHA witnesses are checked against the *actual*
    read-only GGUF data. The final construction receipt is not a quality gate.
    """
    from gguf import GGUFReader

    from chowder_batch.reap_gguf_chunked import (
        _quantized_shape_matches, hash_tensor_span,
    )

    receipt_path = q8.with_name(q8.name + ".reap-chunked.json")
    if (bf16.is_symlink() or q8.is_symlink() or receipt_path.is_symlink()
            or not bf16.is_file() or not q8.is_file()
            or not receipt_path.is_file()
            or receipt_path.stat().st_size > MAX_CHUNKED_RECEIPT_BYTES):
        raise ValueError("chunked GGUF and bounded final receipt must be published files")
    before = (_file_stat(bf16), _file_stat(q8), _file_stat(receipt_path))
    record = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("chunked GGUF final receipt must be an object")
    if (source["sha256"] != FROZEN_SOURCE_SHA256
            or source["spec_sha256"] != FROZEN_SPEC_SHA256
            or _sha256_file(bf16) != source["sha256"]):
        raise ValueError("chunked Q8 source differs from registered full BF16 stage SHA")
    digest = record.get("output_sha256")
    references = record.get("chunked_reference_receipts")
    if (record.get("schema") != "chowder.reap128-v2-gguf-q8-chunked-final.v1"
            or record.get("status") != "Q8_TENSOR_EXACT_CONSTRUCTED_NOT_QUALITY_QUALIFIED"
            or record.get("source_gguf_sha256") != source["sha256"]
            or record.get("frozen_stage_spec_sha256") != source["spec_sha256"]
            or record.get("quantizer_binary_sha256") != FROZEN_QUANTIZER_SHA256
            or record.get("output") != str(q8.resolve())
            or record.get("output_size_bytes") != q8.stat().st_size
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or record.get("quality_gate") != "NOT_EVALUATED"
            or record.get("gpu_fit_gate") != "NOT_EVALUATED"
            or type(record.get("chunk_count")) is not int
            or not 1 <= record["chunk_count"] <= TENSOR_INVENTORY
            or not isinstance(references, dict)
            or not isinstance(references.get("all_tensors"), dict)
            or not isinstance(references.get("source_groups"), list)):
        raise ValueError("chunked Q8 final receipt has invalid source, mode, or non-gate status")
    receipts = references["all_tensors"]
    groups = references["source_groups"]
    if (len(receipts) != TENSOR_INVENTORY
            or len(groups) != record["chunk_count"]
            or any(not isinstance(group, list) or not group
                   or any(not isinstance(name, str) for name in group)
                   for group in groups)):
        raise ValueError("chunked Q8 tensor inventory or group count differs")
    ordered_names = [name for group in groups for name in group]
    if len(ordered_names) != TENSOR_INVENTORY or set(ordered_names) != set(receipts):
        raise ValueError("chunked Q8 groups must cover each of 733 tensors exactly once")
    if _sha256_file(q8) != digest:
        raise ValueError("chunked Q8 output SHA differs from published GGUF bytes")

    original = GGUFReader(str(bf16), "r")
    quantized = GGUFReader(str(q8), "r")
    for reader, expected_type in ((original, 32), (quantized, 7)):
        fields = (("general.file_type", expected_type),
                  ("qwen35moe.block_count", LAYERS),
                  ("qwen35moe.expert_count", 128))
        if any(reader.get_field(field) is None
               or reader.get_field(field).contents() != expected
               for field, expected in fields):
            raise ValueError("chunked Q8 mode or frozen 40x128 geometry mismatch")
    source_tensors = original.tensors
    output_tensors = quantized.tensors
    if (len(source_tensors) != TENSOR_INVENTORY
            or len(output_tensors) != TENSOR_INVENTORY
            or [tensor.name for tensor in source_tensors] != ordered_names
            or [tensor.name for tensor in output_tensors] != ordered_names):
        raise ValueError("chunked Q8 GGUF tensor inventory/order differs from source groups")
    saw_q8 = False
    for src, target in zip(source_tensors, output_tensors):
        claimed = receipts[target.name]
        if (not isinstance(claimed, dict)
                or claimed.get("shape") != [int(v) for v in target.shape]
                or claimed.get("tensor_type") != target.tensor_type.name
                or not isinstance(claimed.get("raw_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", claimed["raw_sha256"]) is None
                or not _quantized_shape_matches(src, target)
                or hash_tensor_span(q8, int(target.data_offset),
                                    int(target.n_bytes)) != claimed["raw_sha256"]):
            raise ValueError(f"chunked Q8 tensor shape/type/payload receipt mismatch: {target.name}")
        saw_q8 |= target.tensor_type.name == "Q8_0"
    if not saw_q8:
        raise ValueError("chunked Q8 mode has no Q8_0 tensor")
    if before != (_file_stat(bf16), _file_stat(q8), _file_stat(receipt_path)):
        raise ValueError("chunked Q8 witnesses or GGUF bytes changed during verification")
    return record


def _stage_identity(bf16: Path, q8: Path) -> dict:
    source = _load_stage(bf16, 0)
    stage_path = q8.with_name(q8.name + ".reap-stage.json")
    chunk_path = q8.with_name(q8.name + ".reap-chunked.json")
    if stage_path.exists() or stage_path.is_symlink():
        if chunk_path.exists() or chunk_path.is_symlink():
            raise ValueError("Q8 GGUF cannot have conflicting monolithic/chunked witnesses")
        target = _load_stage(q8, 1)
        if (source["spec_sha256"] != target["spec_sha256"]
                or target.get("input_sha256") != source["sha256"]):
            raise ValueError("Q8 stage does not descend from the witnessed BF16 GGUF")
    else:
        target = _load_chunked(bf16, q8, source)
    return {"bf16": source, "q8_0": target}


def _file_stat(path: Path) -> tuple[int, int, int]:
    s = path.stat()
    return s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _windows(element_count: int) -> list[int]:
    if element_count < WINDOW_VALUES or element_count % BLOCK_VALUES:
        raise ValueError("sampled quantized tensor needs >=256 values and 32-aligned extent")
    maximum = element_count - WINDOW_VALUES
    return sorted({
        (maximum * j // (BLOCK_COUNT - 1) // BLOCK_VALUES) * BLOCK_VALUES
        for j in range(BLOCK_COUNT)
    })


def _compare_window(bf16_raw, q8_raw, *, element_offset: int, n: int = WINDOW_VALUES) -> dict:
    import numpy as np
    from gguf import GGMLQuantizationType, dequantize

    if element_offset % BLOCK_VALUES or n % BLOCK_VALUES:
        raise ValueError("unaligned Q8_0 comparison window")
    original_start = element_offset * 2
    q8_start = element_offset // BLOCK_VALUES * Q8_BLOCK_BYTES
    original = np.asarray(bf16_raw).reshape(-1)[original_start:original_start + n * 2]
    quantized = np.asarray(q8_raw).reshape(-1)[
        q8_start:q8_start + n // BLOCK_VALUES * Q8_BLOCK_BYTES
    ]
    if original.size != n * 2 or quantized.size != n // BLOCK_VALUES * Q8_BLOCK_BYTES:
        raise ValueError("truncated GGUF quantization window")
    # BF16 is the most significant 16 bits of little-endian IEEE754 float32.
    bit_pattern = original.tobytes()
    reference = (np.frombuffer(bit_pattern, dtype="<u2").astype("<u4") << 16).view("<f4")
    recovered = dequantize(quantized.reshape(-1, Q8_BLOCK_BYTES),
                           GGMLQuantizationType.Q8_0).reshape(-1)
    if reference.size != n or recovered.size != n:
        raise ValueError("unexpected dequantized block shape")
    if not np.isfinite(reference).all() or not np.isfinite(recovered).all():
        raise ValueError("non-finite BF16 source or Q8 dequantized weights")
    delta = recovered - reference
    return {
        "offset": element_offset, "count": n,
        "max_abs_error": float(np.abs(delta).max()),
        "sum_abs_error": float(np.abs(delta).sum(dtype=np.float64)),
        "sum_squared_error": float((delta.astype(np.float64) ** 2).sum()),
        "source_max_abs": float(np.abs(reference).max()),
    }


def measure(bf16: Path, q8: Path) -> dict:
    from gguf import GGMLQuantizationType, GGUFReader

    bf16, q8 = bf16.resolve(), q8.resolve()
    witnesses = _stage_identity(bf16, q8)
    stamps = {str(p): _file_stat(p) for p in (bf16, q8)}
    source_reader = GGUFReader(str(bf16), "r")
    quant_reader = GGUFReader(str(q8), "r")
    left = {tensor.name: tensor for tensor in source_reader.tensors}
    right = {tensor.name: tensor for tensor in quant_reader.tensors}
    if set(left) != set(right):
        raise ValueError("Q8 GGUF tensor-name inventory differs from BF16 parent")
    rows = []
    omitted = []
    for layer in range(LAYERS):
        for suffix in EXPERT_KEYS:
            name = f"blk.{layer}.{suffix}"
            if name not in left:
                raise ValueError(f"missing required expert tensor {name}")
            a, b = left[name], right[name]
            if tuple(a.shape) != tuple(b.shape):
                raise ValueError(f"GGUF expert shape changed in quantization: {name}")
            if a.tensor_type != GGMLQuantizationType.BF16:
                raise ValueError(f"BF16 source expert tensor is not BF16: {name}")
            if b.tensor_type != GGMLQuantizationType.Q8_0:
                omitted.append({"tensor": name, "target_type": b.tensor_type.name})
                continue
            n = math.prod(int(v) for v in a.shape)
            observations = [_compare_window(a.data, b.data, element_offset=i)
                            for i in _windows(n)]
            rows.append({"tensor": name, "shape": [int(v) for v in a.shape],
                         "windows": observations})
    if not rows:
        raise ValueError("no Q8_0 expert tensor sampled: cannot measure quantization error")
    windows = [sample for row in rows for sample in row["windows"]]
    count = sum(w["count"] for w in windows)
    max_abs = max(w["max_abs_error"] for w in windows)
    mae = sum(w["sum_abs_error"] for w in windows) / count
    rmse = math.sqrt(sum(w["sum_squared_error"] for w in windows) / count)
    if any(_file_stat(Path(path)) != stamp for path, stamp in stamps.items()):
        raise ValueError("GGUF weights changed during the sampled quantization measurement")
    chunked = witnesses["q8_0"].get("schema") == "chowder.reap128-v2-gguf-q8-chunked-final.v1"
    return {
        "schema": "chowder.reap128-v2-gguf-q8-bf16-numeric-sample.v1",
        "status": "DESCRIPTIVE_QUANTIZATION_SAMPLE_NOT_QUALITY_GATE",
        "stage_witnesses": witnesses,
        "source_gguf": str(bf16),
        "q8_gguf": str(q8),
        "source_file_stat": list(stamps[str(bf16)]),
        "q8_file_stat": list(stamps[str(q8)]),
        "sample_rule": "3 routed expert types x 40 layers, four deterministic 256-value BF16/Q8 block-aligned windows per tensor",
        "compared_expert_tensors": len(rows),
        "non_q8_expert_tensors": omitted,
        "total_compared_values": count,
        "max_abs_error": max_abs, "mean_abs_error": mae,
        "root_mean_squared_error": rmse,
        "samples": rows,
        "limitations": (
            "Sampled numerical weight comparison only. "
            + ("Full-file GGUF and all 733 output tensor hashes were verified "
               "against the frozen BF16 stage and final chunked receipt. "
               if chunked else
               "Monolithic GGUF full-file hashes are recorded in stage "
               "receipts but not rehashed by this sampler. ")
            + "Does not measure perplexity, GSM8K, HumanEval, GPU fit, speed "
            "or any registered scientific quality/selection threshold."
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--q8", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true",
                        help="run bounded CPU-only GGUF sampled dequantization")
    args = parser.parse_args(argv)
    try:
        source, quantized = args.bf16.resolve(), args.q8.resolve()
        if (not args.out.parent.is_dir() or args.out.exists()
                or args.out.is_symlink()
                or args.out.resolve().is_relative_to(source.parent / source.name)
                or args.out.resolve().is_relative_to(quantized.parent / quantized.name)):
            raise ValueError("new external quantization sample receipt is required")
        if not args.execute:
            witnessed = _stage_identity(source, quantized)
            print(json.dumps({"status": "PREFLIGHT_ONLY_NUMERICS_NOT_MEASURED",
                              "stage_sha256": [witnessed["bf16"]["sha256"],
                                               witnessed["q8_0"].get("sha256",
                                                                     witnessed["q8_0"].get("output_sha256"))]}))
            return 0
        result = measure(source, quantized)
        with args.out.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({"status": result["status"],
                          "tensors": result["compared_expert_tensors"],
                          "values": result["total_compared_values"],
                          "max_abs_error": result["max_abs_error"],
                          "mean_abs_error": result["mean_abs_error"],
                          "receipt": str(args.out)}), flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"Q8 numeric sampling refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
