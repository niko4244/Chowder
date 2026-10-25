"""Source-bound, bounded CPU Q4_K_M GGUF versus BF16 descriptive error sample.

This verifies all 733 published Q4 tensor bytes, including mixed K-block
types, before sampling routed expert windows; it is NOT a quality gate.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

from chowder_batch import reap_gguf_chunked as core
from chowder_batch import reap_gguf_quant_sample as q8
from chowder_batch.reap_gguf_q4_chunked import BINARY_SHA256, Q4_FILE_TYPE

Q4_SCHEMA = "chowder.reap128-v2-gguf-q4-chunked-final.v1"
TYPES = {"Q4_K": (256, 144), "Q5_K": (256, 176),
         "Q6_K": (256, 210), "Q8_0": (32, 34)}
COPIED = {"F32", "F16", "BF16"}
VALUES_PER_WINDOW = 256


def _verified_pair(bf16: Path, quantized: Path) -> tuple[dict, object, object]:
    from gguf import GGUFReader

    receipt_path = quantized.with_name(quantized.name + ".reap-chunked.json")
    if any(path.is_symlink() or not path.is_file() for path in
           (bf16, quantized, receipt_path)) or receipt_path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Q4 publication and immutable source-bound witness required")
    source = q8._load_stage(bf16, 0)
    if (source["sha256"] != q8.FROZEN_SOURCE_SHA256
            or source["spec_sha256"] != q8.FROZEN_SPEC_SHA256
            or q8._sha256_file(bf16) != source["sha256"]):
        raise ValueError("BF16 source is not original frozen GGUF")
    claim = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(claim, dict):
        raise ValueError("Q4 witness must be a JSON object")
    reference = claim.get("chunked_reference_receipts")
    if not isinstance(reference, dict):
        raise ValueError("Q4 tensor witness collection is missing or malformed")
    groups = reference.get("source_groups")
    proofs = reference.get("all_tensors")
    digest = claim.get("output_sha256")
    if (claim.get("schema") != Q4_SCHEMA
            or claim.get("status") != "Q4_K_M_TENSOR_EXACT_CONSTRUCTED_NOT_QUALITY_QUALIFIED"
            or claim.get("output") != str(quantized)
            or claim.get("output_size_bytes") != quantized.stat().st_size
            or claim.get("source_gguf_sha256") != source["sha256"]
            or claim.get("frozen_stage_spec_sha256") != source["spec_sha256"]
            or claim.get("quantizer_binary_sha256") != BINARY_SHA256
            or claim.get("quality_gate") != "NOT_EVALUATED"
            or claim.get("gpu_fit_gate") != "NOT_EVALUATED"
            or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not isinstance(groups, list) or not isinstance(proofs, dict)
            or len(groups) != claim.get("chunk_count") or len(proofs) != 733):
        raise ValueError("Q4 final witness identity or non-gate provenance differs")
    names = [name for group in groups if isinstance(group, list) for name in group]
    if (len(names) != 733 or len(set(names)) != 733
            or set(names) != set(proofs) or len(groups) < 1):
        raise ValueError("Q4 source groups do not exactly cover 733 tensors")
    if q8._sha256_file(quantized) != digest:
        raise ValueError("full Q4 GGUF SHA differs from published receipt")
    original = GGUFReader(str(bf16), "r")
    packed = GGUFReader(str(quantized), "r")
    for reader, kind in ((original, 32), (packed, Q4_FILE_TYPE)):
        if any(reader.get_field(field) is None or reader.get_field(field).contents() != val
               for field, val in (("general.file_type", kind),
                                  ("qwen35moe.block_count", 40),
                                  ("qwen35moe.expert_count", 128),
                                  ("qwen35moe.expert_used_count", 8))):
            raise ValueError("frozen Q4 expert geometry or GGUF mode differs")
    left, right = original.tensors, packed.tensors
    if (len(left) != len(right) or len(left) != 733
            or [t.name for t in left] != names
            or [t.name for t in right] != names):
        raise ValueError("Q4 output tensor order and source inventory differ")
    has_q4 = False
    for src, dst in zip(left, right):
        record = proofs[dst.name]
        if (not isinstance(record, dict)
                or src.tensor_type.name not in {"BF16", "F32"}
                or dst.tensor_type.name not in set(TYPES) | COPIED
                or not core._quantized_shape_matches(src, dst)
                or record.get("shape") != list(map(int, dst.shape))
                or record.get("tensor_type") != dst.tensor_type.name
                or record.get("raw_sha256") != core.hash_tensor_span(
                    quantized, int(dst.data_offset), int(dst.n_bytes))):
            raise ValueError(f"Q4 tensor type/shape/payload witness changed: {dst.name}")
        # A copied original F32/BF16 payload must remain bit-for-bit equal to
        # its actual BF16-stage source, not merely to a self-consistent,
        # potentially re-sealed Q4 final receipt.
        if (dst.tensor_type.name in COPIED and
                (dst.tensor_type != src.tensor_type
                 or core.hash_tensor_span(bf16, int(src.data_offset),
                                          int(src.n_bytes)) != record["raw_sha256"])):
            raise ValueError(f"Q4 copied source tensor ancestry changed: {dst.name}")
        has_q4 |= dst.tensor_type.name == "Q4_K"
    if not has_q4:
        raise ValueError("Q4_K_M recipe has no actual Q4_K payload")
    return claim, original, packed


def _window(bf16_raw, quantized_raw, kind, position: int) -> dict:
    import numpy as np
    from gguf import GGMLQuantizationType, dequantize

    block_values, block_bytes = TYPES[kind]
    if position % block_values:
        raise ValueError("unaligned K-quant block sampling offset")
    original_start = position * 2
    original = np.asarray(bf16_raw).reshape(-1)[original_start:
                                                 original_start + 512]
    start = position // block_values * block_bytes
    packed = np.asarray(quantized_raw).reshape(-1)[
        start:start + VALUES_PER_WINDOW // block_values * block_bytes]
    if original.size != 512 or packed.size != VALUES_PER_WINDOW // block_values * block_bytes:
        raise ValueError("truncated quantized expert comparison window")
    exact = (np.frombuffer(original.tobytes(), dtype="<u2").astype("<u4")
             << 16).view("<f4")
    decoded = dequantize(packed.reshape(-1, block_bytes),
                         GGMLQuantizationType[kind]).reshape(-1)
    if (exact.size != VALUES_PER_WINDOW or decoded.size != VALUES_PER_WINDOW
            or not np.isfinite(exact).all() or not np.isfinite(decoded).all()):
        raise ValueError("Q4 dequant sample contains nonfinite or malformed values")
    delta = decoded - exact
    return {"offset": position, "count": VALUES_PER_WINDOW,
            "max_abs_error": float(np.abs(delta).max()),
            "sum_abs_error": float(np.abs(delta).sum(dtype=np.float64)),
            "sum_squared_error": float((delta.astype(np.float64) ** 2).sum())}


def measure(bf16: Path, q4file: Path) -> dict:
    bf16, q4file = bf16.resolve(), q4file.resolve()
    sidecar = q4file.with_name(q4file.name + ".reap-chunked.json")
    before = tuple(q8._file_stat(p) for p in (bf16, q4file, sidecar))
    receipt, original, quantized = _verified_pair(bf16, q4file)
    left = {t.name: t for t in original.tensors}
    right = {t.name: t for t in quantized.tensors}
    windows, omitted, measured = [], [], []
    for layer in range(40):
        for suffix in q8.EXPERT_KEYS:
            name = f"blk.{layer}.{suffix}"
            src, dst = left[name], right[name]
            if src.tensor_type.name != "BF16":
                raise ValueError(f"not BF16 source expert: {name}")
            kind = dst.tensor_type.name
            if kind not in TYPES:
                omitted.append({"tensor": name, "type": kind})
                continue
            n = math.prod(map(int, src.shape))
            if n < VALUES_PER_WINDOW or n % TYPES[kind][0]:
                raise ValueError(f"invalid quantized expert element count: {name}")
            maximum = n - VALUES_PER_WINDOW
            offsets = sorted({(maximum * j // 3 // TYPES[kind][0])
                              * TYPES[kind][0] for j in range(4)})
            observed = [_window(src.data, dst.data, kind, i) for i in offsets]
            measured.append({"tensor": name, "type": kind, "windows": observed})
            windows.extend(observed)
    if not windows:
        raise ValueError("no Q4 expert payload was available for numerical sampling")
    if before != tuple(q8._file_stat(p) for p in (bf16, q4file, sidecar)):
        raise ValueError("Q4 source, GGUF or sidecar changed during verification")
    n = sum(w["count"] for w in windows)
    return {"schema": "chowder.reap128-v2-q4-bf16-numeric-sample.v1",
            "status": "DESCRIPTIVE_Q4_QUANTIZATION_SAMPLE_NOT_QUALITY_GATE",
            "source_sha256": q8.FROZEN_SOURCE_SHA256,
            "quantized_sha256": receipt["output_sha256"],
            "quantizer_sha256": BINARY_SHA256,
            "source": str(bf16), "model": str(q4file),
            "source_stage_sha256": q8._sha256_file(bf16.with_name(bf16.name + ".reap-stage.json")),
            "final_receipt_sha256": q8._sha256_file(sidecar),
            "verified_tensor_count": 733,
            "compared_expert_tensors": len(measured),
            "omitted_expert_tensors": omitted,
            "total_compared_values": n,
            "max_abs_error": max(w["max_abs_error"] for w in windows),
            "mean_abs_error": sum(w["sum_abs_error"] for w in windows) / n,
            "root_mean_squared_error": math.sqrt(
                sum(w["sum_squared_error"] for w in windows) / n),
            "samples": measured,
            "limitations": "Sampled BF16-to-Q4 reconstruction only; not registered "
                           "GSM8K, HumanEval, CUDA, GPU fit or throughput evidence."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", required=True, type=Path)
    parser.add_argument("--q4", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.out.exists() or args.out.is_symlink() or not args.out.parent.is_dir():
            raise ValueError("new exclusive Q4 numerical receipt required")
        result = measure(args.bf16, args.q4)
        with args.out.open("x", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: result[key] for key in (
            "status", "verified_tensor_count", "compared_expert_tensors",
            "total_compared_values", "max_abs_error", "mean_abs_error",
            "root_mean_squared_error")}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError) as exc:
        print(f"Q4 BF16 numeric sample refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
