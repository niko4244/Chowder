"""Source-bound, bounded-memory descriptive Q8_0 versus BF16 GGUF spot check.

This is an independent CPU-only *weight reconstruction* measurement, not
GSM8K/HumanEval, a full GGUF-byte attestation, inference, or a registered gate.
Both GGUFs must already have been atomically published by the frozen
REAP supervisor with exact stage witnesses and common recipe identity.

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
import sys
from pathlib import Path

BLOCK_VALUES = 32
Q8_BLOCK_BYTES = 34
WINDOW_VALUES = 256
BLOCK_COUNT = 4
LAYERS = 40
EXPERT_KEYS = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")


def _load_stage(path: Path, phase: int) -> dict:
    import re

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


def _stage_identity(bf16: Path, q8: Path) -> dict:
    source = _load_stage(bf16, 0)
    target = _load_stage(q8, 1)
    if (source["spec_sha256"] != target["spec_sha256"]
            or target.get("input_sha256") != source["sha256"]):
        raise ValueError("Q8 stage does not descend from the witnessed BF16 GGUF")
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
            "Sampled numerical weight comparison only. Full GGUF hashes were "
            "recorded by conversion supervisor stage receipts but not rehashed "
            "here. Does not measure perplexity, GSM8K, HumanEval, GPU fit, speed "
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
        witnessed = _stage_identity(source, quantized)
        if not args.execute:
            print(json.dumps({"status": "PREFLIGHT_ONLY_NUMERICS_NOT_MEASURED",
                              "stage_sha256": [witnessed[k]["sha256"]
                                               for k in ("bf16", "q8_0")]}))
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
