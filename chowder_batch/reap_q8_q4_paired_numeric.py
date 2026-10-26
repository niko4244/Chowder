"""Independent bounded-memory Q8_0 vs Q4_K_M paired expert-weight numerics.

Compares the exact same four source-aligned 256-element windows in 120 routed
expert tensors. This is *not* a token-forward, quality, CUDA or speed gate.
The published original GGUFs and their source-bound receipts are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

BF16_SHA = "80341aa776ad9b5f937c21b081b6d76de4a2ea277e5fb88c9ae2634a9da8f4c6"
Q8_SHA = "89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6"
Q4_SHA = "2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7"
Q8_BYTES = 19748435968
Q4_BYTES = 11563915264
TENSOR_COUNT = 733
EXPERT_SUFFIXES = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
Q4_TYPES = {"Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210)}


def _checked_publication_claims(q8: dict, q4: dict) -> None:
    expected = (
        (q8, "chowder.reap128-v2-gguf-q8-chunked-final.v1",
         "Q8_TENSOR_EXACT_CONSTRUCTED_NOT_QUALITY_QUALIFIED", Q8_SHA),
        (q4, "chowder.reap128-v2-gguf-q4-chunked-final.v1",
         "Q4_K_M_TENSOR_EXACT_CONSTRUCTED_NOT_QUALITY_QUALIFIED", Q4_SHA),
    )
    for record, schema, status, digest in expected:
        if (not isinstance(record, dict) or record.get("schema") != schema
                or record.get("status") != status
                or record.get("source_gguf_sha256") != BF16_SHA
                or record.get("output_sha256") != digest
                or record.get("quality_gate") != "NOT_EVALUATED"
                or record.get("gpu_fit_gate") != "NOT_EVALUATED"
                or not isinstance(record.get("chunked_reference_receipts"), dict)
                or not isinstance(record["chunked_reference_receipts"].get("all_tensors"), dict)):
            raise ValueError("Q8/Q4 publication differs from original common BF16 source/non-gate state")


def _aligned_offsets(element_count: int) -> list[int]:
    if type(element_count) is not int or element_count < 256 or element_count % 256:
        raise ValueError("paired Q8/Q4 expert length must be >=256 and K-block aligned")
    maximum = element_count - 256
    return sorted({(maximum * j // 3 // 256) * 256 for j in range(4)})


def _summarize(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("cannot certify an empty sample")
    for sample in samples:
        if type(sample.get("count")) is not int or sample["count"] != 256:
            raise ValueError("missing or malformed 256-element paired window")
        for key in ("max_abs_error", "sum_abs_error", "sum_squared_error"):
            value = sample.get(key)
            if (type(value) not in (float, int) or not math.isfinite(value)
                    or value < 0):
                raise ValueError("nonfinite/negative paired weight error")
    count = sum(row["count"] for row in samples)
    return {
        "status": "DESCRIPTIVE_Q8_Q4_WEIGHT_NUMERICS_NOT_QUALITY_GATE",
        "total_compared_values": count,
        "max_abs_error": max(row["max_abs_error"] for row in samples),
        "mean_abs_error": sum(row["sum_abs_error"] for row in samples) / count,
        "root_mean_squared_error": math.sqrt(sum(row["sum_squared_error"] for row in samples) / count),
    }


def _file_snapshot(path: Path) -> tuple[int, int, int]:
    state = path.stat()
    return state.st_size, state.st_mtime_ns, state.st_ctime_ns


def _hash_guarded(path: Path) -> str:
    """Stream full original model hash; refuse if Windows host headroom degrades."""
    from chowder_batch.reap_prune_supervised import host_state

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            buf = stream.read(8 * 1024 * 1024)
            if not buf:
                break
            digest.update(buf)
            if stream.tell() % (256 * 1024 * 1024) < len(buf):
                h = host_state(Path("F:/"))
                if h.ram_available_gib < 9 or h.commit_available_gib < 10:
                    raise ValueError("conservative RAM/commit guard refused original full-file read")
    return digest.hexdigest()


def _compare_window(q8_raw, q4_raw, kind: str, *, offset: int) -> dict:
    import numpy as np
    from gguf import GGMLQuantizationType, dequantize

    if kind not in Q4_TYPES or offset < 0 or offset % 256:
        raise ValueError("paired expert window is not K-block aligned")
    n_k, kbytes = Q4_TYPES[kind]
    if n_k != 256:
        raise ValueError("unexpected K-block geometry")
    q8_bytes = np.asarray(q8_raw).reshape(-1)[offset // 32 * 34:
                                               (offset + 256) // 32 * 34]
    q4_bytes = np.asarray(q4_raw).reshape(-1)[offset // 256 * kbytes:
                                               (offset // 256 + 1) * kbytes]
    if q8_bytes.size != 272 or q4_bytes.size != kbytes:
        raise ValueError("truncated paired expert payload")
    q8 = dequantize(q8_bytes.reshape(8, 34), GGMLQuantizationType.Q8_0).reshape(-1)
    q4 = dequantize(q4_bytes.reshape(1, kbytes), GGMLQuantizationType[kind]).reshape(-1)
    if q8.size != 256 or q4.size != 256 or not (np.isfinite(q8).all() and np.isfinite(q4).all()):
        raise ValueError("nonfinite or malformed paired Q8/Q4 weights")
    delta = q8.astype(np.float64) - q4.astype(np.float64)
    return {"offset": offset, "count": 256,
            "max_abs_error": float(np.abs(delta).max()),
            "sum_abs_error": float(np.abs(delta).sum()),
            "sum_squared_error": float((delta * delta).sum())}


def measure(q8_file: Path, q4_file: Path) -> dict:
    from gguf import GGUFReader

    q8_file, q4_file = q8_file.resolve(), q4_file.resolve()
    sides = [p.with_name(p.name + ".reap-chunked.json") for p in (q8_file, q4_file)]
    files = [q8_file, q4_file, *sides]
    if (any(p.is_symlink() or not p.is_file() for p in files)
            or any(p.stat().st_size > 2 * 1024 * 1024 for p in sides)
            or q8_file.stat().st_size != Q8_BYTES or q4_file.stat().st_size != Q4_BYTES):
        raise ValueError("published original Q8/Q4 files and bounded receipts are mandatory")
    stamps = [_file_snapshot(p) for p in files]
    q8_claim, q4_claim = [json.loads(p.read_text(encoding="utf8")) for p in sides]
    _checked_publication_claims(q8_claim, q4_claim)
    for path, digest in ((q8_file, Q8_SHA), (q4_file, Q4_SHA)):
        if _hash_guarded(path) != digest:
            raise ValueError("full original published Q8/Q4 GGUF SHA-256 mismatch")
    q8_reader, q4_reader = GGUFReader(str(q8_file), "r"), GGUFReader(str(q4_file), "r")
    q8_ts, q4_ts = q8_reader.tensors, q4_reader.tensors
    if (len(q8_ts) != TENSOR_COUNT or len(q4_ts) != TENSOR_COUNT
            or [t.name for t in q8_ts] != [t.name for t in q4_ts]):
        raise ValueError("original tensor inventory names/order changed")
    q8_map = {t.name: t for t in q8_ts}
    q4_map = {t.name: t for t in q4_ts}
    claim_a = q8_claim["chunked_reference_receipts"]["all_tensors"]
    claim_b = q4_claim["chunked_reference_receipts"]["all_tensors"]
    if (len(claim_a) != TENSOR_COUNT or len(claim_b) != TENSOR_COUNT
            or set(q8_map) != set(claim_a) or set(q4_map) != set(claim_b)):
        raise ValueError("original complete tensor receipt inventory differs")
    recorded = []
    for layer in range(40):
        for suffix in EXPERT_SUFFIXES:
            name = f"blk.{layer}.{suffix}"
            left, right = q8_map[name], q4_map[name]
            shape = list(map(int, left.shape))
            if (shape != list(map(int, right.shape)) or left.tensor_type.name != "Q8_0"
                    or right.tensor_type.name not in Q4_TYPES
                    or claim_a[name]["shape"] != shape or claim_b[name]["shape"] != shape
                    or claim_a[name]["tensor_type"] != "Q8_0"
                    or claim_b[name]["tensor_type"] != right.tensor_type.name):
                raise ValueError(f"source-aligned expert tensor incompatible: {name}")
            windows = [_compare_window(left.data, right.data, right.tensor_type.name, offset=n)
                       for n in _aligned_offsets(math.prod(shape))]
            if len(windows) != 4:
                raise ValueError("original four-window expert sample unexpectedly collapsed")
            recorded.append({"name": name, "type": right.tensor_type.name,
                             "shape": shape, "windows": windows})
    if len(recorded) != 120 or stamps != [_file_snapshot(p) for p in files]:
        raise ValueError("original Q8/Q4 files mutated or required paired tensors missing")
    summary = _summarize([window for item in recorded for window in item["windows"]])
    return {
        "schema": "chowder.reap-v2-q8-q4-independent-paired-weight-sample.v1",
        **summary,
        "q8_sha256": Q8_SHA, "q4_sha256": Q4_SHA,
        "common_original_bf16_sha256": BF16_SHA,
        "q8_sidecar_sha256": hashlib.sha256(sides[0].read_bytes()).hexdigest(),
        "q4_sidecar_sha256": hashlib.sha256(sides[1].read_bytes()).hexdigest(),
        "verified_named_tensor_inventory": TENSOR_COUNT,
        "sampled_expert_tensors": len(recorded),
        "windows_per_expert": 4,
        "samples": recorded,
        "limitations": "Original published GGUF weight sampling only; not a matching-token "
                       "Q8/Q4 forward pass, HumanEval/GSM8K, CUDA parity, GPU-fit or Gen-0 gate.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q8", type=Path, required=True)
    parser.add_argument("--q4", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if (args.out.exists() or args.out.is_symlink() or not args.out.parent.is_dir()
                or args.out.suffix != ".json"):
            raise ValueError("new exclusive external numerical JSON receipt required")
        result = measure(args.q8, args.q4)
        with args.out.open("x", encoding="utf8", newline="\n") as handle:
            json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps({key: result[key] for key in (
            "status", "sampled_expert_tensors", "total_compared_values",
            "max_abs_error", "mean_abs_error", "root_mean_squared_error")}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError) as exc:
        print(f"independent paired Q8/Q4 numerics refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
