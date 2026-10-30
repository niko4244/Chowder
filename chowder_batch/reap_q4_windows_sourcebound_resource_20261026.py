"""Read-only source-pinned resource accounting for the original Windows Q4 200.

Only computes measured sampled availability/placement/cache, not allocator
peaks, model quality eligibility, or unmeasured dense/Q8 ratios. Private raw
generation bytes never appear in this public-safe summary.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re

from chowder_batch.reap_external_full_audit import _strict_json

ARTIFACTS = {
    "resource-samples.jsonl": ("c144f1692e823800d6b4fdcf87ee9f4a261acd8f5f13a207ef1b097886b428e8", 256 * 1024),
    "watchdog-terminal.json": ("f4f861f41b90fc76e7173ad6fab14287e8588139b64580d02c67d2124db01fc9", 1024 * 1024),
    "reap128_v2_q4/server.log": ("fd836dc8715297628bea6c6b9d750868ec9aee89fbafd7e3223c52a367257f5a", 8 * 1024 * 1024),
    "reap128_v2_q4/serve.json": ("6902322460420afe11228f663c9ddf38a67b8d628fe9983f92391d71d49705c8", 1024 * 1024),
    "reap128_v2_q4/gsm8k.jsonl": ("f3bae8ada933f0b85dae4d8946542f1901b02ef9e5ffead8411c4d449ea2472e", 2 * 1024 * 1024),
}
MODEL_SHA = "2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7"
SESSION = "e3f9c268b2bb4d958cc55648c0c4c8e0"
FROZEN = ["--fit", "on", "-np", "4", "-c", "16384", "--jinja"]
RESOURCE_KEYS = ("ram_available_gib", "commit_available_gib", "gpu_free_mib", "disk_free_gib")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_exact(root: Path, relative: str) -> bytes:
    expected, limit = ARTIFACTS[relative]
    path = root / relative
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError("original bounded Q4 Windows private source absent")
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError("original Windows Q4 private source hash differs")
    return raw


def summarize(samples_raw: bytes, terminal_raw: bytes, server_raw: bytes,
              serve_raw: bytes, rows_raw: bytes) -> dict:
    """Fail closed on compromised samples/log/source, no per-row text emitted."""
    if any(not isinstance(blob, bytes) for blob in (
            samples_raw, terminal_raw, server_raw, serve_raw, rows_raw)):
        raise ValueError("input bytes missing")
    terminal = _strict_json(terminal_raw)
    serve = _strict_json(serve_raw)
    speed = serve.get("speed_session")
    if (terminal.get("status") != "COMPLETE_SOURCE_BOUND_200_CANDIDATE_ONLY"
            or type(terminal.get("child_exit_code")) is not int
            or terminal["child_exit_code"] != 0
            or terminal.get("abort_reasons")
            or terminal.get("partial_or_complete_rows") != 200
            or serve.get("gguf_sha256") != MODEL_SHA
            or serve.get("eval_session_id") != SESSION
            or serve.get("flags") != FROZEN
            or type(serve.get("measured_experts_offloaded")) is not int
            or serve["measured_experts_offloaded"] != 0
            or not isinstance(speed, dict)
            or speed.get("schema") != "chowder.reap-q4-same-session-speed.v1"
            or speed.get("session_id") != SESSION
            or speed.get("n") != 200
            or speed.get("rows_sha256") != digest(rows_raw)
            or terminal.get("sample_count") != 148):
        raise ValueError("frozen Windows Q4 serving/session/terminal continuity missing")
    rows = [_strict_json(line) for line in rows_raw.splitlines()]
    if (len(rows) != 200 or len({r.get("id") for r in rows}) != 200
            or {r.get("id") for r in rows} != {f"gsm8k-{i}" for i in range(200)}
            or any(type(r.get("correct")) is not bool
                   or type(r.get("terminated")) is not bool for r in rows)):
        raise ValueError("original scored rows are missing/duplicated/ambiguous")
    if sum(int(r["correct"]) for r in rows) != 187:
        raise ValueError("original Windows Q4 correct count unexpectedly changed")
    samples = [_strict_json(line) for line in samples_raw.splitlines()]
    if len(samples) != 148:
        raise ValueError("full live Q4 resource sample count missing")
    readings = {name: [] for name in RESOURCE_KEYS}
    dates = []
    for row in samples:
        stamp = datetime.fromisoformat(row["time_utc"])
        if stamp.utcoffset() != timedelta(0):
            raise ValueError("non-UTC or naive Windows watchdog sample")
        dates.append(stamp)
        for name in RESOURCE_KEYS:
            value = row.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("nonfinite/invalid original resource telemetry")
            readings[name].append(float(value))
    gaps = [(b - a).total_seconds() for a, b in zip(dates, dates[1:])]
    if any(not 0 < gap <= 10 for gap in gaps):
        raise ValueError("resource watchdog order or continuity mismatch")
    minimum = {key: min(values) for key, values in readings.items()}
    if (minimum["ram_available_gib"] < 7
            or minimum["commit_available_gib"] < 8
            or minimum["disk_free_gib"] < 8
            or readings["gpu_free_mib"][0] < 12288):
        raise ValueError("original Q4 protected runtime/start floors not demonstrated")
    native = server_raw.decode("utf8", errors="replace")
    model = re.findall(r"CUDA0 model buffer size\s*=\s*([\d.]+) MiB", native)
    kv = re.findall(r"CUDA0 KV buffer size\s*=\s*([\d.]+) MiB", native)
    cache = re.findall(
        r"cache state:\s*(\d+) prompts,\s*([\d.]+) MiB \(limits:\s*([\d.]+) MiB",
        native,
    )
    if (len(model) != 1 or len(kv) != 1 or not cache
            or "offloaded 41/41 layers to GPU" not in native
            or "n_seq_max             = 4" not in native
            or "n_ctx_seq             = 4096" not in native
            or float(kv[0]) != 320.0
            or any(float(entry[2]) != 8192.0 for entry in cache)):
        raise ValueError("source-bound native CUDA/KV/cache evidence missing or incompatible")
    cache_max = max(float(entry[1]) for entry in cache)
    if not 0 <= cache_max <= 8192:
        raise ValueError("native prompt-cache consumption inconsistent with 8192MiB cap")
    return {
        "schema": "chowder.reap-v2-original-windows-q4-first200-sampled-resources.v1",
        "status": "SOURCE_BOUND_WINDOWS_Q4_FIRST200_SAMPLED_RESOURCES_CANDIDATE_ONLY",
        "original_model_sha256": MODEL_SHA,
        "serving_session_id": SESSION,
        "source_sha256": {name: digest(blob) for name, blob in zip(
            ARTIFACTS, (samples_raw, terminal_raw, server_raw, serve_raw, rows_raw))},
        "questions": 200, "correct": 187, "frozen_flags": FROZEN,
        "samples": len(samples), "maximum_watchdog_gap_seconds": max(gaps),
        "initial_available": {key: values[0] for key, values in readings.items()},
        "lowest_sampled_available": minimum,
        "largest_sampled_decline_from_initial": {
            key: values[0] - min(values) for key, values in readings.items()},
        "native_cuda_model_buffer_mib": float(model[0]),
        "native_cuda_kv_buffer_mib": float(kv[0]),
        "native_prompt_cache_cap_mib": 8192.0,
        "native_prompt_cache_max_observed_mib": cache_max,
        "native_gpu_layers_offloaded": "41/41",
        "native_context_total": 16384,
        "native_parallel_sequences": 4,
        "cuda_allocator_instantaneous_peak_certified": False,
        "wddm_total_dedicated_resident_bytes_measured": False,
        "original_dense_matched_first200_available": False,
        "speed_ratio_1p3_certified": False,
        "gen0_status": "BLOCKED", "chosen": None, "release_approved": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if (not args.private_run.is_absolute() or not args.private_run.is_dir()
            or not args.out.is_absolute() or args.out.is_symlink()
            or args.out.exists() or not args.out.parent.is_dir()):
        raise ValueError("original private run and exclusive output paths required")
    blobs = [read_exact(args.private_run, name) for name in ARTIFACTS]
    data = summarize(*blobs)
    with args.out.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({
        "status": data["status"], "samples": data["samples"],
        "lowest_sampled_available": data["lowest_sampled_available"],
        "native_prompt_cache_max_observed_mib":
            data["native_prompt_cache_max_observed_mib"],
        "gen0_status": "BLOCKED",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
