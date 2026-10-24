"""Bounded, SHA-verified mounted-shard -> local-scratch copy admission.

CPU only; copies at most 128 MiB from ONE existing private checkpoint shard,
then verifies the local copy and deletes it.  Never stages a complete model,
touches a GPU/TPU, or satisfies a frozen scientific gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

MAX_SAMPLE_BYTES = 128 * 1024 * 1024
MAX_SAMPLE_SECONDS = 120.0
CHUNK_BYTES = 4 * 1024 * 1024
MIN_SCRATCH_FREE_EXTRA_BYTES = 64 * 1024 * 1024


def _refuse_bad_budget(cap_bytes: int, cap_seconds: float) -> None:
    if not (type(cap_bytes) is int and 0 < cap_bytes <= MAX_SAMPLE_BYTES):
        raise ValueError("source copy cap must be between 1 and 128 MiB")
    if not (type(cap_seconds) in (float, int) and 0 < cap_seconds <= MAX_SAMPLE_SECONDS):
        raise ValueError("source copy timeout must be between 0 and 120 seconds")


def copy_verify_prefix(
    dataset: Path,
    scratch_root: Path,
    *,
    cap_bytes: int = MAX_SAMPLE_BYTES,
    cap_seconds: float = MAX_SAMPLE_SECONDS,
) -> dict[str, Any]:
    """Return a sample-only receipt; destructive scratch content is temporary."""
    _refuse_bad_budget(cap_bytes, cap_seconds)
    if not dataset.is_dir():
        return {"status": "DATASET_MOUNT_MISSING", "bounded_bytes_copied": 0}
    if not scratch_root.is_dir():
        return {"status": "SCRATCH_ROOT_MISSING", "bounded_bytes_copied": 0}
    dataset = dataset.resolve(strict=True)
    scratch_root = scratch_root.resolve(strict=True)
    if dataset == scratch_root or scratch_root.is_relative_to(dataset):
        raise ValueError("scratch must not be inside the read-only dataset")
    if dataset.is_relative_to(scratch_root):
        raise ValueError("dataset must not reside inside scratch")
    shards = sorted(dataset.rglob("*.safetensors"))
    if not shards:
        return {"status": "MOUNTED_SHARDS_ABSENT", "bounded_bytes_copied": 0}
    shard = shards[0]
    if not shard.is_file() or shard.is_symlink() or not shard.resolve(strict=True).is_relative_to(dataset):
        raise ValueError("checkpoint shard must be a regular file within the mounted dataset")
    shard_size = shard.stat().st_size
    if shard_size == 0:
        return {"status": "EMPTY_SHARD_REFUSED", "bounded_bytes_copied": 0}
    desired_bytes = min(shard_size, cap_bytes)
    free_bytes = shutil.disk_usage(scratch_root).free
    if free_bytes < desired_bytes + MIN_SCRATCH_FREE_EXTRA_BYTES:
        return {"status": "INSUFFICIENT_SCRATCH", "bounded_bytes_copied": 0,
                "scratch_free_bytes": free_bytes, "required_free_bytes":
                desired_bytes + MIN_SCRATCH_FREE_EXTRA_BYTES}
    source_sha = hashlib.sha256()
    copied_bytes = 0
    read_seconds = 0.0
    write_seconds = 0.0
    copy_start = time.perf_counter()
    # TemporaryDirectory both bounds persistent footprint and cleans up on
    # success or exception. Never touch a model's original mounted bytes.
    with tempfile.TemporaryDirectory(dir=scratch_root, prefix="chowder-stage-probe-") as tmp:
        stage = Path(tmp) / "prefix-sample.bin"
        with shard.open("rb", buffering=CHUNK_BYTES) as source, stage.open("xb") as dest:
            while copied_bytes < desired_bytes:
                if time.perf_counter() - copy_start >= cap_seconds:
                    break
                t0 = time.perf_counter()
                chunk = source.read(min(CHUNK_BYTES, desired_bytes - copied_bytes))
                read_seconds += time.perf_counter() - t0
                if not chunk:
                    break
                t0 = time.perf_counter()
                dest.write(chunk)
                write_seconds += time.perf_counter() - t0
                source_sha.update(chunk)
                copied_bytes += len(chunk)
            # Exercise actual local scratch write and buffer drain, not just
            # a Python buffered copy. This is a tiny bounded diagnostic.
            t0 = time.perf_counter()
            dest.flush()
            os.fsync(dest.fileno())
            write_seconds += time.perf_counter() - t0
        copy_seconds = max(time.perf_counter() - copy_start, 1e-9)
        verified_sha = hashlib.sha256()
        checked_bytes = 0
        verify_start = time.perf_counter()
        with stage.open("rb", buffering=CHUNK_BYTES) as check:
            for block in iter(lambda: check.read(CHUNK_BYTES), b""):
                verified_sha.update(block)
                checked_bytes += len(block)
        verify_seconds = time.perf_counter() - verify_start
        match = checked_bytes == copied_bytes and verified_sha.digest() == source_sha.digest()
        # The remote sample will *never* report a whole checkpoint staged,
        # even if a test shard happens to be <= cap_bytes.
        receipt = {
            "status": ("SAMPLE_COPY_VERIFIED" if match and copied_bytes == desired_bytes
                       else "SAMPLE_COPY_TIME_LIMITED" if match else
                       "SAMPLE_COPY_INTEGRITY_FAILURE"),
            "source_shard_relative_path": str(shard.relative_to(dataset)),
            "source_shard_total_bytes": shard_size,
            "declared_cap_bytes": cap_bytes,
            "declared_copy_timeout_seconds": cap_seconds,
            "bounded_bytes_copied": copied_bytes,
            "source_prefix_sha256_not_checkpoint_hash": source_sha.hexdigest(),
            "scratch_prefix_sha256": verified_sha.hexdigest(),
            "verified_bytes": checked_bytes,
            "prefix_integrity_pass": match,
            "source_read_seconds": round(read_seconds, 4),
            "scratch_write_fsync_seconds": round(write_seconds, 4),
            "copy_wall_seconds": round(copy_seconds, 4),
            "local_verify_seconds": round(verify_seconds, 4),
            "copy_mib_per_second": round(copied_bytes / copy_seconds / 1024**2, 4),
            "scratch_free_gib_before": round(free_bytes / 1024**3, 3),
            "scratch_sample_deleted_after_check": True,
            "original_parent_staged": False,
            "full_checkpoint_staged": False,
            "frozen_gate_d0": "NOT_MEASURED",
        }
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--scratch", type=Path, default=Path("/tmp"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or not args.out.parent.is_dir():
        parser.error("output path must not exist and parent must already exist")
    report = {
        "schema": "chowder.kaggle-stage-verified-prefix.v1",
        "sample": copy_verify_prefix(args.dataset, args.scratch),
        "gpu_or_tpu_initialized": False,
        "weights_downloaded": False,
        "model_loaded": False,
        "trained_steps": 0,
        "gate_d0": "NOT_MEASURED",
    }
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
