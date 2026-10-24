"""Bounded Kaggle dataset-mount streaming test; never loads a model.

Used to diagnose the Oct 2026 0.69 MB/s tensor-by-tensor checkpoint reader
without consuming GPU/TPU quota. Reads at most 64 MiB from ONE mounted shard,
for at most 120 seconds, without copying the entire private checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

MAX_SAMPLE_BYTES = 64 * 1024 * 1024
MAX_SECONDS = 120.0
CHUNK_BYTES = 4 * 1024 * 1024


def sample_mount(root: Path, *, cap_bytes: int = MAX_SAMPLE_BYTES,
                 cap_seconds: float = MAX_SECONDS) -> dict:
    if not (0 < cap_bytes <= MAX_SAMPLE_BYTES and 0 < cap_seconds <= MAX_SECONDS):
        raise ValueError("dataset streaming sample must stay inside fixed bounds")
    if not root.is_dir():
        return {"status": "MOUNT_MISSING", "bytes_read": 0}
    shards = sorted(root.rglob("*.safetensors"))
    if not shards:
        return {"status": "CHECKPOINT_SHARDS_ABSENT", "bytes_read": 0}
    shard = shards[0]
    digest = hashlib.sha256()
    size = shard.stat().st_size
    total = 0
    start = time.perf_counter()
    with shard.open("rb", buffering=CHUNK_BYTES) as stream:
        while total < min(size, cap_bytes) and time.perf_counter() - start < cap_seconds:
            data = stream.read(min(CHUNK_BYTES, size - total, cap_bytes - total))
            if not data:
                break
            digest.update(data)
            total += len(data)
    elapsed = max(time.perf_counter() - start, 1e-9)
    return {
        "status": "SAMPLE_ONLY_NO_FULL_STAGING",
        "shard_relative_path": str(shard.relative_to(root)),
        "shard_size_bytes": size,
        "bounded_bytes_read": total,
        "sample_sha256_not_full_weight_digest": digest.hexdigest(),
        "wall_seconds": round(elapsed, 4),
        "sample_mib_per_s": round(total / elapsed / 1024**2, 5),
        "declared_max_bytes": cap_bytes,
        "declared_max_seconds": cap_seconds,
        "full_checkpoint_staged": False,
        "registered_d0": "NOT_MEASURED",
    }


def main() -> None:
    root = Path("/kaggle/input")
    result = {
        "schema": "chowder.kaggle-checkpoint-mount-sample.v1",
        "dataset": sample_mount(root),
        "tmp_free_gib": round(shutil.disk_usage("/tmp").free / 1024**3, 3),
        "weights_downloaded": False,
        "gpu_initialized": False, "model_loaded": False,
        "registered_d0": "NOT_MEASURED",
    }
    target = Path("/kaggle/working/chowder-checkpoint-stream-sample.json")
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
