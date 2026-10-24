"""CPU-only microbench: original list scan versus verified set-based index scan.

No model tensors or GPU are loaded. The synthetic file contents are 1 byte;
benchmark measures only index JSON parsing and shard filename resolution.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chowder.backends.moe_int8_experts import _shard_files  # noqa: E402


def _original_shard_files(base: Path) -> list[Path]:
    """Reference implementation from f099d3a, for *valid* index fixtures."""
    index_path = base / "model.safetensors.index.json"
    shards: list[Path] = []
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for name in (index.get("weight_map") or {}).values():
            path = base / str(name)
            if path not in shards:
                shards.append(path)
    if not shards:
        shards = sorted(base.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards under {str(base)!r}")
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"the safetensors index names shards that are missing: {missing[:3]}")
    return shards


def _timed(func, root: Path, loops: int) -> float:
    samples = []
    for _ in range(loops):
        begin = time.perf_counter()
        func(root)
        samples.append((time.perf_counter() - begin) * 1000.0)
    return statistics.median(samples)


def benchmark(keys: int, shards: int, loops: int) -> dict[str, float | int]:
    with tempfile.TemporaryDirectory(prefix="moe-shard-bench-") as directory:
        root = Path(directory)
        names = [f"model-{i:05d}-of-{shards:05d}.safetensors" for i in range(1, shards + 1)]
        for name in names:
            (root / name).write_bytes(b"x")
        weight_map = {f"model.layers.{i}.weight": names[i % shards] for i in range(keys)}
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}), encoding="utf-8"
        )
        expected = _original_shard_files(root)
        assert _shard_files(root) == expected, "new resolution changed the valid shard order"
        _original_shard_files(root)
        _shard_files(root)
        old = _timed(_original_shard_files, root, loops)
        new = _timed(_shard_files, root, loops)
        return {"keys": keys, "shards": shards, "loops": loops, "old_median_ms": round(old, 4),
                "new_median_ms": round(new, 4), "speedup": round(old / new, 2)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", type=int, default=1026)
    parser.add_argument("--shards", type=int, default=26)
    parser.add_argument("--loops", type=int, default=40)
    args = parser.parse_args()
    if min(args.keys, args.shards, args.loops) <= 0 or args.keys < args.shards:
        parser.error("--keys must be >= --shards, and each count must be positive")
    print(json.dumps(benchmark(args.keys, args.shards, args.loops)))


if __name__ == "__main__":
    main()
