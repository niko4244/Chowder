"""Read-only postbuild source-bound Qwen BF16 REAP128_v2 qualification.

Production verification requires the registered 704×40 sealed observation,
exactly the pinned original BF16 model, the selection.json, and the completed
pruned model. Default compares metadata plus bounded deterministic expert
samples. --full-compare compares EVERY retained expert row and unmodified
tensor exactly and hashes all output shards before producing a receipt.
No GPU, inference service, quantization, or original files are modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors import safe_open

from chowder_batch.reap_hidden_checkpoint import sha256_file, validate_observation_receipt
from chowder_batch.reap_original_bf16_pin import validate_registered_source_metadata
from chowder_batch.reap_prune import COPY_FILES, EXPERT_KEY, kept_experts
from chowder_batch.reap_rank_experts import validate_selection

LAYERS = 40
EXPERTS = 256
KEEP = 128
CHUNK_TARGET_BYTES = 8 * 1024 ** 2
MAX_SAFETENSORS_HEADER_BYTES = 32 * 1024 ** 2


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _indexed(index: dict) -> dict[str, set[str]]:
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("missing model weight index")
    by_shard: dict[str, set[str]] = {}
    for key, name in mapping.items():
        if (not isinstance(key, str) or not isinstance(name, str)
                or Path(name).name != name or not name.endswith(".safetensors")):
            raise ValueError("unsafe index entry")
        by_shard.setdefault(name, set()).add(key)
    return by_shard


def _identical_bf16_bits(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Detect even signed-zero and NaN-payload differences, unlike float ==."""
    return (left.dtype == right.dtype == torch.bfloat16
            and left.shape == right.shape
            and bool(torch.equal(left.contiguous().view(torch.int16),
                                 right.contiguous().view(torch.int16))))


def _equal_source_row(
    original, pruned, key: str, source_row: int, destination_row: int,
) -> None:
    left = original.get_slice(key)[source_row:source_row + 1]
    right = pruned.get_slice(key)[destination_row:destination_row + 1]
    if not _identical_bf16_bits(left, right):
        raise ValueError(f"expert weight mismatch at {key} src={source_row} dst={destination_row}")


def _source_runs(rows: list[int]) -> list[tuple[int, int, int]]:
    """Consecutive source rows map to consecutive destination rows."""
    if not rows:
        return []
    runs = []
    start = last = rows[0]
    destination = 0
    for i, source_row in enumerate(rows[1:], 1):
        if source_row == last + 1:
            last = source_row
            continue
        runs.append((destination, start, last - start + 1))
        destination, start, last = i, source_row, source_row
    runs.append((destination, start, last - start + 1))
    return runs


def _equal_expert_rows(original, pruned, key: str, rows: list[int]) -> int:
    """Compare ALL BF16 selected rows in contiguous ~8 MiB tensor chunks."""
    source = original.get_slice(key)
    chosen = pruned.get_slice(key)
    stride = max(1, CHUNK_TARGET_BYTES // (math.prod(source.get_shape()[1:]) * 2))
    verified = 0
    for dst_start, src_start, length in _source_runs(rows):
        for offset in range(0, length, stride):
            n = min(stride, length - offset)
            left = source[src_start + offset:src_start + offset + n]
            right = chosen[dst_start + offset:dst_start + offset + n]
            if not _identical_bf16_bits(left, right):
                raise ValueError(f"expert block mismatch at {key}, src={src_start + offset}")
            verified += n
    return verified


def _equal_copied_tensor(original, pruned, key: str) -> None:
    a, b = original.get_slice(key), pruned.get_slice(key)
    shape = a.get_shape()
    if shape != b.get_shape() or a.get_dtype() != b.get_dtype():
        raise ValueError(f"unmodified tensor geometry differs: {key}")
    # Bounded tensor slices; includes the same BF16 dtype and bitwise equality.
    stride = max(1, CHUNK_TARGET_BYTES // (math.prod(shape[1:]) * 2)) if len(shape) > 1 else 4096
    for start in range(0, shape[0], stride):
        stop = min(start + stride, shape[0])
        if not _identical_bf16_bits(a[start:stop], b[start:stop]):
            raise ValueError(f"unmodified tensor bytes differ: {key} at {start}:{stop}")


@contextmanager
def _mapped_bf16_shard(path: Path):
    """Yield validated byte offsets into a READ-ONLY, noncommitted file mapping."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked BF16 shard: {path}")
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"short safetensors header: {path}")
        length = int.from_bytes(length_bytes, "little")
        if not 0 < length <= MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(f"invalid bounded safetensors header length: {path}")
        encoded = stream.read(length)
        if len(encoded) != length:
            raise ValueError(f"truncated safetensors metadata: {path}")
        data = json.loads(encoded)
        if not isinstance(data, dict):
            raise ValueError(f"invalid safetensors header: {path}")
        data.pop("__metadata__", None)
        start = 8 + length
        payload = path.stat().st_size - start
        intervals = []
        for key, info in data.items():
            if not isinstance(info, dict) or info.get("dtype") != "BF16":
                raise ValueError(f"non-BF16 tensor in {path}: {key}")
            shape, offsets = info.get("shape"), info.get("data_offsets")
            if (not isinstance(shape, list) or not shape
                    or any(type(n) is not int or n <= 0 for n in shape)
                    or not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(n) is not int for n in offsets)):
                raise ValueError(f"invalid tensor shape or offsets: {key}")
            begin, end = offsets
            if begin < 0 or end > payload or end - begin != math.prod(shape) * 2:
                raise ValueError(f"invalid tensor byte extent: {key}")
            intervals.append((begin, end))
        cursor = 0
        for begin, end in sorted(intervals):
            if begin != cursor:
                raise ValueError(f"noncontiguous or overlapping tensor extents: {path}")
            cursor = end
        if cursor != payload:
            raise ValueError(f"unindexed bytes inside safetensors payload: {path}")
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            yield mapped, data, start


def _compare_mapped_bf16_shard(
    original_path: Path, output_path: Path, names: set[str],
    kept: dict[int, list[int]],
) -> int:
    """Exact BF16 bytes via two file maps, never materializing weight tensors."""
    with _mapped_bf16_shard(original_path) as (source, source_data, source_start):
        with _mapped_bf16_shard(output_path) as (target, target_data, target_start):
            if set(target_data) != names or not names.issubset(source_data):
                raise ValueError("mmap tensor inventory differs from trusted index")
            checks = 0
            with memoryview(source) as left, memoryview(target) as right:
                for name in sorted(names):
                    raw, converted = source_data[name], target_data[name]
                    shape, resulting_shape = raw["shape"], converted["shape"]
                    source_begin = source_start + raw["data_offsets"][0]
                    target_begin = target_start + converted["data_offsets"][0]
                    selected = EXPERT_KEY.fullmatch(name)
                    if selected:
                        rows = kept[int(selected[1])]
                        if resulting_shape != [len(rows), *shape[1:]]:
                            raise ValueError(f"invalid retained expert geometry: {name}")
                        row_bytes = math.prod(shape[1:]) * 2
                        for dst, src, length in _source_runs(rows):
                            offset = 0
                            count_bytes = length * row_bytes
                            while offset < count_bytes:
                                size = min(CHUNK_TARGET_BYTES, count_bytes - offset)
                                so = source_begin + src * row_bytes + offset
                                to = target_begin + dst * row_bytes + offset
                                if left[so:so + size] != right[to:to + size]:
                                    raise ValueError(f"BF16 expert byte mismatch: {name}, src={src}")
                                offset += size
                        checks += len(rows)
                    else:
                        if resulting_shape != shape:
                            raise ValueError(f"copied BF16 tensor shape changed: {name}")
                        length = raw["data_offsets"][1] - raw["data_offsets"][0]
                        for offset in range(0, length, CHUNK_TARGET_BYTES):
                            size = min(CHUNK_TARGET_BYTES, length - offset)
                            if (left[source_begin + offset:source_begin + offset + size]
                                    != right[target_begin + offset:target_begin + offset + size]):
                                raise ValueError(f"copied BF16 tensor byte mismatch: {name}")
                        checks += 1
            return checks


def inspect(*, source: Path, observed: Path, selection: Path,
            candidate: Path, full_compare: bool = False,
            shard: str | None = None) -> dict:
    """Verified result only after all shape/mapping/provenance checks pass."""
    if shard is not None and not full_compare:
        raise ValueError("a per-shard certificate requires full bitwise comparison")
    source, observed, selection, candidate = (p.resolve() for p in
                                              (source, observed, selection, candidate))
    if source == candidate or candidate.is_relative_to(source) or source.is_relative_to(candidate):
        raise ValueError("pruned candidate overlaps the original")
    if not candidate.is_dir():
        raise ValueError("the pruned BF16 model has not been published")
    validate_registered_source_metadata(source, source)
    original_observation = _load(observed / "observation.json")
    validate_observation_receipt(observed, original_observation, require_registered_original=True)
    upstream = original_observation["checkpoint_receipt"]["source_identity"]
    if upstream["model_dir"] != str(source) or upstream["weights_dir"] != str(source):
        raise ValueError("registered observation and original model paths differ")
    chosen = _load(selection)
    kept = kept_experts(observed, LAYERS, EXPERTS, KEEP, "reap", 20261017)
    validate_selection(chosen, model_dir=source, weights_dir=source,
                       observations=observed, kept=kept, out=candidate)
    recorded = _load(candidate / "reap-prune.json")
    if (recorded.get("method") != "reap" or recorded.get("keep") != KEEP
            or recorded.get("of") != EXPERTS or recorded.get("seed") != 20261017
            or recorded.get("selection_sha256") != sha256_file(selection)
            or recorded.get("observation_digest") != sha256_file(observed / "observation.json")
            or recorded.get("kept_experts") != {str(i): v for i, v in kept.items()}):
        raise ValueError("published prune receipt and frozen selection differ")
    for layer in range(LAYERS):
        fname = observed / f"layer-{layer:02d}.pt"
        if recorded.get("observation_layer_sha256", {}).get(str(layer)) != sha256_file(fname):
            raise ValueError(f"pruned receipt lost layer-{layer:02d} saliency identity")
    if not isinstance(recorded.get("verified_live_upstream_weight_sha256"), dict):
        raise ValueError("missing producer-side original weight verification")
    upstream_weights = {
        key.removeprefix("weights/"): value
        for key, value in upstream["artifacts"].items() if key.startswith("weights/")
    }
    if recorded["verified_live_upstream_weight_sha256"] != upstream_weights:
        raise ValueError("pruner did not attest the original 26 live shard payloads")
    input_config = _load(source / "config.json")
    output_config = _load(candidate / "config.json")
    expected_config = json.loads(json.dumps(input_config))
    expected_config["text_config"]["num_experts"] = KEEP
    expected_config["text_config"]["mtp_num_hidden_layers"] = 0
    if output_config != expected_config:
        raise ValueError("output model configuration differs from frozen prune transformation")
    for name in COPY_FILES:
        original = source / name
        final = candidate / name
        if original.is_file() and (not final.is_file() or sha256_file(final) != sha256_file(original)):
            raise ValueError(f"copied tokenizer/model metadata changed: {name}")
    expected_index = _load(source / "model.safetensors.index.json")
    actual_index = _load(candidate / "model.safetensors.index.json")
    expected_shards = _indexed(expected_index)
    resulting_shards = _indexed(actual_index)
    expected_keys = {key for key in expected_index["weight_map"] if not key.startswith("mtp.")}
    if set(actual_index["weight_map"]) != expected_keys:
        raise ValueError("candidate index does not preserve non-MTP tensor inventory")
    if any(actual_index["weight_map"][key] != expected_index["weight_map"][key]
           for key in expected_keys):
        raise ValueError("candidate reshuffled original shard identity")
    if set(resulting_shards) != set(expected_shards):
        raise ValueError("candidate shard filenames differ")
    if shard is not None and shard not in resulting_shards:
        raise ValueError("requested candidate shard is not in the frozen index")
    source_size = output_size = touched = compared = 0
    output_shas = {}
    for filename in ([shard] if shard is not None else sorted(expected_shards)):
        original = source / filename
        output = candidate / filename
        if output.is_symlink() or not output.is_file() or output.resolve().parent != candidate:
            raise ValueError(f"candidate shard missing or unsafe: {filename}")
        if original.is_symlink() or not original.is_file() or original.resolve().parent != source:
            raise ValueError(f"original source shard missing or unsafe: {filename}")
        if full_compare:
            # Do not call safe_open(framework='pt') for full verification.
            # On Windows it can commit gigabytes merely to map the 26 shards.
            # Parse the bounded raw safetensors metadata and compare mapped
            # file bytes without allocating actual PyTorch weight tensors.
            with _mapped_bf16_shard(original) as (_mapping, first, _start):
                with _mapped_bf16_shard(output) as (_result, last, _output_start):
                    if set(last) != resulting_shards[filename]:
                        raise ValueError(f"candidate shard/header index mismatch: {filename}")
                    for key in sorted(resulting_shards[filename]):
                        if key not in first:
                            raise ValueError(f"indexed input tensor missing from shard: {key}")
                        source_shape, actual_shape = first[key]["shape"], last[key]["shape"]
                        transform = EXPERT_KEY.fullmatch(key)
                        expected_shape = ([KEEP, *source_shape[1:]]
                                          if transform else source_shape)
                        if actual_shape != expected_shape:
                            raise ValueError(f"incorrect tensor geometry: {key}")
                        source_size += math.prod(source_shape) * 2
                        output_size += math.prod(actual_shape) * 2
                        if transform:
                            touched += 1
            compared += _compare_mapped_bf16_shard(
                original, output, resulting_shards[filename], kept,
            )
            output_shas[filename] = sha256_file(output)
            continue
        with safe_open(str(original), framework="pt", device="cpu") as a, safe_open(
            str(output), framework="pt", device="cpu",
        ) as b:
            if set(b.keys()) != resulting_shards[filename]:
                raise ValueError(f"candidate shard/header index mismatch: {filename}")
            for key in sorted(resulting_shards[filename]):
                first, last = a.get_slice(key), b.get_slice(key)
                source_shape, actual_shape = first.get_shape(), last.get_shape()
                if first.get_dtype() != "BF16" or last.get_dtype() != "BF16":
                    raise ValueError(f"candidate has non-BF16 tensor: {key}")
                transform = EXPERT_KEY.fullmatch(key)
                expected_shape = ([KEEP, *source_shape[1:]]
                                  if transform else source_shape)
                if actual_shape != expected_shape:
                    raise ValueError(f"incorrect tensor geometry: {key}")
                source_size += math.prod(source_shape) * 2
                output_size += math.prod(actual_shape) * 2
                if transform:
                    touched += 1
                    rows = kept[int(transform[1])]
                    if not full_compare:
                        for j in sorted({0, 1, 63, 64, 126, 127}):
                            _equal_source_row(a, b, key, rows[j], j)
                            compared += 1
    if shard is None and (touched != LAYERS * 3 or output_size != recorded["total_bytes"]):
        raise ValueError("candidate bytes/routed tensor census differs from receipt")
    if shard is None and actual_index.get("metadata", {}).get("total_size") != output_size:
        raise ValueError("candidate safetensors index total_size is inaccurate")
    original_shard_sha = None
    if shard is not None:
        original_shard_sha = sha256_file(source / shard)
        if original_shard_sha != upstream_weights.get(shard):
            raise ValueError("original source shard changed since frozen BF16 prune: " + shard)
    return {
        "schema": "chowder.reap128-v2-pruned-bf16-verification.v1",
        "status": ("FULL_SOURCE_TENSOR_EQUIVALENCE_SHARD_PASS" if shard is not None
                   else "FULL_SOURCE_TENSOR_EQUIVALENCE_PASS" if full_compare
                   else "HEADER_AND_EXPERT_SAMPLE_PASS"),
        "individual_shard": shard,
        "original_shard_sha256": original_shard_sha,
        "original_source": str(source),
        "candidate": str(candidate),
        "selection_sha256": sha256_file(selection),
        "original_observation_sha256": sha256_file(observed / "observation.json"),
        "candidate_index_sha256": sha256_file(candidate / "model.safetensors.index.json"),
        "candidate_prune_receipt_sha256": sha256_file(candidate / "reap-prune.json"),
        "candidate_shard_sha256": output_shas if full_compare else None,
        "shards": 1 if shard is not None else len(resulting_shards),
        "routed_tensors": touched,
        "row_or_tensor_checks": compared,
        "source_non_mtp_tensor_bytes": source_size,
        "pruned_tensor_bytes": output_size,
        "no_gpu_loaded": True,
        "qualifies_scientific_accuracy": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "observed", "selection", "candidate"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--full-compare", action="store_true")
    parser.add_argument("--shard", help="run complete comparison for one indexed shard; partial receipt only")
    parser.add_argument("--out", type=Path, help="new JSON verification receipt (no overwrite)")
    args = parser.parse_args(argv)
    try:
        result = inspect(
            source=args.source, observed=args.observed, selection=args.selection,
            candidate=args.candidate, full_compare=args.full_compare, shard=args.shard,
        )
        record = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
        if args.out:
            with args.out.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(record)
        print(record, end="")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"REAP128_v2 pruned verification refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
