"""CPU-only admission of the frozen 704-row REAP128_v2 observer input.

Read metadata and safetensors headers only. This checks the inputs *before*
the expensive 40-layer observer allocates GPU tensors. It neither initializes
CUDA nor writes an observation or claims that Gate D0 has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from chowder_batch.reap_calibration import NGRAM, SEED
from chowder_batch.reap_v2_source_identity import (
    FROZEN_V2_BASE_ROWS,
    V2SourceIdentityError,
    frozen_v2_source_identity,
)

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_CALIBRATION_BYTES = 64 * 1024 * 1024
MAX_HEADER_BYTES = 16 * 1024 * 1024
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
               "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
               "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
PREFIX = "model.language_model."
REQUIRED_TENSORS = {
    f"{PREFIX}layers.{i}.mlp.{name}"
    for i in range(40)
    for name in ("experts.gate_up_proj", "experts.down_proj", "gate.weight")
} | {f"{PREFIX}embed_tokens.weight", f"{PREFIX}norm.weight", "lm_head.weight"}


class ObserverAdmissionError(ValueError):
    """The supplied source cannot be admitted to the frozen v2 observation."""


def _check(condition: bool, explanation: str) -> None:
    if not condition:
        raise ObserverAdmissionError(explanation)


def _json(path: Path) -> dict[str, Any]:
    _check(path.is_file() and path.stat().st_size <= MAX_JSON_BYTES,
           f"missing or oversized metadata JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ObserverAdmissionError(f"unreadable metadata JSON: {path}: {exc}") from exc
    _check(isinstance(value, dict), f"metadata JSON must be an object: {path}")
    return value


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_calibration(calibration: Path) -> dict[str, Any]:
    manifest = _json(calibration / "calibration-manifest.json")
    path = calibration / "calibration.jsonl"
    _check(path.is_file() and path.stat().st_size <= MAX_CALIBRATION_BYTES,
           "frozen v2 calibration absent or exceeds metadata inspection budget")
    raw = path.read_bytes()
    actual_sha = _sha(raw)
    _check(manifest.get("mix") == "v2" and manifest.get("calibration_sha256") == actual_sha,
           "v2 calibration/manifest SHA256 or mix mismatch")
    try:
        frozen = frozen_v2_source_identity(calibration, manifest)
    except (V2SourceIdentityError, OSError) as exc:
        raise ObserverAdmissionError(f"registered v2 source identity refused: {exc}") from exc
    _check(manifest.get("seed") == SEED and manifest.get("ngram") == NGRAM
           and type(manifest.get("max_tokens")) is int and manifest["max_tokens"] == 2048,
           "v2 manifest differs from registered seed/13-gram/2048-token packing")
    _check(type(manifest.get("rows")) is int and manifest["rows"] == FROZEN_V2_BASE_ROWS,
           "v2 manifest must declare all 704 registered calibration rows")
    sources = manifest.get("sources")
    _check(isinstance(sources, dict) and bool(sources), "v2 source manifest absent")
    try:
        rows = [json.loads(line) for line in raw.splitlines()]
    except (ValueError, UnicodeError) as exc:
        raise ObserverAdmissionError(f"v2 calibration JSONL invalid: {exc}") from exc
    _check(len(rows) == FROZEN_V2_BASE_ROWS, "v2 calibration has an incorrect row census")
    counts: Counter[str] = Counter()
    token_count = 0
    for i, row in enumerate(rows):
        _check(isinstance(row, dict) and isinstance(row.get("text"), str)
               and bool(row["text"]) and isinstance(row.get("source"), str)
               and row["source"] in sources and type(row.get("n_tokens")) is int
               and 256 <= row["n_tokens"] <= 2048,
               f"v2 calibration row {i} has invalid source/text/length")
        counts[row["source"]] += 1
        token_count += row["n_tokens"]
    _check(type(manifest.get("tokens")) is int and manifest["tokens"] == token_count,
           "v2 calibration manifest token count mismatch")
    for name, stats in sources.items():
        _check(isinstance(stats, dict) and type(stats.get("kept")) is int
               and stats["kept"] == counts[name], f"v2 source {name!r} count mismatch")
    return {"sha256": actual_sha, "rows": len(rows), "declared_tokens": token_count,
            "source_counts": dict(sorted(counts.items())), **frozen}


def _header(path: Path) -> dict[str, Any]:
    """Read only a shard's safetensors header and verify every data span exists."""
    with path.open("rb") as stream:
        length = stream.read(8)
        _check(len(length) == 8, f"safetensors shard truncated before header: {path.name}")
        size = struct.unpack("<Q", length)[0]
        _check(0 < size <= MAX_HEADER_BYTES,
               f"safetensors shard header exceeds bounded inspection: {path.name}")
        raw = stream.read(size)
        _check(len(raw) == size, f"safetensors shard truncated header: {path.name}")
    try:
        header = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ObserverAdmissionError(f"invalid safetensors header {path.name}: {exc}") from exc
    _check(isinstance(header, dict), f"invalid safetensors header object: {path.name}")
    payload_bytes = path.stat().st_size - 8 - size
    spans: list[tuple[int, int]] = []
    for key, desc in header.items():
        if key == "__metadata__":
            continue
        _check(isinstance(desc, dict), f"invalid tensor header for {key}")
        span = desc.get("data_offsets")
        _check(isinstance(span, list) and len(span) == 2
               and all(type(x) is int for x in span)
               and 0 <= span[0] <= span[1] <= payload_bytes,
               f"tensor {key} references bytes outside {path.name}")
        shape = desc.get("shape")
        size_per_element = DTYPE_BYTES.get(desc.get("dtype"))
        _check(isinstance(shape, list) and all(type(x) is int and x >= 0 for x in shape)
               and size_per_element is not None
               and math.prod(shape) * size_per_element == span[1] - span[0],
               f"tensor {key} has an inconsistent dtype/shape/byte range")
        spans.append((span[0], span[1]))
    ordered = sorted(spans)
    _check(all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:])),
           f"safetensors header has overlapping tensor ranges: {path.name}")
    return {k: v for k, v in header.items() if k != "__metadata__"}


def _validate_model(model: Path, weights: Path, declared_tokens: int) -> dict[str, Any]:
    config_path = model / "config.json"
    index_path = model / "model.safetensors.index.json"
    config = _json(config_path)
    tc = config.get("text_config")
    _check(config.get("model_type") == "qwen3_5_moe" and isinstance(tc, dict)
           and tc.get("model_type") == "qwen3_5_moe_text"
           and tc.get("num_hidden_layers") == 40 and tc.get("num_experts") == 256
           and tc.get("num_experts_per_tok") == 8
           and type(tc.get("hidden_size")) is int and tc["hidden_size"] > 0
           and type(tc.get("vocab_size")) is int and tc["vocab_size"] > 0,
           "model is not the original 40-layer 256-expert top-8 Qwen MoE")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        _check((model / name).is_file(), f"model tokenizer artifact missing: {name}")
    index = _json(index_path)
    mapping = index.get("weight_map")
    _check(isinstance(mapping, dict) and REQUIRED_TENSORS <= set(mapping),
           "model shard index lacks observer embedding/norm/head or all routed expert tensors")
    by_file: dict[str, set[str]] = {}
    for key, filename in mapping.items():
        _check(isinstance(key, str) and isinstance(filename, str)
               and filename.endswith(".safetensors") and Path(filename).name == filename
               and "/" not in filename and "\\" not in filename,
               "model index contains invalid tensor name or off-root shard path")
        by_file.setdefault(filename, set()).add(key)
    headers: dict[str, Any] = {}
    for filename, keys in sorted(by_file.items()):
        path = weights / filename
        _check(path.is_file(), f"source checkpoint shard missing: {filename}")
        header = _header(path)
        _check(set(header) == keys, f"safetensors header/index tensor mismatch: {filename}")
        headers.update(header)
    for key in REQUIRED_TENSORS:
        desc = headers[key]
        shape = desc.get("shape")
        _check(desc.get("dtype") == "BF16" and isinstance(shape, list)
               and all(type(x) is int and x > 0 for x in shape),
               f"observer tensor {key} has wrong dtype or shape")
        if key.endswith("mlp.gate.weight") or key.endswith("experts.down_proj") or key.endswith("experts.gate_up_proj"):
            _check(shape[0] == 256, f"observer tensor {key} is not an unpruned 256-expert tensor")
        if key == "lm_head.weight" or key == f"{PREFIX}embed_tokens.weight":
            _check(shape == [tc["vocab_size"], tc["hidden_size"]],
                   f"observer vocabulary tensor {key} does not match model config")
        if key == f"{PREFIX}norm.weight":
            _check(shape == [tc["hidden_size"]], "final norm shape differs from model config")
    return {"config_sha256": _sha(config_path.read_bytes()),
            "index_sha256": _sha(index_path.read_bytes()), "shard_count": len(by_file),
            "indexed_tensors": len(mapping), "hidden_size": tc["hidden_size"],
            "estimated_hidden_state_gib": round(
                2 * tc["hidden_size"] * declared_tokens / (1024 ** 3), 2),
            "source_weight_bytes_not_hashed": sum((weights / name).stat().st_size for name in by_file)}


def admit(*, calibration: Path, model_dir: Path, out: Path,
          weights_dir: Path | None = None,
          allow_existing_out: bool = False) -> dict[str, Any]:
    calib = calibration.resolve(strict=True)
    model = model_dir.resolve(strict=True)
    weights = (weights_dir or model_dir).resolve(strict=True)
    intended = out.resolve()
    _check(calib.is_dir() and model.is_dir() and weights.is_dir(),
           "calibration, model and weight paths must be directories")
    if allow_existing_out:
        # Resume path: the output directory legitimately exists and must be
        # reused. A completed observation is still an absolute refusal.
        _check(intended.is_dir(),
               "resume requires an existing observer output directory")
        _check(not (intended / "observation.json").exists(),
               "resume refused: this observation already completed")
    else:
        _check(not intended.exists(),
               "observer output already exists: refusing overwrite or reuse")
    _check(not any(intended == source or intended.is_relative_to(source)
                   or source.is_relative_to(intended) for source in (calib, model, weights)),
           "observer output overlaps an input directory")
    provenance = _validate_calibration(calib)
    checkpoint = _validate_model(model, weights, provenance["declared_tokens"])
    # This is a SAFE supervised launch TEMPLATE, not the historical direct
    # in-RAM observer. Running reap_observe.py without atomic checkpoint flags
    # would strand the 704-row D0 and cannot satisfy the registered seal gate.
    command = ["python", "-B", "-m", "chowder_batch.reap_observe_supervised", "--launch",
               "--model-dir", str(model), "--calibration", str(calib / "calibration.jsonl"),
               "--out", str(intended)]
    if weights_dir is not None:
        command += ["--weights-dir", str(weights)]
    command += ["--scratch", "<NEW_UNUSED_F_VOLUME_CHECKPOINT_DIR>",
                "--log-dir", "<NEW_UNUSED_C_VOLUME_SUPERVISOR_LOG_DIR>"]
    return {
        "schema": "chowder.reap-v2-observer-admission.v1",
        "status": "INPUTS_ADMITTED_OBSERVATION_NOT_RUN",
        "calibration": provenance,
        "model": {"directory": str(model), "weights_directory": str(weights), **checkpoint},
        "observation_command_not_executed": command,
        "command_requires_operator_actions": (
            "Replace placeholders with unique nonexistent directories; arrange "
            "explicit protected-model-service maintenance, recover host/GPU/commit "
            "headroom and pass the supervisor's three 30s-spaced host samples. "
            "The direct observer command without atomic scratch is NOT a D0 launch."
        ),
        "intended_output": str(intended),
        "bounds": {"full_layers": 40, "sequences": 704,
                   "next_token_nll_position_chunk": 64,
                   "cpu_hidden_memory_estimate_excludes_runtime_and_overhead": True},
        "gate_d0": "NOT_MEASURED: requires the complete 40-layer observer and perplexity <20",
        "source_identity_limit": "Small config/index and shard headers checked. Full BF16 weights "
                                 "are not hashed; pinned revision contents are not attested.",
        "gpu_work_started": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True, help="frozen v2 calibration directory")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="intended new observer output directory")
    args = parser.parse_args(argv)
    try:
        report = admit(calibration=args.calibration, model_dir=args.model_dir,
                       weights_dir=args.weights_dir, out=args.out)
    except (ObserverAdmissionError, OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        print(f"REAP v2 observation preflight refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
