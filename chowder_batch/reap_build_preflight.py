"""Read-only, frozen REAP fallback *build* preflight (not a selection gate).

Validate the exact observation/weight metadata that ``reap_prune.py`` needs
before any checkpoint shard is opened. Print a deterministic JSON command plan;
never execute the command, create directories, or write ``--out``.

The resulting plan is NOT permission to build an out-of-order fallback: the
frozen selection order and all post-build quality/fit/training gates remain in
``reap_select.py`` and the lineage preregistration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

try:
    from chowder_batch.reap_gates import GateError, _d0
    from chowder_batch.reap_select import _validate_v2_observation
    from chowder_batch.reap_v2_source_identity import (
        V2SourceIdentityError, frozen_v2_source_identity,
    )
except ModuleNotFoundError as exc:
    if exc.name != "chowder_batch":
        raise
    # `python -B chowder_batch/reap_build_preflight.py` puts this script's
    # directory (not its repository parent) on sys.path. Match reap_select's
    # direct-script imports while keeping package imports when run with -m.
    from reap_gates import GateError, _d0
    from reap_select import _validate_v2_observation
    from reap_v2_source_identity import V2SourceIdentityError, frozen_v2_source_identity

ARM_SPEC = {
    "reap128_v2": (128, "v2", ("reap128",)),
    "reap160": (160, "v1", ("reap128", "reap128_v2")),
    "reap160_v2": (160, "v2", ("reap128", "reap128_v2", "reap160")),
}
LAYERS = 40
EXPERTS = 256
TOP_K = 8
FROZEN_V1_SHA256 = "d43c1f8d49d8d7c70872367ba8a56e7052a3dd49351463eed292fd122a631725"
# The real observer writes four 256-element double vectors (~11 KiB/pt).
# This limit prevents an adversarial or wrong checkpoint from becoming an
# unbounded CPU load through torch.load even with weights_only=True.
MAX_STATS_BYTES = 1024 * 1024
STATS_KEYS = ("count", "gate_sum", "norm_sum", "reap_sum")
EXPERT_SUFFIXES = ("experts.gate_up_proj", "experts.down_proj", "gate.weight")


class PreflightError(ValueError):
    """The intended REAP build lacks a required frozen prerequisite."""


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise PreflightError(reason)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise PreflightError(f"missing/unreadable JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def _digest(path: Path) -> str:
    # Metadata and observer stats are small; model shards are NEVER hashed.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _existing_dir(path: Path, label: str) -> Path:
    _require(path.is_dir(), f"{label} directory missing: {path}")
    return path.resolve()


def _model_metadata(model: Path, weights: Path) -> dict[str, str]:
    config_path = model / "config.json"
    index_path = model / "model.safetensors.index.json"
    config = _json(config_path)
    tc = config.get("text_config")
    _require(
        config.get("model_type") == "qwen3_5_moe"
        and isinstance(tc, dict) and tc.get("model_type") == "qwen3_5_moe_text"
        and type(tc.get("num_hidden_layers")) is int and tc["num_hidden_layers"] == LAYERS
        and type(tc.get("num_experts")) is int and tc["num_experts"] == EXPERTS
        and type(tc.get("num_experts_per_tok")) is int and tc["num_experts_per_tok"] == TOP_K,
        "model config is not the original 40-layer / 256-expert / top-8 Qwen MoE; "
        "do not prune an already-pruned checkpoint",
    )
    index = _json(index_path)
    mapping = index.get("weight_map")
    _require(isinstance(mapping, dict) and bool(mapping), "model index has no weight_map")
    # The pruner slices exactly these 120 router/expert tensors. Missing any
    # would silently produce a partly pruned checkpoint that cannot load.
    required = {
        f"model.language_model.layers.{layer}.mlp.{suffix}"
        for layer in range(LAYERS) for suffix in EXPERT_SUFFIXES
    }
    _require(required <= set(mapping), f"model index missing {len(required - set(mapping))} routed expert keys")
    shards: set[str] = set()
    for name in mapping.values():
        _require(
            isinstance(name, str) and name.endswith(".safetensors")
            and name not in {".", ".."} and Path(name).name == name
            and "/" not in name and "\\" not in name,
            "model index contains an invalid/off-root shard filename",
        )
        shards.add(name)
    for shard in sorted(shards):
        _require((weights / shard).is_file(), f"indexed model shard missing: {shard}")
    return {"config_sha256": _digest(config_path), "index_sha256": _digest(index_path),
            "shard_count": len(shards)}


def _calibration_provenance(calibration: Path, observation: Path, mix: str) -> dict[str, Any]:
    manifest = _json(calibration / "calibration-manifest.json")
    rows_path = calibration / "calibration.jsonl"
    _require(rows_path.is_file(), "calibration.jsonl missing")
    observed = _json(observation / "observation.json")
    sha = _digest(rows_path)
    _require(
        manifest.get("calibration_sha256") == sha
        and observed.get("calibration_sha256") == sha,
        "calibration SHA must match both manifest and full observer",
    )
    if mix == "v2":
        try:
            _validate_v2_observation(calibration, observation / "observation.json")
        except (GateError, OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
            raise PreflightError(f"v2 full calibration/Ornith census invalid: {exc}") from exc
        _require(manifest.get("mix") == "v2", "v2 arm requires frozen v2 calibration")
        try:
            identity = frozen_v2_source_identity(calibration, manifest)
        except V2SourceIdentityError as exc:
            raise PreflightError(str(exc)) from exc
    else:
        _require(manifest.get("mix") in (None, "v1"), "reap160 must use v1, never v2")
        _require("ornith" not in manifest, "v1 calibration must not declare Ornith")
        rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        _require(
            type(manifest.get("rows")) is int and manifest["rows"] == len(rows) == 384,
            "v1 requires all 384 frozen calibration rows",
        )
        _require(
            type(observed.get("n_seqs")) is int and observed["n_seqs"] == len(rows)
            and type(observed.get("n_tokens")) is int and observed["n_tokens"] >= len(rows),
            "v1 observer did not process every calibration sequence",
        )
        _require(
            type(manifest.get("tokens")) is int and manifest["tokens"] == 651861,
            "v1 calibration token census differs from frozen 651861",
        )
        # This is the complete v1 preregistered calibration SHA, not an
        # arbitrary self-consistent calibration. v2 allows an Ornith slot.
        _require(
            sha == FROZEN_V1_SHA256,
            "v1 calibration differs from preregistered SHA256",
        )

    d0 = _d0(observation / "observation.json")
    _require(d0["status"] == "PASS", f"D0 is {d0['status']}: {d0.get('reason')}")
    layers = observed.get("layers")
    _require(isinstance(layers, list), "D0 layer census absent")
    for index, row in enumerate(layers):
        _require(isinstance(row, dict) and row.get("layer") == index,
                 "observer layers must be in actual 0..39 forward order")
        _require(type(row.get("routed_tokens")) is int and row["routed_tokens"] > 0,
                 f"observer layer {index} missing positive routed-token total")
    return {"calibration_sha256": sha, "observer_sha256": _digest(observation / "observation.json"),
            "observed_sequences": observed["n_seqs"], "calibration_ppl": d0["statistic"],
            **(identity if mix == "v2" else {})}


def _saliency(observation: Path, reported_layers: list[dict[str, Any]]) -> str:
    try:
        import torch
    except ImportError as exc:
        raise PreflightError("CPU PyTorch is required to decode observer .pt statistics") from exc

    collected = hashlib.sha256()
    for layer in range(LAYERS):
        name = f"layer-{layer:02d}.pt"
        path = observation / name
        _require(path.is_file(), f"observer saliency layer missing: {name}")
        _require(path.stat().st_size <= MAX_STATS_BYTES,
                 f"{name}: oversized stats file, refusing unbounded torch.load")
        try:
            values = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, pickle.UnpicklingError) as exc:
            raise PreflightError(f"{name}: invalid weights_only observer stats: {exc}") from exc
        _require(isinstance(values, dict) and all(key in values for key in STATS_KEYS),
                 f"{name}: missing count/gate_sum/norm_sum/reap_sum stats")
        for key in STATS_KEYS:
            value = values[key]
            _require(
                isinstance(value, torch.Tensor) and value.ndim == 1
                and value.numel() == EXPERTS and not value.is_complex()
                and (value.is_floating_point() or key == "count"),
                f"{name}: {key} must be a numeric {EXPERTS}-element vector",
            )
            _require(torch.isfinite(value).all().item() and (value >= 0).all().item(),
                     f"{name}: {key} must be finite and nonnegative")
        counts = values["count"]
        _require(torch.equal(counts, counts.round()), f"{name}: counts must be integral")
        _require((counts > 0).any().item(), f"{name}: no expert was routed")
        _require(int(counts.sum().item()) == reported_layers[layer]["routed_tokens"],
                 f"{name}: expert counts disagree with full observer routed-token total")
        for key in ("gate_sum", "norm_sum", "reap_sum"):
            _require((values[key][counts == 0] == 0).all().item(),
                     f"{name}: unrouted experts carry {key} saliency")
        collected.update(name.encode("ascii"))
        collected.update(bytes.fromhex(_digest(path)))
    return collected.hexdigest()


def build_plan(*, arm: str, calibration: Path, observations: Path,
               model_dir: Path, out: Path, weights_dir: Path | None = None) -> dict[str, Any]:
    _require(arm in ARM_SPEC, "arm must be reap128_v2, reap160, or reap160_v2")
    keep, mix, predecessors = ARM_SPEC[arm]
    model = _existing_dir(Path(model_dir), "model")
    weights = _existing_dir(Path(weights_dir) if weights_dir is not None else model, "weights")
    calibration = _existing_dir(Path(calibration), "calibration")
    observations = _existing_dir(Path(observations), "observations")
    output = Path(out).resolve()
    _require(not output.exists(), "intended output already exists: refuse overwrite")
    _require(output not in (model, weights, calibration, observations)
             and not any(input_dir.is_relative_to(output) or output.is_relative_to(input_dir)
                         for input_dir in (model, weights, calibration, observations)),
             "intended output must not overlap model, weights, calibration, or observations")
    config = _model_metadata(model, weights)
    provenance = _calibration_provenance(calibration, observations, mix)
    observed = _json(observations / "observation.json")
    if mix == "v2" and observed.get("n_seqs") == 704:
        receipt = observed.get("checkpoint_receipt")
        identity = receipt.get("source_identity") if isinstance(receipt, dict) else None
        _require(
            isinstance(identity, dict)
            and identity.get("model_dir") == str(model)
            and identity.get("weights_dir") == str(weights)
            and identity.get("output_dir") == str(observations),
            "registered v2 D0 receipt must bind requested model/weights/observation paths",
        )
    saliency_sha = _saliency(observations, observed["layers"])

    command = ["python", "chowder_batch/reap_prune.py", "--model-dir", str(model)]
    if weights_dir is not None:
        command += ["--weights-dir", str(weights)]
    command += ["--observations", str(observations), "--keep", str(keep),
                "--method", "reap", "--out", str(output)]
    return {
        "schema": "reap-frozen-build-preflight-v1",
        "status": "PREREQUISITES_VALIDATED_NOT_BUILT",
        "arm": arm,
        "frozen_order_predecessors": list(predecessors),
        "required_calibration": mix,
        "keep_experts_per_layer": keep,
        "model_experts_per_layer": EXPERTS,
        "layers": LAYERS,
        "model": {"directory": str(model), "weights_directory": str(weights), **config},
        "calibration": provenance,
        "saliency_layer_digest_sha256": saliency_sha,
        "output_directory_intended_only": str(output),
        "prune_command_not_executed": command,
        "qualification": "NOT_EVALUATED: build preflight does not certify ordered fallback "
                         "selection, quality, GGUF fit/speed, LoRA placement, or owner approval",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=tuple(ARM_SPEC))
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="intended prune output; never created")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(
            arm=args.arm, calibration=args.calibration, observations=args.observations,
            model_dir=args.model_dir, weights_dir=args.weights_dir, out=args.out,
        )
    except (PreflightError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"REAP build preflight refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(plan, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
