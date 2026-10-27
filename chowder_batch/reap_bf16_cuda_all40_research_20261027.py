"""Owned resumable 40-layer BF16 CPU/CUDA *research-only* probe.

Each layer executes in a new bounded original Python child; source shards,
memory and physical GPU are guarded by the layer runner. Completed layer 0/3
original receipts are reused byte-for-byte, never repeated. Does NOT measure
full-model logits, release D0, or original GGUF native serving parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from chowder_batch.reap_bf16_cuda_layer_parity_20261027 import (
    FROZEN_PROOF_SHA256, SCHEMA as LAYER_SCHEMA, digest_file,
    verify_model_metadata,
)

SCHEMA = "chowder.reap128-v2-bf16-cpu-cuda-all40-layer-research.v1"
OWNER_SCHEMA = "chowder.reap128-v2-owned-all40-layer-research.v1"
MIN_DISK_FREE_BYTES = 8 * 1024 ** 3
ORIGINAL_LAYER_RECEIPTS = {
    0: (
        r"F:\reap-v2-bf16-cuda-layer-parity-20261027-0301"
        r"\original-layer0-source-pinned-physical-gpu-cpu-cuda.json",
        "0a632ba0f87db8604159728e79bc5edb3a299cdc195e3fa6150d6599d053c22b",
    ),
    3: (
        r"F:\reap-v2-bf16-cuda-layer-parity-20261027-0301"
        r"\original-layer3-source-pinned-physical-gpu-cpu-cuda.json",
        "9228f76bb505f95680b6b49163644cec57a1eecbe7bd1d83a35aea6369a95f6e",
    ),
}


def exact_owner(model: Path) -> dict:
    cfg, meta, _ = verify_model_metadata(model, 0)
    return {
        "schema": OWNER_SCHEMA,
        "status": "EXPERIMENT_ONLY_NOT_GATE_ELIGIBLE",
        "original_model": str(model.resolve()),
        "layer_count": cfg["num_hidden_layers"],
        "original_frozen_proof_sha256": FROZEN_PROOF_SHA256,
        "config_sha256": meta["config.json"],
        "index_sha256": meta["model.safetensors.index.json"],
        "prune_sha256": meta["reap-prune.json"],
        "run_layers": list(range(40)),
        "reuse_original_layers": {str(k): sha for k, (_, sha) in ORIGINAL_LAYER_RECEIPTS.items()},
        "full_model_logit_parity_proven": False,
        "gen0_status": "BLOCKED", "release_approved": False,
    }


def _canonical_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _load_strict(raw: bytes) -> dict:
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("repeated private receipt key")
            out[key] = value
        return out

    def no_constant(s):
        raise ValueError(f"invalid nonfinite receipt number: {s}")

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=no_constant)
    if not isinstance(value, dict):
        raise ValueError("private layer receipt must be an object")
    return value


def verify_layer_receipt(raw: bytes, *, model: Path, layer: int) -> dict:
    if not 0 <= layer < 40 or not 0 < len(raw) < 256 * 1024:
        raise ValueError("wrong layer or unbounded layer receipt")
    data = _load_strict(raw)
    frozen = data.get("source_verification", {})
    pre, post = data.get("physical_gpu_pre", {}), data.get("physical_gpu_post", {})
    stats = data.get("parity", {})
    if (data.get("schema") != LAYER_SCHEMA
            or data.get("status") != "SOURCE_BOUND_SINGLE_LAYER_BF16_PARITY_RESEARCH_ONLY"
            or data.get("original_model") != str(model.resolve())
            or type(data.get("original_layer")) is not int
            or data["original_layer"] != layer
            or data.get("num_layers_in_original") != 40
            or data.get("input_shape") != [1, 2, 2048]
            or data.get("input_dtype") != "torch.bfloat16"
            or data.get("fixed_input") != "sin(arange*0.03125)*0.5"
            or data.get("gpu_device") != "NVIDIA GeForce RTX 5060 Ti"
            or data.get("original_windows_native_gguf_gate_eligible") is not False
            or data.get("entire_40_layer_logit_parity_proven") is not False
            or data.get("quality_gate") != "UNKNOWN"
            or data.get("gen0_status") != "BLOCKED"
            or data.get("chosen", "missing") is not None
            or data.get("release_approved") is not False
            or not isinstance(frozen, dict)
            or frozen.get("original_frozen_full_source_proof_sha256") != FROZEN_PROOF_SHA256
            or frozen.get("unread_shards_rehashed_this_run") is not False
            or not isinstance(frozen.get("verified_live_selected_shard_sha256"), dict)
            or not frozen["verified_live_selected_shard_sha256"]
            or not isinstance(stats, dict)
            or type(stats.get("elements")) is not int
            or stats["elements"] != 4096
            or type(stats.get("exact_equal_elements")) is not int
            or not 0 <= stats["exact_equal_elements"] <= 4096):
        raise ValueError("original source or research-only layer receipt contract changed")
    for name in ("max_abs_error", "mean_abs_error", "rmse", "max_relative_error_eps1e-3"):
        value = stats.get(name)
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError(f"nonfinite or negative original {name}")
    for gpu in (pre, post):
        if (not isinstance(gpu, dict) or gpu.get("index") != 0
                or type(gpu.get("free_mib")) is not int or gpu["free_mib"] < 8192
                or gpu.get("name") != "NVIDIA GeForce RTX 5060 Ti"
                or gpu.get("total_mib") != 16311):
            raise ValueError("missing genuine native physical GPU source measurement")
    return {"layer": layer, "layer_type": data.get("layer_type"),
            "source_shards": sorted(frozen["verified_live_selected_shard_sha256"]),
            "exact_equal_elements": stats["exact_equal_elements"],
            "max_abs_error": stats["max_abs_error"],
            "mean_abs_error": stats["mean_abs_error"],
            "rmse": stats["rmse"],
            "host_gpu_free_pre_mib": pre["free_mib"],
            "host_gpu_free_post_mib": post["free_mib"]}


def _receipt(model: Path, layer: int, output: Path) -> tuple[dict, str]:
    if not output.is_file() or output.is_symlink() or output.stat().st_size >= 256 * 1024:
        raise ValueError("missing, unbounded or symlinked original layer receipt")
    raw = output.read_bytes()
    return verify_layer_receipt(raw, model=model, layer=layer), hashlib.sha256(raw).hexdigest()


def run(model: Path, out: Path, *, execute: bool, max_new: int = 40) -> dict:
    model = model.resolve()
    if not 1 <= max_new <= 40:
        raise ValueError("owned execution budget must be between 1 and 40")
    owner = exact_owner(model)
    if (out.is_symlink() or out.resolve() == model
            or out.resolve().is_relative_to(model)
            or model.is_relative_to(out.resolve())
            or not out.parent.is_dir()):
        raise ValueError("separate new/existing private owned output folder required")
    if shutil.disk_usage(out.parent).free < MIN_DISK_FREE_BYTES:
        raise RuntimeError("BF16 parity receipts refuse below 8 GiB disk reserve")
    if not execute:
        return {"status": "ORIGINAL_ALL40_PLAN_NOT_EXECUTED", "source": owner,
                "out": str(out), "no_generated_code": True}
    if not out.exists():
        out.mkdir(exist_ok=False)
    manifest = out / "owner.json"
    if manifest.exists():
        if manifest.is_symlink() or manifest.read_bytes() != _canonical_bytes(owner):
            raise ValueError("resume folder does not belong to same immutable BF16 mission")
    else:
        # If a folder already exists with unrelated content, never claim it.
        if any(out.iterdir()):
            raise ValueError("refusing unowned existing receipt folder")
        with manifest.open("xb") as writer:
            writer.write(_canonical_bytes(owner))
    results, digests, new = [], {}, 0
    for layer in range(40):
        path = out / f"layer-{layer:02}.json"
        if layer in ORIGINAL_LAYER_RECEIPTS:
            original_path, sha = ORIGINAL_LAYER_RECEIPTS[layer]
            original = Path(original_path)
            if path.exists() or original.is_symlink() or not original.is_file():
                raise ValueError(f"refusing duplicate or absent immutable original layer {layer}")
            result, digest = _receipt(model, layer, original)
            if digest != sha:
                raise ValueError(f"pinned original earlier layer {layer} SHA mismatch")
        elif path.exists():
            result, digest = _receipt(model, layer, path)
        else:
            if new >= max_new:
                return {"status": "PARTIAL_SOURCE_BOUND_RESEARCH_NOT_QUALIFIED",
                        "validated_layers": len(results), "new_layers_this_call": new,
                        "next_layer": layer, "gen0_status": "BLOCKED",
                        "release_approved": False}
            # Only our child is owned; explicit no-shell argv and bounded CPU/GPU.
            args = [sys.executable, "-m",
                    "chowder_batch.reap_bf16_cuda_layer_parity_20261027",
                    "--model", str(model), "--layer", str(layer), "--out", str(path)]
            proc = subprocess.run(args, capture_output=True, text=True,
                                  check=False, timeout=240)
            if proc.returncode:
                raise RuntimeError(f"original layer {layer} refused (exit {proc.returncode}); "
                                   "preserve prior receipts and inspect owned log")
            result, digest = _receipt(model, layer, path)
            new += 1
        results.append(result)
        digests[str(layer)] = digest
        print(json.dumps({"event": "SOURCE_BOUND_LAYER_RESEARCH_COMPLETE",
                          "layer": layer, "from_prior_original_receipt": layer in ORIGINAL_LAYER_RECEIPTS,
                          "exact": result["exact_equal_elements"],
                          "max_abs_error": result["max_abs_error"]}), flush=True)
    if len(results) != 40 or sorted(int(k) for k in digests) != list(range(40)):
        raise ValueError("unqualified partial 40-layer source run")
    complete = {
        "schema": SCHEMA, "status": "ALL40_SOURCE_BOUND_LAYER_NUMERICS_RESEARCH_ONLY",
        "model": str(model), "owner_sha256": hashlib.sha256(_canonical_bytes(owner)).hexdigest(),
        "source_full_model_ancestry_sha256": FROZEN_PROOF_SHA256,
        "distinct_original_layer_receipts": len(results),
        "original_prior_layer_receipts_reused": sorted(ORIGINAL_LAYER_RECEIPTS),
        "real_cpu_cuda_bf16_compared_elements": sum(r["exact_equal_elements"] * 0 + 4096 for r in results),
        "exact_equal_elements": sum(r["exact_equal_elements"] for r in results),
        "max_absolute_difference": max(r["max_abs_error"] for r in results),
        "min_physical_gpu_free_mib": min(min(r["host_gpu_free_pre_mib"],
                                           r["host_gpu_free_post_mib"]) for r in results),
        "per_layer": results, "source_bound_private_receipt_sha256": digests,
        "entire_model_forward_or_logit_parity_proven": False,
        "native_gguf_original_four_slot_gate_eligible": False,
        "quality_gate": "UNKNOWN", "gen0_status": "BLOCKED",
        "chosen": None, "release_approved": False,
    }
    target = out / "all40-summary.json"
    if target.exists():
        if target.is_symlink() or target.read_bytes() != _canonical_bytes(complete):
            raise ValueError("existing source-pinned summary differs, refusing overwrite")
    else:
        with target.open("xb") as writer:
            writer.write(_canonical_bytes(complete))
    return {"status": complete["status"], "receipt": str(target),
            "sha256": digest_file(target), "layers": 40,
            "gen0_status": "BLOCKED", "release_approved": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-new", type=int, default=40)
    args = parser.parse_args(argv)
    try:
        result = run(args.model, args.out, execute=args.execute, max_new=args.max_new)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["status"] != "PARTIAL_SOURCE_BOUND_RESEARCH_NOT_QUALIFIED" else 3
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, KeyError) as exc:
        print(f"full original BF16 layer research refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
