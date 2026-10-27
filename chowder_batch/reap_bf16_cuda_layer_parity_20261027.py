"""Bounded original REAP128_v2 BF16 CPU/CUDA *layer* research probe.

Loads only an original checkpoint layer, never a model server or generated code.
Uses identical fixed synthetic BF16 activations and Hugging Face layer code.
This is NOT full 40-layer/logit parity, calibrated D0, a release gate, or a
comparison of the original Q4/Q8 native GGUF serving configurations.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


SCHEMA = "chowder.reap128-v2-bf16-cpu-cuda-layer-parity-research.v1"
MIN_GPU_FREE_BYTES = 8 * 1024 ** 3
MIN_GPU_HOST_FREE_MIB = 8192
MIN_PHYSICAL_FREE_BYTES = 7 * 1024 ** 3
MIN_COMMIT_FREE_BYTES = 8 * 1024 ** 3
FROZEN_PROOF_PATH = (Path(__file__).resolve().parent.parent / "docs" / "quals"
                     / "reap128-v2-full-source-equivalence-20261025.json")
FROZEN_PROOF_SHA256 = "0746c8c06764f784fa7134ecdc4a18954a761372f865256c9e0120588cfbbc7d"


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_model_metadata(model: Path, layer: int) -> tuple[dict, dict, dict]:
    """Check the exact original pruned expert geometry before opening weights."""
    if type(layer) is not int or not 0 <= layer < 40:
        raise ValueError("layer must be one of the original 40 decoder layers")
    if not model.is_dir() or model.is_symlink():
        raise ValueError("original BF16 checkpoint must be a real directory")
    paths = {name: model / name for name in
             ("config.json", "model.safetensors.index.json", "reap-prune.json")}
    if any(not p.is_file() or p.is_symlink() or p.stat().st_size > 16 * 1024 ** 2
           for p in paths.values()):
        raise ValueError("missing, symlinked or unbounded original model metadata")
    cfg = json.loads(paths["config.json"].read_bytes())["text_config"]
    prune = json.loads(paths["reap-prune.json"].read_bytes())
    index = json.loads(paths["model.safetensors.index.json"].read_bytes())
    if (cfg.get("num_hidden_layers") != 40 or cfg.get("num_experts") != 128
            or cfg.get("num_experts_per_tok") != 8
            or cfg.get("hidden_size") != 2048
            or prune.get("method") != "reap" or prune.get("keep") != 128
            or prune.get("of") != 256):
        raise ValueError("checkpoint is not original REAP128_v2 40/128/top8")
    keys = index.get("weight_map")
    prefix = f"model.language_model.layers.{layer}."
    if (not isinstance(keys, dict) or not any(k.startswith(prefix) for k in keys)
            or any(not isinstance(k, str) or not isinstance(v, str)
                   or not re.fullmatch(r"model-[0-9]{5}-of-00026\.safetensors", v)
                   for k, v in keys.items())):
        raise ValueError("original shard index identity/format missing")
    needed = sorted({v for k, v in keys.items() if k.startswith(prefix)})
    if any(not (model / name).is_file() or (model / name).is_symlink()
           for name in needed):
        raise ValueError("missing or unsafe original layer shard")
    return cfg, {name: digest_file(path) for name, path in paths.items()}, {
        "tensor_keys": len([k for k in keys if k.startswith(prefix)]),
        "read_shard_names": needed,
        "read_shard_sizes": {name: (model / name).stat().st_size for name in needed},
    }


def verify_frozen_layer_ancestry(model: Path, metadata_sha: dict,
                                 inventory: dict) -> dict:
    """Pin source, selection and the FULL bytes of every touched BF16 shard.

    The earlier source-only verification covers all 26 original shards;
    this lightweight research run rehashes only the shards it really reads.
    It does not turn a single-layer diagnostic into a new 26-shard audit.
    """
    if (FROZEN_PROOF_PATH.is_symlink() or not FROZEN_PROOF_PATH.is_file()
            or digest_file(FROZEN_PROOF_PATH) != FROZEN_PROOF_SHA256):
        raise ValueError("git-frozen full BF16 source verification receipt differs")
    proof = json.loads(FROZEN_PROOF_PATH.read_bytes())
    shards = proof.get("candidate_shard_sha256")
    if (proof.get("schema") != "chowder.reap128-v2-pruned-bf16-verification.v1"
            or proof.get("status") != "FULL_SOURCE_TENSOR_EQUIVALENCE_PASS"
            or Path(proof.get("candidate", "")).resolve() != model.resolve()
            or proof.get("shards") != 26
            or not isinstance(shards, dict) or len(shards) != 26
            or proof.get("candidate_index_sha256") != metadata_sha["model.safetensors.index.json"]
            or proof.get("candidate_prune_receipt_sha256") != metadata_sha["reap-prune.json"]):
        raise ValueError("selected layer differs from frozen independently source-verified BF16 model")
    observed = {}
    for name in inventory["read_shard_names"]:
        if name not in shards or not isinstance(shards[name], str):
            raise ValueError("registered original layer shard missing from full verification")
        observed[name] = digest_file(model / name)
        if observed[name] != shards[name]:
            raise ValueError(f"current original layer shard SHA drift: {name}")
    return {"original_frozen_full_source_proof_sha256": FROZEN_PROOF_SHA256,
            "original_source": proof["original_source"],
            "original_selection_sha256": proof["selection_sha256"],
            "original_observation_sha256": proof["original_observation_sha256"],
            "verified_live_selected_shard_sha256": observed,
            "unread_shards_rehashed_this_run": False}


def verify_windows_admission() -> dict:
    if os.name != "nt":
        raise ValueError("original local BF16 diagnostic is Windows-only")

    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("free_physical", ctypes.c_ulonglong),
                    ("total_pagefile", ctypes.c_ulonglong),
                    ("free_commit", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("free_virtual", ctypes.c_ulonglong),
                    ("free_extended", ctypes.c_ulonglong)]

    entry = Memory()
    entry.length = ctypes.sizeof(entry)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(entry)):
        raise RuntimeError("Windows host RAM/commit census unavailable")
    if (entry.free_physical < MIN_PHYSICAL_FREE_BYTES
            or entry.free_commit < MIN_COMMIT_FREE_BYTES):
        raise RuntimeError("BF16 layer probe refused by physical RAM/commit safety floors")
    return {"free_physical_bytes": entry.free_physical,
            "free_commit_bytes": entry.free_commit}


def parse_physical_gpu_census(raw: str, expected_name: str) -> dict:
    """Reject WDDM virtual allocations as a substitute for physical GPU RAM."""
    try:
        rows = list(csv.reader(raw.splitlines(), skipinitialspace=True))
        if len(rows) != 1 or len(rows[0]) != 6:
            raise ValueError("ambiguous GPU count")
        index, uuid, name, total, used, free = [v.strip() for v in rows[0]]
        total, used, free = int(total), int(used), int(free)
    except (ValueError, TypeError) as exc:
        raise ValueError("malformed native GPU memory census") from exc
    if (index != "0" or not re.fullmatch(r"GPU-[a-zA-Z0-9-]{8,64}", uuid)
            or name != expected_name or total < 16000 or min(used, free) < 0
            or used + free > total + 32 or free < MIN_GPU_HOST_FREE_MIB):
        raise RuntimeError("physical GPU identity or 8-GiB free VRAM admission failed")
    return {"index": 0, "uuid": uuid, "name": name,
            "total_mib": total, "used_mib": used, "free_mib": free}


def physical_gpu_census(expected_name: str) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "0"):
        raise RuntimeError("remapped CUDA GPU is not admitted for this original BF16 probe")
    result = subprocess.run(
        ["nvidia-smi", "--id=0",
         "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
         "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True, timeout=10,
    )
    if result.returncode:
        raise RuntimeError("native physical NVIDIA VRAM measurement unavailable")
    return parse_physical_gpu_census(result.stdout, expected_name)


def compare_finite(a, b) -> dict:
    import torch

    if (not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)
            or a.shape != b.shape or a.dtype != torch.bfloat16
            or b.dtype != torch.bfloat16 or a.numel() == 0):
        raise ValueError("BF16 parity outputs must have identical nonempty tensor shapes")
    left, right = a.detach().float().cpu(), b.detach().float().cpu()
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise ValueError("BF16 layer emitted nonfinite output")
    diff = (left - right).abs()
    return {"elements": left.numel(), "exact_equal_elements": int((left == right).sum()),
            "max_abs_error": float(diff.max()), "mean_abs_error": float(diff.mean()),
            "rmse": float(torch.sqrt((diff * diff).mean())),
            "max_relative_error_eps1e-3": float((diff / left.abs().clamp_min(1e-3)).max())}


def run(model: Path, layer_id: int) -> dict:
    import torch
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

    from chowder_batch.reap_observe import PREFIX, ShardReader, _load_layer_without_duplicate_weights

    cfg, metadata_sha, inventory = verify_model_metadata(model, layer_id)
    host = verify_windows_admission()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("independent local CUDA BF16 capability unavailable")
    device_name = torch.cuda.get_device_name(0)
    native_gpu_pre = physical_gpu_census(device_name)
    before_free, total = torch.cuda.mem_get_info(0)
    if before_free < MIN_GPU_FREE_BYTES:
        raise RuntimeError("BF16 layer probe refused by 8-GiB free VRAM reserve")
    original_ancestry = verify_frozen_layer_ancestry(model, metadata_sha, inventory)
    config = Qwen3_5MoeTextConfig(**cfg)
    config._attn_implementation = "sdpa"
    # Same BF16 input on each device; never draw random or model text samples.
    x = (torch.arange(2 * cfg["hidden_size"], dtype=torch.float32)
         .mul(0.03125).sin().mul(0.5).to(torch.bfloat16)
         .reshape(1, 2, cfg["hidden_size"]))
    position_ids = torch.arange(2, dtype=torch.long).view(1, -1)
    source = ShardReader(model).load(PREFIX + f"layers.{layer_id}.", "cpu")
    if any(t.dtype != torch.bfloat16 for t in source.values() if t.is_floating_point()):
        raise ValueError("original pruned layer has non-BF16 floating weights")
    # Bind loaded original layer tensor bytes, not merely an untrusted receipt.
    tensor_hashes = {}
    for name in sorted(source):
        tensor = source[name].detach().contiguous().cpu()
        tensor_hashes[name] = hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
    stage_sha = hashlib.sha256(json.dumps(tensor_hashes, sort_keys=True).encode()).hexdigest()
    torch.set_num_threads(2)

    def forward(device):
        state = {k: v.to(device) for k, v in source.items()}
        module = _load_layer_without_duplicate_weights(
            M.Qwen3_5MoeDecoderLayer, config, layer_id, torch.bfloat16, state,
        ).eval()
        states, pos = x.to(device), position_ids.to(device)
        rotary = M.Qwen3_5MoeTextRotaryEmbedding(config).to(device)
        rotations = rotary(states, pos[None].expand(3, 1, -1))
        if module.block_type == "full_attention":
            mask = M.create_causal_mask(config=config, inputs_embeds=states,
                                        attention_mask=None, past_key_values=None,
                                        position_ids=pos)
        else:
            mask = M.create_recurrent_attention_mask(config=config,
                                                      inputs_embeds=states, attention_mask=None)
        with torch.inference_mode():
            output = module(states, position_embeddings=rotations,
                            attention_mask=mask, position_ids=pos).detach().cpu()
        del module, state, states, rotary, rotations, mask
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        return output

    cpu_output = forward("cpu")
    gpu_output = forward("cuda")
    stats = compare_finite(cpu_output, gpu_output)
    after_free, _ = torch.cuda.mem_get_info(0)
    native_gpu_post = physical_gpu_census(device_name)
    host_after = verify_windows_admission()
    return {"schema": SCHEMA, "status": "SOURCE_BOUND_SINGLE_LAYER_BF16_PARITY_RESEARCH_ONLY",
            "original_model": str(model.resolve()), "original_layer": layer_id,
            "layer_type": cfg["layer_types"][layer_id], "num_layers_in_original": 40,
            "input_shape": [1, 2, cfg["hidden_size"]],
            "input_dtype": "torch.bfloat16", "fixed_input": "sin(arange*0.03125)*0.5",
            "checkpoint_metadata_sha256": metadata_sha,
            "source_verification": original_ancestry,
            "original_layer_inventory": inventory, "loaded_layer_tensor_sha256": stage_sha,
            "tensor_count_hashed": len(tensor_hashes), "parity": stats,
            "gpu_device": device_name,
            "physical_gpu_pre": native_gpu_pre, "physical_gpu_post": native_gpu_post,
            "wddm_cuda_free_not_used_as_host_gpu_allocation_evidence": True,
            "cuda_memory_pre_free_bytes": before_free,
            "cuda_memory_post_free_bytes": after_free,
            "cuda_memory_total_bytes": total,
            "host_pre": host, "host_post": host_after,
            "original_windows_native_gguf_gate_eligible": False,
            "entire_40_layer_logit_parity_proven": False,
            "quality_gate": "UNKNOWN", "gen0_status": "BLOCKED",
            "chosen": None, "release_approved": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.suffix.lower() != ".json" or args.out.exists() or args.out.is_symlink() or not args.out.parent.is_dir():
            raise ValueError("new exclusive JSON receipt path required")
        result = run(args.model, args.layer)
        with args.out.open("x", encoding="utf8", newline="\n") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"status": result["status"], "layer": args.layer,
                          "parity": result["parity"], "receipt": str(args.out)}))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, KeyError, TypeError) as exc:
        print(f"BF16 CPU/CUDA layer parity refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
