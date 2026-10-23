"""Dependency-light, CPU-safe Kaggle BF16 compute admission receipt.

No model downloads, tensor allocations, training, GPU use, or gate decisions.
When deliberately scheduled on an accelerator, it interrogates native dtype
support and records a HARDWARE candidate, not a qualified model experiment.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

GIB = 1024**3
ORIGINAL_BF16_CHECKPOINT_BYTES = 71_903_776_776
FROZEN_CALIBRATION_DECLARED_TOKENS = 1_192_636
FROZEN_CALIBRATION_ROWS = 704
FROZEN_HIDDEN_GIB = 4.55


def _disk(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    usage = shutil.disk_usage(path)
    return {"path": str(path), "exists": True,
            "total_gib": round(usage.total / GIB, 3),
            "free_gib": round(usage.free / GIB, 3)}


def _memory() -> dict[str, Any]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return {"total_gib": None, "available_gib": None}
    lines = [line.split(":", 1) for line in meminfo.read_text().splitlines() if ":" in line]
    values = {name: int(raw.strip().split()[0]) * 1024 for name, raw in lines}
    return {"total_gib": round(values["MemTotal"] / GIB, 3),
            "available_gib": round(values["MemAvailable"] / GIB, 3)}


def _gpu() -> dict[str, Any]:
    # Torch is imported only when the accelerator probe is deliberately asked
    # to initialize a GPU. The CPU notebook must not import/init CUDA.
    try:
        import torch
    except ImportError:
        return {"available": False, "error": "torch unavailable"}
    devices = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        devices.append({
            "name": props.name,
            "sm": [props.major, props.minor],
            "memory_gib": round(props.total_memory / GIB, 3),
            "native_bf16": bool(props.major >= 8 and torch.cuda.is_bf16_supported()),
        })
    return {"available": bool(devices), "devices": devices}


def _tpu() -> dict[str, Any]:
    # TPU runtime is deliberately queried only when requested, since JAX/XLA
    # may compile and allocate runtime resources during backend discovery.
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        return {"available": False, "error": "jax unavailable"}
    try:
        devices = jax.devices()
    except (RuntimeError, ValueError, OSError) as exc:
        return {"available": False, "error": f"TPU device discovery: {type(exc).__name__}"}
    available = any(device.platform == "tpu" for device in devices)
    result = {
        "available": available,
        "devices": [{"platform": d.platform, "device_kind": d.device_kind,
                     "id": int(d.id)} for d in devices],
        "jax_bfloat16_dtype": str(jnp.dtype("bfloat16")),
    }
    if available:
        try:
            # A tiny *actual* TPU BF16 arithmetic proof, not a Qwen decoder
            # parity/throughput claim. No model weights, no large allocation.
            a = jnp.ones((32, 32), dtype=jnp.bfloat16)
            c = (a @ a).block_until_ready()
            result["bf16_matmul_32x32"] = (
                "PASS" if str(c.dtype) == "bfloat16" and
                bool(jnp.isfinite(c).all().item()) and
                float(c[0, 0].item()) == 32.0 else "FAIL"
            )
        except (RuntimeError, ValueError, OSError) as exc:
            result["bf16_matmul_32x32"] = f"ERROR:{type(exc).__name__}"
    return result


def classify(accelerator: str, details: dict[str, Any], *, model_staged: bool,
             calibration_staged: bool) -> dict[str, Any]:
    """Keep hardware support distinct from original-parent scientific readiness."""
    if accelerator == "cpu":
        hardware = "CPU_ONLY_NOT_REGISTERED_BF16_ACCELERATOR"
    elif accelerator == "gpu":
        devices = details.get("devices", [])
        if not devices or not all(d.get("native_bf16") is True for d in devices):
            hardware = "GPU_REJECTED_NATIVE_BF16_UNAVAILABLE"
        else:
            hardware = "GPU_NATIVE_BF16_CANDIDATE_ONLY"
    elif accelerator == "tpu":
        hardware = ("TPU_NATIVE_BF16_CANDIDATE_ONLY"
                    if details.get("available") is True and
                    details.get("jax_bfloat16_dtype") == "bfloat16" and
                    details.get("bf16_matmul_32x32") == "PASS"
                    else "TPU_REJECTED_RUNTIME_NOT_VERIFIED")
    else:
        raise ValueError("accelerator must be cpu, gpu or tpu")
    blocked = []
    if not model_staged:
        blocked.append("ORIGINAL_BF16_MODEL_NOT_STAGED_OR_IDENTITY_NOT_CHECKED")
    if not calibration_staged:
        blocked.append("REGISTERED_704_ROW_CALIBRATION_NOT_STAGED_OR_IDENTITY_NOT_CHECKED")
    if accelerator == "tpu":
        blocked.append("QWEN3_5_MOE_LAYER_EXACT_PYTORCH_TO_TPU_PARITY_NOT_PROVEN")
    blocked.append("FULL_FROZEN_704_ROW_D0_NOT_RUN")
    return {"hardware": hardware, "unmet_scientific_requirements": blocked,
            "qualified_to_run_registered_observer": False}


def probe(accelerator: str) -> dict[str, Any]:
    """Emit a bounded, independent Kaggle environment record."""
    info: dict[str, Any] = {
        "schema": "chowder.kaggle-secondary-hardware-probe.v1",
        "status": "HARDWARE_ONLY_NOT_D0",
        "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "accelerator_requested": accelerator,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "host_memory": _memory(),
        "disks": [_disk(Path(path)) for path in
                  ("/kaggle/working", "/kaggle/temp", "/tmp", "/kaggle/input")],
        "expected_original_checkpoint_bytes": ORIGINAL_BF16_CHECKPOINT_BYTES,
        "registered_v2": {"rows": FROZEN_CALIBRATION_ROWS,
                          "declared_tokens": FROZEN_CALIBRATION_DECLARED_TOKENS,
                          "bf16_hidden_gib": FROZEN_HIDDEN_GIB},
        "frozen_source_hashes_verified": False,
        "model_loaded": False,
        "weights_downloaded": False,
        "calibration_loaded": False,
        "gate_d0": "NOT_MEASURED",
        "trained_adapters": 0,
    }
    info["accelerator"] = {
        "cpu": lambda: {"available": True, "native_bf16_accelerator": False},
        "gpu": _gpu, "tpu": _tpu,
    }[accelerator]()
    # No presence-only filesystem check may be promoted to an identity check:
    # an arbitrary input file could masquerade as the original model.
    info["admission"] = classify(accelerator, info["accelerator"],
                                 model_staged=False, calibration_staged=False)
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accelerator", choices=("cpu", "gpu", "tpu"),
                        default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = probe(args.accelerator)
    if not args.out.parent.is_dir() or args.out.exists():
        parser.error("output parent must exist and receipt path must not already exist")
    # Exclusive creation: never replace a scientific or hardware receipt.
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"],
                      "admission": result["admission"],
                      "receipt_path": str(args.out),
                      "receipt_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
