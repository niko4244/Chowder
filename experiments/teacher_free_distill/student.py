"""Student selection and memory planning for the teacher-free pilot.

Candidates are pinned to exact Hugging Face revisions (verified via the HF API
on 2026-09-25). Memory planning delegates to Chowder's production planner
(`chowder.memory.plan_memory`) and hardware detection
(`chowder.hardware.detect_hardware`) — no separate framework, no invented
performance numbers: throughput and wall-clock estimates require a measured
pilot run and are deliberately not fabricated here.

GGUF/quantized inference checkpoints are NOT trainable targets; every catalog
entry points at a native safetensors revision for PEFT training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from chowder.hardware import detect_hardware  # noqa: E402
from chowder.memory import HardwareProfile, WorkloadProfile, plan_memory  # noqa: E402

# LoRA hyperparameters for estimation (mirrors the recipe in recipes/*.json).
LORA_RANK = 16
LORA_TARGET_RATIO = 0.01        # fraction of base params wrapped (empirical LoRA coverage)
SEQ_LEN = 2048
MICRO_BATCH = 4

STUDENTS: dict[str, dict] = {
    "qwen3-1.7b": {
        "role": "primary",
        "repo": "Qwen/Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "license": "apache-2.0",
        "dtype": "bfloat16",
        "bytes_per_param": 2,
        "params_est": 1_720_000_000,
        "hidden_size": 2048,
        "layers": 28,
        "chat_template": "native Qwen3 (verified prefix-consistent masking required)",
        "gguf_only": False,
        "license_note": "API-verified apache-2.0; revision pinned to live sha.",
    },
    "qwen2.5-coder-3b": {
        "role": "control",
        "repo": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "revision": "488639f1ff808d1d3d0ba301aef8c11461451ec5",
        "license": "other (Qwen research/community license; operator acceptance required before download)",
        "dtype": "bfloat16",
        "bytes_per_param": 2,
        "params_est": 3_090_000_000,
        "hidden_size": 2048,
        "layers": 36,
        "chat_template": "native Qwen2.5 (verified prefix-consistent masking required)",
        "gguf_only": False,
        "license_note": "License tag is 'other', not a permissive grant. Download/use only after the operator accepts Qwen's license terms.",
    },
}


def workload_estimate(student: dict) -> WorkloadProfile:
    bytes_per_param = student["bytes_per_param"]
    params = student["params_est"]
    trainable = int(params * LORA_TARGET_RATIO * (1 + LORA_RANK / 64))
    frozen_gb = params * bytes_per_param / 1e9
    # LoRA state: bf16 weights+grads plus fp32 AdamW moments/master (~16 B/param ceiling)
    trainable_gb = trainable * 16 / 1e9
    # Activations: per-token residual+MLP traffic, batched; conservative upper band
    activation_gb = (MICRO_BATCH * SEQ_LEN * student["hidden_size"]
                     * student["layers"] * 12 * bytes_per_param) / 1e9
    optimizer_gb = trainable * 8 / 1e9  # fp32 moments spill estimate
    workspace_gb = 1.0
    return WorkloadProfile(frozen_weights_gb=round(frozen_gb, 3),
                           trainable_gb=round(trainable_gb, 3),
                           activation_gb=round(activation_gb, 3),
                           optimizer_gb=round(optimizer_gb, 3),
                           workspace_gb=workspace_gb)


def select(student_id: str) -> dict:
    student = STUDENTS.get(student_id)
    if student is None:
        raise ValueError(f"unknown student: {student_id}")
    snapshot = detect_hardware()
    pools = sorted((a.memory_gb for a in snapshot.accelerators), reverse=True)
    if not pools:
        raise RuntimeError("no accelerator detected; training placement requires operator planning")
    vram_gb = pools[0]
    hardware = HardwareProfile(
        vram_gb=vram_gb,
        ram_gb=snapshot.ram_gb,
        nvme_gb=snapshot.storage_free_gb,
        pcie_gbps=16.0,        # declared transport, not a measured number
        ram_gbps=25.0,         # declared transport, not a measured number
        nvme_gbps=3.0,         # declared transport, not a measured number
        accelerator_vram_gb=tuple(sorted(pools, reverse=True)),
    )
    workload = workload_estimate(student)
    try:
        plan = plan_memory(hardware, workload)
        fits = True
        bottleneck = plan.bottleneck
        vram_total = plan.vram_total
    except ValueError as exc:
        fits, bottleneck, vram_total = False, str(exc), 0.0
    return {
        "student": student,
        "hardware": snapshot.to_dict(),
        "workload_gb": {k: getattr(workload, k) for k in
                        ("frozen_weights_gb", "trainable_gb", "activation_gb",
                         "optimizer_gb", "workspace_gb")},
        "primary_pool_gb": vram_gb,
        "fits_primary_pool": fits,
        "plan_bottleneck": bottleneck,
        "plan_vram_total_gb": round(vram_total, 3),
        "notes": [
            "memory figures are static estimates; throughput and duration require a measured pilot",
            "control student requires explicit license acceptance before any download",
            "GGUF inference checkpoints are not trainable; recipes load native safetensors",
        ],
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", default="qwen3-1.7b", choices=sorted(STUDENTS))
    args = parser.parse_args()
    print(json.dumps(select(args.student), indent=2))


if __name__ == "__main__":
    main()
