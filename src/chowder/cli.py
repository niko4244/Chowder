from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .hardware import detect_hardware
from .memory import HardwareProfile, WorkloadProfile, plan_memory


def _memory_plan(args: argparse.Namespace) -> int:
    hardware = HardwareProfile(
        vram_gb=args.vram,
        ram_gb=args.ram,
        nvme_gb=args.nvme,
        pcie_gbps=args.pcie,
        ram_gbps=args.ram_bw,
        nvme_gbps=args.nvme_bw,
    )
    workload = WorkloadProfile(
        frozen_weights_gb=args.weights,
        trainable_gb=args.trainable,
        activation_gb=args.activations,
        optimizer_gb=args.optimizer,
        workspace_gb=args.workspace,
    )
    print(json.dumps(asdict(plan_memory(hardware, workload)), indent=2))
    return 0


def _hardware_detect(args: argparse.Namespace) -> int:
    print(json.dumps(detect_hardware(args.path).to_dict(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chowder", description="Chowder autonomous post-training engine")
    sub = parser.add_subparsers(dest="command", required=True)

    memory = sub.add_parser("memory-plan", help="Plan tensor residency across VRAM/RAM/NVMe")
    memory.add_argument("--vram", type=float, required=True)
    memory.add_argument("--ram", type=float, required=True)
    memory.add_argument("--nvme", type=float, required=True)
    memory.add_argument("--pcie", type=float, default=24.0)
    memory.add_argument("--ram-bw", type=float, default=50.0)
    memory.add_argument("--nvme-bw", type=float, default=5.0)
    memory.add_argument("--weights", type=float, required=True)
    memory.add_argument("--trainable", type=float, required=True)
    memory.add_argument("--activations", type=float, required=True)
    memory.add_argument("--optimizer", type=float, required=True)
    memory.add_argument("--workspace", type=float, required=True)
    memory.set_defaults(func=_memory_plan)

    hardware = sub.add_parser("hardware-detect", help="Inventory local hardware without ML dependencies")
    hardware.add_argument("--path", default=".", help="Filesystem path whose storage tier should be measured")
    hardware.set_defaults(func=_hardware_detect)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
