from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .calibration import calibrate_hardware
from .hardware import detect_hardware
from .memory import HardwareProfile, WorkloadProfile, plan_memory
from .project import load_project
from .project_runner import run_project
from .run_events import RunEventPayload, format_event


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


def _hardware_calibrate(args: argparse.Namespace) -> int:
    calibration = calibrate_hardware(
        args.path,
        sample_mib=args.sample_mib,
        passes=args.passes,
        include_cuda=not args.no_cuda,
    )
    print(json.dumps(calibration.to_dict(), indent=2))
    return 0


def _project_validate(args: argparse.Namespace) -> int:
    project = load_project(args.project, validate_files=True)
    print(
        json.dumps(
            {
                "ok": True,
                "name": project.name,
                "work_dir": str(project.work_dir),
                "registry_path": str(project.registry_path),
                "experiment_id": project.experiment.experiment_id,
                "goal_metrics": [target.name for target in project.goal.metrics],
                "gpu_hour_budget": project.goal.gpu_hour_budget,
            },
            indent=2,
        )
    )
    return 0


def _train(args: argparse.Namespace) -> int:
    def event_sink(event: RunEventPayload) -> None:
        print(format_event(event), flush=True)

    outcome = run_project(args.project, on_event=event_sink)
    candidate = outcome.generation.candidates[0]
    summary = {
        "project": outcome.project.name,
        "experiment_id": candidate.experiment_id,
        "succeeded": candidate.succeeded,
        "promoted_experiment_id": outcome.promoted_experiment_id,
        "artifact_ref": candidate.artifact.artifact_ref if candidate.artifact else None,
        "metrics": dict(candidate.result.metrics) if candidate.result else None,
        "gpu_hours": candidate.result.gpu_hours if candidate.result else None,
        "error": candidate.error,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if candidate.succeeded else 1


def _tui(args: argparse.Namespace) -> int:
    from .tui import run_tui

    run_tui(args.project)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chowder",
        description="Chowder autonomous post-training engine",
    )
    sub = parser.add_subparsers(dest="command")

    tui = sub.add_parser("tui", help="Open the guided training setup interface")
    tui.add_argument(
        "--project",
        default="chowder-project.json",
        help="Project JSON path to create/use",
    )
    tui.set_defaults(func=_tui)

    train = sub.add_parser("train", help="Run a saved Chowder project headlessly")
    train.add_argument("project", help="Path to a Chowder project JSON file")
    train.set_defaults(func=_train)

    validate = sub.add_parser("project-validate", help="Validate a project without training")
    validate.add_argument("project", help="Path to a Chowder project JSON file")
    validate.set_defaults(func=_project_validate)

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
    hardware.add_argument(
        "--path",
        default=".",
        help="Filesystem path whose storage tier should be measured",
    )
    hardware.set_defaults(func=_hardware_detect)

    calibrate = sub.add_parser(
        "hardware-calibrate",
        help="Measure local storage, host-copy, and optional CUDA transfer throughput",
    )
    calibrate.add_argument("--path", default=".", help="Filesystem path to calibrate")
    calibrate.add_argument("--sample-mib", type=int, default=64)
    calibrate.add_argument("--passes", type=int, default=3)
    calibrate.add_argument(
        "--no-cuda",
        action="store_true",
        help="Skip optional CUDA transfer measurement",
    )
    calibrate.set_defaults(func=_hardware_calibrate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        # The default user experience is the guided TUI; automation can use an
        # explicit subcommand such as `chowder train project.json`.
        args = parser.parse_args(["tui"])
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
