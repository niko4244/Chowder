from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .backends.transformers_peft import TransformersPeftExecutor
from .cycle import ExperimentCycleRunner, GenerationOutcome
from .engine import EvolutionEngine
from .evaluators.transformers_text import TransformersTextEvaluator
from .executors import ExecutionContext
from .hardware import HardwareSnapshot, detect_hardware
from .memory import HardwareProfile
from .project import ProjectSpec, load_project
from .registry import RunRegistry


@dataclass(frozen=True)
class ProjectRunEvent:
    stage: str
    message: str


@dataclass(frozen=True)
class ProjectRunOutcome:
    project: ProjectSpec
    hardware: HardwareSnapshot
    generation: GenerationOutcome

    @property
    def succeeded(self) -> bool:
        return any(candidate.succeeded for candidate in self.generation.candidates)

    @property
    def promoted_experiment_id(self) -> str | None:
        promoted = self.generation.promoted
        return promoted.experiment_id if promoted is not None else None


EventCallback = Callable[[ProjectRunEvent], None]


def _emit(callback: EventCallback | None, stage: str, message: str) -> None:
    if callback is not None:
        callback(ProjectRunEvent(stage=stage, message=message))


def hardware_profile_from_snapshot(snapshot: HardwareSnapshot) -> HardwareProfile:
    """Convert inventory to a conservative execution-context profile.

    Inventory does not measure PCIe/RAM/NVMe throughput, so those bandwidths are
    recorded as 0 (unknown), never invented. GPU pools remain discrete.
    """

    pools = tuple(float(accelerator.memory_gb) for accelerator in snapshot.accelerators)
    contiguous = max(pools, default=0.0)
    reserve = min(1.0, contiguous) if contiguous > 0 else 0.0
    return HardwareProfile(
        vram_gb=contiguous,
        ram_gb=float(snapshot.ram_gb),
        nvme_gb=float(snapshot.storage_free_gb),
        pcie_gbps=0.0,
        ram_gbps=0.0,
        nvme_gbps=0.0,
        reserve_vram_gb=reserve,
        accelerator_vram_gb=pools,
    )


def run_project(
    project_or_path: ProjectSpec | str | Path,
    *,
    on_event: EventCallback | None = None,
) -> ProjectRunOutcome:
    """Execute one real train → evaluate → gate project generation."""

    if isinstance(project_or_path, ProjectSpec):
        project = project_or_path
        project.validate_files()
    else:
        project = load_project(project_or_path, validate_files=True)

    _emit(on_event, "prepare", f"Loaded project {project.name!r}")
    hardware = detect_hardware(project.work_dir)
    profile = hardware_profile_from_snapshot(hardware)
    if hardware.accelerators:
        descriptions = ", ".join(
            f"{accelerator.name} ({accelerator.memory_gb:.1f} GB)"
            for accelerator in hardware.accelerators
        )
        _emit(on_event, "hardware", f"Detected accelerators: {descriptions}")
    else:
        _emit(on_event, "hardware", "No NVIDIA accelerator detected; using CPU-compatible path")

    engine = EvolutionEngine(
        goal=project.goal,
        baseline=project.baseline,
        spent_gpu_hours=project.baseline.gpu_hours,
    )
    trainer = TransformersPeftExecutor()
    evaluator = TransformersTextEvaluator()
    context = ExecutionContext(
        hardware=profile,
        work_dir=str(project.work_dir),
        seed=project.seed,
    )

    with RunRegistry(project.registry_path) as registry:
        runner = ExperimentCycleRunner(
            engine=engine,
            trainer=trainer,
            evaluator=evaluator,
            context=context,
            base_config=project.config,
            registry=registry,
        )
        accepted = engine.propose((project.experiment,))
        if not accepted:
            raise RuntimeError(
                "initial experiment does not fit the configured GPU-hour budget"
            )
        registry.record_experiment(project.experiment)
        _emit(
            on_event,
            "train",
            f"Starting {project.experiment.experiment_id} with {engine.reservation_for(project.experiment.experiment_id):.4g} reserved GPU-hours",
        )
        generation = runner.run_generation(accepted)

    candidate = generation.candidates[0]
    if candidate.error is not None:
        _emit(on_event, "failed", candidate.error)
    elif candidate.result is not None:
        metrics = ", ".join(
            f"{name}={value:.4f}" for name, value in sorted(candidate.result.metrics.items())
        )
        _emit(on_event, "evaluate", f"Evaluation complete: {metrics}")
        if generation.promoted is not None:
            _emit(on_event, "promoted", f"Promoted {generation.promoted.experiment_id}")
        else:
            _emit(on_event, "rejected", "Candidate completed but did not pass the promotion gate")

    return ProjectRunOutcome(
        project=project,
        hardware=hardware,
        generation=generation,
    )
