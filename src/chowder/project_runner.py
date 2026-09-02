from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .backends.transformers_peft import TransformersPeftExecutor
from .cycle import ExperimentCycleRunner, GenerationOutcome
from .engine import EvolutionEngine
from .evaluators.base_text import BaseModelTextEvaluator
from .evaluators.transformers_text import TransformersTextEvaluator
from .executors import EvaluationOutcome, ExecutionContext
from .hardware import HardwareSnapshot, detect_hardware
from .memory import HardwareProfile
from .models import Experiment, ExperimentResult, Hypothesis
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


def _resolved_revision_from_outcome(outcome: EvaluationOutcome) -> str | None:
    """The exact base-model commit the baseline actually measured, if pinned.

    ``BaseModelTextEvaluator`` resolves a floating ref (e.g. no revision, or
    "main") to the exact commit it loaded and records that under
    ``model_provenance``. Reusing it -- rather than letting training
    independently re-resolve the same floating ref later -- is what makes
    "the untouched model" and "the model training actually started from"
    provably the same snapshot, not merely likely the same one.
    """
    provenance = outcome.evidence.get("model_provenance")
    if isinstance(provenance, Mapping):
        commit = provenance.get("resolved_model_commit")
        if isinstance(commit, str) and commit:
            return commit
    return None


def _config_with_bound_revision(config: Mapping[str, Any], revision: str) -> dict[str, Any]:
    bound = dict(config)
    backend = dict(bound.get("backend", {}))
    backend["revision"] = revision
    bound["backend"] = backend
    return bound


def _run_automatic_baseline(
    project: ProjectSpec,
    context: ExecutionContext,
    registry: RunRegistry,
    on_event: EventCallback | None,
) -> tuple[ExperimentResult, str | None]:
    """Evaluate the untouched base model and persist it as the baseline.

    Runs before any training happens, using the exact same evaluator and
    protocol (suites, precision, quantization, seed) that will later score
    the trained candidate -- so ``Goal.require_protocol_match`` is comparing
    like with like, not the user's guess of where the base model already
    stood against a differently-configured post-training run.
    """
    _emit(on_event, "baseline", "Evaluating the untouched base model for an automatic baseline")
    # evaluation_runs/results both carry FOREIGN KEY(experiment_id) REFERENCES
    # experiments(experiment_id) -- the baseline needs a real row there too,
    # the same as any candidate experiment gets via record_experiment below.
    evaluation_config = project.config.get("evaluation")
    estimated_gpu_hours = 0.01
    if isinstance(evaluation_config, Mapping):
        try:
            estimated_gpu_hours = max(0.01, float(evaluation_config.get("estimated_gpu_hours", 0.01)))
        except (TypeError, ValueError):
            pass
    registry.record_experiment(
        Experiment(
            experiment_id="baseline",
            parent_id=None,
            hypothesis=Hypothesis(
                observation="No prior measurement of this base model on this protocol exists",
                suspected_cause="A baseline has never been established for this project",
                intervention="Evaluate the untouched base model under the configured evaluation protocol",
            ),
            config_patch={},
            estimated_gpu_hours=estimated_gpu_hours,
        )
    )
    outcome = BaseModelTextEvaluator().evaluate(config=project.config, context=context)
    evidence: dict[str, Any] = {
        "evaluation_run_id": outcome.run_id,
        "evaluation": dict(outcome.evidence),
        "compute": {
            "evaluation_gpu_hours": outcome.gpu_hours,
            "total_gpu_hours": outcome.gpu_hours,
        },
    }
    protocol_sha = outcome.evidence.get("protocol_sha256")
    if isinstance(protocol_sha, str) and len(protocol_sha) == 64:
        evidence["evaluation_protocol_sha256"] = protocol_sha
    result = ExperimentResult(
        experiment_id="baseline",
        metrics=dict(outcome.metrics),
        gpu_hours=outcome.gpu_hours,
        artifact_ref=None,
        evidence=evidence,
    )
    registry.record_evaluation_outcome(outcome)
    registry.record_result(result)
    metrics_summary = ", ".join(f"{name}={value:.4f}" for name, value in sorted(result.metrics.items()))
    _emit(on_event, "baseline", f"Automatic baseline established: {metrics_summary}")
    return result, _resolved_revision_from_outcome(outcome)


def run_project(
    project_or_path: ProjectSpec | str | Path,
    *,
    on_event: EventCallback | None = None,
) -> ProjectRunOutcome:
    """Execute one real (baseline if automatic) → train → evaluate → gate project generation."""

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

    context = ExecutionContext(
        hardware=profile,
        work_dir=str(project.work_dir),
        seed=project.seed,
    )

    with RunRegistry(project.registry_path) as registry:
        if project.baseline_mode == "auto":
            baseline, resolved_revision = _run_automatic_baseline(
                project, context, registry, on_event
            )
            training_config: Mapping[str, Any] = (
                _config_with_bound_revision(project.config, resolved_revision)
                if resolved_revision
                else project.config
            )
        else:
            assert project.baseline is not None  # enforced by ProjectSpec.__post_init__
            baseline = project.baseline
            training_config = project.config

        engine = EvolutionEngine(
            goal=project.goal,
            baseline=baseline,
            spent_gpu_hours=baseline.gpu_hours,
        )
        trainer = TransformersPeftExecutor()
        evaluator = TransformersTextEvaluator()
        runner = ExperimentCycleRunner(
            engine=engine,
            trainer=trainer,
            evaluator=evaluator,
            context=context,
            base_config=training_config,
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
