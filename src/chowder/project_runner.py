from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .backends.transformers_peft import TransformersPeftExecutor
from .cancellation import CancellationToken
from .cycle import ExperimentCycleRunner, GenerationOutcome
from .engine import EvolutionEngine
from .evaluators.base_text import BaseModelTextEvaluator
from .evaluators.transformers_text import TransformersTextEvaluator
from .executors import EvaluationOutcome, ExecutionContext
from .failures import harvest_transformers_text_failures
from .hardware import HardwareSnapshot, detect_hardware
from .local_corpus_provider import LocalCorpusRepairProvider
from .memory import HardwareProfile
from .models import Experiment, ExperimentResult, Hypothesis
from .project import ProjectSpec, load_project
from .recursive_repair import RecursiveRepairOutcome, run_bounded_autonomous_repair
from .registry import RunRegistry
from .run_events import (
    CheckpointEvent,
    FailureEvent,
    PromotionEvent,
    RepairEvent,
    RunEvent,
    RunEventPayload,
)

# Kept as the pre-existing public name for the generic stage/message event,
# since every current caller (CLI, TUI, tests) matches on `.stage` --
# run_project now also emits the more specific event types from run_events
# (RepairEvent, FailureEvent, PromotionEvent, CheckpointEvent) through the
# same callback, so a caller that only understands RunEvent can keep
# ignoring the rest via isinstance/duck typing.
ProjectRunEvent = RunEvent


@dataclass(frozen=True)
class ProjectRunOutcome:
    project: ProjectSpec
    hardware: HardwareSnapshot
    generation: GenerationOutcome
    repair: RecursiveRepairOutcome | None = None

    @property
    def succeeded(self) -> bool:
        return any(candidate.succeeded for candidate in self.generation.candidates)

    @property
    def promoted_experiment_id(self) -> str | None:
        promoted = self.generation.promoted
        return promoted.experiment_id if promoted is not None else None


EventCallback = Callable[[RunEventPayload], None]


def _emit(
    callback: EventCallback | None,
    registry: RunRegistry | None,
    event: RunEventPayload,
) -> None:
    """Every event is durably persisted (when a registry is open) before
    the live callback runs -- the TUI/CLI's on_event is a convenience for
    live display, not the system of record. A caller that restarts after a
    crash can reconstruct run history from registry.list_events() even if
    nothing was watching on_event at the time.
    """
    if registry is not None:
        registry.record_event(event)
    if callback is not None:
        callback(event)


def _emit_stage(
    callback: EventCallback | None,
    registry: RunRegistry | None,
    stage: str,
    message: str,
    *,
    experiment_id: str | None = None,
) -> None:
    _emit(callback, registry, RunEvent(stage=stage, message=message, experiment_id=experiment_id))


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
    _emit_stage(
        on_event, registry, "baseline", "Evaluating the untouched base model for an automatic baseline"
    )
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
    _emit_stage(
        on_event, registry, "baseline", f"Automatic baseline established: {metrics_summary}"
    )
    return result, _resolved_revision_from_outcome(outcome)


def run_project(
    project_or_path: ProjectSpec | str | Path,
    *,
    on_event: EventCallback | None = None,
    cancellation: CancellationToken | None = None,
) -> ProjectRunOutcome:
    """Execute one real (baseline if automatic) → train → evaluate → gate project generation.

    `cancellation`, if given, is checked before each candidate (and each
    autonomous-repair hop) starts, and is bound to the real trainer/evaluator
    while one is in flight so a request can terminate an already-running
    subprocess rather than only preventing the next one. Not consulted
    during automatic baseline evaluation, which runs before any candidate
    and is typically short relative to training.
    """

    if isinstance(project_or_path, ProjectSpec):
        project = project_or_path
        project.validate_files()
    else:
        project = load_project(project_or_path, validate_files=True)

    with RunRegistry(project.registry_path) as registry:
        _emit_stage(on_event, registry, "prepare", f"Loaded project {project.name!r}")
        hardware = detect_hardware(project.work_dir)
        profile = hardware_profile_from_snapshot(hardware)
        if hardware.accelerators:
            descriptions = ", ".join(
                f"{accelerator.name} ({accelerator.memory_gb:.1f} GB)"
                for accelerator in hardware.accelerators
            )
            _emit_stage(on_event, registry, "hardware", f"Detected accelerators: {descriptions}")
        else:
            _emit_stage(
                on_event,
                registry,
                "hardware",
                "No NVIDIA accelerator detected; using CPU-compatible path",
            )

        context = ExecutionContext(
            hardware=profile,
            work_dir=str(project.work_dir),
            seed=project.seed,
        )

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
            failure_harvester=harvest_transformers_text_failures,
            cancellation=cancellation,
            # Deliberately NOT persisted via _emit/registry.record_event:
            # this fires from a background thread polling the training
            # subprocess (see TransformersPeftExecutor._poll_progress),
            # concurrently with the main thread's use of the same registry
            # connection, and sqlite3 connections are only safe on the
            # thread that created them. Live progress ticks are ephemeral
            # display data, not part of the durable history the way stage
            # transitions, repair/failure/promotion events, and checkpoints
            # already are.
            progress_callback=on_event,
        )
        accepted = engine.propose((project.experiment,))
        if not accepted:
            raise RuntimeError(
                "initial experiment does not fit the configured GPU-hour budget"
            )
        registry.record_experiment(project.experiment)
        _emit_stage(
            on_event,
            registry,
            "train",
            f"Starting {project.experiment.experiment_id} with {engine.reservation_for(project.experiment.experiment_id):.4g} reserved GPU-hours",
            experiment_id=project.experiment.experiment_id,
        )
        generation = runner.run_generation(accepted)
        _emit_candidate_events(on_event, registry, generation.candidates[0])

        repair_outcome: RecursiveRepairOutcome | None = None
        if project.repair is not None and generation.promoted is None:
            repair_spec = project.repair
            _emit(
                on_event,
                registry,
                RepairEvent(
                    target_experiment_id=project.experiment.experiment_id,
                    depth=0,
                    failure_signature=None,
                ),
            )
            provider = LocalCorpusRepairProvider(
                repair_spec.corpus_files,
                max_examples=repair_spec.provider_max_examples,
                min_examples=repair_spec.provider_min_examples,
                examples_per_failure=repair_spec.provider_examples_per_failure,
            )
            repair_outcome = run_bounded_autonomous_repair(
                runner=runner,
                source_generation=generation,
                provider=provider,
                variants=repair_spec.variants,
                policy=repair_spec.policy,
            )
            generation = repair_outcome.final_generation
            for hop in repair_outcome.hops:
                _emit_candidate_events(
                    on_event,
                    registry,
                    hop.outcome.repair_generation.candidates[0],
                )
            _emit(
                on_event,
                registry,
                RepairEvent(
                    target_experiment_id=project.experiment.experiment_id,
                    depth=repair_outcome.depth,
                    stop_reason=repair_outcome.stop_reason.value,
                    stop_detail=repair_outcome.stop_detail,
                ),
            )

        candidate = generation.candidates[0]
        if candidate.error is not None:
            _emit_stage(
                on_event, registry, "failed", candidate.error, experiment_id=candidate.experiment_id
            )
        elif candidate.result is not None:
            metrics = ", ".join(
                f"{name}={value:.4f}" for name, value in sorted(candidate.result.metrics.items())
            )
            _emit_stage(
                on_event,
                registry,
                "evaluate",
                f"Evaluation complete: {metrics}",
                experiment_id=candidate.experiment_id,
            )
            if generation.promoted is not None:
                promoted = generation.promoted
                _emit_stage(
                    on_event,
                    registry,
                    "promoted",
                    f"Promoted {promoted.experiment_id}",
                    experiment_id=promoted.experiment_id,
                )
                _emit(
                    on_event,
                    registry,
                    PromotionEvent(
                        experiment_id=promoted.experiment_id, metrics=dict(promoted.metrics)
                    ),
                )
            else:
                _emit_stage(
                    on_event,
                    registry,
                    "rejected",
                    "Candidate completed but did not pass the promotion gate",
                    experiment_id=candidate.experiment_id,
                )

    return ProjectRunOutcome(
        project=project,
        hardware=hardware,
        generation=generation,
        repair=repair_outcome,
    )


def _emit_candidate_events(
    callback: EventCallback | None, registry: RunRegistry | None, candidate
) -> None:
    """CheckpointEvent/FailureEvent for one candidate's real, already-known
    outcome -- not a live progress stream (that needs the worker to report
    intermediate state, which is a separate, later piece of work), just the
    structured facts already available once training/evaluation for this
    candidate has finished.
    """
    if candidate.artifact is not None:
        checkpoint = candidate.artifact.evidence.get("checkpoint")
        if isinstance(checkpoint, Mapping) and checkpoint.get("trainer_dir"):
            _emit(
                callback,
                registry,
                CheckpointEvent(
                    experiment_id=candidate.experiment_id,
                    checkpoint_dir=str(checkpoint["trainer_dir"]),
                    step=None,
                ),
            )
    if candidate.harvested_failures:
        _emit(
            callback,
            registry,
            FailureEvent(
                experiment_id=candidate.experiment_id,
                failure_count=len(candidate.harvested_failures),
                repair_plan_count=len(candidate.repair_plans),
            ),
        )
