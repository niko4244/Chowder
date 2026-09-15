from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .backend_selection import (
    ROUTER_HEALING_ENGINE,
    create_evaluation_executor,
    create_training_executor,
    normalize_training_config_for_executor,
    resolve_training_engine,
)
from .cancellation import CancellationToken
from .cycle import ExperimentCycleRunner, GenerationOutcome
from .engine import EvolutionEngine
from .evaluators.base_text import BaseModelTextEvaluator
from .executors import EvaluationOutcome, ExecutionContext
from .failures import harvest_transformers_text_failures
from .hardware import HardwareSnapshot, detect_hardware
from .local_corpus_provider import LocalCorpusRepairProvider
from .memory import HardwareProfile
from .models import Experiment, ExperimentResult, ExperimentStatus, Hypothesis
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
    registry_audit: tuple[dict[str, object], ...] = ()

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
) -> tuple[ExperimentResult | None, str | None]:
    """Evaluate the untouched base model and persist it as the baseline.

    Runs before any training happens, using the exact same evaluator and
    protocol (suites, precision, quantization, seed) that will later score
    the trained candidate -- so ``Goal.require_protocol_match`` is comparing
    like with like, not the user's guess of where the base model already
    stood against a differently-configured post-training run.

    For a router project running paired arms, this records the baseline ROW
    but defers its measurement to the candidate's own resident-pair
    evaluation (one model load instead of two) and returns None; the
    deferred-baseline provider in run_project completes the row from that
    measurement before the gate adjudicates.
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
    # The baseline must be measured by the *same* scorer that will score the
    # candidate: a router project's untouched base scored by the PEFT text
    # evaluator would be a different measurement on a different protocol, and
    # comparing it to a router payload's holdout loss would be arithmetic on two
    # unrelated numbers.
    if resolve_training_engine(project.config) == ROUTER_HEALING_ENGINE:
        from .backends.router_healing import RouterHealingEvaluator

        evaluator = RouterHealingEvaluator()
        if evaluator.defers_automatic_baseline(
            experiment=project.experiment, context=context
        ):
            # Paired arms: the candidate's own evaluation measures the base
            # in the same resident process. Record the row now, settle it
            # when the paired evidence lands.
            return None, None
        try:
            outcome = evaluator.evaluate_base(
                config=project.config, context=context
            )
        except Exception:
            # The row exists; a measurement that never completed must not
            # strand it in `planned` ("has not run yet") -- `failed` with no
            # result is the honest record for an attempt that produced no
            # scored outcome.
            registry.update_experiment_status("baseline", ExperimentStatus.FAILED.value)
            raise
    else:
        try:
            outcome = BaseModelTextEvaluator().evaluate(config=project.config, context=context)
        except Exception:
            registry.update_experiment_status("baseline", ExperimentStatus.FAILED.value)
            raise
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
    # A measured baseline is a completed measurement, not a gate verdict: the
    # gate's accept/reject lives on the candidate's row. `parent_tournament`
    # already persists its measured base-model rows as `passed`; the automatic
    # baseline follows the same convention so the durable status finally
    # matches the evidence the row carries.
    registry.update_experiment_status("baseline", ExperimentStatus.PASSED.value)
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
            # The project config is the base resolution the backend resolvers
            # read (e.g. the paired-arms decision at start time); the cycle
            # runner replaces this per candidate with the graph-resolved
            # config, so nothing downstream sees a stale view.
            resolved_config=project.config,
        )

        deferred = False
        if project.baseline_mode == "auto":
            baseline, resolved_revision = _run_automatic_baseline(
                project, context, registry, on_event
            )
            deferred = baseline is None
            training_config: Mapping[str, Any] = (
                _config_with_bound_revision(project.config, resolved_revision)
                if resolved_revision
                else project.config
            )
        else:
            assert project.baseline is not None  # enforced by ProjectSpec.__post_init__
            baseline = project.baseline
            training_config = project.config

        training_config = normalize_training_config_for_executor(training_config)
        engine = EvolutionEngine(
            goal=project.goal,
            # A deferred baseline is a real seam, not a placeholder: the
            # engine runs with no baseline at all and refuses every gate-time
            # operation until the paired evaluation's measurement lands.
            baseline=baseline,
            baseline_deferred=deferred,
            # The row's estimate is the provisional spend so budget admission
            # stays conservative; the provider reconciles it with the
            # measured cost.
            spent_gpu_hours=baseline.gpu_hours if baseline is not None else 0.01,
        )
        trainer = create_training_executor(training_config)
        # Same engine key as the trainer, so a router payload can never be handed
        # to the PEFT text evaluator (or the reverse) and fail inside the
        # library instead of at the dispatch seam.
        evaluator = create_evaluation_executor(training_config)
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
            deferred_baseline=(
                _paired_baseline_completer(registry, on_event) if deferred else None
            ),
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

        # Closeout audit: a result stranded on a non-terminal row is the class
        # of durable-evidence disagreement the automatic-baseline settlement
        # fixed for one writer. The audit keeps the class visible instead of
        # trusting every writer to stay correct forever; the finding is both
        # on the outcome for the caller and persisted as a run event so a
        # restart reconstructs the warning from durable history.
        registry_audit = tuple(registry.audit_stranded_results())
        if registry_audit:
            summary = ", ".join(
                f"{finding['experiment_id']} ({finding['status']})"
                for finding in registry_audit
            )
            _emit_stage(
                on_event,
                registry,
                "registry-audit",
                f"{len(registry_audit)} result(s) stranded on non-terminal rows: {summary}",
            )

    return ProjectRunOutcome(
        project=project,
        hardware=hardware,
        generation=generation,
        repair=repair_outcome,
        registry_audit=registry_audit,
    )


def _paired_baseline_completer(
    registry: RunRegistry,
    on_event: EventCallback | None,
) -> Callable[[object], ExperimentResult]:
    """Complete the deferred baseline row from the paired evaluation.

    Built by run_project when the router project deferred its automatic
    baseline to the candidate's resident pair. The runner calls it with the
    scored candidate outcome BEFORE the gate adjudicates; this closure is
    the single writer of the baseline row (the same writer that created it),
    so the stranded-result discipline holds.
    """

    def complete(candidate_outcome) -> ExperimentResult:
        if candidate_outcome is None:
            # No candidate ever scored: the resident pair that should have
            # measured the base never ran to a score. Settle the row failed
            # and tell the runner the measurement does not exist.
            registry.update_experiment_status("baseline", ExperimentStatus.FAILED.value)
            raise RuntimeError(
                "no candidate evaluation scored, so the deferred baseline has no "
                "resident-pair measurement to complete from"
            )
        evidence = candidate_outcome.evaluation.evidence
        base_loss = evidence.get("base_holdout_loss")
        if not isinstance(base_loss, (int, float)) or not math.isfinite(float(base_loss)):
            registry.update_experiment_status("baseline", ExperimentStatus.FAILED.value)
            raise RuntimeError(
                "the paired evaluation carried no measurable base score; the deferred "
                "baseline is unknown, not zero"
            )
        if evidence.get("arm") != "paired":
            registry.update_experiment_status("baseline", ExperimentStatus.FAILED.value)
            raise RuntimeError(
                f"the candidate evaluation reported arm {evidence.get('arm')!r}, not a "
                "resident pair: the deferral decision and the actual evaluation "
                "disagree, so the baseline is not measured"
            )
        gpu_hours = float(candidate_outcome.evaluation.gpu_hours)
        registry.update_experiment_status("baseline", ExperimentStatus.PASSED.value)
        result = ExperimentResult(
            experiment_id="baseline",
            metrics={"holdout_loss": float(base_loss)},
            gpu_hours=gpu_hours,
            artifact_ref=None,
            evidence={
                "baseline_source": "paired-candidate-evaluation",
                "base_holdout_loss": float(base_loss),
                "compute": {
                    "baseline_source": "paired-candidate-evaluation",
                    "total_gpu_hours": gpu_hours,
                    "model_loads": 1,
                },
            },
        )
        registry.record_result(result)
        metrics_summary = f"holdout_loss={float(base_loss):.4f}"
        _emit_stage(
            on_event,
            registry,
            "baseline",
            f"Automatic baseline established from the resident pair: {metrics_summary}",
        )
        return result

    return complete


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
