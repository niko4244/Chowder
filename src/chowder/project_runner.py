from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from .backend_selection import (
    ROUTER_HEALING_ENGINE,
    create_evaluation_executor,
    create_training_executor,
    normalize_training_config_for_executor,
    resolve_training_engine,
)
from .backends.transformers_peft import TransformersPeftExecutor
from .cancellation import CancellationToken
from .cycle import ExperimentCycleRunner, GenerationOutcome
from .engine import EvolutionEngine
from .evaluators.base_text import BaseModelTextEvaluator
from .evaluators.transformers_text import TransformersTextEvaluator
from .executors import EvaluationOutcome, ExecutionContext
from .failures import harvest_transformers_text_failures
from .goal_lifecycle import GoalLifecycle, GoalLifecycleError
from .improvement.constitution import Constitution, goal_digest
from .hardware import HardwareSnapshot, detect_hardware
from .local_corpus_provider import LocalCorpusRepairProvider
from .memory import HardwareProfile
from .models import Experiment, ExperimentResult, ExperimentStatus, Hypothesis
from .protocol import protocol_fingerprint, result_protocol_fingerprint
from .project import ProjectSpec, load_project
from .provenance import sha256_file
from .recursive_repair import RecursiveRepairOutcome, run_bounded_autonomous_repair
from .registry import (
    GoalObjectiveMigrationApproval as ProtocolContractMigrationApproval,
    GoalObjectiveMigrationProvenance,
    RegistryInvariantError,
    RunRegistry,
)
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
        """Return true only when the frozen lifecycle reports goal completion.

        Training/evaluation success, promotion, and a clean process exit are
        not product success. Bounded stops, refusal, cancellation, crashes,
        and incomplete evidence all remain non-success.
        """
        return self.generation.goal_terminal_state == "STOP_GOALS_MET"

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


def _closeout_registry_audit(
    registry: RunRegistry,
    on_event: EventCallback | None,
) -> tuple[dict[str, object], ...]:
    findings = tuple(registry.audit_stranded_results())
    if findings:
        _emit_stage(
            on_event,
            registry,
            "registry-audit",
            f"Registry audit found {len(findings)} stranded result(s)",
        )
    return findings


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
    if resolve_training_engine(project.config) == ROUTER_HEALING_ENGINE:
        from .backends.router_healing import RouterHealingEvaluator

        evaluator = RouterHealingEvaluator()
        if evaluator.defers_automatic_baseline(
            experiment=project.experiment, context=context
        ):
            return None, None  # type: ignore[return-value]
        try:
            outcome = evaluator.evaluate_base(config=project.config, context=context)
        except Exception:
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
    if not (isinstance(protocol_sha, str) and len(protocol_sha) == 64):
        protocol_sha = outcome.evidence.get("eval_spec_digest")
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
    registry.update_experiment_status("baseline", ExperimentStatus.PASSED.value)
    metrics_summary = ", ".join(f"{name}={value:.4f}" for name, value in sorted(result.metrics.items()))
    _emit_stage(
        on_event, registry, "baseline", f"Automatic baseline established: {metrics_summary}"
    )
    return result, _resolved_revision_from_outcome(outcome)


def _project_benchmark_digest(project: ProjectSpec) -> str:
    """Hash the configured benchmark contract and its current dataset bytes."""
    evaluation = project.config.get("evaluation", {})
    suites = evaluation.get("suites", ()) if isinstance(evaluation, Mapping) else ()
    benchmark_suites: list[dict[str, Any]] = []
    for row in suites:
        if not isinstance(row, Mapping):
            continue
        dataset = Path(str(row.get("dataset", "")))
        if not dataset.is_absolute():
            dataset = project.work_dir / dataset
        benchmark_suites.append(
            {
                "name": str(row.get("name", "")),
                "dataset_sha256": sha256_file(dataset),
                "prompt_field": str(row.get("prompt_field", "prompt")),
                "expected_field": str(row.get("expected_field", "expected")),
                "scoring": str(row.get("scoring", "normalized_exact_match")),
                "max_new_tokens": int(row.get("max_new_tokens", 64)),
                "use_chat_template": bool(row.get("use_chat_template", False)),
            }
        )
    payload = json.dumps(benchmark_suites, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _protocol_contract_digest(project: ProjectSpec) -> str:
    """Hash the complete static evaluation/training contract for this project."""
    config = project.config
    backend = config.get("backend", {})
    evaluation = config.get("evaluation", {})
    if not isinstance(backend, Mapping) or not isinstance(evaluation, Mapping):
        raise ValueError("project protocol contract requires backend and evaluation mappings")

    backend_contract = dict(backend)
    dataset_value = backend_contract.get("dataset")
    if dataset_value:
        dataset = Path(str(dataset_value))
        if not dataset.is_absolute():
            dataset = project.work_dir / dataset
        backend_contract["dataset_sha256"] = sha256_file(dataset)
    backend_contract.pop("dataset", None)

    evaluation_contract = dict(evaluation)
    suites = evaluation_contract.get("suites", ())
    normalized_suites: list[dict[str, object]] = []
    if not isinstance(suites, (list, tuple)):
        raise ValueError("evaluation.suites must be a list")
    for row in suites:
        if not isinstance(row, Mapping):
            raise ValueError("evaluation suite must be a mapping")
        suite = dict(row)
        dataset = Path(str(suite.get("dataset", "")))
        if not dataset.is_absolute():
            dataset = project.work_dir / dataset
        suite["dataset_sha256"] = sha256_file(dataset)
        suite.pop("dataset", None)
        normalized_suites.append(suite)
    evaluation_contract["suites"] = normalized_suites

    payload = {
        "seed": project.seed,
        "config_seed": config.get("seed", project.seed),
        "backend": backend_contract,
        "evaluation": evaluation_contract,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _legacy_unbounded_lifecycle(project: ProjectSpec) -> bool:
    lifecycle = project.config.get("goal_lifecycle")
    return (
        isinstance(lifecycle, Mapping)
        and lifecycle.get("mode") == "legacy_unbounded"
        and all(
            metric.minimum is None and metric.maximum is None
            for metric in project.goal.metrics
        )
    )


def _measured_protocol_digest(result: ExperimentResult | None) -> str | None:
    if result is None:
        return None
    digest = result_protocol_fingerprint(result.evidence)
    if digest is not None:
        return digest
    evaluation = result.evidence.get("evaluation")
    if isinstance(evaluation, Mapping):
        digest = evaluation.get("eval_spec_digest")
        if isinstance(digest, str) and len(digest) == 64:
            return digest
    digest = result.evidence.get("eval_spec_digest")
    return digest if isinstance(digest, str) and len(digest) == 64 else None


def _open_project_lifecycle(
    project: ProjectSpec,
    registry: RunRegistry,
    *,
    baseline: ExperimentResult | None = None,
) -> GoalLifecycle:
    """Construct the sole goal authority for the canonical project path."""
    stored = registry.get_goal_objective(project.objective_version)
    if stored is not None:
        identity = stored["identity"]
        baseline = baseline or next(
            (
                result
                for result in registry.list_results()
                if result.experiment_id == "baseline"
            ),
            None,
        )
        contract_digest = _protocol_contract_digest(project)
        legacy_unbounded = _legacy_unbounded_lifecycle(project)
        current_measured_digest = _measured_protocol_digest(baseline)
        if (
            current_measured_digest is not None
            and current_measured_digest != str(identity["evaluation_protocol_digest"])
        ):
            raise GoalLifecycleError(
                "objective identity changed (evaluation protocol evidence); refusing to resume objective"
            )
        return GoalLifecycle.open(
            registry,
            objective_version=project.objective_version,
            goal=project.goal,
            benchmark_digest=_project_benchmark_digest(project),
            evaluation_protocol_digest=str(identity["evaluation_protocol_digest"]),
            constitution=Constitution(),
            objective_metadata={
                "protocol_contract_digest": contract_digest,
                "legacy_unbounded": legacy_unbounded,
            },
            legacy_unbounded=legacy_unbounded,
            resume=True,
        )

    protocol_digest = _measured_protocol_digest(baseline) or protocol_fingerprint(
        {"evaluation": project.config.get("evaluation", {})}
    )
    contract_digest = _protocol_contract_digest(project)
    legacy_unbounded = _legacy_unbounded_lifecycle(project)
    return GoalLifecycle.open(
        registry,
        objective_version=project.objective_version,
        goal=project.goal,
        benchmark_digest=_project_benchmark_digest(project),
        evaluation_protocol_digest=protocol_digest,
        constitution=Constitution(),
        objective_metadata={
            "protocol_contract_digest": contract_digest,
            "legacy_unbounded": legacy_unbounded,
        },
        legacy_unbounded=legacy_unbounded,
    )


def migrate_legacy_protocol_contract(
    project: ProjectSpec,
    *,
    new_objective_version: str,
    approval: ProtocolContractMigrationApproval | None = None,
) -> ProjectSpec:
    """Migrate only a legacy objective's missing protocol contract.

    This operation is deliberately explicit and never called by ``run_project``.
    It preserves the legacy objective and appends a new objective plus an
    approval/provenance migration record. The returned project is the only
    version permitted to resume after migration.
    """
    if approval is None:
        raise GoalLifecycleError(
            "protocol-contract migration requires explicit human approval"
        )
    if not isinstance(new_objective_version, str) or not new_objective_version.strip():
        raise GoalLifecycleError("migration requires a non-empty new objective version")
    if new_objective_version == project.objective_version:
        raise GoalLifecycleError("protocol-contract migration requires a new objective version")
    project.validate_files()

    with RunRegistry(project.registry_path) as registry:
        stored = registry.get_goal_objective(project.objective_version)
        if stored is None:
            raise GoalLifecycleError(
                f"cannot migrate unknown objective: {project.objective_version}"
            )
        stored_goal = stored["goal"]
        if not isinstance(stored_goal, Mapping):
            raise GoalLifecycleError("persisted objective payload is invalid")
        if "protocol_contract_digest" in stored_goal:
            raise GoalLifecycleError("objective already has a protocol contract")
        stored_identity = stored["identity"]
        if goal_digest(project.goal) != stored_identity["goal_digest"]:
            raise GoalLifecycleError("current goal differs; migration scope is protocol contract only")
        current_benchmark = _project_benchmark_digest(project)
        if current_benchmark != stored_identity["benchmark_digest"]:
            raise GoalLifecycleError("current benchmark differs; migration scope is protocol contract only")

        target_identity = Constitution().new_objective_identity(
            objective_version=new_objective_version,
            goal=project.goal,
            benchmark_digest=str(stored_identity["benchmark_digest"]),
            evaluation_protocol_digest=str(stored_identity["evaluation_protocol_digest"]),
        )
        contract_digest = _protocol_contract_digest(project)
        target_goal_payload = dict(stored_goal)
        target_goal_payload["protocol_contract_digest"] = contract_digest
        source_identity_json = json.dumps(stored_identity, sort_keys=True, separators=(",", ":"))
        target_identity_payload = {
            "objective_version": target_identity.objective_version,
            "goal_digest": target_identity.goal_digest,
            "benchmark_digest": target_identity.benchmark_digest,
            "evaluation_protocol_digest": target_identity.evaluation_protocol_digest,
            "constitution_digest": target_identity.constitution_digest,
        }
        target_identity_json = json.dumps(
            target_identity_payload, sort_keys=True, separators=(",", ":")
        )
        provenance = GoalObjectiveMigrationProvenance(
            operation="legacy_protocol_contract_migration",
            source_objective_version=project.objective_version,
            target_objective_version=new_objective_version,
            source_identity_digest=hashlib.sha256(
                source_identity_json.encode("utf-8")
            ).hexdigest(),
            target_identity_digest=hashlib.sha256(
                target_identity_json.encode("utf-8")
            ).hexdigest(),
            protocol_contract_digest=contract_digest,
            registry_path=str(project.registry_path),
        )
        try:
            registry.record_goal_objective_migration(
                source_objective_version=project.objective_version,
                target_identity=target_identity,
                target_goal_payload=target_goal_payload,
                protocol_contract_digest=contract_digest,
                approval=approval.to_dict(),
                provenance=provenance,
            )
        except (KeyError, TypeError, ValueError, RegistryInvariantError) as exc:
            raise GoalLifecycleError(str(exc)) from exc
    return replace(project, objective_version=new_objective_version)


def run_project(
    project_or_path: ProjectSpec | str | Path,
    *,
    on_event: EventCallback | None = None,
    cancellation: CancellationToken | None = None,
    goal_lifecycle: GoalLifecycle | None = None,
) -> ProjectRunOutcome:
    """Execute one lifecycle-authoritative project generation.

    The canonical path always creates or resumes a frozen ``GoalLifecycle``;
    ``ProjectRunOutcome.succeeded`` is true only for ``STOP_GOALS_MET``.
    Training/evaluation success and promotion are deliberately not enough.

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
        stored_objective = registry.get_goal_objective(project.objective_version)
        lifecycle = goal_lifecycle
        if lifecycle is None and stored_objective is not None:
            lifecycle = _open_project_lifecycle(
                project,
                registry,
                baseline=project.baseline,
            )
        if lifecycle is not None and lifecycle.terminal_state is not None:
            persisted = lifecycle.terminal_result()
            return ProjectRunOutcome(
                project=project,
                hardware=detect_hardware(project.work_dir),
                generation=GenerationOutcome(
                    candidates=(),
                    ranking=(),
                    promoted=None,
                    goal_assessment=persisted.assessment,
                    goal_terminal_state=persisted.terminal_state.value,
                ),
                registry_audit=_closeout_registry_audit(registry, on_event),
            )
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
            resolved_config=project.config,
        )

        if project.baseline_mode == "auto":
            if lifecycle is not None:
                baseline = next(
                    (result for result in registry.list_results() if result.experiment_id == "baseline"),
                    None,
                )
                if baseline is None:
                    raise RuntimeError(
                        "persisted objective has no baseline result; refusing to bypass lifecycle"
                    )
                resolved_revision = None
            else:
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

        if lifecycle is None:
            lifecycle = _open_project_lifecycle(project, registry, baseline=baseline)

        if lifecycle.last_assessment is None and baseline is not None:
            parent_result = lifecycle.assess_parent_result(baseline)
            if parent_result.terminal_state is not None:
                return ProjectRunOutcome(
                    project=project,
                    hardware=hardware,
                    generation=GenerationOutcome(
                        candidates=(),
                        ranking=(),
                        promoted=None,
                        goal_assessment=parent_result.assessment,
                        goal_terminal_state=parent_result.terminal_state.value,
                    ),
                    registry_audit=_closeout_registry_audit(registry, on_event),
                )

        training_config = normalize_training_config_for_executor(training_config)
        baseline_deferred = baseline is None
        engine = EvolutionEngine(
            goal=project.goal,
            baseline=baseline,
            spent_gpu_hours=0.0 if baseline_deferred else baseline.gpu_hours,
            baseline_deferred=baseline_deferred,
        )
        if resolve_training_engine(training_config) == ROUTER_HEALING_ENGINE:
            trainer = create_training_executor(training_config)
            evaluator = create_evaluation_executor(training_config)
        else:
            # Preserve the existing constructor seam for the mature PEFT path;
            # the router backend is the only path that needs factory dispatch.
            trainer = TransformersPeftExecutor()
            evaluator = TransformersTextEvaluator()
        deferred_baseline = None
        if baseline_deferred:
            def complete_deferred_baseline(candidate):
                if candidate is None or candidate.evaluation is None:
                    raise RuntimeError("paired evaluation produced no candidate evidence")
                evaluation_evidence = dict(candidate.evaluation.evidence)
                base_loss = evaluation_evidence.get("base_holdout_loss")
                if not isinstance(base_loss, (int, float)):
                    raise RuntimeError("paired evaluation omitted base_holdout_loss")
                compute = {
                    "baseline_source": "paired-candidate-evaluation",
                    "model_loads": 1,
                    "shared_wall_gpu_hours": candidate.result.gpu_hours if candidate.result else 0.0,
                    "charged_to": candidate.experiment_id,
                    "total_gpu_hours": 0.0,
                }
                evidence = {
                    "baseline_source": "paired-candidate-evaluation",
                    "evaluation": evaluation_evidence,
                    "compute": compute,
                }
                result = ExperimentResult(
                    experiment_id="baseline",
                    metrics={"holdout_loss": float(base_loss)},
                    gpu_hours=0.0,
                    artifact_ref=None,
                    evidence=evidence,
                )
                registry.record_result(result)
                registry.update_experiment_status("baseline", ExperimentStatus.PASSED.value)
                return result
            deferred_baseline = complete_deferred_baseline

        runner = ExperimentCycleRunner(
            engine=engine,
            trainer=trainer,
            evaluator=evaluator,
            deferred_baseline=deferred_baseline,
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
            goal_lifecycle=lifecycle,
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
        if generation.candidates:
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

        if not generation.candidates:
            return ProjectRunOutcome(
                project=project,
                hardware=hardware,
                generation=generation,
                repair=repair_outcome,
                registry_audit=_closeout_registry_audit(registry, on_event),
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
            registry_audit=_closeout_registry_audit(registry, on_event),
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
