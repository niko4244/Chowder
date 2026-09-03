from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping

from .backends.transformers_peft import TransformersPeftExecutor
from .cancellation import CancellationToken
from .config_validation import validate_transformers_backend_config
from .dependency_preflight import check_dependencies, check_disk_space
from .model_compatibility import check_causal_lm_architecture
from .engine import EvolutionEngine
from .evaluators.transformers_text import TransformersTextEvaluator
from .execution_failure import ExecutionFailure, ExecutionStage, normalize_execution_exception
from .executor_investigator import ExecutorFailureAnalysis, analyze_execution_failure
from .executors import EvaluationExecutor, EvaluationOutcome, ExecutionContext, TrainingArtifact, TrainingExecutor
from .failures import FailureRecord, RepairPlan, cluster_failures, plan_repairs
from .investigation import RemediationRegistry
from .models import Experiment, ExperimentResult, ExperimentStatus
from .registry import RunRegistry
from .run_events import TrainingProgressEvent
from .tournament import RankedCandidate

FailureHarvester = Callable[[EvaluationOutcome], tuple[FailureRecord, ...]]


@dataclass(frozen=True)
class CandidateCycleOutcome:
    experiment_id: str
    artifact: TrainingArtifact | None = None
    evaluation: EvaluationOutcome | None = None
    result: ExperimentResult | None = None
    harvested_failures: tuple[FailureRecord, ...] = ()
    repair_plans: tuple[RepairPlan, ...] = ()
    execution_failure: ExecutionFailure | None = None
    executor_analysis: ExecutorFailureAnalysis | None = None
    diagnostic_error: str | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.result is not None and self.error is None


@dataclass(frozen=True)
class GenerationOutcome:
    candidates: tuple[CandidateCycleOutcome, ...]
    ranking: tuple[RankedCandidate, ...]
    promoted: ExperimentResult | None

    @property
    def failures(self) -> tuple[CandidateCycleOutcome, ...]:
        return tuple(candidate for candidate in self.candidates if not candidate.succeeded)

    @property
    def harvested_failures(self) -> tuple[FailureRecord, ...]:
        return tuple(
            failure
            for candidate in self.candidates
            for failure in candidate.harvested_failures
        )

    @property
    def repair_plans(self) -> tuple[RepairPlan, ...]:
        return tuple(plan for candidate in self.candidates for plan in candidate.repair_plans)


def _validated_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    if not metrics:
        raise ValueError("evaluation returned no metrics")
    normalized: dict[str, float] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("evaluation metric names must be non-empty strings")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"evaluation metric {name!r} is not finite")
        normalized[name] = number
    return normalized


def _validated_failures(
    failures: Iterable[FailureRecord],
    *,
    evaluation: EvaluationOutcome,
) -> tuple[FailureRecord, ...]:
    protocol_sha = evaluation.evidence.get("protocol_sha256")
    validated: list[FailureRecord] = []
    seen: set[str] = set()
    for failure in failures:
        if failure.failure_id in seen:
            raise ValueError("failure harvester returned duplicate failure IDs")
        seen.add(failure.failure_id)
        if failure.experiment_id != evaluation.experiment_id:
            raise ValueError("failure record belongs to a different experiment")
        if failure.evaluation_run_id != evaluation.run_id:
            raise ValueError("failure record belongs to a different evaluation run")
        if isinstance(protocol_sha, str) and failure.protocol_sha256 != protocol_sha:
            raise ValueError("failure record protocol does not match evaluation protocol")
        if not math.isfinite(float(failure.score)):
            raise ValueError("failure record score is not finite")
        validated.append(failure)
    return tuple(validated)


def _declared_evaluation_reserve(config: Mapping[str, Any]) -> float:
    evaluation = config.get("evaluation", {})
    if evaluation is None:
        return 0.0
    if not isinstance(evaluation, Mapping):
        raise ValueError("resolved config evaluation section must be a mapping")
    raw = evaluation.get("estimated_gpu_hours", 0.0)
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ValueError("evaluation.estimated_gpu_hours must be finite and non-negative")
    return value


def _validate_trainer_config(trainer: TrainingExecutor, resolved: Mapping[str, Any]) -> None:
    """Dispatch strict config validation only for backends Chowder understands.

    Future/custom executors keep control of their own schemas. The built-in
    Transformers/PEFT executor is strict because silent unknown-key fallback is
    unsafe for autonomous research.
    """

    if str(getattr(trainer, "name", "")) == "transformers-peft":
        validate_transformers_backend_config(resolved)


_TRANSFORMERS_PEFT_TRAINING_PACKAGES = ("torch", "transformers", "peft", "datasets", "accelerate")
_TRANSFORMERS_TEXT_EVAL_PACKAGES = ("torch", "transformers", "peft")


def _check_dependencies(
    trainer: TrainingExecutor,
    evaluator: EvaluationExecutor,
    resolved: Mapping[str, Any],
    context: ExecutionContext,
) -> None:
    """Dependency preflight, dispatched by real type rather than by the
    duck-typed `.name` string _validate_trainer_config uses. Config
    validation legitimately applies to anything claiming to be a
    transformers-peft/-text backend, since it only inspects the resolved
    config dict -- but "are torch/transformers/peft/etc. importable in
    this process" is a question about whether the actual executor class
    can run, not about what name a test double happens to share. An
    isinstance check keeps every existing hand-written trainer/evaluator
    test double (which never subclasses the real executors) unaffected,
    while still catching a real missing dependency for the real backends.
    Runs before engine.resize_reservation(), so a missing package is a
    config-time rejection, not something discovered deep inside a spawned
    subprocess after GPU-hours were already reserved and a process
    already started.
    """
    if isinstance(trainer, TransformersPeftExecutor):
        check_dependencies(
            packages=_TRANSFORMERS_PEFT_TRAINING_PACKAGES,
            quantization=trainer.resolved_quantization(context),
            label="transformers-peft training",
        )
        backend = resolved.get("backend", {})
        training = backend.get("training", {}) if isinstance(backend, Mapping) else {}
        optimizer_tiering_mode = (
            str(training.get("optimizer_tiering", "off")).strip().lower()
            if isinstance(training, Mapping)
            else "off"
        )
        # Only "always" is checked explicitly here, not "auto": auto's own
        # real experiment (chowder.optimizer_tiering.run_optimizer_tiering_
        # experiment) already degrades gracefully to available=False/
        # recommended=False when bitsandbytes is missing -- no separate
        # preflight rejection is needed for that path. "always" bypasses
        # the experiment entirely and would otherwise only fail deep
        # inside the spawned training worker.
        if optimizer_tiering_mode == "always":
            check_dependencies(
                packages=(),
                quantization="4bit",
                label="transformers-peft optimizer tiering",
            )
    if isinstance(evaluator, TransformersTextEvaluator):
        backend = resolved.get("backend", {})
        evaluation = resolved.get("evaluation", {})
        backend = backend if isinstance(backend, Mapping) else {}
        evaluation = evaluation if isinstance(evaluation, Mapping) else {}
        quantization = str(evaluation.get("quantization", "inherit")).lower()
        if quantization == "inherit":
            quantization = str(backend.get("quantization", "none")).lower()
        check_dependencies(
            packages=_TRANSFORMERS_TEXT_EVAL_PACKAGES,
            quantization=quantization,
            label="transformers-text evaluation",
        )


_DEFAULT_MIN_FREE_DISK_GB = 2.0


def _check_disk_space(
    trainer: TrainingExecutor,
    evaluator: EvaluationExecutor,
    resolved: Mapping[str, Any],
    context: ExecutionContext,
) -> None:
    """Free-disk-space preflight, dispatched the same way as
    _check_dependencies: only for the real backends that actually download a
    model and write checkpoints/adapters to context.work_dir. One check
    covers both training and evaluation, since they share the same work_dir
    and run back-to-back within the same candidate.
    """
    if not isinstance(trainer, TransformersPeftExecutor) and not isinstance(
        evaluator, TransformersTextEvaluator
    ):
        return
    backend = resolved.get("backend", {})
    backend = backend if isinstance(backend, Mapping) else {}
    minimum_free_gb = float(backend.get("min_free_disk_gb", _DEFAULT_MIN_FREE_DISK_GB))
    check_disk_space(
        path=context.work_dir,
        minimum_free_gb=minimum_free_gb,
        label="transformers-peft training/evaluation",
    )


def _check_model_architecture(
    trainer: TrainingExecutor,
    evaluator: EvaluationExecutor,
    resolved: Mapping[str, Any],
    context: ExecutionContext,
) -> None:
    """Model-architecture-compatibility preflight, dispatched the same way
    as _check_dependencies/_check_disk_space. Training and evaluation both
    load backend.base_model with AutoModelForCausalLM, so one check covers
    both. Skips (rather than requiring base_model to be set) when it is
    absent -- that is a different, pre-existing validation concern, not
    this check's job.
    """
    if not isinstance(trainer, TransformersPeftExecutor) and not isinstance(
        evaluator, TransformersTextEvaluator
    ):
        return
    backend = resolved.get("backend", {})
    backend = backend if isinstance(backend, Mapping) else {}
    base_model = str(backend.get("base_model", "")).strip()
    if not base_model:
        return
    revision = backend.get("revision")
    check_causal_lm_architecture(
        base_model=base_model,
        revision=str(revision) if revision is not None else None,
        offline=bool(backend.get("offline", False)),
        label="transformers-peft training/evaluation",
    )


@dataclass
class ExperimentCycleRunner:
    """Run one generation through profile → train → evaluate → diagnose → gate."""

    engine: EvolutionEngine
    trainer: TrainingExecutor
    evaluator: EvaluationExecutor
    context: ExecutionContext
    base_config: Mapping[str, Any] = field(default_factory=dict)
    registry: RunRegistry | None = None
    failure_harvester: FailureHarvester | None = None
    executor_remediation_registry: RemediationRegistry = field(default_factory=RemediationRegistry)
    executor_investigation_budget: float = 0.25
    cancellation: CancellationToken | None = None
    progress_callback: Callable[[TrainingProgressEvent], None] | None = None

    def __post_init__(self) -> None:
        budget = float(self.executor_investigation_budget)
        if not math.isfinite(budget) or budget < 0:
            raise ValueError("executor_investigation_budget must be finite and non-negative")

    def _record_status(self, experiment: Experiment) -> None:
        if self.registry is not None:
            self.registry.update_experiment_status(experiment.experiment_id, experiment.status.value)

    def _bind_cancellation(self, executor: object, token: CancellationToken | None) -> None:
        bind = getattr(executor, "bind_cancellation", None)
        if callable(bind):
            bind(token)

    def _bind_progress(self, executor: object, callback) -> None:
        """Optional capability, same shape as _bind_cancellation: only
        TransformersPeftExecutor defines bind_progress_callback today
        (evaluation progress is a separate, not-yet-built piece), so this
        is a silent no-op for the evaluator and for every hand-written test
        double in this suite -- getattr/callable, never isinstance."""
        bind = getattr(executor, "bind_progress_callback", None)
        if callable(bind):
            bind(callback)

    def _run_candidate(self, experiment: Experiment) -> CandidateCycleOutcome:
        if experiment.experiment_id not in self.engine.graph.nodes:
            raise ValueError("experiment must be proposed before execution")
        if not self.engine.has_reservation(experiment.experiment_id):
            raise ValueError("experiment has no active compute reservation")

        if self.cancellation is not None and self.cancellation.requested:
            self.engine.cancel_reservation(
                experiment.experiment_id, status=ExperimentStatus.REJECTED
            )
            experiment.status = ExperimentStatus.REJECTED
            self._record_status(experiment)
            return CandidateCycleOutcome(
                experiment_id=experiment.experiment_id,
                error="cancelled before start",
            )

        resolved = self.engine.resolve_config(experiment.experiment_id, self.base_config)
        run_context = replace(self.context, resolved_config=resolved)

        try:
            _validate_trainer_config(self.trainer, resolved)
            _check_dependencies(self.trainer, self.evaluator, resolved, run_context)
            _check_disk_space(self.trainer, self.evaluator, resolved, run_context)
            _check_model_architecture(self.trainer, self.evaluator, resolved, run_context)
            try:
                training_estimate = self.trainer.profile(experiment, run_context)
            except NotImplementedError:
                training_gpu_hours = self.engine.reservation_for(experiment.experiment_id)
            else:
                training_gpu_hours = training_estimate.gpu_hours
            try:
                evaluation_estimate = self.evaluator.profile(experiment, run_context)
            except NotImplementedError:
                evaluation_gpu_hours = _declared_evaluation_reserve(resolved)
            else:
                evaluation_gpu_hours = evaluation_estimate.gpu_hours
            lifecycle_estimate = training_gpu_hours + evaluation_gpu_hours
            self.engine.resize_reservation(experiment.experiment_id, lifecycle_estimate)
        except Exception as exc:
            self.engine.cancel_reservation(
                experiment.experiment_id,
                status=ExperimentStatus.REJECTED,
            )
            experiment.status = ExperimentStatus.REJECTED
            self._record_status(experiment)
            return CandidateCycleOutcome(
                experiment_id=experiment.experiment_id,
                error=f"preflight {type(exc).__name__}: {exc}",
            )

        experiment.status = ExperimentStatus.RUNNING
        self._record_status(experiment)

        artifact: TrainingArtifact | None = None
        evaluation: EvaluationOutcome | None = None
        harvested: tuple[FailureRecord, ...] = ()
        repair_plans: tuple[RepairPlan, ...] = ()
        diagnostic_error: str | None = None

        training_started = time.perf_counter()
        self._bind_cancellation(self.trainer, self.cancellation)
        self._bind_progress(self.trainer, self.progress_callback)
        try:
            artifact = self.trainer.run(experiment, run_context)
        except Exception as exc:
            elapsed = time.perf_counter() - training_started
            was_cancelled = self.cancellation is not None and self.cancellation.requested
            failure = normalize_execution_exception(
                exc,
                experiment=experiment,
                executor_name=str(getattr(self.trainer, "name", type(self.trainer).__name__)),
                context=run_context,
                wall_seconds=elapsed,
                stage=ExecutionStage.TRAIN,
            )
            analysis: ExecutorFailureAnalysis | None = None
            # A cancellation is a deliberate, expected stop, not an anomaly
            # -- routing it through the Executor Investigator would spend
            # its investigation budget diagnosing something that isn't one.
            if not was_cancelled:
                try:
                    analysis = analyze_execution_failure(
                        failure,
                        context=run_context,
                        registry=self.executor_remediation_registry,
                        gpu_hour_budget=self.executor_investigation_budget,
                        investigation_id=f"investigate-{failure.run_id}",
                        occurred_at=datetime.now(timezone.utc).isoformat(),
                    )
                except Exception as investigator_exc:
                    diagnostic_error = (
                        f"executor investigator {type(investigator_exc).__name__}: "
                        f"{investigator_exc}"
                    )

            self.engine.fail(
                experiment.experiment_id,
                actual_gpu_hours=failure.gpu_hours_spent,
            )
            experiment.status = ExperimentStatus.FAILED
            self._record_status(experiment)
            error = f"{failure.cause_type}: {failure.cause_message}"
            return CandidateCycleOutcome(
                experiment_id=experiment.experiment_id,
                execution_failure=failure,
                executor_analysis=analysis,
                diagnostic_error=diagnostic_error,
                error=f"cancelled: {error}" if was_cancelled else error,
            )
        finally:
            self._bind_cancellation(self.trainer, None)
            self._bind_progress(self.trainer, None)

        try:
            if artifact.experiment_id != experiment.experiment_id:
                raise ValueError("trainer returned an artifact for a different experiment")
            if self.registry is not None:
                self.registry.record_training_artifact(artifact)
        except Exception as exc:
            self.engine.fail(experiment.experiment_id, actual_gpu_hours=artifact.gpu_hours)
            experiment.status = ExperimentStatus.FAILED
            self._record_status(experiment)
            return CandidateCycleOutcome(
                experiment_id=experiment.experiment_id,
                artifact=artifact,
                error=f"{type(exc).__name__}: {exc}",
            )

        evaluation_started = time.perf_counter()
        self._bind_cancellation(self.evaluator, self.cancellation)
        try:
            evaluation = self.evaluator.evaluate(
                experiment=experiment,
                artifact=artifact,
                context=run_context,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - evaluation_started
            was_cancelled = self.cancellation is not None and self.cancellation.requested
            failure = normalize_execution_exception(
                exc,
                experiment=experiment,
                executor_name=str(getattr(self.evaluator, "name", type(self.evaluator).__name__)),
                context=run_context,
                wall_seconds=elapsed,
                stage=ExecutionStage.EVALUATE,
            )
            analysis: ExecutorFailureAnalysis | None = None
            if not was_cancelled:
                try:
                    analysis = analyze_execution_failure(
                        failure,
                        context=run_context,
                        registry=self.executor_remediation_registry,
                        gpu_hour_budget=self.executor_investigation_budget,
                        investigation_id=f"investigate-{failure.run_id}",
                        occurred_at=datetime.now(timezone.utc).isoformat(),
                    )
                except Exception as investigator_exc:
                    diagnostic_error = (
                        f"executor investigator {type(investigator_exc).__name__}: "
                        f"{investigator_exc}"
                    )

            known_compute = artifact.gpu_hours + (failure.gpu_hours_spent or 0.0)
            self.engine.fail(experiment.experiment_id, actual_gpu_hours=known_compute)
            experiment.status = ExperimentStatus.FAILED
            self._record_status(experiment)
            error = f"{failure.cause_type}: {failure.cause_message}"
            return CandidateCycleOutcome(
                experiment_id=experiment.experiment_id,
                artifact=artifact,
                execution_failure=failure,
                executor_analysis=analysis,
                diagnostic_error=diagnostic_error,
                error=f"cancelled: {error}" if was_cancelled else error,
            )
        finally:
            self._bind_cancellation(self.evaluator, None)

        try:
            if evaluation.experiment_id != experiment.experiment_id:
                raise ValueError("evaluator returned evidence for a different experiment")
            if evaluation.source_artifact_ref != artifact.artifact_ref:
                raise ValueError("evaluator did not evaluate the training artifact")
            metrics = _validated_metrics(evaluation.metrics)
            if self.registry is not None:
                self.registry.record_evaluation_outcome(evaluation)

            if self.failure_harvester is not None:
                try:
                    harvested = _validated_failures(
                        self.failure_harvester(evaluation),
                        evaluation=evaluation,
                    )
                    repair_plans = plan_repairs(cluster_failures(harvested))
                    if self.registry is not None:
                        self.registry.record_failures(harvested)
                        for plan in repair_plans:
                            self.registry.record_repair_plan(plan)
                except Exception as exc:
                    diagnostic_error = f"{type(exc).__name__}: {exc}"

            total_gpu_hours = artifact.gpu_hours + evaluation.gpu_hours
            evidence: dict[str, Any] = {
                "training_run_id": artifact.run_id,
                "evaluation_run_id": evaluation.run_id,
                "training": dict(artifact.evidence),
                "evaluation": dict(evaluation.evidence),
                "compute": {
                    "reserved_lifecycle_gpu_hours": self.engine.reservation_for(
                        experiment.experiment_id
                    ),
                    "training_gpu_hours": artifact.gpu_hours,
                    "evaluation_gpu_hours": evaluation.gpu_hours,
                    "total_gpu_hours": total_gpu_hours,
                },
                "diagnostics": {
                    "failure_count": len(harvested),
                    "repair_plan_count": len(repair_plans),
                    "error": diagnostic_error,
                },
            }
            protocol_sha = evaluation.evidence.get("protocol_sha256")
            if isinstance(protocol_sha, str) and len(protocol_sha) == 64:
                evidence["evaluation_protocol_sha256"] = protocol_sha

            result = ExperimentResult(
                experiment_id=experiment.experiment_id,
                metrics=metrics,
                gpu_hours=total_gpu_hours,
                artifact_ref=artifact.artifact_ref,
                evidence=evidence,
            )
            if self.registry is not None:
                self.registry.record_result(result)
            return CandidateCycleOutcome(
                experiment_id=experiment.experiment_id,
                artifact=artifact,
                evaluation=evaluation,
                result=result,
                harvested_failures=harvested,
                repair_plans=repair_plans,
                diagnostic_error=diagnostic_error,
            )
        except Exception as exc:
            self.engine.fail(
                experiment.experiment_id,
                actual_gpu_hours=artifact.gpu_hours + evaluation.gpu_hours,
            )
            experiment.status = ExperimentStatus.FAILED
            self._record_status(experiment)
            return CandidateCycleOutcome(
                experiment_id=experiment.experiment_id,
                artifact=artifact,
                evaluation=evaluation,
                harvested_failures=harvested,
                repair_plans=repair_plans,
                diagnostic_error=diagnostic_error,
                error=f"{type(exc).__name__}: {exc}",
            )

    def run_generation(self, experiments: Iterable[Experiment]) -> GenerationOutcome:
        candidates = tuple(self._run_candidate(experiment) for experiment in experiments)
        results = tuple(candidate.result for candidate in candidates if candidate.result is not None)
        ranking = self.engine.adjudicate(results) if results else ()
        promoted = self.engine.promote(ranking)

        if self.registry is not None:
            for candidate in candidates:
                node = self.engine.graph.nodes.get(candidate.experiment_id)
                if node is not None:
                    self.registry.update_experiment_status(candidate.experiment_id, node.status.value)

        return GenerationOutcome(candidates=candidates, ranking=ranking, promoted=promoted)
