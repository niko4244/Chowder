from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping

from .engine import EvolutionEngine
from .executors import EvaluationExecutor, EvaluationOutcome, ExecutionContext, TrainingArtifact, TrainingExecutor
from .failures import FailureRecord, RepairPlan, cluster_failures, plan_repairs
from .models import Experiment, ExperimentResult, ExperimentStatus
from .registry import RunRegistry
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

    def _record_status(self, experiment: Experiment) -> None:
        if self.registry is not None:
            self.registry.update_experiment_status(experiment.experiment_id, experiment.status.value)

    def _run_candidate(self, experiment: Experiment) -> CandidateCycleOutcome:
        if experiment.experiment_id not in self.engine.graph.nodes:
            raise ValueError("experiment must be proposed before execution")
        if not self.engine.has_reservation(experiment.experiment_id):
            raise ValueError("experiment has no active compute reservation")

        resolved = self.engine.resolve_config(experiment.experiment_id, self.base_config)
        run_context = replace(self.context, resolved_config=resolved)

        # Preflight is intentionally outside the execution/failure accounting
        # path. A profile/config failure consumed no model compute, so release the
        # reservation without charging the experiment as a failed GPU run.
        try:
            training_estimate = self.trainer.profile(experiment, run_context)
            lifecycle_estimate = (
                training_estimate.gpu_hours + _declared_evaluation_reserve(resolved)
            )
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

        try:
            artifact = self.trainer.run(experiment, run_context)
            if artifact.experiment_id != experiment.experiment_id:
                raise ValueError("trainer returned an artifact for a different experiment")
            if self.registry is not None:
                self.registry.record_training_artifact(artifact)

            evaluation = self.evaluator.evaluate(
                experiment=experiment,
                artifact=artifact,
                context=run_context,
            )
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
            known_compute = artifact.gpu_hours if artifact is not None else None
            self.engine.fail(experiment.experiment_id, actual_gpu_hours=known_compute)
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
