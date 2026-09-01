from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping

from .engine import EvolutionEngine
from .executors import EvaluationExecutor, EvaluationOutcome, ExecutionContext, TrainingArtifact, TrainingExecutor
from .models import Experiment, ExperimentResult, ExperimentStatus
from .registry import RunRegistry
from .tournament import RankedCandidate


@dataclass(frozen=True)
class CandidateCycleOutcome:
    experiment_id: str
    artifact: TrainingArtifact | None = None
    evaluation: EvaluationOutcome | None = None
    result: ExperimentResult | None = None
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


@dataclass
class ExperimentCycleRunner:
    """Run one experiment generation through train → evaluate → gate.

    The runner is intentionally framework-neutral. Training and evaluation are
    separate executors, and only the runner can combine their costs/evidence into
    the ``ExperimentResult`` accepted by ``EvolutionEngine.adjudicate``.
    """

    engine: EvolutionEngine
    trainer: TrainingExecutor
    evaluator: EvaluationExecutor
    context: ExecutionContext
    base_config: Mapping[str, Any] = field(default_factory=dict)
    registry: RunRegistry | None = None

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
        experiment.status = ExperimentStatus.RUNNING
        self._record_status(experiment)

        artifact: TrainingArtifact | None = None
        evaluation: EvaluationOutcome | None = None
        try:
            artifact = self.trainer.run(experiment, run_context)
            if artifact.experiment_id != experiment.experiment_id:
                raise ValueError("trainer returned an artifact for a different experiment")
            if artifact.gpu_hours < 0:
                raise ValueError("trainer returned negative gpu_hours")
            if not artifact.artifact_ref:
                raise ValueError("trainer returned an empty artifact_ref")
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
            if evaluation.gpu_hours < 0:
                raise ValueError("evaluator returned negative gpu_hours")
            metrics = _validated_metrics(evaluation.metrics)
            if self.registry is not None:
                self.registry.record_evaluation_outcome(evaluation)

            total_gpu_hours = artifact.gpu_hours + evaluation.gpu_hours
            result = ExperimentResult(
                experiment_id=experiment.experiment_id,
                metrics=metrics,
                gpu_hours=total_gpu_hours,
                artifact_ref=artifact.artifact_ref,
                evidence={
                    "training_run_id": artifact.run_id,
                    "evaluation_run_id": evaluation.run_id,
                    "training": dict(artifact.evidence),
                    "evaluation": dict(evaluation.evidence),
                    "compute": {
                        "training_gpu_hours": artifact.gpu_hours,
                        "evaluation_gpu_hours": evaluation.gpu_hours,
                        "total_gpu_hours": total_gpu_hours,
                    },
                },
            )
            if self.registry is not None:
                self.registry.record_result(result)
            return CandidateCycleOutcome(
                experiment_id=experiment.experiment_id,
                artifact=artifact,
                evaluation=evaluation,
                result=result,
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
                error=f"{type(exc).__name__}: {exc}",
            )

    def run_generation(self, experiments: Iterable[Experiment]) -> GenerationOutcome:
        candidates = tuple(self._run_candidate(experiment) for experiment in experiments)
        results = tuple(
            candidate.result for candidate in candidates if candidate.result is not None
        )
        ranking = self.engine.adjudicate(results) if results else ()
        promoted = self.engine.promote(ranking)

        if self.registry is not None:
            for candidate in candidates:
                node = self.engine.graph.nodes.get(candidate.experiment_id)
                if node is not None:
                    self.registry.update_experiment_status(candidate.experiment_id, node.status.value)

        return GenerationOutcome(candidates=candidates, ranking=ranking, promoted=promoted)
