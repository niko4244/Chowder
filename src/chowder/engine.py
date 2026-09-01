from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .graph import ExperimentGraph
from .models import Experiment, ExperimentResult, ExperimentStatus, Goal
from .tournament import RankedCandidate, rank_candidates


@dataclass
class EvolutionEngine:
    goal: Goal
    baseline: ExperimentResult
    graph: ExperimentGraph = field(default_factory=ExperimentGraph)
    spent_gpu_hours: float = 0.0
    reserved_gpu_hours: float = 0.0
    _reservations: dict[str, float] = field(default_factory=dict)

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.goal.gpu_hour_budget - self.spent_gpu_hours - self.reserved_gpu_hours)

    @property
    def outstanding_candidates(self) -> int:
        return len(self._reservations)

    def has_reservation(self, experiment_id: str) -> bool:
        return experiment_id in self._reservations

    def propose(self, experiments: Iterable[Experiment]) -> tuple[Experiment, ...]:
        accepted: list[Experiment] = []
        for experiment in experiments:
            if self.outstanding_candidates + len(accepted) >= self.goal.max_parallel_candidates:
                break
            if experiment.estimated_gpu_hours <= 0:
                continue
            if experiment.estimated_gpu_hours > self.remaining_budget - sum(e.estimated_gpu_hours for e in accepted):
                continue
            self.graph.add(experiment)
            accepted.append(experiment)

        for experiment in accepted:
            self._reservations[experiment.experiment_id] = experiment.estimated_gpu_hours
            self.reserved_gpu_hours += experiment.estimated_gpu_hours
        return tuple(accepted)

    def resolve_config(
        self, experiment_id: str, base_config: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.graph.resolve_config(experiment_id, base_config)

    def _settle_reservation(
        self, experiment_id: str, *, actual_gpu_hours: float, status: ExperimentStatus
    ) -> None:
        reserved = self._reservations.pop(experiment_id, 0.0)
        self.reserved_gpu_hours = max(0.0, self.reserved_gpu_hours - reserved)
        self.spent_gpu_hours += max(0.0, actual_gpu_hours)
        if experiment_id in self.graph.nodes:
            self.graph.nodes[experiment_id].status = status

    def fail(self, experiment_id: str, *, actual_gpu_hours: float | None = None) -> None:
        """Fail a candidate and conservatively settle its reservation.

        When a backend crashes before it can report actual compute, Chowder
        charges the reserved estimate rather than pretending the failed run was
        free. If partial compute is known, the larger of actual and reserved is
        charged so failures cannot create phantom budget.
        """
        reserved = self._reservations.get(experiment_id, 0.0)
        charge = reserved if actual_gpu_hours is None else max(reserved, actual_gpu_hours)
        self._settle_reservation(
            experiment_id, actual_gpu_hours=charge, status=ExperimentStatus.FAILED
        )

    def adjudicate(self, results: Iterable[ExperimentResult]) -> tuple[RankedCandidate, ...]:
        results = tuple(results)
        if any(not isinstance(result, ExperimentResult) for result in results):
            raise TypeError("adjudicate accepts evaluated ExperimentResult objects only")
        ranked = rank_candidates(goal=self.goal, baseline=self.baseline, candidates=results)
        decisions = {candidate.result.experiment_id: candidate.decision for candidate in ranked}
        for result in results:
            decision = decisions[result.experiment_id]
            status = ExperimentStatus.PASSED if decision.accepted else ExperimentStatus.REJECTED
            self._settle_reservation(
                result.experiment_id, actual_gpu_hours=result.gpu_hours, status=status
            )
        return ranked

    def promote(self, ranked: Iterable[RankedCandidate]) -> ExperimentResult | None:
        for candidate in ranked:
            if candidate.decision.accepted:
                self.baseline = candidate.result
                return candidate.result
        return None
