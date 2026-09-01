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

    def adjudicate(self, results: Iterable[ExperimentResult]) -> tuple[RankedCandidate, ...]:
        results = tuple(results)
        if any(not isinstance(result, ExperimentResult) for result in results):
            raise TypeError("adjudicate accepts evaluated ExperimentResult objects only")
        for result in results:
            reserved = self._reservations.pop(result.experiment_id, 0.0)
            self.reserved_gpu_hours = max(0.0, self.reserved_gpu_hours - reserved)
            self.spent_gpu_hours += max(0.0, result.gpu_hours)
            if result.experiment_id in self.graph.nodes:
                self.graph.nodes[result.experiment_id].status = ExperimentStatus.PASSED
        return rank_candidates(goal=self.goal, baseline=self.baseline, candidates=results)

    def promote(self, ranked: Iterable[RankedCandidate]) -> ExperimentResult | None:
        for candidate in ranked:
            if candidate.decision.accepted:
                self.baseline = candidate.result
                return candidate.result
        return None
