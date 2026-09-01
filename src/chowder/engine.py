from __future__ import annotations

import math
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

    def __post_init__(self) -> None:
        for label, value in (
            ("spent_gpu_hours", self.spent_gpu_hours),
            ("reserved_gpu_hours", self.reserved_gpu_hours),
        ):
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        if self.spent_gpu_hours + self.reserved_gpu_hours > self.goal.gpu_hour_budget + 1e-12:
            raise ValueError("engine spent + reserved GPU-hours exceed goal budget")

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.goal.gpu_hour_budget - self.spent_gpu_hours - self.reserved_gpu_hours)

    @property
    def outstanding_candidates(self) -> int:
        return len(self._reservations)

    def has_reservation(self, experiment_id: str) -> bool:
        return experiment_id in self._reservations

    def reservation_for(self, experiment_id: str) -> float:
        if experiment_id not in self._reservations:
            raise ValueError(f"experiment has no active reservation: {experiment_id}")
        return self._reservations[experiment_id]

    def propose(self, experiments: Iterable[Experiment]) -> tuple[Experiment, ...]:
        accepted: list[Experiment] = []
        staged_hours = 0.0
        for experiment in experiments:
            if self.outstanding_candidates + len(accepted) >= self.goal.max_parallel_candidates:
                break
            if experiment.estimated_gpu_hours > self.remaining_budget - staged_hours:
                continue
            accepted.append(experiment)
            staged_hours += experiment.estimated_gpu_hours

        rows = tuple(accepted)
        self.graph.add_many(rows)
        for experiment in rows:
            self._reservations[experiment.experiment_id] = experiment.estimated_gpu_hours
        self.reserved_gpu_hours += staged_hours
        return rows

    def resize_reservation(self, experiment_id: str, gpu_hours: float) -> float:
        """Resize a planned reservation from a measured/profiled lifecycle estimate."""

        if experiment_id not in self._reservations:
            raise ValueError(f"experiment has no active reservation: {experiment_id}")
        number = float(gpu_hours)
        if not math.isfinite(number) or number < 0:
            raise ValueError("reservation GPU-hours must be finite and non-negative")
        old = self._reservations[experiment_id]
        delta = number - old
        if delta > self.remaining_budget + 1e-12:
            raise ValueError(
                f"profiled lifecycle cost {number:.6g} GPU-hours does not fit remaining budget"
            )
        self._reservations[experiment_id] = number
        self.reserved_gpu_hours += delta
        if self.reserved_gpu_hours < -1e-12:
            raise RuntimeError("reservation accounting became negative")
        self.reserved_gpu_hours = max(0.0, self.reserved_gpu_hours)
        return number

    def cancel_reservation(
        self,
        experiment_id: str,
        *,
        status: ExperimentStatus = ExperimentStatus.REJECTED,
    ) -> None:
        """Release a reservation before execution without charging compute."""

        if experiment_id not in self._reservations:
            raise ValueError(f"experiment has no active reservation: {experiment_id}")
        reserved = self._reservations.pop(experiment_id)
        self.reserved_gpu_hours = max(0.0, self.reserved_gpu_hours - reserved)
        node = self.graph.nodes.get(experiment_id)
        if node is None:
            raise ValueError(f"reserved experiment is missing from graph: {experiment_id}")
        node.status = status

    def withdraw_proposals(self, experiment_ids: Iterable[str]) -> tuple[Experiment, ...]:
        ids = tuple(experiment_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate experiment id in proposal withdrawal")
        for experiment_id in ids:
            if experiment_id not in self._reservations:
                raise ValueError(f"experiment has no active reservation: {experiment_id}")
            node = self.graph.nodes.get(experiment_id)
            if node is None:
                raise ValueError(f"reserved experiment is missing from graph: {experiment_id}")
            if node.status is not ExperimentStatus.PLANNED:
                raise ValueError(
                    f"cannot withdraw experiment {experiment_id} with status {node.status.value}"
                )

        removed = self.graph.remove_many(ids)
        released = 0.0
        for experiment_id in ids:
            released += self._reservations.pop(experiment_id)
        self.reserved_gpu_hours = max(0.0, self.reserved_gpu_hours - released)
        return removed

    def resolve_config(
        self, experiment_id: str, base_config: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.graph.resolve_config(experiment_id, base_config)

    def _settle_reservation(
        self, experiment_id: str, *, actual_gpu_hours: float, status: ExperimentStatus
    ) -> None:
        if experiment_id not in self._reservations:
            raise ValueError(f"experiment has no active reservation: {experiment_id}")
        actual = float(actual_gpu_hours)
        if not math.isfinite(actual) or actual < 0:
            raise ValueError("actual GPU-hours must be finite and non-negative")
        reserved = self._reservations.pop(experiment_id)
        self.reserved_gpu_hours = max(0.0, self.reserved_gpu_hours - reserved)
        self.spent_gpu_hours += actual
        if experiment_id in self.graph.nodes:
            self.graph.nodes[experiment_id].status = status

    def fail(self, experiment_id: str, *, actual_gpu_hours: float | None = None) -> None:
        if experiment_id not in self._reservations:
            raise ValueError(f"experiment has no active reservation: {experiment_id}")
        reserved = self._reservations[experiment_id]
        if actual_gpu_hours is None:
            charge = reserved
        else:
            actual = float(actual_gpu_hours)
            if not math.isfinite(actual) or actual < 0:
                raise ValueError("failed-run actual GPU-hours must be finite and non-negative")
            charge = max(reserved, actual)
        self._settle_reservation(
            experiment_id,
            actual_gpu_hours=charge,
            status=ExperimentStatus.FAILED,
        )

    def adjudicate(self, results: Iterable[ExperimentResult]) -> tuple[RankedCandidate, ...]:
        results = tuple(results)
        if any(not isinstance(result, ExperimentResult) for result in results):
            raise TypeError("adjudicate accepts evaluated ExperimentResult objects only")
        result_ids = [result.experiment_id for result in results]
        if len(result_ids) != len(set(result_ids)):
            raise ValueError("adjudicate received duplicate experiment results")
        missing_reservations = [
            experiment_id for experiment_id in result_ids if experiment_id not in self._reservations
        ]
        if missing_reservations:
            raise ValueError(
                f"cannot adjudicate experiments without active reservations: {missing_reservations}"
            )

        ranked = rank_candidates(goal=self.goal, baseline=self.baseline, candidates=results)
        decisions = {candidate.result.experiment_id: candidate.decision for candidate in ranked}
        for result in results:
            decision = decisions[result.experiment_id]
            status = ExperimentStatus.PASSED if decision.accepted else ExperimentStatus.REJECTED
            self._settle_reservation(
                result.experiment_id,
                actual_gpu_hours=result.gpu_hours,
                status=status,
            )
        return ranked

    def promote(self, ranked: Iterable[RankedCandidate]) -> ExperimentResult | None:
        for candidate in ranked:
            if candidate.decision.accepted:
                self.baseline = candidate.result
                return candidate.result
        return None
