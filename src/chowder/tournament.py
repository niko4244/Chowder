from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .gate import evaluate_candidate
from .models import ExperimentResult, Goal, GateDecision


@dataclass(frozen=True)
class RankedCandidate:
    result: ExperimentResult
    decision: GateDecision
    efficiency: float


def rank_candidates(
    *,
    goal: Goal,
    baseline: ExperimentResult,
    candidates: Iterable[ExperimentResult],
) -> tuple[RankedCandidate, ...]:
    ranked: list[RankedCandidate] = []
    for candidate in candidates:
        decision = evaluate_candidate(goal=goal, baseline=baseline, candidate=candidate)
        # GPU-hour efficiency is deliberately separate from the hard safety gate.
        efficiency = decision.score / max(candidate.gpu_hours, 1e-9)
        ranked.append(RankedCandidate(candidate, decision, efficiency))

    return tuple(
        sorted(
            ranked,
            key=lambda item: (
                item.decision.accepted,
                item.decision.score,
                item.efficiency,
                -item.result.gpu_hours,
            ),
            reverse=True,
        )
    )
