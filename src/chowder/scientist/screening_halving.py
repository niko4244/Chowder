"""Successive halving for the screening lane (docs/COMPUTE_PROVIDERS.md §4).

The growth library (`chowder.successive_halving`) implements budget-driven
elimination against the *local* ExperimentCycleRunner: real bounded training
rounds, checkpoint resume, registry persistence. The screening lane runs the
same schedule against the *scheduler* instead — short/cheap runs routed to the
declared screening lane (Kaggle T4×2 today), with each round giving survivors
`step_multiplier` times the previous round's budget until the survivor set is
small enough to graduate to substantial runs.

This module carries the schedule arithmetic and the deterministic settlement
rule; it executes nothing. Round semantics mirror the library exactly:

- survivors per round = min(candidates, max(min_survivors,
  ceil(candidates × survival_fraction)));
- a candidate with no score was eliminated *by gate* (its run never produced
  a usable observation — the same split as the library's
  rejected_ranking/eliminated_by_gate);
- a candidate scored below the survivor line was eliminated *by cutoff*;
- the survivor line is deterministic: score descending, experiment_id
  ascending as the tiebreak — an LLM never ranks candidates;
- only the FINAL round's survivors graduate (a cheap early round never
  promotes anyone), mirroring "promoted from the last round only".

Budgets: round r costs `round_budget(r) = min(budget_cap_gpu_hours,
initial_budget_gpu_hours × step_multiplier^r)` per candidate. The cap is the
screening-lane discipline (screening runs stay cheap; substantial runs happen
elsewhere), so a long schedule saturates at the cap rather than outgrowing it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScreeningHalving:
    """The schedule: budgets grow by the multiplier, candidates halve."""

    initial_budget_gpu_hours: float = 0.05
    budget_cap_gpu_hours: float = 0.25
    step_multiplier: float = 2.0
    survival_fraction: float = 0.5
    min_survivors: int = 1
    max_rounds: int = 4

    def __post_init__(self) -> None:
        if self.initial_budget_gpu_hours < 0:
            raise ValueError("initial_budget_gpu_hours must be non-negative")
        if self.budget_cap_gpu_hours < self.initial_budget_gpu_hours:
            raise ValueError(
                "budget_cap_gpu_hours must be >= initial_budget_gpu_hours; "
                "the cap bounds screening runs"
            )
        if not 0 < self.survival_fraction < 1:
            raise ValueError("survival_fraction must be strictly between 0 and 1")
        if self.min_survivors < 1:
            raise ValueError("min_survivors must be at least 1")
        if self.step_multiplier <= 1:
            raise ValueError("step_multiplier must be greater than 1")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def round_budget(self, round_index: int) -> float:
        """Per-candidate screening budget for a round: grow by the
        multiplier, saturate at the cap."""
        if round_index < 0:
            raise ValueError("round_index must be non-negative")
        return min(
            self.budget_cap_gpu_hours,
            self.initial_budget_gpu_hours * (self.step_multiplier**round_index),
        )

    def survivor_count(self, candidates: int) -> int:
        """How many candidates advance from a round (mirrors the library)."""
        if candidates < 0:
            raise ValueError("candidates must be non-negative")
        if candidates == 0:
            return 0
        return min(
            candidates,
            max(self.min_survivors, math.ceil(candidates * self.survival_fraction)),
        )

    def is_final_round(self, round_index: int, survivors: int) -> bool:
        """Graduate after this round when the survivor set is small enough
        or the round budget is exhausted (mirrors the library's stop rule)."""
        if survivors < 0:
            raise ValueError("survivors must be non-negative")
        return (
            survivors <= self.min_survivors
            or round_index + 1 >= self.max_rounds
        )


@dataclass(frozen=True)
class HalvingRoundOutcome:
    """One settled round: what ran, at what budget, who advanced and why."""

    round_index: int
    budget_gpu_hours: float
    submitted_experiment_ids: tuple[str, ...]
    survivor_experiment_ids: tuple[str, ...]
    eliminated_by_gate_experiment_ids: tuple[str, ...]
    eliminated_by_cutoff_experiment_ids: tuple[str, ...]
    #: (experiment_id, score) best-first — the audit trail of the cut.
    scores: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class ScreeningHalvingOutcome:
    """The full schedule; `final_survivors` graduate to substantial runs."""

    rounds: tuple[HalvingRoundOutcome, ...]
    final_survivors: tuple[str, ...]

    @property
    def total_rounds(self) -> int:
        return len(self.rounds)

    @property
    def total_screening_gpu_hours(self) -> float:
        """Sum of per-candidate budgets over every submitted run (the
        screening lane's own cost; measured wall time is charged by the
        mission ledger when observations land)."""
        return sum(
            round_outcome.budget_gpu_hours * len(round_outcome.submitted_experiment_ids)
            for round_outcome in self.rounds
        )


def settle_screening_round(
    scores: Mapping[str, float | None],
    halving: ScreeningHalving,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[tuple[str, float], ...]]:
    """Settle one round from per-candidate scores. Deterministic, mechanical.

    - score None → eliminated by gate (no usable observation);
    - otherwise ranked score-descending (experiment_id ascending tiebreak);
      the top `survivor_count(len(competing))` survive, the rest are cut.
    """
    by_gate = tuple(sorted(k for k, v in scores.items() if v is None))
    competing = {k: float(v) for k, v in scores.items() if v is not None}
    keep = halving.survivor_count(len(competing))
    ranked: tuple[tuple[str, float], ...] = tuple(
        sorted(competing.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    survivors = tuple(k for k, _ in ranked[:keep])
    by_cutoff = tuple(k for k, _ in ranked[keep:])
    return survivors, by_gate, by_cutoff, ranked


def screening_score_from_node(node: Any, score_node: Any) -> float | None:
    """A candidate's round score from its research node: None when the node
    never carried an observation (gate), else the tree's deterministic score.
    The tree — not an LLM — is the authority on what a run was worth."""
    observation_ids = getattr(node, "observation_ids", ())
    if not observation_ids:
        return None
    return float(score_node(node))
