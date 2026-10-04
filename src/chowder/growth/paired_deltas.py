"""Paired causal deltas: compare parent and candidate under the same
conditions, or do not compare at all.

An aggregate mean can hide exactly the regression that matters -- a candidate
that wins 60 tasks by a little and loses 40 by a lot can look fine on the
mean. Paired evidence (same prompt, same seed, same decoding settings, same
evaluator, same hardware class, same protocol) exposes task-level wins and
losses, per-dimension deltas, and a bootstrap interval that says whether the
delta is a measurement or noise. Where the pairing keys disagree, the pairing
refuses: a delta computed across a protocol change is not causal evidence,
it is two different experiments wearing one table.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = [
    "PairingRefusal",
    "PairedDelta",
    "PairedOutcome",
    "pair_outcomes",
    "bootstrap_delta_interval",
]


class PairingRefusal(RuntimeError):
    """The two arms were not measured under the same protocol."""


#: The pairing keys that must match for a delta to be causal evidence.
_PAIRING_KEYS: tuple[str, ...] = (
    "decoding",
    "seed",
    "evaluator",
    "evaluator_version",
    "hardware_class",
    "protocol_sha256",
)


@dataclass(frozen=True)
class PairedOutcome:
    """One measured row from one arm, ready to pair."""

    task_id: str
    score: float
    protocol: Mapping[str, Any]
    #: Optional dimension-level sub-scores (tool validity, termination...).
    dimensions: Mapping[str, float] | None = None


@dataclass(frozen=True)
class PairedDelta:
    """The causal comparison of one candidate arm against its parent."""

    dimension: str
    paired_tasks: int
    wins: int
    losses: int
    ties: int
    parent_mean: float
    candidate_mean: float
    mean_delta: float
    #: Task-level deltas in preregistered task order.
    task_deltas: tuple[float, ...]

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        return self.wins / decided if decided else 0.0


def _protocol_key(protocol: Mapping[str, Any]) -> tuple:
    return tuple(
        repr(protocol.get(key)) for key in _PAIRING_KEYS
    )


def pair_outcomes(
    *,
    dimension: str,
    parent: Sequence[PairedOutcome],
    candidate: Sequence[PairedOutcome],
) -> PairedDelta:
    """Pair parent and candidate rows by task under identical protocol keys.

    Refuses when: a task exists on only one arm (the arms measured different
    task sets -- pairing the intersection would silently drop the missing
    rows, which is where regressions hide), or the protocol keys differ on any
    paired task.
    """
    parent_by_task = {row.task_id: row for row in parent}
    candidate_by_task = {row.task_id: row for row in candidate}
    parent_tasks = set(parent_by_task)
    candidate_tasks = set(candidate_by_task)
    if parent_tasks != candidate_tasks:
        missing_in_parent = sorted(candidate_tasks - parent_tasks)
        missing_in_candidate = sorted(parent_tasks - candidate_tasks)
        raise PairingRefusal(
            f"the arms measured different task sets for {dimension!r}: "
            f"missing from parent {missing_in_parent}, missing from candidate "
            f"{missing_in_candidate}; pairing the intersection would hide the "
            "missing rows"
        )
    task_deltas: list[float] = []
    wins = losses = ties = 0
    parent_sum = candidate_sum = 0.0
    for task_id in sorted(parent_tasks):
        parent_row = parent_by_task[task_id]
        candidate_row = candidate_by_task[task_id]
        if _protocol_key(parent_row.protocol) != _protocol_key(candidate_row.protocol):
            differing = [
                key
                for key in _PAIRING_KEYS
                if repr(parent_row.protocol.get(key)) != repr(candidate_row.protocol.get(key))
            ]
            raise PairingRefusal(
                f"task {task_id!r} changed protocol keys {differing} between "
                "the arms; a delta across a protocol change is not causal "
                "evidence"
            )
        delta = candidate_row.score - parent_row.score
        task_deltas.append(delta)
        parent_sum += parent_row.score
        candidate_sum += candidate_row.score
        if delta > 1e-12:
            wins += 1
        elif delta < -1e-12:
            losses += 1
        else:
            ties += 1
    paired = len(task_deltas)
    return PairedDelta(
        dimension=dimension,
        paired_tasks=paired,
        wins=wins,
        losses=losses,
        ties=ties,
        parent_mean=parent_sum / paired if paired else math.nan,
        candidate_mean=candidate_sum / paired if paired else math.nan,
        mean_delta=sum(task_deltas) / paired if paired else math.nan,
        task_deltas=tuple(task_deltas),
    )


def bootstrap_delta_interval(
    delta: PairedDelta,
    *,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 20261003,
) -> tuple[float, float]:
    """A percentile bootstrap CI over the paired task deltas.

    Deterministic for a given seed. Returns the (low, high) percentile bounds
    of the mean delta under resampling -- a mean outside the interval around
    zero is a measured effect, not noise.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if not delta.task_deltas:
        raise ValueError("no paired deltas to resample")
    rng = random.Random(seed)
    deltas = list(delta.task_deltas)
    means: list[float] = []
    for _ in range(iterations):
        sample = [deltas[rng.randrange(len(deltas))] for _ in range(len(deltas))]
        means.append(sum(sample) / len(sample))
    means.sort()
    tail = (1.0 - confidence) / 2.0
    low = means[max(0, int(tail * iterations))]
    high = means[min(iterations - 1, int((1.0 - tail) * iterations) - 1)]
    return (low, high)
