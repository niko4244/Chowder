"""Bandit-style reordering of not-yet-run experiment candidates.

Successive halving and the tournament gate answer "was this candidate any
good" only *after* it has already spent real GPU-hours. This module answers
a cheaper question asked *before* any GPU-hours are spent: "which candidates
in this batch look most promising to run first, given everything we've
learned from every experiment run so far?"

It never touches the hard regression gate (`gate.evaluate_candidate`) or the
tournament ranking (`tournament.rank_candidates`) that already decide
acceptance/promotion after a real run. It only *reorders* a pool of
not-yet-run `Experiment` objects before they are handed to
`EvolutionEngine.propose()` / `Cycle.run_round()` -- a low-confidence
candidate can still run, just later.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import log, sqrt
from typing import Any, Mapping, Sequence

from .gate import evaluate_candidate
from .models import Experiment, ExperimentResult, Goal

_EXPLORATION_WEIGHT = 2.0


def dotted_paths(config_patch: Mapping[str, Any], prefix: str = "") -> frozenset[str]:
    """Flatten a config_patch into the dotted leaf key-paths it touches.

    Two experiments that both set ``backend.training.learning_rate`` (to
    different values) are the same *arm* -- the bandit tracks which knobs
    are being turned, not which values were tried for them.

    Public because this is now the single definition of an "arm" shared
    with `intervention_outcomes.py`, which groups historical outcomes by
    the same arm this module bandits over -- two definitions that silently
    drifted apart would make the two modules disagree about which
    experiments were the same intervention.
    """
    paths: set[str] = set()
    for key, value in config_patch.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            paths |= dotted_paths(value, prefix=path)
        else:
            paths.add(path)
    return frozenset(paths)


@dataclass
class _ArmStatistics:
    pulls: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.pulls if self.pulls else 0.0


def _ucb1_score(arm: _ArmStatistics, *, total_pulls: int) -> float:
    if arm.pulls == 0:
        return float("inf")
    return arm.mean_reward + _EXPLORATION_WEIGHT * sqrt(log(total_pulls) / arm.pulls)


def prioritize_candidates(
    candidates: Sequence[Experiment],
    *,
    history: Sequence[tuple[Experiment, ExperimentResult]],
    goal: Goal,
    baseline: ExperimentResult,
) -> tuple[Experiment, ...]:
    """Reorder not-yet-run *candidates* by UCB1 score over historical arms.

    Each arm is the set of dotted config_patch key-paths an experiment
    touches (see `dotted_paths`). Reward is the same gpu-hour-normalized
    gate score `tournament.rank_candidates` uses (`decision.score /
    gpu_hours`), replayed against *history* through the same hard gate every
    real candidate goes through -- this function never grants promotion or
    bypasses that gate, it only decides which not-yet-run candidate to spend
    GPU-hours on first.

    Cold start: an arm with zero historical pulls (including every arm when
    *history* is empty) scores `+inf`. `sorted` is stable, so candidates
    whose arms have never been tried keep their original relative order --
    an empty history introduces no bias.
    """
    if not candidates:
        return ()

    arms: dict[frozenset[str], _ArmStatistics] = defaultdict(_ArmStatistics)
    for experiment, result in history:
        decision = evaluate_candidate(goal=goal, baseline=baseline, candidate=result)
        reward = decision.score / max(result.gpu_hours, 1e-9)
        arm = arms[dotted_paths(experiment.config_patch)]
        arm.pulls += 1
        arm.total_reward += reward

    total_pulls = max(sum(arm.pulls for arm in arms.values()), 1)

    def score(experiment: Experiment) -> float:
        arm = arms.get(dotted_paths(experiment.config_patch))
        if arm is None:
            return float("inf")
        return _ucb1_score(arm, total_pulls=total_pulls)

    return tuple(sorted(candidates, key=lambda experiment: -score(experiment)))
