from __future__ import annotations

from typing import Sequence

from .investigation import HypothesisTrial


def rank_trials(trials: Sequence[HypothesisTrial]) -> tuple[HypothesisTrial, ...]:
    """Order trials by how much evidence backs them, most-corroborated first.

    Task 5's minimal ranking: probe-result count only, no cost tiebreak --
    Task 6 adds that once ranking actually has to choose between more than
    the one trial the walking skeleton exercises. Populates each trial's
    existing `.rank` field as it scores them, since nothing sets that field
    before a ranking pass runs.
    """
    for trial in trials:
        trial.rank = float(len(trial.probe_results))
    return tuple(sorted(trials, key=lambda t: t.rank, reverse=True))
