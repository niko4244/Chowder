from __future__ import annotations

from typing import Sequence

from .investigation import HypothesisTrial


def rank_trials(trials: Sequence[HypothesisTrial]) -> tuple[HypothesisTrial, ...]:
    """Order trials by how much evidence backs them, most-corroborated first;
    break ties by estimated cost, cheapest first.

    Probe-result count is the primary key (Task 5). The cost tiebreak
    (Task 6) matters as soon as a signature_kind has more than one
    candidate hypothesis attached to the same probe evidence -- e.g. two
    competing fixes for a CUDA_OOM incident both get the same probes
    recorded against them, so without a tiebreak their order would be
    arbitrary. Populates each trial's existing `.rank` field as it scores
    them, since nothing sets that field before a ranking pass runs.
    """
    for trial in trials:
        trial.rank = float(len(trial.probe_results))
    return tuple(sorted(trials, key=lambda t: (-t.rank, t.estimated_gpu_hours)))
