from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .cycle import CandidateCycleOutcome, ExperimentCycleRunner, GenerationOutcome
from .graph import deep_merge_config
from .models import Experiment, ExperimentResult

_DEFAULT_CHECKPOINT_FRACTION = 0.5


@dataclass(frozen=True)
class HalvingRoundOutcome:
    """One real round of successive halving: every candidate in this
    round ran for real (a real, bounded-budget training + independent
    evaluation), went through the same hard regression gate every
    Chowder candidate always goes through (rank_candidates via
    ExperimentCycleRunner.run_round), and had its real reservation
    settled for real spent GPU-hours -- successive halving changes how
    much budget a candidate gets and who advances, never how a
    candidate's own result is judged.
    """

    round_index: int
    max_steps: int
    generation: GenerationOutcome
    survivor_experiment_ids: tuple[str, ...]
    eliminated_by_gate_experiment_ids: tuple[str, ...]
    eliminated_by_cutoff_experiment_ids: tuple[str, ...]


@dataclass(frozen=True)
class SuccessiveHalvingOutcome:
    """The full run: every real round, in order, plus the final
    promotion decision (made once, against the LAST round's real
    results -- no earlier, cheap-budget round's winner is ever promoted
    over the real current baseline; see ExperimentCycleRunner.run_round's
    own docstring for why)."""

    rounds: tuple[HalvingRoundOutcome, ...]
    promoted: ExperimentResult | None

    @property
    def total_rounds(self) -> int:
        return len(self.rounds)

    @property
    def total_gpu_hours(self) -> float:
        return sum(
            candidate.result.gpu_hours
            for round_outcome in self.rounds
            for candidate in round_outcome.generation.candidates
            if candidate.result is not None
        )


def _latest_checkpoint_dir(artifact_ref: str) -> Path | None:
    """The real checkpoint a survivor's next round resumes from --
    globbing the same trainer/checkpoint-N layout every real checkpoint/
    resume test in this codebase already relies on, picking the highest
    real step number actually written."""
    trainer_dir = Path(artifact_ref) / "trainer"
    if not trainer_dir.is_dir():
        return None
    checkpoints = [p for p in trainer_dir.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None

    def _step(path: Path) -> int:
        try:
            return int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            return -1

    return max(checkpoints, key=_step)


def _round_max_steps(*, initial_max_steps: int, round_index: int, step_multiplier: float) -> int:
    return max(1, round(initial_max_steps * (step_multiplier**round_index)))


def _round_experiment_config_patch(*, max_steps: int, resume_from_checkpoint: Path | None) -> dict[str, Any]:
    save_steps = max(1, math.ceil(max_steps * _DEFAULT_CHECKPOINT_FRACTION))
    training: dict[str, Any] = {
        "max_steps": max_steps,
        "save_strategy": "steps",
        "save_steps": save_steps,
    }
    backend: dict[str, Any] = {"training": training}
    if resume_from_checkpoint is not None:
        backend["resume_from_checkpoint"] = str(resume_from_checkpoint)
    return {"backend": backend}


def _next_round_experiment(
    *,
    parent_experiment: Experiment,
    parent_outcome: CandidateCycleOutcome,
    round_index: int,
    previous_max_steps: int,
    max_steps: int,
) -> Experiment | None:
    """Build the child experiment that continues a real survivor into
    the next, more expensive round -- via the existing parent/child
    config-patch lineage (ExperimentGraph.resolve_config walks root to
    leaf), not a hand-reconstructed full config. Returns None if the
    survivor has no real checkpoint to resume from (its round did not
    write one -- a real, honest reason to stop advancing that candidate
    rather than silently restart it from scratch)."""
    assert parent_outcome.result is not None
    checkpoint_dir = _latest_checkpoint_dir(parent_outcome.result.artifact_ref)
    if checkpoint_dir is None:
        return None
    parent_gpu_hours = max(parent_outcome.result.gpu_hours, 1e-6)
    # Scale the new round's *estimated* GPU-hours from the real measured
    # cost of the round just completed -- evidence-based, not a guess,
    # and only ever a starting point: the normal preflight (Trainer.
    # profile / engine.resize_reservation) still refines it for real
    # once this candidate is actually proposed and profiled.
    step_ratio = max_steps / max(1, previous_max_steps)
    estimated_gpu_hours = parent_gpu_hours * step_ratio
    return Experiment(
        experiment_id=f"{parent_experiment.experiment_id}-r{round_index}",
        parent_id=parent_experiment.experiment_id,
        hypothesis=parent_experiment.hypothesis,
        config_patch=_round_experiment_config_patch(
            max_steps=max_steps, resume_from_checkpoint=checkpoint_dir
        ),
        estimated_gpu_hours=estimated_gpu_hours,
        tags=parent_experiment.tags,
    )


def run_successive_halving(
    runner: ExperimentCycleRunner,
    experiments: tuple[Experiment, ...],
    *,
    initial_max_steps: int,
    step_multiplier: float = 2.0,
    survival_fraction: float = 0.5,
    min_survivors: int = 1,
    max_rounds: int | None = None,
) -> SuccessiveHalvingOutcome:
    """Run many candidates cheaply, progressively allocating more real
    GPU-hours only to survivors -- Priority 4's successive halving.

    Each round is a real ExperimentCycleRunner.run_round() call: real
    bounded-length training (max_steps), real independent evaluation,
    the same hard regression gate every Chowder candidate goes through,
    and real reservation settlement for real spent GPU-hours. Survivors
    advance into the next round via a real checkpoint resume (built
    through the existing parent/child experiment config-patch lineage,
    not a hand-reconstructed config) with step_multiplier times the
    previous round's budget. Only the FINAL round's winner is ever
    promoted -- no early, cheap-budget round's winner becomes the new
    baseline, however well it ranks in its own round.

    initial_max_steps/step_multiplier/survival_fraction/min_survivors
    are documented starting points, not claimed-optimal constants,
    matching every other threshold in this codebase's search/placement
    mechanisms -- the right schedule genuinely depends on the workload.

    experiments must already carry `backend.training.max_steps` at
    initial_max_steps in their own config (or leave it to this
    function to inject on round 0 -- see the round-0 handling below,
    which applies the same config_patch shape every later round uses,
    so round 0 is not a structurally different case from round 1+).
    """
    if not 0 < survival_fraction < 1:
        raise ValueError("survival_fraction must be strictly between 0 and 1")
    if min_survivors < 1:
        raise ValueError("min_survivors must be at least 1")
    if step_multiplier <= 1:
        raise ValueError("step_multiplier must be greater than 1")

    rounds: list[HalvingRoundOutcome] = []
    current_experiments: dict[str, Experiment] = {
        experiment.experiment_id: experiment for experiment in experiments
    }
    round_index = 0

    while current_experiments:
        max_steps = _round_max_steps(
            initial_max_steps=initial_max_steps, round_index=round_index, step_multiplier=step_multiplier
        )
        round_input = tuple(
            replace(
                experiment,
                config_patch=deep_merge_config(
                    experiment.config_patch,
                    _round_experiment_config_patch(max_steps=max_steps, resume_from_checkpoint=None),
                ),
            )
            if round_index == 0
            else experiment
            for experiment in current_experiments.values()
        )
        accepted = runner.engine.propose(round_input)
        generation = runner.run_round(accepted, promote=False)

        accepted_ranking = tuple(item for item in generation.ranking if item.decision.accepted)
        rejected_ranking = tuple(item for item in generation.ranking if not item.decision.accepted)
        survivor_count = min(
            len(accepted_ranking),
            max(min_survivors, math.ceil(len(accepted_ranking) * survival_fraction)),
        )
        survivors = accepted_ranking[:survivor_count]
        cutoff = accepted_ranking[survivor_count:]

        rounds.append(
            HalvingRoundOutcome(
                round_index=round_index,
                max_steps=max_steps,
                generation=generation,
                survivor_experiment_ids=tuple(item.result.experiment_id for item in survivors),
                eliminated_by_gate_experiment_ids=tuple(
                    item.result.experiment_id for item in rejected_ranking
                ),
                eliminated_by_cutoff_experiment_ids=tuple(
                    item.result.experiment_id for item in cutoff
                ),
            )
        )

        is_final_round = (
            len(survivors) <= min_survivors
            or (max_rounds is not None and round_index + 1 >= max_rounds)
            or not survivors
        )
        if is_final_round:
            promoted = runner.engine.promote(generation.ranking)
            return SuccessiveHalvingOutcome(rounds=tuple(rounds), promoted=promoted)

        next_max_steps = _round_max_steps(
            initial_max_steps=initial_max_steps, round_index=round_index + 1, step_multiplier=step_multiplier
        )
        candidate_lookup = {
            outcome.experiment_id: outcome for outcome in generation.candidates
        }
        next_experiments: dict[str, Experiment] = {}
        for ranked in survivors:
            experiment_id = ranked.result.experiment_id
            parent_experiment = next(e for e in accepted if e.experiment_id == experiment_id)
            outcome = candidate_lookup[experiment_id]
            child = _next_round_experiment(
                parent_experiment=parent_experiment,
                parent_outcome=outcome,
                round_index=round_index + 1,
                previous_max_steps=max_steps,
                max_steps=next_max_steps,
            )
            if child is not None:
                next_experiments[child.experiment_id] = child

        if not next_experiments:
            promoted = runner.engine.promote(generation.ranking)
            return SuccessiveHalvingOutcome(rounds=tuple(rounds), promoted=promoted)

        current_experiments = next_experiments
        round_index += 1

    return SuccessiveHalvingOutcome(rounds=tuple(rounds), promoted=None)
