"""Bisect a rejected candidate's own checkpoints to find where, during that
single training run, a regression was first introduced -- using the exact
independent evaluator and hard gate every candidate already goes through,
never a training-loss proxy.

`checkpoint_discovery.py` finds and validates checkpoints against a NEW
run's config (to decide whether it's safe to resume from one); this module
answers a different question about an ALREADY-FINISHED, already-rejected
run: which of its own checkpoints, independently re-evaluated and gated
against the same baseline the final candidate was gated against, was the
first to actually regress.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .executors import EvaluationExecutor, ExecutionContext, TrainingArtifact
from .gate import evaluate_candidate
from .models import Experiment, ExperimentResult, GateDecision, Goal
from .provenance import sha256_directory


@dataclass(frozen=True)
class CheckpointVerdict:
    checkpoint_dir: str
    step: int
    result: ExperimentResult
    decision: GateDecision


@dataclass(frozen=True)
class CheckpointBisectOutcome:
    verdicts: tuple[CheckpointVerdict, ...]  # ascending by step

    @property
    def first_regressing(self) -> CheckpointVerdict | None:
        """The earliest checkpoint (by step) the real gate rejected, or None
        if every checkpoint the run wrote was gate-acceptable."""
        return next((verdict for verdict in self.verdicts if not verdict.decision.accepted), None)


def _step_from_name(name: str) -> int:
    try:
        return int(name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        raise ValueError(f"not a checkpoint-N directory name: {name!r}") from None


def _checkpoints_for_run(artifact_ref: str) -> tuple[Path, ...]:
    """The same trainer/checkpoint-N layout every checkpoint/resume path in
    this codebase already relies on (see successive_halving.py's own
    `_latest_checkpoint_dir`), but returning every checkpoint the run wrote,
    not just the most recent one."""
    trainer_dir = Path(artifact_ref) / "trainer"
    if not trainer_dir.is_dir():
        return ()
    checkpoints = [p for p in trainer_dir.glob("checkpoint-*") if p.is_dir()]
    return tuple(sorted(checkpoints, key=lambda p: _step_from_name(p.name)))


def evaluate_all_checkpoints(
    *,
    experiment: Experiment,
    rejected_result: ExperimentResult,
    evaluator: EvaluationExecutor,
    context: ExecutionContext,
    goal: Goal,
    baseline: ExperimentResult,
) -> CheckpointBisectOutcome:
    """Independently re-evaluate every real checkpoint written during a
    rejected candidate's own training run, gating each one against the same
    baseline the final candidate itself was gated against.

    Linear scan, not binary search: a run's checkpoint count is typically
    single digits, so the real cost is dominated by launching each
    independent evaluation subprocess, not by how many comparisons decide
    which checkpoint to look at next. Bisection only pays for itself once a
    run accumulates dozens of checkpoints, which no run in this codebase's
    own tests or production configs does yet -- a future binary-search
    variant can reuse this exact per-checkpoint evaluation step without
    changing it.
    """
    verdicts: list[CheckpointVerdict] = []
    for checkpoint_dir in _checkpoints_for_run(rejected_result.artifact_ref):
        step = _step_from_name(checkpoint_dir.name)
        artifact = TrainingArtifact(
            run_id=f"{rejected_result.experiment_id}-checkpoint-{step}",
            experiment_id=experiment.experiment_id,
            artifact_ref=str(checkpoint_dir),
            gpu_hours=0.0,
            evidence={"artifact_sha256": sha256_directory(checkpoint_dir)},
        )
        evaluation = evaluator.evaluate(experiment=experiment, artifact=artifact, context=context)
        result = ExperimentResult(
            experiment_id=experiment.experiment_id,
            metrics=dict(evaluation.metrics),
            gpu_hours=evaluation.gpu_hours,
            artifact_ref=str(checkpoint_dir),
            evidence={
                "checkpoint_step": step,
                "checkpoint_sha256": artifact.evidence["artifact_sha256"],
                "evaluation": dict(evaluation.evidence),
            },
        )
        decision = evaluate_candidate(goal=goal, baseline=baseline, candidate=result)
        verdicts.append(CheckpointVerdict(str(checkpoint_dir), step, result, decision))

    return CheckpointBisectOutcome(tuple(verdicts))
