from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .autonomous_repair import (
    AutonomousRepairOutcome,
    RepairTarget,
    _repairable_target,
    run_single_hop_autonomous_repair,
)
from .cycle import ExperimentCycleRunner, GenerationOutcome
from .failures import FailureRecord
from .repair_candidates import RepairVariant
from .repair_requests import RepairSourceProvider


class RecursiveRepairStopReason(str, Enum):
    PROMOTED = "promoted"
    MAX_DEPTH = "max_depth"
    NO_PROGRESS = "no_progress"
    REPEATED_FAILURE = "repeated_failure"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_ADMISSIBLE_CANDIDATE = "no_admissible_candidate"
    NO_REPAIRABLE_DIAGNOSTIC = "no_repairable_diagnostic"


@dataclass(frozen=True)
class RecursiveRepairPolicy:
    """Finite stopping policy for autonomous repair recursion."""

    max_depth: int = 3
    min_score_improvement: float = 1e-4
    max_failure_signature_occurrences: int = 1
    replay_ratio: float | None = 1.0

    def __post_init__(self) -> None:
        if self.max_depth <= 0:
            raise ValueError("recursive repair max_depth must be positive")
        if not math.isfinite(float(self.min_score_improvement)):
            raise ValueError("min_score_improvement must be finite")
        if self.min_score_improvement < 0:
            raise ValueError("min_score_improvement cannot be negative")
        if self.max_failure_signature_occurrences <= 0:
            raise ValueError("max_failure_signature_occurrences must be positive")
        if self.replay_ratio is not None:
            ratio = float(self.replay_ratio)
            if not math.isfinite(ratio) or ratio <= 0 or ratio > 10:
                raise ValueError("replay_ratio must be finite and in (0, 10]")


@dataclass(frozen=True)
class RecursiveRepairHop:
    depth: int
    target_experiment_id: str
    failure_signature: str
    target_score: float
    score_improvement: float | None
    outcome: AutonomousRepairOutcome
    remaining_budget_after: float


@dataclass(frozen=True)
class RecursiveRepairOutcome:
    initial_generation: GenerationOutcome
    final_generation: GenerationOutcome
    hops: tuple[RecursiveRepairHop, ...]
    stop_reason: RecursiveRepairStopReason
    stop_detail: str

    @property
    def promoted(self):
        if self.final_generation.promoted is not None:
            return self.final_generation.promoted
        for hop in reversed(self.hops):
            if hop.outcome.promoted is not None:
                return hop.outcome.promoted
        return self.initial_generation.promoted

    @property
    def depth(self) -> int:
        return len(self.hops)


def _stable_failure_rows(target: RepairTarget) -> tuple[FailureRecord, ...]:
    wanted = set(target.cluster.failure_ids)
    rows = tuple(
        failure
        for failure in target.candidate.harvested_failures
        if failure.failure_id in wanted
    )
    if len(rows) != len(wanted):
        raise ValueError("repair target failure evidence is incomplete")
    return rows


def failure_signature(target: RepairTarget) -> str:
    """Hash the stable benchmark failure state, excluding experiment/run identity.

    Failure IDs cannot be used because they intentionally include experiment and
    evaluation-run identity. This signature instead captures the evaluator,
    protocol, suite, failure kind, source role, and the hidden benchmark rows by
    row index plus prompt/expected hashes. Raw holdout text never leaves this
    internal diagnostic boundary.
    """

    rows = _stable_failure_rows(target)
    payload = {
        "version": 1,
        "evaluator": target.cluster.evaluator,
        "suite": target.cluster.suite,
        "protocol_sha256": target.cluster.protocol_sha256,
        "source_role": target.cluster.source_role.value,
        "failure_kind": target.cluster.failure_kind,
        "rows": sorted(
            (
                {
                    "row_index": int(row.row_index),
                    "prompt_sha256": hashlib.sha256(
                        row.prompt.encode("utf-8")
                    ).hexdigest(),
                    "expected_sha256": hashlib.sha256(
                        row.expected.encode("utf-8")
                    ).hexdigest(),
                }
                for row in rows
            ),
            key=lambda row: (
                row["row_index"], row["prompt_sha256"], row["expected_sha256"]
            ),
        ),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _score_for(generation: GenerationOutcome, experiment_id: str) -> float:
    for ranked in generation.ranking:
        if ranked.result.experiment_id == experiment_id:
            return float(ranked.decision.score)
    raise ValueError("repair target is missing from generation ranking")


def _score_improvement(previous: float, current: float) -> float:
    previous_finite = math.isfinite(previous)
    current_finite = math.isfinite(current)
    if previous_finite and current_finite:
        return current - previous
    if not previous_finite and current_finite:
        return math.inf
    if previous_finite and not current_finite:
        return -math.inf
    return 0.0


def _ranked_repairable_targets(
    generation: GenerationOutcome,
) -> tuple[RepairTarget, ...]:
    targets: list[RepairTarget] = []
    for ranked in generation.ranking:
        if ranked.decision.accepted:
            continue
        try:
            targets.append(
                _repairable_target(
                    generation,
                    candidate_id=ranked.result.experiment_id,
                )
            )
        except ValueError as exc:
            if "no independently repairable diagnostics" in str(exc):
                continue
            raise
    return tuple(targets)


def _select_novel_target(
    generation: GenerationOutcome,
    *,
    signature_counts: dict[str, int],
    max_occurrences: int,
) -> tuple[RepairTarget | None, str | None, bool]:
    targets = _ranked_repairable_targets(generation)
    if not targets:
        return None, None, False
    for target in targets:
        signature = failure_signature(target)
        if signature_counts.get(signature, 0) < max_occurrences:
            return target, signature, True
    return None, None, True


def _budget_stop_reason(runner: ExperimentCycleRunner) -> RecursiveRepairStopReason:
    if runner.engine.remaining_budget <= 0:
        return RecursiveRepairStopReason.BUDGET_EXHAUSTED
    return RecursiveRepairStopReason.NO_ADMISSIBLE_CANDIDATE


def run_bounded_autonomous_repair(
    *,
    runner: ExperimentCycleRunner,
    source_generation: GenerationOutcome,
    provider: RepairSourceProvider,
    variants: Iterable[RepairVariant],
    policy: RecursiveRepairPolicy = RecursiveRepairPolicy(),
) -> RecursiveRepairOutcome:
    """Run finite, evidence-preserving recursive repair.

    A hop is permitted only when a gate-rejected candidate has independently
    repairable diagnostics, its stable failure signature has not exhausted the
    policy limit, there is sufficient measured score progress after the first
    hop, and the engine can admit another repair population within budget.
    """

    variant_rows = tuple(variants)
    if not variant_rows:
        raise ValueError("bounded recursive repair requires at least one variant")
    if any(variant.lora_patch for variant in variant_rows):
        raise ValueError("bounded continuation repair cannot change LoRA topology")

    if source_generation.promoted is not None:
        return RecursiveRepairOutcome(
            initial_generation=source_generation,
            final_generation=source_generation,
            hops=(),
            stop_reason=RecursiveRepairStopReason.PROMOTED,
            stop_detail="source generation already contains a promoted candidate",
        )

    current = source_generation
    hops: list[RecursiveRepairHop] = []
    signature_counts: dict[str, int] = {}
    previous_target_score: float | None = None

    for depth in range(1, policy.max_depth + 1):
        if runner.engine.remaining_budget <= 0:
            return RecursiveRepairOutcome(
                source_generation,
                current,
                tuple(hops),
                RecursiveRepairStopReason.BUDGET_EXHAUSTED,
                "GPU-hour budget is exhausted",
            )

        target, signature, had_repairable = _select_novel_target(
            current,
            signature_counts=signature_counts,
            max_occurrences=policy.max_failure_signature_occurrences,
        )
        if target is None or signature is None:
            reason = (
                RecursiveRepairStopReason.REPEATED_FAILURE
                if had_repairable
                else RecursiveRepairStopReason.NO_REPAIRABLE_DIAGNOSTIC
            )
            detail = (
                "all repairable failure signatures reached their recurrence limit"
                if had_repairable
                else "generation contains no independently repairable rejected diagnostics"
            )
            return RecursiveRepairOutcome(
                source_generation, current, tuple(hops), reason, detail
            )

        target_score = _score_for(current, target.candidate.experiment_id)
        gain: float | None = None
        if previous_target_score is not None:
            gain = _score_improvement(previous_target_score, target_score)
            if gain < policy.min_score_improvement:
                return RecursiveRepairOutcome(
                    source_generation,
                    current,
                    tuple(hops),
                    RecursiveRepairStopReason.NO_PROGRESS,
                    (
                        f"best novel rejected candidate improved by {gain:.6g}; "
                        f"minimum required is {policy.min_score_improvement:.6g}"
                    ),
                )

        signature_counts[signature] = signature_counts.get(signature, 0) + 1
        try:
            one_hop = run_single_hop_autonomous_repair(
                runner=runner,
                source_generation=current,
                provider=provider,
                variants=variant_rows,
                candidate_id=target.candidate.experiment_id,
                replay_ratio=policy.replay_ratio,
            )
        except ValueError as exc:
            message = str(exc)
            if (
                "fits the remaining GPU-hour budget" in message
                or "no parallel candidate slots" in message
                or "produced no budget-admissible candidates" in message
            ):
                reason = _budget_stop_reason(runner)
                return RecursiveRepairOutcome(
                    source_generation,
                    current,
                    tuple(hops),
                    reason,
                    message,
                )
            raise

        hop = RecursiveRepairHop(
            depth=depth,
            target_experiment_id=target.candidate.experiment_id,
            failure_signature=signature,
            target_score=target_score,
            score_improvement=gain,
            outcome=one_hop,
            remaining_budget_after=runner.engine.remaining_budget,
        )
        hops.append(hop)
        current = one_hop.repair_generation
        previous_target_score = target_score

        if current.promoted is not None:
            return RecursiveRepairOutcome(
                source_generation,
                current,
                tuple(hops),
                RecursiveRepairStopReason.PROMOTED,
                f"repair hop {depth} produced a promoted candidate",
            )

    return RecursiveRepairOutcome(
        source_generation,
        current,
        tuple(hops),
        RecursiveRepairStopReason.MAX_DEPTH,
        f"reached configured repair depth {policy.max_depth}",
    )
