from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable, Mapping
from uuid import uuid4

from .autonomous_repair import (
    AutonomousRepairOutcome,
    RepairTarget,
    _repairable_target,
    run_single_hop_autonomous_repair,
)
from .cycle import ExperimentCycleRunner, GenerationOutcome
from .failures import FailureRecord
from .models import ExperimentResult, Goal
from .repair_candidates import RepairVariant
from .repair_requests import RepairSourceProvider
from .recursive_trace import RecursiveRepairTraceStore


class RecursiveRepairStopReason(str, Enum):
    PROMOTED = "promoted"
    MAX_DEPTH = "max_depth"
    NO_PROGRESS = "no_progress"
    REPEATED_FAILURE = "repeated_failure"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_ADMISSIBLE_CANDIDATE = "no_admissible_candidate"
    NO_REPAIRABLE_DIAGNOSTIC = "no_repairable_diagnostic"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class RecursiveRepairPolicy:
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
    session_id: str | None = None
    starting_depth: int = 0

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
        return self.starting_depth + len(self.hops)

    @property
    def new_hops(self) -> int:
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


def _portable_number(value: float | None) -> float | str | None:
    if value is None:
        return None
    if math.isfinite(value):
        return float(value)
    return "inf" if value > 0 else "-inf"


def _generation_candidate_ids(generation: GenerationOutcome) -> tuple[str, ...]:
    return tuple(candidate.experiment_id for candidate in generation.candidates)


def _policy_payload(policy: RecursiveRepairPolicy) -> dict[str, Any]:
    return {
        "max_depth": policy.max_depth,
        "min_score_improvement": policy.min_score_improvement,
        "max_failure_signature_occurrences": policy.max_failure_signature_occurrences,
        "replay_ratio": policy.replay_ratio,
    }


def _variant_metadata(variants: tuple[RepairVariant, ...]) -> list[dict[str, Any]]:
    return [
        {
            "name": variant.name,
            "estimated_gpu_hours": variant.estimated_gpu_hours,
            "training_patch": dict(variant.training_patch),
            "lora_patch": dict(variant.lora_patch),
            "expected_deltas": dict(variant.expected_deltas),
        }
        for variant in variants
    ]


def _goal_snapshot(goal: Goal) -> dict[str, Any]:
    return {
        "metrics": [
            {
                "name": target.name,
                "minimum": target.minimum,
                "maximum": target.maximum,
                "weight": target.weight,
                "regression_tolerance": target.regression_tolerance,
                "direction": target.direction.value,
            }
            for target in goal.metrics
        ],
        "gpu_hour_budget": goal.gpu_hour_budget,
        "max_parallel_candidates": goal.max_parallel_candidates,
        "minimum_promotion_gain": goal.minimum_promotion_gain,
        "require_protocol_match": goal.require_protocol_match,
    }


def _result_snapshot(result: ExperimentResult) -> dict[str, Any]:
    return {
        "experiment_id": result.experiment_id,
        "metrics": dict(result.metrics),
        "gpu_hours": result.gpu_hours,
        "artifact_ref": result.artifact_ref,
        "evidence": dict(result.evidence),
    }


def _engine_snapshot(runner: ExperimentCycleRunner) -> dict[str, Any]:
    return {
        "goal": _goal_snapshot(runner.engine.goal),
        "baseline": _result_snapshot(runner.engine.baseline),
    }


def _executor_name(executor: object) -> str:
    return str(getattr(executor, "name", type(executor).__name__))


def _execution_snapshot(runner: ExperimentCycleRunner) -> dict[str, Any]:
    return {
        "version": 1,
        "base_config": dict(runner.base_config),
        "seed": int(runner.context.seed),
        "trainer_name": _executor_name(runner.trainer),
        "evaluator_name": _executor_name(runner.evaluator),
        "hardware": asdict(runner.context.hardware),
    }


def _validated_variants(variants: Iterable[RepairVariant]) -> tuple[RepairVariant, ...]:
    rows = tuple(variants)
    if not rows:
        raise ValueError("bounded recursive repair requires at least one variant")
    if any(variant.lora_patch for variant in rows):
        raise ValueError("bounded continuation repair cannot change LoRA topology")
    return rows


def _execute_bounded_loop(
    *,
    runner: ExperimentCycleRunner,
    current_generation: GenerationOutcome,
    provider: RepairSourceProvider,
    variants: tuple[RepairVariant, ...],
    policy: RecursiveRepairPolicy,
    session_id: str,
    starting_depth: int,
    signature_counts: Mapping[str, int],
    previous_target_score: float | None,
    begin_session: bool,
    recovery_claim_token: str | None = None,
) -> RecursiveRepairOutcome:
    if starting_depth < 0 or starting_depth > policy.max_depth:
        raise ValueError("starting_depth is outside recursive repair policy")
    if not session_id.strip():
        raise ValueError("recursive repair session_id is required")
    counts = dict(signature_counts)
    if any(
        not isinstance(key, str)
        or len(key) != 64
        or not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        for key, value in counts.items()
    ):
        raise ValueError("recursive repair signature checkpoint is invalid")

    initial_generation = current_generation
    current = current_generation
    new_hops: list[RecursiveRepairHop] = []
    previous_score = previous_target_score
    store = (
        RecursiveRepairTraceStore(runner.registry.path)
        if runner.registry is not None
        else None
    )
    if not begin_session and store is None:
        raise ValueError("resuming recursive repair requires a RunRegistry")
    if not begin_session and not recovery_claim_token:
        raise ValueError("resuming recursive repair requires a recovery claim token")
    session_started = not begin_session
    write_claim_token = recovery_claim_token if not begin_session else None

    def total_depth() -> int:
        return starting_depth + len(new_hops)

    def state_payload() -> dict[str, Any]:
        promoted_id = (
            current.promoted.experiment_id if current.promoted is not None else None
        )
        return {
            "depth_completed": total_depth(),
            "current_candidate_ids": list(_generation_candidate_ids(current)),
            "signature_counts": dict(sorted(counts.items())),
            "previous_target_score": _portable_number(previous_score),
            "remaining_budget": runner.engine.remaining_budget,
            "promoted_experiment_id": promoted_id,
        }

    def finish(
        reason: RecursiveRepairStopReason,
        detail: str,
    ) -> RecursiveRepairOutcome:
        if store is not None:
            store.finish(
                session_id=session_id,
                stop_reason=reason.value,
                stop_detail=detail,
                state=state_payload(),
                claim_token=write_claim_token,
            )
        return RecursiveRepairOutcome(
            initial_generation=initial_generation,
            final_generation=current,
            hops=tuple(new_hops),
            stop_reason=reason,
            stop_detail=detail,
            session_id=session_id,
            starting_depth=starting_depth,
        )

    try:
        if store is not None and begin_session:
            provider_name = str(getattr(provider, "name", type(provider).__name__))
            provider_version = str(getattr(provider, "version", "unknown"))
            store.begin(
                session_id=session_id,
                policy=_policy_payload(policy),
                metadata={
                    "provider_name": provider_name,
                    "provider_version": provider_version,
                    "variants": _variant_metadata(variants),
                    "baseline_experiment_id": runner.engine.baseline.experiment_id,
                    "engine_snapshot": _engine_snapshot(runner),
                    "execution_snapshot": _execution_snapshot(runner),
                },
                initial_candidate_ids=_generation_candidate_ids(current_generation),
                state=state_payload(),
            )
            session_started = True
        elif store is not None:
            existing = store.get_session(session_id)
            if existing is None or existing.get("status") != "running":
                raise ValueError("recursive repair resume session is not running")
            claim = store.get_recovery_claim(session_id)
            if claim is None or claim.get("claim_token") != recovery_claim_token:
                raise ValueError("recursive repair resume claim does not match")

        if current.promoted is not None:
            return finish(
                RecursiveRepairStopReason.PROMOTED,
                "current generation already contains a promoted candidate",
            )
        if runner.cancellation is not None and runner.cancellation.requested:
            return finish(
                RecursiveRepairStopReason.CANCELLED,
                "cancellation was requested",
            )
        if total_depth() >= policy.max_depth:
            return finish(
                RecursiveRepairStopReason.MAX_DEPTH,
                f"reached configured repair depth {policy.max_depth}",
            )

        for depth in range(starting_depth + 1, policy.max_depth + 1):
            if runner.engine.remaining_budget <= 0:
                return finish(
                    RecursiveRepairStopReason.BUDGET_EXHAUSTED,
                    "GPU-hour budget is exhausted",
                )
            # Checked again each iteration, not just once before the loop:
            # a hop that itself got cancelled mid-training/evaluation
            # returns a normal (not raised) rejected-candidate outcome, so
            # this is what turns that into a clean CANCELLED stop on the
            # next iteration instead of it falling through to
            # NO_REPAIRABLE_DIAGNOSTIC/REPEATED_FAILURE as if it were an
            # ordinary dead end.
            if runner.cancellation is not None and runner.cancellation.requested:
                return finish(
                    RecursiveRepairStopReason.CANCELLED,
                    "cancellation was requested",
                )

            target, signature, had_repairable = _select_novel_target(
                current,
                signature_counts=counts,
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
                return finish(reason, detail)

            target_score = _score_for(current, target.candidate.experiment_id)
            gain: float | None = None
            if previous_score is not None:
                gain = _score_improvement(previous_score, target_score)
                if gain < policy.min_score_improvement:
                    return finish(
                        RecursiveRepairStopReason.NO_PROGRESS,
                        (
                            f"best novel rejected candidate improved by {gain:.6g}; "
                            f"minimum required is {policy.min_score_improvement:.6g}"
                        ),
                    )

            counts[signature] = counts.get(signature, 0) + 1
            try:
                one_hop = run_single_hop_autonomous_repair(
                    runner=runner,
                    source_generation=current,
                    provider=provider,
                    variants=variants,
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
                    return finish(_budget_stop_reason(runner), message)
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
            new_hops.append(hop)
            current = one_hop.repair_generation
            previous_score = target_score

            if store is not None:
                produced_ids = _generation_candidate_ids(current)
                promoted_id = (
                    current.promoted.experiment_id
                    if current.promoted is not None
                    else None
                )
                store.record_hop(
                    session_id=session_id,
                    depth=depth,
                    target_experiment_id=target.candidate.experiment_id,
                    failure_signature=signature,
                    target_score=target_score,
                    score_improvement=gain,
                    remaining_budget_after=runner.engine.remaining_budget,
                    produced_candidate_ids=produced_ids,
                    promoted_experiment_id=promoted_id,
                    state=state_payload(),
                    claim_token=write_claim_token,
                )

            if current.promoted is not None:
                return finish(
                    RecursiveRepairStopReason.PROMOTED,
                    f"repair hop {depth} produced a promoted candidate",
                )

        return finish(
            RecursiveRepairStopReason.MAX_DEPTH,
            f"reached configured repair depth {policy.max_depth}",
        )
    except Exception as exc:
        if store is not None and session_started:
            try:
                store.fail(
                    session_id=session_id,
                    error_detail=f"{type(exc).__name__}: {exc}",
                    state=state_payload(),
                    claim_token=write_claim_token,
                )
            except Exception:
                pass
        raise
    finally:
        if store is not None:
            store.close()


def run_bounded_autonomous_repair(
    *,
    runner: ExperimentCycleRunner,
    source_generation: GenerationOutcome,
    provider: RepairSourceProvider,
    variants: Iterable[RepairVariant],
    policy: RecursiveRepairPolicy = RecursiveRepairPolicy(),
) -> RecursiveRepairOutcome:
    variant_rows = _validated_variants(variants)
    return _execute_bounded_loop(
        runner=runner,
        current_generation=source_generation,
        provider=provider,
        variants=variant_rows,
        policy=policy,
        session_id=uuid4().hex,
        starting_depth=0,
        signature_counts={},
        previous_target_score=None,
        begin_session=True,
    )


def _resume_bounded_autonomous_repair_from_checkpoint(
    *,
    runner: ExperimentCycleRunner,
    current_generation: GenerationOutcome,
    provider: RepairSourceProvider,
    variants: Iterable[RepairVariant],
    policy: RecursiveRepairPolicy,
    session_id: str,
    starting_depth: int,
    signature_counts: Mapping[str, int],
    previous_target_score: float | None,
    recovery_claim_token: str,
) -> RecursiveRepairOutcome:
    variant_rows = _validated_variants(variants)
    return _execute_bounded_loop(
        runner=runner,
        current_generation=current_generation,
        provider=provider,
        variants=variant_rows,
        policy=policy,
        session_id=session_id,
        starting_depth=starting_depth,
        signature_counts=signature_counts,
        previous_target_score=previous_target_score,
        begin_session=False,
        recovery_claim_token=recovery_claim_token,
    )
