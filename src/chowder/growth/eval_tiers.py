"""Evaluation tiers: how much evaluation a candidate deserves.

Running every expensive benchmark every training iteration would burn the
local GPU-hour budget on ceremony. Tiers bound cost by decision stakes:

- TIER_0_SMOKE: broken tokenizer / lost formatting / adapter not applied.
- TIER_1_TARGETED: benchmarks tied directly to the current curriculum.
- TIER_2_PROTECTED: the stable battery that must not regress.
- TIER_3_BROAD: all major applicable skill domains.
- TIER_4_FRONTIER: expensive suites, only for serious promotion candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

TIER_0_SMOKE = "TIER_0_SMOKE"
TIER_1_TARGETED = "TIER_1_TARGETED"
TIER_2_PROTECTED = "TIER_2_PROTECTED"
TIER_3_BROAD = "TIER_3_BROAD"
TIER_4_FRONTIER = "TIER_4_FRONTIER"

TIER_ORDER = {
    TIER_0_SMOKE: 0,
    TIER_1_TARGETED: 1,
    TIER_2_PROTECTED: 2,
    TIER_3_BROAD: 3,
    TIER_4_FRONTIER: 4,
}


@dataclass(frozen=True)
class EvalPlan:
    tier: str
    benchmarks: tuple[str, ...]
    estimated_gpu_hours: float
    rationale: str


def plan_eval_tier(
    *,
    stage: str,
    benchmarks_by_tier: Mapping[str, Sequence[str]],
    gpu_hours_per_benchmark: Mapping[str, float],
    budget_gpu_hours: float,
    targeted_benchmarks: Sequence[str] = (),
) -> EvalPlan:
    """Choose the highest tier that fits the remaining budget.

    ``stage``: "mid_training_checkpoint" (tiers 0-1 only), "candidate_screen"
    (tiers 0-2), "promotion_candidate" (all tiers).
    """
    allowed: dict[int, str] = {
        "mid_training_checkpoint": 1,
        "candidate_screen": 2,
        "promotion_candidate": 4,
    }
    max_rank = allowed.get(stage)
    if max_rank is None:
        raise ValueError(f"unknown eval stage: {stage}")

    # Always include tier 0 smoke benchmarks when they fit.
    ordered_tiers = [TIER_0_SMOKE, TIER_1_TARGETED, TIER_2_PROTECTED, TIER_3_BROAD, TIER_4_FRONTIER]
    selected: list[str] = []
    estimated = 0.0
    tier_reached = TIER_0_SMOKE
    for tier in ordered_tiers:
        if TIER_ORDER[tier] > max_rank:
            break
        if tier == TIER_1_TARGETED and targeted_benchmarks:
            tier_benchmarks = tuple(targeted_benchmarks)
        else:
            tier_benchmarks = tuple(benchmarks_by_tier.get(tier, ()))
        for benchmark_id in tier_benchmarks:
            cost = gpu_hours_per_benchmark.get(benchmark_id, 0.0)
            if estimated + cost > budget_gpu_hours:
                continue
            selected.append(benchmark_id)
            estimated += cost
        if tier_benchmarks:
            tier_reached = tier

    rationale = {
        "mid_training_checkpoint": "cheap catch of catastrophic failures mid-training",
        "candidate_screen": "protected battery screens out candidates before expensive evals",
        "promotion_candidate": "full battery: only serious candidates reach tier 4",
    }[stage]
    if tier_reached == TIER_0_SMOKE and not selected:
        rationale = "budget too small for any benchmark; refusing to fake coverage"
    return EvalPlan(
        tier=tier_reached,
        benchmarks=tuple(selected),
        estimated_gpu_hours=round(estimated, 4),
        rationale=rationale,
    )
