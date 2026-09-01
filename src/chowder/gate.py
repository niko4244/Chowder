from __future__ import annotations

from .models import GateDecision, Goal, ExperimentResult


def evaluate_candidate(
    *,
    goal: Goal,
    baseline: ExperimentResult,
    candidate: ExperimentResult,
) -> GateDecision:
    """Evaluate a candidate for *promotion*, not final-goal completion.

    Final targets describe where evolution should end. They must not prevent
    incremental, non-regressing improvements from becoming the next baseline.
    """
    regressions: dict[str, float] = {}
    unmet: list[str] = []
    missing: list[str] = []
    weighted_gain = 0.0
    weight_total = 0.0

    for target in goal.metrics:
        if target.name not in candidate.metrics:
            missing.append(target.name)
            continue

        value = float(candidate.metrics[target.name])
        if target.name not in baseline.metrics:
            missing.append(f"baseline:{target.name}")
            continue

        base_value = float(baseline.metrics[target.name])
        utility_delta = target.utility_delta(base_value, value)
        weighted_gain += utility_delta * target.weight
        weight_total += target.weight

        if utility_delta < -target.regression_tolerance:
            regressions[target.name] = utility_delta
        if not target.target_met(value):
            unmet.append(target.name)

    score = weighted_gain / weight_total if weight_total else float("-inf")
    accepted = not regressions and not missing and score > goal.minimum_promotion_gain
    goal_met = not unmet and not missing

    if missing:
        reason = "rejected: evaluation evidence is incomplete"
    elif regressions:
        reason = "rejected: regression tolerance exceeded"
    elif score <= goal.minimum_promotion_gain:
        reason = "rejected: candidate did not improve enough"
    elif goal_met:
        reason = "accepted: improved without protected regressions and final goal is met"
    else:
        reason = "accepted: incremental improvement without protected regressions"

    return GateDecision(
        accepted=accepted,
        score=score,
        regressions=regressions,
        unmet_targets=tuple(unmet),
        missing_metrics=tuple(missing),
        goal_met=goal_met,
        reason=reason,
    )
