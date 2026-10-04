from __future__ import annotations

from .models import GateDecision, Goal, ExperimentResult
from .protocol import result_protocol_fingerprint


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

    if goal.require_protocol_match:
        baseline_protocol = result_protocol_fingerprint(baseline.evidence)
        candidate_protocol = result_protocol_fingerprint(candidate.evidence)
        if baseline_protocol is None:
            missing.append("baseline:evaluation_protocol")
        if candidate_protocol is None:
            missing.append("evaluation_protocol")
        if (
            baseline_protocol is not None
            and candidate_protocol is not None
            and baseline_protocol != candidate_protocol
        ):
            return GateDecision(
                accepted=False,
                score=float("-inf"),
                regressions={},
                unmet_targets=(),
                missing_metrics=(),
                goal_met=False,
                reason="rejected: evaluation protocol does not match baseline",
            )

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

    # Runtime safety gates are hard vetoes, not weighted objectives.  Missing
    # evidence is deliberately a veto too: a text score cannot certify that a
    # model actually observed a green repair or avoided nonexistent reads.
    runtime_reasons: list[str] = []
    runtime_checks = (
        ("runtime_reward", "runtime_reward_min", "min"),
        ("runtime_nonexistent_read_rate", "runtime_nonexistent_read_rate_max", "max"),
    )
    for metric_name, attribute, bound in runtime_checks:
        threshold = getattr(goal, attribute)
        if threshold is None:
            continue
        if metric_name not in candidate.metrics:
            missing.append(metric_name)
            runtime_reasons.append(f"{metric_name}:missing")
            continue
        value = float(candidate.metrics[metric_name])
        passed = value >= threshold if bound == "min" else value <= threshold
        if not passed:
            runtime_reasons.append(f"{metric_name}:{value:g}>{threshold:g}" if bound == "max" else f"{metric_name}:{value:g}<{threshold:g}")

    score = weighted_gain / weight_total if weight_total else float("-inf")
    accepted = not regressions and not missing and not runtime_reasons and score > goal.minimum_promotion_gain
    goal_met = not unmet and not missing

    if missing:
        reason = "rejected: evaluation evidence is incomplete"
    elif runtime_reasons:
        reason = "rejected: runtime safety gate failed: " + ", ".join(runtime_reasons)
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
