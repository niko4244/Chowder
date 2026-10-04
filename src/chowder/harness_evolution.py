"""RRSI-inspired regularized selection for runtime harness evolution.

The model is frozen; the evolving object is the harness. This module keeps the
selection rules explicit and auditable: sparse proposals, leakage screening,
noise-aware acceptance, cost-aware acceptance, and structural pruning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class HarnessMetrics:
    score: float
    cost: float
    green_rate: float
    nonexistent_read_rate: float
    policy_tokens: float = 0.0


@dataclass(frozen=True)
class HarnessProposal:
    name: str
    component: str
    changes: tuple[str, ...]
    metrics: HarnessMetrics


def annealed_edit_budget(round_index: int, total_rounds: int, *, minimum: int = 1, maximum: int = 3) -> int:
    if total_rounds < 1 or not 0 <= round_index < total_rounds:
        raise ValueError("round_index must be within total_rounds")
    if minimum < 1 or maximum < minimum:
        raise ValueError("invalid edit budget bounds")
    progress = 0.5 * (1.0 + math.cos(math.pi * round_index / total_rounds))
    return min(maximum, max(minimum, math.ceil(minimum + (maximum - minimum) * progress)))


def leakage_free(proposal: HarnessProposal, forbidden_terms: Iterable[str]) -> bool:
    """Reject edits that encode task/entity-specific answers or paths."""
    text = " ".join((proposal.name, proposal.component, *proposal.changes)).lower()
    return not any(term.lower() in text for term in forbidden_terms)


def accept_candidate(
    incumbent: HarnessMetrics,
    candidate: HarnessMetrics,
    *,
    noise_band: float = 0.0,
    beta0: float = 0.10,
    beta1: float = 0.50,
) -> tuple[bool, str]:
    """Conservatively accept only gains that pay for their extra cost."""
    delta_score = candidate.score - incumbent.score
    if delta_score <= noise_band:
        return False, "gain is within the empirical noise band"
    if incumbent.cost <= 0:
        return False, "incumbent cost must be positive"
    delta_cost_ratio = (candidate.cost - incumbent.cost) / incumbent.cost
    if delta_cost_ratio > beta0 + beta1 * delta_score:
        return False, "cost growth is not justified by score gain"
    if candidate.nonexistent_read_rate > incumbent.nonexistent_read_rate:
        return False, "candidate regresses nonexistent-read safety"
    return True, "accepted: regularized gain survives safety and cost gates"


def productive_components(history: Iterable[HarnessProposal], *, window: int = 5) -> tuple[str, ...]:
    """Keep components with a positive accepted score delta in the window."""
    rows = list(history)[-window:]
    deltas: dict[str, float] = {}
    for row in rows:
        deltas[row.component] = deltas.get(row.component, 0.0) + row.metrics.score
    return tuple(sorted(component for component, delta in deltas.items() if delta > 0))


def metrics_from_benchmark(result: Mapping[str, Any]) -> HarnessMetrics:
    """Project a run_live_benchmark result onto the selector's metric view."""
    metrics = result["metrics"]
    return HarnessMetrics(
        score=float(metrics["runtime_reward"]),
        cost=float(metrics.get("runtime_execution_cost", metrics.get("policy_tokens", 0.0))),
        green_rate=float(metrics["runtime_green_rate"]),
        nonexistent_read_rate=float(metrics["runtime_nonexistent_read_rate"]),
        policy_tokens=float(metrics.get("policy_tokens", 0.0)),
    )


def result_digest(metrics: HarnessMetrics) -> Mapping[str, float]:
    return {
        "score": metrics.score,
        "cost": metrics.cost,
        "green_rate": metrics.green_rate,
        "nonexistent_read_rate": metrics.nonexistent_read_rate,
        "policy_tokens": metrics.policy_tokens,
    }


def select_first_round(
    incumbent: HarnessMetrics,
    candidate: HarnessMetrics,
    *,
    evolve_incumbent: HarnessMetrics,
    evolve_candidate: HarnessMetrics,
    heldout_incumbent: HarnessMetrics,
    heldout_candidate: HarnessMetrics,
    proposal: HarnessProposal,
    forbidden_terms: Iterable[str],
    round_index: int = 0,
    total_rounds: int = 4,
    noise_band: float = 0.0,
    beta0: float = 0.10,
    beta1: float = 0.50,
) -> dict[str, object]:
    """Evaluate one sparse proposal on evolve and untouched held-out tasks."""
    budget = annealed_edit_budget(round_index, total_rounds)
    if len(proposal.changes) > budget:
        return {"accepted": False, "reason": "proposal exceeds annealed edit budget", "budget": budget}
    if not leakage_free(proposal, forbidden_terms):
        return {"accepted": False, "reason": "leakage screening rejected task-specific edit", "budget": budget}
    evolve_ok, evolve_reason = accept_candidate(
        evolve_incumbent, evolve_candidate, noise_band=noise_band, beta0=beta0, beta1=beta1
    )
    if not evolve_ok:
        return {"accepted": False, "reason": "evolve split: " + evolve_reason, "budget": budget}
    transfer_delta = heldout_candidate.score - heldout_incumbent.score
    if transfer_delta < -noise_band:
        return {
            "accepted": False,
            "reason": "held-out transfer regressed",
            "budget": budget,
            "transfer_delta": transfer_delta,
        }
    if heldout_candidate.nonexistent_read_rate > heldout_incumbent.nonexistent_read_rate:
        return {"accepted": False, "reason": "held-out safety regressed", "budget": budget}
    return {
        "accepted": True,
        "reason": "accepted: evolve gain transferred without held-out regression",
        "budget": budget,
        "transfer_delta": transfer_delta,
        "evolve": result_digest(evolve_candidate),
        "heldout": result_digest(heldout_candidate),
        "incumbent": result_digest(incumbent),
        "candidate": result_digest(candidate),
    }
