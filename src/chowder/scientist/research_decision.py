"""ResearchDecision: the closed decision vocabulary of the research loop.

After evidence lands, the director reaches exactly one of these decisions per
hypothesis/branch. `promote_candidate` is the one decision scientist mode can
*request* — and it is a request: the actual promotion verdict is reached only
by the existing production certification + promotion gate. Deciding "the
model is better" is not, and will never be, a research-layer authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DECISION_KINDS = (
    "expand_branch",
    "reject_hypothesis",
    "request_replication",
    "request_transfer",
    "promote_candidate",      # a request; production gate decides
    "investigate_anomaly",
    "stop_plateau",
    "stop_budget",
    "stop_no_admissible_hypothesis",
    "human_review",
)

TERMINAL_DECISIONS = frozenset({
    "stop_plateau", "stop_budget", "stop_no_admissible_hypothesis", "human_review",
})


@dataclass(frozen=True)
class ResearchDecision:
    kind: str
    subject_id: str            # hypothesis / branch / proposal the decision is about
    reason: str
    evidence_run_ids: tuple[str, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in DECISION_KINDS:
            raise ValueError(f"unknown decision kind: {self.kind}; known: {DECISION_KINDS}")
        if not self.subject_id:
            raise ValueError("a decision names its subject")
        if not self.reason.strip():
            raise ValueError("a decision states its reason")
        if self.kind == "promote_candidate" and not self.evidence_run_ids:
            raise ValueError(
                "a promotion request without evidence runs is not a request; "
                "the production gate would refuse it anyway"
            )

    @property
    def terminal(self) -> bool:
        return self.kind in TERMINAL_DECISIONS

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "subject_id": self.subject_id,
            "reason": self.reason,
            "evidence_run_ids": list(self.evidence_run_ids),
            "payload": dict(self.payload),
        }
