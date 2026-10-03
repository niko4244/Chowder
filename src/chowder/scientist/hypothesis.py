"""Hypothesis: the reason an experiment exists.

A hypothesis is falsifiable research, not a hunch: it records the observation
that motivated it, the suspected mechanism, a predicted effect, the evidence
for and against it, and — critically — its own falsification conditions. The
director refuses experiments proposed without one ("no hypothesis, just try
LR 2e-5" is not scientific research).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ResearchQuestion:
    """The capability question a hypothesis answers; not a benchmark name."""

    text: str
    capability: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("a research question needs a question")
        if not self.capability.strip():
            raise ValueError("a research question names the capability it probes")

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "capability": self.capability}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchQuestion":
        unknown = sorted(set(data) - {"text", "capability"})
        if unknown:
            raise ValueError(f"unknown research-question keys: {unknown}")
        return cls(text=str(data["text"]), capability=str(data["capability"]))


@dataclass(frozen=True)
class Hypothesis:
    """A falsifiable proposed answer to a research question."""

    hypothesis_id: str
    research_question: ResearchQuestion
    observation: str
    suspected_mechanism: str
    predicted_effect: str
    novelty_basis: str = ""
    supporting_evidence: tuple[str, ...] = ()   # run_ids / finding ids
    contradicting_evidence: tuple[str, ...] = ()
    uncertainty: str = "medium"                 # low | medium | high
    falsification_conditions: tuple[str, ...] = ()
    proposed_experiments: tuple[str, ...] = ()  # proposal ids, set by director
    expected_information_gain: float = 0.0
    status: str = "open"                        # open | supported | rejected | withdrawn
    provider: str = ""                          # provenance: which provider proposed it
    mission_id: str = ""

    UNCERTAINTY_LEVELS = ("low", "medium", "high")
    STATUSES = ("open", "supported", "rejected", "withdrawn")

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            raise ValueError("hypothesis_id is required")
        for field_name in ("observation", "suspected_mechanism", "predicted_effect"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"a hypothesis must state its {field_name}")
        if self.uncertainty not in self.UNCERTAINTY_LEVELS:
            raise ValueError(f"unknown uncertainty: {self.uncertainty}")
        if self.status not in self.STATUSES:
            raise ValueError(f"unknown hypothesis status: {self.status}")
        if not self.falsification_conditions:
            raise ValueError(
                "a hypothesis without falsification conditions is not falsifiable; "
                "state what evidence would refute it"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "research_question": self.research_question.to_dict(),
            "observation": self.observation,
            "suspected_mechanism": self.suspected_mechanism,
            "predicted_effect": self.predicted_effect,
            "novelty_basis": self.novelty_basis,
            "supporting_evidence": list(self.supporting_evidence),
            "contradicting_evidence": list(self.contradicting_evidence),
            "uncertainty": self.uncertainty,
            "falsification_conditions": list(self.falsification_conditions),
            "proposed_experiments": list(self.proposed_experiments),
            "expected_information_gain": self.expected_information_gain,
            "status": self.status,
            "provider": self.provider,
            "mission_id": self.mission_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Hypothesis":
        _KEYS = (
            "hypothesis_id", "research_question", "observation", "suspected_mechanism",
            "predicted_effect", "novelty_basis", "supporting_evidence",
            "contradicting_evidence", "uncertainty", "falsification_conditions",
            "proposed_experiments", "expected_information_gain", "status", "provider",
            "mission_id",
        )
        unknown = sorted(set(data) - set(_KEYS))
        if unknown:
            raise ValueError(f"unknown hypothesis keys (fail-closed): {unknown}")
        missing = [k for k in ("hypothesis_id", "research_question", "observation",
                               "suspected_mechanism", "predicted_effect") if k not in data]
        if missing:
            raise ValueError(f"hypothesis is missing required keys: {missing}")
        return cls(
            hypothesis_id=str(data["hypothesis_id"]),
            research_question=ResearchQuestion.from_dict(dict(data["research_question"])),
            observation=str(data["observation"]),
            suspected_mechanism=str(data["suspected_mechanism"]),
            predicted_effect=str(data["predicted_effect"]),
            novelty_basis=str(data.get("novelty_basis", "")),
            supporting_evidence=tuple(str(e) for e in data.get("supporting_evidence", ())),
            contradicting_evidence=tuple(str(e) for e in data.get("contradicting_evidence", ())),
            uncertainty=str(data.get("uncertainty", "medium")),
            falsification_conditions=tuple(
                str(f) for f in data.get("falsification_conditions", ())
            ),
            proposed_experiments=tuple(str(p) for p in data.get("proposed_experiments", ())),
            expected_information_gain=float(data.get("expected_information_gain", 0.0)),
            status=str(data.get("status", "open")),
            provider=str(data.get("provider", "")),
            mission_id=str(data.get("mission_id", "")),
        )
