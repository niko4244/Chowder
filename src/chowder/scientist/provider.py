"""ScientistProvider: the pluggable research-intelligence boundary.

Chowder does not generate scientific hypotheses itself — this is a pluggable
boundary, mirroring `TrainingDataProvider` and `HypothesisGenerator`. A
provider sees only the sanitized `ResearchContext` Chowder exports, and
returns typed data objects. It never sees candidate evaluation content,
protected evaluation content, registry internals, or policy documents, and it
can never write anything: every returned object crosses back through Chowder's
admission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .findings import ResearchFinding
from .hypothesis import Hypothesis
from .mission import ResearchMission
from .observation import ExperimentObservation
from .proposal import ExperimentProposal


@dataclass(frozen=True)
class CarriedEvidence:
    """A prior-evidence fact exported to the provider — always marked carried
    so prior evidence can never masquerade as a fresh measurement (T8)."""

    statement: str
    source_run_ids: tuple[str, ...]
    capability: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "source_run_ids": list(self.source_run_ids),
            "capability": self.capability,
            "carried": True,
        }


@dataclass(frozen=True)
class SkillSummary:
    """One skill's exported estimate (UNKNOWN stays None, never zero)."""

    skill: str
    estimate: float | None
    confidence: float
    uncertainty: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "estimate": self.estimate,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True)
class ResearchContext:
    """The sanitized world-view a provider is allowed to see."""

    mission: ResearchMission
    skill_estimates: tuple[SkillSummary, ...]
    open_failure_categories: tuple[dict[str, Any], ...]   # category + count, no content
    attempted_mechanisms: tuple[dict[str, Any], ...]      # mechanism + outcome + run ids
    carried_evidence: tuple[CarriedEvidence, ...]
    remaining_gpu_hours: float
    architecture: str = ""                                # family metadata only

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission": self.mission.to_dict(),
            "skill_estimates": [s.to_dict() for s in self.skill_estimates],
            "open_failure_categories": [dict(c) for c in self.open_failure_categories],
            "attempted_mechanisms": [dict(m) for m in self.attempted_mechanisms],
            "carried_evidence": [c.to_dict() for c in self.carried_evidence],
            "remaining_gpu_hours": self.remaining_gpu_hours,
            "architecture": self.architecture,
        }


@dataclass(frozen=True)
class ProviderUnavailability(Exception):
    """Raised by providers that cannot run (missing runtime, no key, no
    home). Chowder refuses loudly rather than fabricating research."""

    provider: str
    reason: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"provider {self.provider!r} unavailable: {self.reason}"


@runtime_checkable
class ScientistProvider(Protocol):
    """The research-intelligence seam.

    `propose_hypotheses` turns an exported context into falsifiable
    hypotheses. `propose_experiments` turns hypotheses into typed proposals
    that Chowder must admit. `interpret` turns grounded observations into a
    proposed finding — Chowder still decides the claim statuses. `export_state`
    carries provider-internal research state (e.g. a BFTS journal) across
    restarts as opaque data.
    """

    name: str

    def available(self) -> bool:
        """Whether this provider can run at all (runtime present, key set)."""
        ...

    def propose_hypotheses(self, context: ResearchContext,
                           *, count: int = 3) -> tuple[Hypothesis, ...]:
        ...

    def propose_experiments(self, context: ResearchContext,
                            hypotheses: tuple[Hypothesis, ...]) -> tuple[ExperimentProposal, ...]:
        ...

    def interpret(
        self,
        context: ResearchContext,
        hypothesis: Hypothesis,
        observations: tuple[ExperimentObservation, ...],
    ) -> ResearchFinding:
        ...

    def export_state(self) -> dict[str, Any]:
        """Opaque provider-internal state for restart; Chowder stores it but
        never interprets it."""
        ...
