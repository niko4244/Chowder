"""Research knowledge base: recorded results with applicability to our scale.

The KB exists so curriculum and recipe decisions can cite *evidence* rather
than folklore. Every entry records the experiment's scale and compute so a
671B/H100 result is never silently transplanted into a 9B/one-GPU setting;
``applicable_to`` forces the caller to justify relevance at our scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TOPIC_SFT = "sft"
TOPIC_PREFERENCE = "preference"
TOPIC_REJECTION_SAMPLING = "rejection_sampling"
TOPIC_REASONING_DATA = "reasoning_data"
TOPIC_CURRICULUM = "curriculum"
TOPIC_SELF_TRAINING = "self_training"
TOPIC_DISTILLATION = "distillation"
TOPIC_SYNTHETIC = "synthetic"
TOPIC_DATA_QUALITY = "data_quality"
TOPIC_FORGETTING = "catastrophic_forgetting"
TOPIC_CONTINUAL = "continual_learning"
TOPIC_MERGING = "model_merging"
TOPIC_LORA = "lora"
TOPIC_POST_TRAINING = "post_training"
TOPIC_TEST_TIME = "test_time_compute"
TOPIC_AGENTIC = "agentic_training"
TOPIC_SELF_IMPROVEMENT = "self_improvement"

ALL_TOPICS: tuple[str, ...] = (
    TOPIC_SFT,
    TOPIC_PREFERENCE,
    TOPIC_REJECTION_SAMPLING,
    TOPIC_REASONING_DATA,
    TOPIC_CURRICULUM,
    TOPIC_SELF_TRAINING,
    TOPIC_DISTILLATION,
    TOPIC_SYNTHETIC,
    TOPIC_DATA_QUALITY,
    TOPIC_FORGETTING,
    TOPIC_CONTINUAL,
    TOPIC_MERGING,
    TOPIC_LORA,
    TOPIC_POST_TRAINING,
    TOPIC_TEST_TIME,
    TOPIC_AGENTIC,
    TOPIC_SELF_IMPROVEMENT,
)


@dataclass(frozen=True)
class ResearchEntry:
    """One recorded research result with transferability metadata."""

    entry_id: str
    citation: str
    claim: str
    topics: tuple[str, ...]
    model_scale_b: float  # parameters in billions (e.g. 671.0)
    data_scale_tokens: int | None
    compute: str  # e.g. "H100 x512, 2 weeks"
    result: str
    limitations: tuple[str, ...] = field(default_factory=tuple)
    url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "citation": self.citation,
            "claim": self.claim,
            "topics": list(self.topics),
            "model_scale_b": self.model_scale_b,
            "data_scale_tokens": self.data_scale_tokens,
            "compute": self.compute,
            "result": self.result,
            "limitations": list(self.limitations),
            "url": self.url,
        }


@dataclass(frozen=True)
class Applicability:
    """Whether a result transfers to Chowder's 9B/local-hardware setting."""

    entry_id: str
    applicable: bool
    confidence: str  # high | medium | low
    reason: str


# Chowder's target deployment scale.
OUR_SCALE_B = 9.0

# A result trained at more than this multiple of our scale is treated as
# requiring explicit justification before its recipe transfers.
SCALE_MULTIPLE_CAUTION = 8.0


def applicability(entry: ResearchEntry, *, our_scale_b: float = OUR_SCALE_B) -> Applicability:
    """Mechanical applicability screening with an honest reason."""
    reasons: list[str] = []
    confidence = "high"

    if entry.model_scale_b > our_scale_b * SCALE_MULTIPLE_CAUTION:
        reasons.append(
            f"model scale {entry.model_scale_b:g}B exceeds our {our_scale_b:g}B "
            f"deployment by more than {SCALE_MULTIPLE_CAUTION:g}x"
        )
        confidence = "low"
    elif entry.model_scale_b > our_scale_b:
        reasons.append(
            f"model scale {entry.model_scale_b:g}B is above our {our_scale_b:g}B target"
        )
        confidence = "medium"

    if "TPU" in entry.compute or "tpu" in entry.compute:
        reasons.append("compute used TPUs; our setting is a single consumer GPU")
        confidence = min(confidence, "low") if confidence == "high" else "low"

    if not reasons:
        reasons.append(
            f"scale {entry.model_scale_b:g}B is at or below our target and compute is transferable"
        )

    applicable = confidence in {"high", "medium"} or "scale" not in reasons[0]
    return Applicability(
        entry_id=entry.entry_id,
        applicable=applicable,
        confidence=confidence,
        reason="; ".join(reasons),
    )


class ResearchKB:
    """A small, append-only registry of research results."""

    def __init__(self) -> None:
        self._entries: dict[str, ResearchEntry] = {}

    def add(self, entry: ResearchEntry) -> ResearchEntry:
        if entry.entry_id in self._entries:
            raise ValueError(f"research entry already recorded: {entry.entry_id}")
        if not entry.topics:
            raise ValueError("research entry must declare at least one topic")
        unknown = [t for t in entry.topics if t not in ALL_TOPICS]
        if unknown:
            raise ValueError(f"unknown topics: {unknown}")
        self._entries[entry.entry_id] = entry
        return entry

    def get(self, entry_id: str) -> ResearchEntry:
        entry = self._entries.get(entry_id)
        if entry is None:
            raise KeyError(entry_id)
        return entry

    def by_topic(self, topic: str) -> tuple[ResearchEntry, ...]:
        return tuple(e for e in self._entries.values() if topic in e.topics)

    def entries(self) -> tuple[ResearchEntry, ...]:
        return tuple(self._entries.values())

    def applicable_to_our_scale(
        self, entry: ResearchEntry
    ) -> Applicability:
        """Screen an entry for transferability to Chowder's local setting."""
        return applicability(entry)

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [e.to_dict() for e in self._entries.values()]}


__all__ = [
    "ALL_TOPICS",
    "Applicability",
    "OUR_SCALE_B",
    "ResearchEntry",
    "ResearchKB",
    "applicability",
]
