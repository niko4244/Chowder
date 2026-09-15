"""Chowder capability taxonomy and profile aggregation.

The growth system measures a model, aggregates raw benchmark measurements
into a structured skill profile, and keeps every raw number visible behind
the estimates -- raw benchmark scores are never hidden behind skill names
and never silently overridden by them.

Skills are dotted paths, two levels deep:
``reasoning.abstract``, ``math.olympiad``, ``coding.debugging``, ...
``capability.ALL_SKILLS`` is the canonical closed list; a benchmark mapping
to an unknown skill is a registry error, not a silent invention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

SKILL_ROOTS = (
    "reasoning",
    "math",
    "coding",
    "tools",
    "research",
    "knowledge",
    "instruction",
    "context",
    "agents",
    "self_improvement",
    "multilingual",
    "multimodal",
    "professional",
    "science",
    "health",
    "safety",
)

_SKILL_LEAVES: dict[str, tuple[str, ...]] = {
        "reasoning": ("abstract", "scientific", "causal"),
        "math": ("arithmetic", "algebra", "competition", "olympiad", "proof"),
        "coding": ("generation", "debugging", "repo", "agentic", "efficiency"),
        "tools": ("function_calling", "multi_turn", "state_handling", "recovery"),
        "research": ("browsing", "citation", "synthesis", "conflict_resolution"),
        "knowledge": ("factuality", "calibration", "abstention"),
        "instruction": ("multi_constraint", "formatting", "persistence"),
        "context": ("retrieval", "synthesis", "distractor_resistance", "retention"),
        "agents": ("planning", "recovery", "long_horizon", "terminal_work"),
        "self_improvement": (
            "debug_training",
            "design_experiment",
            "instrument_repair",
            "contamination_detection",
        ),
        "multilingual": ("comprehension", "generation"),
        "multimodal": ("image", "document", "video"),
        "professional": ("analysis", "documents", "workflows"),
        "science": ("recall", "quantitative", "experimental_design", "literature"),
        "health": ("knowledge", "reasoning"),
        "safety": ("robustness", "refusal_calibration", "injection_resistance", "honesty"),
    }

ALL_SKILLS: tuple[str, ...] = tuple(
    f"{root}.{leaf}" for root, leaves in _SKILL_LEAVES.items() for leaf in leaves
)

_SKILL_SET = frozenset(ALL_SKILLS)


def assert_known_skills(skills: tuple[str, ...], *, context: str) -> None:
    """Refuse an unknown skill path: a mapping to a skill outside the closed
    list is a registry authoring error, not a new capability invention."""
    for skill in skills:
        if skill not in _SKILL_SET:
            raise ValueError(f"{context}: unknown skill {skill!r} (not in capability.ALL_SKILLS)")


@dataclass(frozen=True)
class SkillEstimate:
    """One skill's normalized estimate with its evidence trail."""

    skill: str
    estimate: float  # normalized 0..1
    confidence: float  # 0..1, evidence-weighted
    evidence: tuple[str, ...]  # benchmark IDs contributing

    def __post_init__(self) -> None:
        if not 0.0 <= self.estimate <= 1.0:
            raise ValueError(f"{self.skill}: estimate must be in [0, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"{self.skill}: confidence must be in [0, 1]")


@dataclass(frozen=True)
class CapabilityProfile:
    """A model's measured capability profile at one version.

    ``raw_scores`` keeps every underlying benchmark measurement visible
    (benchmark@version -> raw metric value in [0, 1] where possible).
    ``skills`` holds the normalized estimates. Neither hides the other.
    """

    model_version: str
    raw_scores: Mapping[str, float]
    skills: tuple[SkillEstimate, ...]
    unsupported: tuple[str, ...] = ()  # benchmarks N/A for this modality
    tainted: tuple[str, ...] = ()  # contaminated: shown raw, excluded from estimates
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for benchmark_id in self.tainted:
            if benchmark_id not in self.raw_scores and benchmark_id not in self.unsupported:
                raise ValueError(
                    f"tainted benchmark {benchmark_id!r} has no raw score or unsupported marker"
                )

    def skill(self, skill: str) -> SkillEstimate | None:
        for estimate in self.skills:
            if estimate.skill == skill:
                return estimate
        return None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapabilityProfile":
        """Rebuild a profile from :meth:`to_dict` output (e.g. an eval pass)."""
        return cls(
            model_version=data["model_version"],
            raw_scores=dict(data.get("raw_scores", {})),
            skills=tuple(
                SkillEstimate(
                    skill=s["skill"],
                    estimate=float(s["estimate"]),
                    confidence=float(s["confidence"]),
                    evidence=tuple(s.get("evidence", ())),
                )
                for s in data.get("skills", ())
            ),
            unsupported=tuple(data.get("unsupported", ())),
            tainted=tuple(data.get("tainted", ())),
            notes=dict(data.get("notes", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "raw_scores": dict(self.raw_scores),
            "skills": [
                {
                    "skill": s.skill,
                    "estimate": s.estimate,
                    "confidence": s.confidence,
                    "evidence": list(s.evidence),
                }
                for s in self.skills
            ],
            "unsupported": list(self.unsupported),
            "tainted": list(self.tainted),
            "notes": dict(self.notes),
        }


def build_profile(
    *,
    model_version: str,
    raw_scores: Mapping[str, float],
    skill_weights: Mapping[str, Mapping[str, float]],
    confidences: Mapping[str, float] | None = None,
    unsupported: tuple[str, ...] = (),
    tainted: tuple[str, ...] = (),
) -> CapabilityProfile:
    """Aggregate raw benchmark scores into skill estimates.

    ``skill_weights`` maps skill -> {benchmark_id: weight}. A benchmark score
    contributes to a skill estimate as a weighted mean; confidence combines
    declared benchmark confidence with evidence count.

    Tainted benchmarks still appear in ``raw_scores`` (raw measurements are
    never hidden) but are EXCLUDED from skill estimates -- a contaminated
    score must not drive training decisions.
    """
    confidences = confidences or {}
    tainted_set = set(tainted)
    per_skill: dict[str, list[tuple[float, float]]] = {}
    for skill, weights in skill_weights.items():
        pairs: list[tuple[float, float]] = []
        for benchmark_id, weight in weights.items():
            if benchmark_id in tainted_set:
                continue
            if benchmark_id in raw_scores:
                pairs.append((float(raw_scores[benchmark_id]), float(weight)))
        if pairs:
            total_weight = sum(w for _, w in pairs)
            per_skill[skill] = pairs

    estimates = []
    for skill in sorted(per_skill):
        pairs = per_skill[skill]
        total_weight = sum(w for _, w in pairs)
        estimate = sum(v * w for v, w in pairs) / total_weight
        declared = [float(confidences.get(b, 0.8)) for b, _ in pairs]
        confidence = min(1.0, (sum(declared) / len(declared)) * min(1.0, len(pairs) / 3.0))
        evidence = tuple(sorted(b for b, _ in pairs))
        estimates.append(
            SkillEstimate(
                skill=skill,
                estimate=estimate,
                confidence=confidence,
                evidence=evidence,
            )
        )

    return CapabilityProfile(
        model_version=model_version,
        raw_scores=dict(raw_scores),
        skills=tuple(estimates),
        unsupported=tuple(sorted(unsupported)),
        tainted=tuple(sorted(tainted_set)),
    )


def profile_delta(
    before: CapabilityProfile, after: CapabilityProfile
) -> dict[str, dict[str, float]]:
    """Per-skill change between two profiles, plus raw per-benchmark deltas.

    Missing skills on either side are omitted rather than assumed zero.
    """
    before_skills = {s.skill: s.estimate for s in before.skills}
    after_skills = {s.skill: s.estimate for s in after.skills}
    deltas: dict[str, dict[str, float]] = {"skills": {}, "raw": {}}
    for skill in sorted(set(before_skills) & set(after_skills)):
        deltas["skills"][skill] = after_skills[skill] - before_skills[skill]
    for benchmark_id in sorted(set(before.raw_scores) & set(after.raw_scores)):
        deltas["raw"][benchmark_id] = (
            after.raw_scores[benchmark_id] - before.raw_scores[benchmark_id]
        )
    return deltas


def frontier_parity(chowder: float, frontier: float) -> float | None:
    """chowder/frontier as a percentage where the scale makes it meaningful
    (both finite, frontier > 0). Returns None otherwise -- use standardized
    gaps instead of a fake ratio."""
    if frontier is None or chowder is None:
        return None
    frontier_f = float(frontier)
    chowder_f = float(chowder)
    if not math.isfinite(frontier_f) or not math.isfinite(chowder_f) or frontier_f <= 0.0:
        return None
    return chowder_f / frontier_f
