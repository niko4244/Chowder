"""The autonomous growth control plane: durable state, profiling, target choice.

The single-generation engine below this module (training, evaluation,
certification) is trusted and unchanged.  What was missing is the layer that
lets one generation *teach the next*: today a campaign constructs a fresh
in-memory ``FailureBank()`` every time, so nothing accumulates; the capability
profile is a flat mean smeared across every known skill; and the next target is
chosen by a human.

This module supplies the durable, evidence-attributed pieces:

* :class:`GrowthState` -- append-only learning memory (failures, interventions,
  target proposals, capability history, stopping state) that survives a process
  restart and is read back before the next generation is planned;
* :func:`build_skill_profile` -- benchmark -> skill attribution, so an estimate
  for a skill is computed **only** from the benchmarks that actually measure it,
  and a skill nobody measured stays *unknown* rather than zero;
* :func:`classify_intervention` -- whether a weakness is even the kind of thing
  the current training path can fix, from evidence and intervention history;
* :class:`NextTargetSelector` -- picks the next target from durable evidence
  with an explicit, inspectable score, and records *why not* the alternatives.

Nothing here can see a campaign's candidate results: the selector's inputs are
the parent's measured profile, memory and policy.  Protected benchmarks are
gates, never selection objectives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from chowder.evals.result import MEASURED_PARENT, MEASURED_THIS_GENERATION, UNMEASURED

from .failure_bank import FailureBank, FailureRecord

#: The measurement provenance that may contribute a capability estimate.  A
#: carried/grey row is reference context, not a measurement of this model.
_ESTIMATING_ORIGINS = frozenset({MEASURED_THIS_GENERATION, MEASURED_PARENT})

#: How a weakness should be treated.  Deciding "train it with SFT" for every
#: weakness is how an autonomous loop repeats the same useless experiment.
TREATMENT_CLASSES = (
    "targeted_repair",
    "sft",
    "continued_pretrain",
    "preference",
    "data_acquisition",
    "evaluation_needed",
    "architecture_research",
    "untrainable_with_current_path",
)

#: Treatment classes that may proceed to training without a human.
AUTONOMOUS_TREATMENTS = frozenset(
    {"targeted_repair", "sft", "continued_pretrain", "preference", "data_acquisition"}
)

#: Treatment classes that must not start a campaign without human review.
REVIEW_TREATMENTS = frozenset(
    {"evaluation_needed", "architecture_research", "untrainable_with_current_path"}
)

#: The smallest sample count at which a measurement is fully weighted.  Below
#: it, confidence scales with support rather than being assumed.
FULL_CONFIDENCE_SAMPLES = 16


# --------------------------------------------------------------------------
# benchmark -> skill profiling
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillEstimateEvidence:
    """One skill's estimate, with the evidence it was computed from.

    ``estimate is None`` means **unknown**, not zero: a skill nobody measured
    has no estimate.  ``confidence`` is 0 in that case, and ``uncertainty`` is
    1.  A downstream chooser is expected to read unknown as "measure, don't
    train".
    """

    skill: str
    estimate: float | None
    confidence: float
    uncertainty: float
    supporting_benchmarks: tuple[str, ...]
    benchmark_measurements: Mapping[str, float]
    n_measurements: int
    generation: str
    provenance: str
    aggregation: str

    @property
    def measured(self) -> bool:
        return self.estimate is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "estimate": self.estimate,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "supporting_benchmarks": list(self.supporting_benchmarks),
            "benchmark_measurements": dict(self.benchmark_measurements),
            "n_measurements": self.n_measurements,
            "generation": self.generation,
            "provenance": self.provenance,
            "aggregation": self.aggregation,
            "measured": self.measured,
        }


@dataclass(frozen=True)
class SkillProfile:
    """One generation's capability profile, by evidence-specific skill."""

    generation: str
    estimates: tuple[SkillEstimateEvidence, ...]
    notes: str = ""

    def for_skill(self, skill: str) -> SkillEstimateEvidence | None:
        for estimate in self.estimates:
            if estimate.skill == skill:
                return estimate
        return None

    @property
    def measured(self) -> tuple[SkillEstimateEvidence, ...]:
        return tuple(value for value in self.estimates if value.measured)

    @property
    def unmeasured(self) -> tuple[SkillEstimateEvidence, ...]:
        return tuple(value for value in self.estimates if not value.measured)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "estimates": [value.to_dict() for value in self.estimates],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SkillProfile":
        return cls(
            generation=str(data.get("generation", "")),
            estimates=tuple(
                SkillEstimateEvidence(
                    skill=str(item["skill"]),
                    estimate=(
                        None if item.get("estimate") is None else float(item["estimate"])
                    ),
                    confidence=float(item.get("confidence", 0.0)),
                    uncertainty=float(
                        item.get("uncertainty", 1.0 - float(item.get("confidence", 0.0)))
                    ),
                    supporting_benchmarks=tuple(item.get("supporting_benchmarks", ())),
                    benchmark_measurements=dict(item.get("benchmark_measurements", {})),
                    n_measurements=int(item.get("n_measurements", 0)),
                    generation=str(item.get("generation", "")),
                    provenance=str(item.get("provenance", "")),
                    aggregation=str(item.get("aggregation", "")),
                )
                for item in data.get("estimates", ())
            ),
            notes=str(data.get("notes", "")),
        )


def _support_confidence(run: Any, entry: Any) -> float:
    """How much one measurement should count, from its support and provenance."""
    if str(run.measurement_origin) not in _ESTIMATING_ORIGINS:
        return 0.0
    samples = int(getattr(run, "n_samples", 0) or 0)
    if samples <= 0:
        # An aggregate-only row is a real measurement with weak support, not a
        # fake one: it counts, at reduced confidence.
        support = 0.5
    else:
        support = min(1.0, samples / FULL_CONFIDENCE_SAMPLES)
    # A parent-side measurement is as real as the candidate's, but the campaign
    # measures the candidate, so it carries slightly less weight for planning.
    origin_weight = 1.0 if str(run.measurement_origin) == MEASURED_THIS_GENERATION else 0.9
    return support * origin_weight


def build_skill_profile(
    *,
    generation: str,
    runs: Sequence[Any],
    skills: Sequence[str] | None = None,
    registry: Any = None,
    aggregation: str = "support-weighted mean of the skill's own benchmarks",
) -> SkillProfile:
    """Attribute each measurement to the skills its benchmark declares.

    The estimate for a skill is computed only from the benchmarks that measure
    *that* skill (via the benchmark registry), weighted by each measurement's
    support and provenance.  A skill with no measurement keeps
    ``estimate=None`` -- unknown, never zero.  Different skills therefore get
    different estimates from the same benchmark set, which is the property the
    old flat mean destroyed.
    """
    from .capability import ALL_SKILLS
    from .catalog import default_registry

    registry = registry if registry is not None else default_registry()
    universe: list[str] = list(skills) if skills is not None else list(ALL_SKILLS)

    # skill -> benchmark -> (weighted score contribution, weight)
    per_skill: dict[str, dict[str, tuple[float, float]]] = {}
    provenance: dict[str, set[str]] = {}
    for run in runs:
        qualified_id = str(getattr(run, "benchmark_qualified_id", ""))
        entry = registry.get(qualified_id) if registry is not None else None
        if entry is None:
            # A benchmark the registry does not declare cannot be attributed to
            # a skill; it is skipped rather than smeared over every skill.
            continue
        weight = _support_confidence(run, entry)
        if weight <= 0.0:
            continue
        score = getattr(run, "score", None)
        if score is None:
            continue
        for skill in tuple(entry.skills):
            if skill not in per_skill:
                per_skill[skill] = {}
                provenance[skill] = set()
            bucket = per_skill[skill].setdefault(qualified_id, (0.0, 0.0))
            per_skill[skill][qualified_id] = (
                bucket[0] + float(score) * weight,
                bucket[1] + weight,
            )
            provenance[skill].add(str(getattr(run, "measurement_origin", "")))

    estimates: list[SkillEstimateEvidence] = []
    for skill in universe:
        bucket = per_skill.get(skill, {})
        if not bucket:
            estimates.append(
                SkillEstimateEvidence(
                    skill=skill,
                    estimate=None,
                    confidence=0.0,
                    uncertainty=1.0,
                    supporting_benchmarks=(),
                    benchmark_measurements={},
                    n_measurements=0,
                    generation=generation,
                    provenance="",
                    aggregation=aggregation,
                )
            )
            continue
        measurements = {
            qualified_id: (total / weight)
            for qualified_id, (total, weight) in bucket.items()
            if weight > 0
        }
        total_weight = sum(weight for _total, weight in bucket.values())
        estimate = sum(
            measurements[qualified_id] * bucket[qualified_id][1]
            for qualified_id in measurements
        ) / total_weight
        # Confidence grows with the number of distinct benchmarks that support
        # the skill, capped below 1: one benchmark is never certain evidence.
        support_ratio = min(1.0, len(measurements) / 2.0)
        confidence = round(min(0.95, 0.5 * support_ratio + 0.45 * min(1.0, total_weight / 2.0)), 4)
        estimates.append(
            SkillEstimateEvidence(
                skill=skill,
                estimate=round(estimate, 6),
                confidence=confidence,
                uncertainty=round(1.0 - confidence, 6),
                supporting_benchmarks=tuple(sorted(measurements)),
                benchmark_measurements={
                    key: round(value, 6) for key, value in sorted(measurements.items())
                },
                n_measurements=len(measurements),
                generation=generation,
                provenance="+".join(sorted(provenance.get(skill, ()))),
                aggregation=aggregation,
            )
        )
    return SkillProfile(generation=generation, estimates=tuple(estimates))


# --------------------------------------------------------------------------
# durable learning memory
# --------------------------------------------------------------------------

#: The append-only files a :class:`GrowthState` keeps.
STATE_FILES = {
    "failures": "failure-bank.jsonl",
    "interventions": "intervention-history.jsonl",
    "targets": "target-history.jsonl",
    "capabilities": "capability-history.jsonl",
}
STOPPING_FILE = "stopping-state.json"


def _append_jsonl(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(document, sort_keys=True, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return rows


@dataclass
class GrowthState:
    """Durable, append-only learning memory for the growth loop.

    One generation teaches the next through this and only this: failures found
    in generation N are read back before generation N+1 is planned, intervened
    attempts are remembered so the same unsuccessful experiment is not repeated
    blindly, and every capability profile is kept rather than overwritten.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- failures ----------------------------------------------------------

    def failure_bank(self) -> FailureBank:
        """Rebuild the bank from durable memory (not a fresh empty one)."""
        records = [
            record
            for record in (
                _failure_from_dict(row) for row in _read_jsonl(self._path("failures"))
            )
            if record is not None
        ]
        return FailureBank.from_records(records)

    def persist_failures(self, records: Iterable[FailureRecord]) -> int:
        """Append failures; returns how many were written (never rewrites)."""
        written = 0
        for record in records:
            _append_jsonl(self._path("failures"), record.to_dict())
            written += 1
        return written

    def reopen_failures(self, *, generation: str | None = None) -> tuple[FailureRecord, ...]:
        return self.failure_bank().open_failures(generation=generation)

    def repaired_classes(self) -> tuple[str, ...]:
        return self.failure_bank().repaired_classes()

    # -- interventions -----------------------------------------------------

    def record_intervention(
        self,
        *,
        target_skill: str,
        training_type: str,
        cycle_id: str,
        generation: str,
        cost_gpu_hours: float = 0.0,
        candidate_result: str = "",
        promotion_result: str = "",
        measured_effect: float | None = None,
        regressions: Sequence[str] = (),
        promoted_identity: tuple[str, str] | None = None,
    ) -> dict[str, Any]:
        document = {
            "target_skill": str(target_skill),
            "training_type": str(training_type),
            #: The (path, sha256) this generation promoted, recorded so a resumed
            #: session knows which adapter the next generation must train from.
            #: Absent rather than guessed when the generation did not promote.
            "promoted_identity": (
                [str(promoted_identity[0]), str(promoted_identity[1])]
                if promoted_identity
                else None
            ),
            "cycle_id": str(cycle_id),
            "generation": str(generation),
            "cost_gpu_hours": float(cost_gpu_hours),
            "candidate_result": str(candidate_result),
            "promotion_result": str(promotion_result),
            "measured_effect": measured_effect,
            "regressions": list(regressions),
        }
        _append_jsonl(self._path("interventions"), document)
        return document

    def interventions(self, *, target_skill: str | None = None) -> tuple[dict[str, Any], ...]:
        rows = _read_jsonl(self._path("interventions"))
        if target_skill is None:
            return tuple(rows)
        return tuple(row for row in rows if row.get("target_skill") == target_skill)

    def attempts_on(self, target_skill: str) -> int:
        return len(self.interventions(target_skill=target_skill))

    def successful_effects(self, target_skill: str) -> tuple[float, ...]:
        return tuple(
            float(row["measured_effect"])
            for row in self.interventions(target_skill=target_skill)
            if isinstance(row.get("measured_effect"), (int, float))
        )

    # -- targets -----------------------------------------------------------

    def record_target(self, proposal: "TargetProposal") -> dict[str, Any]:
        document = proposal.to_dict()
        _append_jsonl(self._path("targets"), document)
        return document

    def targets(self, *, parent_version: str | None = None) -> tuple[dict[str, Any], ...]:
        rows = _read_jsonl(self._path("targets"))
        if parent_version is None:
            return tuple(rows)
        return tuple(row for row in rows if row.get("parent_version") == parent_version)

    def same_target_attempts(self, *, target_skill: str) -> int:
        return sum(
            1 for row in self.targets() if row.get("target_skill") == target_skill
        )

    # -- capability history ------------------------------------------------

    def record_capability(self, profile: SkillProfile) -> None:
        _append_jsonl(self._path("capabilities"), profile.to_dict())

    def capability_history(self, *, skill: str | None = None) -> tuple[SkillProfile, ...]:
        profiles = [
            SkillProfile.from_dict(row) for row in _read_jsonl(self._path("capabilities"))
        ]
        if skill is None:
            return tuple(profiles)
        return tuple(profile for profile in profiles if profile.for_skill(skill) is not None)

    # -- stopping state ----------------------------------------------------

    def stopping_state(self) -> dict[str, Any]:
        path = self.root / STOPPING_FILE
        if not path.is_file():
            return {}
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            return {}
        return dict(document) if isinstance(document, Mapping) else {}

    def set_stopping_state(self, document: Mapping[str, Any]) -> None:
        path = self.root / STOPPING_FILE
        path.write_text(
            json.dumps(dict(document), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    # -- helpers -----------------------------------------------------------

    def _path(self, kind: str) -> Path:
        return self.root / STATE_FILES[kind]


def _failure_from_dict(row: Mapping[str, Any]) -> FailureRecord | None:
    required = (
        "failure_id",
        "model_version",
        "benchmark_qualified_id",
        "sample_ref",
        "prompt_digest_hex",
        "output_digest_hex",
        "expected_behavior",
        "score",
        "verifier_evidence",
        "first_seen_generation",
        "last_seen_generation",
    )
    if any(key not in row for key in required):
        return None
    try:
        return FailureRecord(
            failure_id=str(row["failure_id"]),
            model_version=str(row["model_version"]),
            benchmark_qualified_id=str(row["benchmark_qualified_id"]),
            sample_ref=str(row["sample_ref"]),
            prompt_digest_hex=str(row["prompt_digest_hex"]),
            output_digest_hex=str(row["output_digest_hex"]),
            expected_behavior=str(row["expected_behavior"]),
            score=float(row["score"]),
            categories=tuple(row.get("categories", ())),
            verifier_evidence=str(row["verifier_evidence"]),
            confidence=float(row.get("confidence", 1.0)),
            first_seen_generation=str(row["first_seen_generation"]),
            last_seen_generation=str(row["last_seen_generation"]),
            recurrence_count=int(row.get("recurrence_count", 1)),
            repaired=bool(row.get("repaired", False)),
            repair_generation=(
                None
                if row.get("repair_generation") in (None, "")
                else str(row["repair_generation"])
            ),
            notes=str(row.get("notes", "")),
            extra=dict(row.get("extra", {})),
        )
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# intervention classification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InterventionDecision:
    treatment: str
    reason: str

    @property
    def requires_review(self) -> bool:
        return self.treatment in REVIEW_TREATMENTS

    def to_dict(self) -> dict[str, Any]:
        return {
            "treatment": self.treatment,
            "reason": self.reason,
            "requires_review": self.requires_review,
        }


def classify_intervention(
    *,
    estimate: SkillEstimateEvidence | None,
    attempts: int = 0,
    structural: bool = False,
    external_knowledge: bool = False,
    calibration_defect: bool = False,
    max_same_target_attempts: int = 2,
) -> InterventionDecision:
    """Classify how a weakness should be treated, before any campaign exists.

    The order encodes the falsification the mission asks for: a weakness with no
    evidence is a *measurement* problem, not a training problem; a known
    structural limit is research, not another LoRA run; and a target already
    tried to its ceiling stops rather than repeating.
    """
    if estimate is None or not estimate.measured:
        return InterventionDecision(
            "evaluation_needed",
            "the skill has no measurement, so the next action is to measure it "
            "rather than train against an unknown",
        )
    if structural:
        return InterventionDecision(
            "architecture_research",
            "the weakness is a known structural limitation of the current path, "
            "so the same training path cannot fix it",
        )
    if calibration_defect:
        return InterventionDecision(
            "preference",
            "the measured defect is a calibration/preference property, so the "
            "treatment is a preference intervention rather than plain SFT",
        )
    if external_knowledge:
        return InterventionDecision(
            "data_acquisition",
            "the weakness is missing knowledge the model was never trained on, "
            "so it needs data, not protocol examples",
        )
    if attempts >= max_same_target_attempts:
        return InterventionDecision(
            "untrainable_with_current_path",
            f"this target has already consumed {attempts} attempt(s) without "
            "moving the metric, so repeating it is refused rather than merely "
            "deprioritized",
        )
    if attempts == 0 and (estimate.estimate or 0.0) < 0.5:
        return InterventionDecision(
            "targeted_repair",
            "a low, well-measured skill with no prior attempt is the cleanest "
            "repair target",
        )
    return InterventionDecision(
        "sft",
        "a measured weakness with room left on this path; supervise on "
        "non-protected examples for the skill",
    )


# --------------------------------------------------------------------------
# target selection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetScoreFactors:
    """Every factor that went into a target's score, so it is inspectable."""

    weakness: float
    confidence: float
    importance: float
    recurrence: float
    frontier_gap: float
    trainability: float
    novelty: float
    efficiency: float
    regression_risk: float
    repeat_penalty: float
    uncertainty_penalty: float
    total: float

    def to_dict(self) -> dict[str, float]:
        return {
            "weakness": self.weakness,
            "confidence": self.confidence,
            "importance": self.importance,
            "recurrence": self.recurrence,
            "frontier_gap": self.frontier_gap,
            "trainability": self.trainability,
            "novelty": self.novelty,
            "efficiency": self.efficiency,
            "regression_risk": self.regression_risk,
            "repeat_penalty": self.repeat_penalty,
            "uncertainty_penalty": self.uncertainty_penalty,
            "total": self.total,
        }


@dataclass(frozen=True)
class TargetProposal:
    """The next generation's improvement target, reproducible from its inputs."""

    parent_version: str
    target_skill: str
    target_benchmarks: tuple[str, ...]
    weakness_evidence: tuple[str, ...]
    priority: float
    confidence: float
    expected_trainability: float
    regression_risks: tuple[str, ...]
    suggested_training_type: str
    expected_cost_gpu_hours: float
    why_not_other_targets: Mapping[str, str]
    factors: TargetScoreFactors
    treatment_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_version": self.parent_version,
            "target_skill": self.target_skill,
            "target_benchmarks": list(self.target_benchmarks),
            "weakness_evidence": list(self.weakness_evidence),
            "priority": self.priority,
            "confidence": self.confidence,
            "expected_trainability": self.expected_trainability,
            "regression_risks": list(self.regression_risks),
            "suggested_training_type": self.suggested_training_type,
            "expected_cost_gpu_hours": self.expected_cost_gpu_hours,
            "why_not_other_targets": dict(self.why_not_other_targets),
            "factors": self.factors.to_dict(),
            "treatment_reason": self.treatment_reason,
        }


#: Skills that measure reasoning are the ones most likely to regress when a
#: narrow repair is trained; they are named as regression risks.
_REGRESSION_PRONE_PREFIXES = ("reasoning.", "math.", "coding.")


class NextTargetSelector:
    """Choose the next generation's target from durable evidence.

    The score is an explicit product/sum of named factors (see
    :class:`TargetScoreFactors`), not "the lowest benchmark".  Crucially the
    selector's inputs contain no candidate results from an undeclared campaign:
    it reads the parent's measured profile, its memory and its policy, so a
    target can never be picked by peeking at how a candidate scored.
    """

    def __init__(
        self,
        *,
        registry: Any = None,
        protected_skills: Sequence[str] = (),
        frontier: Mapping[str, float] | None = None,
        expected_cost_gpu_hours: float = 0.4,
        max_same_target_attempts: int = 2,
        importance: Mapping[str, float] | None = None,
        structural_skills: Sequence[str] = (),
    ) -> None:
        self.registry = registry
        self.protected_skills = frozenset(protected_skills)
        #: Skills the *policy* declares to be a known structural limit of the
        #: current training path. A weakness there is routed to human review by
        #: the intervention classifier instead of being trained again, which is
        #: the one thing the selector cannot infer from measurements alone.
        self.structural_skills = frozenset(structural_skills)
        self.frontier = dict(frontier or {})
        self.expected_cost_gpu_hours = float(expected_cost_gpu_hours)
        self.max_same_target_attempts = int(max_same_target_attempts)
        self.importance = dict(importance or {})

    # -- scoring -----------------------------------------------------------

    def score_skill(
        self,
        estimate: SkillEstimateEvidence,
        *,
        state: GrowthState,
        recurrence: int = 0,
    ) -> TargetScoreFactors:
        """Score one skill, with every factor named."""
        if not estimate.measured:
            # Unknown skills are deliberately not scored as weak: they are the
            # selector's "measure first" case, handled by the classifier.
            return TargetScoreFactors(
                weakness=0.0,
                confidence=0.0,
                importance=0.0,
                recurrence=0.0,
                frontier_gap=0.0,
                trainability=0.0,
                novelty=0.0,
                efficiency=0.0,
                regression_risk=0.0,
                repeat_penalty=0.0,
                uncertainty_penalty=1.0,
                total=0.0,
            )
        value = float(estimate.estimate or 0.0)
        weakness = max(0.0, 1.0 - value)
        confidence = estimate.confidence
        importance = float(self.importance.get(estimate.skill, 1.0))
        recurrence_factor = min(1.0, recurrence / 3.0)
        frontier = self.frontier.get(estimate.skill)
        frontier_gap = (
            max(0.0, float(frontier) - value) if frontier is not None else 0.0
        )
        attempts = state.attempts_on(estimate.skill)
        effects = state.successful_effects(estimate.skill)
        helped = any(effect > 0.0 for effect in effects)
        # Trainability: fewer attempts and prior success mean a better bet.
        trainability = max(0.0, 1.0 - 0.25 * attempts) * (1.25 if helped else 1.0)
        novelty = 0.0 if attempts >= self.max_same_target_attempts else 1.0 / (1.0 + attempts)
        efficiency = 1.0 / (1.0 + self.expected_cost_gpu_hours)
        regression_risk = self._regression_risk(estimate.skill, state)
        repeat_penalty = 0.5 * min(1.0, attempts / max(1, self.max_same_target_attempts))
        uncertainty_penalty = 1.0 - confidence

        base = (
            weakness
            * confidence
            * importance
            * (1.0 + recurrence_factor)
            * (1.0 + frontier_gap)
            * trainability
            * (0.5 + novelty)
            * efficiency
        )
        # Penalties damp the score multiplicatively rather than subtracting from
        # it, so a heavily-penalized target reads as *less attractive* instead of
        # as a negative priority, and a very weak-but-uncertain skill cannot be
        # pushed below a genuinely better one by the uncertainty term alone.
        total = (
            base
            * (1.0 - 0.5 * regression_risk)
            * (1.0 - 0.5 * repeat_penalty)
            * (1.0 - 0.5 * uncertainty_penalty)
        )
        return TargetScoreFactors(
            weakness=round(weakness, 6),
            confidence=round(confidence, 6),
            importance=round(importance, 6),
            recurrence=round(recurrence_factor, 6),
            frontier_gap=round(frontier_gap, 6),
            trainability=round(trainability, 6),
            novelty=round(novelty, 6),
            efficiency=round(efficiency, 6),
            regression_risk=round(regression_risk, 6),
            repeat_penalty=round(repeat_penalty, 6),
            uncertainty_penalty=round(uncertainty_penalty, 6),
            total=round(total, 6),
        )

    def _regression_risk(self, skill: str, state: GrowthState) -> float:
        """Risk of training this target harming an already-good capability.

        A narrow repair on a formatting-style skill risks the reasoning
        capabilities it shares data and gradients with; a target that already
        caused a measured regression carries more.
        """
        risk = 0.0
        for row in state.interventions(target_skill=skill):
            risk += 0.1 * len(tuple(row.get("regressions", ()) or ()))
        if any(skill.startswith(prefix) for prefix in ("instruction.", "formatting.")):
            risk += 0.15
        if skill in self.protected_skills:
            risk += 0.25
        return round(min(1.0, risk), 6)

    # -- selection ---------------------------------------------------------

    def propose(
        self,
        *,
        parent_version: str,
        profile: SkillProfile,
        state: GrowthState,
        category_counts: Mapping[str, int] | None = None,
        known_skills: Sequence[str] | None = None,
        max_targets_reported: int = 4,
        exclude_skills: Sequence[str] = (),
    ) -> TargetProposal:
        """Return the best-scoring measured target, with the runners-up named.

        The returned proposal is a pure function of its arguments: the same
        profile, state and policy always produce the same proposal.

        ``exclude_skills`` lets the loop ask a *different* question after a
        target has been exhausted (see ``growth_loop.detect_plateau``): "what is
        the best target other than these?" Slots that were already tried to
        their limit are not re-proposed while a treatment or a skill remains,
        which is what keeps a loop from spending its envelope re-running the
        same unsuccessful experiment.
        """
        counts = dict(category_counts or {})
        excluded = {str(skill) for skill in exclude_skills}
        candidates: list[tuple[SkillEstimateEvidence, TargetScoreFactors, int]] = []
        for estimate in profile.estimates:
            if not estimate.measured:
                continue
            if estimate.skill in self.protected_skills:
                # Protected capabilities are gates, never optimization targets.
                continue
            if estimate.skill in excluded:
                continue
            recurrence = int(counts.get(estimate.skill, 0))
            factors = self.score_skill(estimate, state=state, recurrence=recurrence)
            candidates.append((estimate, factors, recurrence))

        if not candidates:
            raise ValueError(
                "no measured, non-protected skill to target: the honest next "
                "action is to measure a capability (evaluation_needed), not to "
                "invent a target"
            )

        candidates.sort(key=lambda item: (-item[1].total, item[0].skill))
        best_estimate, best_factors, _best_recurrence = candidates[0]
        decision = classify_intervention(
            estimate=best_estimate,
            attempts=state.attempts_on(best_estimate.skill),
            structural=best_estimate.skill in self.structural_skills,
            max_same_target_attempts=self.max_same_target_attempts,
        )
        why_not: dict[str, str] = {}
        # The unmeasured skills are named first: "we did not have evidence" is
        # the most useful thing a reader can learn about why an alternative was
        # not chosen, and it must not be crowded out by the scored runners-up.
        for estimate in profile.unmeasured:
            if estimate.skill in self.protected_skills:
                continue
            why_not.setdefault(
                estimate.skill, "insufficient evidence (unmeasured, not zero)"
            )
        for estimate, factors, _ in candidates[1:]:
            if len(why_not) >= max_targets_reported:
                break
            why_not.setdefault(
                estimate.skill,
                f"scored {factors.total:.4f} vs {best_factors.total:.4f} "
                f"(weakness {factors.weakness:.3f}, confidence {factors.confidence:.3f})",
            )

        regression_risks = tuple(
            estimate.skill
            for estimate in profile.measured
            if estimate.skill != best_estimate.skill
            and estimate.skill.startswith(_REGRESSION_PRONE_PREFIXES)
        )
        return TargetProposal(
            parent_version=parent_version,
            target_skill=best_estimate.skill,
            target_benchmarks=best_estimate.supporting_benchmarks,
            weakness_evidence=tuple(
                f"{benchmark}={score:.4f}"
                for benchmark, score in sorted(
                    best_estimate.benchmark_measurements.items()
                )
            ),
            priority=best_factors.total,
            confidence=best_factors.confidence,
            expected_trainability=best_factors.trainability,
            regression_risks=regression_risks,
            suggested_training_type=decision.treatment,
            expected_cost_gpu_hours=self.expected_cost_gpu_hours,
            why_not_other_targets=why_not,
            factors=best_factors,
            treatment_reason=decision.reason,
        )
