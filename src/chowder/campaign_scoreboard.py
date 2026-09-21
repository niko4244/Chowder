"""Public-benchmark campaign scoreboard and the Fable standing reference.

Two scoreboards exist in this program and must never be conflated:

1. The **protected nine-dimension suite** (`parent_suite_content.py`) —
   private, original content, contamination-guarded, the tournament's
   ranking instrument.
2. **This module** — the public-benchmark campaign scoreboard (MMLU,
   GSM8K, HumanEval, MATH historical continuity targets plus any added
   public benchmarks) and the standing DavidAU Fable comparison.

Reporting separation (program policy, enforced structurally): every
`BenchmarkScore` names a *public* benchmark declared in this module's
target list; protected-suite scores live in `parent_eval.py` records and
never enter this module, and nothing here computes an aggregate that
mixes the two. A `FableReference` carries its own dimension list for the
same reason — the higher-level Pareto view that joins them is a *report
assembled elsewhere from both evidence stores*, never a field here.

Honesty rules (matching this codebase's evidence discipline):

- A `BenchmarkScore` without a real `evidence_ref` (an evaluation-run
  experiment id, artifact reference, or equivalent durable handle) is a
  claim, not evidence — construction fails.
- The Fable comparison records only *measured* entries; a dimension
  without a real measurement on either side is absent, and
  `parity_claim_allowed` returns False until every declared dimension
  has evidence on both sides. This module never emits the parity
  sentence itself — the claim is written by humans from the rendered
  table, because only a human can weigh "meaningful capability" the way
  the program doc requires.
- Historical targets are recorded exactly as the program's original
  goals state them (strictly greater than the threshold, per the
  "> 0.90" wording); nothing here lowers them, and the scoreboard's
  digest covers them so a silently retargeted scoreboard is detectable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

#: The program's original continuity targets, recorded verbatim. These
#: are floors from an earlier campaign, not the full objective — the
#: program doc explicitly keeps them while demanding stronger coding
#: and reasoning evaluation on top.
@dataclass(frozen=True)
class HistoricalBenchmarkTarget:
    """One public benchmark's continuity target.

    `threshold` is exclusive (the program's own wording is "> 0.90"):
    a score exactly at the threshold has not met the target.
    """

    benchmark: str
    threshold: float
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, str) or not self.benchmark.strip():
            raise ValueError("benchmark name must be a non-empty string")
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float)):
            raise ValueError("threshold must be a number")
        if not 0.0 < float(self.threshold) <= 1.0:
            raise ValueError(
                f"threshold must be in (0, 1]; got {self.threshold} for {self.benchmark!r}"
            )
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")

    def passed(self, score: float) -> bool:
        return float(score) > self.threshold


#: The program's original continuity targets, recorded verbatim. These
#: are floors from an earlier campaign, not the full objective — the
#: program doc explicitly keeps them while demanding stronger coding
#: and reasoning evaluation on top.
HISTORICAL_TARGETS: tuple[HistoricalBenchmarkTarget, ...] = (
    HistoricalBenchmarkTarget("MMLU", 0.90, "multiple-choice knowledge"),
    HistoricalBenchmarkTarget("GSM8K", 0.90, "grade-school math word problems"),
    HistoricalBenchmarkTarget("HumanEval", 0.60, "Python code generation (legacy floor; stronger coding evaluation is planned on top of it)"),
    HistoricalBenchmarkTarget("MATH", 0.40, "competition mathematics"),
)


@dataclass(frozen=True)
class BenchmarkScore:
    """One measured public-benchmark score for one model.

    `evidence_ref` is mandatory: a durable handle to the evaluation run
    that produced the number (registry experiment id, artifact path, or
    equivalent). A score nobody can trace is not admissible here.
    """

    benchmark: str
    score: float
    evidence_ref: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, str) or not self.benchmark.strip():
            raise ValueError("benchmark name must be a non-empty string")
        if self.benchmark.startswith("suite-"):
            raise ValueError(
                f"{self.benchmark!r} is a protected-suite name: protected scores "
                "live in parent_eval records and never enter the public scoreboard"
            )
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise ValueError("score must be a number")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError(f"score must be in [0, 1]; got {self.score} for {self.benchmark!r}")
        if not isinstance(self.evidence_ref, str) or not self.evidence_ref.strip():
            raise ValueError(
                f"score for {self.benchmark!r} has no evidence_ref: an untraceable "
                "number is a claim, not evidence"
            )


@dataclass(frozen=True)
class CampaignScoreboard:
    """The public-benchmark scoreboard for one model snapshot.

    `baseline` is the model-generation-0 reference the campaign measures
    deltas against (typically the selected parent's own scoreboard).
    `best` carries the best score each benchmark has ever recorded for
    this lineage; a `render` row's `best` column is
    max(best_prior, current). Both are keyed by benchmark name and every
    entry must itself carry an evidence_ref.
    """

    model_label: str
    model_manifest_sha256: str | None
    targets: tuple[HistoricalBenchmarkTarget, ...]
    scores: tuple[BenchmarkScore, ...]
    baseline: Mapping[str, BenchmarkScore] = field(default_factory=dict)
    best_prior: Mapping[str, float] = field(default_factory=dict)
    scoreboard_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model_label, str) or not self.model_label.strip():
            raise ValueError("model_label must be a non-empty string")
        if self.model_manifest_sha256 is not None and (
            not isinstance(self.model_manifest_sha256, str)
            or len(self.model_manifest_sha256) != 64
        ):
            raise ValueError("model_manifest_sha256 must be 64 hex chars or None")
        if not self.targets:
            raise ValueError("a scoreboard with no targets records nothing")
        names = [t.benchmark for t in self.targets]
        if len(set(names)) != len(names):
            raise ValueError("duplicate benchmark in targets")
        seen: set[str] = set()
        for score in self.scores:
            if score.benchmark in seen:
                raise ValueError(f"duplicate score for {score.benchmark!r}")
            seen.add(score.benchmark)
            if score.benchmark not in names:
                raise ValueError(
                    f"score for undeclared benchmark {score.benchmark!r}; declare it "
                    "as a target first — an undeclared benchmark cannot be gated"
                )
        for name, entry in self.baseline.items():
            if not isinstance(entry, BenchmarkScore):
                raise ValueError(f"baseline entry for {name!r} is not a BenchmarkScore")
        unknown_best = set(self.best_prior) - set(names)
        if unknown_best:
            raise ValueError(f"best_prior references undeclared benchmarks: {sorted(unknown_best)}")
        object.__setattr__(self, "scoreboard_sha256", self._digest())

    def _digest(self) -> str:
        payload = json.dumps(
            {
                "model_label": self.model_label,
                "model_manifest_sha256": self.model_manifest_sha256,
                "targets": [
                    {"benchmark": t.benchmark, "threshold": t.threshold, "description": t.description}
                    for t in self.targets
                ],
                "scores": [
                    {"benchmark": s.benchmark, "score": s.score, "evidence_ref": s.evidence_ref}
                    for s in self.scores
                ],
                "baseline": {k: v.score for k, v in sorted(self.baseline.items())},
                "best_prior": dict(sorted(self.best_prior.items())),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def score_for(self, benchmark: str) -> BenchmarkScore | None:
        for score in self.scores:
            if score.benchmark == benchmark:
                return score
        return None

    def render(self) -> dict[str, Any]:
        """The scoreboard table: baseline / current / best / delta / target / pass.

        Rows exist for every declared target. A benchmark never measured
        renders with `current: None`, `pass: False` — honest absence, not
        a zero, because a zero would look like a real (bad) measurement.
        """
        rows: dict[str, dict[str, Any]] = {}
        for target in self.targets:
            score = self.score_for(target.benchmark)
            baseline = self.baseline.get(target.benchmark)
            best = max(
                [self.best_prior.get(target.benchmark, float("-inf"))]
                + ([score.score] if score is not None else [])
            )
            rows[target.benchmark] = {
                "description": target.description,
                "baseline": None if baseline is None else baseline.score,
                "current": None if score is None else score.score,
                "best": None if best == float("-inf") else best,
                "delta": None
                if score is None or baseline is None
                else round(score.score - baseline.score, 12),
                "target": target.threshold,
                "evidence_ref": None if score is None else score.evidence_ref,
                "pass": False if score is None else target.passed(score.score),
            }
        return {
            "model_label": self.model_label,
            "model_manifest_sha256": self.model_manifest_sha256,
            "scoreboard_sha256": self.scoreboard_sha256,
            "rows": rows,
            "all_targets_passed": all(row["pass"] for row in rows.values()),
            "measured_count": sum(1 for row in rows.values() if row["current"] is not None),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_label": self.model_label,
            "model_manifest_sha256": self.model_manifest_sha256,
            "targets": [
                {"benchmark": t.benchmark, "threshold": t.threshold, "description": t.description}
                for t in self.targets
            ],
            "scores": [
                {
                    "benchmark": s.benchmark,
                    "score": s.score,
                    "evidence_ref": s.evidence_ref,
                    "details": dict(s.details),
                }
                for s in self.scores
            ],
            "baseline": {k: v.to_dict() for k, v in sorted(self.baseline.items())},
            "best_prior": dict(sorted(self.best_prior.items())),
            "scoreboard_sha256": self.scoreboard_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CampaignScoreboard":
        scoreboard = cls(
            model_label=data["model_label"],
            model_manifest_sha256=data.get("model_manifest_sha256"),
            targets=tuple(
                HistoricalBenchmarkTarget(**entry) for entry in data["targets"]
            ),
            scores=tuple(
                BenchmarkScore(
                    benchmark=entry["benchmark"],
                    score=entry["score"],
                    evidence_ref=entry["evidence_ref"],
                    details=entry.get("details", {}),
                )
                for entry in data["scores"]
            ),
            baseline={
                k: BenchmarkScore(**v) for k, v in data.get("baseline", {}).items()
            },
            best_prior=dict(data.get("best_prior", {})),
        )
        recorded = data.get("scoreboard_sha256")
        if recorded is not None and recorded != scoreboard.scoreboard_sha256:
            raise ValueError(
                "scoreboard digest mismatch: the serialized scoreboard was modified "
                f"after signing (recorded {str(recorded)[:12]}…, computed "
                f"{scoreboard.scoreboard_sha256[:12]}…)"
            )
        return scoreboard


def advance_best(
    prior: Mapping[str, float], scoreboard: CampaignScoreboard
) -> dict[str, float]:
    """Fold a new scoreboard into the lineage's best-so-far map."""
    merged = dict(prior)
    for score in scoreboard.scores:
        merged[score.benchmark] = max(merged.get(score.benchmark, float("-inf")), score.score)
    return merged


# ---- Fable standing reference --------------------------------------------------

#: The comparison axes the program doc requires for every Fable milestone.
#: Capability dimensions are scored; latency/vram/active-parameters are
#: measured in their own units and rendered alongside, never averaged into
#: a capability number.
FABLE_COMPARISON_DIMENSIONS: tuple[str, ...] = (
    "reasoning",
    "coding",
    "debugging",
    "knowledge",
    "instruction_following",
    "agentic_performance",
    "self_correction",
    "calibration",
    "thinking_efficiency",
    "verbosity",
    "creative_quality",
    "latency",
    "vram",
    "active_parameters_per_token",
)

FABLE_MODEL_LABEL = "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU"


@dataclass(frozen=True)
class FableComparisonEntry:
    """One measured dimension of the ours-vs-Fable comparison.

    Both sides require evidence refs: a comparison where either number
    cannot be traced is a rumor about at least one model.
    """

    dimension: str
    our_score: float
    our_evidence_ref: str
    fable_score: float
    fable_evidence_ref: str

    def __post_init__(self) -> None:
        if self.dimension not in FABLE_COMPARISON_DIMENSIONS:
            raise ValueError(
                f"{self.dimension!r} is not a declared Fable comparison dimension"
            )
        if self.dimension.startswith("suite-"):
            raise ValueError(
                "protected-suite names cannot be Fable comparison dimensions"
            )
        for label, value in (("our_score", self.our_score), ("fable_score", self.fable_score)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{label} must be a non-negative number")
        for label, ref in (
            ("our_evidence_ref", self.our_evidence_ref),
            ("fable_evidence_ref", self.fable_evidence_ref),
        ):
            if not isinstance(ref, str) or not ref.strip():
                raise ValueError(f"{label} must be a non-empty evidence reference")


@dataclass(frozen=True)
class FableReference:
    """The standing ours-vs-Fable comparison for one of our model states.

    `fable_manifest_sha256` is None until parent D is cached and
    manifested locally — an honest gap, not a placeholder.
    """

    our_model_label: str
    our_model_manifest_sha256: str | None
    fable_revision: str
    fable_manifest_sha256: str | None
    entries: tuple[FableComparisonEntry, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.our_model_label, str) or not self.our_model_label.strip():
            raise ValueError("our_model_label must be a non-empty string")
        if not isinstance(self.fable_revision, str) or not self.fable_revision.strip():
            raise ValueError(
                "fable_revision must be the pinned revision; the reference compares "
                "a specific checkpoint, not a moving repository"
            )
        seen: set[str] = set()
        for entry in self.entries:
            if entry.dimension in seen:
                raise ValueError(f"duplicate entry for dimension {entry.dimension!r}")
            seen.add(entry.dimension)

    def measured_dimensions(self) -> tuple[str, ...]:
        return tuple(entry.dimension for entry in self.entries)

    def parity_claim_allowed(self) -> bool:
        """True only when every declared dimension has a measured, evidenced
        entry on both sides. This gates the *permission to write the
        claim sentence* — it does not assert the claim, which requires a
        human judgment over the rendered table."""
        return set(self.measured_dimensions()) == set(FABLE_COMPARISON_DIMENSIONS)

    def render(self) -> dict[str, Any]:
        rows = {}
        for entry in self.entries:
            rows[entry.dimension] = {
                "our_score": entry.our_score,
                "our_evidence_ref": entry.our_evidence_ref,
                "fable_score": entry.fable_score,
                "fable_evidence_ref": entry.fable_evidence_ref,
                "delta": round(entry.our_score - entry.fable_score, 12),
            }
        return {
            "our_model_label": self.our_model_label,
            "fable_model_label": FABLE_MODEL_LABEL,
            "fable_revision": self.fable_revision,
            "fable_manifest_sha256": self.fable_manifest_sha256,
            "rows": rows,
            "unmeasured_dimensions": [
                d for d in FABLE_COMPARISON_DIMENSIONS if d not in rows
            ],
            "parity_claim_allowed": self.parity_claim_allowed(),
        }


# ---- Efficiency metrics --------------------------------------------------------
#: Named formulas from the program doc so every scoreboard computes them
#: identically. Pure arithmetic; the inputs come from measured state.

EFFICIENCY_METRICS: tuple[str, ...] = (
    "capability_per_active_billion",
    "capability_per_gpu_second",
    "coding_score_per_active_billion",
    "reasoning_score_per_active_billion",
)


def capability_per_active_billion(capability: float, active_params: float) -> float:
    """capability score (0..1) divided by active billions of parameters."""
    if capability < 0:
        raise ValueError("capability must be non-negative")
    if active_params <= 0:
        raise ValueError("active_params must be positive")
    return capability / active_params


def capability_per_gpu_second(capability: float, gpu_seconds: float) -> float:
    if gpu_seconds <= 0:
        raise ValueError("gpu_seconds must be positive")
    return capability / gpu_seconds
