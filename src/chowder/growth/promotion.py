"""Promotion rules: whether a candidate becomes the next generation.

Multi-objective and predeclared: promotion compares the candidate against
its parent on target improvement, protected regressions, broad battery,
calibration, reliability, evidence integrity (contamination), and the
resource envelope. Frontier context is recorded but NEVER decides
promotion -- a candidate below GPT-class can still be a legitimate
generation if it beats its parent without regressions.

Verdicts: PROMOTED / REJECTED / INCONCLUSIVE / TAINTED.

- TAINTED: the contamination firewall flags the evidence; a tainted run can
  never be promoted regardless of the numbers.
- INCONCLUSIVE: the evidence is too thin (unknown benchmarks, insufficient
  samples) to decide -- refuses to certify rather than assuming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from chowder.evals.result import (
    CARRIED_REFERENCE,
    MEASURED_PARENT,
    MEASURED_THIS_GENERATION,
    UNMEASURED,
)

from .statistics import compare


@dataclass(frozen=True)
class BenchmarkResult:
    """One measured score with its samples (for statistical comparison).

    ``measurement_origin`` records which system actually produced the row.
    A promotion gate may only be *satisfied* by a row measured on the
    generation being adjudicated; parent-measured or carried rows are
    context, never evidence for the candidate side.
    """

    benchmark_qualified_id: str
    score: float  # 0..1 normalized
    samples: tuple[float, ...] = ()  # per-question/per-run scores
    contamination: str = "UNKNOWN"  # from the firewall manifest
    measurement_confidence: float = 0.8
    measurement_origin: str = UNMEASURED

    @property
    def candidate_measured(self) -> bool:
        return self.measurement_origin == MEASURED_THIS_GENERATION

    @property
    def gate_eligible(self) -> bool:
        """May this row satisfy a promotion gate at all?

        Only rows actually measured on this generation count. Carried and
        parent rows may render in reports; unmeasured rows are the absence
        of evidence.
        """
        return self.measurement_origin == MEASURED_THIS_GENERATION


@dataclass(frozen=True)
class PromotionInput:
    """Everything the predeclared promotion decision sees."""

    candidate_version: str
    parent_version: str
    target_benchmarks: tuple[str, ...]  # benchmarks measuring the cycle's skills
    candidate_results: Mapping[str, BenchmarkResult]
    parent_results: Mapping[str, BenchmarkResult]
    protected_benchmarks: tuple[str, ...]
    broad_battery_benchmarks: tuple[str, ...]
    calibration_benchmarks: tuple[str, ...] = ()
    reliability_benchmarks: tuple[str, ...] = ()  # pass@k-capable evals
    min_target_improvement: float = 0.02
    max_protected_regression: float = 0.02
    max_broad_regression: float = 0.05
    max_calibration_regression: float = 0.02
    max_reliability_regression: float = 0.02
    device_gpu_hours: float = 0.0
    device_gpu_hours_ceiling: float | None = None


@dataclass(frozen=True)
class PromotionDecision:
    verdict: str  # PROMOTED | REJECTED | INCONCLUSIVE | TAINTED
    reasons: tuple[str, ...]
    checks: Mapping[str, str]  # check name -> improved/regressed/flat/inconclusive/ok/violated
    target_deltas: Mapping[str, float] = field(default_factory=dict)
    protected_deltas: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "checks": dict(self.checks),
            "target_deltas": dict(self.target_deltas),
            "protected_deltas": dict(self.protected_deltas),
        }


def evaluate_promotion(data: PromotionInput) -> PromotionDecision:
    """The single, predeclared promotion rule. Same inputs -> same verdict."""
    reasons: list[str] = []
    checks: dict[str, str] = {}
    target_deltas: dict[str, float] = {}
    protected_deltas: dict[str, float] = {}

    # 1. Evidence integrity first: any KNOWN_CONTAMINATION or POSSIBLE on the
    # candidate's target/protected results taints the whole decision.
    tainted = [
        qualified_id
        for qualified_id, result in data.candidate_results.items()
        if result.contamination in {"KNOWN_CONTAMINATION", "POSSIBLE"}
    ]
    if tainted:
        return PromotionDecision(
            verdict="TAINTED",
            reasons=(f"contamination flagged on: {', '.join(sorted(tainted))}",),
            checks={"evidence_integrity": "violated"},
        )
    unknown = [
        qualified_id
        for qualified_id, result in data.candidate_results.items()
        if result.contamination == "UNKNOWN"
    ]
    if unknown:
        checks["evidence_integrity"] = "inconclusive"
        reasons.append(f"contamination unchecked on: {', '.join(sorted(unknown))}")
    else:
        checks["evidence_integrity"] = "ok"

    def _paired(benchmark_id: str) -> tuple[Sequence[float], Sequence[float]] | None:
        candidate = data.candidate_results.get(benchmark_id)
        parent = data.parent_results.get(benchmark_id)
        if (
            candidate is None
            or parent is None
            or not candidate.samples
            or not parent.samples
        ):
            return None
        # Identical arrays on both sides are the signature of a carried row
        # relabeled as candidate evidence: they manufacture fake statistical
        # confidence (a degenerate "flat" or a floor-to-ceiling "improved"
        # built from one real measurement). Only candidate-measured rows may
        # supply the candidate side of a pair.
        if not candidate.gate_eligible:
            return None
        return (parent.samples, candidate.samples)

    # 2. Target improvement: at least one target benchmark statistically
    #    improved by the declared minimum; none may regress significantly.
    improved_targets = 0
    target_missing: list[str] = []
    for benchmark_id in data.target_benchmarks:
        candidate = data.candidate_results.get(benchmark_id)
        if candidate is not None and not candidate.gate_eligible:
            # A target row without candidate-measured provenance is not a
            # measurement of this candidate, no matter what its label says.
            target_missing.append(benchmark_id)
            reasons.append(
                f"target benchmark has no candidate-measured evidence "
                f"(origin {candidate.measurement_origin}): {benchmark_id}"
            )
            continue
        pair = _paired(benchmark_id)
        if pair is None:
            target_missing.append(benchmark_id)
            continue
        before, after = pair
        result = compare(before, after, min_effect=data.min_target_improvement)
        candidate_score = data.candidate_results[benchmark_id].score
        parent_score = data.parent_results[benchmark_id].score
        target_deltas[benchmark_id] = candidate_score - parent_score
        checks[f"target:{benchmark_id}"] = result.verdict
        if result.verdict == "improved":
            improved_targets += 1
        elif result.verdict == "regressed":
            reasons.append(f"target benchmark regressed: {benchmark_id}")
    if target_missing:
        for benchmark_id in target_missing:
            checks[f"target:{benchmark_id}"] = "inconclusive"
        reasons.append(f"target benchmarks missing samples: {', '.join(target_missing)}")
    if improved_targets == 0:
        checks["target_improvement"] = "not met"
    else:
        checks["target_improvement"] = "met"

    # 3. Protected regressions: no protected benchmark may drop more than the
    #    declared tolerance (statistically where samples exist). A protected
    #    benchmark without per-sample evidence is INCONCLUSIVE, not ok --
    #    promotion refuses to certify what it cannot verify.
    protected_violations = 0
    protected_inconclusive = 0
    for benchmark_id in data.protected_benchmarks:
        candidate = data.candidate_results.get(benchmark_id)
        parent = data.parent_results.get(benchmark_id)
        if candidate is None or parent is None:
            checks[f"protected:{benchmark_id}"] = "inconclusive"
            protected_inconclusive += 1
            reasons.append(f"protected benchmark unmeasured: {benchmark_id}")
            continue
        if not candidate.gate_eligible:
            # The row exists but was not measured on the candidate (carried
            # or parent-origin). A parent score relabeled as candidate
            # evidence is exactly what this gate must never certify -- and a
            # parent pinned at 0.0 must not create a "cannot regress"
            # shortcut. Not measured => not certified.
            checks[f"protected:{benchmark_id}"] = "inconclusive"
            protected_inconclusive += 1
            reasons.append(
                f"protected benchmark has no candidate-measured evidence "
                f"(origin {candidate.measurement_origin}): {benchmark_id}"
            )
            continue
        delta = candidate.score - parent.score
        protected_deltas[benchmark_id] = delta
        pair = _paired(benchmark_id)
        if pair is not None and len(pair[0]) >= 2 and len(pair[1]) >= 2:
            result = compare(*pair, min_effect=0.0)
            if result.verdict == "regressed":
                checks[f"protected:{benchmark_id}"] = "violated"
                protected_violations += 1
                continue
            if delta < -data.max_protected_regression:
                checks[f"protected:{benchmark_id}"] = "violated"
                protected_violations += 1
                continue
            checks[f"protected:{benchmark_id}"] = "ok"
        else:
            # No paired samples. The aggregate delta is still real evidence:
            # a drop beyond tolerance violates outright, and an aggregate
            # that does NOT drop cannot be scored for significance -- that
            # is a one-sided pass the rule can certify without pretending
            # to a significance test it never ran. (A carried protected
            # measurement -- identical row by construction -- lands here:
            # delta exactly 0, no regression, certified as not-regressed.)
            if delta < -data.max_protected_regression:
                checks[f"protected:{benchmark_id}"] = "violated"
                protected_violations += 1
            else:
                checks[f"protected:{benchmark_id}"] = "not-regressed"
                reasons.append(
                    f"protected benchmark aggregate-only (no paired samples; "
                    f"delta {delta:+.4f} within tolerance): {benchmark_id}"
                )
    if protected_violations:
        checks["protected_regression"] = "violated"
    elif protected_inconclusive:
        checks["protected_regression"] = "inconclusive"
    else:
        checks["protected_regression"] = "ok"
    if protected_violations:
        reasons.append(f"{protected_violations} protected regression(s)")

    # 4. Broad battery: aggregate must not materially deteriorate. Only
    #    candidate-measured rows feed the means -- carried parent values in
    #    the aggregate would compare the parent against itself.
    broad_before = [
        data.parent_results[b].score for b in data.broad_battery_benchmarks if b in data.parent_results
    ]
    broad_after = [
        data.candidate_results[b].score
        for b in data.broad_battery_benchmarks
        if b in data.candidate_results
        and data.candidate_results[b].gate_eligible
    ]
    broad_declared = len(data.broad_battery_benchmarks)
    if broad_before and broad_after and len(broad_after) == broad_declared:
        mean_before = sum(broad_before) / len(broad_before)
        mean_after = sum(broad_after) / len(broad_after)
        if mean_after < mean_before - data.max_broad_regression:
            checks["broad_battery"] = "violated"
            reasons.append(
                f"broad battery dropped {mean_before - mean_after:.3f} "
                f"({mean_before:.3f} -> {mean_after:.3f})"
            )
        else:
            checks["broad_battery"] = "ok"
    else:
        checks["broad_battery"] = "inconclusive"
        if broad_declared and len(broad_after) < broad_declared:
            missing = sorted(
                set(data.broad_battery_benchmarks)
                - {
                    b
                    for b in data.broad_battery_benchmarks
                    if b in data.candidate_results
                    and data.candidate_results[b].gate_eligible
                }
            )
            reasons.append(
                "broad battery insufficiently measured (no candidate-measured "
                f"evidence for: {', '.join(missing)})"
            )
        else:
            reasons.append("broad battery insufficiently measured")

    # 5. Calibration: hallucination/overconfidence must not increase past
    #    tolerance.
    calibration_violations = 0
    calibration_inconclusive = 0
    for benchmark_id in data.calibration_benchmarks:
        candidate = data.candidate_results.get(benchmark_id)
        parent = data.parent_results.get(benchmark_id)
        if candidate is None or parent is None or not candidate.gate_eligible:
            checks[f"calibration:{benchmark_id}"] = "inconclusive"
            calibration_inconclusive += 1
            reasons.append(f"calibration benchmark unmeasured on candidate: {benchmark_id}")
            continue
        if candidate.score < parent.score - data.max_calibration_regression:
            checks[f"calibration:{benchmark_id}"] = "violated"
            calibration_violations += 1
        else:
            checks[f"calibration:{benchmark_id}"] = "ok"
    if not data.calibration_benchmarks:
        checks["calibration"] = "unmeasured"
    elif calibration_violations:
        checks["calibration"] = "violated"
    elif calibration_inconclusive:
        checks["calibration"] = "inconclusive"
    else:
        checks["calibration"] = "ok"
    if calibration_violations:
        reasons.append(f"{calibration_violations} calibration regression(s)")

    # 5b. Reliability: pass@k-capable evals must not regress past tolerance.
    #     A hard gate like calibration, but with one stricter rule: a *declared*
    #     reliability set with no results is inconclusive, not ok. The set is a
    #     predeclaration -- dodging the gate by never running the eval would
    #     otherwise be a free pass, which is exactly what a predeclared rule
    #     must not allow. An empty set is honestly "unmeasured": nothing was
    #     promised, so nothing can be violated.
    reliability_violations = 0
    reliability_inconclusive = 0
    for benchmark_id in data.reliability_benchmarks:
        candidate = data.candidate_results.get(benchmark_id)
        parent = data.parent_results.get(benchmark_id)
        if candidate is None or parent is None or not candidate.gate_eligible:
            checks[f"reliability:{benchmark_id}"] = "inconclusive"
            reliability_inconclusive += 1
            reasons.append(f"reliability benchmark unmeasured on candidate: {benchmark_id}")
            continue
        delta = candidate.score - parent.score
        if delta < -data.max_reliability_regression:
            checks[f"reliability:{benchmark_id}"] = "violated"
            reliability_violations += 1
        else:
            checks[f"reliability:{benchmark_id}"] = "ok"
    if not data.reliability_benchmarks:
        checks["reliability"] = "unmeasured"
    elif reliability_violations:
        checks["reliability"] = "violated"
    elif reliability_inconclusive:
        checks["reliability"] = "inconclusive"
    else:
        checks["reliability"] = "ok"
    if reliability_violations:
        reasons.append(f"{reliability_violations} reliability regression(s)")

    # 6. Resource envelope.
    if data.device_gpu_hours_ceiling is not None:
        if data.device_gpu_hours > data.device_gpu_hours_ceiling:
            checks["resource_envelope"] = "violated"
            reasons.append(
                f"device GPU-h {data.device_gpu_hours:.4f} exceeds ceiling "
                f"{data.device_gpu_hours_ceiling:.4f}"
            )
        else:
            checks["resource_envelope"] = "ok"
    else:
        checks["resource_envelope"] = "unmeasured"

    # Verdict assembly. Hard failures decide first: a protected regression,
    # calibration violation, reliability violation, or resource-envelope
    # breach is REJECTED no matter how good the target looks. TAINTED and
    # INCONCLUSIVE are decided above; what remains is REJECTED vs PROMOTED.
    hard_failures = (
        (checks["protected_regression"] == "violated")
        or (checks["calibration"] == "violated")
        or (checks["reliability"] == "violated")
        or (checks["resource_envelope"] == "violated")
    )
    if hard_failures:
        return PromotionDecision(
            verdict="REJECTED",
            reasons=tuple(reasons) or ("predeclared checks failed",),
            checks=checks,
            target_deltas=target_deltas,
            protected_deltas=protected_deltas,
        )
    if improved_targets == 0 and not target_missing:
        # Complete, clean evidence and the candidate still improved nothing it
        # set out to improve: that is a rejection, not missing evidence.
        if checks["evidence_integrity"] == "ok" and checks["broad_battery"] == "ok":
            return PromotionDecision(
                verdict="REJECTED",
                reasons=tuple(reasons) or ("no target benchmark improved",),
                checks=checks,
                target_deltas=target_deltas,
                protected_deltas=protected_deltas,
            )
    if improved_targets > 0 and checks["protected_regression"] == "ok" and not calibration_violations:
        # A declared reliability set must have been measured to promote:
        # "unmeasured" means the set was never declared (backward-compatible
        # default), "inconclusive" means it was declared and dodged. Same for
        # a declared calibration set.
        reliability_measured = checks["reliability"] in {"ok", "unmeasured"}
        calibration_measured = checks["calibration"] in {"ok", "unmeasured"}
        if (
            checks["broad_battery"] == "ok"
            and checks["evidence_integrity"] == "ok"
            and reliability_measured
            and calibration_measured
        ):
            return PromotionDecision(
                verdict="PROMOTED",
                reasons=("all predeclared promotion checks passed",),
                checks=checks,
                target_deltas=target_deltas,
                protected_deltas=protected_deltas,
            )
        if checks["broad_battery"] == "inconclusive" or checks["evidence_integrity"] == "inconclusive":
            return PromotionDecision(
                verdict="INCONCLUSIVE",
                reasons=tuple(reasons) or ("promotion evidence incomplete",),
                checks=checks,
                target_deltas=target_deltas,
                protected_deltas=protected_deltas,
            )
    return PromotionDecision(
        verdict="INCONCLUSIVE",
        reasons=tuple(reasons) or ("promotion evidence incomplete",),
        checks=checks,
        target_deltas=target_deltas,
        protected_deltas=protected_deltas,
    )
