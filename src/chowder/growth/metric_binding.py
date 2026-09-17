"""Metric binding: a measured run -> a declared 0..1 promotion score.

The growth cycle could train and evaluate a candidate, but nothing produced the
``BenchmarkResult``s ``evaluate_promotion`` consumes. That gap is arithmetic:
a run reports a raw metric (a loss of 3.05, an accuracy of 0.31) and promotion
compares 0..1 scores in which higher is always better. Closing it with an
invented conversion -- ``1 - loss``, a hardcoded min/max, a percentage divided
by a hundred -- is how a rejection becomes a promotion.

So the conversion is never computed here. It is *read* from
:mod:`chowder.growth.catalog`'s ``METRIC_SEMANTICS``, which declares each
metric's polarity and, where one exists, its 0..1 scale with provenance. This
module's job is to apply that declaration and to refuse everything it cannot
apply:

- a benchmark absent from the registry;
- an entry that declares no 0..1 scale (``kernelgen@2025-06`` today);
- a run whose metric name disagrees with the entry's;
- a run that reports no score, or reports a support level saying this model was
  never measured;
- a raw value outside the declared domain;
- a run measured on a different generation than the one being bound.

Every refusal is named in the report. A benchmark missing from a promotion
input is therefore never *silently* missing: it is either declared unscaleable
or named as a refusal, and promotion sees the difference.

Two things are deliberately **not** refused, and the reasons matter:

- an aggregate that disagrees with the mean of its per-sample scores. Adapters
  legitimately aggregate differently (dropping malformed items, reporting a
  median). The binder records ``sample_mean`` so the disagreement is visible
  evidence instead of a hidden constraint.
- ``UNKNOWN`` contamination. It is recorded on the result, where
  ``evaluate_promotion`` already treats it as an inconclusive integrity check
  rather than as clean.

Scales are read from the registry and from nowhere else. A run may carry a
``normalization`` blob in its metadata and the binder ignores it: a candidate
does not get to choose the denominator its own promotion is judged against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from chowder.evals.result import SUPPORTED, BenchmarkRun

from .benchmark_registry import (
    BenchmarkEntry,
    BenchmarkRegistry,
    Normalization,
    NormalizationRefused,
)
from .promotion import (
    BenchmarkResult,
    PromotionDecision,
    PromotionInput,
    evaluate_promotion,
)


class PromotionBindingError(ValueError):
    """The promotion *configuration* is wrong, so no run could be bound.

    Distinct from :class:`BindingRefusal`, which is about one measurement that
    could not be converted. This is raised when the declared promotion sets or
    the registry itself are inconsistent -- naming a benchmark nobody declared,
    for example -- because such a defect would silently change what promotion
    compares rather than merely dropping one number.
    """


@dataclass(frozen=True)
class BindingRefusal:
    """One run that could not be placed on a declared scale, and why."""

    qualified_id: str
    reason: str
    generation_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_qualified_id": self.qualified_id,
            "reason": self.reason,
            "generation_version": self.generation_version,
        }


@dataclass(frozen=True)
class BoundMeasurement:
    """One run successfully placed on its declared scale."""

    qualified_id: str
    generation_version: str
    raw: float
    score: float
    metric: str
    direction: str
    normalization_kind: str
    result: BenchmarkResult
    support: str = SUPPORTED
    measurement_kind: str = ""
    sample_count: int = 0
    #: Mean of the normalized per-sample scores, when the run reported any.
    #: Recorded so an aggregate that contradicts its own samples is visible
    #: evidence rather than a hidden constraint.
    sample_mean: float | None = None
    artifact_ref: str = ""

    @property
    def aggregate_agrees_with_samples(self) -> bool | None:
        if self.sample_mean is None:
            return None
        return math.isclose(self.score, self.sample_mean, rel_tol=0.0, abs_tol=1e-9)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_qualified_id": self.qualified_id,
            "generation_version": self.generation_version,
            "raw": self.raw,
            "score": self.score,
            "metric": self.metric,
            "direction": self.direction,
            "normalization_kind": self.normalization_kind,
            "support": self.support,
            "measurement_kind": self.measurement_kind,
            "sample_count": self.sample_count,
            "sample_mean": self.sample_mean,
            "artifact_ref": self.artifact_ref,
            "contamination": self.result.contamination,
        }


@dataclass(frozen=True)
class BindingReport:
    """Every run from one generation's binding attempt: bound, or refused."""

    generation_version: str
    results: Mapping[str, BenchmarkResult]
    measurements: Mapping[str, BoundMeasurement] = field(default_factory=dict)
    refusals: tuple[BindingRefusal, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_version": self.generation_version,
            "bound": {k: v.to_dict() for k, v in self.measurements.items()},
            "results": {
                k: {
                    "score": v.score,
                    "samples": list(v.samples),
                    "contamination": v.contamination,
                }
                for k, v in self.results.items()
            },
            "refusals": [r.to_dict() for r in self.refusals],
        }


@dataclass(frozen=True)
class PromotionAssembly:
    """A bound promotion input, its verdict, and the refusals behind both."""

    promotion_input: PromotionInput
    decision: PromotionDecision
    report: BindingReport
    parent_report: BindingReport

    @property
    def refusals(self) -> tuple[BindingRefusal, ...]:
        """Both sides' refusals, candidate first."""
        return self.report.refusals + self.parent_report.refusals

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "candidate": self.report.to_dict(),
            "parent": self.parent_report.to_dict(),
        }


class MetricBinder:
    """Applies declared metric semantics to measured runs.

    ``contamination`` is the ``benchmarks`` section of a generation's
    contamination manifest (``{qualified_id: {"status": ...}}``). A benchmark
    absent from it binds with ``UNKNOWN``, which promotion already treats as an
    inconclusive integrity check -- absence is not silently clean.
    """

    def __init__(
        self,
        registry: BenchmarkRegistry,
        *,
        contamination: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.registry = registry
        self._contamination = {
            qualified_id: dict(entry)
            for qualified_id, entry in (contamination or {}).items()
        }

    @classmethod
    def from_manifest(
        cls, registry: BenchmarkRegistry, manifest: Mapping[str, Any]
    ) -> "MetricBinder":
        """Build a binder from a firewall ``contamination_manifest.json``."""
        section = manifest.get("benchmarks", {})
        if not isinstance(section, Mapping):
            raise PromotionBindingError(
                "contamination manifest 'benchmarks' section must be a mapping of "
                f"qualified id -> status, got {type(section).__name__}"
            )
        return cls(registry, contamination=section)

    # ---------------- contamination ----------------

    def contamination_status(self, qualified_id: str) -> str:
        """The firewall's verdict for one benchmark, or UNKNOWN if unchecked."""
        entry = self._contamination.get(qualified_id)
        if not entry:
            return "UNKNOWN"
        status = entry.get("status")
        if not isinstance(status, str) or not status.strip():
            return "UNKNOWN"
        return status.strip().upper()

    # ---------------- binding ----------------

    def bind(
        self,
        run: BenchmarkRun,
        *,
        generation_version: str | None = None,
    ) -> BoundMeasurement | BindingRefusal:
        """Place one run on its declared scale, or refuse for a named reason.

        ``generation_version`` pins the generation being bound: a run measured
        on another generation belongs to another comparison, and binding it
        would silently compare a candidate against a parent that was never
        measured. ``None`` means "do not check" (single-sided use).
        """
        qualified_id = run.benchmark_qualified_id
        if generation_version is not None and run.generation_version != generation_version:
            return BindingRefusal(
                qualified_id,
                f"{qualified_id}: run records generation {run.generation_version!r} but "
                f"this binding is for generation {generation_version!r}; a measurement "
                "from another generation is not evidence about this one",
                generation_version=run.generation_version,
            )

        entry = self.registry.get(qualified_id)
        if entry is None:
            return BindingRefusal(
                qualified_id,
                f"{qualified_id}: not in the registry, so it declares no direction and "
                "no 0..1 scale; an undeclared benchmark cannot be scored",
                generation_version=run.generation_version,
            )

        if run.support != SUPPORTED:
            return BindingRefusal(
                qualified_id,
                f"{qualified_id}: the run reports support {run.support} "
                f"({run.support.lower().replace('_', ' ')}), not a measurement of this "
                "model; an unmeasured benchmark is never bound as a score",
                generation_version=run.generation_version,
            )

        if run.score is None:
            return BindingRefusal(
                qualified_id,
                f"{qualified_id}: the run reports no score; absent evidence is not zero",
                generation_version=run.generation_version,
            )

        if run.metric != entry.primary_metric:
            return BindingRefusal(
                qualified_id,
                f"{qualified_id}: the run's metric {run.metric!r} is not the registry's "
                f"declared primary metric {entry.primary_metric!r}; a declared scale "
                "belongs to one metric, not to a benchmark's name",
                generation_version=run.generation_version,
            )

        normalization = entry.normalization
        if normalization is None:
            return BindingRefusal(
                qualified_id,
                f"{qualified_id}: the registry declares no declared 0..1 scale for "
                f"metric {entry.primary_metric!r}, so its measurement cannot enter a "
                "promotion input; declare anchors with provenance or leave it unbound",
                generation_version=run.generation_version,
            )

        try:
            score = normalization.score(
                run.score, direction=entry.direction, qualified_id=qualified_id
            )
            samples = tuple(
                normalization.score(
                    raw_sample, direction=entry.direction, qualified_id=qualified_id
                )
                for raw_sample in run.per_sample_scores
            )
        except NormalizationRefused as refusal:
            return BindingRefusal(
                qualified_id,
                str(refusal),
                generation_version=run.generation_version,
            )

        result = BenchmarkResult(
            benchmark_qualified_id=qualified_id,
            score=score,
            samples=samples,
            contamination=self.contamination_status(qualified_id),
        )
        measurement = BoundMeasurement(
            qualified_id=qualified_id,
            generation_version=run.generation_version,
            raw=float(run.score),
            score=score,
            metric=run.metric,
            direction=entry.direction,
            normalization_kind=normalization.kind,
            result=result,
            support=run.support,
            measurement_kind=run.measurement_kind,
            sample_count=len(samples),
            sample_mean=(sum(samples) / len(samples)) if samples else None,
            artifact_ref=run.raw_artifact_ref,
        )
        return measurement


    def bind_all(
        self,
        runs: Iterable[BenchmarkRun],
        *,
        generation_version: str | None = None,
    ) -> BindingReport:
        """Bind every run of one generation; bound and refused both survive.

        The report keeps *both* halves, which is the point: a caller reading
        only ``results`` can see exactly which measurements are absent, and the
        refusals say why.
        """
        results: dict[str, BenchmarkResult] = {}
        measurements: dict[str, BoundMeasurement] = {}
        refusals: list[BindingRefusal] = []
        for run in runs:
            outcome = self.bind(run, generation_version=generation_version)
            if isinstance(outcome, BindingRefusal):
                refusals.append(outcome)
                continue
            if outcome.qualified_id in results:
                refusals.append(
                    BindingRefusal(
                        outcome.qualified_id,
                        f"{outcome.qualified_id}: bound twice in one generation; a "
                        "duplicate measurement would silently pick one of two scores",
                        generation_version=outcome.generation_version,
                    )
                )
                continue
            measurements[outcome.qualified_id] = outcome
            results[outcome.qualified_id] = outcome.result
        return BindingReport(
            generation_version=generation_version or "",
            results=results,
            measurements=measurements,
            refusals=tuple(refusals),
        )

    # ---------------- promotion assembly ----------------

    def promotion_input(
        self,
        *,
        candidate_version: str,
        parent_version: str,
        candidate_runs: Sequence[BenchmarkRun],
        parent_runs: Sequence[BenchmarkRun],
        target_benchmarks: Sequence[str],
        protected_benchmarks: Sequence[str] = (),
        broad_battery_benchmarks: Sequence[str] = (),
        calibration_benchmarks: Sequence[str] = (),
        reliability_benchmarks: Sequence[str] = (),
        min_target_improvement: float = 0.02,
        max_protected_regression: float = 0.02,
        max_broad_regression: float = 0.05,
        max_calibration_regression: float = 0.02,
        device_gpu_hours: float = 0.0,
        device_gpu_hours_ceiling: float | None = None,
    ) -> PromotionAssembly:
        """Bind both generations' runs and adjudicate with the declared rule."""
        self._check_declared(
            target_benchmarks, protected_benchmarks, broad_battery_benchmarks,
            calibration_benchmarks, reliability_benchmarks,
        )
        report = self.bind_all(candidate_runs, generation_version=candidate_version)
        parent_report = self.bind_all(parent_runs, generation_version=parent_version)
        data = PromotionInput(
            candidate_version=candidate_version,
            parent_version=parent_version,
            target_benchmarks=tuple(target_benchmarks),
            candidate_results=report.results,
            parent_results=parent_report.results,
            protected_benchmarks=tuple(protected_benchmarks),
            broad_battery_benchmarks=tuple(broad_battery_benchmarks),
            calibration_benchmarks=tuple(calibration_benchmarks),
            reliability_benchmarks=tuple(reliability_benchmarks),
            min_target_improvement=min_target_improvement,
            max_protected_regression=max_protected_regression,
            max_broad_regression=max_broad_regression,
            max_calibration_regression=max_calibration_regression,
            device_gpu_hours=device_gpu_hours,
            device_gpu_hours_ceiling=device_gpu_hours_ceiling,
        )
        return PromotionAssembly(
            promotion_input=data,
            decision=evaluate_promotion(data),
            report=report,
            parent_report=parent_report,
        )

    def _check_declared(self, *sets: Sequence[str]) -> None:
        """Every benchmark a promotion set names must be registered.

        A typo here does not fail loudly on its own: the named benchmark simply
        has no results, which promotion reads as "insufficient evidence" rather
        than as a configuration error. That is the difference between a broken
        promotion rule and a weak candidate, so it refuses instead.
        """
        names = (
            "target_benchmarks",
            "protected_benchmarks",
            "broad_battery_benchmarks",
            "calibration_benchmarks",
            "reliability_benchmarks",
        )
        for name, declared in zip(names, sets):
            for qualified_id in declared:
                if self.registry.get(qualified_id) is None:
                    raise PromotionBindingError(
                        f"promotion set {name} names {qualified_id!r}, which is not in "
                        "the registry; a promotion set is a declaration, and a "
                        "declaration about an undeclared benchmark cannot be compared"
                    )

    # ---------------- entry access ----------------

    def entry(self, qualified_id: str) -> BenchmarkEntry | None:
        return self.registry.get(qualified_id)

    def normalization_for(self, qualified_id: str) -> Normalization | None:
        entry = self.registry.get(qualified_id)
        return entry.normalization if entry is not None else None
