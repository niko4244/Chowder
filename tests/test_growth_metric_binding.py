"""The promotion half of the growth loop is only as honest as its binding.

A growth cycle can train a real candidate, but ``evaluate_promotion`` consumes
``BenchmarkResult``s on a 0..1 better-direction scale -- and nothing in the
shipped system produced them. Turning a measured metric into that scale is
arithmetic, and arithmetic on an undeclared scale is how a rejection becomes a
promotion.

These tests pin the contract that closes it:

- every registry entry **declares** its metric's direction and, when it has
  one, its 0..1 scale -- with provenance for a scale that is not the metric's
  own range;
- the binder converts a measured run onto that declared scale and **refuses**
  everything it cannot convert (unknown benchmark, unequal metric name, no
  declared scale, value outside the declared domain, unsupported run, wrong
  generation) rather than inventing a number;
- the declared sets are checked for registry membership, because a promotion
  set naming a benchmark nobody declared is a configuration defect;
- the identity control over real measurements must not promote.

Refusals are named in the report, so a benchmark that is absent from the
promotion input is never silently absent: it is either declared unmeasurable
or named as a refusal.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from chowder.evals.result import (
    AGENT_HARNESS,
    NOT_APPLICABLE_MODALITY,
    RAW_MODEL,
    SUPPORTED,
    BenchmarkRun,
)
from chowder.growth.benchmark_registry import (
    DIRECTIONS,
    NORMALIZATION_KINDS,
    BenchmarkEntry,
    BenchmarkRegistry,
    Normalization,
    NormalizationRefused,
)
from chowder.growth.catalog import default_registry
from chowder.growth.cycle import CycleConfig, GrowthCycle
from chowder.growth.contamination import ContaminationFirewall
from chowder.growth.curriculum import CurriculumEngine
from chowder.growth.failure_bank import FailureBank
from chowder.growth.lineage import GenerationLedger, RegressionMemory
from chowder.growth.frontier_reference import SnapshotStore
from chowder.growth.metric_binding import (
    BindingRefusal,
    BoundMeasurement,
    MetricBinder,
    PromotionBindingError,
)

TARGET_ID = "local_target@2026-09"
PROTECTED_ID = "local_protected@2026-09"
BROAD_ID = "local_broad@2026-09"

# A declared loss scale, worst -> best. Anchors are fixture data here; the
# point of every test below is that they are read from the registry.
LOSS_NORMALIZATION = Normalization(
    kind="anchored_linear",
    zero_at=3.5,
    one_at=2.6,
    provenance="fixture: parent measured 3.0-3.4 on this protocol before the run",
)
RATE_NORMALIZATION = Normalization(kind="identity")


def _entry(**overrides) -> BenchmarkEntry:
    base = dict(
        benchmark_id="local_target",
        version="2026-09",
        name="Local holdout protocol (target)",
        category="instruction",
        subcategory="local protocol",
        status="RUNNABLE_PUBLIC",
        lifecycle="ACTIVE_DIAGNOSTIC",
        tier=1,
        scorer="chowder_custom",
        primary_metric="holdout_loss",
        direction="lower_is_better",
        normalization=LOSS_NORMALIZATION,
        skills=("instruction.formatting",),
        dataset_source="local://holdout",
        implementation_source="chowder custom adapter",
        source="Chowder",
        license="internal",
        release_date="2026-09-16",
        adapter="chowder_custom",
        split_policy="none",
    )
    base.update(overrides)
    return BenchmarkEntry(**base)


def _registry(*entries: BenchmarkEntry) -> BenchmarkRegistry:
    return BenchmarkRegistry(entries)


def _run(
    qualified_id: str,
    *,
    score: float | None,
    metric: str,
    samples: tuple[float, ...] = (),
    generation_version: str = "gen1",
    support: str = SUPPORTED,
    metadata: dict | None = None,
) -> BenchmarkRun:
    return BenchmarkRun(
        benchmark_qualified_id=qualified_id,
        adapter="chowder_custom",
        generation_version=generation_version,
        score=score,
        support=support,
        measurement_kind=RAW_MODEL,
        n_samples=len(samples),
        per_sample_scores=samples,
        metric=metric,
        metadata=metadata or {},
    )


def _samples(*values: float) -> tuple[float, ...]:
    return tuple(values)


# --------------------------------------------------------------------------
# The declaration itself
# --------------------------------------------------------------------------


def test_every_catalog_entry_declares_a_direction_and_a_normalization_decision():
    """No entry may leave either undecided.

    ``normalization=None`` is a *declaration* that the metric has no 0..1
    scale, not an omission -- the field has no default, so a catalog row that
    forgets it cannot be constructed at all. This test pins that the rows
    actually carry values and that exactly the declared number of them say
    "no scale".
    """
    registry = default_registry()
    unscaled: list[str] = []
    for entry in registry:
        assert entry.direction in DIRECTIONS, entry.qualified_id
        if entry.normalization is None:
            unscaled.append(entry.qualified_id)
        else:
            assert isinstance(entry.normalization, Normalization)
            assert entry.normalization.kind in NORMALIZATION_KINDS
    assert unscaled == ["kernelgen@2025-06"], (
        "the catalog's undeclared-scale set changed; a metric that gains or "
        "loses a declared scale must be a deliberate edit here"
    )


def test_a_rate_metric_declares_identity_and_an_unbounded_metric_declares_anchors():
    registry = default_registry()
    accuracy = registry.require("gpqa_diamond@2025-05-30")
    assert accuracy.direction == "higher_is_better"
    assert accuracy.normalization is not None
    assert accuracy.normalization.kind == "identity"


def test_entry_refuses_an_unknown_direction():
    with pytest.raises(ValueError, match="unknown direction"):
        _entry(direction="better")


def test_entry_refuses_an_unknown_normalization_kind():
    with pytest.raises(ValueError, match="unknown normalization kind"):
        _entry(normalization=Normalization(kind="vibes"))


def test_anchored_linear_requires_provenance():
    with pytest.raises(ValueError, match="provenance"):
        _entry(
            normalization=Normalization(kind="anchored_linear", zero_at=3.5, one_at=2.6, provenance="")
        )


def test_anchored_linear_requires_two_distinct_finite_anchors():
    with pytest.raises(ValueError, match="two distinct finite anchors"):
        _entry(normalization=Normalization(kind="anchored_linear", zero_at=3.5, one_at=3.5, provenance="x"))
    with pytest.raises(ValueError, match="two distinct finite anchors"):
        _entry(normalization=Normalization(kind="anchored_linear", zero_at=3.5, one_at=None, provenance="x"))


def test_identity_must_not_carry_anchors():
    with pytest.raises(ValueError, match="identity"):
        _entry(normalization=Normalization(kind="identity", zero_at=1.0, one_at=2.0))


def test_a_catalog_row_cannot_redeclare_its_metric_semantics():
    """Polarity and scale belong to the metric NAME, not to a row.

    Two rows declaring the same metric name with different scales would make
    every bound score of that metric ambiguous -- and the ambiguity would be
    invisible at the point of promotion. A benchmark that genuinely needs a
    different scale needs a distinct metric name.
    """
    from chowder.growth.catalog import METRIC_SEMANTICS, _with_semantics

    assert {"accuracy", "pass@1", "speedup_at_correctness"} <= set(METRIC_SEMANTICS)
    # The live catalog is the proof that no row does it today.
    default_registry()
    with pytest.raises(ValueError, match="redeclares"):
        _with_semantics(
            {
                "benchmark_id": "rogue",
                "primary_metric": "accuracy",
                "direction": "lower_is_better",
            }
        )


def test_a_metric_with_no_declared_semantics_is_refused_not_defaulted():
    from chowder.growth.catalog import semantics_for

    with pytest.raises(ValueError, match="declares no direction/normalization"):
        semantics_for("vibes_per_second")


@pytest.mark.parametrize(
    "direction,zero_at,one_at",
    [
        ("lower_is_better", 2.6, 3.5),  # worse must be the higher loss
        ("higher_is_better", 3.5, 2.6),  # better must be the higher value
    ],
)
def test_declared_scale_must_agree_with_declared_direction(direction, zero_at, one_at):
    with pytest.raises(ValueError, match="declared scale contradicts the declared direction"):
        _entry(
            direction=direction,
            normalization=Normalization(
                kind="anchored_linear", zero_at=zero_at, one_at=one_at, provenance="x"
            ),
        )


# --------------------------------------------------------------------------
# Conversion onto the declared scale
# --------------------------------------------------------------------------


def test_identity_maps_a_rate_straight_through():
    entry = _entry(
        primary_metric="accuracy",
        direction="higher_is_better",
        normalization=RATE_NORMALIZATION,
    )
    normalization = entry.normalization
    assert normalization is not None
    assert normalization.score(0.42, direction=entry.direction, qualified_id=entry.qualified_id) == 0.42


def test_lower_is_better_anchored_linear_maps_worst_to_zero_and_best_to_one():
    score = LOSS_NORMALIZATION.score
    assert score(3.5, direction="lower_is_better", qualified_id=TARGET_ID) == 0.0
    assert score(2.6, direction="lower_is_better", qualified_id=TARGET_ID) == 1.0
    assert score(3.05, direction="lower_is_better", qualified_id=TARGET_ID) == pytest.approx(0.5, abs=1e-9)


def test_a_value_outside_the_declared_domain_is_refused_not_clamped():
    # Anchored scales name their own domain, so the refusal can say what it is.
    with pytest.raises(NormalizationRefused, match="outside the declared domain"):
        LOSS_NORMALIZATION.score(3.9, direction="lower_is_better", qualified_id=TARGET_ID)
    with pytest.raises(NormalizationRefused, match="outside the declared domain"):
        LOSS_NORMALIZATION.score(1.0, direction="lower_is_better", qualified_id=TARGET_ID)
    # An identity scale's domain *is* 0..1, which is also what catches an
    # adapter reporting 84.0 for 84%.
    with pytest.raises(NormalizationRefused, match="outside the 0..1 range"):
        RATE_NORMALIZATION.score(1.4, direction="higher_is_better", qualified_id=TARGET_ID)
    with pytest.raises(NormalizationRefused, match="outside the 0..1 range"):
        RATE_NORMALIZATION.score(84.0, direction="higher_is_better", qualified_id=TARGET_ID)


def test_a_run_reporting_a_percentage_is_refused_not_divided_by_a_hundred():
    """An adapter reporting 84.0 for 84% is a unit mismatch, not a score."""
    entry = _entry(
        primary_metric="accuracy",
        direction="higher_is_better",
        normalization=RATE_NORMALIZATION,
    )
    binder = _binder(entries=(entry,))
    outcome = binder.bind(_run(TARGET_ID, score=84.0, metric="accuracy"))
    assert isinstance(outcome, BindingRefusal)
    assert "outside the 0..1 range" in outcome.reason
    assert isinstance(
        binder.bind(_run(TARGET_ID, score=0.84, metric="accuracy")), BoundMeasurement
    )


def test_a_non_finite_measurement_is_refused():
    binder = _binder()
    for raw in (float("nan"), float("inf")):
        outcome = binder.bind(_run(TARGET_ID, score=raw, metric="holdout_loss"))
        assert isinstance(outcome, BindingRefusal)
        assert "not a finite number" in outcome.reason


# --------------------------------------------------------------------------
# The binder
# --------------------------------------------------------------------------


def _binder(*, manifest: dict | None = None, entries: tuple[BenchmarkEntry, ...] | None = None) -> MetricBinder:
    registry = _registry(*(entries or (_entry(),)))
    return MetricBinder(registry, contamination=(manifest or {}).get("benchmarks", {}))


def test_a_measured_run_binds_onto_the_declared_scale():
    binder = _binder()
    outcome = binder.bind(_run(TARGET_ID, score=3.05, metric="holdout_loss", samples=_samples(3.0, 3.1)))
    assert isinstance(outcome, BoundMeasurement)
    assert outcome.raw == 3.05
    assert outcome.score == pytest.approx(0.5, abs=1e-9)
    assert outcome.result.benchmark_qualified_id == TARGET_ID
    assert outcome.result.score == pytest.approx(0.5, abs=1e-9)
    assert outcome.normalization_kind == "anchored_linear"


def test_per_sample_scores_are_normalized_onto_the_same_declared_scale():
    """Per-question evidence must land in the same space as the aggregate.

    ``evaluate_promotion`` runs its significance test on ``samples``, so
    leaving them in raw metric units while the aggregate is normalized would
    compare two different scales -- and for a loss metric it would compare
    higher-is-better in the wrong direction.
    """
    binder = _binder()
    outcome = binder.bind(
        _run(TARGET_ID, score=3.05, metric="holdout_loss", samples=_samples(3.0, 3.1))
    )
    assert isinstance(outcome, BoundMeasurement)
    assert outcome.result.samples == pytest.approx((0.5555555, 0.4444444), abs=1e-6)


def test_an_unsupported_run_is_refused_and_never_bound_as_zero():
    binder = _binder()
    outcome = binder.bind(
        _run(TARGET_ID, score=None, metric="holdout_loss", support=NOT_APPLICABLE_MODALITY)
    )
    assert isinstance(outcome, BindingRefusal)
    assert "not applicable" in outcome.reason


def test_a_supported_run_with_no_score_is_refused_as_absent_not_zero():
    """A run the model *could* have been measured on, but wasn't.

    ``score=None`` with a SUPPORTED support level is a harness that ran and
    produced nothing. Binding it as 0.0 would turn an absent measurement into
    the worst possible result.
    """
    binder = _binder()
    outcome = binder.bind(_run(TARGET_ID, score=None, metric="holdout_loss"))
    assert isinstance(outcome, BindingRefusal)
    assert "no score" in outcome.reason


def test_a_metric_name_that_disagrees_with_the_registry_is_refused():
    binder = _binder()
    outcome = binder.bind(_run(TARGET_ID, score=0.31, metric="pass@1"))
    assert isinstance(outcome, BindingRefusal)
    assert "metric" in outcome.reason and "holdout_loss" in outcome.reason


def test_an_entry_without_a_declared_scale_is_refused():
    entry = _entry(normalization=None)
    binder = _binder(entries=(entry,))
    outcome = binder.bind(_run(TARGET_ID, score=0.31, metric="holdout_loss"))
    assert isinstance(outcome, BindingRefusal)
    assert "no declared 0..1 scale" in outcome.reason


def test_a_benchmark_absent_from_the_registry_is_refused():
    binder = _binder()
    outcome = binder.bind(_run("mystery@2026-01", score=0.5, metric="holdout_loss"))
    assert isinstance(outcome, BindingRefusal)
    assert "not in the registry" in outcome.reason


def test_a_run_from_the_wrong_generation_is_refused():
    binder = _binder()
    outcome = binder.bind(
        _run(TARGET_ID, score=3.05, metric="holdout_loss", generation_version="gen0"),
        generation_version="gen1",
    )
    assert isinstance(outcome, BindingRefusal)
    assert "generation" in outcome.reason


def test_anchors_are_read_from_the_registry_never_from_the_run():
    """A run cannot supply its own scale.

    The run below carries metadata that would move the score onto a different
    scale if the binder honoured it. Reading a scale from the measurement is
    how a candidate gets to choose the denominator its own promotion is
    judged against, so the binder must ignore it.
    """
    binder = _binder()
    outcome = binder.bind(
        _run(
            TARGET_ID,
            score=3.05,
            metric="holdout_loss",
            metadata={
                "normalization": {"kind": "anchored_linear", "zero_at": 3.2, "one_at": 3.0},
                "at_min": 3.2,
                "at_max": 3.0,
                "one_at": 3.0,
                "zero_at": 3.2,
                "score": 0.95,
            },
        )
    )
    assert isinstance(outcome, BoundMeasurement)
    assert outcome.score == pytest.approx(0.5, abs=1e-9)


def test_contamination_status_comes_from_the_manifest_not_the_run():
    binder = _binder(manifest={"benchmarks": {TARGET_ID: {"status": "KNOWN_CONTAMINATION"}}})
    outcome = binder.bind(
        _run(
            TARGET_ID,
            score=3.05,
            metric="holdout_loss",
            metadata={"contamination": "CLEAN"},
        )
    )
    assert isinstance(outcome, BoundMeasurement)
    assert outcome.result.contamination == "KNOWN_CONTAMINATION"


def test_an_unchecked_benchmark_binds_with_unknown_contamination():
    binder = _binder()
    outcome = binder.bind(_run(TARGET_ID, score=3.05, metric="holdout_loss"))
    assert isinstance(outcome, BoundMeasurement)
    assert outcome.result.contamination == "UNKNOWN"


def test_the_report_names_every_refusal_so_a_score_is_never_silently_absent():
    binder = _binder()
    report = binder.bind_all(
        [
            _run(TARGET_ID, score=3.05, metric="holdout_loss"),
            _run(BROAD_ID, score=0.4, metric="accuracy"),
        ]
    )
    assert set(report.results) == {TARGET_ID}
    assert [refusal.qualified_id for refusal in report.refusals] == [BROAD_ID]
    assert "not in the registry" in report.refusals[0].reason


def test_binding_the_same_benchmark_twice_refuses_the_second():
    """Two measurements of one benchmark would silently pick one score."""
    binder = _binder()
    report = binder.bind_all(
        [
            _run(TARGET_ID, score=3.05, metric="holdout_loss"),
            _run(TARGET_ID, score=2.70, metric="holdout_loss"),
        ]
    )
    assert report.results[TARGET_ID].score == pytest.approx(0.5, abs=1e-9)
    assert len(report.refusals) == 1
    assert "bound twice" in report.refusals[0].reason


def test_a_binder_can_be_built_from_a_firewall_manifest():
    manifest = {"benchmarks": _manifest(TARGET_ID, PROTECTED_ID)["benchmarks"]}
    binder = MetricBinder.from_manifest(_registry(_entry()), manifest)
    assert binder.contamination_status(TARGET_ID) == "CLEAN"
    assert binder.contamination_status(PROTECTED_ID) == "CLEAN"
    outcome = binder.bind(_run(TARGET_ID, score=3.05, metric="holdout_loss"))
    assert isinstance(outcome, BoundMeasurement)
    assert outcome.result.contamination == "CLEAN"


def test_an_aggregate_that_contradicts_its_samples_is_recorded_not_hidden():
    """Adapters aggregate differently; the disagreement is evidence, not a veto."""
    binder = _binder()
    outcome = binder.bind(
        _run(TARGET_ID, score=3.05, metric="holdout_loss", samples=_samples(2.7, 2.8))
    )
    assert isinstance(outcome, BoundMeasurement)
    assert outcome.sample_mean == pytest.approx(5 / 6, abs=1e-9)
    assert outcome.aggregate_agrees_with_samples is False


def test_a_bound_measurement_with_no_samples_reports_no_sample_mean():
    binder = _binder()
    outcome = binder.bind(_run(TARGET_ID, score=3.05, metric="holdout_loss"))
    assert isinstance(outcome, BoundMeasurement)
    assert outcome.sample_mean is None
    assert outcome.aggregate_agrees_with_samples is None


# --------------------------------------------------------------------------
# Promotion assembly and adjudication
# --------------------------------------------------------------------------


def _three_entry_registry() -> tuple[BenchmarkEntry, ...]:
    return (
        _entry(),
        _entry(
            benchmark_id="local_protected",
            version="2026-09",
            name="Local protected protocol",
            primary_metric="protected_loss",
            tier=2,
            skills=("instruction.formatting",),
        ),
        _entry(
            benchmark_id="local_broad",
            version="2026-09",
            name="Local broad protocol",
            primary_metric="broad_loss",
            tier=3,
            skills=("instruction.formatting",),
        ),
    )


def _four_entry_registry() -> tuple[BenchmarkEntry, ...]:
    """The three-benchmark battery plus one pass@k-capable reliability eval."""
    return _three_entry_registry() + (
        _entry(
            benchmark_id="local_reliability",
            version="2026-09",
            name="Local reliability protocol (pass@k capable)",
            primary_metric="reliability_pass_rate",
            direction="higher_is_better",
            normalization=RATE_NORMALIZATION,
            tier=3,
            skills=("instruction.formatting",),
        ),
    )


# Measured raw per-item losses. Both sides are real shapes from the local
# protocol; only the candidate's target values differ.
PARENT_TARGET = _samples(3.0, 3.1, 3.0, 3.1)
STABLE = _samples(3.0, 3.0, 3.1, 3.0)
IMPROVED_TARGET = _samples(2.7, 2.8, 2.7, 2.8)


def _manifest(*ids: str, status: str = "CLEAN") -> dict:
    return {"benchmarks": {qualified_id: {"status": status} for qualified_id in ids}}


RELIABILITY_ID = "local_reliability@2026-09"

# Reliability is a pass rate on an identity scale, so raw samples are already
# the better-direction 0..1 scores the promotion rule compares.
RELIABILITY_STABLE = _samples(0.9, 0.9, 0.9, 0.9)
RELIABILITY_REGRESSED = _samples(0.8, 0.8, 0.8, 0.8)


def _reliability_run(
    raw_samples: tuple[float, ...], *, generation_version: str
) -> BenchmarkRun:
    return _loss_run(
        RELIABILITY_ID,
        "reliability_pass_rate",
        raw_samples,
        generation_version=generation_version,
    )


def _loss_run(
    qualified_id: str, metric: str, raw_samples: tuple[float, ...], *, generation_version: str
) -> BenchmarkRun:
    return _run(
        qualified_id,
        score=sum(raw_samples) / len(raw_samples),
        metric=metric,
        samples=raw_samples,
        generation_version=generation_version,
    )


def _attempt(
    generation_version: str, *, target: tuple[float, ...], protected=STABLE, broad=STABLE
) -> list[BenchmarkRun]:
    return [
        _loss_run(TARGET_ID, "holdout_loss", target, generation_version=generation_version),
        _loss_run(PROTECTED_ID, "protected_loss", protected, generation_version=generation_version),
        _loss_run(BROAD_ID, "broad_loss", broad, generation_version=generation_version),
    ]


def _binder_for_adjudication(
    *, entries: tuple[BenchmarkEntry, ...] | None = None, status: str = "CLEAN"
) -> MetricBinder:
    return MetricBinder(
        _registry(*(entries or _three_entry_registry())),
        contamination=_manifest(
            TARGET_ID, PROTECTED_ID, BROAD_ID, RELIABILITY_ID, status=status
        )["benchmarks"],
    )


def _assemble(binder: MetricBinder, *, candidate_target: tuple[float, ...]):
    return binder.promotion_input(
        candidate_version="gen1",
        parent_version="gen0",
        candidate_runs=_attempt("gen1", target=candidate_target),
        parent_runs=_attempt("gen0", target=PARENT_TARGET),
        target_benchmarks=(TARGET_ID,),
        protected_benchmarks=(PROTECTED_ID,),
        broad_battery_benchmarks=(BROAD_ID,),
    )


def test_a_candidate_that_improves_its_target_without_regressions_is_promoted():
    binder = _binder_for_adjudication()
    assembly = _assemble(binder, candidate_target=IMPROVED_TARGET)
    assert assembly.decision.verdict == "PROMOTED", assembly.decision.reasons
    # The anchor span is 0.9 raw (3.5 -> 0.0, 2.6 -> 1.0). The parent's mean
    # 3.05 sits at exactly 0.5; the candidate's 2.75 at 5/6, so the target
    # delta is (3.05 - 2.75) / 0.9. Derived from the declared anchors, not
    # from a recorded run.
    assert assembly.decision.target_deltas[TARGET_ID] == pytest.approx(0.3 / 0.9, abs=1e-9)


def test_the_identity_control_over_the_same_measurements_is_not_promoted():
    """Same real measurements on both sides: nothing improved, nothing promotes."""
    binder = _binder_for_adjudication()
    assembly = _assemble(binder, candidate_target=PARENT_TARGET)
    assert assembly.decision.verdict == "REJECTED"
    assert "no target benchmark improved" in assembly.decision.reasons


def test_a_tainted_candidate_benchmark_yields_tainted():
    binder = _binder_for_adjudication(status="KNOWN_CONTAMINATION")
    assembly = _assemble(binder, candidate_target=IMPROVED_TARGET)
    assert assembly.decision.verdict == "TAINTED"


def test_a_missing_target_benchmark_yields_inconclusive_not_promoted():
    """A partial battery refuses to certify: absent evidence is not improvement."""
    binder = _binder_for_adjudication()
    assembly = binder.promotion_input(
        candidate_version="gen1",
        parent_version="gen0",
        candidate_runs=[_run(TARGET_ID, score=None, metric="holdout_loss", support=NOT_APPLICABLE_MODALITY)],
        parent_runs=_attempt("gen0", target=PARENT_TARGET),
        target_benchmarks=(TARGET_ID,),
        protected_benchmarks=(PROTECTED_ID,),
        broad_battery_benchmarks=(BROAD_ID,),
    )
    assert assembly.decision.verdict == "INCONCLUSIVE"
    assert any(refusal.qualified_id == TARGET_ID for refusal in assembly.report.refusals)


def test_a_promotion_set_naming_an_unregistered_benchmark_is_refused():
    binder = _binder_for_adjudication()
    with pytest.raises(PromotionBindingError, match="not in the registry"):
        binder.promotion_input(
            candidate_version="gen1",
            parent_version="gen0",
            candidate_runs=_attempt("gen1", target=IMPROVED_TARGET),
            parent_runs=_attempt("gen0", target=PARENT_TARGET),
            target_benchmarks=("typo_target@2026-09",),
            protected_benchmarks=(PROTECTED_ID,),
            broad_battery_benchmarks=(BROAD_ID,),
        )


def test_the_device_envelope_reaches_the_promotion_input():
    binder = _binder_for_adjudication()
    assembly = binder.promotion_input(
        candidate_version="gen1",
        parent_version="gen0",
        candidate_runs=_attempt("gen1", target=IMPROVED_TARGET),
        parent_runs=_attempt("gen0", target=PARENT_TARGET),
        target_benchmarks=(TARGET_ID,),
        protected_benchmarks=(PROTECTED_ID,),
        broad_battery_benchmarks=(BROAD_ID,),
        device_gpu_hours=0.02,
        device_gpu_hours_ceiling=0.05,
    )
    assert assembly.promotion_input.device_gpu_hours == 0.02
    assert assembly.promotion_input.device_gpu_hours_ceiling == 0.05
    assert assembly.decision.checks["resource_envelope"] == "ok"


# --------------------------------------------------------------------------
# Cycle wiring: one Model N -> N+1 attempt, adjudicated end to end
# --------------------------------------------------------------------------


class _NullPlanner:
    def propose(self, items, *, count: int = 4):  # noqa: ANN001, ARG002
        return ()


def _cycle(tmp_path: Path) -> GrowthCycle:
    return GrowthCycle(
        CycleConfig(
            cycle_id="cycle-001",
            parent_version="gen0",
            candidate_version="gen1",
            device_gpu_hours_ceiling=1.0,
            target_benchmarks=(TARGET_ID,),
            protected_benchmarks=(PROTECTED_ID,),
            broad_battery=(BROAD_ID,),
            recipe_count=1,
        ),
        curriculum=CurriculumEngine(),
        planner=_NullPlanner(),
        failure_bank=FailureBank(),
        firewall=ContaminationFirewall(),
        ledger=GenerationLedger(tmp_path / "ledger"),
        regression_memory=RegressionMemory(tmp_path / "probes"),
        snapshots=SnapshotStore(tmp_path / "snapshots"),
        train_fn=lambda recipe, items: {},  # noqa: ARG005
    )


def test_the_cycle_adjudicates_a_measured_attempt_end_to_end(tmp_path: Path):
    cycle = _cycle(tmp_path)
    binder = _binder_for_adjudication()
    assembly = cycle.decide_promotion_from_runs(
        binder,
        candidate_runs=_attempt("gen1", target=IMPROVED_TARGET),
        parent_runs=_attempt("gen0", target=PARENT_TARGET),
        device_gpu_hours=0.02,
    )
    assert assembly.decision.verdict == "PROMOTED"
    assert assembly.decision.checks["evidence_integrity"] == "ok"


def test_the_cycle_passes_its_declared_tolerances_into_promotion(tmp_path: Path):
    """A tolerance is a declared gate; the cycle's config must reach it.

    The same measurements that promote under the default 0.02 bar must not
    promote under a 0.5 bar. If the config never reached promotion, both runs
    would agree and this test would be the only place the gap could show.
    """
    strict = GrowthCycle(
        CycleConfig(
            cycle_id="cycle-003",
            parent_version="gen0",
            candidate_version="gen1",
            device_gpu_hours_ceiling=1.0,
            target_benchmarks=(TARGET_ID,),
            protected_benchmarks=(PROTECTED_ID,),
            broad_battery=(BROAD_ID,),
            recipe_count=1,
            min_target_improvement=0.5,
        ),
        curriculum=CurriculumEngine(),
        planner=_NullPlanner(),
        failure_bank=FailureBank(),
        firewall=ContaminationFirewall(),
        ledger=GenerationLedger(tmp_path / "ledger"),
        regression_memory=RegressionMemory(tmp_path / "probes"),
        snapshots=SnapshotStore(tmp_path / "snapshots"),
        train_fn=lambda recipe, items: {},  # noqa: ARG005
    )
    default_cycle = _cycle(tmp_path)
    binder = _binder_for_adjudication()
    permissive = default_cycle.decide_promotion_from_runs(
        binder,
        candidate_runs=_attempt("gen1", target=IMPROVED_TARGET),
        parent_runs=_attempt("gen0", target=PARENT_TARGET),
    )
    demanding = strict.decide_promotion_from_runs(
        binder,
        candidate_runs=_attempt("gen1", target=IMPROVED_TARGET),
        parent_runs=_attempt("gen0", target=PARENT_TARGET),
    )
    assert permissive.promotion_input.min_target_improvement == 0.02
    assert demanding.promotion_input.min_target_improvement == 0.5
    assert permissive.decision.verdict == "PROMOTED"
    assert demanding.decision.verdict != "PROMOTED"


def test_the_cycle_passes_its_device_ceiling_into_promotion(tmp_path: Path):
    """The cycle's own budget ceiling is a hard promotion gate, not a note.

    A candidate that cleared every measured bar by training past the device
    envelope it was declared under must not promote: the envelope is part of
    the predeclaration, and a verdict computed without it would read as
    success for a run the program said it would not pay for.
    """
    def _cycle_with_ceiling(ceiling: float) -> GrowthCycle:
        return GrowthCycle(
            CycleConfig(
                cycle_id="cycle-004",
                parent_version="gen0",
                candidate_version="gen1",
                device_gpu_hours_ceiling=ceiling,
                target_benchmarks=(TARGET_ID,),
                protected_benchmarks=(PROTECTED_ID,),
                broad_battery=(BROAD_ID,),
                recipe_count=1,
            ),
            curriculum=CurriculumEngine(),
            planner=_NullPlanner(),
            failure_bank=FailureBank(),
            firewall=ContaminationFirewall(),
            ledger=GenerationLedger(tmp_path / f"ledger-{ceiling}"),
            regression_memory=RegressionMemory(tmp_path / f"probes-{ceiling}"),
            snapshots=SnapshotStore(tmp_path / f"snapshots-{ceiling}"),
            train_fn=lambda recipe, items: {},  # noqa: ARG005
        )

    binder = _binder_for_adjudication()
    within = _cycle_with_ceiling(1.0).decide_promotion_from_runs(
        binder,
        candidate_runs=_attempt("gen1", target=IMPROVED_TARGET),
        parent_runs=_attempt("gen0", target=PARENT_TARGET),
        device_gpu_hours=0.02,
    )
    over = _cycle_with_ceiling(0.01).decide_promotion_from_runs(
        binder,
        candidate_runs=_attempt("gen1", target=IMPROVED_TARGET),
        parent_runs=_attempt("gen0", target=PARENT_TARGET),
        device_gpu_hours=0.02,
    )
    assert within.promotion_input.device_gpu_hours_ceiling == 1.0
    assert within.decision.checks["resource_envelope"] == "ok"
    assert within.decision.verdict == "PROMOTED"
    assert over.promotion_input.device_gpu_hours_ceiling == 0.01
    assert over.decision.checks["resource_envelope"] == "violated"
    assert over.decision.verdict == "REJECTED"


def test_the_cycle_refuses_a_promotion_set_naming_an_unregistered_benchmark(tmp_path: Path):
    """A promotion set is a declaration; naming an undeclared benchmark refuses."""
    cycle = GrowthCycle(
        CycleConfig(
            cycle_id="cycle-002",
            parent_version="gen0",
            candidate_version="gen1",
            device_gpu_hours_ceiling=1.0,
            target_benchmarks=("typo_target@2026-09",),
            protected_benchmarks=(PROTECTED_ID,),
            broad_battery=(BROAD_ID,),
            recipe_count=1,
        ),
        curriculum=CurriculumEngine(),
        planner=_NullPlanner(),
        failure_bank=FailureBank(),
        firewall=ContaminationFirewall(),
        ledger=GenerationLedger(tmp_path / "ledger"),
        regression_memory=RegressionMemory(tmp_path / "probes"),
        snapshots=SnapshotStore(tmp_path / "snapshots"),
        train_fn=lambda recipe, items: {},  # noqa: ARG005
    )
    with pytest.raises(PromotionBindingError, match="not in the registry"):
        cycle.decide_promotion_from_runs(
            _binder_for_adjudication(),
            candidate_runs=_attempt("gen1", target=IMPROVED_TARGET),
            parent_runs=_attempt("gen0", target=PARENT_TARGET),
        )


def test_the_cycle_records_a_rejected_candidate_as_evidence(tmp_path: Path):
    cycle = _cycle(tmp_path)
    binder = _binder_for_adjudication()
    assembly = cycle.decide_promotion_from_runs(
        binder,
        candidate_runs=_attempt("gen1", target=PARENT_TARGET),
        parent_runs=_attempt("gen0", target=PARENT_TARGET),
    )
    outcome = cycle.finalize(
        assembly.decision,
        base_model={"model_id": "tiny"},
        dataset_manifest_ref="manifest.json",
        curriculum_manifest_ref="curriculum.json",
        recipe={"recipe_id": "recipe-00"},
        training_evidence_ref="evidence.json",
        evaluation_report_ref="report.json",
    )
    assert outcome.verdict == "REJECTED"
    # The record is a plain serializable mapping, not a live object: a rejected
    # candidate is durable evidence, so it must survive a round trip through
    # JSON the same way a promoted one does.
    assert isinstance(outcome.promotion, dict)
    assert dataclasses.asdict(outcome) is not None  # the outcome itself serializes
    assert json.loads(json.dumps(outcome.to_dict()))["promotion"]["verdict"] == "REJECTED"
    assert outcome.to_dict()["promotion"]["verdict"] == "REJECTED"
    assert outcome.to_dict()["promotion"]["checks"]["target_improvement"] == "not met"


def test_a_reliability_regression_rejects_a_candidate_that_improved_its_target():
    """Reliability is a hard gate: passing targets cannot buy back a drop.

    PromotionInput has declared reliability_benchmarks since the growth system
    landed, and the module docstring has always promised a reliability
    comparison -- but evaluate_promotion never read the set. This test pins
    the promised check: a pass@k-capable eval that drops past the declared
    tolerance rejects the candidate no matter what the target improved.
    """
    binder = _binder_for_adjudication(entries=_four_entry_registry())
    assembly = binder.promotion_input(
        candidate_version="gen1",
        parent_version="gen0",
        candidate_runs=_attempt("gen1", target=IMPROVED_TARGET)
        + [_reliability_run(RELIABILITY_REGRESSED, generation_version="gen1")],
        parent_runs=_attempt("gen0", target=PARENT_TARGET)
        + [_reliability_run(RELIABILITY_STABLE, generation_version="gen0")],
        target_benchmarks=(TARGET_ID,),
        protected_benchmarks=(PROTECTED_ID,),
        broad_battery_benchmarks=(BROAD_ID,),
        reliability_benchmarks=(RELIABILITY_ID,),
    )
    assert assembly.decision.checks[f"reliability:{RELIABILITY_ID}"] == "violated"
    assert assembly.decision.checks["reliability"] == "violated"
    assert assembly.decision.verdict == "REJECTED"
    assert any("reliability regression" in reason for reason in assembly.decision.reasons)


def test_a_stable_reliability_check_leaves_a_clean_promotion_intact():
    binder = _binder_for_adjudication(entries=_four_entry_registry())
    assembly = binder.promotion_input(
        candidate_version="gen1",
        parent_version="gen0",
        candidate_runs=_attempt("gen1", target=IMPROVED_TARGET)
        + [_reliability_run(RELIABILITY_STABLE, generation_version="gen1")],
        parent_runs=_attempt("gen0", target=PARENT_TARGET)
        + [_reliability_run(RELIABILITY_STABLE, generation_version="gen0")],
        target_benchmarks=(TARGET_ID,),
        protected_benchmarks=(PROTECTED_ID,),
        broad_battery_benchmarks=(BROAD_ID,),
        reliability_benchmarks=(RELIABILITY_ID,),
    )
    assert assembly.decision.checks["reliability"] == "ok"
    assert assembly.decision.verdict == "PROMOTED", assembly.decision.reasons


def test_a_declared_reliability_set_that_was_never_measured_is_inconclusive_not_ok():
    """Declaring the set is a predeclaration; skipping the measurement is not ok.

    A protected benchmark unmeasured is already inconclusive rather than ok.
    Reliability follows the same rule: an empty set is 'unmeasured' (nothing
    was promised), but a *declared* set with no results must block promotion,
    or a candidate could dodge the gate by simply never running the eval.
    """
    binder = _binder_for_adjudication(entries=_four_entry_registry())
    assembly = binder.promotion_input(
        candidate_version="gen1",
        parent_version="gen0",
        candidate_runs=_attempt("gen1", target=IMPROVED_TARGET),
        parent_runs=_attempt("gen0", target=PARENT_TARGET),
        target_benchmarks=(TARGET_ID,),
        protected_benchmarks=(PROTECTED_ID,),
        broad_battery_benchmarks=(BROAD_ID,),
        reliability_benchmarks=(RELIABILITY_ID,),
    )
    assert assembly.decision.checks["reliability"] == "inconclusive"
    assert assembly.decision.checks[f"reliability:{RELIABILITY_ID}"] == "inconclusive"
    assert assembly.decision.verdict != "PROMOTED"


def test_an_undeclared_reliability_set_reports_unmeasured_and_promotes():
    """The empty default stays backward compatible: nothing promised, nothing gated."""
    binder = _binder_for_adjudication()
    assembly = _assemble(binder, candidate_target=IMPROVED_TARGET)
    assert assembly.decision.checks["reliability"] == "unmeasured"
    assert assembly.decision.verdict == "PROMOTED", assembly.decision.reasons


def test_the_cycle_passes_its_reliability_set_into_promotion(tmp_path: Path):
    """The cycle's reliability declaration must actually reach promotion.

    The binder has accepted a reliability set since the binding landed, but
    no cycle could supply one: CycleConfig had no field, so the support was
    unreachable and every growth cycle silently ran without the gate. The
    same evidence must reject under a cycle that declares the set and
    promote under one that does not -- that difference is the proof the knob
    is wired.
    """
    def _cycle_with_reliability(reliability: tuple[str, ...], tag: str) -> GrowthCycle:
        return GrowthCycle(
            CycleConfig(
                cycle_id=f"cycle-reliability-{tag}",
                parent_version="gen0",
                candidate_version="gen1",
                device_gpu_hours_ceiling=1.0,
                target_benchmarks=(TARGET_ID,),
                protected_benchmarks=(PROTECTED_ID,),
                broad_battery=(BROAD_ID,),
                reliability_benchmarks=reliability,
                recipe_count=1,
            ),
            curriculum=CurriculumEngine(),
            planner=_NullPlanner(),
            failure_bank=FailureBank(),
            firewall=ContaminationFirewall(),
            ledger=GenerationLedger(tmp_path / f"ledger-{tag}"),
            regression_memory=RegressionMemory(tmp_path / f"probes-{tag}"),
            snapshots=SnapshotStore(tmp_path / f"snapshots-{tag}"),
            train_fn=lambda recipe, items: {},  # noqa: ARG005
        )

    binder = _binder_for_adjudication(entries=_four_entry_registry())
    candidate_runs = _attempt("gen1", target=IMPROVED_TARGET) + [
        _reliability_run(RELIABILITY_REGRESSED, generation_version="gen1")
    ]
    parent_runs = _attempt("gen0", target=PARENT_TARGET) + [
        _reliability_run(RELIABILITY_STABLE, generation_version="gen0")
    ]
    gated = _cycle_with_reliability((RELIABILITY_ID,), "gated").decide_promotion_from_runs(
        binder, candidate_runs=candidate_runs, parent_runs=parent_runs
    )
    unguarded = _cycle_with_reliability((), "unguarded").decide_promotion_from_runs(
        binder, candidate_runs=candidate_runs, parent_runs=parent_runs
    )
    assert unguarded.promotion_input.reliability_benchmarks == ()
    assert unguarded.decision.verdict == "PROMOTED"
    assert gated.promotion_input.reliability_benchmarks == (RELIABILITY_ID,)
    assert gated.decision.checks["reliability"] == "violated"
    assert gated.decision.verdict == "REJECTED"
