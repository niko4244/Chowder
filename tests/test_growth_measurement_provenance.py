"""Measurement provenance: candidate gates only candidate measurements.

The critical fabrication this suite pins shut: a parent row relabeled with
the candidate's generation string. The generation label is caller-assigned;
the measurement origin is evidence. Promotion must read the origin.
"""

from __future__ import annotations

import dataclasses

import pytest

from chowder.evals.result import (
    CARRIED_REFERENCE,
    MEASURED_PARENT,
    MEASURED_THIS_GENERATION,
    UNMEASURED,
    BenchmarkRun,
)
from chowder.growth.benchmark_registry import BenchmarkEntry, Normalization
from chowder.growth.metric_binding import MetricBinder, PromotionBindingError
from chowder.growth.promotion import (
    BenchmarkResult,
    PromotionInput,
    evaluate_promotion,
)


def _entry(qualified_id: str) -> BenchmarkEntry:
    return BenchmarkEntry(
        benchmark_id=qualified_id.split("@")[0],
        version=qualified_id.split("@")[1],
        name=qualified_id,
        category="math",
        subcategory="test",
        status="RUNNABLE_PUBLIC",
        lifecycle="ACTIVE_DIAGNOSTIC",
        tier=1,
        scorer="lm_eval",
        primary_metric="accuracy",
        direction="higher_is_better",
        normalization=Normalization(kind="identity"),
        skills=("math.algebra",),
        dataset_source="local://test",
        implementation_source="lm-evaluation-harness",
        source="Chowder",
        license="internal",
        release_date="2024-04-01",
        adapter="lm_eval",
        split_policy="none",
    )


def _registry(*entries: BenchmarkEntry):
    from chowder.growth.benchmark_registry import BenchmarkRegistry

    return BenchmarkRegistry(entries or (_entry("math500@2024-04"),))


def _run(
    qid: str,
    version: str,
    score: float,
    *,
    origin: str = MEASURED_THIS_GENERATION,
    samples: tuple[float, ...] = (),
) -> BenchmarkRun:
    return BenchmarkRun(
        benchmark_qualified_id=qid,
        adapter="lm_eval",
        generation_version=version,
        score=score,
        support="SUPPORTED",
        measurement_kind="raw_model",
        n_samples=len(samples),
        metric="accuracy",
        per_sample_scores=samples,
        measurement_origin=origin,
    )


def _result(
    qid: str,
    score: float,
    *,
    origin: str = MEASURED_THIS_GENERATION,
    samples: tuple[float, ...] = (),
) -> BenchmarkResult:
    return BenchmarkResult(
        benchmark_qualified_id=qid,
        score=score,
        samples=samples,
        contamination="CLEAN",
        measurement_origin=origin,
    )


# ---------------- binding-level provenance ----------------


def test_parent_row_relabeled_as_candidate_refuses_binding() -> None:
    from chowder.growth.metric_binding import BindingRefusal

    binder = MetricBinder(_registry())
    parent_row = _run(
        "math500@2024-04",
        "gen1",  # relabeled!
        0.0,
        origin=MEASURED_PARENT,
        samples=(0.0,) * 8,
    )
    refusal = binder.bind(parent_row, generation_version="gen1", role="candidate")
    assert isinstance(refusal, BindingRefusal)
    assert "parent-measured" in refusal.reason


def test_carried_reference_row_refuses_on_candidate_side() -> None:
    from chowder.growth.metric_binding import BindingRefusal

    binder = MetricBinder(_registry())
    carried = _run(
        "math500@2024-04", "gen1", 0.0, origin=CARRIED_REFERENCE, samples=(0.0,) * 8
    )
    refusal = binder.bind(carried, generation_version="gen1", role="candidate")
    assert isinstance(refusal, BindingRefusal)
    assert "carried reference" in refusal.reason


def test_legacy_row_without_provenance_refuses_on_candidate_side() -> None:
    """Historical rows predate the origin field; they are not silently trusted."""
    from chowder.growth.metric_binding import BindingRefusal

    binder = MetricBinder(_registry())
    legacy = _run("math500@2024-04", "gen1", 0.0, origin=UNMEASURED, samples=(0.0,) * 8)
    refusal = binder.bind(legacy, generation_version="gen1", role="candidate")
    assert isinstance(refusal, BindingRefusal)
    assert "MEASURED_THIS_GENERATION" in refusal.reason


def test_candidate_measured_row_binds_on_candidate_side() -> None:
    binder = MetricBinder(_registry())
    row = _run("math500@2024-04", "gen1", 0.25, samples=(0.0, 0.5))
    outcome = binder.bind(row, generation_version="gen1", role="candidate")
    assert outcome.result.measurement_origin == MEASURED_THIS_GENERATION


def test_parent_side_accepts_parent_measured_and_legacy_rows() -> None:
    binder = MetricBinder(_registry())
    legacy_parent = _run("math500@2024-04", "gen0", 0.0, origin=UNMEASURED)
    assert binder.bind(legacy_parent, generation_version="gen0", role="parent") is not None
    parent_row = _run(
        "math500@2024-04", "gen0", 0.0, origin=MEASURED_THIS_GENERATION
    )
    assert binder.bind(parent_row, generation_version="gen0", role="parent") is not None


# ---------------- gate-level semantics ----------------


def test_parent_floor_zero_with_candidate_unmeasured_is_not_regression_pass() -> None:
    """Parent 0.0 + candidate never evaluated: inconclusive, never 'ok'."""
    data = PromotionInput(
        candidate_version="gen1",
        parent_version="gen0",
        target_benchmarks=(),
        candidate_results={},
        parent_results={
            "math500@2024-04": _result("math500@2024-04", 0.0, origin=UNMEASURED)
        },
        protected_benchmarks=("math500@2024-04",),
        broad_battery_benchmarks=("math500@2024-04",),
    )
    decision = evaluate_promotion(data)
    assert decision.checks["protected:math500@2024-04"] == "inconclusive"
    assert decision.checks["protected_regression"] == "inconclusive"
    assert decision.checks["broad_battery"] == "inconclusive"


def test_carried_row_cannot_certify_not_regressed_even_at_identical_score() -> None:
    """The exact gen1 defect: identical copied arrays must not read 'ok'."""
    data = PromotionInput(
        candidate_version="gen1",
        parent_version="gen0",
        target_benchmarks=(),
        candidate_results={
            "math500@2024-04": _result(
                "math500@2024-04",
                0.0,
                origin=CARRIED_REFERENCE,
                samples=(0.0,) * 48,
            )
        },
        parent_results={
            "math500@2024-04": _result(
                "math500@2024-04", 0.0, origin=UNMEASURED, samples=(0.0,) * 48
            )
        },
        protected_benchmarks=("math500@2024-04",),
        broad_battery_benchmarks=("math500@2024-04",),
    )
    decision = evaluate_promotion(data)
    assert decision.checks["protected:math500@2024-04"] == "inconclusive"
    assert decision.checks["protected_regression"] == "inconclusive"
    assert decision.checks["broad_battery"] == "inconclusive"


def test_both_arms_actually_measured_at_floor_is_evaluable() -> None:
    """Parent 0.0 measured, candidate 0.0 measured: real adjudication happens."""
    data = PromotionInput(
        candidate_version="gen1",
        parent_version="gen0",
        target_benchmarks=(),
        candidate_results={
            "math500@2024-04": _result(
                "math500@2024-04", 0.0, samples=(0.0,) * 24
            )
        },
        parent_results={
            "math500@2024-04": _result(
                "math500@2024-04", 0.0, origin=UNMEASURED, samples=(0.0,) * 24
            )
        },
        protected_benchmarks=("math500@2024-04",),
        broad_battery_benchmarks=("math500@2024-04",),
    )
    decision = evaluate_promotion(data)
    assert decision.checks["protected:math500@2024-04"] == "ok"
    assert decision.checks["protected_regression"] == "ok"
    assert decision.checks["broad_battery"] == "ok"


def test_candidate_regression_beyond_tolerance_violates() -> None:
    data = PromotionInput(
        candidate_version="gen1",
        parent_version="gen0",
        target_benchmarks=(),
        candidate_results={
            "math500@2024-04": _result(
                "math500@2024-04", 0.0, samples=(0.0,) * 24
            )
        },
        parent_results={
            "math500@2024-04": _result(
                "math500@2024-04", 0.10, origin=UNMEASURED, samples=(0.10,) * 24
            )
        },
        protected_benchmarks=("math500@2024-04",),
        broad_battery_benchmarks=("math500@2024-04",),
        max_protected_regression=0.02,
    )
    decision = evaluate_promotion(data)
    assert decision.checks["protected:math500@2024-04"] == "violated"
    assert decision.verdict == "REJECTED"


def test_identical_copied_arrays_cannot_manufacture_target_improvement() -> None:
    """Copied arrays with a relabeled generation must not pair at all."""
    binder = MetricBinder(_registry(_entry("skill@v1")))
    carried = _run(
        "skill@v1", "gen1", 1.0, origin=CARRIED_REFERENCE, samples=(1.0,) * 16
    )
    parent = _run("skill@v1", "gen0", 0.0, origin=UNMEASURED, samples=(0.0,) * 16)
    assembly = binder.promotion_input(
        candidate_version="gen1",
        parent_version="gen0",
        candidate_runs=[carried],
        parent_runs=[parent],
        target_benchmarks=("skill@v1",),
    )
    # The carried row is refused on the candidate side, so the target gate
    # cannot see it at all.
    assert assembly.report.refusals
    assert "skill@v1" not in assembly.report.results
    assert assembly.decision.checks["target_improvement"] == "not met"


def test_benchmark_run_rejects_unknown_origin() -> None:
    with pytest.raises(ValueError, match="measurement_origin"):
        BenchmarkRun(
            benchmark_qualified_id="math500@2024-04",
            adapter="lm_eval",
            generation_version="gen1",
            score=0.0,
            measurement_origin="SOMETHING_ELSE",
        )


def test_result_to_dict_roundtrips_origin() -> None:
    r = _result("math500@2024-04", 0.5, origin=MEASURED_PARENT)
    assert dataclasses.asdict(r)["measurement_origin"] == MEASURED_PARENT
