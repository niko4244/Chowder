"""Campaign scoreboard tests (public benchmarks + Fable standing reference).

Enforces the properties the program doc requires: historical targets
recorded verbatim and never lowered; every score traceable to evidence;
honest absence rendered as None, not zero; a signed scoreboard digest;
and structural separation between public and protected scoreboards.
"""
from __future__ import annotations

import pytest

from chowder.campaign_scoreboard import (
    EFFICIENCY_METRICS,
    FABLE_COMPARISON_DIMENSIONS,
    HISTORICAL_TARGETS,
    BenchmarkScore,
    CampaignScoreboard,
    FableComparisonEntry,
    FableReference,
    advance_best,
    capability_per_active_billion,
    capability_per_gpu_second,
)


def _scoreboard(**overrides) -> CampaignScoreboard:
    kwargs = dict(
        model_label="chowder-qwen38-gen0",
        model_manifest_sha256="a" * 64,
        targets=HISTORICAL_TARGETS,
        scores=(
            BenchmarkScore("MMLU", 0.87, "eval-run-1"),
            BenchmarkScore("GSM8K", 0.91, "eval-run-1"),
            BenchmarkScore("HumanEval", 0.62, "eval-run-2"),
        ),
    )
    kwargs.update(overrides)
    return CampaignScoreboard(**kwargs)


def test_historical_targets_recorded_verbatim() -> None:
    assert [(t.benchmark, t.threshold) for t in HISTORICAL_TARGETS] == [
        ("MMLU", 0.90),
        ("GSM8K", 0.90),
        ("HumanEval", 0.60),
        ("MATH", 0.40),
    ]


def test_threshold_is_exclusive() -> None:
    mmlu = HISTORICAL_TARGETS[0]
    assert mmlu.passed(0.90) is False  # exactly at the threshold has not met "> 0.90"
    assert mmlu.passed(0.900001) is True
    assert mmlu.passed(0.89) is False


def test_score_without_evidence_ref_is_rejected() -> None:
    with pytest.raises(ValueError, match="not evidence"):
        BenchmarkScore("MMLU", 0.87, "  ")


def test_protected_suite_names_are_refused() -> None:
    with pytest.raises(ValueError, match="never enter the public scoreboard"):
        BenchmarkScore("suite-reasoning-v1", 0.9, "eval-run-1")


def test_score_for_undeclared_benchmark_is_rejected() -> None:
    with pytest.raises(ValueError, match="undeclared benchmark"):
        _scoreboard(scores=(BenchmarkScore("ARC-AGI", 0.1, "eval-run-9"),))


def test_unmeasured_benchmark_renders_as_none_not_zero() -> None:
    rendered = _scoreboard().render()
    math_row = rendered["rows"]["MATH"]
    assert math_row["current"] is None
    assert math_row["pass"] is False
    assert math_row["best"] is None  # no measurement ever: honest absence
    assert rendered["all_targets_passed"] is False
    assert rendered["measured_count"] == 3


def test_render_columns_and_pass_fail() -> None:
    baseline = {
        "MMLU": BenchmarkScore("MMLU", 0.85, "eval-run-0"),
        "GSM8K": BenchmarkScore("GSM8K", 0.93, "eval-run-0"),
    }
    rendered = _scoreboard(baseline=baseline).render()
    mmlu = rendered["rows"]["MMLU"]
    assert mmlu["baseline"] == 0.85
    assert mmlu["current"] == 0.87
    assert mmlu["delta"] == pytest.approx(0.02)
    assert mmlu["pass"] is False  # 0.87 < 0.90 target
    gsm8k = rendered["rows"]["GSM8K"]
    assert gsm8k["pass"] is True  # 0.91 > 0.90
    assert gsm8k["delta"] == pytest.approx(-0.02)  # regression against baseline


def test_best_column_carries_best_ever() -> None:
    rendered = _scoreboard(
        best_prior={"HumanEval": 0.71}, baseline={"HumanEval": BenchmarkScore("HumanEval", 0.55, "eval-run-0")}
    ).render()
    row = rendered["rows"]["HumanEval"]
    assert row["best"] == 0.71  # best_prior survives a worse current score
    assert row["delta"] == pytest.approx(0.07)


def test_scoreboard_digest_detects_tampering() -> None:
    sb = _scoreboard()
    data = sb.to_dict()
    data["scores"][0]["score"] = 0.99  # silently retarget the scoreboard
    with pytest.raises(ValueError, match="modified after signing"):
        CampaignScoreboard.from_dict(data)
    round_trip = CampaignScoreboard.from_dict(sb.to_dict())
    assert round_trip.scoreboard_sha256 == sb.scoreboard_sha256


def test_targets_cannot_be_lowered_in_a_serialized_scoreboard() -> None:
    sb = _scoreboard()
    data = sb.to_dict()
    data["targets"][3]["threshold"] = 0.10  # quietly "pass" MATH
    with pytest.raises(ValueError, match="modified after signing"):
        CampaignScoreboard.from_dict(data)


def test_advance_best_folds_new_scores() -> None:
    sb = _scoreboard()
    merged = advance_best({"MMLU": 0.88}, sb)
    assert merged["MMLU"] == 0.88  # prior best survives
    assert merged["GSM8K"] == 0.91
    assert merged["HumanEval"] == 0.62


def test_fable_entry_requires_both_evidence_refs() -> None:
    with pytest.raises(ValueError, match="fable_evidence_ref"):
        FableComparisonEntry("coding", 0.6, "eval-run-1", 0.5, " ")
    with pytest.raises(ValueError, match="not a declared Fable comparison dimension"):
        FableComparisonEntry("suite-coding-v1", 0.6, "a", 0.5, "b")


def test_fable_parity_requires_every_dimension() -> None:
    partial = FableReference(
        our_model_label="gen0",
        our_model_manifest_sha256=None,
        fable_revision="deadbeef",
        fable_manifest_sha256=None,
        entries=(
            FableComparisonEntry("coding", 0.6, "eval-1", 0.5, "fable-1"),
            FableComparisonEntry("reasoning", 0.7, "eval-1", 0.6, "fable-1"),
        ),
    )
    assert partial.parity_claim_allowed() is False
    assert set(partial.render()["unmeasured_dimensions"]) == set(FABLE_COMPARISON_DIMENSIONS) - {
        "coding",
        "reasoning",
    }

    full = FableReference(
        our_model_label="gen0",
        our_model_manifest_sha256="a" * 64,
        fable_revision="deadbeef",
        fable_manifest_sha256="b" * 64,
        entries=tuple(
            FableComparisonEntry(d, 0.6, "eval-1", 0.5, "fable-1")
            for d in FABLE_COMPARISON_DIMENSIONS
        ),
    )
    assert full.parity_claim_allowed() is True
    assert full.render()["unmeasured_dimensions"] == []


def test_efficiency_metrics_and_edges() -> None:
    assert capability_per_active_billion(0.8, 4.0) == pytest.approx(0.2)
    assert capability_per_gpu_second(1.0, 100.0) == pytest.approx(0.01)
    with pytest.raises(ValueError):
        capability_per_active_billion(0.8, 0.0)
    with pytest.raises(ValueError):
        capability_per_active_billion(-0.1, 4.0)
    assert set(EFFICIENCY_METRICS) == {
        "capability_per_active_billion",
        "capability_per_gpu_second",
        "coding_score_per_active_billion",
        "reasoning_score_per_active_billion",
    }
