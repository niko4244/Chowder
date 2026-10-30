"""Bounded, independent Q8-vs-Q4 sampled-weight research, no model inference."""

import math

import pytest

from chowder_batch.reap_q8_q4_paired_numeric import (
    Q4_SHA, Q8_SHA, _aligned_offsets, _checked_publication_claims, _summarize,
)


def _claim(kind):
    return {
        "schema": f"chowder.reap128-v2-gguf-{kind}-chunked-final.v1",
        "source_gguf_sha256": "80341aa776ad9b5f937c21b081b6d76de4a2ea277e5fb88c9ae2634a9da8f4c6",
        "output_sha256": Q8_SHA if kind == "q8" else Q4_SHA,
        "status": ("Q8_TENSOR_EXACT_CONSTRUCTED_NOT_QUALITY_QUALIFIED" if kind == "q8"
                   else "Q4_K_M_TENSOR_EXACT_CONSTRUCTED_NOT_QUALITY_QUALIFIED"),
        "quality_gate": "NOT_EVALUATED", "gpu_fit_gate": "NOT_EVALUATED",
        "chunked_reference_receipts": {
            "all_tensors": {"blk.0.ffn_down_exps.weight": {
                "shape": [128, 2048, 512], "tensor_type": "Q8_0" if kind == "q8" else "Q4_K",
                "raw_sha256": "a" * 64}},
        },
    }


def test_copublished_models_require_one_bf16_source_and_no_prior_gate():
    assert _checked_publication_claims(_claim("q8"), _claim("q4")) is None


@pytest.mark.parametrize("which,key,bad", [
    ("q8", "source_gguf_sha256", "0" * 64),
    ("q4", "source_gguf_sha256", "0" * 64),
    ("q8", "quality_gate", "PASS"),
    ("q4", "gpu_fit_gate", "PASS"),
    ("q8", "status", "QUALITY_PASS"),
    ("q4", "output_sha256", "garbled"),
    ("q8", "schema", "wrong"),
])
def test_divergent_ancestry_or_fake_quality_promotion_refused(which, key, bad):
    left, right = _claim("q8"), _claim("q4")
    (left if which == "q8" else right)[key] = bad
    with pytest.raises(ValueError):
        _checked_publication_claims(left, right)


def test_same_four_deterministic_q4_block_aligned_offsets_for_both_models():
    offsets = _aligned_offsets(128 * 512 * 2048)
    assert len(offsets) == 4
    assert offsets[0] == 0
    assert offsets[-1] <= 128 * 512 * 2048 - 256
    assert all(x % 256 == 0 for x in offsets)


@pytest.mark.parametrize("elements", [0, 255, 257, -256, 258])
def test_unaligned_or_small_tensors_fail_closed(elements):
    with pytest.raises(ValueError):
        _aligned_offsets(elements)


def test_statistical_weight_error_summary_is_exact_and_non_gate():
    rows = [{"count": 256, "max_abs_error": 2.0, "sum_abs_error": 256.0,
             "sum_squared_error": 512.0},
            {"count": 256, "max_abs_error": 1.0, "sum_abs_error": 128.0,
             "sum_squared_error": 128.0}]
    output = _summarize(rows)
    assert output["total_compared_values"] == 512
    assert output["max_abs_error"] == 2.0
    assert output["mean_abs_error"] == 0.75
    assert output["root_mean_squared_error"] == math.sqrt(1.25)
    assert output["status"] == "DESCRIPTIVE_Q8_Q4_WEIGHT_NUMERICS_NOT_QUALITY_GATE"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, True])
def test_nonnumeric_and_negative_error_statistics_cannot_be_scored(bad):
    with pytest.raises(ValueError):
        _summarize([{"count": 256, "max_abs_error": bad, "sum_abs_error": 0.0,
                     "sum_squared_error": 0.0}])
