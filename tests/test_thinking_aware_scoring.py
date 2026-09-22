"""Tests for protocol v2: thinking-aware answer extraction and its markers.

Motivated by real evidence: retry6 of the A/B tournament scored 0.0 on all
54 items because Qwen3.8 emits chain-of-thought, closes it with
``</think>``, and answers afterwards -- while the worker scored the whole
raw generation (preamble included) against the expected value. The real
retry6 knowledge item (thinking + `</think>` + `W` vs expected `w`) is
pinned here as a regression test.
"""

from __future__ import annotations

import chowder.parent_eval as pe
from chowder.evaluators.base_text_worker import (
    _final_answer,
    _reasoning_answer,
    _score,
)


# ---- extraction ---------------------------------------------------------------


def test_closed_thinking_extracts_final_answer() -> None:
    raw = "We need answer user: \"symbol?\" Tungsten symbol W.\n</think>\n\nW"
    assert _final_answer(raw) == "\n\nW"


def test_no_marker_means_whole_prediction() -> None:
    assert _final_answer("Mercury") == "Mercury"


def test_unclosed_thinking_means_no_answer() -> None:
    assert _final_answer("<think>let me work this out") == ""


def test_uses_last_close_marker() -> None:
    raw = "<think>first</think>intermediate <think>revised</think>FINAL"
    assert _final_answer(raw) == "FINAL"


# ---- scoring through the extraction -------------------------------------------


def test_normalized_match_ignores_thinking_preamble() -> None:
    raw = 'We need answer user: "tungsten?" Need final only.\n</think>\n\nW'
    assert _score(raw, "w", "normalized_exact_match") == 1.0


def test_unclosed_thinking_scores_zero() -> None:
    assert _score("<think>still reasoning about it", "w", "normalized_exact_match") == 0.0


def test_plain_prediction_unchanged() -> None:
    assert _score("Mercury", "mercury", "normalized_exact_match") == 1.0
    assert _score("Venus", "mercury", "normalized_exact_match") == 0.0


# ---- reasoning_answer_match: templates that open <think> themselves ----------


def test_reasoning_answer_reads_span_after_first_close_marker() -> None:
    # Spark-X2.5 shape: the generation prompt ends with <think>, so the
    # generation starts by closing it. The trailing </think> is the model's
    # end-of-turn marker, not part of the answer.
    assert _reasoning_answer("</think>zblorb-61</think>") == "zblorb-61"


def test_reasoning_answer_without_trailing_marker() -> None:
    assert _reasoning_answer("</think>zblorb-60") == "zblorb-60"


def test_reasoning_answer_unclosed_thinking_is_empty() -> None:
    assert _reasoning_answer("<think>still reasoning") == ""


def test_reasoning_answer_plain_prediction_unchanged() -> None:
    assert _reasoning_answer("zblorb-60") == "zblorb-60"


def test_reasoning_answer_match_scores_real_spark_generations() -> None:
    assert _score("</think>zblorb-61</think>", "zblorb-61", "reasoning_answer_match") == 1.0
    assert _score("</think>zblorb-62</think>", "zblorb-61", "reasoning_answer_match") == 0.0
    # The old rule reads the empty tail after the LAST marker and would score
    # this correct answer as a miss -- that behavior gap is why this mode exists.
    assert _score("</think>zblorb-61</think>", "zblorb-61", "normalized_exact_match") == 0.0


# ---- reasoning_final_number_match: reasoning span + final-number rule -------


def test_reasoning_final_number_reads_number_from_reasoning_span() -> None:
    # Spark-style trailing </think> marker: final_number_match reads the empty
    # tail after the LAST marker and would score a correct answer as a miss.
    assert _score("</think>$0.20 per sock... #### 3</think>", "3", "final_number_match") == 0.0
    assert (
        _score("</think>$0.20 per sock... #### 3</think>", "3", "reasoning_final_number_match")
        == 1.0
    )


def test_reasoning_final_number_matches_plain_output() -> None:
    # No think markers: the span is the whole prediction, number compared.
    assert _score("The answer is #### 72", "72", "reasoning_final_number_match") == 1.0


def test_reasoning_final_number_comma_and_decimal_normalization() -> None:
    assert _score("</think>70,000 total</think>", "70000", "reasoning_final_number_match") == 1.0
    assert _score("</think>72.0</think>", "72", "reasoning_final_number_match") == 1.0


def test_reasoning_final_number_rejects_unclosed_thinking() -> None:
    # Budget exhausted mid-reasoning: no answer span, no number, score 0.
    assert _score("<think>let me work this out... 42", "42", "reasoning_final_number_match") == 0.0


def test_reasoning_final_number_mismatch_scores_zero() -> None:
    assert _score("</think>#### 25</think>", "24", "reasoning_final_number_match") == 0.0


def test_reasoning_answer_match_rejects_unclosed_thinking() -> None:
    assert _score("<think>mid-reasoning", "zblorb-61", "reasoning_answer_match") == 0.0


def test_real_retry6_knowledge_item_regression() -> None:
    """The actual retry6 generation that wrongly scored 0.0 under v1."""
    raw = (
        'We need answer user: "What is the chemical symbol for the element '
        'tungsten? Respond with the symbol only." Need final only symbol. '
        "Tungsten symbol W. Ensure no extra.\n</think>\n\nW"
    )
    assert _score(raw, "w", "normalized_exact_match") == 1.0


# ---- protocol v2 markers --------------------------------------------------------


def _spec(protocol_version: str) -> pe.ParentEvalSpec:
    suites = tuple(
        pe.ParentSuiteSpec(
            name=f"suite-{dim}-v1",
            dimension=dim,
            dataset=f"datasets/{dim}.jsonl",
        )
        for dim in pe.PARENT_DIMENSIONS
    )
    return pe.ParentEvalSpec(
        suites=suites, quantization="4bit", protocol_version=protocol_version
    )


def test_protocol_version_is_in_the_digest() -> None:
    assert _spec("v1").digest() != _spec("v2").digest()
    assert _spec("v2").digest() == _spec("v2").digest()


def test_suite_budget_default_raised_for_thinking_models() -> None:
    suite = pe.ParentSuiteSpec(name="s", dimension="coding", dataset="d.jsonl")
    assert suite.max_new_tokens == 256
    assert pe.ParentSuiteSpec.from_dict({"name": "s", "dimension": "coding",
                                         "dataset": "d.jsonl"}).max_new_tokens == 256
