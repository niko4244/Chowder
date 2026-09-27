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
from chowder.evaluators.base_text_worker import _final_answer, _score


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


def test_reopened_thinking_after_empty_block_means_no_answer() -> None:
    # Real Condition A shape: empty block, reopened, truncated mid-sentence.
    raw = "<think>\n\n</think>\n\n<think> Okay, the degree drops by 1 (since the"
    assert _final_answer(raw) == ""
    assert _score(raw, "1", "final_number_match") == 0.0


def test_unfinished_generation_scores_zero() -> None:
    assert _score("The answer is 64", "64", "final_number_match", finished=False) == 0.0
    assert _score("The answer is 64", "64", "final_number_match", finished=True) == 1.0


def test_normalized_match_ignores_thinking_preamble() -> None:
    raw = 'We need answer user: "tungsten?" Need final only.\n</think>\n\nW'
    assert _score(raw, "w", "normalized_exact_match") == 1.0


def test_unclosed_thinking_scores_zero() -> None:
    assert _score("<think>still reasoning about it", "w", "normalized_exact_match") == 0.0


def test_plain_prediction_unchanged() -> None:
    assert _score("Mercury", "mercury", "normalized_exact_match") == 1.0
    assert _score("Venus", "mercury", "normalized_exact_match") == 0.0


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


def test_boxed_answer_wins_over_trailing_restatement() -> None:
    # Real A3 GSM8K rows: the committed answer is boxed, then context is restated.
    assert _score(r"**Answer:** Claire eats \boxed{7} dozens of eggs in 4 weeks.", "7", "final_number_match") == 1.0
    assert _score(r"It is \boxed{8} cm shorter from 1:00 PM to 5:00 PM.", "8", "final_number_match") == 1.0
    assert _score(r"so \boxed{\frac{3}{4}} then", "4", "final_number_match") == 1.0  # last number inside the box
    assert _score(r"\boxed{12} was wrong, the answer is \boxed{15}.", "15", "final_number_match") == 1.0


def test_decimal_money_answers_normalise() -> None:
    assert _score(r"\boxed{18.00}", "18", "final_number_match") == 1.0
    assert _score(r"\boxed{\$18.50}", "18.5", "final_number_match") == 1.0
    assert _score(r"\boxed{100}", "10", "final_number_match") == 0.0  # integers keep their zeros
    assert _score(r"\boxed{0.0}", "0", "final_number_match") == 1.0


def test_unbalanced_last_box_falls_back_to_the_previous_box() -> None:
    assert _score(r"so \boxed{42} and 7 more ... \boxed{42", "42", "final_number_match") == 1.0


def test_no_box_keeps_the_last_number_rule() -> None:
    assert _score("The total is 64 dollars", "64", "final_number_match") == 1.0
    assert _score(r"\boxed{7 unbalanced, 9", "9", "final_number_match") == 1.0
