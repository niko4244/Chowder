"""The answer-extraction scorer, and the documented bug it must not reproduce.

The frontier repo's FINDINGS-GSM8K-EVAL-BUG.md records two harness defects that
manufactured failures rather than measuring them:

  1. an extractor using r"[-]?\d+(?:\.\d+)?" split "$70,000" into ["70", "000"] and
     took the last, scoring a CORRECT answer wrong -- a self-improvement loop then
     "trained on a failure" that was not one;
  2. a 256-token generation budget cut verbose reasoning off before the final
     number, so an intermediate value was scored instead.

(2) is a suite setting (max_new_tokens); (1) is this code's responsibility, so the
real case is pinned here.
"""

import pytest

from chowder.evaluators.transformers_text_worker import _final_number, _score
from chowder.evaluators.base_text_worker import _score as _base_score


def test_the_comma_bug_case_from_the_findings_scores_correct():
    """"profit of $70,000" vs "70000" -- the exact case a naive regex got wrong."""
    assert _score("He made a profit of $70,000", "70000", "final_number_match") == 1.0


@pytest.mark.parametrize("text,want", [
    ("the answer is 18", "18"),
    ("profit of $70,000", "70000"),
    ("so 1,234.50 dollars", "1234.50"),
    ("result: -42", "-42"),
    ("it costs 8.0", "8"),            # trailing .0 normalised
    ("ends with 12.", "12"),          # trailing dot dropped
    ("no digits here", None),
    ("", None),
])
def test_final_number_extraction(text, want):
    assert _final_number(text) == want


def test_the_last_number_wins_so_shown_work_does_not_break_it():
    shown = "First 3 * 4 = 12, then 12 + 6 = 18. Final Answer: 18"
    assert _score(shown, "18", "final_number_match") == 1.0
    # an intermediate value must NOT pass
    assert _score(shown, "12", "final_number_match") == 0.0


def test_a_wrong_number_still_fails():
    assert _score("the answer is 19", "18", "final_number_match") == 0.0


def test_missing_numbers_score_zero_rather_than_crashing():
    assert _score("I cannot answer", "18", "final_number_match") == 0.0
    assert _score("18", "no expected number", "final_number_match") == 0.0


def test_exact_match_modes_are_unchanged():
    assert _score("18", "18", "exact_match") == 1.0
    assert _score(" 18 ", "18", "normalized_exact_match") == 1.0
    assert _score("the answer is 18", "18", "exact_match") == 0.0


def test_unsupported_scoring_still_refuses():
    with pytest.raises(ValueError, match="unsupported scoring"):
        _score("18", "18", "vibes")


def test_the_base_model_worker_strips_thinking_then_extracts():
    """base_text_worker drops a <think> block before scoring; the number must come
    from the answer, not from the discarded reasoning."""
    text = "<think>maybe 99</think> Final Answer: 18"
    assert _base_score(text, "18", "final_number_match") == 1.0
    assert _base_score(text, "99", "final_number_match") == 0.0


def test_both_workers_accept_the_mode():
    from chowder.evaluators.transformers_text import _ALLOWED_SCORING

    assert "final_number_match" in _ALLOWED_SCORING
