"""Every text evaluator worker must score by exactly one rule.

They did not. `base_text_worker` discarded an unclosed ``<think>`` block before
extracting an answer; `transformers_text_worker` scored the raw generation. Chowder
runs the automatic baseline through the first and the candidate through the second,
so the two sides of a comparison were scored by *different rules* -- which voided
the capability arm of the 2026-09-11 GSM8K run, where all 50 baseline responses had
an unclosed ``<think>`` and the recorded 0.00 was an artifact of the asymmetry rather
than a measurement (`docs/PRUNED_9B_REAL_TRAINING_CORRECTION.md`).

Three layers here, because an agreement table alone can be satisfied by a second
implementation that happens to pass the table and then drifts:

  1. identity   -- both workers expose the *same function object*
  2. behaviour  -- a shared case table, asserted through both workers
  3. source     -- neither worker file re-declares its own scoring helpers
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chowder.evaluators import scoring
from chowder.evaluators.base_text_worker import _score as base_score
from chowder.evaluators.transformers_text_worker import _score as candidate_score


# ---- 1. identity ---------------------------------------------------------------


def test_both_workers_expose_the_same_scoring_function() -> None:
    assert base_score is scoring.score
    assert candidate_score is scoring.score


def test_both_workers_expose_the_same_helpers() -> None:
    from chowder.evaluators import base_text_worker as base
    from chowder.evaluators import transformers_text_worker as candidate

    for name, shared in (
        ("_normalize", scoring.normalize),
        ("_final_answer", scoring.final_answer),
        ("_final_number", scoring.final_number),
    ):
        assert getattr(base, name) is shared, f"base_text_worker.{name} drifted"
        assert getattr(candidate, name) is shared, f"transformers_text_worker.{name} drifted"


# ---- 2. behaviour ---------------------------------------------------------------

#: The real generation shape that made this matter. The pruned 9B degenerated on
#: every prompt, never closed its <think>, and this loop happens to stop on the
#: CORRECT number -- so the old raw-text rule scored it 1.0. That is how a checkpoint
#: degenerate on 8/8 prompts still posted 0.125 in the dense-vs-pruned control.
DEGENERATE_LOOP_ENDING_CORRECT = (
    "<think>So 20*20=400. 400/20=20. So 20*20=400. 400/20=20. "
    "So 20*20=400. 400/20=20. So the answer is 18"
)

CASES: list[tuple[str, str, str, float]] = [
    # (scoring mode, prediction, expected, want)
    # -- the divergence that voided an arm: no closing </think> means no answer
    ("final_number_match", DEGENERATE_LOOP_ENDING_CORRECT, "18", 0.0),
    ("final_number_match", "<think>it must be 18", "18", 0.0),
    ("normalized_exact_match", "<think>still reasoning about it", "w", 0.0),
    ("exact_match", "<think>nearly there 18", "18", 0.0),
    # -- closed thinking: the answer is what follows the LAST close marker
    ("final_number_match", "<think>maybe 99</think> Final Answer: 18", "18", 1.0),
    ("final_number_match", "<think>maybe 99</think> Final Answer: 18", "99", 0.0),
    ("final_number_match", "<think>a</think>12 <think>b</think>18", "18", 1.0),
    ("normalized_exact_match", "reasoning here\n</think>\n\nW", "w", 1.0),
    # -- no markers at all: the whole prediction is the answer
    ("final_number_match", "First 3 * 4 = 12, then 12 + 6 = 18. Answer: 18", "18", 1.0),
    ("final_number_match", "First 3 * 4 = 12, then 12 + 6 = 18. Answer: 18", "12", 0.0),
    ("final_number_match", "He made a profit of $70,000", "70000", 1.0),
    ("final_number_match", "the answer is 19", "18", 0.0),
    ("final_number_match", "I cannot answer", "18", 0.0),
    ("final_number_match", "18", "no expected number", 0.0),
    ("exact_match", "18", "18", 1.0),
    ("exact_match", "the answer is 18", "18", 0.0),
    ("normalized_exact_match", " 18 ", "18", 1.0),
    ("normalized_exact_match", "Mercury", "mercury", 1.0),
]


@pytest.mark.parametrize("mode,prediction,expected,want", CASES)
def test_both_workers_agree_on_the_case_table(
    mode: str, prediction: str, expected: str, want: float
) -> None:
    assert base_score(prediction, expected, mode) == want
    assert candidate_score(prediction, expected, mode) == want


def test_the_lenient_rule_would_have_scored_the_degenerate_loop_correct() -> None:
    """Pins WHY the strict rule is the right one, not merely the stricter one.

    Reading a number out of an unclosed <think> scores the chain of thought -- the
    one region of the output explicitly not the answer -- and here it would reward a
    pure repetition loop with a 1.0.
    """
    assert scoring.final_number(DEGENERATE_LOOP_ENDING_CORRECT) == "18"  # raw text
    assert scoring.final_answer(DEGENERATE_LOOP_ENDING_CORRECT) == ""  # no answer span
    assert scoring.score(DEGENERATE_LOOP_ENDING_CORRECT, "18", "final_number_match") == 0.0


def test_unsupported_scoring_refuses_in_both_workers() -> None:
    for score in (base_score, candidate_score):
        with pytest.raises(ValueError, match="unsupported scoring"):
            score("18", "18", "vibes")


# ---- 3. source ------------------------------------------------------------------


def test_no_worker_redeclares_its_own_scorer() -> None:
    """A future edit must not paste a local copy back in. Identity and the case
    table would both still pass if a worker shadowed only part of the rule."""
    import chowder

    evaluators = Path(chowder.__file__).resolve().parent / "evaluators"
    for name in ("base_text_worker.py", "transformers_text_worker.py"):
        source = (evaluators / name).read_text(encoding="utf-8")
        assert "from .scoring import" in source, f"{name} no longer imports the shared rule"
        for declaration in (
            "def _score(",
            "def _final_number(",
            "def _final_answer(",
            "def _normalize(",
            "_FINAL_NUMBER = re.compile",
        ):
            assert declaration not in source, f"{name} re-declares {declaration!r}"
