"""The one scoring rule, shared by every text evaluator worker.

It used to be two. `base_text_worker` and `transformers_text_worker` each carried
their own copy of `_normalize`, `_FINAL_NUMBER`, `_final_number` and `_score`, and
the copies had drifted on the case that matters most for reasoning models: the base
worker discarded an unclosed ``<think>`` block before extracting an answer, and the
transformers worker scored the raw generation.

That is not a cosmetic difference. Chowder runs the automatic baseline through one
worker and the candidate through the other, so the two sides of a comparison were
not scored by the same rule. In the 2026-09-11 GSM8K run **all 50** baseline
responses had an unclosed ``<think>``, which made the recorded baseline 0.00 *by
construction* and the arm uninterpretable
(`docs/PRUNED_9B_REAL_TRAINING_CORRECTION.md`).

The strict rule wins, and not because it is more conservative
---------------------------------------------------------------
An unclosed ``<think>`` means the generation budget ran out mid-reasoning: the model
never emitted an answer span at all. Reading a number out of that text scores the
*chain of thought*, which is the one region of the output explicitly not the answer.

The lenient rule does not merely flatter such a response, it can score it **correct**.
In the dense-vs-pruned control the pruned checkpoint degenerated on 8 of 8 prompts
and still posted 0.125, because `final_number_match` takes the last number and one
repetition loop happened to stop on the right one
(`docs/PRUNED_9B_REAL_TRAINING_RESULT.md`). A scorer that rewards that is measuring
luck. Scoring it as a miss is the honest reading: failing to finish thinking inside
the configured budget is a real capability limit of the protocol.

Consequence, stated plainly: thinking models under tight `max_new_tokens` score
*lower* here than under the old transformers-worker rule. That is the correction,
not a regression. Raise `max_new_tokens` if the budget is the binding constraint --
do not loosen the scorer to hide it.
"""
from __future__ import annotations

import re

__all__ = ["normalize", "final_answer", "final_number", "score"]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def final_answer(prediction: str) -> str:
    """Extract a thinking model's final answer from its raw generation.

    Qwen3-style reasoning models emit chain-of-thought, then a ``</think>`` close
    marker, then the answer. The answer is everything after the LAST close marker;
    without any marker the whole prediction is the answer (non-thinking models, or
    thinking disabled). An *unclosed* ``<think>`` means the budget was exhausted
    mid-reasoning, so there is no answer yet and the extraction is empty.
    """
    if "</think>" in prediction:
        return prediction.rsplit("</think>", 1)[1]
    if "<think>" in prediction:
        return ""
    return prediction


#: Final-number extraction for arithmetic word problems (GSM8K and similar).
#: The regex is comma-aware and the LAST number wins, both learned the hard way:
#: the frontier repo's FINDINGS-GSM8K-EVAL-BUG.md records an extractor using
#: r"[-]?\d+(?:\.\d+)?" that split "$70,000" into ["70", "000"] and scored a CORRECT
#: answer wrong, so a self-improvement loop then "trained on a failure" that was not
#: one. Commas are stripped and a trailing ".0" normalised, so
#: "70,000" == "70000" == "70000.0".
_FINAL_NUMBER = re.compile(r"[-]?\d[\d,]*(?:\.\d+)?")


def final_number(text: str) -> str | None:
    matches = _FINAL_NUMBER.findall(text or "")
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    if raw.endswith(".0"):
        raw = raw[:-2]
    if raw.endswith("."):
        raw = raw[:-1]
    return raw or None


def score(prediction: str, expected: str, scoring: str) -> float:
    """Score one prediction. Thinking-aware extraction applies to every mode."""
    answer = final_answer(prediction)
    if scoring == "exact_match":
        return float(answer.strip() == expected.strip())
    if scoring == "normalized_exact_match":
        return float(normalize(answer) == normalize(expected))
    if scoring == "final_number_match":
        # Compare the last number on each side, not the whole string: a model that
        # shows its work cannot exact-match a bare answer, and scoring it wrong
        # would manufacture failures rather than measure them.
        got = final_number(answer)
        want = final_number(expected)
        if got is None or want is None:
            return 0.0
        return float(got == want)
    raise ValueError(f"unsupported scoring: {scoring}")
