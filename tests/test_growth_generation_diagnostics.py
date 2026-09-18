"""The generation-diagnostics instrument: the ported rules and their refusals.

The frozen judge's T1-T10 read one instrument row. Before this module the only
thing that could produce that row was ``docs/gen1/run_gen1_cycle.py``, so T1-T10
were ``UNKNOWN`` for every run a production path could make. These tests pin the
two properties that make the port real rather than nominal:

* the aggregates are the frozen instrument's rules, computed from the worker's
  own per-item generations -- hand-computed cases, not round-trips;
* the facts the aggregates are defined over are read from what the worker
  observed and validated, so a generation nobody recorded termination for
  refuses instead of being counted as a termination failure.
"""

from __future__ import annotations

import json

import pytest

from chowder.growth.candidate_evaluation import CandidateEvaluationRefusal
from chowder.growth.generation_diagnostics import (
    GENERATION_DIAGNOSTICS_UNMEASURED,
    GenerationDiagnostics,
)


def _item(
    *,
    prompt: str,
    expected: str = "answer",
    prediction: str = "answer",
    generated_tokens: int = 4,
    eos_terminated: bool = True,
) -> dict:
    return {
        "prompt": prompt,
        "expected": expected,
        "prediction": prediction,
        "score": 1.0,
        "generated_tokens": generated_tokens,
        "eos_terminated": eos_terminated,
    }


def test_the_frozen_rules_are_computed_from_the_generations() -> None:
    """EOS, cap, unclosed think, loops and trigrams, by hand.

    Four items: one clean termination, one that ran to the cap, one that never
    closed its ``<think>``, and one that repeated a line three times. The
    expected numbers are the arithmetic of the frozen definitions, not whatever
    the implementation happened to produce.
    """
    items = [
        _item(prompt="one", prediction="answer", generated_tokens=4),
        _item(prompt="two", prediction="rambling on", generated_tokens=8, eos_terminated=False),
        _item(prompt="three", prediction="<think>thinking forever", generated_tokens=4),
        _item(prompt="four", prediction="loop\nloop\nloop", generated_tokens=4),
    ]

    diagnostics = GenerationDiagnostics.from_items(
        items, max_new_tokens=8, seed=1234, source="predictions-target.jsonl"
    )

    assert diagnostics.n_prompts == 4
    # Three of four stopped on EOS; the cap-hitter is the one that did not.
    assert diagnostics.eos_termination_rate == pytest.approx(0.75)
    assert diagnostics.max_token_cap_rate == pytest.approx(0.25)
    assert diagnostics.unclosed_think_rate == pytest.approx(0.25)
    assert diagnostics.obvious_loop_count == 1
    assert diagnostics.max_new_tokens == 8
    assert diagnostics.seed == 1234


def test_short_completions_are_trivially_distinct() -> None:
    """Fewer than three words cannot repeat a trigram, so the ratio is 1.0."""
    diagnostics = GenerationDiagnostics.from_items(
        [_item(prompt="one", prediction="answer")],
        max_new_tokens=8,
        seed=1234,
        source="predictions-target.jsonl",
    )

    assert diagnostics.distinct_trigram_ratio_mean == pytest.approx(1.0)
    assert diagnostics.distinct_trigram_ratio_min == pytest.approx(1.0)


def test_a_repeated_phrase_lowers_the_trigram_ratio() -> None:
    """A loop is visible in the ratio as well as in the loop counter."""
    looping = "the cat sat the cat sat the cat sat the cat sat"
    diagnostics = GenerationDiagnostics.from_items(
        [_item(prompt="one", prediction=looping)],
        max_new_tokens=64,
        seed=1,
        source="predictions.jsonl",
    )

    assert diagnostics.obvious_loop_count == 0  # its lines are not three in a row
    assert 0.0 < diagnostics.distinct_trigram_ratio_mean < 1.0


def test_the_metadata_carries_the_keys_the_frozen_judge_reads() -> None:
    """Flat keys, exactly where T1-T10 look for them."""
    diagnostics = GenerationDiagnostics.from_items(
        [_item(prompt="Reply with exactly: ping", expected="ping", prediction="ping")],
        max_new_tokens=8,
        seed=1234,
        source="predictions.jsonl",
    )

    metadata = diagnostics.to_metadata()

    for key in (
        "per_prompt",
        "eos_termination_rate",
        "max_token_cap_rate",
        "unclosed_think_rate",
        "obvious_loop_count",
        "distinct_trigram_ratio_mean",
    ):
        assert key in metadata, f"the judge reads metadata[{key!r}]"
    entry = metadata["per_prompt"][0]
    assert entry["prompt"] == "Reply with exactly: ping"
    assert entry["expected"] == "ping"
    assert entry["completion"] == "ping"
    assert entry["eos_terminated"] is True
    assert entry["cap_hit"] is False
    # JSON-serialisable: this metadata is written into the arm the judge loads.
    assert json.loads(json.dumps(metadata)) == metadata


@pytest.mark.parametrize(
    "item, expected_reason",
    [
        ({"prompt": "p", "expected": "e", "prediction": "e", "eos_terminated": True}, "generated_tokens"),
        ({"prompt": "p", "expected": "e", "prediction": "e", "generated_tokens": 3}, "eos_terminated"),
        (
            {"prompt": "p", "expected": "e", "prediction": "e", "generated_tokens": 3, "eos_terminated": "yes"},
            "eos_terminated",
        ),
        (
            {"prompt": "p", "expected": "e", "prediction": "e", "generated_tokens": -1, "eos_terminated": True},
            "generated_tokens",
        ),
    ],
)
def test_a_generation_that_was_not_observed_refuses(item: dict, expected_reason: str) -> None:
    """No observation is not a zero: an unrecorded fact cannot become a rate."""
    with pytest.raises(CandidateEvaluationRefusal) as error:
        GenerationDiagnostics.from_items(
            [item], max_new_tokens=8, seed=1, source="predictions.jsonl"
        )

    assert GENERATION_DIAGNOSTICS_UNMEASURED in str(error.value)
    assert expected_reason in str(error.value)


def test_termination_and_the_cap_cannot_both_be_true() -> None:
    """A worker whose two facts disagree is refused, not reconciled.

    ``eos_terminated`` means the generation stopped *short* of the cap. Both
    being true would mean one of the facts is wrong, and this module has no way
    to know which -- so it refuses rather than pick.
    """
    with pytest.raises(CandidateEvaluationRefusal) as error:
        GenerationDiagnostics.from_items(
            [_item(prompt="p", generated_tokens=8, eos_terminated=True)],
            max_new_tokens=8,
            seed=1,
            source="predictions.jsonl",
        )

    assert GENERATION_DIAGNOSTICS_UNMEASURED in str(error.value)
    assert "termination" in str(error.value)


def test_no_generations_is_not_a_diagnostic() -> None:
    with pytest.raises(CandidateEvaluationRefusal) as error:
        GenerationDiagnostics.from_items(
            [], max_new_tokens=8, seed=1, source="predictions.jsonl"
        )

    assert GENERATION_DIAGNOSTICS_UNMEASURED in str(error.value)


def test_the_same_items_always_produce_the_same_metadata() -> None:
    """Determinism: the aggregates are a function of the bytes, nothing else."""
    items = [
        _item(prompt=f"prompt {index}", prediction=f"distinct words number {index}")
        for index in range(5)
    ]

    first = GenerationDiagnostics.from_items(
        items, max_new_tokens=16, seed=7, source="predictions.jsonl"
    ).to_metadata()
    second = GenerationDiagnostics.from_items(
        items, max_new_tokens=16, seed=7, source="predictions.jsonl"
    ).to_metadata()

    assert first == second
