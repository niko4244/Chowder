"""The io-ceiling benchmark: two forms of the same expert work, and one arithmetic.

The benchmark exists to answer two questions the campaign budget rests on --
how fast this link copies, and what the expert GEMMs cost in each form. Both
answers are worthless if the two forms are not computing the same thing, so the
shapes are checked before either is timed. These tests run the shape check on a
CPU: they need no card, and the defect they guard against (an index built as a
slice of the token rows, when there are more routed rows than tokens) raises
before a single timer starts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chowder_batch"))
sys.path.insert(0, str(ROOT / "src"))

import moe_int8_io_ceiling as ceiling  # noqa: E402

torch = pytest.importorskip("torch")


def test_both_expert_forms_produce_the_same_row_count():
    """The 2026-10-21 defect: `x[rows % tokens]` with `rows` a slice raised
    `TypeError: unsupported operand type(s) for %: 'slice' and 'int'`, so the
    benchmark could not be reproduced from the committed script at all."""
    out = ceiling.measure_experts(
        tokens=16, topk=4, experts=8, hidden=8, moe_inter=4, reps=1, device="cpu"
    )
    assert out["rows_per_expert"] == 8  # (16 * 4) // 8
    assert out["speedup"] > 0
    assert out["padded_rows_waste"] == 0.0


def test_more_routed_rows_than_tokens_is_the_case_the_artifact_has():
    """At the shipped shapes there are 16,320 routed rows over 2,048 tokens:
    the assignment has to wrap, and a slice cannot express that."""
    out = ceiling.measure_experts(
        tokens=8, topk=8, experts=4, hidden=8, moe_inter=4, reps=1, device="cpu"
    )
    assert out["rows_per_expert"] == 16
    assert out["speedup"] > 0


def test_the_copy_arithmetic_still_says_the_target_is_unreachable():
    out = ceiling.step_ceiling(18.82, 3, 5.52, 9.0)
    assert out["target_reachable"] is False
    assert out["h2d_seconds"] > 9.0
