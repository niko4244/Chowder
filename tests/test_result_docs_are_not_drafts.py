"""A result document must not ship with unfilled numbers.

`PRUNED_9B_RERUN_RESULT.md` was written as a skeleton *before* the run finished, so
its verdict rules were fixed before the numbers existed and filling them in produces
the verdict mechanically. That is the point of pre-writing it.

The risk it creates is a draft being read as a result: a `«TBD»` left next to a
verdict looks like a claim. So the contract is self-enforcing -- while the document
carries its SKELETON banner, placeholders are expected; the moment the banner is
removed to declare it final, every placeholder must be gone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PLACEHOLDER = "«TBD»"
BANNER = "**SKELETON"

DOCS = Path(__file__).resolve().parent.parent / "docs"


def _result_docs() -> list[Path]:
    return sorted(DOCS.glob("*RESULT*.md"))


def test_there_are_result_docs_to_check() -> None:
    """Guard against the glob silently matching nothing and passing vacuously."""
    assert _result_docs(), f"no *RESULT*.md under {DOCS}"


@pytest.mark.parametrize("doc", _result_docs(), ids=lambda p: p.name)
def test_a_finalised_result_doc_has_no_placeholders(doc: Path) -> None:
    text = doc.read_text(encoding="utf-8")
    if BANNER in text:
        pytest.skip(f"{doc.name} is still marked as a skeleton")
    assert PLACEHOLDER not in text, (
        f"{doc.name} has no SKELETON banner but still contains {PLACEHOLDER} -- "
        "either finish the numbers or put the banner back; a placeholder next to a "
        "verdict reads as a claim"
    )


def test_a_skeleton_must_actually_be_marked_as_one() -> None:
    """The inverse: a doc full of placeholders must carry the banner, so it cannot be
    mistaken for a finished result just because this test skips it."""
    for doc in _result_docs():
        text = doc.read_text(encoding="utf-8")
        if PLACEHOLDER in text:
            assert BANNER in text, (
                f"{doc.name} contains {PLACEHOLDER} but is not marked SKELETON"
            )
