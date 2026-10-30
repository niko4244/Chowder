"""Standalone, fail-closed original GSM8K final-number scorer witness.

This is a research-only source-bound check. It neither approves any frozen
gate nor changes the contributor-owned frozen-gate evidence bridge. The
original ``reap_eval`` grader uses ``final_answer`` and ``final_number`` from
``chowder.evaluators.scoring``; their original GSM8K semantics are reproduced
below without importing an ambient editable Python package.

The production scorer source must match the independently reviewed worktree
bytes before this verifier returns a result. A drifted scorer is a blocker,
not an invitation to reinterpret historical labels.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re


ORIGINAL_SCORER_SHA256 = (
    "d2e59aca5c0a621bac8d4979d522c9b6107f811610567d21c92a2b53960860eb"
)
_WORKTREE_ROOT = Path(__file__).resolve().parent.parent
_ORIGINAL_SCORER_PATH = _WORKTREE_ROOT / "src/chowder/evaluators/scoring.py"
_FINAL_NUMBER = re.compile(r"[-]?\d[\d,]*(?:\.\d+)?")


class OriginalScorerSourceError(ValueError):
    """The original grader's exact source bytes are not authenticated."""


def require_original_scorer_source(*, source_path: Path | None = None) -> str:
    """Authenticate the original worktree scorer without importing it.

    ``source_path`` is intended for isolated adversarial tests only. A caller
    may not supply an alternate expected digest.
    """
    path = _ORIGINAL_SCORER_PATH if source_path is None else source_path
    if not path.is_file() or path.is_symlink():
        raise OriginalScorerSourceError("original GSM8K scorer source missing")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ORIGINAL_SCORER_SHA256:
        raise OriginalScorerSourceError("original GSM8K scorer source drift")
    return ORIGINAL_SCORER_SHA256


def _final_answer(prediction: str) -> str:
    if "</think>" in prediction:
        return prediction.rsplit("</think>", 1)[1]
    if "<think>" in prediction:
        return ""
    return prediction


def _final_number(text: str) -> str | None:
    matches = _FINAL_NUMBER.findall(text or "")
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    if raw.endswith(".0"):
        raw = raw[:-2]
    if raw.endswith("."):
        raw = raw[:-1]
    return raw or None


def score_original_gsm8k(
    prediction: str, expected: str, *, terminated: bool,
) -> bool:
    """Reproduce original ``reap_eval._score`` GSM8K correctness semantics.

    Unlike a dynamic import, a monkeypatched ambient scoring package cannot
    change the result. This function still requires source authentication;
    it does not certify prompt provenance or native request/response linkage.
    """
    require_original_scorer_source()
    if type(prediction) is not str or type(expected) is not str:
        raise TypeError("original GSM8K prediction/expected must be strings")
    if type(terminated) is not bool:
        raise TypeError("original GSM8K terminated must be a boolean")
    if not terminated:
        return False
    got = _final_number(_final_answer(prediction))
    want = _final_number(expected)
    return got is not None and want is not None and got == want
