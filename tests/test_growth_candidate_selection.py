"""Candidate-selection integrity.

Selection must be deterministic and decided from training-side evidence
only. Protected/broad final scores are promotion gates, not optimization
targets -- the selection function structurally cannot read them.
"""

from __future__ import annotations

import pytest

from chowder.growth.cycle import select_candidate


def _attempt(recipe_id: str, *, succeeded: bool = True, loss: float | None = None, idx: int = 0) -> dict:
    evidence = {
        "recipe_id": recipe_id,
        "status": "SUCCEEDED" if succeeded else "FAILED",
        "candidate_succeeded": succeeded,
        "artifact_ref": f"artifacts/{recipe_id}/adapter" if succeeded else None,
        "artifact_sha256": "a" * 64 if succeeded else None,
        "candidate_metrics": {"train_loss": loss} if loss is not None else {},
        "attempt_index": idx,
    }
    # The tempting-but-forbidden fields: selection must ignore these even
    # when present in the evidence mapping.
    evidence["protected_scores"] = {"math500@2024-04": 1.0}
    evidence["broad_scores"] = {"gpqa_diamond@2025-05-30": 1.0}
    return evidence


def test_first_successful_is_deterministic_in_preregistered_order() -> None:
    results = (
        _attempt("recipe-00", succeeded=False),
        _attempt("recipe-01", succeeded=True, idx=1),
        _attempt("recipe-02", succeeded=True, idx=2),
    )
    chosen = select_candidate(results)
    assert chosen["recipe_id"] == "recipe-01"


def test_first_by_loss_uses_only_recorded_train_loss() -> None:
    results = (
        _attempt("recipe-00", loss=1.9),
        _attempt("recipe-01", loss=1.2),
        _attempt("recipe-02", loss=1.5),
    )
    chosen = select_candidate(results, order="first_by_loss")
    assert chosen["recipe_id"] == "recipe-01"


def test_protected_scores_in_evidence_cannot_influence_selection() -> None:
    """A recipe whose evidence block claims a perfect protected score must
    not win selection through it."""
    loser = _attempt("recipe-00", loss=1.9)
    winner_by_loss = _attempt("recipe-01", loss=1.2)
    results = (loser, winner_by_loss)
    chosen = select_candidate(results, order="first_by_loss")
    assert chosen["recipe_id"] == "recipe-01"


def test_selection_never_returns_failed_or_artifactless_attempts() -> None:
    results = (
        _attempt("recipe-00", succeeded=False),
        _attempt("recipe-01", succeeded=True),
    )
    results[1]["artifact_ref"] = None
    assert select_candidate(results) is None
    assert select_candidate(()) is None


def test_unknown_policy_refuses() -> None:
    with pytest.raises(ValueError, match="unknown candidate-selection policy"):
        select_candidate((_attempt("r0"),), order="best_protected_score")
