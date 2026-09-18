"""Ledger correction semantics: supersede without mutating.

Original record bytes must remain unchanged; revisions append; the
effective verdict resolves deterministically; the audit trail stays
inspectable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chowder.growth.lineage import GenerationLedger
from chowder.growth.promotion import PromotionDecision


def _decision(verdict: str) -> PromotionDecision:
    return PromotionDecision(
        verdict=verdict,
        reasons=("test",),
        checks={"target_improvement": "met"},
    )


def _record(ledger: GenerationLedger, verdict: str = "PROMOTED") -> None:
    ledger.record(
        version="gen1",
        parent_version="gen0",
        cycle_id="c1",
        base_model={"path": "F:/llm-models/tiny"},
        dataset_manifest_ref="d.json",
        curriculum_manifest_ref="c.json",
        recipe={"lr": 1e-4},
        training_evidence_ref="t.json",
        evaluation_report_ref="e.json",
        promotion=_decision(verdict),
    )


def test_original_record_untouched_by_revision(tmp_path: Path) -> None:
    _record(GenerationLedger(tmp_path))
    before = (tmp_path / "generations.json").read_bytes()
    ledger = GenerationLedger(tmp_path)
    ledger.append_adjudication_revision(
        generation_version="gen1",
        reason_codes=["CARRIED_EVIDENCE_TREATED_AS_CANDIDATE", "BUDGET_SETTLEMENT_INCOMPLETE"],
        policy_version="promotion-policy-v2",
        new_verdict="INCONCLUSIVE",
        evidence_refs=["docs/gen1/re_adjudication.json"],
    )
    after = (tmp_path / "generations.json").read_bytes()
    assert before == after  # original bytes unchanged


def test_revision_appends_and_chains(tmp_path: Path) -> Path:
    path = tmp_path
    _record(GenerationLedger(path))
    ledger = GenerationLedger(path)
    first = ledger.append_adjudication_revision(
        generation_version="gen1",
        reason_codes=["INTEGRITY_AUDIT"],
        policy_version="v2",
        new_verdict="INCONCLUSIVE",
    )
    assert first.supersedes_adjudication_id == "original"
    second = ledger.append_adjudication_revision(
        generation_version="gen1",
        reason_codes=["FRESH_CANDIDATE_MEASUREMENT"],
        policy_version="v2",
        new_verdict="PROMOTED",
    )
    assert second.supersedes_adjudication_id == first.revision_id
    revisions = GenerationLedger(path).revisions_for("gen1")
    assert [r.revision_id for r in revisions] == [
        "gen1-adjudication-001",
        "gen1-adjudication-002",
    ]
    return path


def test_effective_verdict_is_deterministic(tmp_path: Path) -> None:
    path = test_revision_appends_and_chains(tmp_path)
    # Reload from disk each time: resolution is a function of the file, not
    # of any in-memory state.
    for _ in range(3):
        assert GenerationLedger(path).effective_verdict("gen1") == "PROMOTED"


def test_effective_verdict_falls_back_to_original(tmp_path: Path) -> None:
    _record(GenerationLedger(tmp_path), verdict="REJECTED")
    assert GenerationLedger(tmp_path).effective_verdict("gen1") == "REJECTED"


def test_revision_of_unknown_generation_refuses(tmp_path: Path) -> None:
    ledger = GenerationLedger(tmp_path)
    with pytest.raises(KeyError):
        ledger.append_adjudication_revision(
            generation_version="genX",
            reason_codes=["X"],
            policy_version="v2",
            new_verdict="INCONCLUSIVE",
        )


def test_original_digest_binds_revision_to_on_disk_record(tmp_path: Path) -> None:
    _record(GenerationLedger(tmp_path))
    ledger = GenerationLedger(tmp_path)
    revision = ledger.append_adjudication_revision(
        generation_version="gen1",
        reason_codes=["AUDIT"],
        policy_version="v2",
        new_verdict="INCONCLUSIVE",
    )
    on_disk = json.loads((tmp_path / "generations.json").read_text(encoding="utf-8"))
    import hashlib

    expected = hashlib.sha256(
        json.dumps(on_disk[0]["promotion"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert revision.original_decision_digest == expected
