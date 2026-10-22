"""N5 bypass reproductions: upward revisions and unapproved direct ledger writes.

All artifacts are CPU-only synthetic fixtures; no model or GPU is accessed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chowder.growth.lineage import GenerationLedger
from chowder.growth.owner_approval import _grant
from chowder.growth.promotion import PromotionDecision
from chowder.growth.promotion_transaction import prepare_transaction


SHA = "a" * 64


def _decision(verdict: str) -> PromotionDecision:
    return PromotionDecision(verdict=verdict, reasons=("synthetic fixture",), checks={})


def _record(ledger: GenerationLedger, *, model: dict, verdict: str,
            owner_approval=None, candidate_artifact_sha256: str = "") -> None:
    ledger.record(
        version="gen2", parent_version="gen1", cycle_id="N5-test",
        base_model=model, dataset_manifest_ref="data", curriculum_manifest_ref="curriculum",
        recipe={}, training_evidence_ref="train", evaluation_report_ref="eval",
        promotion=_decision(verdict), owner_approval=owner_approval,
        candidate_artifact_sha256=candidate_artifact_sha256,
    )


def _reap(tmp_path: Path) -> Path:
    model = tmp_path / "reap-model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_moe", "text_config": {"num_experts": 160}
    }), encoding="utf-8")
    return model


def test_rejected_reap_cannot_be_revised_to_promoted_without_n5(tmp_path):
    model = _reap(tmp_path)
    ledger = GenerationLedger(tmp_path / "ledger")
    _record(ledger, model={"base_model_path": str(model), "lineage": "reap-moe"}, verdict="REJECTED")
    with pytest.raises(ValueError, match="ADJUDICATION_PROMOTION_UNAUTHORIZED"):
        ledger.append_adjudication_revision(
            generation_version="gen2", reason_codes=("CLAIMED_CORRECTION",),
            policy_version="synthetic", new_verdict="PROMOTED",
        )
    assert ledger.effective_verdict("gen2") == "REJECTED"
    assert ledger.revisions_for("gen2") == ()


def test_downgraded_generation_cannot_repromote_via_revision(tmp_path):
    root = tmp_path / "ledger"
    root.mkdir()
    primary = tmp_path / "primary"
    primary.mkdir()
    (primary / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    # Explicit legacy non-REAP record, predating transactional promotion.
    # The new record() API must not mint such a record without PREPARED.
    (root / "generations.json").write_text(json.dumps([{
        "version": "gen2", "parent_version": "gen1", "cycle_id": "N5-test",
        "base_model": {"path": str(primary), "revision": "legacy-primary"},
        "promotion": _decision("PROMOTED").to_dict(),
    }]), encoding="utf-8")
    ledger = GenerationLedger(root)
    ledger.append_adjudication_revision(
        generation_version="gen2", reason_codes=("AUDIT",),
        policy_version="synthetic", new_verdict="INCONCLUSIVE",
    )
    with pytest.raises(ValueError, match="ADJUDICATION_PROMOTION_UNAUTHORIZED"):
        ledger.append_adjudication_revision(
            generation_version="gen2", reason_codes=("UNVERIFIED_RECOVERY",),
            policy_version="synthetic", new_verdict="PROMOTED",
        )
    assert ledger.effective_verdict("gen2") == "INCONCLUSIVE"


def test_promoted_direct_record_without_model_identity_refuses(tmp_path):
    ledger = GenerationLedger(tmp_path / "ledger")
    with pytest.raises(ValueError, match="PROMOTION_BASE_UNDECLARED"):
        _record(ledger, model={}, verdict="PROMOTED")
    assert not (tmp_path / "ledger" / "generations.json").exists()


def test_promoted_direct_record_with_only_revision_refuses(tmp_path):
    ledger = GenerationLedger(tmp_path / "ledger")
    with pytest.raises(ValueError, match="PROMOTION_BASE_UNDECLARED"):
        _record(ledger, model={"revision": "synthetic-primary@sha"}, verdict="PROMOTED")
    assert not (tmp_path / "ledger" / "generations.json").exists()


def test_forged_in_process_grant_not_sufficient_without_frozen_judge(tmp_path):
    model = _reap(tmp_path)
    ledger = GenerationLedger(tmp_path / "ledger")
    prepare_transaction(tmp_path, cycle_id="N5-test", candidate_version="gen2",
                        run_document={"cycle_id": "N5-test"},
                        promotion=_decision("PROMOTED").to_dict(),
                        manifest_digest="a" * 64)
    # The old in-process _grant helper can mint an apparently valid MAC without
    # showing anyone a TTY. N5 release must independently refuse this token.
    forged = _grant("N5-test", "gen2", SHA, judge_certified=True)
    with pytest.raises(ValueError, match="FROZEN_REAP_JUDGE_REQUIRED"):
        _record(
            ledger, model={"base_model_path": str(model), "lineage": "reap-moe"},
            verdict="PROMOTED", owner_approval=forged, candidate_artifact_sha256=SHA,
        )
    assert not (tmp_path / "ledger" / "generations.json").exists()


def test_legacy_reap_promoted_without_transaction_is_not_readable(tmp_path):
    model = _reap(tmp_path)
    root = tmp_path / "ledger"
    root.mkdir()
    # Simulate a pre-N5 direct mutation of the JSON ledger. No transaction
    # marker, frozen judge receipt, or owner evidence exists.
    payload = [{
        "version": "gen2", "parent_version": "gen1", "cycle_id": "N5-test",
        "base_model": {"base_model_path": str(model), "lineage": "reap-moe"},
        "promotion": _decision("PROMOTED").to_dict(),
    }]
    (root / "generations.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="PROMOTION_UNCOMMITTED"):
        GenerationLedger(root).effective_verdict("gen2")


@pytest.mark.parametrize("variant", ["nonexistent", "configless", "unknown_config", "disguised_reap", "relative"])
def test_markerless_promoted_history_cannot_fake_nonreap_model_path(tmp_path, variant):
    base = tmp_path / "claimed-primary"
    if variant not in ("nonexistent", "relative"):
        base.mkdir()
    if variant == "unknown_config":
        (base / "config.json").write_text("{}", encoding="utf-8")
    if variant == "disguised_reap":
        (base / "config.json").write_text(json.dumps({
            "model_type": "qwen3_5_moe", "text_config": {"num_experts": 128}
        }), encoding="utf-8")
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    (ledger_dir / "generations.json").write_text(json.dumps([{
        "version": "gen2", "parent_version": "gen1", "cycle_id": "N5-test",
        "base_model": {"path": "claimed-primary" if variant == "relative" else str(base),
                       "lineage": "unknown"},
        "promotion": _decision("PROMOTED").to_dict(),
    }]), encoding="utf-8")
    ledger = GenerationLedger(ledger_dir)
    with pytest.raises(ValueError, match="PROMOTION_UNCOMMITTED"):
        ledger.get("gen2")
    assert "gen2" not in ledger.versions()


def test_manually_written_upward_revision_does_not_change_effective_verdict(tmp_path):
    ledger = GenerationLedger(tmp_path / "ledger")
    _record(ledger, model={"path": str(tmp_path / "primary"), "revision": "synthetic-primary@sha"}, verdict="REJECTED")
    revisions = tmp_path / "ledger" / "adjudication_revisions.json"
    revisions.write_text(json.dumps([{
        "generation_version": "gen2", "supersedes_adjudication_id": "original",
        "revision_id": "gen2-adjudication-001", "reason_codes": ["NO_OWNER_SIGNOFF"],
        "policy_version": "fake", "new_verdict": "PROMOTED", "evidence_refs": [],
        "recorded_at": "2026-10-22T00:00:00Z",
        "original_decision_digest": "0" * 64,
    }]), encoding="utf-8")
    with pytest.raises(ValueError, match="ADJUDICATION_PROMOTION_UNAUTHORIZED"):
        GenerationLedger(tmp_path / "ledger").effective_verdict("gen2")


@pytest.mark.parametrize("model", [
    {"path": "Z:/not-real/primary", "revision": "spoofed-primary"},
    {"base_model_path": "Z:/not-real/primary", "lineage": "unknown"},
])
def test_new_promoted_record_cannot_claim_nonreap_path_to_bypass_transaction(tmp_path, model):
    ledger = GenerationLedger(tmp_path / "ledger")
    with pytest.raises(ValueError, match="PROMOTION_TRANSACTION_REQUIRED"):
        _record(ledger, model=model, verdict="PROMOTED")
    assert not (tmp_path / "ledger" / "generations.json").exists()


def test_prepared_transaction_alone_never_makes_promoted_record_readable(tmp_path):
    ledger = GenerationLedger(tmp_path / "ledger")
    model = {"path": "Z:/not-real/nonreap", "revision": "synthetic-primary"}
    decision = _decision("PROMOTED")
    prepare_transaction(
        tmp_path, cycle_id="N5-test", candidate_version="gen2",
        run_document={"cycle_id": "N5-test"}, promotion=decision.to_dict(),
        manifest_digest="a" * 64,
    )
    _record(ledger, model=model, verdict="PROMOTED")
    with pytest.raises(ValueError, match="PROMOTION_UNCOMMITTED"):
        ledger.get("gen2")
