"""Owner N5: qualifying a candidate never silently advances a REAP generation."""

import io
import json
from types import SimpleNamespace

import pytest

from chowder.growth import owner_approval as oa


SHA = "a" * 64


class _Console(io.StringIO):
    def isatty(self):
        return True


def test_default_headless_and_environment_spoof_fail_closed(monkeypatch):
    monkeypatch.setenv("CHOWDER_AUTO_APPROVE", "1")
    monkeypatch.setenv("CHOWDER_OWNER_APPROVED", "true")
    monkeypatch.setattr(oa.sys, "stdin", io.StringIO("APPROVE c gen2 " + SHA + "\n"))
    monkeypatch.setattr(oa.sys, "stdout", io.StringIO())
    assert oa.authorize_promotion(cycle_id="c", candidate_version="gen2", artifact_sha256=SHA,
                                  judge_certified=True) is None


def test_interactive_approval_requires_exact_full_bound_phrase(monkeypatch):
    bad = _Console("APPROVE c gen2 " + "b" * 64 + "\n")
    monkeypatch.setattr(oa.sys, "stdin", bad)
    monkeypatch.setattr(oa.sys, "stdout", _Console())
    assert oa.authorize_promotion(cycle_id="c", candidate_version="gen2", artifact_sha256=SHA,
                                  judge_certified=True) is None
    monkeypatch.setattr(oa.sys, "stdin", _Console("APPROVE c gen2 " + SHA + "\n"))
    assert oa.authorize_promotion(cycle_id="c", candidate_version="gen2", artifact_sha256=SHA) is None
    monkeypatch.setattr(oa.sys, "stdin", _Console("APPROVE c gen2 " + SHA + "\n"))
    grant = oa.authorize_promotion(cycle_id="c", candidate_version="gen2", artifact_sha256=SHA,
                                    judge_certified=True)
    assert oa.verify_grant(grant, cycle_id="c", candidate_version="gen2", artifact_sha256=SHA)
    assert not oa.verify_grant(grant, cycle_id="wrong", candidate_version="gen2", artifact_sha256=SHA)
    assert not oa.verify_grant(grant, cycle_id="c", candidate_version="gen3", artifact_sha256=SHA)
    assert not oa.verify_grant(grant, cycle_id="c", candidate_version="gen2", artifact_sha256="b" * 64)


def test_malformed_mac_never_promotes():
    fake = oa.ApprovalGrant("c", "gen2", SHA, True, "0" * 64)
    assert not oa.verify_grant(fake, cycle_id="c", candidate_version="gen2", artifact_sha256=SHA)
    assert not oa.verify_grant(None, cycle_id="c", candidate_version="gen2", artifact_sha256=SHA)


def test_reap_classification_uses_content_not_environment_toggle(monkeypatch, tmp_path):
    parent = tmp_path / "model"
    parent.mkdir()
    config = parent / "config.json"
    config.write_text(json.dumps({"model_type": "qwen3_5_moe", "text_config": {"num_experts": 160}}))
    assert oa.requires_owner_approval(parent)
    config.write_text(json.dumps({"model_type": "qwen3_5_moe", "text_config": {"num_experts": 128}}))
    assert oa.requires_owner_approval(parent)
    monkeypatch.setenv("CHOWDER_IGNORE_HUMAN_APPROVAL", "1")
    assert oa.requires_owner_approval(parent)
    config.write_text("{}")
    assert not oa.requires_owner_approval(parent)
    (parent / "reap-prune.json").write_text('{"method":"reap"}')
    assert oa.requires_owner_approval(parent)


def test_direct_cycle_finalize_without_base_identity_cannot_promote():
    """A direct low-level call must not evade N5 by omitting model identity."""
    from chowder.growth.cycle import GrowthCycle

    cycle = SimpleNamespace(config=SimpleNamespace(cycle_id="c", parent_version="gen1", candidate_version="gen2"))
    decision = SimpleNamespace(verdict="PROMOTED", reasons=(), checks={})
    with pytest.raises(ValueError, match="PROMOTION_BASE_UNDECLARED"):
        GrowthCycle.finalize(
            cycle, decision,
            base_model={}, dataset_manifest_ref="dataset", curriculum_manifest_ref="curriculum",
            recipe={}, training_evidence_ref="train", evaluation_report_ref="eval",
        )


def test_direct_lineage_record_refuses_reap_without_approval(tmp_path):
    from chowder.growth.lineage import GenerationLedger
    from chowder.growth.promotion_transaction import prepare_transaction
    from chowder.growth.promotion import PromotionDecision

    base = tmp_path / "base"
    base.mkdir()
    (base / "reap-prune.json").write_text('{"method":"reap"}', encoding="utf-8")
    fake_decision = PromotionDecision(verdict="PROMOTED", reasons=(), checks={})
    ledger = GenerationLedger(tmp_path / "ledger")
    prepare_transaction(tmp_path, cycle_id="c", candidate_version="gen2",
                        run_document={"cycle_id": "c"},
                        promotion=fake_decision.to_dict(), manifest_digest="a" * 64)
    with pytest.raises(ValueError, match="HUMAN_APPROVAL_REQUIRED"):
        ledger.record(
            version="gen2", parent_version="gen1", cycle_id="c",
            base_model={"base_model_path": str(base), "lineage": "reap-moe"},
            dataset_manifest_ref="dataset", curriculum_manifest_ref="curriculum",
            recipe={}, training_evidence_ref="train", evaluation_report_ref="eval",
            promotion=fake_decision,
        )
    assert not (tmp_path / "ledger" / "generations.json").exists()


def test_direct_cycle_finalize_with_reap_model_refuses_without_owner_grant(tmp_path):
    from chowder.growth.cycle import GrowthCycle

    base = tmp_path / "reap"
    base.mkdir()
    (base / "reap-prune.json").write_text('{"method":"reap"}', encoding="utf-8")
    cycle = SimpleNamespace(config=SimpleNamespace(cycle_id="c", parent_version="gen1", candidate_version="gen2"))
    decision = SimpleNamespace(verdict="PROMOTED", reasons=(), checks={})
    with pytest.raises(ValueError, match="HUMAN_APPROVAL_REQUIRED"):
        GrowthCycle.finalize(
            cycle, decision,
            base_model={"base_model_path": str(base), "lineage": "reap-moe"},
            dataset_manifest_ref="dataset", curriculum_manifest_ref="curriculum",
            recipe={}, training_evidence_ref="train", evaluation_report_ref="eval",
        )
