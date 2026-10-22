"""A promotion's durable transaction must not switch parent generations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chowder.growth.lineage import GenerationLedger
from chowder.growth.promotion import PromotionDecision
from chowder.growth.promotion_transaction import (
    load_transaction,
    prepare_transaction,
    reconcile_transaction,
    transaction_path,
)


def _prepare_record_and_terminal(
    root: Path, *, run_parent: str = "gen1", ledger_parent: str = "gen1",
    outcome_parent: str = "gen1",
) -> tuple[GenerationLedger, Path]:
    base = root / "dense-base"
    base.mkdir(parents=True)
    (base / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    decision = PromotionDecision(verdict="PROMOTED", reasons=("fixture",), checks={})
    run = {
        "cycle_id": "transaction-parent", "candidate_version": "gen2",
        "parent_version": run_parent, "verdict": "PROMOTED", "phases": [],
    }
    prepare_transaction(
        root, cycle_id="transaction-parent", candidate_version="gen2",
        run_document=run, promotion=decision.to_dict(), manifest_digest="a" * 64,
    )
    ledger = GenerationLedger(root / "ledger")
    ledger.record(
        version="gen2", parent_version=ledger_parent, cycle_id="transaction-parent",
        base_model={"base_model_path": str(base), "lineage": "primary"},
        dataset_manifest_ref="data", curriculum_manifest_ref="curriculum", recipe={},
        training_evidence_ref="train", evaluation_report_ref="eval", promotion=decision,
    )
    (root / "campaign-run.json").write_text(json.dumps({
        **run,
        "cycle_outcome": {
            "cycle_id": "transaction-parent", "parent_version": outcome_parent,
            "candidate_version": "gen2", "verdict": "PROMOTED",
            "promotion": decision.to_dict(),
        },
    }), encoding="utf-8")
    (root / "campaign-journal.jsonl").write_text(
        json.dumps({"phase": "campaign_start", "status": "started"}) + "\n"
        + json.dumps({"phase": "terminal", "status": "PROMOTED"}) + "\n", encoding="utf-8",
    )
    marker = transaction_path(root / "ledger", "gen2")
    return ledger, marker


@pytest.mark.parametrize("parents", [
    ("gen1", "gen0", "gen1"),
    ("gen1", "gen1", "gen0"),
    ("gen1", "gen0", "gen0"),
])
def test_reconcile_refuses_parent_ancestry_divergence(tmp_path: Path, parents):
    ledger, marker = _prepare_record_and_terminal(
        tmp_path, run_parent=parents[0], ledger_parent=parents[1],
        outcome_parent=parents[2],
    )
    assert reconcile_transaction(
        tmp_path, cycle_id="transaction-parent", candidate_version="gen2",
        journal_verdict="PROMOTED", manifest_digest="a" * 64,
    ) is None
    assert load_transaction(marker)["state"] == "PREPARED"
    with pytest.raises(ValueError, match="PROMOTION_UNCOMMITTED"):
        ledger.get("gen2")


def test_reconcile_accepts_exact_parent_and_read_refuses_late_outcome_change(tmp_path: Path):
    ledger, marker = _prepare_record_and_terminal(tmp_path)
    assert load_transaction(marker)["parent_version"] == "gen1"
    assert reconcile_transaction(
        tmp_path, cycle_id="transaction-parent", candidate_version="gen2",
        journal_verdict="PROMOTED", manifest_digest="a" * 64,
    ) is not None
    assert load_transaction(marker)["state"] == "COMMITTED"
    assert ledger.get("gen2").parent_version == "gen1"
    record_path = tmp_path / "campaign-run.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["cycle_outcome"]["parent_version"] = "gen0"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="PROMOTION_UNCOMMITTED"):
        GenerationLedger(tmp_path / "ledger").get("gen2")
