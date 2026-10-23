"""The standalone settlement CLI must not certify a self-hashed forged ledger."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from chowder.cli import main as chowder_main
from chowder.growth import campaign_runner
from chowder.growth.campaign_runner import run_campaign
from chowder.growth.compute_cost import CycleCostLedger, ledger_digest
from chowder.growth.compute_cost import ComputeCost
from chowder.growth.promotion_transaction import content_digest
from test_growth_campaign_runner import (
    OVERRUN_WALL_GPU_HOURS,
    _RecordingRunner,
    _campaign,
    _patch_seams,
    _tight_campaign_budget,
    _verbs,
)


def settle(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    manifest_path: Path, accounting_path: Path,
) -> tuple[int, dict]:
    monkeypatch.setattr(
        sys, "argv",
        ["chowder", "growth", "campaign", "settle", str(manifest_path), str(accounting_path)],
    )
    code = chowder_main()
    payload = json.loads(capsys.readouterr().out)
    return code, payload


def test_fabricated_zero_entry_self_hashed_ledger_is_never_compliant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest, _runner, _ = _campaign(tmp_path)
    root = Path(manifest.state_root)
    root.mkdir(parents=True, exist_ok=True)
    ledger = root / "cycle_compute_accounting.json"
    CycleCostLedger(manifest.cycle_id).write(ledger)
    code, payload = settle(
        monkeypatch, capsys, tmp_path / "inputs" / "campaign.json", ledger,
    )
    assert code == 1
    assert payload["budget_compliant"] is False
    assert "CAMPAIGN_SETTLEMENT_UNVERIFIED" in payload["refusal_reason"]


def test_real_terminal_campaign_accounting_settles_under_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest, runner, _ = _campaign(tmp_path, with_ancestor=True)
    _patch_seams(monkeypatch, runner)
    run = run_campaign(manifest)
    assert run.cost["accounting_digest"]
    trained = _verbs(runner).count("train")
    code, payload = settle(
        monkeypatch, capsys, tmp_path / "inputs" / "campaign.json",
        Path(run.cost["accounting_path"]),
    )
    assert code == 0, payload
    assert payload["budget_compliant"] is True
    assert payload["actual"]["wall_gpu_hours"] == pytest.approx(run.cost["wall_gpu_hours"])
    assert _verbs(runner).count("train") == trained  # CLI is read-only


@pytest.mark.parametrize("tamper", ["clear", "omit_one"])
def test_settlement_requires_terminal_attempt_summary_for_each_paid_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], tamper: str,
) -> None:
    """The independently written terminal run must name every paid attempt."""
    manifest, runner, _ = _campaign(tmp_path, recipe_count=2, with_ancestor=True)
    _patch_seams(monkeypatch, runner)
    run = run_campaign(manifest)
    root = Path(manifest.state_root)
    run_path = root / "campaign-run.json"
    saved = json.loads(run_path.read_text(encoding="utf-8"))
    assert len(saved["attempts"]) == 2
    saved["attempts"] = [] if tamper == "clear" else saved["attempts"][:1]
    run_path.write_text(json.dumps(saved), encoding="utf-8")
    code, result = settle(
        monkeypatch, capsys, tmp_path / "inputs" / "campaign.json",
        Path(run.cost["accounting_path"]),
    )
    assert code == 1, result
    assert result["budget_compliant"] is False
    assert "terminal attempt summary" in result["refusal_reason"]


@pytest.mark.parametrize("tamper", ["duplicate_first", "drop_all", "out_of_order"])
def test_settlement_refuses_nonbijective_or_unordered_attempt_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], tamper: str,
) -> None:
    """One receipt must independently prove each paid, ordered attempt."""
    manifest, runner, _ = _campaign(tmp_path, recipe_count=2, with_ancestor=True)
    _patch_seams(monkeypatch, runner)
    run = run_campaign(manifest)
    root = Path(manifest.state_root)
    journal = root / "campaign-journal.jsonl"
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    indices = [i for i, row in enumerate(rows) if row.get("phase") == "attempt_receipt"]
    assert len(indices) == 2, rows
    if tamper == "duplicate_first":
        # The old verifier compared only counts, rereading 000000 twice and
        # never checking 000001 despite the real paid second invocation.
        rows[indices[1]] = dict(rows[indices[0]])
    elif tamper == "drop_all":
        # No receipt events and no receipt files formerly passed the equal
        # length check (zero == zero) while training rows remained charged.
        rows = [row for row in rows if row.get("phase") != "attempt_receipt"]
        for file in (root / "campaign-attempt-receipts").glob("*.json"):
            file.unlink()
    else:
        rows[indices[0]], rows[indices[1]] = rows[indices[1]], rows[indices[0]]
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    code, payload = settle(
        monkeypatch, capsys, tmp_path / "inputs" / "campaign.json",
        Path(run.cost["accounting_path"]),
    )
    assert code == 1, payload
    assert payload["budget_compliant"] is False
    assert "CAMPAIGN_SETTLEMENT_UNVERIFIED" in payload["refusal_reason"]


@pytest.mark.parametrize("tamper", ["false_success", "string_status", "clear_settlement"])
def test_settlement_rejects_disagreement_with_terminal_settlement_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], tamper: str,
) -> None:
    manifest, runner, _ = _campaign(tmp_path, with_ancestor=True)
    _patch_seams(monkeypatch, runner)
    run = run_campaign(manifest)
    assert run.verdict != "REFUSED"
    record_path = Path(manifest.state_root) / "campaign-run.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["settlement"]["budget_compliant"] is True
    if tamper == "false_success":
        record["settlement"]["budget_compliant"] = False
    elif tamper == "string_status":
        record["settlement"]["budget_compliant"] = "true"
    else:
        record["settlement"] = {}
    record_path.write_text(json.dumps(record), encoding="utf-8")
    code, payload = settle(
        monkeypatch, capsys, tmp_path / "inputs" / "campaign.json",
        Path(run.cost["accounting_path"]),
    )
    assert code == 1, payload
    assert payload["budget_compliant"] is False
    assert "CAMPAIGN_SETTLEMENT_UNVERIFIED" in payload["refusal_reason"]


def test_settlement_revalidates_original_trainer_evidence_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest, runner, _ = _campaign(tmp_path, with_ancestor=True)
    _patch_seams(monkeypatch, runner)
    run = run_campaign(manifest)
    receipt_file = Path(manifest.state_root) / "campaign-attempt-receipts" / "000000.json"
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    evidence_file = Path(receipt["evidence"]["evidence_path"])
    original = json.loads(evidence_file.read_text(encoding="utf-8"))
    original["status"] = "FAILED"
    evidence_file.write_text(json.dumps(original), encoding="utf-8")
    code, payload = settle(
        monkeypatch, capsys, tmp_path / "inputs" / "campaign.json",
        Path(run.cost["accounting_path"]),
    )
    assert code == 1, payload
    assert "CAMPAIGN_SETTLEMENT_UNVERIFIED" in payload["refusal_reason"]


@pytest.mark.parametrize("tamper_completion", [False, True])
def test_nonreplayable_executor_has_cost_bound_completion_not_native_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], tamper_completion: bool,
) -> None:
    """Remote/custom backends must not be forced to invent native receipts."""
    manifest, runner, _ = _campaign(tmp_path, recipe_count=1, with_ancestor=True)
    _patch_seams(monkeypatch, runner)

    class RemoteFixture:
        firewall = campaign_runner.ContaminationFirewall()

        def admit(self, recipe):
            return None

        def __call__(self, recipe, items):
            return {
                "status": "FAILED", "attempt": "remote-attempt-01",
                "train_started": True, "measured_gpu_hours": None,
                "actual_cost": ComputeCost.from_wall_only(
                    0.01, source="verified remote executor fixture"
                ).to_dict(),
            }

    run = run_campaign(manifest, train_fn=RemoteFixture())
    root = Path(manifest.state_root)
    assert run.verdict == "REFUSED"
    assert len(run.attempts) == 1, "failed paid invocations retain terminal provenance"
    assert not (root / "campaign-attempt-receipts").exists()
    journal = root / "campaign-journal.jsonl"
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    completions = [row for row in rows if row.get("phase") == "attempt"
                   and row.get("status") != "started"]
    assert len(completions) == 1
    assert completions[0].get("detail")
    if tamper_completion:
        completions[0]["detail"] = "f" * 64
        journal.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    code, payload = settle(
        monkeypatch, capsys, tmp_path / "inputs" / "campaign.json",
        Path(run.cost["accounting_path"]),
    )
    if tamper_completion:
        assert code == 1
        assert "CAMPAIGN_SETTLEMENT_UNVERIFIED" in payload["refusal_reason"]
    else:
        assert code == 0, payload  # cost-only compliance; training verdict was REFUSED
        assert payload["budget_compliant"] is True
        assert payload["actual"]["wall_gpu_hours"] == pytest.approx(0.01)


@pytest.mark.parametrize("tamper_start_mode", [False, True, "start_and_run_backend"])
def test_native_receipt_cannot_be_downgraded_to_nonreplayable_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], tamper_start_mode: bool | str,
) -> None:
    manifest, runner, _ = _campaign(tmp_path, with_ancestor=True)
    _patch_seams(monkeypatch, runner)
    run = run_campaign(manifest)
    root = Path(manifest.state_root)
    journal = root / "campaign-journal.jsonl"
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    starts = [row for row in rows if row.get("phase") == "attempt" and row.get("status") == "started"]
    receipts = [row for row in rows if row.get("phase") == "attempt_receipt"]
    assert starts and len(starts) == len(receipts)
    ledger = json.loads(Path(run.cost["accounting_path"]).read_text(encoding="utf-8"))
    cost = ledger["entries"][0]["cost"]
    if tamper_start_mode:
        starts[0]["proof_mode"] = "completion"
    if tamper_start_mode == "start_and_run_backend":
        record_path = root / "campaign-run.json"
        document = json.loads(record_path.read_text(encoding="utf-8"))
        phase = next(p for p in document["phases"] if p.get("phase") == "training-backend")
        phase["backend"]["executor"] = "caller-supplied"
        phase["backend"]["provider"] = None
        record_path.write_text(json.dumps(document), encoding="utf-8")
    event = receipts[0]
    event["phase"] = "attempt"
    event["detail"] = content_digest({
        "identity": starts[0]["identity"],
        "recipe_id": event["recipe_id"],
        "attempt": event["attempt"],
        "status": event["status"],
        "cost": ComputeCost.from_dict(cost).to_dict(),
    })
    (root / "campaign-attempt-receipts" / "000000.json").unlink()
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    code, payload = settle(
        monkeypatch, capsys, tmp_path / "inputs" / "campaign.json",
        Path(run.cost["accounting_path"]),
    )
    assert code == 1, payload
    assert "CAMPAIGN_SETTLEMENT_UNVERIFIED" in payload["refusal_reason"]


@pytest.mark.parametrize("invalid_status", ["RUNNING", "BOGUS", ""])
def test_nonnative_completion_rejects_nonterminal_status_even_with_valid_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], invalid_status: str,
) -> None:
    manifest, runner, _ = _campaign(tmp_path, recipe_count=1, with_ancestor=True)
    _patch_seams(monkeypatch, runner)

    class RemoteFixture:
        firewall = campaign_runner.ContaminationFirewall()

        def admit(self, recipe):
            return None

        def __call__(self, recipe, items):
            return {
                "status": "FAILED", "attempt": "remote-attempt-01",
                "train_started": True, "measured_gpu_hours": None,
                "actual_cost": ComputeCost.from_wall_only(
                    0.01, source="verified remote executor fixture"
                ).to_dict(),
            }

    run = run_campaign(manifest, train_fn=RemoteFixture())
    root = Path(manifest.state_root)
    journal = root / "campaign-journal.jsonl"
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    completion = next(
        row for row in rows
        if row.get("phase") == "attempt" and row.get("status") == "FAILED"
    )
    accounting = Path(run.cost["accounting_path"])
    ledger = json.loads(accounting.read_text(encoding="utf-8"))
    completion["status"] = invalid_status
    completion["detail"] = content_digest({
        "identity": completion["identity"],
        "recipe_id": completion["recipe_id"],
        "attempt": completion["attempt"],
        "status": invalid_status,
        "cost": ComputeCost.from_dict(ledger["entries"][0]["cost"]).to_dict(),
    })
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    code, payload = settle(
        monkeypatch, capsys, tmp_path / "inputs" / "campaign.json", accounting,
    )
    assert code == 1, payload
    assert "CAMPAIGN_SETTLEMENT_UNVERIFIED" in payload["refusal_reason"]


@pytest.mark.parametrize(
    "tamper",
    [
        "ledger_only", "run_digest", "run_total", "journal",
        "terminal_gpu_hours", "paid_rows", "receipt", "receipt_cost",
    ],
)
def test_settlement_refuses_tampered_digest_record_or_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], tamper: str,
) -> None:
    manifest, runner, _ = _campaign(tmp_path, with_ancestor=True)
    _patch_seams(monkeypatch, runner)
    run = run_campaign(manifest)
    root = Path(manifest.state_root)
    accounting = Path(run.cost["accounting_path"])
    record_path = root / "campaign-run.json"
    if tamper in {"ledger_only", "paid_rows"}:
        doc = json.loads(accounting.read_text(encoding="utf-8"))
        if tamper == "ledger_only":
            doc["entries"][0]["notes"] = "malicious rehash"
            doc["digest_sha256"] = ledger_digest(doc)
        else:
            # An attacker rehashing the entire edited accounting document
            # still has no independent completed campaign binding.
            doc = CycleCostLedger(manifest.cycle_id).render()
        accounting.write_text(json.dumps(doc), encoding="utf-8")
    elif tamper == "run_digest":
        doc = json.loads(record_path.read_text(encoding="utf-8"))
        doc["cost"]["accounting_digest"] = "0" * 64
        record_path.write_text(json.dumps(doc), encoding="utf-8")
    elif tamper == "run_total":
        doc = json.loads(record_path.read_text(encoding="utf-8"))
        doc["cost"]["wall_gpu_hours"] = 0
        record_path.write_text(json.dumps(doc), encoding="utf-8")
    elif tamper == "journal":
        (root / "campaign-journal.jsonl").write_text(
            '{"phase":"campaign_start","status":"started"}\n', encoding="utf-8",
        )
    elif tamper == "terminal_gpu_hours":
        journal = root / "campaign-journal.jsonl"
        lines = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
        lines[-1]["gpu_hours"] = 0.0
        journal.write_text(
            "".join(json.dumps(row) + "\n" for row in lines), encoding="utf-8",
        )
    else:
        receipt = root / "campaign-attempt-receipts" / "000000.json"
        entry = json.loads(receipt.read_text(encoding="utf-8"))
        if tamper == "receipt":
            entry["spec_digest"] = "f" * 64
        else:
            entry["evidence"]["actual_cost"]["wall_gpu_hours"] += 0.01
        receipt.write_text(json.dumps(entry), encoding="utf-8")
    code, payload = settle(
        monkeypatch, capsys, tmp_path / "inputs" / "campaign.json", accounting,
    )
    assert code == 1
    assert payload["budget_compliant"] is False
    assert "CAMPAIGN_SETTLEMENT_UNVERIFIED" in payload["refusal_reason"]


def test_self_hashed_zero_ledger_with_forged_terminal_run_still_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest, runner, _ = _campaign(tmp_path, with_ancestor=True)
    _patch_seams(monkeypatch, runner)
    result = run_campaign(manifest)
    root = Path(manifest.state_root)
    accounting = Path(result.cost["accounting_path"])
    zero = CycleCostLedger(manifest.cycle_id).render()
    accounting.write_text(json.dumps(zero), encoding="utf-8")
    record = root / "campaign-run.json"
    document = json.loads(record.read_text(encoding="utf-8"))
    document["cost"] = {
        **document["cost"],
        "accounting_digest": zero["digest_sha256"],
        "device_gpu_hours": 0.0,
        "wall_gpu_hours": 0.0,
    }
    record.write_text(json.dumps(document), encoding="utf-8")
    code, payload = settle(
        monkeypatch, capsys, tmp_path / "inputs" / "campaign.json", accounting,
    )
    assert code == 1
    assert payload["budget_compliant"] is False
    assert "zero-entry/zero-spend" in payload["refusal_reason"]


def test_settlement_rejects_copy_of_valid_ledger_outside_its_state_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest, runner, _ = _campaign(tmp_path, with_ancestor=True)
    _patch_seams(monkeypatch, runner)
    result = run_campaign(manifest)
    alternate = tmp_path / "copied-ledger.json"
    alternate.write_bytes(Path(result.cost["accounting_path"]).read_bytes())
    code, payload = settle(
        monkeypatch, capsys, tmp_path / "inputs" / "campaign.json", alternate,
    )
    assert code == 1
    assert payload["budget_compliant"] is False
    assert "canonical ledger" in payload["refusal_reason"]


def test_settlement_cannot_relabel_prior_overspend_by_enlarging_manifest_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest, runner, _ = _campaign(
        tmp_path, with_ancestor=True,
        budget=_tight_campaign_budget(),
        runner=_RecordingRunner(gpu_hours=OVERRUN_WALL_GPU_HOURS),
    )
    _patch_seams(monkeypatch, runner)
    run = run_campaign(manifest)
    manifest_path = tmp_path / "inputs" / "campaign.json"
    declaration = json.loads(manifest_path.read_text(encoding="utf-8"))
    declaration["budget"]["wall_gpu_hours_ceiling_campaign"] = 99.0
    manifest_path.write_text(json.dumps(declaration), encoding="utf-8")
    code, payload = settle(
        monkeypatch, capsys, manifest_path, Path(run.cost["accounting_path"]),
    )
    assert code == 1
    assert payload["budget_compliant"] is False
    assert "campaign admission identity" in payload["refusal_reason"]


def test_real_overbudget_campaign_settlement_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest, runner, _ = _campaign(
        tmp_path, with_ancestor=True,
        budget=_tight_campaign_budget(),
        runner=_RecordingRunner(gpu_hours=OVERRUN_WALL_GPU_HOURS),
    )
    _patch_seams(monkeypatch, runner)
    run = run_campaign(manifest)
    assert run.verdict == "REFUSED"
    assert run.cost["wall_gpu_hours"] > manifest.budget.wall_gpu_hours_ceiling_campaign
    code, payload = settle(
        monkeypatch, capsys, tmp_path / "inputs" / "campaign.json",
        Path(run.cost["accounting_path"]),
    )
    assert code == 1
    assert payload["budget_compliant"] is False
    assert payload["budget_failure_reasons"]
