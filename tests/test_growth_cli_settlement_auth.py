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
