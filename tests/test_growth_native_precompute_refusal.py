"""A native preflight refusal is not an unreceipted paid GPU attempt."""

import json
from pathlib import Path

import pytest

from chowder.growth.campaign_runner import run_campaign
from test_growth_campaign_runner import _campaign


def test_unstarted_native_preflight_is_recorded_but_never_fabricates_receipt(tmp_path: Path):
    manifest, _runner, _document = _campaign(tmp_path)
    run = run_campaign(manifest)
    assert run.verdict == "REFUSED"
    assert run.cost["wall_gpu_hours"] == pytest.approx(0.0)
    root = Path(manifest.state_root)
    rows = [json.loads(row) for row in (root / "campaign-journal.jsonl").read_text().splitlines()]
    starts = [row for row in rows if row.get("phase") == "attempt" and row.get("status") == "started"]
    assert starts and all(row.get("proof_mode") == "receipt" for row in starts)
    assert not list((root / "campaign-attempt-receipts").glob("*.json"))
