"""Fail-closed external settlement admission for a completed campaign.

An accounting ledger's SHA256 is only a self-check. The CLI cannot treat an
arbitrary caller-provided digest as an attestation that training was charged:
it needs the *independently published* same-root campaign-run record, its
terminal journal, and matching paid attempt boundaries.

These mutable local files are not cryptographic signatures. This join rejects
standalone and casually edited cost evidence; resisting an attacker who can
rewrite the entire state root requires a separately signed/append-only anchor.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .campaign_runner import _journal_outcome
from .compute_cost import ComputeCost, checked_ledger_total
from .promotion_transaction import content_digest
from .campaign_resume import declared_inputs_digest, validate_evidence


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(f"CAMPAIGN_SETTLEMENT_UNVERIFIED: {detail}")


def verified_settlement_total(
    *,
    cycle_id: str,
    parent_version: str,
    candidate_version: str,
    state_root: Path,
    accounting: Path,
    manifest: Any,
) -> ComputeCost:
    """Return measured spend only when a full, concluded campaign names it."""
    root = state_root.resolve()
    expected_accounting = root / "cycle_compute_accounting.json"
    _require(
        accounting.resolve() == expected_accounting,
        "accounting path is not the campaign state root's canonical ledger",
    )
    try:
        ledger_doc = json.loads(accounting.read_text(encoding="utf-8"))
        run = json.loads((root / "campaign-run.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(
            "CAMPAIGN_SETTLEMENT_UNVERIFIED: recorded ledger or terminal "
            "campaign-run.json missing or unreadable"
        ) from error
    _require(isinstance(run, dict), "campaign-run.json is not an object")
    _require(isinstance(ledger_doc, dict), "accounting ledger is not an object")
    _require(
        run.get("cycle_id") == cycle_id
        and run.get("parent_version") == parent_version
        and run.get("candidate_version") == candidate_version,
        "recorded campaign identity does not match the declaration",
    )
    journal = root / "campaign-journal.jsonl"
    terminal = _journal_outcome(journal)
    _require(
        terminal is not None and run.get("verdict") == terminal,
        "durable terminal journal and recorded verdict are missing or inconsistent",
    )
    recorded_cost = run.get("cost")
    _require(isinstance(recorded_cost, Mapping), "recorded campaign has no cost record")
    declared_path = recorded_cost.get("accounting_path")
    _require(
        isinstance(declared_path, str)
        and Path(declared_path).resolve() == expected_accounting,
        "campaign-run.json does not bind this exact accounting path",
    )
    recorded_digest = recorded_cost.get("accounting_digest")
    _require(
        isinstance(recorded_digest, str)
        and len(recorded_digest) == 64
        and recorded_digest == ledger_doc.get("digest_sha256"),
        "ledger digest is not independently bound to campaign-run.json",
    )
    try:
        # Production supports a truly free synthetic/cached evaluator. That
        # zero row does not prove work was paid: the independent run, journal
        # and separately positive training entries below provide that proof.
        total = checked_ledger_total(
            ledger_doc, cycle_id=cycle_id, allow_zero_evaluation=True,
        )
    except (ValueError, TypeError) as error:
        raise ValueError(f"CAMPAIGN_SETTLEMENT_UNVERIFIED: {error}") from error
    _require(
        isinstance(recorded_cost.get("wall_gpu_hours"), (int, float))
        and isinstance(recorded_cost.get("device_gpu_hours"), (int, float))
        and abs(recorded_cost["wall_gpu_hours"] - total.wall_gpu_hours) < 1e-10
        and abs(recorded_cost["device_gpu_hours"] - total.device_gpu_hours) < 1e-10,
        "campaign-run.json cost totals differ from the reaggregated ledger",
    )
    # This CLI computes compliance itself, but it must not produce a positive
    # result contradicting the terminal campaign's own recorded settlement.
    # A REFUSED early-exit can intentionally lack settlement (for example,
    # evaluator failure after charged training); in that case only a fresh
    # accounting answer is available, not an independently certified verdict.
    from .campaign import settle_campaign

    actual_settlement = settle_campaign(manifest, total=total).to_dict()
    recorded_settlement = run.get("settlement")
    _require(isinstance(recorded_settlement, dict), "recorded settlement is malformed")
    if run.get("verdict") != "REFUSED" or recorded_settlement:
        _require(
            type(recorded_settlement.get("budget_compliant")) is bool
            and recorded_settlement == actual_settlement,
            "terminal campaign settlement disagrees with recomputed budget result",
        )
    entries = ledger_doc.get("entries") or []
    paid = [
        row for row in entries
        if row.get("kind") in ("training", "failed_attempt", "evaluation", "load_probe")
    ]
    _require(
        bool(paid) and any(
            row["cost"].get("wall_gpu_hours", 0) > 0
            or row["cost"].get("device_gpu_hours", 0) > 0
            for row in paid
        ),
        "zero-entry/zero-spend ledger cannot prove paid campaign work",
    )
    # The journal is written *before* the trainer invocation. An accounting
    # document listing fewer training entries than started invocations can be
    # a forged winner-only/cost-omitting ledger. Never accept it.
    try:
        lines = journal.read_text(encoding="utf-8").splitlines(keepends=True)
        _require(bool(lines) and all(line.endswith("\n") for line in lines), "torn campaign journal")
        journal_rows = [json.loads(line) for line in lines]
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        raise ValueError("CAMPAIGN_SETTLEMENT_UNVERIFIED: unreadable attempt journal") from error
    # Freeze the actual admission declaration, not merely three human-readable
    # IDs. Otherwise the caller could enlarge its budget in a copied manifest
    # and turn the exact same historical overspend into a CLI success.
    from chowder.worker_env import chowder_source_identity

    identity = content_digest({
        "declaration": content_digest(asdict(manifest)),
        "declared_files": declared_inputs_digest(manifest),
        "source_sha256": chowder_source_identity()["source_sha256"],
    })
    campaign_starts = [
        row for row in journal_rows
        if isinstance(row, dict) and row.get("phase") == "campaign_start"
    ]
    _require(
        len(campaign_starts) == 1
        and campaign_starts[0].get("identity") == identity,
        "campaign admission identity does not match this manifest, input "
        "bytes and source version",
    )
    terminal_row = journal_rows[-1]
    _require(
        isinstance(terminal_row, dict)
        and terminal_row.get("phase") == "terminal"
        and type(terminal_row.get("gpu_hours")) in (int, float)
        and abs(terminal_row["gpu_hours"] - total.wall_gpu_hours) < 1e-10,
        "terminal journal's separately recorded spend disagrees with ledger",
    )
    started = [
        row for row in journal_rows
        if isinstance(row, dict)
        and row.get("phase") == "attempt"
        and row.get("status") == "started"
    ]
    charged_attempts = [
        row for row in paid if row["kind"] in ("training", "failed_attempt")
    ]
    _require(bool(started), "journal contains no paid training invocation")
    _require(
        len(charged_attempts) == len(started),
        "paid training entry count does not match journaled invocation starts",
    )
    from collections import Counter

    _require(
        Counter(str(row.get("recipe_id")) for row in charged_attempts)
        == Counter(str(row.get("recipe_id")) for row in started),
        "charged recipe identities do not match journaled attempts",
    )
    # Local training also emits a receipt digest *in the journal*, separately
    # from the aggregate ledger. Cross-check every such receipt against the
    # originally charged attempt row, not just against its own self-hash. A
    # rehashed winner-only/substituted cost entry must not pass.
    indexed_starts = [
        (pos, row) for pos, row in enumerate(journal_rows)
        if isinstance(row, dict) and row.get("phase") == "attempt"
        and row.get("status") == "started"
    ]
    indexed_receipts = [
        (pos, row) for pos, row in enumerate(journal_rows)
        if isinstance(row, dict) and row.get("phase") == "attempt_receipt"
    ]
    receipt_dir = root / "campaign-attempt-receipts"
    receipt_paths = set(receipt_dir.glob("*.json")) if receipt_dir.is_dir() else set()
    expected_receipt_paths = {
        receipt_dir / f"{index:06d}.json" for index in range(len(started))
    }
    _require(
        len(indexed_receipts) == len(indexed_starts)
        and receipt_paths == expected_receipt_paths,
        "every paid attempt needs one canonical, durable indexed receipt",
    )
    # A simple length check can be bypassed by duplicating receipt index 0
    # while leaving index 1 unchecked; enforce a complete ordered bijection,
    # with each receipt following its own start and preceding the next start.
    for expected_index, (position, row) in enumerate(indexed_receipts):
        index = row.get("index")
        _require(
            type(index) is int and index == expected_index,
            "receipt has duplicate, missing or out-of-order attempt index",
        )
        start_position, started_row = indexed_starts[expected_index]
        _require(
            started_row.get("index") == expected_index
            and start_position < position
            and (expected_index + 1 == len(indexed_starts)
                 or position < indexed_starts[expected_index + 1][0]),
            "receipt is not ordered after its matching attempt start",
        )
        started_row = started[index]
        receipt_file = receipt_dir / f"{index:06d}.json"
        _require(receipt_file in receipt_paths, "receipt event has no indexed receipt file")
        try:
            receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as error:
            raise ValueError("CAMPAIGN_SETTLEMENT_UNVERIFIED: corrupt attempt receipt") from error
        _require(
            isinstance(receipt, dict) and receipt.get("index") == index
            and receipt.get("protocol") == 1
            and receipt.get("spec_digest") == started_row.get("identity")
            and receipt.get("spec_digest") == row.get("identity")
            and content_digest(receipt) == row.get("detail"),
            "receipt digest/identity does not match the attempt journal",
        )
        evidence = receipt.get("evidence")
        billed = charged_attempts[index]
        _require(
            isinstance(evidence, dict)
            and evidence.get("attempt") == row.get("attempt")
            and evidence.get("recipe_id") == row.get("recipe_id")
            and billed.get("recipe_id") == row.get("recipe_id")
            and row.get("status") == evidence.get("status")
            and billed.get("kind") == (
                "training" if evidence.get("status") == "SUCCEEDED" else "failed_attempt"
            ),
            "receipt attempt/recipe does not match charged training",
        )
        try:
            validate_evidence(root, evidence)
        except (ValueError, TypeError, OSError) as error:
            raise ValueError(
                f"CAMPAIGN_SETTLEMENT_UNVERIFIED: receipt's original attempt "
                f"evidence or checkpoint cannot be verified: {error}"
            ) from error
        try:
            source_cost = ComputeCost.from_dict(evidence["actual_cost"])
            billed_cost = ComputeCost.from_dict(billed["cost"])
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError("CAMPAIGN_SETTLEMENT_UNVERIFIED: invalid receipt cost") from error
        _require(
            source_cost == billed_cost,
            "durable attempt receipt cost differs from settled ledger entry",
        )
    return total
