"""Crash-safe coordination of campaign verdicts and promoted lineage.

The ledger and campaign journal live in separate files and cannot share an
atomic rename. A durable PREPARED marker therefore prevents lineage readers
from admitting a generation until the exact campaign verdict, ledger entry,
and terminal journal agree. The COMMITTED marker remains as an attestation.

Recovery may commit an already-complete transaction; it never invents a
missing terminal verdict, recomputes a decision, or retries spent training.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


def content_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def transaction_path(ledger_root: Path, candidate_version: str) -> Path:
    # Candidate versions are manifest-controlled; they must never be filenames.
    safe_name = hashlib.sha256(candidate_version.encode("utf-8")).hexdigest()
    return ledger_root / "promotion-transactions" / (safe_name + ".json")


def atomic_json(path: Path, value: Mapping[str, Any] | list[Any]) -> None:
    """Publish complete JSON bytes; a process kill cannot truncate the prior file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    if os.name != "nt":
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def load_transaction(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError, OSError) as exc:
        raise ValueError("PROMOTION_TRANSACTION_INVALID: unreadable transaction") from exc
    if not isinstance(payload, dict) or payload.get("protocol") != 1:
        raise ValueError("PROMOTION_TRANSACTION_INVALID: unknown transaction format")
    if payload.get("state") not in {"PREPARED", "COMMITTED"}:
        raise ValueError("PROMOTION_TRANSACTION_INVALID: unknown transaction state")
    return payload


def prepare_transaction(
    root: Path, *, cycle_id: str, candidate_version: str,
    run_document: Mapping[str, Any], promotion: Mapping[str, Any]
) -> None:
    """Write the intent before GrowthCycle.finalize can touch lineage."""
    path = transaction_path(root / "ledger", candidate_version)
    if path.exists():
        raise ValueError("PROMOTION_TRANSACTION_EXISTS: generation already has a transaction")
    # An earlier, transaction-less generation may use the same version.
    # Refuse without placing a PREPARED marker over that existing lineage.
    if _matching_generation(root, candidate_version) is not None:
        raise ValueError("PROMOTION_TRANSACTION_EXISTS: generation already recorded")
    atomic_json(path, {
        "protocol": 1,
        "state": "PREPARED",
        "cycle_id": cycle_id,
        "candidate_version": candidate_version,
        "run_digest": content_digest(run_document),
        "promotion_digest": content_digest(promotion),
    })


def _matching_generation(root: Path, candidate_version: str) -> dict[str, Any] | None:
    try:
        rows = json.loads((root / "ledger" / "generations.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(rows, list):
        return None
    matches = [
        item for item in rows
        if isinstance(item, dict) and item.get("version") == candidate_version
    ]
    return matches[0] if len(matches) == 1 else None


def reconcile_transaction(
    root: Path, *, cycle_id: str, candidate_version: str,
    journal_verdict: str | None
) -> dict[str, Any] | None:
    """Return a verified durable verdict, finishing PREPARED iff all evidence exists.

    A partially written or contradictory state remains PREPARED and invisible
    to lineage consumers. Repeated recovery makes no duplicate entries.
    """
    path = transaction_path(root / "ledger", candidate_version)
    marker = load_transaction(path)
    if marker is None:
        return None
    if (
        marker.get("cycle_id") != cycle_id
        or marker.get("candidate_version") != candidate_version
        or journal_verdict != "PROMOTED"
    ):
        return None
    try:
        document = json.loads((root / "campaign-run.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(document, dict):
        return None
    outcome = document.get("cycle_outcome")
    if (
        document.get("verdict") != "PROMOTED"
        or document.get("cycle_id") != cycle_id
        or document.get("candidate_version") != candidate_version
        or not isinstance(outcome, dict)
        or outcome.get("cycle_id") != cycle_id
        or outcome.get("candidate_version") != candidate_version
        or outcome.get("verdict") != "PROMOTED"
    ):
        return None
    without_outcome = {key: value for key, value in document.items() if key != "cycle_outcome"}
    if content_digest(without_outcome) != marker.get("run_digest"):
        return None
    generation = _matching_generation(root, candidate_version)
    promotion = generation.get("promotion") if generation else None
    if (
        generation is None
        or generation.get("cycle_id") != cycle_id
        or not isinstance(promotion, dict)
        or promotion.get("verdict") != "PROMOTED"
        or content_digest(promotion) != marker.get("promotion_digest")
        or outcome.get("promotion") != promotion
    ):
        return None
    generation_digest = content_digest(generation)
    if marker["state"] == "COMMITTED":
        if marker.get("generation_digest") != generation_digest:
            return None
    else:
        atomic_json(path, {**marker, "state": "COMMITTED", "generation_digest": generation_digest})
    return document


def assert_committed(ledger_root: Path, record: Mapping[str, Any]) -> None:
    """Block raw get/effective_verdict for any generation awaiting commit."""
    path = transaction_path(ledger_root, str(record["version"]))
    marker = load_transaction(path)
    if marker is None:  # Historical/non-campaign generation.
        return
    if (
        marker.get("state") != "COMMITTED"
        or marker.get("candidate_version") != record["version"]
        or marker.get("cycle_id") != record["cycle_id"]
        or marker.get("generation_digest") != content_digest(record)
        or marker.get("promotion_digest") != content_digest(record["promotion"])
    ):
        raise ValueError("PROMOTION_UNCOMMITTED: campaign terminal commitment not proven")
