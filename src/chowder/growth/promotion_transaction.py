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
    run_document: Mapping[str, Any], promotion: Mapping[str, Any],
    manifest_digest: str,
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
        "parent_version": run_document.get("parent_version"),
        "manifest_digest": manifest_digest,
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


def _reap_approval_in_record(root: Path, document: Mapping[str, Any], generation: Mapping[str, Any]) -> bool:
    """Refuse recovery/read of a REAP PROMOTED result with no bound N5 evidence.

    The immutable transaction digest binds the approval phase to this run's
    selected adapter. This is audit evidence of an interactive confirmation,
    not cryptographic proof of the operator's identity.
    """
    from .owner_approval import requires_owner_approval

    base_model = generation.get("base_model") or {}
    if not isinstance(base_model, Mapping):
        return False
    model_path = str(base_model.get("base_model_path") or "")
    if base_model.get("lineage") != "reap-moe" and (
        not model_path or not requires_owner_approval(model_path)
    ):
        return True
    selection = document.get("selection")
    if not isinstance(selection, Mapping):
        return False
    artifact = selection.get("artifact_sha256")
    if not isinstance(artifact, str) or len(artifact) != 64:
        return False
    phases = document.get("phases")
    if not isinstance(phases, list):
        return False
    matches = [
        item for item in phases
        if isinstance(item, Mapping) and item.get("phase") == "human_approval"
        and item.get("verdict") == "approved"
        and item.get("cycle_id") == document.get("cycle_id")
        and item.get("candidate_version") == document.get("candidate_version")
        and item.get("artifact_sha256") == artifact
        and item.get("frozen_judge_passed") is True
    ]
    return len(matches) == 1


def reconcile_transaction(
    root: Path, *, cycle_id: str, candidate_version: str,
    journal_verdict: str | None, manifest_digest: str,
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
        or marker.get("manifest_digest") != manifest_digest
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
        or not isinstance(document.get("parent_version"), str)
        or not document["parent_version"]
        or (marker.get("parent_version") is not None
            and marker["parent_version"] != document["parent_version"])
        or not isinstance(outcome, dict)
        or outcome.get("cycle_id") != cycle_id
        or outcome.get("candidate_version") != candidate_version
        or outcome.get("parent_version") != document["parent_version"]
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
        or generation.get("parent_version") != document["parent_version"]
        or not isinstance(promotion, dict)
        or promotion.get("verdict") != "PROMOTED"
        or content_digest(promotion) != marker.get("promotion_digest")
        or outcome.get("promotion") != promotion
    ):
        return None
    if not _reap_approval_in_record(root, document, generation):
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
        # Historical REAP promotions cannot acquire N5 authority merely by
        # omitting the transaction: require explicit provenance migration and
        # owner release instead of making a legacy row visible as PROMOTED.
        if isinstance(record.get("promotion"), Mapping) and record["promotion"].get("verdict") == "PROMOTED":
            from .owner_approval import requires_owner_approval

            identity = record.get("base_model")
            if not isinstance(identity, Mapping):
                raise ValueError("PROMOTION_UNCOMMITTED: promoted model identity is missing")
            path_value = str(identity.get("base_model_path") or identity.get("path") or "")
            if not path_value or not Path(path_value).is_absolute():
                raise ValueError("PROMOTION_UNCOMMITTED: promoted model has no absolute verifiable base path")
            # Old records may lack transactions, but a nonexistent or
            # unclassifiable base is not evidence that this was non-REAP.
            # Recheck physical identity on every read, including after reload.
            base = Path(path_value)
            try:
                config = json.loads((base / "config.json").read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError) as exc:
                raise ValueError(
                    "PROMOTION_UNCOMMITTED: historical promoted base identity cannot be verified"
                ) from exc
            if (
                not base.is_dir()
                or not isinstance(config, dict)
                or not isinstance(config.get("model_type"), str)
                or not config["model_type"].strip()
                or identity.get("lineage") not in (None, "", "primary", "dense")
                or config["model_type"] == "qwen3_5_moe"
                or requires_owner_approval(base)
            ):
                raise ValueError(
                    "PROMOTION_UNCOMMITTED: historical promoted base is unverified or "
                    "may require REAP N5 approval and a committed transaction"
                )
        return
    if (
        marker.get("state") != "COMMITTED"
        or marker.get("candidate_version") != record["version"]
        or marker.get("cycle_id") != record["cycle_id"]
        or marker.get("generation_digest") != content_digest(record)
        or marker.get("promotion_digest") != content_digest(record["promotion"])
    ):
        raise ValueError("PROMOTION_UNCOMMITTED: campaign terminal commitment not proven")
    root = ledger_root.parent
    try:
        document = json.loads((root / "campaign-run.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        document = None
    if not isinstance(document, Mapping) or not _reap_approval_in_record(root, document, record):
        raise ValueError("PROMOTION_UNCOMMITTED: REAP owner approval evidence not proven")
    # Keep this binding checked across processes; the approval phase is not
    # merely an unverified string in a separately mutable campaign JSON.
    if content_digest({k: v for k, v in document.items() if k != "cycle_outcome"}) != marker.get("run_digest"):
        raise ValueError("PROMOTION_UNCOMMITTED: campaign record changed after approval")
    outcome = document.get("cycle_outcome")
    if (
        document.get("verdict") != "PROMOTED"
        or document.get("cycle_id") != record["cycle_id"]
        or document.get("candidate_version") != record["version"]
        or not isinstance(document.get("parent_version"), str)
        or not document["parent_version"]
        or document["parent_version"] != record.get("parent_version")
        or (marker.get("parent_version") is not None
            and marker["parent_version"] != document["parent_version"])
        or not isinstance(outcome, Mapping)
        or outcome.get("cycle_id") != record["cycle_id"]
        or outcome.get("candidate_version") != record["version"]
        or outcome.get("parent_version") != document["parent_version"]
        or outcome.get("verdict") != "PROMOTED"
        or outcome.get("promotion") != record["promotion"]
    ):
        raise ValueError("PROMOTION_UNCOMMITTED: campaign terminal outcome disagrees with lineage")
    # The digest intentionally excludes cycle_outcome because it is published
    # after finalization. Verify that late field and the fsynced journal on
    # every read; a COMMITTED marker alone cannot keep a stale promotion live.
    from .campaign_runner import _journal_outcome

    if _journal_outcome(root / "campaign-journal.jsonl") != "PROMOTED":
        raise ValueError("PROMOTION_UNCOMMITTED: final terminal journal is absent or inconsistent")
