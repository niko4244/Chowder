"""Replay verified, settled campaign attempts after a hard process interruption.

Each completed training invocation has a separately fsynced receipt and a
matching append-only journal event. Replaying a receipt is free; rerunning a
possibly executing or unpriced attempt is not. Candidate search uses these
replayed results to restore its survivor checkpoint and continue the next
round through the original validated training backend.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

from .compute_cost import ComputeCost
from .promotion_transaction import atomic_json, content_digest
from .training_binding import EVIDENCE_FILE, STATUS_SUCCEEDED, directory_digest

RESUME_UNVERIFIED = "CAMPAIGN_RESUME_UNVERIFIED"
RECEIPTS_DIR = "campaign-attempt-receipts"


class ResumeRefusal(ValueError):
    pass


def _refuse(reason: str) -> ResumeRefusal:
    return ResumeRefusal(f"{RESUME_UNVERIFIED}: {reason}")


def receipt_path(root: Path, index: int) -> Path:
    return root / RECEIPTS_DIR / f"{index:06d}.json"


def _contained(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def declared_inputs_digest(manifest: Any) -> str:
    """Pin on-disk declaration inputs, including trainer data and backend config.

    Base model and parent adapter trees have their own authoritative hashes
    through campaign identity preflight; hashing the full model a second time
    at each campaign boundary would be needlessly expensive.
    """
    pinned: dict[str, str] = {}
    for item in fields(manifest):
        name = item.name
        if not name.endswith("_path") or name in {"base_model_path", "parent_adapter_path"}:
            continue
        declared = getattr(manifest, name)
        if not declared:
            continue
        source = Path(str(declared))
        if not source.is_file():
            raise _refuse(f"declared input {name} is not a regular file")
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        pinned[name] = digest.hexdigest()
    return content_digest(pinned)


def validate_evidence(root: Path, evidence: Mapping[str, Any]) -> None:
    """Accept only attempt-owned, content-bound terminal evidence and priced spend."""
    name = evidence.get("attempt")
    if not isinstance(name, str) or not name.startswith("attempt-") or not name[8:].isdigit():
        raise _refuse("attempt has no canonical attempt-XX identity")
    attempt_dir = root / "attempts" / name
    if not attempt_dir.is_dir() or not _contained(attempt_dir, root):
        raise _refuse("attempt directory missing or outside campaign state")
    if Path(str(evidence.get("attempt_dir") or "")).resolve() != attempt_dir.resolve():
        raise _refuse("attempt directory identity disagrees")
    evidence_file = attempt_dir / EVIDENCE_FILE
    if Path(str(evidence.get("evidence_path") or "")).resolve() != evidence_file.resolve():
        raise _refuse("attempt evidence path disagrees")
    try:
        original = json.loads(evidence_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise _refuse("attempt evidence is missing, corrupt, or not yet durable") from exc
    # Campaign's backend wrapper adds its independently declared provenance
    # after the training binding saved its own raw evidence. It is bound by the
    # receipt, while all trainer-authored fields must match byte-for-byte.
    native = {key: value for key, value in evidence.items() if key != "backend"}
    if not isinstance(original, dict) or original != native:
        raise _refuse("attempt evidence differs from the trainer's durable original")
    if evidence.get("status") not in {"SUCCEEDED", "FAILED", "REFUSED"}:
        raise _refuse("attempt has no terminal status")
    if evidence.get("status") == STATUS_SUCCEEDED:
        source_identity = evidence.get("source_identity")
        if not isinstance(source_identity, dict) or source_identity.get("verified") is not True:
            raise _refuse("successful attempt lacks verified worker source identity")
        if evidence.get("attempt_row_status") not in {"passed", "failed", "rejected"}:
            raise _refuse("successful attempt has no terminal registry result")
        settlement = evidence.get("budget_settlement")
        if not isinstance(settlement, dict) or settlement.get("budget_compliant") is not True:
            raise _refuse("successful attempt has no budget-compliant settlement")
    cost = evidence.get("actual_cost")
    if not isinstance(cost, dict):
        raise _refuse("attempt has no measured or conservatively observed cost")
    try:
        actual = ComputeCost.from_dict(cost)
    except (TypeError, ValueError, KeyError) as exc:
        raise _refuse("attempt cost is invalid") from exc
    if not actual.incremental:
        raise _refuse("attempt's spent cost must be incremental")
    if evidence.get("train_started") is True and not actual.wall_gpu_hours and not actual.device_measured:
        raise _refuse("started training has an unverifiable zero cost")
    artifact_ref = evidence.get("artifact_ref")
    if evidence.get("status") == STATUS_SUCCEEDED and not artifact_ref:
        raise _refuse("successful attempt has no artifact")
    if artifact_ref:
        artifact = Path(str(artifact_ref))
        if not artifact.is_dir() or not _contained(artifact, attempt_dir):
            raise _refuse("attempt's artifact is missing or outside its directory")
        recorded = evidence.get("artifact_sha256")
        digest, files = directory_digest(artifact)
        if not isinstance(recorded, str) or not files or recorded != digest:
            raise _refuse("checkpoint/artifact SHA256 inventory differs from attempt receipt")
        if [dict(item) for item in files] != evidence.get("artifact_files"):
            raise _refuse("checkpoint/artifact file inventory differs from attempt receipt")


def commit_receipt(
    root: Path, *, index: int, spec_digest: str, evidence: Mapping[str, Any]
) -> str:
    validate_evidence(root, evidence)
    path = receipt_path(root, index)
    if not _contained(path.parent, root):
        raise _refuse("attempt receipt directory resolves outside campaign state")
    if path.exists():
        raise _refuse(f"attempt index {index} was already committed")
    payload = {
        "protocol": 1,
        "index": index,
        "spec_digest": spec_digest,
        "evidence": dict(evidence),
    }
    atomic_json(path, payload)
    return content_digest(payload)


def load_receipts(root: Path, *, manifest_digest: str) -> list[dict[str, Any]]:
    """Only fully paired, single-session journal starts and receipts may replay."""
    journal = root / "campaign-journal.jsonl"
    if not journal.is_file():
        return []
    try:
        text = journal.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise _refuse("unreadable campaign journal") from exc
    rows: list[dict[str, Any]] = []
    for line in text.splitlines(keepends=True):
        if not line.endswith("\n"):
            raise _refuse("torn journal line")
        try:
            row = json.loads(line)
        except (TypeError, ValueError) as exc:
            raise _refuse("malformed journal line") from exc
        if not isinstance(row, dict):
            raise _refuse("journal row is not a record")
        rows.append(row)
    # A prior terminal campaign may predate the interrupted one. Recovery
    # operates only on the most recent campaign_start, never borrowed receipts.
    starts = [i for i, row in enumerate(rows) if row.get("phase") == "campaign_start"]
    if not starts:
        raise _refuse("no campaign start marker")
    active = rows[starts[-1]:]
    if active[0].get("identity") != manifest_digest:
        raise _refuse("manifest declaration changed since interrupted start")
    started: dict[str, Any] | None = None
    receipts: list[dict[str, Any]] = []
    for row in active[1:]:
        phase = row.get("phase")
        if phase in {"candidate_evaluation_started", "terminal", "promotion_prepare"}:
            raise _refuse("evaluation, promotion, or terminal work already began")
        if phase == "attempt":
            if row.get("status") == "started":
                if started is not None:
                    raise _refuse("another attempt started before the prior one settled")
                if (
                    type(row.get("index")) is not int
                    or row["index"] != len(receipts)
                    or not isinstance(row.get("identity"), str)
                ):
                    raise _refuse("attempt start has no unique invocation identity")
                # Automatic replay is restricted to a native, priced receipt.
                # The pre-execution choice cannot be missing or downgraded to
                # a remote completion after GPU work has already happened.
                if row.get("proof_mode") != "receipt":
                    raise _refuse("attempt start has no immutable native receipt proof mode")
                started = row
            elif row.get("status") not in (None, "started"):
                # Legacy event or unbound completion is inadmissible.
                raise _refuse("unbound attempt completion event")
        elif phase == "attempt_receipt":
            index = len(receipts)
            if (
                started is None
                or type(row.get("index")) is not int
                or row["index"] != index
                or row.get("identity") != started.get("identity")
                or row.get("recipe_id") != started.get("recipe_id")
            ):
                raise _refuse("receipt does not match exactly one attempt start")
            path = receipt_path(root, index)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                raise _refuse("missing/corrupt receipt") from exc
            if (
                not isinstance(payload, dict)
                or payload.get("protocol") != 1
                or type(payload.get("index")) is not int
                or payload["index"] != index
                or payload.get("spec_digest") != started.get("identity")
                or content_digest(payload) != row.get("detail")
            ):
                raise _refuse("receipt bytes or invocation identity differ")
            evidence = payload.get("evidence")
            if (
                not isinstance(evidence, dict)
                or evidence.get("attempt") != row.get("attempt")
                or evidence.get("recipe_id") != row.get("recipe_id")
                or evidence.get("status") != row.get("status")
            ):
                raise _refuse("receipt attempt/recipe identity differs")
            validate_evidence(root, evidence)
            receipts.append(payload)
            started = None
    if started is not None:
        raise _refuse("a started attempt has no settled receipt; spend is unknown")
    if set((root / RECEIPTS_DIR).glob("*.json")) != {
        receipt_path(root, i) for i in range(len(receipts))
    }:
        raise _refuse("extra or missing attempt receipts")
    return receipts
