"""Read-only, source-bound LiveCodeBench v6 transport recovery inventory.

This is *not* a generation driver or a benchmark scorer. It cannot authorize
cross-session pooling: a replacement serving session must be independently
admitted and its results evaluated as a separate research cohort. In
particular, an HTTP error is never a wrong model answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from chowder_batch import reap_v2_lcb_v6_append_only_witness_20261028 as witness
from chowder_batch import reap_v2_lcb_v6_checkpoint_readonly_audit_20261028 as lcb
from chowder_batch import reap_v2_lcb_v6_transport_integrity_20261028 as transport

SCHEMA = "chowder.reap-v2-lcb-v6-recovery-inventory.v1"


def build_inventory(source: list[tuple[str, str]], rows: list[dict], *,
                    journal_sha256: str) -> dict:
    """Partition canonical IDs into responded, transport-failed, and untouched.

    Every observed row must match the original source order and frozen prompt
    protocol. A model response with empty extracted code is still a response,
    never a transport failure. No candidate code is copied into the inventory.
    """
    if (not isinstance(journal_sha256, str) or len(journal_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in journal_sha256)):
        raise ValueError("original journal SHA-256 required")
    if (not source or any(type(qid) is not str or not qid or
                          type(filename) is not str or not filename
                          for qid, filename in source)
            or len({qid for qid, _ in source}) != len(source)):
        raise ValueError("canonical source IDs missing or duplicated")
    classification = transport.classify(source, rows)
    responded = []
    failed = []
    untouched = []
    for index, (qid, filename) in enumerate(source):
        item = {"index": index, "question_id": qid, "source_file": filename}
        if index >= len(rows):
            untouched.append(item)
        elif rows[index]["error"] is None:
            responded.append(item)
        else:
            failed.append(item)
    if (len(responded) != classification["successful_transport_rows"]
            or len(failed) != classification["failed_transport_rows_not_scored"]
            or len(failed) + len(untouched) != classification["unresolved_source_cases"]):
        raise ValueError("independent transport partition disagrees")
    return {
        "schema": SCHEMA,
        "status": "RESEARCH_RECOVERY_INVENTORY_ONLY",
        "canonical_population": len(source),
        "journal_sha256": journal_sha256,
        "journal_rows": len(rows),
        "responded_original_session_unattested": responded,
        "failed_transport_not_model_answers": failed,
        "never_attempted": untouched,
        "unresolved_count": len(failed) + len(untouched),
        "source_order_preserved": True,
        "requires_new_independently_admitted_model_session": True,
        "cross_session_pooling_allowed": False,
        "official_score": None,
        "candidate_execution_authorized": False,
        "frozen_gen0": {"status": "BLOCKED", "chosen": None,
                        "release_approved": False},
    }


def inspect() -> dict:
    # This audits the frozen original Git source, six dataset hashes, original
    # generator/evaluator source and independent partial-checkpoint consistency.
    provenance = transport.inspect()
    source = [
        (row["question_id"], name)
        for name in lcb.FILES
        for row in lcb.read_jsonl(lcb.DATA / name)
    ]
    path = lcb.OUT / "gen_q4.jsonl"
    if path.is_symlink() or not path.is_file():
        raise ValueError("original generation journal is not a regular file")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != provenance["generation_journal_sha256"]:
        raise ValueError("journal changed between independent source audits")
    result = build_inventory(source, transport.parse_complete_journal(raw),
                             journal_sha256=digest)
    if (result["unresolved_count"] != provenance["unresolved_source_cases"]
            or result["journal_rows"] != provenance["journal_rows"]
            or lcb.sha(path) != digest):
        raise ValueError("journal changed while inventory was constructed")
    return {
        **result,
        "source_git_commit": provenance["source_git_commit"],
        "source_dataset_sha256": provenance["source_dataset_sha256"],
        "generator_source_sha256": provenance["generator_source_sha256"],
        "evaluator_source_sha256": provenance["evaluator_source_sha256"],
        "checkpoint_sha256": provenance["official_checkpoint_sha256"],
        "checkpoint_consistency_only": provenance[
            "previously_scored_prefix_internal_consistency_only"],
        "cpu_code_isolation_independently_attested": False,
        "research_only": True,
    }


def _verify_complete_journal(journal: Path, expected_sha256: str) -> tuple[int, int]:
    """Hash the original regular file through one stable handle and identity."""
    before = witness._regular_unlinked(journal)
    digest = hashlib.sha256()
    with journal.open("rb") as stream:
        if not witness._same_identity(before, os.fstat(stream.fileno())):
            raise ValueError("original journal replaced before seal read")
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        if not witness._same_identity(before, os.fstat(stream.fileno())):
            raise ValueError("original journal replaced during seal read")
    after = witness._regular_unlinked(journal)
    if not witness._same_identity(before, after):
        raise ValueError("original journal replaced during seal verification")
    if digest.hexdigest() != expected_sha256:
        raise ValueError("original journal changed before evidence seal")
    return before.st_dev, before.st_ino


def _seal_exclusive_inventory(target: Path, journal: Path, result: dict) -> None:
    """Atomically publish only a reverified full-journal inventory, never overwrite."""
    if (result.get("schema") != SCHEMA or result.get("official_score") is not None
            or result.get("cross_session_pooling_allowed") is not False
            or result.get("frozen_gen0") != {"status": "BLOCKED", "chosen": None,
                                               "release_approved": False}):
        raise ValueError("inventory is not a fail-closed recovery certificate")
    original_identity = _verify_complete_journal(journal, result["journal_sha256"])
    pending = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.pending-",
            delete=False,
        ) as stream:
            pending = Path(stream.name)
            stream.write((json.dumps(result, sort_keys=True, indent=2,
                                     allow_nan=False) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        if _verify_complete_journal(journal, result["journal_sha256"]) != original_identity:
            raise ValueError("original journal identity changed before evidence seal")
        # A hard link publishes atomically and fails if the destination exists.
        os.link(pending, target)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    try:
        result = inspect()
        if args.snapshot is not None:
            target = args.snapshot
            if (not target.is_absolute() or target.exists() or target.is_symlink()
                    or target.parent.resolve() != Path("F:/").resolve()):
                raise ValueError("new exclusive F: snapshot path required")
            _seal_exclusive_inventory(target, lcb.OUT / "gen_q4.jsonl", result)
        print(json.dumps({key: value for key, value in result.items()
                          if key not in ("responded_original_session_unattested",
                                         "failed_transport_not_model_answers",
                                         "never_attempted")}, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "REFUSED_LCB_RECOVERY_INVENTORY",
                          "reason": str(exc)[:350], "frozen_gen0_status": "BLOCKED"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
