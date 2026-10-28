"""Source-bound, append-only witness of the *live* original LCB v6 journal.

Unlike a final journal certificate, this proves only the immutable prefix
observed twice. A concurrently appended partial JSON row is excluded and
explicitly counted. Never execute candidate code, restart a model, rewrite
the original journal, combine serving sessions, or issue an official score.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

from chowder_batch import reap_v2_lcb_v6_checkpoint_readonly_audit_20261028 as lcb
from chowder_batch import reap_v2_lcb_v6_transport_integrity_20261028 as transport

SCHEMA = "chowder.reap-v2-lcb-v6-live-append-only-prefix-witness.v1"
MAX_INFLIGHT_TAIL = 1024 * 1024


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _regular_unlinked(path: Path) -> os.stat_result:
    """Reject Windows reparse points, symlinks and nonregular evidence paths."""
    for candidate in (path, path.parent):
        info = candidate.lstat()
        if (getattr(info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
            raise ValueError(f"reparse point in evidence path: {candidate}")
        if candidate.is_symlink():
            raise ValueError(f"symlink in evidence path: {candidate}")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("evidence journal is not a regular file")
    return info


def _same_identity(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def complete_prefix(raw: bytes) -> tuple[bytes, int]:
    """Keep only newline-terminated rows, never parse a torn append."""
    if not raw:
        raise ValueError("empty generation journal")
    end = raw.rfind(b"\n")
    if end < 0:
        raise ValueError("no complete generation row")
    prefix = raw[:end + 1]
    tail_bytes = len(raw) - len(prefix)
    if tail_bytes > MAX_INFLIGHT_TAIL:
        raise ValueError("excessive in-flight generation tail")
    return prefix, tail_bytes


def verify_append_only(prefix: bytes, subsequent: bytes) -> int:
    """Prove a second observation retained the exact original byte prefix."""
    if not prefix or not prefix.endswith(b"\n"):
        raise ValueError("witness requires a complete original prefix")
    if len(subsequent) < len(prefix) or subsequent[:len(prefix)] != prefix:
        raise ValueError("original generation journal truncated or rewritten")
    return len(subsequent) - len(prefix)


def source_provenance() -> tuple[list[tuple[str, str]], dict]:
    """Independently validate frozen Git, all six datasets and both drivers."""
    head = subprocess.check_output(
        ["git", "-C", str(lcb.SOURCE), "rev-parse", "HEAD"], timeout=15,
    ).decode("utf-8").strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(lcb.SOURCE), "status", "--porcelain"], timeout=15,
    )
    if head != lcb.SOURCE_COMMIT or dirty.strip():
        raise ValueError("original LiveCodeBench Git source identity drift")
    if lcb.sha(lcb.GENERATOR) != lcb.GENERATOR_SHA:
        raise ValueError("original generation driver SHA drift")
    if lcb.sha(lcb.EVALUATOR) != lcb.EVALUATOR_SHA:
        raise ValueError("original evaluator SHA drift")
    runtime_count = 0
    for file in (lcb.SOURCE / "lcb_runner").rglob("*.py"):
        if lcb.sha(file) != lcb.sha(lcb.RUNTIME / file.relative_to(lcb.SOURCE)):
            raise ValueError("official evaluator runtime SHA drift")
        runtime_count += 1
    if runtime_count == 0:
        raise ValueError("missing original evaluator runtime")
    canonical = []
    hashes = {}
    for filename, expected in zip(lcb.FILES, lcb.DATA_SHAS, strict=True):
        path = lcb.DATA / filename
        digest = lcb.sha(path)
        if digest != expected:
            raise ValueError(f"original dataset SHA drift: {filename}")
        hashes[filename] = digest
        canonical.extend(
            (row["question_id"], filename) for row in lcb.read_jsonl(path)
        )
    if len(canonical) != 1055 or len({item[0] for item in canonical}) != 1055:
        raise ValueError("original canonical task population mismatch")
    return canonical, {
        "source_git_commit": head,
        "source_dataset_sha256": hashes,
        "generator_source_sha256": lcb.GENERATOR_SHA,
        "evaluator_source_sha256": lcb.EVALUATOR_SHA,
        "runtime_python_files_cross_hashed": runtime_count,
        "partial_checkpoint_sha256_not_a_score": lcb.sha(
            lcb.OUT / "metrics_q4_checkpoint.json"
        ),
    }


def build_witness(
    canonical: list[tuple[str, str]], initial: bytes, subsequent: bytes,
    *, provenance: dict,
) -> dict:
    """Pure, adversarially testable witness without candidate-code execution."""
    prefix, inflight = complete_prefix(initial)
    growth = verify_append_only(prefix, subsequent)
    rows = transport.parse_complete_journal(prefix)
    classified = transport.classify(canonical, rows)
    if classified["successful_transport_rows"] + classified[
        "failed_transport_rows_not_scored"
    ] != len(rows):
        raise ValueError("original transport partition is inconsistent")
    return {
        "schema": SCHEMA,
        "status": "LIVE_APPEND_ONLY_PREFIX_WITNESSED_RESEARCH_ONLY",
        "prefix_sha256": _sha_bytes(prefix),
        "prefix_bytes": len(prefix),
        "prefix_rows": len(rows),
        "initial_observed_bytes": len(initial),
        "initial_inflight_tail_bytes_excluded": inflight,
        "subsequent_observed_bytes": len(subsequent),
        "subsequent_bytes_beyond_prefix": growth,
        "exact_prefix_reverified": True,
        "source_provenance": provenance,
        "transport_classification": classified,
        "candidate_text_or_code_in_witness": False,
        "original_serving_session_independently_attested": False,
        "cpu_code_isolation_independently_attested": False,
        "cross_session_pooling_allowed": False,
        "official_score": None,
        "frozen_gen0": {
            "status": "BLOCKED", "chosen": None, "release_approved": False,
        },
    }


def inspect() -> dict:
    path = lcb.OUT / "gen_q4.jsonl"
    first_info = _regular_unlinked(path)
    initial = path.read_bytes()
    second_info = _regular_unlinked(path)
    if not _same_identity(first_info, second_info):
        raise ValueError("generation journal replaced during initial read")
    canonical, provenance = source_provenance()
    third_info = _regular_unlinked(path)
    subsequent = path.read_bytes()
    fourth_info = _regular_unlinked(path)
    if (not _same_identity(first_info, third_info)
            or not _same_identity(first_info, fourth_info)):
        raise ValueError("generation journal replaced during witness")
    result = build_witness(
        canonical, initial, subsequent, provenance=provenance,
    )
    result["journal_file_identity"] = {
        "device": first_info.st_dev,
        "inode": first_info.st_ino,
    }
    return result


def _verify_prefix_still_present(path: Path, witness: dict) -> None:
    info = _regular_unlinked(path)
    identity = witness["journal_file_identity"]
    if (info.st_dev, info.st_ino) != (
        identity["device"], identity["inode"]
    ):
        raise ValueError("original journal file identity changed before seal")
    with path.open("rb") as stream:
        if not _same_identity(info, os.fstat(stream.fileno())):
            raise ValueError("original journal replaced during seal read")
        prefix = stream.read(witness["prefix_bytes"])
        if not _same_identity(info, os.fstat(stream.fileno())):
            raise ValueError("original journal replaced during seal read")
    if _sha_bytes(prefix) != witness["prefix_sha256"]:
        raise ValueError("original journal prefix changed before evidence seal")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    try:
        witness = inspect()
        witness["observed_utc"] = datetime.now(timezone.utc).isoformat()
        if args.snapshot is not None:
            target = args.snapshot
            if (not target.is_absolute() or target.exists() or target.is_symlink()
                    or target.parent.resolve() != Path("F:/").resolve()):
                raise ValueError("new exclusive F: witness path required")
            _verify_prefix_still_present(lcb.OUT / "gen_q4.jsonl", witness)
            with target.open("xb") as stream:
                stream.write(
                    (json.dumps(witness, sort_keys=True, indent=2) + "\n").encode()
                )
            _verify_prefix_still_present(lcb.OUT / "gen_q4.jsonl", witness)
        print(json.dumps(witness, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError,
            subprocess.SubprocessError) as exc:
        print(json.dumps({
            "status": "REFUSED_LIVE_LCB_PREFIX_WITNESS",
            "reason": str(exc)[:350],
            "frozen_gen0_status": "BLOCKED",
        }))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
