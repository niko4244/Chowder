"""Safely continue ACTUAL source-bound HumanEval Q4/Q8 research execution.

The original one-shot V38 runner deliberately refuses on infrastructure
failure. Its completed fsynced journal is a genuinely scored, immutable
prefix, NOT a failed or finished 164-suite. This separate runner audits every
existing row against the immutable original task/model/program sources and
the independent VM-host witnesses before appending the next case.

Never resumes both arms concurrently on this 4-CPU, 4-GiB Podman VM;
never starts another GPU model or modifies the frozen V2 Gate2 protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from contextlib import contextmanager
from collections.abc import Iterator

import psutil

from chowder_batch import reap_humaneval_v38_actual_execution_20261027 as v38
from chowder_batch import reap_v38_independent_receipt_audit_worker13_20261027 as verifier

AUTHORIZED_ROOTS = {
    "q4": Path("F:/reap-v2-human164-v38-private-q4-sealed-original-20261027-1325"),
    "q8": Path("F:/reap-v2-human164-v38-private-q8-sealed-original-20261027-1329"),
}
SCHEMA = "chowder.original-human164-v38-source-bound-resume.v1"


def _bytes(path: Path, limit: int) -> bytes:
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= limit:
        raise ValueError("missing, linked or oversized original research file")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if not 0 < len(data) <= limit:
        raise ValueError("original research file changed or exceeded bounded read")
    return data


def original_prefix(arm: str, root: Path) -> tuple[dict, bytes, bytes]:
    if arm not in AUTHORIZED_ROOTS or root != AUTHORIZED_ROOTS[arm]:
        raise ValueError("cannot use foreign or different-generation original research root")
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ValueError("original research root missing, linked or not absolute")
    journal = root / (arm + "-actual-independent-per-case.private.jsonl")
    manifest = root / "original-input-manifest.json"
    raw = _bytes(journal, verifier.FILE_SIZE_LIMIT)
    recorded_source = _bytes(manifest, 64 * 1024)
    verified = verifier.audit_bytes(arm, raw, saved_manifest=recorded_source)
    if (root / (arm + "-complete.private.json")).exists():
        raise ValueError("already marked complete; do not duplicate original score")
    if verified["verified_contiguous_rows"] >= v38.ORIGINAL_POPULATION:
        raise ValueError("full original 164 journal already exists without finalization")
    return verified, raw, recorded_source


def _assert_no_other_owners() -> None:
    """Refuse actual concurrent Q4/Q8 runs and orphaned active research pods."""
    own = os.getpid()
    for process in psutil.process_iter(("pid", "name", "cmdline")):
        try:
            args = process.info.get("cmdline") or []
            if process.info["pid"] == own:
                continue
            # Parent PowerShell commands may quote a whole invocation even
            # though no second Python evaluator exists.
            if (process.info.get("name") or "").lower() not in ("python.exe", "pythonw.exe", "py.exe"):
                continue
            line = " ".join(args).lower()
            if ("reap_humaneval_v38_actual_execution_20261027" in line
                    and "--research-execute" in line):
                raise RuntimeError("another original V38 HumanEval evaluator is active")
            if ("reap_humaneval_v38_guarded_resume_20261027" in line
                    and "--execute" in line):
                raise RuntimeError("another guarded HumanEval resume is active")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            raise RuntimeError("process ownership census unreadable") from None
    result = subprocess.run(["podman", "ps", "--format", "{{.Names}}"],
                            capture_output=True, text=True, timeout=20, check=False)
    if result.returncode:
        raise RuntimeError("cannot inspect active owned research containers")
    if any(re.fullmatch(r"w10-he-v38-[a-f0-9]{16}", x.strip())
           for x in result.stdout.splitlines()):
        raise RuntimeError("still-active original research candidate container")


@contextmanager
def _exclusive_execution_lease() -> Iterator[None]:
    """Cross-process Windows advisory lock, released automatically on crash."""
    if os.name != "nt":
        raise RuntimeError("original research resume is Windows-host only")
    import msvcrt

    lock = Path("F:/reap-v2-human164-v38-resume-serial-owner.private.lock")
    with lock.open("a+b") as stream:
        stream.seek(0)
        stream.write(b"R")
        stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError("a HumanEval research resume already holds the VM") from error
        try:
            _assert_no_other_owners()
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def continue_original(arm: str, root: Path, *, execute: bool, max_new: int = 164) -> dict:
    if type(max_new) is not int or not 0 <= max_new <= 164:
        raise ValueError("invalid bounded original number of new cases")
    verified, raw, original_manifest = original_prefix(arm, root)
    count = verified["verified_contiguous_rows"]
    if not execute:
        return {**{k: v for k, v in verified.items() if k != "rows"},
                "status": "READ_ONLY_VERIFIED_ORIGINAL_PREFIX_NO_INFERENCE"}
    # No generated program bytes or private original data can be opened for
    # execution until independently approved exact V38 loader/observer pins.
    v38.original_research_admission()
    with _exclusive_execution_lease():
        after, current, current_manifest = original_prefix(arm, root)
        if (current != raw or current_manifest != original_manifest
                or after["journal_snapshot_sha256"] != verified["journal_snapshot_sha256"]):
            raise RuntimeError("original source journal changed before exclusive resume")
        canonical = v38._read_canonical()
        source = v38._bound_source(v38.ARMS[arm].source, v38.ARMS[arm].source_sha256,
                                   64 * 1024 * 1024)
        source_rows = {v38._strict_json(row)["id"]: v38._strict_json(row)
                       for row in source.splitlines()}
        original = v38.original_program_manifest(arm)
        if original["ordered_program_manifest_sha256"] != verified["ordered_program_manifest_sha256"]:
            raise RuntimeError("original cohort/program identity changed at resume")
        destination = root / (arm + "-actual-independent-per-case.private.jsonl")
        correct = verified["verified_independent_correct_prefix"]
        executed = verified["verified_actual_executions"]
        until = min(164, count + max_new)
        for index in range(count, until):
            _assert_no_other_owners()
            task = canonical[index]
            row_source = source_rows[task["id"]]
            source_fingerprint = original["source_rows_in_original_order"][index]
            item = {
                "index": index, "id": task["id"], "arm": original["arm"],
                "source_session": original["original_serving_session"],
                "original_model_sha256": original["original_model_sha256"],
                "original_terminated": row_source["terminated"],
                "original_reported_correct_unverified": row_source["correct"],
                "program_sha256": source_fingerprint["candidate_program_sha256"],
                "frozen_v2_gate2_eligible": False,
            }
            if not row_source["terminated"]:
                item["execution_status"] = "SOURCE_BOUND_UNTERMINATED_NOT_EXECUTED"
                item["independent_research_correct"] = False
            else:
                program = v38.exact_program(task, row_source)
                if (hashlib.sha256(program.encode("utf-8")).hexdigest()
                        != source_fingerprint["candidate_program_sha256"]):
                    raise RuntimeError("original source candidate changed before native execution")
                # Native disposable Podman and outside host witness. Any
                # cleanup/attestation error ABORTS, never appends a fake score.
                witness = v38._run_research_candidate(program)
                item["execution_status"] = witness["execution_status"]
                item["independent_research_correct"] = witness["independent_research_correct"]
                item["live_external_vm_host_witness"] = witness
                executed += 1
            candidate_line = (json.dumps(item, sort_keys=True, allow_nan=False)
                              + "\n").encode("utf-8")
            # Independently inspect the *whole prior prefix + candidate row*
            # before any disk mutation. All source hashes and native witness
            # identities must be valid.
            verifier.audit_bytes(arm, raw + candidate_line, saved_manifest=original_manifest)
            with destination.open("ab") as stream:
                if stream.tell() != len(raw):
                    raise RuntimeError("journal modified concurrently before append")
                stream.write(candidate_line)
                stream.flush()
                os.fsync(stream.fileno())
            raw += candidate_line
            correct += int(item["independent_research_correct"])
            print(json.dumps({"event": "ORIGINAL_NATIVE_HUMANEVAL_SCORED",
                              "arm": arm, "ordinal": index,
                              "scored": index + 1, "actual_executed": executed,
                              "correct": correct, "journal_sha256": hashlib.sha256(raw).hexdigest()},
                             sort_keys=True), flush=True)
        latest = verifier.audit_bytes(arm, raw, saved_manifest=original_manifest)
        if latest["verified_contiguous_rows"] != 164:
            return {**{k: v for k, v in latest.items() if k != "rows"},
                    "status": "GENUINE_SOURCE_BOUND_PREFIX_RESUMABLE_NOT_COMPLETE"}
        summary = {
            "schema": "chowder.reap-v2-original-human164-independent-research.v38",
            "status": "COMPLETE_164_ORIGINAL_CODE_ISOLATED_RESEARCH_ONLY_NOT_FROZEN_GATE2",
            "arm": original["arm"], "original_model_sha256": original["original_model_sha256"],
            "original_rows_sha256": original["original_rows_raw_sha256"],
            "canonical_sha256": v38.CANONICAL_SHA256,
            "candidate_identity_manifest_sha256": original["ordered_program_manifest_sha256"],
            "original_rows": 164, "executed_rows": latest["verified_actual_executions"],
            "unterminated_rows_source_skipped": latest["verified_source_unterminated_nonpasses"],
            "independently_research_scored_correct": latest["verified_independent_correct_prefix"],
            "independent_score": latest["verified_independent_correct_prefix"] / 164,
            "per_case_evidence_sha256": hashlib.sha256(raw).hexdigest(),
            "original_v2_attestation_approved": False,
            "windows_native_gate_eligible": False,
            "gen0_status": "BLOCKED", "chosen": None, "release_approved": False,
        }
        with (root / (arm + "-complete.private.json")).open("xb") as stream:
            stream.write((json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        return summary


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--arm", choices=["q4", "q8"], required=True)
    cli.add_argument("--execute", action="store_true")
    cli.add_argument("--max-new", type=int, default=164)
    args = cli.parse_args()
    try:
        result = continue_original(args.arm, AUTHORIZED_ROOTS[args.arm],
                                   execute=args.execute, max_new=args.max_new)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"},
                         sort_keys=True), flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, TypeError, KeyError,
            subprocess.SubprocessError, psutil.Error) as error:
        print(json.dumps({"status": "REFUSED_OR_INTERRUPTED_NO_NEW_UNSCORED_CASE",
                          "arm": args.arm, "reason": str(error)[:800],
                          "frozen_gen0_status": "BLOCKED",
                          "release_approved": False}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
