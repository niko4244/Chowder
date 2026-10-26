"""Independent original-Q8 HumanEval replay inside disposable rootless Podman.

No model inference, host execution of generated code, GPU, network or host
mounts. The EXACT prior private source-bound Q8 machine/audit/preflight must
match independently witnessed SHA-256 identities; other arms fail closed.
Only the 162 terminated original Q8 rows are executed; original unclosed
generations are deterministic non-passes without executing their text.
Research only: this does not certify Windows, native CUDA or Gen-0 selection.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from chowder_batch.reap_external_full_audit import (
    EVALUATOR_SHA, Q8_SHA, SCORER_SHA, _strict_json,
)
from chowder_batch.reap_humaneval_podman_probe import (
    IMAGE_SHA256, command, validate,
)

CANONICAL_SHA = "5144d31cf337b25ce262b780ed103b128b1e9d289d3c0d15f4fe34933bb705d5"
MACHINE_SHA = "92ca9969ef5959714f4787ed10a4375c69c2c4a80a4842daabc01c2360543fc0"
TRUSTED_AUDIT_SHA = "107395f480693922bd279a50138338db14c78441320a7060914f0d82cdd73683"
PREFLIGHT_SHA = "516feda89b1b7dd25aedaed9098d0c4965f0eac858f2ddd7a1841a940718732e"
PROBE_SHA = "3321c43285dc9711e09538f334692a5e9482ab5c83d0e12319df4fdfa6b1a0fb"
ORIGINAL_ROW_SHA = "cd3c9d1577611a032b36e69a3f96bd272d118819074352503f143bcd68cac301"
SESSION = "95876063255e45159f57e29306e0e885"
# The successful fixed-source 768-MiB census is not sufficient proof of
# descendant teardown, pinned original Python 3.13 runtime equivalence, or a
# trusted v2 external receipt producer. No command-line flag may override this
# source-owned research qualification gate while those checks remain open.
EXECUTABLE_REPLAY_QUALIFIED = False
# Fixed BENIGN Python 3.13.17 rootless OCI profile was observed on the VM.
# This does not replace the separately pinned historical 3.11-slim harmless
# probe or make the current replay-profile command equivalent to the original
# Kaggle Python 3.13 scorer. Its use requires NEW trusted admission evidence.
PY313_BENIGN_IMAGE_SHA256 = "72036a705a7cff38924efd674e60654666cb283faf7b3fad160b34007f7395bd"
INDEPENDENT_VM_PID_TREE_EXTINCTION_VERIFIED = False
ORIGINAL_10_SECOND_TIMEOUT_EQUIVALENCE_VERIFIED = False
ORIGINAL_CANDIDATE_FILE_AND_CWD_EQUIVALENCE_VERIFIED = False
TRUSTED_PER_TASK_REPLAY_V2_ATTESTATION_VERIFIED = False


def execution_qualification_gaps() -> list[str]:
    """Exact fail-closed gates; the harmless census cannot unlock execution."""
    gaps = []
    if IMAGE_SHA256 != PY313_BENIGN_IMAGE_SHA256:
        gaps.append("PINNED_PY313_REPLAY_IMAGE_REQUIRED")
    if not INDEPENDENT_VM_PID_TREE_EXTINCTION_VERIFIED:
        gaps.append("INDEPENDENT_VM_PID_TREE_EXTINCTION_REQUIRED")
    if not ORIGINAL_10_SECOND_TIMEOUT_EQUIVALENCE_VERIFIED:
        gaps.append("ORIGINAL_10_SECOND_PYTHON_TIMEOUT_EQUIVALENCE_REQUIRED")
    if not ORIGINAL_CANDIDATE_FILE_AND_CWD_EQUIVALENCE_VERIFIED:
        gaps.append("ORIGINAL_CANDIDATE_FILE_AND_CWD_EQUIVALENCE_REQUIRED")
    if not TRUSTED_PER_TASK_REPLAY_V2_ATTESTATION_VERIFIED:
        gaps.append("TRUSTED_PER_TASK_REPLAY_V2_ATTESTATION_REQUIRED")
    if not EXECUTABLE_REPLAY_QUALIFIED:
        gaps.append("GENERATED_EXECUTION_SOURCE_DISABLED")
    return gaps


def require_execution_qualification() -> None:
    gaps = execution_qualification_gaps()
    if gaps:
        raise RuntimeError(
            "generated HumanEval execution disabled: " + ", ".join(gaps)
        )


def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def read_bound(path: Path, expected_sha: str, limit: int) -> bytes:
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or not 0 < path.stat().st_size <= limit):
        raise ValueError("private replay original source not an ordinary bounded file")
    raw = path.read_bytes()
    if digest(raw) != expected_sha:
        raise ValueError("independently trusted original source SHA mismatch")
    return raw


def exact_program(item: dict, row: dict) -> str:
    """Mirror original immutable scorer's HumanEval code composition, no exec."""
    text = row["text"]
    if not isinstance(text, str) or len(text) > 1024 * 1024:
        raise ValueError("original generated code absent or over size bound")
    reply = text.rsplit("</think>", 1)[1] if "</think>" in text else (
        "" if "<think>" in text else text
    )
    blocks = re.findall(r"\x60\x60\x60(?:python)?\n(.*?)\x60\x60\x60", reply, flags=re.S)
    code = blocks[-1] if blocks else reply
    entry = item["entry_point"]
    if f"def {entry}" not in code:
        code = item["header"] + code
    program = code + "\n\n" + item["test"] + f"\n\ncheck({entry})\n"
    if not 0 < len(program.encode("utf8")) <= 2 * 1024 * 1024:
        raise ValueError("rendered original HumanEval candidate exceeds bound")
    return program


def host_safety_guard() -> None:
    """CPU-only container work must not eat protected Windows RAM/commit."""
    if os.name != "nt":
        raise ValueError("original private Podman supervisor requires authorized Windows host")

    class Memory(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    memory = Memory()
    memory.dwLength = ctypes.sizeof(Memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise RuntimeError("original Windows physical/commit memory census failed")
    if memory.ullAvailPhys < 7 * 1024 ** 3 or memory.ullAvailPageFile < 8 * 1024 ** 3:
        raise RuntimeError("protected Windows 7-GiB physical or 8-GiB commit floor breached")


def replay_profile_command(name: str) -> list[str]:
    """Build the exact fixed 768-MiB replay profile; no shell or host mounts."""
    if (not isinstance(name, str) or not re.fullmatch(
        r"reap-q8-independent-human-[0-9a-f]{8}-(?:0|[1-9][0-9]{0,2})", name
    ) or int(name.rsplit("-", 1)[-1]) >= 164):
        raise ValueError("refusing foreign or malformed private container name")
    args = command(IMAGE_SHA256)
    args[args.index("--memory=268435456")] = "--memory=805306368"
    args[args.index("--memory-swap=268435456")] = "--memory-swap=805306368"
    args[args.index("--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=8m")] = (
        "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=64m"
    )
    # The container runtime's own watchdog, PID isolation and cgroup memory
    # bound remain active even if the host CLI loses its connection.
    args[2:2] = ["--name=" + name, "--timeout=14", "--init",
                 "--ulimit=cpu=10:11", "--ulimit=fsize=1048576:1048576",
                 "--ulimit=nofile=64:64"]
    return args


def podman_exec(program: str, name: str) -> tuple[bool, str, int | None, float]:
    """Never execute generated code on host, only passed through container stdin."""
    args = replay_profile_command(name)
    host_safety_guard()
    # The cleanup path is allowed to address only the unique *new* item owner.
    # Never kill/force-remove an existing container merely because it shares
    # an attacker-provided or stale name with this replay.
    initial = subprocess.run(
        ["podman", "container", "exists", name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=10, check=False,
    )
    if initial.returncode != 1:
        raise RuntimeError("isolated replay container name already owned or unverifiable")
    started = time.monotonic()
    try:
        result = subprocess.run(args, input=program, text=True,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=23,
                                check=False)
        container_state = subprocess.run(
            ["podman", "container", "exists", name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=10, check=False,
        )
        if container_state.returncode != 1:
            raise RuntimeError("completed isolated item still has a live or unverifiable native owner")
        host_safety_guard()
        # Podman reserves 125/126/127 for its CLI, invocation and executable
        # failures. 137/143 and out-of-range statuses can mean resource
        # enforcement or an unlaunched/dead container rather than a completed
        # Python assertion. Counting those as normal false test outcomes could
        # accidentally "match" the originally failed HumanEval tasks and mint
        # an independent 164/164 replay without having executed them.
        # Fail the WHOLE scientific replay instead of treating them as misses.
        code = result.returncode
        if (type(code) is not int or code < 0 or code > 124):
            raise RuntimeError(
                "Podman infrastructure/runtime exit cannot attest executed HumanEval code"
            )
        return result.returncode == 0, ("EXIT_0" if result.returncode == 0 else
                                         "NONZERO_EXIT"), result.returncode, (
                                             time.monotonic() - started)
    except subprocess.TimeoutExpired:
        # A timed-out CLI is not evidence the container has stopped. Explicitly
        # stop and remove the unique owned container, never another workload.
        for subcmd in (["podman", "kill", name], ["podman", "rm", "--force", name]):
            try:
                subprocess.run(subcmd, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=10, check=False)
            except subprocess.SubprocessError as exc:
                raise RuntimeError("could not securely reap timed-out private container") from exc
        # Neither 'kill' nor 'rm' returning is evidence that the child PID
        # namespace and descendants vanished. A Podman error (125), or a
        # lingering container, must never be recorded as safely terminated.
        final = subprocess.run(
            ["podman", "container", "exists", name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=10, check=False,
        )
        if final.returncode != 1:
            raise RuntimeError("timed-out private container still exists or owner unverified")
        host_safety_guard()
        return False, "WALL_TIMEOUT_CONTAINER_KILLED", None, time.monotonic() - started


def check_source_binding(paths: argparse.Namespace) -> tuple[list[dict], list[dict], dict]:
    from chowder_batch.reap_humaneval_private_preflight import preflight

    machine_blob = read_bound(paths.machine, MACHINE_SHA, 2 * 1024 * 1024)
    audit_blob = read_bound(paths.trusted_audit, TRUSTED_AUDIT_SHA, 2 * 1024 * 1024)
    preflight_blob = read_bound(paths.preflight, PREFLIGHT_SHA, 2 * 1024 * 1024)
    source_blob = read_bound(paths.canonical, CANONICAL_SHA, 2 * 1024 * 1024)
    raw_blob = read_bound(paths.rows, ORIGINAL_ROW_SHA, 64 * 1024 * 1024)
    read_bound(paths.original_evaluator, EVALUATOR_SHA, 2 * 1024 * 1024)
    read_bound(paths.original_scorer, SCORER_SHA, 2 * 1024 * 1024)
    probe_blob = read_bound(paths.probe, PROBE_SHA, 1024 * 1024)
    machine = _strict_json(machine_blob)
    original_audit = _strict_json(audit_blob)
    witness = _strict_json(preflight_blob)
    probe = _strict_json(probe_blob)
    if (not isinstance(probe, dict)
            or probe.get("status") != "CAPABILITY_PROBE_ONLY_NOT_HUMANEVAL_APPROVAL"
            or probe.get("image_id") != IMAGE_SHA256
            or probe.get("generated_programs_executed") != 0):
        raise ValueError("independent benign rootless sandbox admission missing")
    validate(probe["census"])
    actual = preflight(
        arm="reap128_v2_q8", machine=machine, machine_sha=MACHINE_SHA,
        trusted_audit=original_audit, audit_sha=TRUSTED_AUDIT_SHA,
        canonical_raw=source_blob, rows_raw=raw_blob,
        evaluator_sha=EVALUATOR_SHA, scorer_sha=SCORER_SHA,
    )
    if actual != witness or witness.get("sandbox_verified") is not False:
        raise ValueError("non-executing original preflight cannot be verified exactly")
    canonical = _strict_json(source_blob)
    rows = [_strict_json(line) for line in raw_blob.splitlines()]
    if (not isinstance(canonical, list) or len(canonical) != 164
            or len(rows) != 164
            or any(not isinstance(row, dict) for row in rows)):
        raise ValueError("full source-bound canonical task list malformed")
    return canonical, rows, witness


def replay(paths: argparse.Namespace) -> dict:
    require_execution_qualification()
    host_safety_guard()
    canonical, rows, preflight = check_source_binding(paths)
    by_id = {row["id"]: row for row in rows}
    results = []
    start = time.monotonic()
    for index, original in enumerate(canonical):
        row = by_id[original["id"]]
        result = {"id": original["id"], "program_sha256": None,
                  "original_correct": row["correct"],
                  "terminated": row["terminated"]}
        if row["terminated"]:
            program = exact_program(original, row)
            result["program_sha256"] = digest(program.encode())
            name = "reap-q8-independent-human-" + SESSION[:8] + "-" + str(index)
            passed, reason, code, elapsed = podman_exec(program, name)
            result.update({"independent_correct": passed, "reason": reason,
                           "exit_code": code, "elapsed_seconds": round(elapsed, 3),
                           "execution_mode": "ROOTLESS_PODMAN_ISOLATED"})
        else:
            result.update({"independent_correct": False, "reason": "SKIPPED_UNTERMINATED",
                           "exit_code": None, "elapsed_seconds": 0.0,
                           "execution_mode": "NOT_EXECUTED"})
        results.append(result)
        host_safety_guard()
        if (index + 1) % 25 == 0:
            print(f"SOURCE_BOUND_PRIVATE_PODMAN_PROGRESS {index + 1}/164", flush=True)
    matched = sum(item["original_correct"] == item["independent_correct"] for item in results)
    correct = sum(item["independent_correct"] for item in results)
    executed = sum(item["execution_mode"] == "ROOTLESS_PODMAN_ISOLATED" for item in results)
    return {
        "schema": "chowder.reap-v2-private-rootless-oci-human164-replay.v1",
        "status": ("PASS_SOURCE_BOUND_164_ROOTLESS_OCI_REEXECUTION_PROVISIONAL_LINUX"
                   if matched == 164 and correct == 152 and executed == 162 else
                   "REPLAY_DISAGREEMENT_UNQUALIFIED"),
        "arm": "reap128_v2_q8", "model_sha256": Q8_SHA,
        "session_id": SESSION, "original_machine_receipt_sha256": MACHINE_SHA,
        "original_independent_full_audit_sha256": TRUSTED_AUDIT_SHA,
        "original_preexecution_preflight_sha256": PREFLIGHT_SHA,
        "original_humaneval_row_sha256": ORIGINAL_ROW_SHA,
        "canonical_humaneval_raw_sha256": CANONICAL_SHA,
        "original_evaluator_sha256": EVALUATOR_SHA,
        "original_scorer_sha256": SCORER_SHA,
        "actual_rootless_probe_sha256": PROBE_SHA,
        "python_oci_image_id": IMAGE_SHA256,
        "source_rows": 164, "actually_executed_programs": executed,
        "skipped_original_unterminated": 164 - executed,
        "correct": correct, "matched_recorded_scores": matched,
        "wall_seconds": round(time.monotonic() - start, 3),
        "per_task": results,
        "gen0_status": "BLOCKED", "chosen": None,
        "original_windows_gate_eligible": False, "release_approved": False,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute-qualified", action="store_true",
                   help="Explicitly admit isolated execution after source/probe preflight")
    for key in ("machine", "trusted_audit", "preflight", "canonical", "rows",
                "original_evaluator", "original_scorer", "probe", "out"):
        p.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    args = p.parse_args(argv)
    try:
        if args.out.exists() or args.out.is_symlink() or not args.out.parent.is_dir():
            raise ValueError("original private replay output must be exclusive")
        if not args.execute_qualified:
            check_source_binding(args)
            print(json.dumps({"status": "SOURCE_ONLY_PREEXECUTION_ADMISSION",
                              "generated_programs_executed": 0,
                              "private_output_written": False,
                              "execution_qualification_gaps": execution_qualification_gaps(),
                              "gen0_status": "BLOCKED"}))
            return 0
        result = replay(args)
        # Even a correctness disagreement is substantive private scientific
        # evidence, not a claim of pass. Refuse overwriting previous receipts.
        with args.out.open("x", encoding="utf8", newline="\n") as fp:
            json.dump(result, fp, indent=2, sort_keys=True, allow_nan=False)
            fp.write("\n")
        print(json.dumps({"status": result["status"], "correct": result["correct"],
                          "matched_recorded_scores": result["matched_recorded_scores"],
                          "actually_executed_programs": result["actually_executed_programs"]}))
        return 0 if result["status"].startswith("PASS_") else 2
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError,
            RuntimeError) as exc:
        print("Independent Q8 OCI replay REFUSED: " + str(exc)[:200], file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
