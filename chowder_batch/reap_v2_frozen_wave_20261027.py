"""Frozen original Q4/Q8/dense/control GSM8K execution wave on a dedicated host.

Plan mode never loads weights or modifies files. An explicit `--execute --arm X`
request requires `--authorize-owned-run --receipt-dir` and an independently
qualified original Windows physical host. This routes through the native
CREATE_SUSPENDED+private Job Object owner supervisor, which authorizes ONLY the
child process it created and cancels only its Job on pressure or owner drift.
Its original offline source lock and fresh full-byte model hash must PASS first;
merely observing memory capacity is not authorization to run. Preserve native
model weights, benchmark files and services. The
existing `reap_serve.py`/`reap_eval.py` perform actual independent model load,
bound live serving receipts, original GSM8K inference and scoring. NEVER
execute HumanEval using reap_eval's unapproved local `python -I` sandbox.

Note: a Q4 dense / Q4 candidate comparison consists of distinct owned serving
sessions on the same admitted host/protocol, each with its own sealed first-200
receipt; it cannot literally use one resident model-serving process for both
different checkpoints. This code does not compute or approve a speed ratio.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

from chowder_batch import reap_eval, reap_serve, reap_select

REPO = Path(__file__).resolve().parent.parent
FLAGS = ("--fit", "on", "-np", "4", "-c", "16384", "--jinja")
PINNED = {
    "dense_v2_q4": "2c6d9466129e8802839566b35728e2506aaf1e8705cd137009e2f08feb9dd049",
    "reap128_v2_q4": "2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7",
    "dense_v2": "2ff57a74a36939c5baabf40a0d3eb06d4424ecc94cf7d4ef04b3ac02059097c2",
    "reap128_v2": "89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6",
    "random128_v2": "601871e00403658708eb4eebf3b05f9dc54a93b326e31f2f87b48cb59f92241b",
}
EXPECTED_BYTES = {"dense_v2_q4": 21_166_757_984,
                  "reap128_v2_q4": 11_563_915_264,
                  "dense_v2": 36_903_139_424,
                  "reap128_v2": 19_748_435_968,
                  "random128_v2": 19_748_435_872}
NATIVE_ADMISSION_ARM = {
    "dense_v2_q4": "dense_q4",
    "reap128_v2_q4": "q4",
    "dense_v2": "dense_q8",
    "reap128_v2": "q8",
    "random128_v2": "random128_q8",
}
BLOCKED_HUMANEVAL_EXECUTION = (
    "Frozen HumanEval model generation and executable original 164 needs an "
    "independently source-admitted disposable generated-code sandbox with "
    "witnessed end-to-end execution; reap_eval's local python -I is not it."
)


@dataclass(frozen=True)
class Study:
    arm: str
    suite: str
    limit: int
    model_sha256: str


STUDIES = tuple(Study(arm, "gsm8k", limit, sha) for arm, limit, sha in (
    ("dense_v2_q4", 200, PINNED["dense_v2_q4"]),
    ("reap128_v2_q4", 200, PINNED["reap128_v2_q4"]),
    ("dense_v2", 0, PINNED["dense_v2"]),
    ("reap128_v2", 0, PINNED["reap128_v2"]),
    ("random128_v2", 0, PINNED["random128_v2"]),
))


class AdmissionRefused(ValueError):
    """No attempt to start a new server under this admission."""


def study_for(arm: str) -> Study:
    found = next((study for study in STUDIES if study.arm == arm), None)
    if found is None:
        raise ValueError("arm is not in registered safe frozen original GSM8K wave")
    return found


def assert_frozen_code_identity() -> None:
    if (tuple(reap_serve.FLAGS) != FLAGS
            or tuple(reap_select.FROZEN_SERVE_FLAGS) != FLAGS
            or reap_eval.FROZEN_V2_QUALITY_SOURCES["gsm8k"] != (
                1319, "913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89"
            )
            or reap_eval.MAX_TOKENS["gsm8k"] != 1024
            or reap_eval.SYSTEM != "You are a helpful assistant."
            or any(reap_serve.REGISTERED_QUALITY_REFERENCE_SHA256.get(arm) != PINNED[arm]
                   for arm in ("dense_v2_q4", "dense_v2", "random128_v2"))
            or reap_serve.REGISTERED_Q8_GGUF_SHA256 != PINNED["reap128_v2"]
            or reap_serve.REGISTERED_Q4_GGUF_SHA256 != PINNED["reap128_v2_q4"]):
        raise AdmissionRefused("original frozen serve/model/source/decode code identity drift")


def validate_fresh_output(output: Path, arm: str) -> None:
    study_for(arm)
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("dedicated run output is linked or not a directory")
    if output.exists() and (output / arm).exists():
        # The original serving wrapper supports safe resumption of other suites;
        # this orchestrator does not adopt any historical sessions or rows.
        raise ValueError("existing arm output: use another exclusive wave directory")


def command(study: Study, *, model: Path, output: Path, server: Path, port: int) -> list[str]:
    if study != study_for(study.arm) or not 1024 <= port <= 65535:
        raise ValueError("unregistered study identity or invalid loopback port")
    return [sys.executable, str(REPO / "chowder_batch" / "reap_serve.py"),
            "--gguf", str(model), "--arm", study.arm, "--port", str(port),
            "--out", str(output), "--server", str(server),
            "--", "--port", str(port), "--arm", study.arm, "--out", str(output),
            "--suites", "gsm8k", "--parallel", "4", "--limit", str(study.limit)]


def plan(model_for, *, output: Path, server: Path, port: int) -> dict:
    assert_frozen_code_identity()
    entries = []
    for study in STUDIES:
        model = Path(model_for(study.arm))
        if not model.is_file() or model.is_symlink():
            status = "BLOCKED_MISSING_MODEL"
        elif study.arm in EXPECTED_BYTES and model.stat().st_size != EXPECTED_BYTES[study.arm]:
            status = "BLOCKED_MODEL_SIZE_MISMATCH"
        elif (output / study.arm).exists():
            status = "BLOCKED_PREEXISTING_SESSION"
        else:
            status = "PREPARED_NOT_EXECUTED_REQUIRES_FRESH_ADMISSION"
        entries.append({"arm": study.arm, "suite": study.suite,
                        "item_count": study.limit or 1319,
                        "original_gguf_sha256": study.model_sha256,
                        "model_path": str(model), "status": status,
                        "command_when_admitted": command(
                            study, model=model, output=output, server=server, port=port,
                        )})
    return {"schema": "chowder.reap-v2-frozen-dedicated-wave.v1",
            "worktree": str(REPO), "frozen_flags": list(FLAGS),
            "source_gsm8k_sha256": reap_eval.FROZEN_V2_QUALITY_SOURCES["gsm8k"][1],
            "model_hash_validation": "performed at execution by reap_serve; plan is not proof",
            "model_execution_occurred": False,
            "original_q4_first200": "SEPARATE_MODEL_SESSIONS_SAME_ADMITTED_HOST_REQUIRED",
            "humaneval": "BLOCKED_REQUIRES_ORIGINAL_APPROVED_SANDBOX",
            "human_eval_execution_blocker": BLOCKED_HUMANEVAL_EXECUTION,
            "studies": entries,
            "gen0": {"status": "BLOCKED", "chosen": None, "release_approved": False}}


def admit_for_execution(*, study: Study, model: Path, server: Path, port: int) -> tuple[bool, list[str]]:
    """Two-stage native check; READY is necessary, not an approval to release."""
    from chowder_batch import reap_v2_dedicated_runner_admission_20261027 as native

    if not model.is_file() or model.is_symlink() or not server.is_file() or server.is_symlink():
        return False, ["ORIGINAL_MODEL_OR_NATIVE_SERVER_MISSING"]
    native_arm = NATIVE_ADMISSION_ARM[study.arm]
    snap = native.snapshot(model_path=model, server_path=server,
                           port=port, arm=native_arm)
    # A source-locked dedicated-runner probe cannot be replaced by claimed
    # capacity from another machine or a static JSON manifest.
    decision = native.assess(snap, native_arm, port=port)
    if decision.get("status") != "READY_FOR_CAPACITY_PROBE_ONLY":
        return False, [str(v) for v in decision.get("reasons", ["NATIVE_ADMISSION_REFUSED"])]
    # The capacity-only checker is expressly not sufficient to authorize a
    # new model load. Direct invocation of this wrapper is still blocked until
    # the native launch-owner API has independently established exclusivity.
    return False, ["NO_INDEPENDENT_NATIVE_LAUNCH_OWNER_ADMISSION"]


def execute(study: Study, *, model: Path, output: Path, server: Path, port: int,
            receipt_dir: Path | None = None, execute_authorized: bool = False) -> dict:
    if execute_authorized is not True or receipt_dir is None:
        raise AdmissionRefused("EXPLICIT_OWNED_RUN_AND_NEW_RECEIPT_DIRECTORY_REQUIRED")
    validate_fresh_output(output, study.arm)
    assert_frozen_code_identity()
    # `python -m ...` executes this file as __main__; the owner supervisor
    # imports its canonical chowder_batch module, producing a different Python
    # Study dataclass *type* despite the same four frozen fields. Rebind only
    # after exact value-wise source/model/policy validation. Without this,
    # every correctly authorized native CLI run fails FROZEN_STUDY_DRIFT before
    # its real RAM/GPU admission is even inspected.
    from chowder_batch import reap_v2_frozen_wave_20261027 as registered_wave

    registered = registered_wave.study_for(study.arm)
    if (study.arm, study.suite, study.limit, study.model_sha256) != (
        registered.arm, registered.suite, registered.limit, registered.model_sha256
    ):
        raise AdmissionRefused("REGISTERED_FROZEN_STUDY_IDENTITY_CHANGED")
    # Deliberately no direct subprocess or launching based on a caller-supplied
    # admission JSON. The supervisor rereads Windows physical capacity, ports,
    # GPUs, original model bytes and canonical offline sources before creating
    # its own private, kill-on-close Job, including the serving wrapper itself.
    from chowder_batch.reap_v2_owner_bound_runtime_supervisor_20261027 import run_frozen_arm

    return run_frozen_arm(
        study=registered, model=model, server=server, output=output,
        receipt_dir=receipt_dir, port=port, execute_authorized=True,
    )


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--arm", help="One original registered arm; required for --execute")
    cli.add_argument("--model", type=Path, help="Original model GGUF (execute only)")
    cli.add_argument("--models-dir", type=Path, default=Path("C:/dedicated-reap/models"))
    cli.add_argument("--server", type=Path, default=Path(reap_serve.DEFAULT_SERVER))
    cli.add_argument("--out", type=Path, default=Path("C:/dedicated-reap/eval-wave-v1"))
    cli.add_argument("--port", type=int, default=8790)
    cli.add_argument("--execute", action="store_true")
    cli.add_argument("--authorize-owned-run", action="store_true",
                     help="Explicitly authorize only a fresh supervised child on admitted hardware")
    cli.add_argument("--receipt-dir", type=Path,
                     help="Existing empty exclusive receipt folder on admitted dedicated host")
    cli.add_argument("--snapshot", type=Path, help="New, exclusive JSON plan path")
    args = cli.parse_args()
    try:
        if args.execute:
            if args.arm is None or args.model is None:
                raise AdmissionRefused("--execute requires one --arm and --model")
            result = execute(study_for(args.arm), model=args.model, output=args.out,
                             server=args.server, port=args.port,
                             receipt_dir=args.receipt_dir,
                             execute_authorized=args.authorize_owned_run)
        else:
            result = plan(lambda arm: args.models_dir / (arm + ".gguf"),
                          output=args.out, server=args.server, port=args.port)
        if args.snapshot:
            if args.snapshot.is_symlink() or args.snapshot.exists():
                raise ValueError("refusing to overwrite any previous plan")
            with args.snapshot.open("x", encoding="utf-8") as file:
                json.dump(result, file, sort_keys=True, indent=2, allow_nan=False)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0 if not args.execute or result.get("status") == "CHILD_EXIT_ZERO_RUNTIME_WITNESSED_ONLY" else 2
    except (AdmissionRefused, OSError, ValueError, RuntimeError, TypeError) as exc:
        print(json.dumps({"status": "REFUSED_NOT_ADMITTED", "reason": str(exc),
                          "model_executed": False, "gen0": {
                              "status": "BLOCKED", "chosen": None, "release_approved": False}}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
