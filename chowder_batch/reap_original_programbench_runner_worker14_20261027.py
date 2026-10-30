"""Source-bound staging for real ProgramBench cleanroom model submissions.

This module never manufactures a solution or benchmark score. Inference must run
against the official task_cleanroom image without internet; official ProgramBench
alone grades the resulting complete program source archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from chowder_batch.reap_original_programbench200_population_20261027 import audit

REGISTERED_Q4_GGUF_SHA256 = "2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7"
SOURCE_COMMIT = "1fe64c87a1318998636850cead00b0db80e2d928"


def task_prompt_identity(source_commit: str, population_sha: str, task: dict) -> str:
    """Hash the source-bound task contract; no unverified free-form prompt identity."""
    bound = {"protocol": "programbench.task-prompt-source-binding.v1",
             "source_commit": source_commit, "population_sha256": population_sha,
             "instance_id": task["instance_id"],
             "task_yaml_sha256": task["original_task_yaml_sha256"],
             "tests_json_sha256": task["original_tests_json_sha256"],
             "repository": task["original_repository"],
             "repository_commit": task["original_commit"]}
    return hashlib.sha256(json.dumps(bound, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_archive(path: Path) -> dict:
    """Refuse archive traversal, links, device nodes and ambiguous file names."""
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 256_000_000:
        raise ValueError("missing, linked or oversized generated source archive")
    entries = set()
    source_files = 0
    total_size = 0
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            name = member.name.replace("\\", "/")
            parts = name.split("/")
            if (not name or name.startswith("/") or "\\" in member.name or
                    any(p in ("", ".", "..") for p in parts) or
                    ":" in name or name in entries or
                    not (member.isfile() or member.isdir())):
                raise ValueError("unsafe generated source archive entry")
            entries.add(name)
            if member.isfile():
                source_files += 1
                total_size += member.size
                if total_size > 512_000_000:
                    raise ValueError("generated source archive expanded size exceeds limit")
    if "compile.sh" not in entries or source_files < 2:
        raise ValueError("model generated program missing root compile.sh or source")
    return {"submission_sha256": sha256(path), "archive_files": source_files,
            "archive_expanded_bytes": total_size}


def stage(official: Path, task: str, source_archive: Path, out: Path,
          inference_receipt: Path) -> dict:
    original = audit(official)
    if original["source_commit"] != SOURCE_COMMIT:
        raise ValueError("official source commit mismatch")
    tasks = {r["instance_id"]: r for r in original["source_bound_task_inventory"]}
    if task not in tasks:
        raise ValueError("not one of the original 200 benchmark tasks")
    if not inference_receipt.is_file() or inference_receipt.is_symlink():
        raise ValueError("missing Q4 candidate-generation claims")
    receipt = json.loads(inference_receipt.read_bytes())
    mandatory = ("model_sha256", "quantization", "prompt_sha256", "response_sha256",
                 "task_cleanroom_image_digest", "network_isolation_verified",
                 "generated_archive_sha256", "source_commit", "population_sha256",
                 "instance_id", "task_yaml_sha256", "tests_json_sha256")
    if (not isinstance(receipt, dict) or any(not receipt.get(k) for k in mandatory)
            or receipt["quantization"] != "Q4_K_M"
            or any(not isinstance(receipt[k], str) or len(receipt[k]) != 64
                   or any(c not in "0123456789abcdef" for c in receipt[k])
                   for k in ("model_sha256", "prompt_sha256", "response_sha256",
                             "generated_archive_sha256"))):
        raise ValueError("incomplete or malformed Q4 candidate-generation claims")
    selected = tasks[task]
    if receipt["model_sha256"] != REGISTERED_Q4_GGUF_SHA256:
        raise ValueError("unregistered original Q4 GGUF SHA256")
    bindings = {"source_commit": SOURCE_COMMIT,
                "population_sha256": original["original_task_manifest_sha256"],
                "instance_id": task,
                "task_yaml_sha256": selected["original_task_yaml_sha256"],
                "tests_json_sha256": selected["original_tests_json_sha256"],
                "prompt_sha256": task_prompt_identity(SOURCE_COMMIT,
                       original["original_task_manifest_sha256"], selected)}
    if any(receipt[key] != expected for key, expected in bindings.items()):
        raise ValueError("candidate prompt or immutable official task/source identity mismatch")
    archive_info = checked_archive(source_archive)
    if receipt["generated_archive_sha256"] != archive_info["submission_sha256"]:
        raise ValueError("model-generated source archive does not match inference receipt")
    if out.exists() or out.is_symlink() or not out.parent.is_dir() or not out.is_absolute():
        raise ValueError("exclusive absolute evidence output directory required")
    work = out / task
    work.mkdir(parents=True)
    shutil.copyfile(source_archive, work / "submission.tar.gz")
    shutil.copyfile(inference_receipt, work / "inference_receipt.json")
    result = {
        "schema": "chowder.original-programbench-q4-unverified-candidate-stage.v2",
        "status": "UNVERIFIED_Q4_CANDIDATE_STAGED_NOT_ADMITTED_NOT_SCORED",
        "source_commit": original["source_commit"],
        "official_population_manifest_sha256": original["original_task_manifest_sha256"],
        "task": task, "task_source": selected, "source_bound_prompt_identity_sha256": bindings["prompt_sha256"],
        "registered_q4_gguf_sha256": REGISTERED_Q4_GGUF_SHA256,
        "candidate_claims_sha256": sha256(inference_receipt),
        "model_origin_independently_verified": False,
        "cleanroom_independently_admitted": False,
        "independent_external_witness_verified": False,
        "self_asserted_network_isolation": receipt["network_isolation_verified"],
        "self_asserted_image_digest": receipt["task_cleanroom_image_digest"],
        "independent_admission_required": True,
        **archive_info, "official_evaluator_command": ["programbench", "eval", str(out),
                                                   "--filter", task, "--workers", "1",
                                                   "--branch-workers", "1", "--docker-cpus", "1"],
        "official_score": None, "scored": False, "frozen_gen0_status": "BLOCKED",
    }
    (work / "stage_receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--generated-archive", type=Path, required=True)
    parser.add_argument("--inference-receipt", type=Path, required=True)
    parser.add_argument("--private-out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = stage(args.official_checkout, args.task, args.generated_archive,
                       args.private_out, args.inference_receipt)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, tarfile.TarError, subprocess.SubprocessError,
            json.JSONDecodeError, KeyError, TypeError) as err:
        print(json.dumps({"status": "BLOCKED_ORIGINAL_PROGRAMBENCH_Q4_SUBMISSION",
                          "reason": f"{type(err).__name__}: {str(err)[:300]}",
                          "official_score": None, "frozen_gen0_status": "BLOCKED"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
