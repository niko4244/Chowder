"""Read-only original ProgramBench candidate census; never executes generated code.

An archive plus self-written JSON is not independent original-model provenance.
The official evaluator must run only after separate cleanroom admission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chowder_batch.reap_original_programbench200_population_20261027 import audit
from chowder_batch.reap_original_programbench_runner_worker14_20261027 import (
    checked_archive, sha256, SOURCE_COMMIT, REGISTERED_Q4_GGUF_SHA256,
)


def inspect(official: Path, candidate_root: Path) -> dict:
    source = audit(official)
    if source["source_commit"] != SOURCE_COMMIT:
        raise ValueError("unregistered official ProgramBench release")
    if not candidate_root.is_absolute() or not candidate_root.is_dir() or candidate_root.is_symlink():
        raise ValueError("candidate directory must be an existing absolute physical directory")
    expected = {r["instance_id"]: r for r in source["source_bound_task_inventory"]}
    records = []
    for folder in sorted(candidate_root.iterdir(), key=lambda p: p.name):
        if folder.name not in expected:
            if folder.is_dir() or folder.is_symlink():
                raise ValueError("unregistered candidate task directory")
            continue
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("linked or invalid candidate task directory")
        archive = folder / "submission.tar.gz"
        if not archive.exists():
            continue
        identity = checked_archive(archive)
        receipt = folder / "inference_receipt.json"
        claims = None
        if receipt.exists():
            if receipt.is_symlink() or not receipt.is_file():
                raise ValueError("linked or invalid inference claim file")
            if receipt.stat().st_size > 2_000_000:
                raise ValueError("oversized inference claim file")
            obj = json.loads(receipt.read_bytes())
            if not isinstance(obj, dict):
                raise ValueError("inference claims must be a JSON object")
            claims = {
                "claim_file_sha256": sha256(receipt),
                "claimed_model_sha_matches_registered_q4":
                    obj.get("model_sha256") == REGISTERED_Q4_GGUF_SHA256,
                "claimed_archive_sha_matches_bytes":
                    obj.get("generated_archive_sha256") == identity["submission_sha256"],
            }
        records.append({"task": folder.name, "task_yaml_sha256": expected[folder.name]["original_task_yaml_sha256"],
                        "tests_json_sha256": expected[folder.name]["original_tests_json_sha256"],
                        **identity, "self_asserted_inference_claims": claims,
                        "independent_model_origin_verified": False,
                        "independent_cleanroom_admission_verified": False,
                        "original_official_score": None})
    return {"schema": "chowder.programbench200.original-submission-presence-audit.v1",
            "status": "UNSCORED_INDEPENDENT_ADMISSION_REQUIRED",
            "official_source_commit": SOURCE_COMMIT,
            "official_population_manifest_sha256": source["original_task_manifest_sha256"],
            "registered_population": 200, "registered_behavioral_tests": 248089,
            "candidate_archives_present": len(records), "candidates": records,
            "official_scored_rows": 0, "official_score": None,
            "independent_admission_verified": False, "frozen_gen0_status": "BLOCKED"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official-checkout", type=Path, required=True)
    p.add_argument("--candidate-root", type=Path, required=True)
    a = p.parse_args()
    try:
        print(json.dumps(inspect(a.official_checkout, a.candidate_root), sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "BLOCKED_INVALID_PROGRAMBENCH_EVIDENCE",
                          "reason": str(exc)[:260], "official_score": None}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
