"""CPU-only attestation gate before a PRIVATE original-Q8 AutomationBench run.

This module never downloads Kaggle output, installs wheels, launches a kernel,
loads a model, or executes generated code. It accepts only the actual original
private CPU result receipt, verifies source and official-public Q4 comparator
provenance, and reports distinct six/guard2 launch blockers. It intentionally
does not promote original Gen-0 or convert public tasks into official leaderboard
results. In particular a Kaggle COMPLETE status is not an admission receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from chowder_batch import reap_q8_automationbench_kaggle_20261026 as q8
from chowder_batch import reap_q8_automationbench_offline_wheelhouse_20261027 as wh


PRIVATE_ROOT = Path("F:/reap-v2-q8-automationbench-private-20261026")
Q4_PRIVATE_ROOT = Path("F:/reap-v2-automationbench-private-20261026")
CPU_KERNEL_SOURCE = PRIVATE_ROOT / "cpu-install-official600-20261027-v1" / "reap-q8-official600-locked-cpu-preflight.py"
CPU_KERNEL_SOURCE_SHA = "be47e025458178707df87b598944341ab71f5ea26cb6a5dc1086e00fb5e7d1e8"
SOURCE_TAR = PRIVATE_ROOT / "official-source-bin-private-dataset-v1" / "automationbench-v1.0.6-official.tar.bin"
WHEEL_TAR = PRIVATE_ROOT / "offline-wheelhouse-sourcebound-private-dataset-v2" / "automationbench-cp313-glibc235-original-108-locked-wheels.tar.bin"
WHEEL_TAR_SHA = "e7d1cc7946b17995eacc723f995702874e9a7b5fb7efe9471e5d2939bc1af6ab"
WHEEL_TAR_BYTES = 143_677_440
SOURCE_TAR_BYTES = 18_257_920
EXPECTED_SOURCE_SHA = q8.OFFICIAL_COMMIT
PRIVATE_WHEELS_DATASET = "nikmarco/reap128v2-automationbench-wheels-cp313-20261027"
PRIVATE_SOURCE_DATASET = "nikmarco/reap128v2-automationbench-sourcebin-20261027"
PRIVATE_Q8_DATASET = "nikmarco/reap128v2-q8-original-20261026"
EXPECTED_CPU_TARGET = PurePosixPath("/kaggle/working/reap-q8-official600-cpu-private-target")
EXPECTED_CPU_FILE_NAME = "reap-q8-original-automationbench-cpu600-private.json"
Q8_NOTEBOOKS = {
    "six": ("six-v2-sourcebin", "reap128v2-q8-automationbench-six-20261026.py",
            "887437535bea8bcaaa9dd08d3ce776b32a05bcb020b6f888a3082111ea90f1c2"),
    "guard2": ("guard2-v2-sourcebin", "reap128v2-q8-automationbench-guard2-20261026.py",
               "bad3fd79841084d6e8a79ca523affb74d72eaef737afa5f84fec30eca4d1e1cb"),
}


def _sha(path: Path, *, max_size: int) -> str:
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= max_size:
        raise ValueError("missing/symlinked/unbounded source-bound artifact: " + path.name)
    return wh.sha256_file(path)


def _private_receipt(receipt: Path, root: Path) -> dict[str, Any]:
    if not receipt.is_absolute() or receipt.name != EXPECTED_CPU_FILE_NAME:
        raise ValueError("receipt must be the authentic named private CPU output JSON")
    resolved, boundary = receipt.resolve(strict=False), root.resolve(strict=True)
    if boundary not in resolved.parents or not receipt.is_file() or receipt.is_symlink():
        raise ValueError("private receipt must be a real file within private evidence root")
    if not 0 < receipt.stat().st_size <= 1024 * 1024:
        raise ValueError("private CPU receipt missing/oversized")
    raw = receipt.read_bytes()
    record = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if type(record) is not dict:
        raise ValueError("CPU receipt structure invalid")
    return {"record": record, "sha256": hashlib.sha256(raw).hexdigest()}


def _validate_cpu_receipt(data: dict) -> dict:
    """Require an actual source-bound successful original CPU attempt, not a status string alone."""
    required = {
        "schema": "reap-q8-automationbench-real-cpu600-qual.v1",
        "status": "OFFICIAL_SOURCE_LOCKED_600_CPU_BOOTSTRAP_ADMITTED_NO_MODEL_SCORE",
        "gpu_enabled": False, "internet_enabled": False,
        "weights_loaded": False, "model_score_claimed": False,
        "gen0_status": "BLOCKED",
    }
    for key, expected in required.items():
        if type(data.get(key)) is not type(expected) or data[key] != expected:
            raise ValueError("source-locked CPU receipt unsuccessful or changed: " + key)
    if any(key in data for key in ("raw_binary_passed_count", "model_score", "official_private_score")):
        raise ValueError("CPU receipt cannot assert original model quality/score")
    pre = data.get("inspect")
    installed = data.get("install")
    if type(pre) is not dict or type(installed) is not dict:
        raise ValueError("CPU receipt has no completed wheel audit and import proof")
    if (pre.get("status") != "CPU_OFFLINE_SOURCES_AND_WHEELS_VERIFIED_NOT_INSTALLED"
            or installed.get("status") != "CPU_OFFLINE_600_SOURCE_LOADER_PASS_NOT_A_MODEL_SCORE"):
        raise ValueError("CPU dependency or original source loader did not PASS")
    for label, item in (("inspect", pre), ("install", installed)):
        for key, expected in (("package_count", 108),
                              ("locked_download_bytes", 143_537_351),
                              ("official_commit", EXPECTED_SOURCE_SHA),
                              ("gen0_status", "BLOCKED"),
                              ("weights_loaded", False), ("gpu_requested", False)):
            if type(item.get(key)) is not type(expected) or item[key] != expected:
                raise ValueError(f"{label} independent source/locked-wheel field mismatch: {key}")
    if installed.get("scorer_source_mutated") is not False or installed.get("model_inference_started") is not False:
        raise ValueError("CPU-only install admitted scorer mutation or model inference")
    if PurePosixPath(installed.get("site_packages", "")) != EXPECTED_CPU_TARGET / "packages":
        raise ValueError("offline packages were not isolated under intended Kaggle working target")
    if PurePosixPath(installed.get("official_source", "")) != EXPECTED_CPU_TARGET / "official-source":
        raise ValueError("CPU source unpacked outside isolated Kaggle working target")
    # Source .tar.bin and CPU bootstrap source are pinned separately. The CPU
    # proof is by the signed-local script's branch predicates, not direct row
    # transcript evidence (CPU job is not an inference benchmark).
    return {"status": "ORIGINAL_CPU_RECEIPT_SOURCE_LOCKED_PROVISIONAL_ADMISSION",
            "original_600_prompt_sha256": q8.PUBLIC_PROMPTS_SHA256,
            "original_600_source_verified_by_cpu_runner": True,
            "independent_600_model_score": None,
            "actual_gpu_evidence": False, "gen0_status": "BLOCKED"}


def audit_source_and_pilots(private_root: Path = PRIVATE_ROOT,
                            q4_root: Path = Q4_PRIVATE_ROOT) -> dict:
    if private_root != PRIVATE_ROOT or q4_root != Q4_PRIVATE_ROOT:
        raise ValueError("source identity roots are fixed and cannot be re-pointed")
    if (_sha(CPU_KERNEL_SOURCE, max_size=256 * 1024) != CPU_KERNEL_SOURCE_SHA
            or _sha(SOURCE_TAR, max_size=64 * 1024 * 1024) != wh.SOURCE_ARCHIVE_SHA256
            or SOURCE_TAR.stat().st_size != SOURCE_TAR_BYTES
            or _sha(WHEEL_TAR, max_size=180 * 1024 * 1024) != WHEEL_TAR_SHA
            or WHEEL_TAR.stat().st_size != WHEEL_TAR_BYTES):
        raise ValueError("original CPU/source/wheel archive hash or byte mismatch")
    if _sha(q8.OFFICIAL_SOURCE / "uv.lock", max_size=4 * 1024 * 1024) != wh.LOCK_SHA256:
        raise ValueError("official uv.lock changed")
    q8.verify_official_checkout(q8.OFFICIAL_SOURCE)
    profiles = {}
    for mode, (folder, filename, notebook_sha) in Q8_NOTEBOOKS.items():
        notebook = private_root / folder / filename
        if _sha(notebook, max_size=256 * 1024) != notebook_sha:
            raise ValueError("Q8 prepared notebook source changed: " + mode)
        metadata = json.loads((notebook.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
        if (metadata.get("is_private") is not True or metadata.get("enable_internet") is not False
                or metadata.get("enable_gpu") is not True
                or metadata.get("dataset_sources") != [PRIVATE_Q8_DATASET, PRIVATE_SOURCE_DATASET]):
            raise ValueError("Q8 notebook preregistered dataset or runtime metadata changed")
        comparator = q8.read_q4_evidence(mode, q4_root)
        profiles[mode] = {
            "q4_public_original_passes": 0,
            "exact_task_count": comparator["task_count"],
            "exact_task_names": comparator["ordered_task_names"],
            "q4_official_json_sha256": comparator["q4_json_sha256"],
            "q4_native_manifest_sha256": comparator["q4_native_manifest_sha256"],
            "q8_prepared_notebook_sha256": notebook_sha,
            "serving_flags": list(q8.PROFILES[mode]["flags"]),
            "status": "BLOCKED_ORIGINAL_Q8_NOTEBOOK_LACKS_OFFLINE_DEPENDENCY_DATASET_AND_ISOLATED_RUNTIME",
        }
    return {"source_tar_sha256": wh.SOURCE_ARCHIVE_SHA256, "wheel_tar_sha256": WHEEL_TAR_SHA,
            "original_cpu_kernel_source_sha256": CPU_KERNEL_SOURCE_SHA,
            "locked_uv_sha256": wh.LOCK_SHA256, "official_commit": EXPECTED_SOURCE_SHA,
            "matched_profiles": profiles, "gen0_status": "BLOCKED"}


def admission(receipt: Path | None = None) -> dict:
    source = audit_source_and_pilots()
    if receipt is None:
        return {**source, "status": "BLOCKED_ORIGINAL_CPU_RECEIPT_NOT_PROVIDED",
                "cpu_proof": None, "model_or_gpu_launched": False}
    record = _private_receipt(receipt, PRIVATE_ROOT)
    proof = _validate_cpu_receipt(record["record"])
    return {**source, "status": "CPU_ONLY_SOURCE_AND_DEPENDENCIES_ADMITTED_Q8_LAUNCH_STILL_BLOCKED",
            "cpu_receipt_sha256": record["sha256"], "cpu_proof": proof,
            "model_or_gpu_launched": False,
            "next_required_action": "PRIME_REGENERATE_TWO_SEPARATE_PRIVATE_Q8_KERNELS_WITH_EXACT_WHEEL_DATASET_AND_ISOLATED_RUNTIME"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-cpu-receipt", type=Path, help="Actual completed private Kaggle CPU receipt")
    args = parser.parse_args()
    print(json.dumps(admission(args.original_cpu_receipt), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
