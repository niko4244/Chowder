"""Source-locked Q8 missing-29 HLE pre-load clean-cache admission experiment.

CPU-only preparation. Never launches Kaggle, changes weights/serving flags,
relaxes a resource floor, or promotes the non-official HLE heuristic to a score.
The previous run failed *during* native load at the unchanged 10-GiB cgroup
floor, before the existing successful post-load GGUF cache advisory could run.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

from chowder_batch import reap_q8_hle_missing29_resume_20261027 as original


BASE_SCRIPT_SHA256 = "741564b6717ce33036ed8bf6623e434324c37f0343af346dd310c5a4202c37bc"
SCRIPT_NAME = "reap128v2-q8-hle-missing29-private-20261027.py"
PATCH_ID = "source-pinned-preload-clean-gguf-cache-advisory-v1"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("pinned original missing29 source anchor absent or ambiguous")
    return source.replace(old, new)


def patch_script(raw: bytes) -> bytes:
    """Add an advisory *before* any measured floor violation, never after one."""
    if sha(raw) != BASE_SCRIPT_SHA256:
        raise ValueError("original missing29 generated kernel SHA changed")
    source = raw.decode("utf-8")
    if ("if h['cgroup_unused_gib'] is None or h['cgroup_unused_gib'] < 10:" not in source
            or "if R['after_postload_reclaim']['cgroup_unused_gib'] < 14:" not in source
            or "if 'offloaded 41/41 layers to GPU' not in native_start:" not in source
            or "Q8_GGUF_BYTES = 19748435968" not in source):
        raise ValueError("original model/resource/offload admission not preserved")
    source = replace_once(
        source, "    'samples': [],",
        "    'samples': [],\n"
        "    'during_load_clean_reclaims': [],\n"
        f"    'preload_advisory_version': {PATCH_ID!r},",
    )
    source = replace_once(
        source,
        "        while time.monotonic() - boot < 600:\n"
        "            h = host()\n"
        "            check_runtime(h)\n",
        "        while time.monotonic() - boot < 600:\n"
        "            h = host()\n"
        "            # Advisory on the already SHA-verified original read-only GGUF.\n"
        "            # Never attempt to rescue a measured breach of the 10-GiB floor.\n"
        "            if (h['cgroup_unused_gib'] is not None and\n"
        "                    10 <= h['cgroup_unused_gib'] < 11.5):\n"
        "                check_runtime(h)\n"
        "                if (not hasattr(os, 'posix_fadvise') or\n"
        "                        not hasattr(os, 'POSIX_FADV_DONTNEED') or\n"
        "                        weights.is_symlink() or not weights.is_file() or\n"
        "                        weights.stat().st_size != Q8_GGUF_BYTES):\n"
        "                    raise RuntimeError('pinned GGUF clean-cache advisory unavailable')\n"
        "                before = h\n"
        "                with weights.open('rb', buffering=0) as r_handle:\n"
        "                    os.posix_fadvise(r_handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)\n"
        "                time.sleep(1)\n"
        "                h = host()\n"
        "                R['during_load_clean_reclaims'].append({'before': before, 'after': h})\n"
        "            check_runtime(h)\n",
    )
    ast.parse(source)
    compile(source, SCRIPT_NAME, "exec")
    return source.encode("utf-8")


def prepare(out: Path) -> dict:
    """Create a NEW exclusive private kernel; no submission or model execution."""
    if not out.is_absolute() or out.exists() or out.is_symlink():
        raise ValueError("exclusive absolute output directory required")
    source_receipt = original.prepare(out)
    script = out / SCRIPT_NAME
    patched = patch_script(script.read_bytes())
    script.write_bytes(patched)
    metadata = json.loads((out / "kernel-metadata.json").read_bytes())
    if (metadata.get("id") != original.KAGGLE_ID
            or metadata.get("is_private") is not True
            or metadata.get("enable_gpu") is not True
            or metadata.get("enable_internet") is not False
            or metadata.get("machine_shape") != "NvidiaTeslaT4"):
        raise ValueError("source-locked private Kaggle metadata changed")
    return {
        "status": "PREPARED_ONLY_UNSUBMITTED_NONOFFICIAL_HLE_MISSING29",
        "original_generated_script_sha256": BASE_SCRIPT_SHA256,
        "new_generated_script_sha256": sha(patched),
        "advisory_version": PATCH_ID,
        "original_missing29_source_id_sha256": source_receipt["missing29_id_set_sha256"],
        "unchanged_original_q8_model_sha256": source_receipt["q8_original_model_sha256"],
        "original_serving_flags": source_receipt["native_flags"],
        "unchanged_runtime_cgroup_floor_gib": 10,
        "unchanged_postload_cgroup_floor_gib": 14,
        "official_hle_judge_score": "UNAVAILABLE",
        "scientific_selection": "BLOCKED", "chosen": None,
        "release_approved": False,
        "output_directory": str(out),
    }


def audit(fresh_dir: Path) -> dict:
    """Require measured post-advisory floors before original two-session audit."""
    receipt = original._object(fresh_dir / original.NEW_RECEIPT)
    if receipt.get("preload_advisory_version") != PATCH_ID:
        raise ValueError("fresh HLE receipt not from source-locked advisory profile")
    events = receipt.get("during_load_clean_reclaims")
    if not isinstance(events, list) or len(events) > 600:
        raise ValueError("unbounded or missing model-loading cache witness")
    for event in events:
        if not isinstance(event, dict) or set(event) != {"before", "after"}:
            raise ValueError("invalid clean-cache admission event")
        for sample in (event["before"], event["after"]):
            if (not isinstance(sample, dict)
                    or type(sample.get("cgroup_unused_gib")) not in (int, float)
                    or not math.isfinite(sample["cgroup_unused_gib"])
                    or sample["cgroup_unused_gib"] < 10
                    or type(sample.get("physical_free_gib")) not in (int, float)
                    or not math.isfinite(sample["physical_free_gib"])
                    or sample["physical_free_gib"] < 7
                    or not isinstance(sample.get("gpu_free_mib"), list)
                    or len(sample["gpu_free_mib"]) != 2
                    or any(type(x) is not int or x < 4096
                           for x in sample["gpu_free_mib"])):
                raise ValueError("resource floor breach during model-loading advisory")
    result = original.audit_two_sessions(fresh_dir)
    result["source_locked_preload_advisory"] = PATCH_ID
    result["during_load_clean_reclaim_events"] = len(events)
    return result
