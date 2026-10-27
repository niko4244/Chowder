"""Source-locked Q8 HLE missing29 cgroup abort and safer next-host census.

No Kaggle calls, process launches, model weights, generated code, scorer or
source mutation. The completed *kernel* was a refused model load, not a 29-row
success. This prevents repeating a known-ineligible original host before an
independently larger runner has actual resources. A prestart PASS alone never
proves postload memory, 41/41 CUDA offload or HLE score.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from chowder_batch import reap_q8_hle_missing29_resume_20261027 as source

ROOT = Path(r"F:\reap-v2-q8-hle-missing29-output-20261027-feature-first-complete-recovery")
RECEIPT_SHA = "2489554a5ade3cbfec74001d5d1fcf8b7be6028bf283b564fcdf0ecb5a27a99d"
LOG_SHA = "2d92b64bf690815514fbb12dbcef49c41a31a256d1a24381d42270b84e64b09f"
ERROR = "Q8 conservative 10 GiB cgroup headroom floor breached"
MIN_CGROUP_UNUSED_GIB = 48.0  # Prospective, more conservative than failed 28.806.
MIN_HOST_PHYSICAL_AVAILABLE_GIB = 48.0
MIN_GPU_FREE_MIB = 12_288
MIN_DISK_FREE_GIB = 8.0


def failed_original_evidence(root: Path = ROOT) -> dict:
    """Verify exact private failure bytes. Never promote a kernel COMPLETE status."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("source-bound original Q8 failed output directory absent")
    raw = source._read(root / source.NEW_RECEIPT, expected=RECEIPT_SHA)
    report = source._json_bytes(raw)
    log = source._read(root / source.NEW_LOG, expected=LOG_SHA)
    if not isinstance(report, dict):
        raise ValueError("source-bound failed research receipt is not an object")
    initial = report.get("initial", {})
    expected_cmd = [
        "--fit", "on", "-np", "2", "-c", "16384", "--jinja",
        "--port", "8793", "-m",
        "/kaggle/input/datasets/nikmarco/reap128v2-q8-original-20261026/"
        "reap128_v2-q8_0-20261025.gguf",
    ]
    cmd = report.get("cmd")
    if (report.get("schema") != "reap-v2-q8-matched128-text-only-hle-research.v1"
            or report.get("status") != "REFUSED_OR_ABORTED"
            or report.get("error") != ERROR
            or report.get("scientific_selection") != "BLOCKED"
            or report.get("chosen", "missing") is not None
            or report.get("release_approved") is not False
            or report.get("amendment4_gate_eligible") is not False
            or report.get("official_hle_score_claimed") is not False
            or report.get("samples") != []
            or report.get("resume_generator_template_sha256") != source.V3_SCRIPT_SHA
            or report.get("resume_prior_v4_rows_sha256") != source.V4_ROWS_SHA
            or report.get("resume_prior_v4_session_id") != source.V4_SESSION
            or report.get("resume_missing29_ids_sha256") != source.MISSING_IDS_SHA
            or report.get("resume_q4_original_rows_sha256") != source.Q4_SESSION_SHA
            or report.get("protocol_sha256") != source.PROTOCOL_SHA
            or report.get("protocol_file_sha256") != source.PROTOCOL_FILE_SHA
            or report.get("data", {}).get("q8_gguf_sha256") != source.frozen.Q8_SHA256
            or report.get("data", {}).get("server_binary_sha256") != source.SERVER_SHA
            or report.get("server_log_sha256") != LOG_SHA
            or report.get("server_log_bytes") != len(log)
            or type(cmd) is not list or len(cmd) != 12
            or cmd[1:] != expected_cmd
            or not isinstance(cmd[0], str) or not cmd[0].endswith("/llama-server")
            or b"load_tensors: loading model tensors" not in log
            or b"41/41 layers" in log
            or not isinstance(initial, dict)):
        raise ValueError("original aborted Q8 HLE source/protocol/model log mismatch")
    for field, expected in (("cgroup_unused_gib", 28.806),
                            ("physical_free_gib", 29.77)):
        if type(initial.get(field)) not in (float, int) or initial[field] != expected:
            raise ValueError("original failed Q8 HLE runtime measurement changed")
    if initial.get("gpu_free_mib") != [14_912, 14_912]:
        raise ValueError("original Q8 HLE GPU startup census changed")
    if list(root.glob(source.NEW_PATTERN)):
        raise ValueError("aborted new session unexpectedly contains new scored rows")
    return {
        "status": "ORIGINAL_COMPLETE_KERNEL_WITH_ZERO_HLE_ROWS_CGROUP_ABORT",
        "original_prior_q8_rows": 99,
        "new_q8_valid_rows": 0,
        "q8_combined_original_valid_rows": 99,
        "error": ERROR,
        "initial_cgroup_unused_gib": initial["cgroup_unused_gib"],
        "initial_physical_free_gib": initial["physical_free_gib"],
        "initial_gpu_free_mib": initial["gpu_free_mib"],
        "observed_consumption_gib_lower_bound_before_completed_load":
            initial["cgroup_unused_gib"] - 10.0,
        "model_load_completed": False,
        "new_native_complete_session": False,
        "original_failed_receipt_sha256": RECEIPT_SHA,
        "original_native_log_sha256": LOG_SHA,
        "official_hle_judge": "UNAVAILABLE",
        "scientific_selection": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }


def admit_measured_snapshot(snapshot: dict) -> dict:
    """Prospective strict host floor; caller snapshots are *not* a trust root."""
    if not isinstance(snapshot, dict):
        raise ValueError("missing host measurement")
    scalars = ("cgroup_unused_gib", "host_physical_available_gib", "disk_free_gib")
    if any(type(snapshot.get(key)) not in (float, int)
           or not math.isfinite(snapshot[key]) or snapshot[key] < 0
           for key in scalars):
        raise ValueError("unavailable or nonfinite native host census")
    gpu = snapshot.get("gpu_free_mib")
    if (not isinstance(gpu, list) or len(gpu) != 2
            or any(type(x) is not int or x < 0 for x in gpu)):
        raise ValueError("two physical GPU free-memory measurements required")
    gates = {
        "cgroup_unused_48_gib": snapshot["cgroup_unused_gib"] >= MIN_CGROUP_UNUSED_GIB,
        "host_physical_available_48_gib":
            snapshot["host_physical_available_gib"] >= MIN_HOST_PHYSICAL_AVAILABLE_GIB,
        "both_gpu_free_12288_mib": all(x >= MIN_GPU_FREE_MIB for x in gpu),
        "local_disk_free_8_gib": snapshot["disk_free_gib"] >= MIN_DISK_FREE_GIB,
    }
    return {
        "status": ("MEASURED_PRESTART_FLOORS_SATISFIED_NOT_MODEL_ADMITTED"
                   if all(gates.values()) else "REFUSED_KNOWN_INSUFFICIENT_HLE_RUNNER"),
        "prospective_gates": gates,
        "snapshot": snapshot,
        "full_model_postload_resource_pass": False,
        "forty_one_cuda_layers_proven": False,
        "hles_scored": 0,
        "new_job_submitted": False,
        "gen0_status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }


def measure_original_linux_host() -> dict:
    """Linux runner-native read-only sensors. Not usable for Windows host attest."""
    if sys.platform != "linux":
        raise ValueError("native Linux container resource census required")
    cgroup = Path("/sys/fs/cgroup")
    limit = (cgroup / "memory.max").read_text().strip()
    used = (cgroup / "memory.current").read_text().strip()
    if not limit.isdecimal() or not used.isdecimal() or int(used) > int(limit):
        raise ValueError("finite Linux cgroup-v2 bound and charge required")
    available = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) == 3 and fields[2] == "kB":
                available = int(fields[1]) * 1024
    if available is None:
        raise ValueError("real physical Linux MemAvailable missing")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    if result.returncode:
        raise ValueError("real native two-GPU memory census unavailable")
    lines = [line.strip() for line in result.stdout.splitlines()]
    if len(lines) != 2 or not all(x.isdecimal() for x in lines):
        raise ValueError("two real GPU free-memory counts required")
    return {
        "cgroup_unused_gib": (int(limit) - int(used)) / 1024 ** 3,
        "host_physical_available_gib": available / 1024 ** 3,
        "gpu_free_mib": [int(x) for x in lines],
        "disk_free_gib": shutil.disk_usage("/kaggle/working").free / 1024 ** 3,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-real-linux-host", action="store_true")
    args = parser.parse_args(argv)
    try:
        evidence = failed_original_evidence()
        print(json.dumps(evidence, sort_keys=True))
        if args.probe_real_linux_host:
            actual = admit_measured_snapshot(measure_original_linux_host())
            print(json.dumps(actual, sort_keys=True))
            return 0 if "SATISFIED" in actual["status"] else 2
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        print(json.dumps({"status": "BLOCKED_UNVERIFIED_FAILED_HLE_CGROUP_EVIDENCE",
                          "reason": type(error).__name__,
                          "gen0_status": "BLOCKED", "release_approved": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
