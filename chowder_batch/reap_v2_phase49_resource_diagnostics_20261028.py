"""Phase49: read-only, independently pinned original Windows memory diagnostics.

The program reads historical frozen Q4 evidence and computes conservative
resource requirements. It never samples/runs a current native process, calls
nvidia-smi, loads weights, starts a GPU/Podman job, or modifies services. Its
capacity screen is NOT execution admission; a fresh independent Win32 observer,
original registered full-byte hashes, source lock, unique Job owner, and a new
source-approved launch remain mandatory elsewhere. Frozen flags never change.

Windows PagefileUsage/psutil.pagefile is *private committed charge*, not proof
of actual pagefile I/O. Windows process page_fault_count combines soft/hard
faults: it does not prove swap thrashing, reads or writes. Sampled working sets
are NOT whole-lifetime peak or disjoint memory from model mmap/cache/CUDA.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

GIB = 1024 ** 3
MIB = 1024 ** 2
SCHEMA = "chowder.reap128-v2-phase49-original-windows-resource-diagnostic.v1"
Q4_MODEL_SHA = "2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7"
FROZEN_FOUR_SLOT_FLAGS = ("--fit", "on", "-np", "4", "-c", "16384", "--jinja")
ORIGINAL_ONE_SLOT_FLAGS = ("--fit", "on", "-np", "1", "-c", "73728",
                           "--repeat-penalty", "1.1", "--dry-multiplier", "0.8",
                           "--temperature", "0.7", "--jinja")
HISTORIC_OWNER = "b6aa1f37e81943344217ede4429066c468bce2176a74eb8b15fcc0fe4a4e0529"

# Immutable historical evidence only; never glob a changing active log.
ABORT_PATH = Path("F:/reap128-v2-q4-registered-speed200-20261025-supervisor.json")
ABORT_SHA = "836e4e886f11819075d75a461eec028f0453b9fd3d00e2c9553e0d3177dd81ed"
ABORT_LOG = Path("C:/llm-work/reap/eval-v2-isolated-20261025/reap128_v2_q4/server.log")
ABORT_LOG_SHA = "94057821a77c7eaf15384850d1b155cb540904ef10e206a5f394289faa6012f1"
ABORT_ROWS = Path("C:/llm-work/reap/eval-v2-isolated-20261025/reap128_v2_q4/gsm8k.jsonl")
ABORT_ROWS_SHA = "8eaf2e83a5cf63d436b515434048c82fc3ddc00fc6b5144876d657e9f14b8684"
ONE_SLOT_REPORT = Path("F:/reap-v2-q4-guardv3-native-windows-resource-sustained-20261027-1510.json")
ONE_SLOT_REPORT_SHA = "e6bd81d114dbcc328f280dda59c7e12cd756fde976f54875c6bd645cda9f307b"
FOUR_SLOT_LOG = Path("F:/reap-v2-q4-windows-full1319-phase34-original-20261027-0152/reap128_v2_q4/server.log")
FOUR_SLOT_LOG_SHA = "a01acc0de9507de35fc1ab1a715bf6d104010362025862c24c475cb246b0ce4f"
PHASE48_SNAPSHOT = Path("evidence/reap-v2-phase48-live-dedicated-admission-refused-20261027.json")
PHASE48_SNAPSHOT_SHA = "9467f1184d492853cad8c1904dcbaf3a8f28eabd1a6b03961c6d8dbf29fbf558"
FULL1319_RESOURCE_SAMPLES = Path("F:/reap-v2-q4-windows-full1319-phase34-original-20261027-0152/resource-samples.jsonl")
FULL1319_RESOURCE_SAMPLES_SHA = "72f9ab7fbbc345b4cfb4e19990a1a44f2ca1364156a71331f653ce66a7141641"
FULL1319_TERMINAL = Path("F:/reap-v2-q4-windows-full1319-phase34-original-20261027-0152/full1319-terminal.json")
FULL1319_TERMINAL_SHA = "fabd6e8381ab68f73473b8a57bd80a94c2d2cf675968b7527fa5a37795bb0697"

CONSERVATIVE = {"minimum_ram_available_bytes": 48 * GIB,
                "minimum_commit_headroom_bytes": 48 * GIB,
                "minimum_disk_free_bytes": 8 * GIB,
                "minimum_gpu_total_mib": 48 * 1024,
                "minimum_gpu_free_mib": 40 * 1024,
                "original_12_row_runtime_ram_floor_bytes": 7 * GIB,
                "original_12_row_runtime_commit_floor_bytes": 8 * GIB,
                "original_4_slot_start_gpu_free_mib": 12288}


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require_sha(raw: bytes, expected: str) -> None:
    if (type(expected) is not str or re.fullmatch(r"[0-9a-f]{64}", expected) is None
            or sha256_bytes(raw) != expected):
        raise ValueError("original source SHA-256 changed or unregistered")


def _unique_pairs(items: list[tuple[str, object]]) -> dict:
    result = {}
    for k, v in items:
        if k in result:
            raise ValueError("duplicate JSON object key")
        result[k] = v
    return result


def _bad_constant(token: str) -> None:
    raise ValueError(f"nonfinite original JSON constant: {token}")


def strict_json(raw: bytes) -> dict:
    if not isinstance(raw, bytes) or not 0 < len(raw) <= 32 * MIB:
        raise ValueError("missing or unbounded JSON source")
    parsed = json.loads(raw, object_pairs_hook=_unique_pairs, parse_constant=_bad_constant)
    if type(parsed) is not dict:
        raise ValueError("original receipt must be an object")
    return parsed


def valid_number(value: object, field: str, *, integer: bool = False) -> int | float:
    if (type(value) not in ((int,) if integer else (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{field} requires finite nonnegative number")
    return value


def read_pinned(path: Path, expected_sha: str, *, max_bytes: int = 32 * MIB) -> bytes:
    if (path.is_symlink() or not path.is_file() or
            not 0 < path.stat().st_size <= max_bytes):
        raise ValueError("original source file absent/unsafe/oversized")
    raw = path.read_bytes()
    require_sha(raw, expected_sha)
    return raw


def analyze_abort_samples(source: dict) -> dict:
    """Direct cause of the historical aborted 12-row episode, not its allocator."""
    if (type(source) is not dict
            or source.get("status") != "ABORT_UNCHANGED_RUNTIME_MEMORY_FLOOR"
            or type(source.get("exit_code")) is not int
            or source["exit_code"] == 0
            or type(source.get("memory_floors_gib")) is not dict
            or source["memory_floors_gib"] != {"ram": 7, "commit": 8}):
        raise ValueError("original abort/frozen runtime floors source mismatch")
    initial = source.get("initial")
    samples = source.get("samples")
    if type(initial) is not dict or type(samples) is not list or len(samples) < 2 or len(samples) > 4096:
        raise ValueError("original resource series missing")
    initial_ram = valid_number(initial.get("ram_available_gib"), "initial RAM")
    initial_commit = valid_number(initial.get("commit_available_gib"), "initial commit")
    minimum_ram = math.inf
    minimum_commit = math.inf
    last_seconds = -1
    for sample in samples:
        if type(sample) is not dict:
            raise ValueError("invalid historical resource sample")
        at = valid_number(sample.get("s"), "historical elapsed seconds")
        if at <= last_seconds:
            raise ValueError("historic resource clock is nonmonotonic")
        last_seconds = at
        minimum_ram = min(minimum_ram, valid_number(sample.get("ram_available_gib"), "RAM"))
        minimum_commit = min(minimum_commit, valid_number(sample.get("commit_available_gib"), "commit"))
    ram_breach, commit_breach = minimum_ram < 7, minimum_commit < 8
    if not (ram_breach or commit_breach):
        raise ValueError("source claims abort but no verified floor crossing")
    return {"samples": len(samples), "minimum_ram_available_gib": minimum_ram,
            "minimum_commit_available_gib": minimum_commit,
            "ram_floor_crossed": ram_breach, "commit_floor_crossed": commit_breach,
            "maximum_observed_host_ram_drop_gib": max(0., initial_ram - minimum_ram),
            "maximum_observed_host_commit_drop_gib": max(0., initial_commit - minimum_commit),
            "direct_abort_cause": (
                "PHYSICAL_AND_COMMIT_BELOW_FROZEN_RUNTIME_FLOORS" if ram_breach and commit_breach
                else "PHYSICAL_RAM_BELOW_FROZEN_RUNTIME_FLOOR" if ram_breach
                else "COMMIT_BELOW_FROZEN_RUNTIME_FLOOR"),
            "confirmed_allocation_root_cause": "UNKNOWN_HOST_WIDE_COUNTERS_NOT_PROCESS_ATTRIBUTED",
            "historical_partial_rows_not_score": True}


def historical_cache_sizing(startup_log: bytes, abort_history: dict) -> dict:
    """Conservative prior-failure-bound original start floors; NOT a peak forecast."""
    data = startup_log.decode("utf-8", errors="strict")
    caps = re.findall(r"prompt cache is enabled, size limit:\s*(\d+) MiB", data)
    usage = re.findall(r"cache state:\s*\d+ prompts,\s*([0-9.]+) MiB", data)
    if len(caps) != 1 or not usage:
        raise ValueError("original historical host prompt cache evidence incomplete")
    cap = valid_number(int(caps[0]), "prompt cache limit")
    measured = [valid_number(float(value), "prompt cache used") for value in usage]
    if cap != 8192 or max(measured) > cap:
        raise ValueError("original host prompt cache limit or sampled use differs")
    if (abort_history.get("direct_abort_cause") !=
            "PHYSICAL_RAM_BELOW_FROZEN_RUNTIME_FLOOR"):
        raise ValueError("cannot size from unqualified original historic abort")
    remaining = (cap - max(measured)) / 1024
    required_ram = 7 + abort_history["maximum_observed_host_ram_drop_gib"] + 2 + remaining
    required_commit = 8 + abort_history["maximum_observed_host_commit_drop_gib"] + 2 + remaining
    return {"verified_cache_limit_mib": cap,
            "maximum_observed_cache_mib": max(measured),
            "remaining_cache_allocation_ceiling_gib": remaining,
            "historical_conservative_first200_ram_start_gib": max(16, required_ram),
            "historical_conservative_first200_commit_start_gib": max(16, required_commit),
            "source_bound_original_gpu_start_mib": 12288,
            "original_runtime_ram_reserve_gib": 7,
            "original_runtime_commit_reserve_gib": 8,
            "extra_safety_margin_gib_each": 2,
            "observed_host_decline_may_include_other_processes": True,
            "remaining_cache_is_ceiling_not_extra_disjoint_peak": True,
            "execution_approved": False}


def original_four_slot_watchdog(raw: bytes, terminal: dict) -> dict:
    """Validate all authentic 1319-slot host samples; not process WS or peak."""
    if (not isinstance(raw, bytes) or not raw.endswith(b"\n")
            or type(terminal) is not dict
            or terminal.get("schema") != "chowder.reap128-v2-q4-windows-full1319.v1"
            or terminal.get("status") != "COMPLETE_ORIGINAL_WINDOWS_Q4_FULL1319_CANDIDATE_ONLY"
            or terminal.get("child_exit_code") != 0
            or terminal.get("native_shutdown_verified") is not True
            or terminal.get("frozen_flags") != list(FROZEN_FOUR_SLOT_FLAGS)
            or terminal.get("original_model_sha256") != Q4_MODEL_SHA
            or terminal.get("gen0_status") != "BLOCKED"
            or terminal.get("release_approved") is not False):
        raise ValueError("original four-slot frozen terminal/session witness changed")
    lines = raw.splitlines()
    if terminal.get("sample_count") != 822 or len(lines) != 822:
        raise ValueError("frozen complete four-slot source resource sample count mismatch")
    mins = {"ram_available_gib": math.inf, "commit_available_gib": math.inf,
            "disk_free_gib": math.inf, "gpu_free_mib": math.inf}
    last = None
    first_values = None
    for line in lines:
        item = strict_json(line)
        if set(item) != {*mins, "time_utc"}:
            raise ValueError("frozen four-slot host resource sample schema changed")
        try:
            stamp = datetime.fromisoformat(item["time_utc"])
        except (ValueError, TypeError) as error:
            raise ValueError("invalid original native sample clock") from error
        if stamp.utcoffset() is None or (last is not None and stamp <= last):
            raise ValueError("frozen four-slot watchdog clock out of order")
        last = stamp
        for key in mins:
            mins[key] = min(mins[key], valid_number(item[key], key))
        if first_values is None:
            first_values = {key: item[key] for key in mins}
    if (mins["ram_available_gib"] < 7
            or mins["commit_available_gib"] < 8
            or mins["disk_free_gib"] < 8):
        raise ValueError("original four-slot GSM8K native runtime reserve breached")
    # The earlier 12-row conservative attempt floor is NOT sufficient to
    # predict a sustained full1319 run: the preserved complete-session
    # source shows a larger host-wide decline. These are analytic prestart
    # resource estimates only, never authorization or model-attributed usage.
    full_ram_decline = first_values["ram_available_gib"] - mins["ram_available_gib"]
    full_commit_decline = first_values["commit_available_gib"] - mins["commit_available_gib"]
    return {"sample_count": len(lines), "sampled_host_minimums": mins,
            "first_sampled_host_values": first_values,
            "maximum_host_physical_decline_from_first_sample_gib": full_ram_decline,
            "maximum_host_commit_decline_from_first_sample_gib": full_commit_decline,
            "conservative_full1319_physical_prestart_gib": 7 + 2 + full_ram_decline,
            "conservative_full1319_commit_prestart_gib": 8 + 2 + full_commit_decline,
            "not_model_attributed_not_execution_approval": True,
            "status": "PASS_HISTORICAL_FOUR_SLOT_HOST_RAM_COMMIT_DISK_FLOORS",
            "host_gpu_free_is_not_model_exclusive_peak": True,
            "model_specific_four_slot_working_set_and_peak": "UNKNOWN",
            "model_loading_now": False}


def check_native_sample(source: dict, owner: str, native_pid: int, evaluator_pid: int) -> dict:
    """Validate one preserved Win32 snapshot with exact session/PID guard."""
    if (type(source) is not dict or type(owner) is not str
            or len(owner) != 64 or not re.fullmatch(r"[0-9a-f]{64}", owner)
            or type(native_pid) is not int or native_pid <= 0
            or type(evaluator_pid) is not int or evaluator_pid <= 0
            or source.get("source_owner_sha256") != owner
            or source.get("original_native_pid") != native_pid
            or source.get("original_evaluator_pid") != evaluator_pid):
        raise ValueError("Win32 native PID/evaluator/owner evidence identity mismatch")
    fields = ("physical_available_bytes", "physical_total_bytes", "commit_headroom_bytes",
              "commit_limit_bytes", "commit_total_bytes", "commit_peak_bytes",
              "native_wset_bytes", "native_private_bytes", "native_pagefile_bytes",
              "native_page_faults_cumulative", "host_gpu_total_mib", "host_gpu_used_mib",
              "host_gpu_free_mib", "host_gpu_unaccounted_wddm_reserved_mib",
              "observed_unix_seconds", "completed_receipt_files")
    for key in fields:
        valid_number(source.get(key), key, integer=key not in ("observed_unix_seconds",))
    if (source["commit_total_bytes"] + source["commit_headroom_bytes"]
            != source["commit_limit_bytes"]
            or source["commit_peak_bytes"] < source["commit_total_bytes"]
            or source["physical_available_bytes"] > source["physical_total_bytes"]):
        raise ValueError("original Windows commit/physical counters inconsistent")
    if (source["host_gpu_total_mib"] != source["host_gpu_used_mib"]
            + source["host_gpu_free_mib"]
            + source["host_gpu_unaccounted_wddm_reserved_mib"]):
        raise ValueError("original WDDM total=used+free+reserved GPU accounting changed")
    if source["native_private_bytes"] != source["native_pagefile_bytes"]:
        raise ValueError("Win32 pagefile/private charge alias source changed")
    return {**{key: source[key] for key in fields},
            "pagefile_io_bytes": None, "hard_page_faults": None}


def analyze_native_samples(samples: list[dict], owner: str,
                           native_pid: int, evaluator_pid: int) -> dict:
    if type(samples) is not list or not 2 <= len(samples) <= 120:
        raise ValueError("original bounded Windows native sample count missing")
    trusted = [check_native_sample(s, owner, native_pid, evaluator_pid) for s in samples]
    for earlier, later in zip(trusted, trusted[1:]):
        if later["observed_unix_seconds"] <= earlier["observed_unix_seconds"]:
            raise ValueError("native samples out of time order")
        if later["native_page_faults_cumulative"] < earlier["native_page_faults_cumulative"]:
            raise ValueError("native Windows pagefault counter reset or PID reused")
        if later["completed_receipt_files"] < earlier["completed_receipt_files"]:
            raise ValueError("source-bound benchmark completed receipts decreased")
    return {
        "sample_count": len(trusted), "owner_sha256": owner,
        "native_pid": native_pid, "evaluator_pid": evaluator_pid,
        "duration_seconds": trusted[-1]["observed_unix_seconds"] - trusted[0]["observed_unix_seconds"],
        "max_sampled_native_working_set_bytes": max(x["native_wset_bytes"] for x in trusted),
        "max_sampled_native_private_commit_bytes": max(x["native_private_bytes"] for x in trusted),
        "min_sampled_physical_available_bytes": min(x["physical_available_bytes"] for x in trusted),
        "min_sampled_commit_headroom_bytes": min(x["commit_headroom_bytes"] for x in trusted),
        "min_host_wddm_gpu_free_mib": min(x["host_gpu_free_mib"] for x in trusted),
        "page_fault_count_delta": (trusted[-1]["native_page_faults_cumulative"]
                                   - trusted[0]["native_page_faults_cumulative"]),
        "hard_page_faults_and_pagefile_io_proven": False,
        "native_lifetime_peak_proven": False,
        "host_gpu_used_is_exclusive_process_allocation": False,
        "private_commit_is_not_actual_pagefile_disk_io": True,
    }


def check_one_slot_provenance(source: dict, owner: str) -> dict:
    a = source.get("source_bound_native_allocator") if type(source) is dict else None
    if (source.get("schema") not in (None, "chowder.reap128-v2-original600-windows-native-resource-observation.v1")
            or source.get("source_owner_sha256") != owner
            or source.get("native_original_flags") != list(ORIGINAL_ONE_SLOT_FLAGS)
            or source.get("original_model_sha256", Q4_MODEL_SHA) != Q4_MODEL_SHA
            or type(a) is not dict or a.get("slots") != 1
            or a.get("native_context_tokens") != 73728
            or a.get("native_cuda_offloaded_layers") != 41
            or a.get("native_cuda_total_layers") != 41
            or a.get("native_cpu_expert_offload") is not False
            or a.get("original_gguf_sha256") != Q4_MODEL_SHA):
        raise ValueError("original one-slot model/flags/allocator identity mismatch")
    values = a.get("measured_original_cuda_allocator_mib")
    if type(values) is not dict:
        raise ValueError("original native CUDA allocator not sourced")
    keys = ("cuda_kv_buffer_mib", "cuda_model_buffer_mib",
            "cuda_compute_buffer_mib", "cuda_recurrent_state_buffer_mib",
            "prompt_cache_limit_mib")
    for key in keys:
        valid_number(values.get(key), key)
    if (values["cuda_kv_buffer_mib"] != 1440.0
            or values["cuda_model_buffer_mib"] != 10619.88
            or values["cuda_compute_buffer_mib"] != 172.05
            or values["cuda_recurrent_state_buffer_mib"] != 62.81
            or values["prompt_cache_limit_mib"] != 8192.0):
        raise ValueError("original native allocator quantities changed")
    return {"slots": 1, "context_tokens": 73728, "cuda_kv_mib": 1440.0,
            "cuda_model_mib": 10619.88, "cuda_compute_mib": 172.05,
            "cuda_recurrent_mib": 62.81, "host_prompt_cache_ceiling_mib": 8192.0,
            "gpu_cuda_reported_buffer_sum_mib": round(10619.88 + 172.05 + 62.81 + 1440., 2),
            "report_buffer_sum_is_not_process_gpu_peak": True}


def parse_four_slot_native_startup(log: bytes) -> dict:
    """Parse only native original frozen 4-way fit markers, never model load."""
    if not isinstance(log, bytes) or not 0 < len(log) <= 32 * MIB:
        raise ValueError("missing/bounded original four-slot startup source")
    prefix = log[:131072].decode("utf-8", errors="strict")
    def single(pattern: str) -> str:
        values = re.findall(pattern, prefix, re.MULTILINE)
        if len(values) != 1:
            raise ValueError("four-slot original native fit/KV geometry absent or ambiguous")
        return values[0] if isinstance(values[0], str) else values[0]
    if (prefix.count("common_fit_params: successfully fit params to free device memory") != 1
            or prefix.count("offloaded 41/41 layers to GPU") != 1):
        raise ValueError("four-slot native original fit and CUDA placement unverified")
    seq = int(single(r"llama_context:\s*n_seq_max\s*=\s*(\d+)"))
    kv = float(single(r"llama_kv_cache:\s*CUDA0 KV buffer size\s*=\s*([\d.]+) MiB"))
    layout = re.findall(r"load_model: initializing, n_slots = (\d+), n_ctx_slot = (\d+), kv_unified = 'false'", prefix)
    cap = int(single(r"prompt cache is enabled, size limit:\s*(\d+) MiB"))
    if len(layout) != 1 or (int(layout[0][0]), int(layout[0][1])) != (4, 4096):
        raise ValueError("four-slot original native n_ctx/slot geometry mismatch")
    if seq != 4 or kv != 320.0 or cap != 8192:
        raise ValueError("four-slot original native KV, prompt cache, or n_seq mismatch")
    return {"slots": 4, "context_tokens": 16384,
            "per_slot_context_tokens": 4096, "cuda_kv_mib": kv,
            "prompt_cache_ceiling_mib": cap,
            "native_fit_marker_verified": True,
            "gpu_cuda_layers": "41/41", "cpu_expert_offload_proven_from_this_log": False}


def compare_measured_kv(four_slot_mib: float, one_slot_mib: float,
                        four_slot_context: int, one_slot_context: int) -> dict:
    for name, n in (("4-way CUDA KV", four_slot_mib), ("1-way CUDA KV", one_slot_mib)):
        if valid_number(n, name) <= 0:
            raise ValueError("original measured CUDA KV must be positive")
    if (type(four_slot_context) is not int or type(one_slot_context) is not int
            or four_slot_context != 16384 or one_slot_context != 73728):
        raise ValueError("original frozen 4-way vs distinct one-way context changed")
    lhs = four_slot_mib / four_slot_context
    rhs = one_slot_mib / one_slot_context
    return {"four_slot_cuda_kv_mib": four_slot_mib,
            "experimental_one_slot_cuda_kv_mib": one_slot_mib,
            "measured_mib_per_context_token": lhs,
            "same_measured_kv_bytes_per_token": math.isclose(lhs, rhs, abs_tol=1e-12),
            "observed_kv_geometry_ratio": one_slot_mib / four_slot_mib,
            "measured_profiles_are_not_same_frozen_protocol": True,
            "any_other_context_profile_admitted": False}


def capacity_screen(observed: dict) -> dict:
    """Conservative offline resource SCREEN. It cannot authorize execution."""
    if type(observed) is not dict or type(observed.get("gpu")) is not dict:
        raise ValueError("original independent source capacity screen missing counters")
    gpu = observed["gpu"]
    for key in ("physical_available_bytes", "commit_headroom_bytes", "disk_free_bytes"):
        valid_number(observed.get(key), key, integer=True)
    for key in ("total_mib", "used_mib", "free_mib", "reserved_mib"):
        valid_number(gpu.get(key), key, integer=True)
    if gpu["total_mib"] != gpu["used_mib"] + gpu["free_mib"] + gpu["reserved_mib"]:
        raise ValueError("GPU WDDM memory total disagrees with used/free/reserved")
    reasons = []
    req = CONSERVATIVE
    for field, limit, reason in (
        ("physical_available_bytes", req["minimum_ram_available_bytes"], "RAM_48G_FREE"),
        ("commit_headroom_bytes", req["minimum_commit_headroom_bytes"], "COMMIT_48G_FREE"),
        ("disk_free_bytes", req["minimum_disk_free_bytes"], "DISK_8G_FREE"),
    ):
        if observed[field] < limit:
            reasons.append(reason)
    for field, limit, reason in (
        ("total_mib", req["minimum_gpu_total_mib"], "GPU_48G_CLASS"),
        ("free_mib", req["minimum_gpu_free_mib"], "GPU_40G_DEDICATED"),
        ("free_mib", req["original_4_slot_start_gpu_free_mib"], "GPU_12G_NATIVE_FLOOR"),
    ):
        if gpu[field] < limit:
            reasons.append(reason)
    for name in ("active_conflicts", "listeners"):
        if type(observed.get(name)) is not list:
            raise ValueError("independent foreign-process/port inventory missing")
        if observed[name]:
            reasons.append("ACTIVE_PROCESSES" if name == "active_conflicts" else "ACTIVE_LISTENERS")
    if (observed.get("port_census_complete") is not True
            or observed.get("process_census_complete") is not True):
        reasons.append("INCOMPLETE_PROCESS_OR_PORT_CENSUS")
    return {"status": "CAPACITY_SCREEN_ONLY_NO_EXECUTION_AUTHORIZATION",
            "provisional_resource_capacity_only": not reasons,
            "reasons": sorted(set(reasons)), "original_required_floors": dict(CONSERVATIVE),
            "execution_approved": False, "proves_frozen_model_fit": False,
            "proves_model_specific_lifetime_peaks": False,
            "source_signature_is_not_independent_observer": True,
            "gen0_status": "BLOCKED", "chosen": None, "release_approved": False}


def audit_preserved() -> dict:
    """No argument-controlled replacement of the pinned original Windows inputs."""
    abort = strict_json(read_pinned(ABORT_PATH, ABORT_SHA, max_bytes=256 * 1024))
    history = analyze_abort_samples(abort)
    if (abort.get("schema") != "reap128-v2-q4-registered-200-supervisor.v1"
            or abort.get("initial", {}).get("gpu_free_mib") != 13993
            or len(abort.get("samples", [])) != 50
            or abs(history["minimum_ram_available_gib"] - 6.98) > 1e-9
            or abs(history["minimum_commit_available_gib"] - 10.41) > 1e-9):
        raise ValueError("source-bound authentic 12-row historical abort changed")
    original_rows = read_pinned(ABORT_ROWS, ABORT_ROWS_SHA, max_bytes=MIB)
    if not original_rows.endswith(b"\n"):
        raise ValueError("original partial rows truncated")
    rows = [strict_json(line) for line in original_rows.splitlines()]
    if (len(rows) != 12 or {r.get("id") for r in rows} !=
            {f"gsm8k-{i}" for i in range(12)}):
        raise ValueError("historical 12-row source-bound incomplete prefix changed")
    abort_log = read_pinned(ABORT_LOG, ABORT_LOG_SHA, max_bytes=MIB)
    first200_sizing = historical_cache_sizing(abort_log, history)
    abort_four = parse_four_slot_native_startup(abort_log)
    four_log = read_pinned(FOUR_SLOT_LOG, FOUR_SLOT_LOG_SHA)
    confirmed_four = parse_four_slot_native_startup(four_log)
    if abort_four != confirmed_four:
        raise ValueError("historical partial/full frozen Q4 native KV layout drift")
    terminal = strict_json(read_pinned(FULL1319_TERMINAL, FULL1319_TERMINAL_SHA, max_bytes=MIB))
    full_host = original_four_slot_watchdog(
        read_pinned(FULL1319_RESOURCE_SAMPLES, FULL1319_RESOURCE_SAMPLES_SHA, max_bytes=MIB),
        terminal)
    one = strict_json(read_pinned(ONE_SLOT_REPORT, ONE_SLOT_REPORT_SHA, max_bytes=MIB))
    one_info = check_one_slot_provenance(one, HISTORIC_OWNER)
    original = analyze_native_samples(one.get("raw_native_windows_observations"),
                                      HISTORIC_OWNER, one.get("native_original_pid"),
                                      one.get("original_evaluator_pid"))
    if (one.get("samples") != original["sample_count"]
            or one.get("max_native_wset_bytes") != original["max_sampled_native_working_set_bytes"]
            or one.get("max_native_private_bytes") != original["max_sampled_native_private_commit_bytes"]
            or one.get("min_native_windows_commit_headroom_bytes") != original["min_sampled_commit_headroom_bytes"]
            or one.get("min_physical_host_available_bytes") != original["min_sampled_physical_available_bytes"]
            or one.get("min_host_gpu_free_mib") != original["min_host_wddm_gpu_free_mib"]
            or one.get("native_page_fault_count_delta") != original["page_fault_count_delta"]
            or one.get("host_gpu_used_is_model_exclusive") is not False
            or one.get("gen0_status") != "BLOCKED"
            or one.get("release_approved") is not False):
        raise ValueError("one-slot authentic Win32 resource summary/receipts disagree")
    gpu_kv = compare_measured_kv(confirmed_four["cuda_kv_mib"],
                                 one_info["cuda_kv_mib"], 16384, 73728)
    if gpu_kv["same_measured_kv_bytes_per_token"] is not True:
        raise ValueError("native Q4 CUDA KV buffer/context geometry changed")
    prior = strict_json(read_pinned(PHASE48_SNAPSHOT, PHASE48_SNAPSHOT_SHA, max_bytes=MIB))
    if (prior.get("schema") != "chowder.reap-v2-dedicated-windows-runner-admission.v1"
            or prior.get("registered_flags") != list(FROZEN_FOUR_SLOT_FLAGS)
            or prior.get("original_model", {}).get("sha256") != Q4_MODEL_SHA
            or prior.get("report", {}).get("status") != "REFUSED"
            or prior.get("never_launches_model") is not True):
        raise ValueError("original Phase48 native admission source or refusal drift")
    observed = prior.get("observed")
    screen = capacity_screen(observed)
    if screen["provisional_resource_capacity_only"] is not False:
        raise ValueError("original Phase48 refused host is deceptively marked viable")
    return {
        "schema": SCHEMA, "status": "PASS_PRESERVED_RESOURCE_EVIDENCE_DIAGNOSTIC_ONLY",
        "sources": {"historical_abort_sha256": ABORT_SHA, "historical_12_rows_sha256": ABORT_ROWS_SHA,
                    "historical_4_slot_log_sha256": ABORT_LOG_SHA,
                    "full1319_4_slot_log_sha256": FOUR_SLOT_LOG_SHA,
                    "full1319_host_resource_samples_sha256": FULL1319_RESOURCE_SAMPLES_SHA,
                    "full1319_terminal_sha256": FULL1319_TERMINAL_SHA,
                    "one_slot_resource_report_sha256": ONE_SLOT_REPORT_SHA,
                    "phase48_refusal_sha256": PHASE48_SNAPSHOT_SHA,
                    "original_q4_gguf_registered_sha256_not_rehashed": Q4_MODEL_SHA},
        "historical_12_rows": history,
        "original_first200_conservative_prestart_sizing": first200_sizing,
        "frozen_four_slot_native": confirmed_four,
        "frozen_four_slot_gsm8k1319_historical_watchdog": full_host,
        "experimental_one_slot_native": one_info,
        "different_context_measured_cuda_kv": gpu_kv,
        "native_windows_experimental_24_second_sample_window": original,
        "phase48_historical_host_capacity": screen,
        "independent_gpu_process_allocation_peak": "UNKNOWN",
        "hard_faults_swap_io_and_pagefile_traffic": "UNKNOWN",
        "four_slot_sustained_host_qualification": "UNKNOWN",
        "execution_approved": False, "gen0_status": "BLOCKED",
        "chosen": None, "release_approved": False,
    }


def main() -> int:
    try:
        report = audit_preserved()
        print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as exc:
        print(json.dumps({"schema": SCHEMA, "status": "UNKNOWN_SOURCE_OR_RESOURCE_EVIDENCE_DRIFT",
                          "reason": str(exc)[:300], "execution_approved": False,
                          "gen0_status": "BLOCKED", "chosen": None,
                          "release_approved": False}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
