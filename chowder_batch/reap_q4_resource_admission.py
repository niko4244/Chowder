"""Read-only admission for genuine REAP128_v2 Q4 first-200 qualification.

This guards against repeating a *known* host-memory violation. It does not
start/stop models, edit serving flags, score benchmarks, or assert a scientific
gate. The source-bound preceding abort is a lower bound, not a forecast of a
successful 200-item run. Production requires the frozen original receipt SHA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path

from chowder_batch.reap_prune_supervised import host_state

GIB = 1024 ** 3
ORIGINAL_ABORT_SHA256 = "836e4e886f11819075d75a461eec028f0453b9fd3d00e2c9553e0d3177dd81ed"
ORIGINAL_LOG_SHA256 = "94057821a77c7eaf15384850d1b155cb540904ef10e206a5f394289faa6012f1"
ORIGINAL_ABORT = Path("F:/reap128-v2-q4-registered-speed200-20261025-supervisor.json")
ORIGINAL_SERVER_LOG = Path("C:/llm-work/reap/eval-v2-isolated-20261025/reap128_v2_q4/server.log")
ORIGINAL_GPU_START_FLOOR_MIB = 12288
ORIGINAL_RAM_RUNTIME_FLOOR_GIB = 7
ORIGINAL_COMMIT_RUNTIME_FLOOR_GIB = 8
ORIGINAL_DISK_FLOOR_GIB = 8
MIN_ADDITIONAL_HEADROOM_GIB = 2


def _strict_number(value: object, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field} must be a finite number, not a bool/string")
    if not math.isfinite(value) or value < 0 or (positive and value <= 0):
        raise ValueError(f"{field} is invalid/nonfinite")
    return float(value)


def _read_sha(path: Path, expected_sha256: str) -> bytes:
    if (not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
            or path.is_symlink() or not path.is_file()):
        raise ValueError("trusted source evidence missing, symlink or invalid digest")
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != expected_sha256:
        raise ValueError("trusted source bytes differ from the original pinned SHA-256")
    return contents


def validate_aborted_speed_history(path: Path, *, expected_sha256: str) -> dict:
    """Admit only a genuinely aborted original first-200 4-way Q4 experiment."""
    try:
        raw = _read_sha(Path(path), expected_sha256)
        rec = json.loads(raw, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
        initial = rec["initial"]
        samples = rec["samples"]
        floor = rec["memory_floors_gib"]
        command = rec["command"]
        if (rec.get("schema") != "reap128-v2-q4-registered-200-supervisor.v1"
                or rec.get("status") != "ABORT_UNCHANGED_RUNTIME_MEMORY_FLOOR"
                or not isinstance(command, list)
                or any(not isinstance(v, str) for v in command)
                or not isinstance(samples, list) or len(samples) < 2
                or not isinstance(initial, dict) or not isinstance(floor, dict)):
            raise ValueError("missing original scientific abort structure")
        # Freeze the real Q4, four-concurrent, official first-200 command.
        required = {"--gguf": "G:/reap128-v2-q4_k_m-20261025.gguf",
                    "--arm": "reap128_v2_q4", "--suites": "gsm8k",
                    "--limit": "200", "--parallel": "4"}
        for option, value in required.items():
            # Serving and evaluator may each declare --arm. All occurrences
            # must agree with this original source-bound experiment.
            positions = [i for i, item in enumerate(command[:-1]) if item == option]
            if not positions or any(command[i + 1] != value for i in positions):
                raise ValueError(f"scientific source does not bind {option} {value}")
        if (_strict_number(floor["ram"], "original RAM floor")
                != ORIGINAL_RAM_RUNTIME_FLOOR_GIB
                or _strict_number(floor["commit"], "original commit floor")
                != ORIGINAL_COMMIT_RUNTIME_FLOOR_GIB):
            raise ValueError("original frozen runtime floor altered")
        if _strict_number(initial["gpu_free_mib"], "original free VRAM") < ORIGINAL_GPU_START_FLOOR_MIB:
            raise ValueError("historical GPU session never passed original admission")
        r0 = _strict_number(initial["ram_available_gib"], "original available RAM", positive=True)
        c0 = _strict_number(initial["commit_available_gib"], "original available commit", positive=True)
        ram_values, commit_values = [], []
        last_time = -1.0
        for entry in samples:
            if not isinstance(entry, dict):
                raise ValueError("sample is not a resource measurement")
            t = _strict_number(entry["s"], "sample time")
            if t <= last_time:
                raise ValueError("resource samples are out of order or duplicated")
            last_time = t
            ram_values.append(_strict_number(entry["ram_available_gib"], "sample RAM"))
            commit_values.append(_strict_number(entry["commit_available_gib"], "sample commit"))
        if (min(ram_values) > ORIGINAL_RAM_RUNTIME_FLOOR_GIB
                and min(commit_values) > ORIGINAL_COMMIT_RUNTIME_FLOOR_GIB):
            raise ValueError("source claimed abort without measured original floor violation")
        return {
            "source_sha256": expected_sha256,
            "observed_rows": 12,  # verified separately from the original 12-row JSONL
            "original_ram_gib": r0, "original_commit_gib": c0,
            "minimum_observed_ram_gib": min(ram_values),
            "minimum_observed_commit_gib": min(commit_values),
            "measured_ram_drop_gib": max(0.0, r0 - min(ram_values)),
            "measured_commit_drop_gib": max(0.0, c0 - min(commit_values)),
            "history_is_lower_bound_only": True,
        }
    except (OSError, UnicodeError, KeyError, TypeError, IndexError, json.JSONDecodeError) as exc:
        raise ValueError("original Q4 abort receipt is incomplete or unreadable") from exc


def parse_host_prompt_cache(path: Path, *, expected_sha256: str) -> dict:
    """Bound potential extra RAM by the original server's own logged cache."""
    data = _read_sha(path, expected_sha256).decode("utf-8", errors="replace")
    cap = re.findall(r"prompt cache is enabled, size limit:\s*([0-9]+) MiB", data)
    samples = re.findall(r"cache state:\s*([0-9]+) prompts,\s*([0-9.]+) MiB", data)
    if len(cap) != 1 or not samples:
        raise ValueError("original host prompt cache cap/usage cannot be verified")
    maximum = _strict_number(int(cap[0]), "host prompt-cache cap", positive=True)
    observed = max(_strict_number(float(size), "observed host prompt cache")
                   for _, size in samples)
    if maximum > 1024 * 128 or observed > maximum:
        raise ValueError("host prompt-cache evidence is inconsistent")
    return {"cap_mib": maximum, "maximum_observed_mib": observed,
            "potential_remaining_gib": (maximum - observed) / 1024,
            "source_sha256": expected_sha256}


def admit_host(history: dict, *, ram_available_gib: float, commit_available_gib: float,
               gpu_free_mib: float, disk_free_gib: float,
               ram_headroom_gib: float = MIN_ADDITIONAL_HEADROOM_GIB,
               commit_headroom_gib: float = MIN_ADDITIONAL_HEADROOM_GIB,
               potential_remaining_cache_gib: float = 0.0) -> dict:
    """Never predict PASS. Exceed the previous resource drop plus original floor."""
    values = {}
    problems = []
    for name, value in (("RAM", ram_available_gib), ("commit", commit_available_gib),
                        ("VRAM", gpu_free_mib), ("disk", disk_free_gib)):
        try:
            values[name] = _strict_number(value, f"available {name}")
        except ValueError as exc:
            problems.append(str(exc))
            values[name] = 0.0
    for name, value in (("RAM headroom", ram_headroom_gib),
                        ("commit headroom", commit_headroom_gib)):
        if _strict_number(value, name) < MIN_ADDITIONAL_HEADROOM_GIB:
            raise ValueError("resource headroom cannot be lower than original safety margin")
    extra_cache = _strict_number(potential_remaining_cache_gib, "potential host prompt-cache growth")
    required_ram = max(16.0, ORIGINAL_RAM_RUNTIME_FLOOR_GIB
                       + _strict_number(history["measured_ram_drop_gib"], "historical RAM drop")
                       + ram_headroom_gib + extra_cache)
    required_commit = max(16.0, ORIGINAL_COMMIT_RUNTIME_FLOOR_GIB
                          + _strict_number(history["measured_commit_drop_gib"], "historical commit drop")
                          + commit_headroom_gib + extra_cache)
    if values["RAM"] < required_ram:
        problems.append("available physical RAM below witnessed drop plus original floor and headroom")
    if values["commit"] < required_commit:
        problems.append("available Windows commit below witnessed drop plus original floor and headroom")
    if values["VRAM"] < ORIGINAL_GPU_START_FLOOR_MIB:
        problems.append("free VRAM below original 12-GiB start admission")
    if values["disk"] < ORIGINAL_DISK_FLOOR_GIB:
        problems.append("free disk below original 8-GiB runtime reserve")
    return {"status": ("REFUSED" if problems else
                       "ADMITTED_TO_ATTEMPT_NOT_SCIENTIFICALLY_QUALIFIED"),
            "reasons": problems, "minimum_ram_gib": required_ram,
            "minimum_commit_gib": required_commit,
            "minimum_gpu_mib": ORIGINAL_GPU_START_FLOOR_MIB,
            "minimum_disk_gib": ORIGINAL_DISK_FLOOR_GIB,
            "potential_remaining_host_cache_gib": extra_cache,
            "measured_prior_abort_lower_bound": True,
            "scientific_gate": "UNKNOWN", "quality_gate": "UNKNOWN", "speed_gate": "UNKNOWN",
            "no_process_launched": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="Exclusive report file outside protected models and prior receipts")
    args = parser.parse_args(argv)
    if os.name != "nt":
        raise RuntimeError("this resource preflight is Windows-specific")
    out = args.out.resolve()
    if not out.parent.is_dir() or out.exists() or out.is_symlink():
        raise ValueError("original admission report requires a new exclusive output")
    history = validate_aborted_speed_history(ORIGINAL_ABORT, expected_sha256=ORIGINAL_ABORT_SHA256)
    cache = parse_host_prompt_cache(ORIGINAL_SERVER_LOG, expected_sha256=ORIGINAL_LOG_SHA256)
    h = host_state(out.parent)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                          "--format=csv,noheader,nounits"], capture_output=True,
                         text=True, timeout=15, check=True)
    gpu_free = _strict_number(int(gpu.stdout.strip().splitlines()[0]), "current GPU free memory")
    decision = admit_host(history, ram_available_gib=h.ram_available_gib,
                          commit_available_gib=h.commit_available_gib,
                          gpu_free_mib=gpu_free,
                          disk_free_gib=h.output_disk_free_gib,
                          potential_remaining_cache_gib=cache["potential_remaining_gib"])
    result = {"schema": "reap128-v2-verified-host-admission.v1", "history": history,
              "original_host_prompt_cache": cache,
              "current": {"ram_available_gib": h.ram_available_gib,
                          "commit_available_gib": h.commit_available_gib,
                          "gpu_free_mib": gpu_free, "disk_free_gib": h.output_disk_free_gib},
              "decision": decision}
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(decision, sort_keys=True))
    return 0 if decision["status"].startswith("ADMITTED_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
