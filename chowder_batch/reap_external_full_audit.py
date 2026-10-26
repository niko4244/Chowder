"""Read-only independent audit of complete, frozen-population external REAP rows.

This is NOT an Amendment 4 Windows serving-owner attestation. It cannot turn
Kaggle quality/control/speed results into eligible Gen-0 selector inputs.
Never executes generated HumanEval code: an independently isolated execution
receipt must be provided and verified separately for that score cross-check.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
import re
import sys
from pathlib import Path

FROZEN_FLAGS = ["--fit", "on", "-np", "4", "-c", "16384", "--jinja"]
Q4_SHA = "2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7"
Q8_SHA = "89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6"
RANDOM_SHA = "601871e00403658708eb4eebf3b05f9dc54a93b326e31f2f87b48cb59f92241b"
SERVER_SHA = "f10032228f1d80b0f6639cd3bcdcc7194bbc6638368641cac7f1396775b5186d"
MAX_ORIGINAL_SERVER_LOG_BYTES = 32 * 1024 * 1024
EVALUATOR_SHA = "9736714f21834a45c23599a50fa92ef3bebc6cf70967aca7e2329d26b4d73ee1"
SCORER_SHA = "d2e59aca5c0a621bac8d4979d522c9b6107f811610567d21c92a2b53960860eb"
MODELS = {
    "reap128_v2_q4": (
        "reap-v2-q4-kaggle-complete-provisional-eval.v1",
        "COMPLETE_FULL_DATA_PROVISIONAL_NOT_FROZEN_WINDOWS_GATE", "q4_sha256", Q4_SHA,
    ),
    "reap128_v2_q8": (
        "reap-v2-q8-original-full-with-measured-postload-cache-reclaim.v1",
        "COMPLETE_Q8_RECLAIM_FULL_PROVISIONAL_NOT_FROZEN_WINDOWS_GATE", "q8_sha256", Q8_SHA,
    ),
    "random128_v2_q8": (
        "reap-v2-frozen-independent-random128-full-postload-reclaim.v1",
        "COMPLETE_RANDOM128_RECLAIM_FULL_PROVISIONAL_NOT_FROZEN_WINDOWS_GATE",
        "random128_sha256", RANDOM_SHA,
    ),
}

def _check_postload_resource_evidence(receipt: dict) -> dict:
    """Read the *measured* Q8/control cgroup admission, never infer Windows fit.

    Original complete postload notebooks measure true unused cgroup capacity
    independently of any theoretically reclaimable file cache. Their 10-GiB
    stop protects the preregistered 8-GiB runtime floor. Checking final
    hashes/rows without these measurements would allow a false resource PASS.
    """
    def amount(value: object, name: str) -> float:
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"missing/nonfinite/invalid measured {name}")
        return float(value)

    def host(field: str) -> dict:
        value = receipt.get(field)
        if not isinstance(value, dict):
            raise ValueError(f"missing original postload resource witness {field}")
        ram = amount(value.get("physical_free_gib"), field + ".physical_free_gib")
        unused = amount(value.get("cgroup_unused_gib"), field + ".cgroup_unused_gib")
        gpu = value.get("gpu_free_mib")
        if (not isinstance(gpu, list) or len(gpu) != 2
                or any(type(x) is not int or x < 0 for x in gpu)):
            raise ValueError("missing or corrupt actual two-GPU VRAM inventory")
        if ram < 7 or unused < 10:
            raise ValueError("original real runtime RAM/cgroup safety floor violated")
        return {"ram": ram, "unused": unused, "gpu": gpu}

    initial = host("initial")
    pre = host("pre_reclaim_host")
    post = host("post_reclaim_host")
    before = host("before_inference_host")
    after = host("after_inference_host")
    if (initial["ram"] < 24 or initial["unused"] < 24
            or any(free < 12288 for free in initial["gpu"])):
        raise ValueError("original pinned two-GPU native model-load admission not proven")
    recovery = post["unused"] - pre["unused"]
    reported = amount(receipt.get("postload_clean_file_cache_recovered_gib"),
                      "reported postload recovery")
    if (recovery < 4 or post["unused"] < 14
            or before["unused"] < 10
            or not math.isclose(recovery, reported, rel_tol=0, abs_tol=0.00101)):
        raise ValueError("actual postload 4-GiB recovery/14-GiB headroom not proven")

    def stats(field: str) -> dict:
        value = receipt.get(field)
        if not isinstance(value, dict):
            raise ValueError(f"missing original cgroup file/anon attribution {field}")
        return {k: amount(value.get(k + "_gib"), field + "." + k + "_gib")
                for k in ("file", "anon")}

    file_pre = stats("pre_reclaim_cgroup")
    file_post = stats("post_reclaim_cgroup")
    if file_pre["file"] - file_post["file"] < 4:
        raise ValueError("claimed postload cache recovery has no measured file-cache drop")
    samples = receipt.get("samples")
    if not isinstance(samples, list) or not (2 <= len(samples) <= 20000):
        raise ValueError("original monitored runtime resource samples missing or unbounded")
    ram_min = min(initial["ram"], pre["ram"], post["ram"], before["ram"], after["ram"])
    unused_min = min(initial["unused"], pre["unused"], post["unused"],
                     before["unused"], after["unused"])
    gpu_min = [initial["gpu"][0], initial["gpu"][1]]
    for record in (pre, post, before, after):
        gpu_min = [min(left, right) for left, right in zip(gpu_min, record["gpu"])]
    runtime_samples = 0
    last_runtime_stamp = None
    first_runtime_stamp = None
    max_runtime_gap = 0.0
    boot_ended = False
    last_boot_seconds = -1.0
    for i, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError("malformed supervised resource sample")
        ram = amount(sample.get("physical_free_gib"), f"samples[{i}].ram")
        unused = amount(sample.get("cgroup_unused_gib"), f"samples[{i}].cgroup")
        gpu = sample.get("gpu_free_mib")
        if (not isinstance(gpu, list) or len(gpu) != 2
                or any(type(v) is not int or v < 0 for v in gpu)
                or ram < 7 or unused < 10):
            raise ValueError("unqualified/malformed sampled GPU/RAM/cgroup runtime")
        # Original pinned notebooks mark startup observations with seconds and
        # actual postload inference watchdog observations with UTC timestamp.
        # A completed full run with only startup samples has no attested
        # continuous scoring-time monitor and must not be called monitored.
        if "time_utc" in sample:
            when = sample["time_utc"]
            if not isinstance(when, str):
                raise ValueError("native runtime watchdog timestamp is not a string")
            try:
                parsed = datetime.fromisoformat(when)
            except ValueError as exc:
                raise ValueError("native runtime watchdog timestamp malformed") from exc
            if parsed.utcoffset() != timedelta(0):
                raise ValueError("native runtime watchdog timestamp is not UTC")
            if last_runtime_stamp is not None:
                gap = (parsed - last_runtime_stamp).total_seconds()
                if not 0 < gap <= 15:
                    raise ValueError("native scoring watchdog gap, ordering or sampling cadence invalid")
                max_runtime_gap = max(max_runtime_gap, gap)
            else:
                first_runtime_stamp = parsed
            last_runtime_stamp = parsed
            boot_ended = True
            runtime_samples += 1
        elif ("seconds" not in sample or
              type(sample["seconds"]) not in (int, float) or
              not math.isfinite(sample["seconds"]) or sample["seconds"] < 0):
            raise ValueError("original startup sample has neither boot nor runtime time witness")
        else:
            seconds = sample["seconds"]
            if boot_ended or seconds < last_boot_seconds:
                raise ValueError("original native boot observation is out of recorded time order")
            last_boot_seconds = seconds
        ram_min = min(ram_min, ram)
        unused_min = min(unused_min, unused)
        gpu_min = [min(left, right) for left, right in zip(gpu_min, gpu)]
    if runtime_samples < 2 or first_runtime_stamp is None or last_runtime_stamp is None:
        raise ValueError("complete suite lacks sustained scoring-time resource watchdog evidence")
    suites = receipt.get("suite_results")
    if not isinstance(suites, dict) or set(suites) != {"gsm8k", "humaneval"}:
        raise ValueError("complete original suite durations unavailable for watchdog coverage")
    expected_elapsed = 0.0
    for suite, count in (("gsm8k", 1319), ("humaneval", 164)):
        summary = suites[suite]
        if not isinstance(summary, dict) or type(summary.get("n")) is not int or summary["n"] != count:
            raise ValueError("original complete suite count missing from watchdog evidence")
        wall = summary.get("suite_elapsed_wall_seconds")
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
            raise ValueError("invalid original full-suite monotonic wall duration")
        expected_elapsed += wall
    measured_span = (last_runtime_stamp - first_runtime_stamp).total_seconds()
    # Original notebooks start their 5-second watchdog immediately before
    # GSM8K and stop it after all HumanEval requests. Compare independently
    # sampled UTC duration to the two measured monotonic serving intervals,
    # allowing 2% cross-clock tolerance but not missing entire sub-cohorts.
    if not (0.98 * expected_elapsed - 2 <= measured_span <=
            1.02 * expected_elapsed + 15):
        raise ValueError("native watchdog does not cover measured full suite duration")
    return {
        "status": "COMPLETE_LINUX_SOURCE_MEASURED_POSTLOAD_RESOURCE_WITNESS",
        "initial_actual_gpu_free_mib": initial["gpu"],
        "measured_postload_recovery_gib": reported,
        "measured_file_cache_decrease_gib": round(file_pre["file"] - file_post["file"], 3),
        "minimum_measured_cgroup_unused_gib": unused_min,
        "minimum_measured_physical_available_gib": ram_min,
        "monitored_sample_count": len(samples),
        "monitored_runtime_sample_count": runtime_samples,
        "monitored_runtime_span_seconds": round(measured_span, 3),
        "registered_full_suite_wall_seconds": round(expected_elapsed, 3),
        "maximum_runtime_sample_gap_seconds": round(max_runtime_gap, 3),
        "minimum_observed_gpu_free_mib": gpu_min,
        "largest_sampled_drop_from_start_gpu_free_mib": [
            max(0, original - lowest) for original, lowest in zip(initial["gpu"], gpu_min)
        ],
        "gpu_peak_instantaneous_claimed": False,
        "original_windows_commit_attested": False,
        "model_fit_windows_gate_eligible": False,
    }


def _check_machine(receipt: dict, arm: str) -> None:
    from chowder_batch.reap_eval import FROZEN_V2_QUALITY_SOURCES

    if arm not in MODELS or not isinstance(receipt, dict):
        raise ValueError("only the original registered Q4/Q8/random128 external model is permitted")
    schema, status, key, digest = MODELS[arm]
    if (receipt.get("schema") != schema or receipt.get("status") != status
            or receipt.get("flags") != FROZEN_FLAGS
            or receipt.get("scientific_selection") != "UNKNOWN"
            or receipt.get("not_official_windows_serving_attestation") is not True
            or not isinstance(receipt.get("session_id"), str)
            or not re.fullmatch(r"[0-9a-f]{32}", receipt["session_id"])):
        raise ValueError("unqualified/aborted/model-mixed/fabricated original external session")
    data = receipt.get("data")
    sources = receipt.get("source_sha256")
    if (not isinstance(data, dict) or data.get(key) != digest
            or data.get("server_binary_sha256") != SERVER_SHA
            or not isinstance(sources, dict)
            or sources.get("eval") != EVALUATOR_SHA
            or sources.get("score") != SCORER_SHA
            or any(sources.get(suite + "_full") != source_sha
                   for suite, (_, source_sha) in FROZEN_V2_QUALITY_SOURCES.items())):
        raise ValueError("frozen original model, evaluator or canonical input source SHA mismatch")
    if not isinstance(receipt.get("suite_results"), dict):
        raise ValueError("no complete external suite results in original serving session")
    if arm in ("reap128_v2_q8", "random128_v2_q8"):
        _check_postload_resource_evidence(receipt)


def _number(value: object, label: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError(f"nonfinite, negative or invalid {label}")


def _check_rows(suite: str, source: list[dict], rows: list[dict], session: str) -> dict:
    from chowder.evaluators.scoring import score
    from chowder_batch.reap_eval import _trigram_ratio

    if suite not in ("gsm8k", "humaneval") or len(rows) != len(source) or not source:
        raise ValueError("full canonical cohort missing rows")
    ids = {item["id"]: item for item in source}
    if len(ids) != len(source) or len({row.get("id") for row in rows}) != len(rows):
        raise ValueError("duplicate canonical cohort IDs")
    if {row.get("id") for row in rows} != set(ids):
        raise ValueError("missing/foreign cohort ID or wrong population")
    correct = terminated = total_tokens = 0
    for row in rows:
        if (row.get("eval_session_id") != session
                or type(row.get("correct")) is not bool
                or type(row.get("terminated")) is not bool
                or not isinstance(row.get("text"), str)
                or type(row.get("completion_tokens")) is not int):
            raise ValueError("stale session or impossible scored row fields")
        for field in ("completion_tokens", "seconds", "predicted_per_second", "trigram_ratio"):
            _number(row.get(field), field, nullable=field == "predicted_per_second")
        if not 0 <= row["trigram_ratio"] <= 1:
            raise ValueError("out-of-range trigram ratio")
        if row["trigram_ratio"] != round(_trigram_ratio(row["text"]), 4):
            raise ValueError("source output degeneration metric does not recompute")
        if row["correct"] and not row["terminated"]:
            raise ValueError("source evaluator cannot score an unterminated answer correct")
        if suite == "gsm8k":
            expected = bool(row["terminated"] and score(
                row["text"], ids[row["id"]]["expected"], "final_number_match") == 1.0)
            if row["correct"] != expected:
                raise ValueError("GSM8K original final-answer scorer cannot reproduce saved row")
        correct += row["correct"]
        terminated += row["terminated"]
        total_tokens += row["completion_tokens"]
    return {"n": len(rows), "correct": correct, "terminated": terminated,
            "accuracy": correct / len(rows), "completion_tokens": total_tokens}


def _strict_json(raw: bytes) -> object:
    """Reject duplicate keys and unsupported numeric constants in evidence."""
    def unique_pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON evidence key")
            result[key] = value
        return result

    def refuse_constant(value):
        raise ValueError("invalid JSON numeric constant: " + value)

    return json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=refuse_constant)


def _read_json(path: Path, *, limit: int) -> tuple[dict, str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("external private evidence file missing, linked or unbounded")
    raw = path.read_bytes()
    obj = _strict_json(raw)
    if not isinstance(obj, dict):
        raise ValueError("external provenance JSON is not an object")
    return obj, hashlib.sha256(raw).hexdigest()


def _read_jsonl(path: Path) -> tuple[list[dict], str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("unbounded/missing/symlink original scored cohort")
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("unflushed original output JSONL")
    rows = [_strict_json(line) for line in raw.splitlines()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("original scored rows must be structured JSON objects")
    return rows, hashlib.sha256(raw).hexdigest()


def _check_original_native_log(receipt: dict, log_bytes: bytes) -> dict:
    """Verify original native CUDA placement and frozen four-way KV geometry."""
    expected = receipt.get("server_log_sha256")
    if (not isinstance(log_bytes, bytes) or len(log_bytes) > MAX_ORIGINAL_SERVER_LOG_BYTES
            or not re.fullmatch(r"[0-9a-f]{64}", str(expected))
            or expected != hashlib.sha256(log_bytes).hexdigest()
            or receipt.get("server_log_bytes") != len(log_bytes)):
        raise ValueError("original native server log byte length/hash mismatch")
    text = log_bytes.decode("utf8", errors="replace")
    required = (
        r"load_tensors:\s+offloaded\s+41/41\s+layers to GPU",
        r"llama_context:\s+n_ctx_seq_max\s*=\s*4|llama_context:\s+n_seq_max\s*=\s*4",
        r"llama_context:\s+n_ctx\s*=\s*16384",
        r"llama_context:\s+n_ctx_seq\s*=\s*4096",
        r"llama_kv_cache:\s+size\s*=\s*320\.00\s+MiB",
        r"llama_server:\s+model loaded",
        r"llama_server:\s+listening on http://127\.0\.0\.1:8790",
    )
    if any(re.search(pattern, text) is None for pattern in required):
        raise ValueError("pinned native CUDA offload/KV/loopback topology missing from original log")
    return {"sha256": expected, "native_gpu_offloaded_layers": "41/41",
            "n_seq_max": 4, "n_ctx": 16384, "n_ctx_seq": 4096,
            "total_kv_buffer_mib": 320.0, "loopback_bound": True}


def audit(machine: Path, gsm8k: Path, humaneval: Path, server_log: Path, arm: str,
          independent_humaneval: Path | None = None) -> dict:
    from chowder_batch.reap_eval import FROZEN_V2_QUALITY_SOURCES, _digest, _load

    receipt, receipt_sha = _read_json(machine, limit=2 * 1024 * 1024)
    _check_machine(receipt, arm)
    resource_witness = (_check_postload_resource_evidence(receipt)
                        if arm in ("reap128_v2_q8", "random128_v2_q8") else None)
    if (server_log.is_symlink() or not server_log.is_file()
            or server_log.stat().st_size > MAX_ORIGINAL_SERVER_LOG_BYTES):
        raise ValueError("original complete native server log missing or unbounded")
    native_placement = _check_original_native_log(receipt, server_log.read_bytes())
    if set(receipt["suite_results"]) != {"gsm8k", "humaneval"}:
        raise ValueError("no complete, unique full-suite machine summary")
    session = receipt["session_id"]
    verified = {}
    for suite, path in (("gsm8k", gsm8k), ("humaneval", humaneval)):
        original = _load(suite)
        expected_n, original_sha = FROZEN_V2_QUALITY_SOURCES[suite]
        if len(original) != expected_n or _digest(original) != original_sha:
            raise ValueError("locally loaded frozen input population does not match preregistration")
        rows, row_sha = _read_jsonl(path)
        result = _check_rows(suite, original, rows, session)
        summary = receipt["suite_results"][suite]
        if (not isinstance(summary, dict) or summary.get("file_sha256") != row_sha
                or summary.get("session_id") != session or summary.get("n") != expected_n):
            raise ValueError("original complete session summary and byte-level rows disagree")
        for field, calculated in (("accuracy", result["accuracy"]),
                                  ("terminated_rate", result["terminated"] / expected_n),
                                  ("mean_completion_tokens", result["completion_tokens"] / expected_n)):
            _number(summary.get(field), field)
            if not math.isclose(summary[field], calculated, abs_tol=1e-10, rel_tol=0):
                raise ValueError(f"original external suite summary {field} fails independent replay")
        wall = summary.get("suite_elapsed_wall_seconds")
        _number(wall, "actual suite wall seconds")
        if wall <= 0:
            raise ValueError("no measured complete-cohort wall time")
        verified[suite] = {**result, "row_sha256": row_sha,
                           "suite_wall_seconds": wall,
                           "generated_tokens_per_wall_second": result["completion_tokens"] / wall}
    replay = "PENDING_INDEPENDENT_LOW_PRIVILEGE_CODE_EXECUTION"
    if independent_humaneval is not None:
        independent, replay_sha = _read_json(independent_humaneval, limit=2 * 1024 * 1024)
        if (independent.get("status") != "PASS_INDEPENDENT_ISOLATED_HUMANEVAL_FULL_164_PROVISIONAL_LINUX"
                or independent.get("original_row_sha256") != verified["humaneval"]["row_sha256"]
                or independent.get("session_id") != session
                or independent.get("independent_full_rows") != 164
                or independent.get("matched_recorded_scores") != 164
                or independent.get("correct") != verified["humaneval"]["correct"]):
            raise ValueError("independent HumanEval re-execution outcome/session/row hash mismatch")
        replay = f"PASS_FULL_164_LOW_PRIVILEGE_REEXECUTION_SHA256:{replay_sha}"
    return {
        "schema": "chowder.reap-v2-original-population-external-independent-audit.v1",
        "status": "COMPLETE_LINUX_COHORT_VERIFIED_PROVISIONAL_NOT_WINDOWS_GATE",
        "model": arm,
        "model_sha256": MODELS[arm][3],
        "session_id": session,
        "original_machine_receipt_sha256": receipt_sha,
        "original_native_cuda_placement": native_placement,
        "original_measured_postload_resource_witness": resource_witness,
        "human_eval_independent_execution": replay,
        "suites": verified,
        "original_windows_gate_eligible": False,
        "speed_ratio_certified": False,
        "gen0_status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", type=Path, required=True)
    parser.add_argument("--gsm8k", type=Path, required=True)
    parser.add_argument("--humaneval", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--independent-humaneval", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.is_symlink() or args.out.exists() or not args.out.parent.is_dir():
            raise ValueError("original external evidence audit output must be a new exclusive path")
        result = audit(args.machine, args.gsm8k, args.humaneval, args.server_log,
                       args.model, args.independent_humaneval)
        with args.out.open("x", encoding="utf8", newline="\n") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": result["status"], "model": result["model"],
                          "session_id": result["session_id"],
                          "suites": result["suites"],
                          "human_eval_independent_execution": result["human_eval_independent_execution"]}))
        return 0
    except (ValueError, OSError, TypeError, KeyError, UnicodeError) as exc:
        print(f"external original-frozen cohort audit refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
