"""Re-score the ORIGINAL sealed Windows Q4 first-200 GSM8K run without GPU.

This is read-only for the historical model/evaluator receipts. It never
executes model-generated Python, launches a server, or modifies the original
selection. Only a new exclusive PRIVATE output file may be written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from chowder_batch import reap_eval, reap_gates, reap_serve

MODEL_SHA = "2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7"
SESSION_SCHEMA = "chowder.reap-q4-windows-monitored-first200.v1"
ORIGINAL_FLAGS = ["--fit", "on", "-np", "4", "-c", "16384", "--jinja"]
ORIGINAL_ARM = "reap128_v2_q4"
EXPECTED_IDS = {f"gsm8k-{i}" for i in range(200)}
FLOORS = {"ram_available_gib": 7.0, "commit_available_gib": 8.0,
          "disk_free_gib": 8.0}


def _load_json(path: Path, *, max_bytes: int = 4 * 1024 * 1024) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError(f"missing/unsafe original private receipt: {path.name}")
    parsed = json.loads(path.read_bytes(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if type(parsed) is not dict:
        raise ValueError("original receipt must be an object")
    return parsed


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_rows(arm: Path, *, expected_sha256: str) -> list[dict]:
    path = arm / "gsm8k.jsonl"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("unbounded or missing original private GSM8K row file")
    raw = path.read_bytes()
    if _sha(path) != expected_sha256 or not raw.endswith(b"\n"):
        raise ValueError("sealed original GSM8K result hash/final newline mismatch")
    rows = [json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
            for line in raw.splitlines()]
    if len(rows) != 200 or {r.get("id") for r in rows} != EXPECTED_IDS:
        raise ValueError("frozen first200 GSM8K source IDs incomplete/substituted")
    for row in rows:
        if (type(row) is not dict or set(row) != {"id", "correct", "terminated", "trigram_ratio",
                                                  "completion_tokens", "seconds", "predicted_per_second", "text"}
                or type(row["correct"]) is not bool or type(row["terminated"]) is not bool
                or type(row["text"]) is not str or type(row["completion_tokens"]) is not int
                or row["completion_tokens"] < 0):
            raise ValueError("original GSM8K result row schema invalid")
        for name in ("trigram_ratio", "seconds", "predicted_per_second"):
            value = row[name]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or value < 0):
                raise ValueError("original GSM8K numeric result row invalid")
    return rows


def verify_rows_with_original_scorer(rows: list[dict], items: list[dict]) -> dict:
    """Reproduce the original frozen GSM8K score; never execute generated code."""
    if len(items) != 1319 or {r.get("id") for r in items[:200]} != EXPECTED_IDS:
        raise ValueError("original offline canonical GSM8K source changed")
    if reap_eval._digest(items) != reap_eval.FROZEN_V2_QUALITY_SOURCES["gsm8k"][1]:
        raise ValueError("canonical full GSM8K source item SHA changed")
    source = {item["id"]: item for item in items[:200]}
    matched = 0
    for row in rows:
        observed = reap_eval._score("gsm8k", source[row["id"]], {
            "finish_reason": "stop" if row["terminated"] else "length",
            "text": row["text"], "completion_tokens": row["completion_tokens"],
            "seconds": row["seconds"],
            "timings": {"predicted_per_second": row["predicted_per_second"]},
        })
        if (observed["id"] != row["id"] or observed["correct"] != row["correct"]
                or observed["terminated"] != row["terminated"]
                or observed["trigram_ratio"] != row["trigram_ratio"]):
            raise ValueError("original scorer independently disagrees with saved row")
        matched += int(observed["correct"])
    return {"rescore_correct": matched, "rescore_denominator": 200,
            "original_scorer_agreed_all_rows": True}


def _audit_telemetry(path: Path, *, expected_samples: int) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("original watchdog telemetry missing or unbounded")
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    if len(rows) != expected_samples or not 0 < len(rows) < 100_000:
        raise ValueError("watchdog sample count incomplete/altered")
    minima = {}
    for name, minimum in FLOORS.items():
        numbers = [r.get(name) for r in rows]
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in numbers):
            raise ValueError("watchdog numeric telemetry missing/nonfinite")
        minima[name] = min(numbers)
        if minima[name] < minimum:
            raise ValueError("original watchdog runtime safety floor breached")
    gpu = [r.get("gpu_free_mib") for r in rows]
    if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in gpu):
        raise ValueError("watchdog WDDM GPU telemetry missing/nonfinite")
    minima["gpu_free_mib"] = min(gpu)
    return {"sample_count": len(rows), "minimum_available": minima,
            "resource_sampling_hash_sha256": _sha(path),
            "gpu_measure_is_host_wide_not_isolated_model_peak": True}


def audit(folder: Path, *, items: list[dict] | None = None) -> dict:
    base = folder.resolve(strict=True)
    if not base.is_dir() or folder.is_symlink():
        raise ValueError("original sealed experiment directory missing/symlink")
    arm = base / ORIGINAL_ARM
    terminal = _load_json(base / "watchdog-terminal.json")
    serving = _load_json(arm / "serve.json")
    summary = _load_json(arm / "summary.json")
    proof = serving.get("speed_session")
    if (terminal.get("schema") != SESSION_SCHEMA
            or terminal.get("status") != "COMPLETE_SOURCE_BOUND_200_CANDIDATE_ONLY"
            or terminal.get("native_shutdown_verified") is not True
            or terminal.get("gen0_status") != "BLOCKED"
            or terminal.get("chosen", "unexpected") is not None
            or terminal.get("release_approved") is not False
            or terminal.get("abort_reasons") not in (None, [])
            or type(terminal.get("child_exit_code")) is not int
            or terminal["child_exit_code"] != 0
            or terminal.get("source_bound_speed_session") != proof
            or serving.get("gguf_sha256") != MODEL_SHA
            or serving.get("flags") != ORIGINAL_FLAGS
            or type(serving.get("measured_experts_offloaded")) is not int
            or serving["measured_experts_offloaded"] != 0):
        raise ValueError("source-bound original Windows terminal/session/model contract mismatch")
    if not reap_gates._verify_new_q4_speed_session(base, ORIGINAL_ARM, serving):
        raise ValueError("independent original Windows speed receipt refused")
    if (terminal.get("summary_sha256") != _sha(arm / "summary.json")
            or terminal.get("serve_json_sha256") != _sha(arm / "serve.json")
            or terminal.get("rows_sha256") != _sha(arm / "gsm8k.jsonl")
            or terminal.get("partial_or_complete_rows") != 200):
        raise ValueError("original watchdog evidence hashes or row population changed")
    rows = _strict_rows(arm, expected_sha256=terminal["rows_sha256"])
    rescored = verify_rows_with_original_scorer(rows, items if items is not None else reap_eval._load("gsm8k"))
    scored = summary.get("gsm8k")
    if (type(scored) is not dict or scored.get("n") != 200
            or scored.get("file_sha256") != terminal["rows_sha256"]
            or scored.get("accuracy") != rescored["rescore_correct"] / 200
            or scored.get("terminated_rate") != sum(r["terminated"] for r in rows) / 200):
        raise ValueError("saved GSM8K summary contradicts source-rescored rows")
    telemetry = _audit_telemetry(base / "resource-samples.jsonl",
                                 expected_samples=terminal["sample_count"])
    return {"status": "ORIGINAL_WINDOWS_Q4_FIRST200_INDEPENDENTLY_RESCORED_CANDIDATE_ONLY",
            "arm": ORIGINAL_ARM, "original_q4_gguf_sha256": MODEL_SHA,
            "original_source_sha256": proof["source_items_sha256"],
            "session_id": proof["session_id"], "first200_rows_sha256": terminal["rows_sha256"],
            "correct": rescored["rescore_correct"], "total": 200,
            "accuracy": rescored["rescore_correct"] / 200,
            "natural_termination": sum(r["terminated"] for r in rows),
            "mean_item_output_tokens_per_second": statistics.fmean(r["predicted_per_second"] for r in rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "watchdog_wall_seconds": terminal.get("elapsed_wall_seconds"),
            "model_cpu_expert_offload": 0,
            "native_shutdown_verified": True, "original_speed_receipt_verified": True,
            "independent_frozen_original_scorer_agreement": True,
            **telemetry, "dense_first200_matched_baseline": "ABSENT",
            "human_eval_independent_sandbox": "NOT_PROVEN",
            "gen0_status": "BLOCKED", "chosen": None, "release_approved": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.out.is_absolute() or args.out.exists() or args.out.is_symlink():
        raise ValueError("independent audit requires a fresh exclusive PRIVATE output filename")
    if args.out.parent.is_symlink() or not args.out.parent.is_dir():
        raise ValueError("private audit output parent is not a real existing directory")
    result = audit(args.original)
    with args.out.open("x", encoding="utf8") as out:
        json.dump(result, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps({key: result[key] for key in ("status", "correct", "total", "accuracy",
                                                  "mean_item_output_tokens_per_second",
                                                  "first200_rows_sha256", "gen0_status")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
