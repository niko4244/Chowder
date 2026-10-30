"""Independent, read-only cross-check of original REAP128_v2 frozen evidence.

Checks actual canonical GSM8K completions against the unchanged production
scorer and cross-binds source/model/session/receipt hashes. Previously completed
BF16 all-26-shard hashing is *identified*, not repeated; the source shards are
not opened here. Research evidence never supplies a Gen0 selection verdict.

The CLI opens only existing files. No native server, GPU, sandbox, Podman,
generated code, process, or service is ever started by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

from chowder_batch import reap_eval
from chowder_batch import reap_bf16_cuda_all40_research_20261027 as bf16
from chowder_batch import reap_external_full_audit as external
from chowder_batch import reap_q4_windows_benchmark_watchdog_20261026 as watchdog
from chowder_batch import reap_serve

SCHEMA = "chowder.reap128-v2-cross-original-source-receipt-worker12.v1"
N = 1319
Q4 = watchdog.MODEL_SHA
Q8 = external.Q8_SHA
SOURCE_SHA = reap_eval.FROZEN_V2_QUALITY_SOURCES["gsm8k"][1]
FULL = Path("F:/reap-v2-q4-windows-full1319-phase34-original-20261027-0152")
FIRST = Path("F:/reap-v2-q4-windows-200-20261026-phase33-original")
LINUX_Q4 = Path("F:/reap-v2-kaggle-q4-full-complete-private-evidence-20261026")
LINUX_Q8 = Path("F:/reap-v2-q8-postload-full-complete-original-evidence-20261026-phase4")
LINUX_Q4_AUDIT = Path("F:/reap-v2-sourcebound-q4-with-native-gpu-log-audit-20261026.json")
LINUX_Q8_AUDIT = Path("F:/reap-v2-q8-postload-original-independent-source-audit-20261026-phase4.json")
LINUX_PAIR = Path("F:/reap-v2-original-full-q8-vs-q4-gsm8k-paired-descriptive-20261026-phase4.json")
BF16_ROOT = Path("F:/reap-v2-bf16-all40-measured-20261027-reversible-lease-v1")
BF16_AUDIT = Path("F:/reap-v2-original-bf16-all40-independently-audited-20261027-0840.json")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _no_constant(token: str) -> None:
    raise ValueError(f"nonfinite JSON value: {token}")


def _read(path: Path, limit: int = 64 * 1024 * 1024) -> bytes:
    if (path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= limit):
        raise ValueError(f"missing/unsafe/unbounded original receipt: {path}")
    return path.read_bytes()


def _json(raw: bytes) -> dict:
    value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_no_constant)
    if type(value) is not dict:
        raise ValueError("original receipt root must be an object")
    return value


def _doc(path: Path, *, limit: int = 2 * 1024 * 1024) -> tuple[dict, str]:
    raw = _read(path, limit)
    return _json(raw), _sha(raw)


def canonical_gsm8k() -> tuple[list[dict], str]:
    """Read the same cached frozen full original dataset, without inference."""
    source = reap_eval._load("gsm8k")
    sha = reap_eval._digest(source)
    if (len(source) != N or sha != SOURCE_SHA
            or {x["id"] for x in source} != {f"gsm8k-{i}" for i in range(N)}):
        raise ValueError("original frozen GSM8K source population changed")
    return source, sha


def historical_resource_samples(raw: bytes, count: int) -> dict:
    """Re-derive original sampled host headroom, without probing live devices."""
    if not 0 < len(raw) <= 4 * 1024 * 1024 or not raw.endswith(b"\n"):
        raise ValueError("original historical resource samples incomplete")
    names = ("ram_available_gib", "commit_available_gib", "disk_free_gib", "gpu_free_mib")
    minimum = {name: math.inf for name in names}
    last_time = None
    lines = raw.splitlines()
    if len(lines) != count or count < 2:
        raise ValueError("original historical resource samples count mismatch")
    for line in lines:
        row = _json(line)
        if set(row) != {*names, "time_utc"}:
            raise ValueError("original host sample schema drift")
        try:
            stamp = datetime.fromisoformat(row["time_utc"])
        except (TypeError, ValueError) as exc:
            raise ValueError("original host sample UTC timestamp missing") from exc
        if stamp.utcoffset() is None or (last_time is not None and stamp <= last_time):
            raise ValueError("original historical sample clock out of order")
        last_time = stamp
        for name in names:
            x = row[name]
            if type(x) not in (int, float) or not math.isfinite(x) or x < 0:
                raise ValueError("original historical resource metric invalid")
            minimum[name] = min(minimum[name], x)
    if (minimum["ram_available_gib"] < 7
            or minimum["commit_available_gib"] < 8
            or minimum["disk_free_gib"] < 8):
        raise ValueError("original Q4 native continuous RAM/commit/disk floor violated")
    return {"sample_count": count, "minimum_host_available": minimum,
            "host_gpu_is_not_independent_model_allocation": True,
            "status": "PASS_HISTORICAL_SAMPLED_RAM_COMMIT_DISK_FLOORS"}


def score_rows(raw: bytes, source: list[dict], *, count: int, session: str | None) -> dict:
    """Re-score original text, EOS and trigram using the original readout."""
    if not 0 < len(raw) <= 64 * 1024 * 1024 or not raw.endswith(b"\n"):
        raise ValueError("original GSM8K row stream missing/truncated/oversized")
    corpus = {item["id"]: item for item in source}
    if len(corpus) != N or len(source) != N:
        raise ValueError("frozen GSM8K answer corpus incomplete")
    ledger: dict[str, dict] = {}
    correct = terminated = tokens = degenerate = 0
    speeds: list[float] = []
    for line in raw.splitlines():
        row = _json(line)
        ident = row.get("id")
        if (type(ident) is not str or ident not in corpus or ident in ledger
                or type(row.get("correct")) is not bool
                or type(row.get("terminated")) is not bool
                or type(row.get("text")) is not str
                or type(row.get("completion_tokens")) is not int
                or row["completion_tokens"] < 0):
            raise ValueError("original row missing/duplicated/invalid")
        expected_keys = {"id", "correct", "terminated", "trigram_ratio",
                         "completion_tokens", "seconds", "predicted_per_second", "text"}
        if session is not None:
            expected_keys.add("eval_session_id")
        if set(row) != expected_keys or (session is not None and row["eval_session_id"] != session):
            raise ValueError("original row field/session shape mismatch")
        for key in ("seconds", "predicted_per_second", "trigram_ratio"):
            v = row[key]
            if type(v) not in (float, int) or not math.isfinite(v) or v < 0:
                raise ValueError("nonfinite/negative original row metric")
        recalculated = reap_eval._score("gsm8k", corpus[ident], {
            "finish_reason": "stop" if row["terminated"] else "length",
            "text": row["text"], "completion_tokens": row["completion_tokens"],
            "seconds": row["seconds"],
            "timings": {"predicted_per_second": row["predicted_per_second"]},
        })
        if any(recalculated[k] != row[k] for k in
               ("id", "correct", "terminated", "trigram_ratio")):
            raise ValueError(f"original frozen scorer disagrees on {ident}")
        ledger[ident] = row
        correct += int(row["correct"])
        terminated += int(row["terminated"])
        tokens += row["completion_tokens"]
        degenerate += int(row["trigram_ratio"] < 0.5)
        speeds.append(row["predicted_per_second"])
    if len(ledger) != count:
        raise ValueError("original GSM8K cohort incomplete")
    expected_ids = set(corpus) if count == N else {f"gsm8k-{i}" for i in range(count)}
    if set(ledger) != expected_ids:
        raise ValueError("original GSM8K cohort ID coverage changed")
    return {"count": count, "correct": correct, "terminated": terminated,
            "completion_tokens": tokens, "degenerate": degenerate,
            "rows_sha256": _sha(raw), "mean_item_predicted_per_second": sum(speeds) / count,
            "by_id": ledger}


def _windows(arm: Path, source: list[dict], *, count: int) -> dict:
    serving, serving_sha = _doc(arm / "serve.json")
    identity, _ = _doc(arm / "gsm8k.inputs.json")
    summary, _ = _doc(arm / "summary.json")
    if (serving.get("gguf_sha256") != Q4
            or serving.get("flags") != reap_serve.FLAGS
            or serving.get("measured_experts_offloaded") != 0
            or serving.get("server_port") != watchdog.PORT
            or type(serving.get("server_pid")) is not int
            or not isinstance(serving.get("eval_session_id"), str)
            or identity.get("source_items_sha256") != SOURCE_SHA
            or identity.get("source_items_n") != N
            or identity.get("suite") != "gsm8k"
            or identity.get("model") != {k: serving[k] for k in ("gguf", "gguf_sha256", "flags")}):
        raise ValueError("Windows Q4 native serving or frozen input witness changed")
    reap_serve._verify_offload_receipt(serving, arm / "server.log")
    scored = score_rows(_read(arm / "gsm8k.jsonl"), source, count=count, session=None)
    result = summary.get("gsm8k")
    if (type(result) is not dict or result.get("n") != count
            or result.get("file_sha256") != scored["rows_sha256"]
            or result.get("accuracy") != scored["correct"] / count
            or result.get("terminated_rate") != scored["terminated"] / count
            or result.get("degenerate_rate") != scored["degenerate"] / count
            or result.get("mean_completion_tokens") != scored["completion_tokens"] / count):
        raise ValueError("Windows Q4 native summary differs from source-rescored rows")
    return {**{k: v for k, v in scored.items() if k != "by_id"},
            "session_id": serving["eval_session_id"], "serving_sha256": serving_sha,
            "native_offload": "41/41 CUDA, zero CPU expert offload; native startup log verified"}


def audit_windows(source: list[dict]) -> dict:
    full = _windows(FULL / "reap128_v2_q4", source, count=N)
    terminal, terminal_sha = _doc(FULL / "full1319-terminal.json")
    full_resources = historical_resource_samples(
        _read(FULL / "resource-samples.jsonl", 4 * 1024 * 1024), terminal.get("sample_count"))
    original = terminal.get("source_bound_scoring")
    if (terminal.get("status") != "COMPLETE_ORIGINAL_WINDOWS_Q4_FULL1319_CANDIDATE_ONLY"
            or terminal.get("child_exit_code") != 0
            or terminal.get("native_shutdown_verified") is not True
            or terminal.get("original_model_sha256") != Q4
            or terminal.get("frozen_flags") != reap_serve.FLAGS
            or terminal.get("gen0_status") != "BLOCKED"
            or terminal.get("chosen", "missing") is not None
            or terminal.get("release_approved") is not False
            or not isinstance(original, dict)
            or any(original.get(k) != full[v] for k, v in
                   (("population", "count"), ("correct", "correct"),
                    ("naturally_terminated", "terminated"),
                    ("completion_tokens", "completion_tokens"),
                    ("degenerate", "degenerate"), ("rows_sha256", "rows_sha256"),
                    ("serving_receipt_sha256", "serving_sha256")))
            or original.get("source_sha256") != SOURCE_SHA):
        raise ValueError("original complete Windows 1319 terminal/serving/rows disagree")
    short = _windows(FIRST / "original-first200" / "reap128_v2_q4", source, count=200)
    watchdog_receipt, _ = _doc(FIRST / "original-first200" / "watchdog-terminal.json")
    rescored, _ = _doc(FIRST / "independent-rescored-first200.private.json")
    first_resources = historical_resource_samples(
        _read(FIRST / "original-first200" / "resource-samples.jsonl", 4 * 1024 * 1024),
        watchdog_receipt.get("sample_count"))
    if (rescored.get("minimum_available") != first_resources["minimum_host_available"]):
        raise ValueError("first200 native min observed resource summary does not match samples")
    speed = watchdog_receipt.get("source_bound_speed_session")
    if (watchdog_receipt.get("child_exit_code") != 0
            or watchdog_receipt.get("native_shutdown_verified") is not True
            or watchdog_receipt.get("partial_or_complete_rows") != 200
            or watchdog_receipt.get("status") != "COMPLETE_SOURCE_BOUND_200_CANDIDATE_ONLY"
            or not isinstance(speed, dict)
            or speed.get("session_id") != short["session_id"]
            or speed.get("rows_sha256") != short["rows_sha256"]
            or speed.get("gguf_sha256") != Q4
            or speed.get("n") != 200
            or rescored.get("first200_rows_sha256") != short["rows_sha256"]
            or rescored.get("original_source_sha256") != SOURCE_SHA
            or rescored.get("correct") != short["correct"]
            or rescored.get("completion_tokens") != short["completion_tokens"]
            or rescored.get("natural_termination") != short["terminated"]
            or not math.isclose(rescored.get("mean_item_output_tokens_per_second", -1),
                                short["mean_item_predicted_per_second"], abs_tol=1e-9)
            or rescored.get("dense_first200_matched_baseline") != "ABSENT"
            or rescored.get("gen0_status") != "BLOCKED"):
        raise ValueError("Q4 first200 same-session throughput provenance/rescore differs")
    return {
        "status": "PASS_SOURCE_BOUND_WINDOWS_CANDIDATE_ONLY",
        "original_q4_gguf_sha256_claim": Q4,
        "full1319": {k: v for k, v in full.items() if k != "mean_item_predicted_per_second"},
        "terminal_sha256": terminal_sha,
        "full1319_historical_host_resource_samples": full_resources,
        "first200": {k: v for k, v in short.items() if k != "native_offload"},
        "first200_historical_host_resource_samples": first_resources,
        "dense_paired_native_first200_speed_gate": "UNKNOWN_MISSING_DENSE_SOURCE_SESSION",
        "original_frozen_gen0": "BLOCKED",
    }


def audit_linux(source: list[dict]) -> dict:
    pair, pair_sha = _doc(LINUX_PAIR)
    if (pair.get("schema") != "reap-v2-original-external-gsm8k-paired-private-descriptive.v1"
            or pair.get("status") != "SOURCE_BOUND_COMPLETE_LINUX_GSM8K_PAIRED_DESCRIPTIVE_ONLY"
            or pair.get("gen0_status") != "BLOCKED"
            or pair.get("chosen", "missing") is not None
            or pair.get("release_approved") is not False):
        raise ValueError("original external paired publication status changed")
    out = {}
    sides = {
        "q4": (LINUX_Q4_AUDIT, LINUX_Q4 / "reap-v2-a2a6b83e345145839cc333561b2f3baf-gsm8k.jsonl",
               "reap128_v2_q4", Q4),
        "q8": (LINUX_Q8_AUDIT, LINUX_Q8 / "reap-q8-reclaim-full-95876063255e45159f57e29306e0e885-gsm8k.jsonl",
               "reap128_v2_q8", Q8),
    }
    for arm, (receipt_path, rows_path, model, sha) in sides.items():
        audit, audit_sha = _doc(receipt_path)
        if (audit.get("schema") != "chowder.reap-v2-original-population-external-independent-audit.v1"
                or audit.get("status") != "COMPLETE_LINUX_COHORT_VERIFIED_PROVISIONAL_NOT_WINDOWS_GATE"
                or audit.get("model") != model or audit.get("model_sha256") != sha
                or pair.get("audit_sha256", {}).get(arm) != audit_sha
                or pair.get("source_row_sha256", {}).get(arm) != audit.get("suites", {}).get("gsm8k", {}).get("row_sha256")):
            raise ValueError("original Linux external model/audit/source receipt differs")
        scored = score_rows(_read(rows_path), source, count=N, session=audit.get("session_id"))
        results = audit["suites"]["gsm8k"]
        if (results.get("correct") != scored["correct"]
                or results.get("n") != N
                or results.get("terminated") != scored["terminated"]
                or results.get("completion_tokens") != scored["completion_tokens"]
                or results.get("accuracy") != scored["correct"] / N
                or results.get("row_sha256") != scored["rows_sha256"]):
            raise ValueError("original Linux audit differs from independently rescored rows")
        out[arm] = {"scored": scored, "audit_sha256": audit_sha, "model_sha256_claim": sha,
                    "session_id": audit["session_id"]}
    counts = {"both_correct": 0, "both_wrong": 0, "candidate_only_correct": 0,
              "reference_only_correct": 0}
    for ident in out["q8"]["scored"]["by_id"]:
        q4 = out["q4"]["scored"]["by_id"][ident]["correct"]
        q8 = out["q8"]["scored"]["by_id"][ident]["correct"]
        key = ("both_correct" if q4 and q8 else "both_wrong" if not q4 and not q8
               else "candidate_only_correct" if q8 else "reference_only_correct")
        counts[key] += 1
    published = pair.get("gsm8k")
    if (not isinstance(published, dict)
            or any(published.get(k) != v for k, v in counts.items())
            or published.get("n") != N
            or published.get("candidate_session") != out["q8"]["session_id"]
            or published.get("reference_session") != out["q4"]["session_id"]
            or published.get("delta_accuracy") != (counts["candidate_only_correct"] - counts["reference_only_correct"]) / N
            or published.get("paired_confidence_interval") is not None
            or published.get("amendment4_eligibility") is not False
            or published.get("speed_ratio_certified") is not False):
        raise ValueError("original source-paired Linux descriptive result differs")
    return {"status": "PASS_LINUX_RESEARCH_DESCRIPTIVE_ONLY",
            "q4_correct": out["q4"]["scored"]["correct"],
            "q8_correct": out["q8"]["scored"]["correct"],
            "q4_model_sha256_claim": Q4, "q8_model_sha256_claim": Q8,
            "q4_rows_sha256": out["q4"]["scored"]["rows_sha256"],
            "q8_rows_sha256": out["q8"]["scored"]["rows_sha256"],
            "paired_counts": counts, "paired_receipt_sha256": pair_sha,
            "native_windows_q8_vs_q4_paired_gate": "UNKNOWN_DIFFERENT_HOST_AND_PROTOCOL",
            "native_paired_throughput_gate": "UNKNOWN_NO_MATCHED_NATIVE_SESSIONS"}


def audit_bf16_receipts() -> dict:
    summary, summary_sha = _doc(BF16_ROOT / "all40-summary.json")
    prior, prior_sha = _doc(BF16_AUDIT)
    if (prior.get("status") != "SOURCE_BOUND_40_LAYER_RESEARCH_RECEIPTS_INDEPENDENTLY_VERIFIED"
            or prior.get("all40_summary_sha256") != summary_sha
            or prior.get("unique_source_shards_rehashed") != 26
            or prior.get("original_40_layers") != 40
            or prior.get("frozen_numerical_gate") != "UNKNOWN"
            or summary.get("status") != "ALL40_SOURCE_BOUND_LAYER_NUMERICS_RESEARCH_ONLY"
            or summary.get("distinct_original_layer_receipts") != 40
            or summary.get("real_cpu_cuda_bf16_compared_elements") != 163840
            or summary.get("entire_model_forward_or_logit_parity_proven") is not False
            or summary.get("gen0_status") != "BLOCKED"
            or summary.get("release_approved") is not False):
        raise ValueError("BF16 original prior independently rehashed audit or summary drift")
    model = Path(summary["model"])
    pinned = summary.get("source_bound_private_receipt_sha256")
    if type(pinned) is not dict or set(pinned) != {str(i) for i in range(40)}:
        raise ValueError("incomplete original BF16 numerical receipt hashes")
    exact = 0
    max_abs = 0.0
    for idx in range(40):
        if idx in bf16.ORIGINAL_LAYER_RECEIPTS:
            path, fixed = bf16.ORIGINAL_LAYER_RECEIPTS[idx]
            if pinned[str(idx)] != fixed:
                raise ValueError("reused historical BF16 layer pin drift")
            receipt_path = Path(path)
        else:
            receipt_path = BF16_ROOT / f"layer-{idx:02d}.json"
        raw = _read(receipt_path, 256 * 1024)
        if _sha(raw) != pinned[str(idx)] or prior.get("per_layer_receipt_sha256", {}).get(str(idx)) != pinned[str(idx)]:
            raise ValueError("original BF16 layer receipt hash differs")
        row = bf16.verify_layer_receipt(raw, model=model, layer=idx)
        if row != summary["per_layer"][idx]:
            raise ValueError("original BF16 layer arithmetic/identity differs")
        exact += row["exact_equal_elements"]
        max_abs = max(max_abs, row["max_abs_error"])
    if (exact != summary.get("exact_equal_elements")
            or max_abs != summary.get("max_absolute_difference")
            or prior.get("exact_equal_elements") != exact
            or prior.get("max_absolute_difference") != max_abs):
        raise ValueError("original BF16 layer-sum arithmetic differs")
    return {"status": "PASS_EXISTING_40_LAYER_RECEIPTS_NOT_NEW_SHARD_REHASH",
            "original_prior_26_shard_rehash_receipt_sha256": prior_sha,
            "original_full40_summary_sha256": summary_sha,
            "verified_layer_receipts": 40, "compared_elements": 163840,
            "exact_equal_elements": exact, "max_abs_difference": max_abs,
            "full_model_chained_logit_parity": "UNKNOWN",
            "native_gen0_numerical_gate": "UNKNOWN"}


def audit_all() -> dict:
    source, source_sha = canonical_gsm8k()
    return {"schema": SCHEMA,
            "source_sha256": source_sha,
            "source_population": len(source),
            "original_model_file_bytes_rehashed_now": False,
            "original_bf16_26_shards_rehashed_now": False,
            "windows_q4": audit_windows(source),
            "linux_q8_q4": audit_linux(source),
            "bf16": audit_bf16_receipts(),
            "gen0_status": "BLOCKED", "chosen": None,
            "release_approved": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="new exclusive JSON receipt path (optional)")
    args = parser.parse_args()
    try:
        result = audit_all()
        encoded = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        if args.out is not None:
            if args.out.exists() or args.out.is_symlink() or not args.out.parent.is_dir():
                raise ValueError("audit output must be a new exclusive file")
            with args.out.open("xb") as target:
                target.write(encoded)
        print(encoded.decode(), end="")
        return 0
    except (OSError, KeyError, ValueError, TypeError, RuntimeError) as exc:
        print(json.dumps({"schema": SCHEMA, "status": "UNKNOWN_SOURCE_OR_EVIDENCE_DRIFT",
                          "reason": str(exc), "gen0_status": "BLOCKED",
                          "release_approved": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
