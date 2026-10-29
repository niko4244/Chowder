"""Independent CPU-only witness for the completed original Windows Q4 GSM8K1319.

Correlates immutable SHA-pinned F: model-output rows with the independently
archived native server log, and independently regrades every canonical answer.
It never invokes a model, changes F: evidence, or certifies a frozen gate.
Native timing fingerprints do not cryptographically authenticate response text.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from chowder_batch import reap_original_gsm8k_independent_grade_20261029 as grade
from chowder_batch import reap_original_linux_gsm8k_independent_pair_20261029 as pair
from chowder_batch import reap_original_linux_native_log_witness_20261029 as native


ROOT = Path("F:/reap-v2-q4-windows-full1319-phase34-original-20261027-0152")
ARM = ROOT / "reap128_v2_q4"
PINS = {
    "full1319-terminal.json": "fabd6e8381ab68f73473b8a57bd80a94c2d2cf675968b7527fa5a37795bb0697",
    "reap128_v2_q4/gsm8k.jsonl": "66d0f39545dd8769e837ed5772a265871cfe148bb9a529ea2188454d72c1243f",
    "reap128_v2_q4/server.log": "a01acc0de9507de35fc1ab1a715bf6d104010362025862c24c475cb246b0ce4f",
    "reap128_v2_q4/serve.json": "ba677d78f30f33b67a1357c20cccc8d9218d0633e739cab21811912e2a66a7dd",
    "reap128_v2_q4/summary.json": "14781c321bdfe86f92060149dafc4a02ed7d35d1aabd2b850d80e17465128bf3",
    "reap128_v2_q4/gsm8k.inputs.json": "86cdff1a7eecbbcae566862ce4b80e5f1edd6b1e388e31a24528ec7ba0e67da2",
}
LIMITS = {"server.log": 32 * 1024 * 1024, "gsm8k.jsonl": 8 * 1024 * 1024}
MODEL_SHA = "2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7"
SESSION = "535b50372b6f42de8c500b7126dac2d5"
ROW_KEYS = pair.ROW_KEYS - {"eval_session_id"}
EXPECTED_BINDING_SHA = "e7d0561fa4472335d56dc4d9236631e93eefd8b3cd43e45a461241b626e66082"


class WindowsQ4WitnessError(ValueError):
    """Original Windows Q4 source, scoring or native timing evidence differs."""


def regrade_windows_rows(raw: bytes, canonical: tuple) -> dict:
    """Regrade all 1,319 archived responses without production scorer imports."""
    if not raw.endswith(b"\n") or not 0 < len(raw) <= LIMITS["gsm8k.jsonl"]:
        raise WindowsQ4WitnessError("original Windows row archive incomplete")
    rows = {}
    total_tokens = 0
    degenerate = 0
    for line in raw.splitlines():
        row = pair._json(line)
        ident = row.get("id")
        if (set(row) != ROW_KEYS or type(ident) is not str or ident in rows
                or type(row.get("completion_tokens")) is not int
                or row["completion_tokens"] < 2 or type(row.get("text")) is not str
                or type(row.get("terminated")) is not bool
                or type(row.get("correct")) is not bool):
            raise WindowsQ4WitnessError("invalid or duplicate Windows Q4 archived row")
        for key in ("seconds", "predicted_per_second", "trigram_ratio"):
            value = row[key]
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise WindowsQ4WitnessError("nonfinite Windows Q4 archived numeric field")
        if row["predicted_per_second"] <= 0 or row["trigram_ratio"] != pair._trigram(row["text"]):
            raise WindowsQ4WitnessError("Windows Q4 archived timing or trigram differs")
        rows[ident] = row
        total_tokens += row["completion_tokens"]
        degenerate += int(row["trigram_ratio"] < 0.5)
    try:
        scored = grade.verify_rows(rows, canonical)
    except grade.IndependentGradeError as exc:
        raise WindowsQ4WitnessError("Windows Q4 canonical independent regrading failed") from exc
    if (scored["n"] != 1319 or scored["correct"] != 1224
            or scored["naturally_terminated"] != 1281 or total_tokens != 459264
            or degenerate != 0):
        raise WindowsQ4WitnessError("Windows Q4 full original score/tokens disagree")
    return {**scored, "completion_tokens": total_tokens, "degenerate": degenerate}


def validate_original_receipts(receipts: dict[str, dict], log: bytes) -> None:
    """Require exact original model, frozen protocol and complete terminal claim."""
    terminal = receipts["full1319-terminal.json"]
    serve = receipts["serve.json"]
    inputs = receipts["gsm8k.inputs.json"]
    summary = receipts["summary.json"].get("gsm8k")
    scored = terminal.get("source_bound_scoring")
    expected_model = {"gguf": "G:\\reap128-v2-q4_k_m-20261025.gguf",
                      "gguf_sha256": MODEL_SHA, "flags": native.FLAGS}
    if (terminal.get("schema") != "chowder.reap128-v2-q4-windows-full1319.v1"
            or terminal.get("status") != "COMPLETE_ORIGINAL_WINDOWS_Q4_FULL1319_CANDIDATE_ONLY"
            or terminal.get("arm") != "reap128_v2_q4"
            or terminal.get("child_exit_code") != 0
            or terminal.get("native_shutdown_verified") is not True
            or terminal.get("frozen_flags") != native.FLAGS
            or terminal.get("original_model_sha256") != MODEL_SHA
            or terminal.get("gen0_status") != "BLOCKED"
            or terminal.get("chosen", "missing") is not None
            or terminal.get("release_approved") is not False
            or not isinstance(scored, dict)
            or scored.get("rows_sha256") != PINS["reap128_v2_q4/gsm8k.jsonl"]
            or scored.get("serving_receipt_sha256") != PINS["reap128_v2_q4/serve.json"]
            or scored.get("source_sha256") != grade.FROZEN_GSM8K_SHA256
            or scored.get("population") != 1319 or scored.get("correct") != 1224
            or scored.get("naturally_terminated") != 1281
            or scored.get("completion_tokens") != 459264
            or scored.get("degenerate") != 0
            or scored.get("all_rows_rescored") is not True
            or scored.get("native_offload_verified") is not True):
        raise WindowsQ4WitnessError("original Windows Q4 terminal/scorer receipt drift")
    if (not isinstance(serve, dict)
            or any(serve.get(k) != v for k, v in expected_model.items())
            or serve.get("eval_session_id") != SESSION
            or serve.get("server_port") != 8790
            or serve.get("log_verbosity") != "4"
            or serve.get("measured_experts_offloaded") != 0
            or serve.get("server_log_startup_bytes") != 18600
            or serve.get("server_log_startup_sha256") != hashlib.sha256(log[:18600]).hexdigest()
            or not any("offloaded 41/41 layers to GPU" in line
                       for line in serve.get("offload_log_lines", []))
            or b"offloaded 41/41 layers to GPU" not in log):
        raise WindowsQ4WitnessError("original Windows Q4 serving/native source differs")
    if (inputs.get("schema") != "chowder.reap-eval-input-identity.v1"
            or inputs.get("source_items_sha256") != grade.FROZEN_GSM8K_SHA256
            or inputs.get("source_items_n") != 1319
            or inputs.get("suite") != "gsm8k"
            or inputs.get("model") != expected_model
            or inputs.get("decode") != {"enable_thinking": False, "max_tokens": 1024,
                                         "seed": 7, "system": "You are a helpful assistant.",
                                         "temperature": 0.0, "top_k": 1}):
        raise WindowsQ4WitnessError("original Windows Q4 frozen prompt/decode source differs")
    if (not isinstance(summary, dict) or summary.get("n") != 1319
            or summary.get("file_sha256") != PINS["reap128_v2_q4/gsm8k.jsonl"]
            or summary.get("accuracy") != 1224 / 1319
            or summary.get("terminated_rate") != 1281 / 1319
            or summary.get("degenerate_rate") != 0.0
            or summary.get("mean_completion_tokens") != 459264 / 1319):
        raise WindowsQ4WitnessError("original Windows Q4 summary differs")


def verify_original_windows_q4_native() -> dict:
    """Read original pinned F: bytes once, independently regrade and correlate."""
    raw = {}
    for relative, expected_sha in PINS.items():
        name = Path(relative).name
        raw[name] = pair._read_pinned(ROOT / relative, expected_sha,
                                      LIMITS.get(name, 2 * 1024 * 1024))
    receipts = {name: pair._json(raw[name]) for name in
                ("full1319-terminal.json", "serve.json", "gsm8k.inputs.json", "summary.json")}
    validate_original_receipts(receipts, raw["server.log"])
    canonical = grade.load_canonical()
    scored = regrade_windows_rows(raw["gsm8k.jsonl"], canonical)
    parsed = native.parse_native_lifecycles(raw["server.log"], expected_n=1319)
    if (parsed["n"] != 1319 or parsed["native_releases"] != 1319
            or parsed["max_simultaneously_active_slots"] != 4
            or not parsed["cold_first_slot_requests"]
            or parsed["cached_prompt_savings"] != Counter({0: 4, 11: 1315})
            or sum(parsed["new_prompt_tokens"].elements()) != 137469
            or parsed["prompt_eval_tokens"] != 123004
            or sum(parsed["completion_tokens"].elements()) != 459264):
        raise WindowsQ4WitnessError("Windows Q4 native task/cache/token accounting differs")
    binding = native.bind_archived_rows_to_native_timings(
        (raw["gsm8k.jsonl"], b""), parsed["per_task_eval_timing"]
    )
    if binding["n"] != 1319 or binding["mapping_sha256"] != EXPECTED_BINDING_SHA:
        raise WindowsQ4WitnessError("Windows Q4 per-item native timing mapping differs")
    return {
        "schema": "chowder.reap-original-windows-q4-native-per-item-witness.v1",
        "source_sha256": grade.FROZEN_GSM8K_SHA256,
        "original_model_gguf_sha256_claimed": MODEL_SHA,
        "original_windows_session": SESSION,
        "independently_regraded_correct": scored["correct"],
        "independently_regraded_n": scored["n"],
        "naturally_terminated": scored["naturally_terminated"],
        "independently_regraded_degenerate": scored["degenerate"],
        "native_complete_task_lifecycles": parsed["n"],
        "native_slot_releases": parsed["native_releases"],
        "native_generation_tokens": scored["completion_tokens"],
        "per_item_native_timing_matches": binding["n"],
        "per_item_native_timing_mapping_sha256": binding["mapping_sha256"],
        "native_log_sha256": PINS["reap128_v2_q4/server.log"],
        "native_response_text_cryptographically_authenticated": False,
        "model_weights_rehashed_at_original_execution": False,
        "same_session_dense_reference_available": False,
        "frozen_gate1_qualified": False,
        "frozen_gate2_qualified": False,
        "research_only": True,
        "status": "BLOCKED", "chosen": None, "release_approved": False,
    }


def main() -> int:
    try:
        print(json.dumps(verify_original_windows_q4_native(), indent=2, sort_keys=True))
        return 0
    except (WindowsQ4WitnessError, native.NativeWitnessError,
            pair.IndependentPairError, grade.IndependentGradeError,
            OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "REFUSED_ORIGINAL_WINDOWS_Q4_NATIVE_WITNESS",
                          "reason": str(exc)[:300], "release_approved": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
