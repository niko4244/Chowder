"""Read-only original Linux Q4/Q8 native-server request/token witness.

This verifies the *already completed* original research-only 1,319 GSM8K +
164 HumanEval cohorts against separately SHA-pinned native llama-server logs.
It never starts inference, runs generated code, or changes F: evidence.

Native server request counts, slot lifecycles, prompt-token multisets and
completion-token multisets can be corroborated. Native logs do NOT contain
the actual response text or an independently signed request/response chain:
per-item native response attribution and frozen Gen-0 gates remain UNPROVEN.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re

from chowder_batch import reap_original_linux_gsm8k_independent_pair_20261029 as pair
from chowder_batch import reap_original_gsm8k_independent_grade_20261029 as grade


ROOT = Path("F:/")
N = 1319 + 164
FLAGS = ["--fit", "on", "-np", "4", "-c", "16384", "--jinja"]
SOURCE_SHA = {
    "gsm8k_full": grade.FROZEN_GSM8K_SHA256,
    "humaneval_full": "49fb688c6788b0531803dfd1cb85a69926ec5a7179e74cdfffc53ba52a92b420",
    "eval": "9736714f21834a45c23599a50fa92ef3bebc6cf70967aca7e2329d26b4d73ee1",
    "score": "d2e59aca5c0a621bac8d4979d522c9b6107f811610567d21c92a2b53960860eb",
}
ARMS = {
    "q4": {
        "log": ROOT / "reap-v2-kaggle-q4-full-native-log-20261026"
        / "reap-v2-kaggle-q4-server.log",
        "log_sha": "5dfd24b1beac5445cd194b684bdadae0cd37cbe6d34a6337f9c6b1e55e8b9b48",
        "native": ROOT / "reap-v2-kaggle-q4-full-complete-private-evidence-20261026"
        / "reap-v2-kaggle-q4-native-smoke.json",
        "native_sha": "8f9ed207f0fa314281b6ddb9ed62e7ccd7f267a602ac5d1336f393202810e392",
        "native_schema": "reap-v2-q4-kaggle-complete-provisional-eval.v1",
        "native_status": "COMPLETE_FULL_DATA_PROVISIONAL_NOT_FROZEN_WINDOWS_GATE",
        "human": ROOT / "reap-v2-kaggle-q4-full-complete-private-evidence-20261026"
        / "reap-v2-a2a6b83e345145839cc333561b2f3baf-humaneval.jsonl",
        "human_sha": "92ec6439be77a7dbeb1a80448a0bb664e3cb6113a9cc0d356f7dbe300db69c43",
        "model_key": "q4_sha256",
        "total_eval_tokens": 497474,
    },
    "q8": {
        "log": ROOT / "reap-v2-q8-postload-full-complete-original-evidence-20261026-phase4"
        / "reap-v2-q8-postload-full-server.log",
        "log_sha": "6cd4a9c4c3e1dfa56b2a766a79fbec912288af2a528b05abde2f3c187d74485b",
        "native": ROOT / "reap-v2-q8-postload-full-complete-original-evidence-20261026-phase4"
        / "reap-v2-q8-postload-full-native.json",
        "native_sha": "92ca9969ef5959714f4787ed10a4375c69c2c4a80a4842daabc01c2360543fc0",
        "native_schema": "reap-v2-q8-original-full-with-measured-postload-cache-reclaim.v1",
        "native_status": "COMPLETE_Q8_RECLAIM_FULL_PROVISIONAL_NOT_FROZEN_WINDOWS_GATE",
        "human": ROOT / "reap-v2-q8-postload-full-complete-original-evidence-20261026-phase4"
        / "reap-q8-reclaim-full-95876063255e45159f57e29306e0e885-humaneval.jsonl",
        "human_sha": "cd3c9d1577611a032b36e69a3f96bd272d118819074352503f143bcd68cac301",
        "model_key": "q8_sha256",
        "total_eval_tokens": 507841,
    },
}
_SLOT = re.compile(
    r"\bslot\s+(operator\(\)|print_timing|release): id\s+(\d+)"
    r"\s+\| task (\d+) \| (.*)$"
)
_NEW = re.compile(
    r"new prompt, n_ctx_slot = (\d+), n_keep = (\d+), task\.n_tokens = (\d+)$"
)
_TIME = re.compile(r"(prompt eval time|eval time) =\s+([0-9.]+) ms /\s+(\d+) tokens.*$")
_RELEASE = re.compile(r"stop processing: n_tokens = (\d+), truncated = ([01])$")


class NativeWitnessError(ValueError):
    """Original native evidence is incomplete, mismatched or unverifiable."""


def parse_native_lifecycles(raw: bytes, *, expected_n: int = N) -> dict:
    """Reconstruct prompt, generation and actual slot-release per task id."""
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise NativeWitnessError("native log is not UTF-8") from exc
    events: dict[int, dict] = {}
    active_slots: dict[int, int] = {}
    seen_tasks: set[int] = set()
    first_task_per_slot: dict[int, int] = {}
    peak_active_slots = 0
    for ordinal, line in enumerate(lines):
        match = _SLOT.search(line)
        if not match:
            continue
        operation, slot_text, task_text, detail = match.groups()
        detail = detail.strip()
        if not (detail.startswith("new prompt") or "eval time =" in detail
                or operation == "release"):
            continue
        task = int(task_text)
        slot = int(slot_text)
        if task < 0 or slot not in range(4):
            raise NativeWitnessError("native task/slot outside original four-slot serving")
        record = events.setdefault(task, {"slot": slot})
        if record["slot"] != slot:
            raise NativeWitnessError("native task changed GPU serving slot")
        if operation == "release":
            parsed = _RELEASE.fullmatch(detail)
            if parsed is None or parsed.group(2) != "0":
                raise NativeWitnessError("native release malformed or truncated")
            kind, value = "release", int(parsed.group(1))
        elif detail.startswith("new prompt"):
            parsed = _NEW.fullmatch(detail)
            if operation != "operator()" or parsed is None:
                raise NativeWitnessError("malformed native new-prompt event")
            ctx, keep, tokens = map(int, parsed.groups())
            if ctx != 4096 or keep != 0 or not 0 < tokens <= ctx:
                raise NativeWitnessError("native prompt violates frozen context/fit")
            kind, value = "new", tokens
        else:
            parsed = _TIME.fullmatch(detail)
            if operation != "print_timing" or parsed is None:
                raise NativeWitnessError("malformed native token-timing event")
            label, milliseconds, tokens_text = parsed.groups()
            if not 0 < float(milliseconds) < 1e9:
                raise NativeWitnessError("invalid native event timing")
            value = int(tokens_text)
            if value <= 0:
                raise NativeWitnessError("native event has no tokens")
            kind = "prompt_eval" if label == "prompt eval time" else "eval"
        if kind in record:
            raise NativeWitnessError("duplicate native task lifecycle event")
        if kind == "new":
            if task in seen_tasks:
                raise NativeWitnessError("native task id reused after completion")
            if slot in active_slots:
                raise NativeWitnessError("overlapping native requests on one serving slot")
            seen_tasks.add(task)
            active_slots[slot] = task
            first_task_per_slot.setdefault(slot, task)
            peak_active_slots = max(peak_active_slots, len(active_slots))
        elif active_slots.get(slot) != task:
            raise NativeWitnessError(
                "native timing event out of order or without active matching slot"
            )
        if kind == "eval" and "prompt_eval" not in record:
            raise NativeWitnessError("native generation timing preceded prompt evaluation")
        if kind == "release":
            if "eval" not in record:
                raise NativeWitnessError("native slot released before generation timing")
            if value != record["new"][1] + record["eval"][1] - 1:
                raise NativeWitnessError("native release token accounting mismatch")
            del active_slots[slot]
        record[kind] = ((ordinal, value, milliseconds)
                        if kind in ("prompt_eval", "eval") else (ordinal, value))
    if active_slots:
        raise NativeWitnessError(
            "native serving slots left active: incomplete evaluation lifecycle"
        )
    if len(events) != expected_n:
        raise NativeWitnessError("native server log does not contain the complete cohort")
    for record in events.values():
        if set(record) != {"slot", "new", "prompt_eval", "eval", "release"}:
            raise NativeWitnessError("native request lacks a complete evaluation lifecycle")
        if not (record["new"][0] < record["prompt_eval"][0]
                < record["eval"][0] < record["release"][0]):
            raise NativeWitnessError("native task lifecycle is out of order")
        if record["prompt_eval"][1] > record["new"][1]:
            raise NativeWitnessError("native prompt evaluation exceeds declared prompt")
    saved = Counter(
        record["new"][1] - record["prompt_eval"][1] for record in events.values()
    )
    cold_first_slots = all(
        events[task]["new"][1] == events[task]["prompt_eval"][1]
        for task in first_task_per_slot.values()
    )
    return {
        "n": len(events),
        "new_prompt_tokens": Counter(record["new"][1] for record in events.values()),
        "prompt_eval_tokens": sum(record["prompt_eval"][1] for record in events.values()),
        "completion_tokens": Counter(record["eval"][1] for record in events.values()),
        "native_releases": sum("release" in record for record in events.values()),
        "cached_prompt_savings": saved,
        "cold_first_slot_requests": cold_first_slots,
        "max_simultaneously_active_slots": peak_active_slots,
        "per_task_eval_timing": {
            task: (record["eval"][1], record["eval"][2])
            for task, record in events.items()
        },
    }


def bind_archived_rows_to_native_timings(
    raw_rows: tuple[bytes, bytes], native: dict[int, tuple[int, str]],
) -> dict:
    """Unique numeric timing fingerprint; never authenticates response text.

    The archived llama-server response's predicted_per_second and the native
    server's independently pinned eval milliseconds encode the same duration.
    The native log rounds milliseconds to two decimal places, so matching uses
    a conservative 0.0051 ms rounding interval. This is correlation, not a
    signed request/response transcript.
    """
    by_tokens: dict[int, list[tuple[int, Decimal]]] = defaultdict(list)
    for task, (tokens, milliseconds) in native.items():
        try:
            elapsed = Decimal(milliseconds)
        except (InvalidOperation, TypeError) as exc:
            raise NativeWitnessError("invalid native per-task evaluation time") from exc
        if tokens < 2 or not elapsed.is_finite() or elapsed <= 0:
            raise NativeWitnessError("invalid native per-task timing fingerprint")
        by_tokens[tokens].append((task, elapsed))
    bound: dict[str, int] = {}
    used_tasks: set[int] = set()
    for raw in raw_rows:
        for line in raw.splitlines():
            row = pair._json(line)
            ident = row.get("id")
            tokens = row.get("completion_tokens")
            rate = row.get("predicted_per_second")
            if (type(ident) is not str or ident in bound or type(tokens) is not int
                    or tokens < 2 or type(rate) not in (int, float)):
                raise NativeWitnessError("invalid archived native timing fingerprint")
            try:
                predicted_rate = Decimal(str(rate))
                if not predicted_rate.is_finite() or predicted_rate <= 0:
                    raise NativeWitnessError("nonpositive archived prediction rate")
                elapsed = Decimal(tokens - 1) * 1000 / predicted_rate
            except (InvalidOperation, ZeroDivisionError) as exc:
                raise NativeWitnessError("invalid archived prediction rate") from exc
            matches = [
                task for task, native_ms in by_tokens.get(tokens, ())
                if abs(native_ms - elapsed) <= Decimal("0.0051")
            ]
            if len(matches) != 1 or matches[0] in used_tasks:
                raise NativeWitnessError(
                    "archived row lacks unique unused native timing fingerprint"
                )
            bound[ident] = matches[0]
            used_tasks.add(matches[0])
    if used_tasks != set(native):
        raise NativeWitnessError("native timing fingerprint lacks complete row coverage")
    canonical = json.dumps(bound, sort_keys=True, separators=(",", ":")).encode()
    return {"n": len(bound), "unique_native_tasks": len(used_tasks),
            "mapping_sha256": hashlib.sha256(canonical).hexdigest()}


def human_completion_tokens(raw: bytes, session: str) -> Counter:
    """Read exact original HumanEval rows; do not re-execute or rescore them."""
    if not raw.endswith(b"\n") or not 0 < len(raw) <= 2 * 1024 * 1024:
        raise NativeWitnessError("original HumanEval archive is incomplete")
    seen: set[str] = set()
    tokens: Counter = Counter()
    for line in raw.splitlines():
        try:
            row = pair._json(line)
        except pair.IndependentPairError as exc:
            raise NativeWitnessError("malformed original HumanEval row") from exc
        ident = row.get("id")
        if (set(row) != pair.ROW_KEYS or type(ident) is not str
                or not re.fullmatch(r"HumanEval/(?:0|[1-9]\d*)", ident)
                or ident in seen or row.get("eval_session_id") != session
                or type(row.get("completion_tokens")) is not int
                or row["completion_tokens"] <= 0
                or type(row.get("terminated")) is not bool
                or type(row.get("correct")) is not bool
                or type(row.get("text")) is not str
                or row.get("trigram_ratio") != pair._trigram(row["text"])):
            raise NativeWitnessError("original HumanEval row/session/token invalid")
        seen.add(ident)
        tokens[row["completion_tokens"]] += 1
    if seen != {f"HumanEval/{i}" for i in range(164)}:
        raise NativeWitnessError("original HumanEval population is not canonical 164")
    return tokens


def verify_original_native_log_pair() -> dict:
    """Source-pinned aggregate witness only, never a frozen-gate certificate."""
    canonical = grade.load_canonical()
    prompts = {}
    results = {}
    for arm, pinned in ARMS.items():
        source = pair.ARMS[arm]
        native_raw = pair._read_pinned(pinned["native"], pinned["native_sha"], 2 * 1024 * 1024)
        native = pair._json(native_raw)
        if (native.get("schema") != pinned["native_schema"]
                or native.get("status") != pinned["native_status"]
                or native.get("session_id") != source["session"]
                or native.get("flags") != FLAGS
                or native.get("source_sha256") != SOURCE_SHA
                or native.get("data", {}).get(pinned["model_key"]) != source["model_sha"]
                or native.get("server_log_sha256") != pinned["log_sha"]
                or native.get("scientific_selection") != "UNKNOWN"
                or native.get("not_official_windows_serving_attestation") is not True):
            raise NativeWitnessError("native original source/session/protocol receipt drift")
        suites = native.get("suite_results")
        if not isinstance(suites, dict) or set(suites) != {"gsm8k", "humaneval"}:
            raise NativeWitnessError("native suite population drift")
        for name, expected_sha, count in (
            ("gsm8k", source["rows_sha"], 1319),
            ("humaneval", pinned["human_sha"], 164),
        ):
            suite = suites[name]
            if (not isinstance(suite, dict) or suite.get("file_sha256") != expected_sha
                    or suite.get("session_id") != source["session"]
                    or suite.get("n") != count):
                raise NativeWitnessError("native suite receipt does not bind original rows")
        log_raw = pair._read_pinned(pinned["log"], pinned["log_sha"], 32 * 1024 * 1024)
        if native.get("server_log_bytes") != len(log_raw):
            raise NativeWitnessError("native server-log length does not match receipt")
        parsed = parse_native_lifecycles(log_raw)
        if (parsed["cached_prompt_savings"] != Counter({0: 4, 11: 1479})
                or not parsed["cold_first_slot_requests"]
                or parsed["max_simultaneously_active_slots"] != 4
                or parsed["native_releases"] != N):
            raise NativeWitnessError("original four-slot prompt-cache accounting drift")
        gsm_raw = pair._read_pinned(source["rows"], source["rows_sha"], 8 * 1024 * 1024)
        gsm = pair.regrade_rows(gsm_raw, canonical, source["session"])
        human_raw = pair._read_pinned(pinned["human"], pinned["human_sha"], 2 * 1024 * 1024)
        human_tokens = human_completion_tokens(human_raw, source["session"])
        native_row_binding = bind_archived_rows_to_native_timings(
            (gsm_raw, human_raw), parsed["per_task_eval_timing"]
        )
        gsm_tokens = Counter(pair._json(line)["completion_tokens"] for line in gsm_raw.splitlines())
        if parsed["completion_tokens"] != gsm_tokens + human_tokens:
            raise NativeWitnessError("native per-request token multiset differs from archived rows")
        if (sum(parsed["completion_tokens"].elements()) != pinned["total_eval_tokens"]
                or sum(parsed["new_prompt_tokens"].elements()) != 167973
                or parsed["prompt_eval_tokens"] != 151704
                or gsm["tokens"] != source["tokens"]):
            raise NativeWitnessError("original native token totals contradict source evidence")
        prompts[arm] = parsed["new_prompt_tokens"]
        results[arm] = {
            "native_requests": parsed["n"],
            "native_generation_tokens": sum(parsed["completion_tokens"].elements()),
            "native_prompt_evaluated_tokens": parsed["prompt_eval_tokens"],
            "native_declared_prompt_tokens": sum(parsed["new_prompt_tokens"].elements()),
            "native_cold_prompt_requests": parsed["cached_prompt_savings"][0],
            "native_cached_prompt_requests": parsed["cached_prompt_savings"][11],
            "native_cached_prompt_tokens_saved": 11 * parsed["cached_prompt_savings"][11],
            "native_slot_overlap_refused": True,
            "native_nontruncated_slot_releases": parsed["native_releases"],
            "native_archived_rows_uniquely_timing_correlated": native_row_binding["n"],
            "native_archived_timing_mapping_sha256": native_row_binding["mapping_sha256"],
            "archived_gsm8k_items": gsm["n"],
            "archived_humaneval_items": sum(human_tokens.values()),
            "native_server_log_sha256": pinned["log_sha"],
            "native_receipt_sha256": pinned["native_sha"],
            "session_id": source["session"],
        }
    if prompts["q4"] != prompts["q8"]:
        raise NativeWitnessError("original Q4/Q8 native prompt-token multisets differ")
    return {
        "schema": "chowder.reap-original-linux-native-log-aggregate-witness.v1",
        "arms": results,
        "same_canonical_prompt_token_multiset": True,
        "native_server_request_and_token_accounting_correlated": True,
        "per_item_native_timing_fingerprints_uniquely_correlated": True,
        "per_item_native_response_text_authenticated": False,
        "independent_cryptographic_execution_witness": False,
        "frozen_windows_gate1_qualified": False,
        "frozen_dense_gate2_qualified": False,
        "research_only": True,
        "status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }


def main() -> int:
    try:
        print(json.dumps(verify_original_native_log_pair(), indent=2, sort_keys=True))
        return 0
    except (NativeWitnessError, pair.IndependentPairError, grade.IndependentGradeError,
            OSError, TypeError, KeyError, UnicodeError) as exc:
        print(json.dumps({"status": "REFUSED_ORIGINAL_NATIVE_WITNESS",
                          "reason": str(exc)[:300], "release_approved": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
