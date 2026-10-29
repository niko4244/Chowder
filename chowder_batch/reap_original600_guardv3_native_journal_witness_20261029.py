"""CPU-only original guard.v3 proxy/native session accounting, research only.

Read the original F: evidence through the existing Windows handle-locked
snapshot and the exact Git-pinned historical scorer. Reconcile EVERY native
task launch/release with the original guard.v3 request journal counts, while
reporting unexplained native launches and unfinished native requests rather
than treating them as successful model calls.

No historical HTTP request/response bodies were independently attested.
These are segmented one-slot research sessions, NOT the frozen four-slot
Gen-0 comparison. No inference, generated code, or F: writes occur here.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from chowder_batch import reap_original600_guardv3_git_pinned_reader_20261027 as historical
from chowder_batch.reap_original600_guardv3_stable_snapshot_20261029 import stable_snapshot


SCHEMA = "chowder.reap-original600-guardv3-native-journal-accounting.v1"
SESSIONS = (
    ("1793126619", "1793126629"),
    ("1793134653", "1793134663"),
    ("1793142187", "1793142196"),
    ("1793147057", "1793147067"),
    ("1793147367", "1793147377"),
    ("1793232246", "1793232308"),
)
UNSERVED_LOG = "native-session-1793147244.private.log"
UNSERVED_SHA = "59872df0f07c066f5fac8ea506098a7ce1980ec05ad4802ade36f5c3f06ce033"
# Original F: bytes independently observed in the historical source tree.
# Never learn these expected values from a receipt or today's F: bytes.
PINNED_NATIVE = {
    "1793126619": "8b47f0c04fc958977b31b434a4dcb9611415b583dd91a0d37b935b9262dce9d2",
    "1793134653": "d7ec0e5f7bf18f655ae3fcfac15166b76805196f5ccbfe69cdbfcf52e211533e",
    "1793142187": "65d9364dccf16a538a0290b533f20dd4b13aac7cf0febc3cc4ae24479019c222",
    "1793147057": "80f1ae9fa82a500e28da79fe6e2c335999aea0a4a763ac3ae52d76455f883173",
    "1793147367": "d582c126315369b012b6a772639cac8a89a75c68295b3fb6a56aa5e459f09e76",
    "1793232246": "47e8773f2d7c79b63956386f57b780cbfdd617fb4aa9cf0338ba47347256c5bd",
}
PINNED_GUARD = {
    "1793126629": "abcaa1187322e95299520a9e0a556580a21bc0ae079ec6cb5ae5b151f5ce9fe6",
    "1793134663": "30a52dad75c8a9490ddcb395804f5cb331429d6cbe8f679779c436ebceb60ca9",
    "1793142196": "50239973919a23e8e61ac7b3e670e98aa612624d26d20d22f5044a6d0b23b441",
    "1793147067": "a7b210345ec932a1ebb5c5e9d0887cb0b6175b9c343ac88e7f837d50500bf5b1",
    "1793147377": "f2bd3f233159ea0952e22a0a8e0f1e73178518e9cc86b96df69694c29e74c9a3",
    "1793232308": "0fdf1b5c0e4f65ac81441be57d78a19bb0c3f8761e559bfd75eecd26ec060cd9",
}
MAX_NATIVE = 32 * 1024 * 1024
MAX_JOURNAL = 2 * 1024 * 1024
LAUNCH = re.compile(
    rb" I slot\s+(?:launch_slot_|operator\(\)): id\s+(\d+) \| task\s+(\d+)"
    rb" \| processing task, is_child = (\d+)\r?$"
)
RELEASE = re.compile(
    rb" I slot\s+release: id\s+(\d+) \| task\s+(\d+)"
    rb" \| stop processing: n_tokens = (\d+), truncated = (\d+)\r?$"
)
PROMPT = re.compile(
    rb" I slot print_timing: id\s+(\d+) \| task\s+(\d+)"
    rb" \| prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens"
)
DECODE = re.compile(
    rb" I slot print_timing: id\s+(\d+) \| task\s+(\d+)"
    rb" \|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens"
)


class NativeJournalError(ValueError):
    """Original proxy/native evidence is incomplete, ambiguous or inconsistent."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_pinned(raw: bytes, expected: str, label: str) -> None:
    if _sha(raw) != expected:
        raise NativeJournalError("original source SHA drift: " + label)


def _strict_json(raw: bytes) -> dict:
    def unique(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise NativeJournalError("duplicate original journal JSON key")
            result[key] = value
        return result

    def reject_constant(value: str):
        raise NativeJournalError("nonfinite original journal JSON number: " + value)

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                           parse_constant=reject_constant)
    except (UnicodeError, ValueError) as exc:
        raise NativeJournalError("original journal JSON invalid") from exc
    if type(value) is not dict:
        raise NativeJournalError("original journal row must be object")
    return value


def parse_journal(raw: bytes) -> dict:
    """Require a complete contiguous original guard.v3 request-index ledger."""
    if not 0 < len(raw) <= MAX_JOURNAL or not raw.endswith(b"\n"):
        raise NativeJournalError("original guard.v3 journal missing complete final line")
    events = {"forwarded": 0, "TOOL_ARGUMENTS_UNREPAIRABLE": 0}
    rows = []
    for index, line in enumerate(raw.splitlines(), 1):
        row = _strict_json(line)
        event = row.get("event")
        if (type(row.get("request_index")) is not int
                or row["request_index"] != index
                or row.get("policy") != "guard.v3"
                or event not in events
                or type(row.get("prompt_tokens_after")) is not int
                or not 0 < row["prompt_tokens_after"] <= 73728
                or type(row.get("max_tokens")) is not int
                or not 0 < row["max_tokens"] <= 16384
                or type(row.get("tool_arguments_unrepairable")) is not int
                or row["tool_arguments_unrepairable"] < 0):
            raise NativeJournalError("original guard.v3 journal request invalid")
        events[event] += 1
        rows.append((index, event, row["prompt_tokens_after"]))
    if not rows:
        raise NativeJournalError("empty original guard.v3 request journal")
    return {"requests": len(rows), "events": events,
            "request_sequence_sha256": _sha(json.dumps(
                rows, separators=(",", ":"), allow_nan=False).encode())}


def parse_native(raw: bytes, *, allow_empty: bool = False) -> dict:
    """Require exact single-slot native lifecycle and timing event ordering.

    A final launched task may be unfinished in a stopped session. It is
    counted as UNKNOWN, never silently scored or assigned to a guard request.
    """
    if not 0 < len(raw) <= MAX_NATIVE or not raw.endswith(b"\n"):
        raise NativeJournalError("native log is missing or has incomplete final line")
    launched: set[int] = set()
    released: set[int] = set()
    active: int | None = None
    prompt_seen: set[int] = set()
    decode_seen: set[int] = set()
    untimed_releases: set[int] = set()
    generation_tokens = 0
    last_id = -1
    for line in raw.splitlines():
        if b" | processing task, is_child =" in line:
            match = LAUNCH.search(line)
            if match is None:
                raise NativeJournalError("unparsed original native launch")
            slot, task, child = map(int, match.groups())
            if (slot != 0 or child != 0 or active is not None
                    or task <= last_id or task in launched):
                raise NativeJournalError("overlapping, reordered or child native launch")
            launched.add(task)
            last_id = task
            active = task
        elif b" I slot print_timing:" in line and b"prompt eval time =" in line:
            match = PROMPT.search(line)
            if match is None:
                raise NativeJournalError("unparsed original native prompt timing")
            slot, task = int(match[1]), int(match[2])
            ms, tokens = float(match[3]), int(match[4])
            if (slot != 0 or task != active or task in prompt_seen
                    or not math.isfinite(ms) or not 0 < ms < 1e9
                    or not 0 < tokens <= 73728):
                raise NativeJournalError("orphan or duplicate native prompt timing")
            prompt_seen.add(task)
        elif b" I slot print_timing:" in line and b"eval time =" in line:
            match = DECODE.search(line)
            if match is None:
                raise NativeJournalError("unparsed original native decode timing")
            slot, task, ms, tokens = int(match[1]), int(match[2]), float(match[3]), int(match[4])
            if (slot != 0 or task != active or task in decode_seen
                    or task not in prompt_seen
                    or not math.isfinite(ms) or not 0 < ms < 1e9
                    or not 0 < tokens <= 16384):
                raise NativeJournalError("orphan, duplicate or invalid native decode timing")
            decode_seen.add(task)
            generation_tokens += tokens
        elif b" | stop processing: n_tokens =" in line:
            match = RELEASE.search(line)
            if match is None:
                raise NativeJournalError("unparsed original native release")
            slot, task, tokens, truncated = map(int, match.groups())
            if (slot != 0 or task != active or task in released
                    or tokens <= 0 or truncated != 0):
                raise NativeJournalError(
                    f"unpaired/truncated native release task={task} active={active} "
                    f"prompt={task in prompt_seen} decode={task in decode_seen} "
                    f"tokens={tokens} truncated={truncated}")
            if task not in prompt_seen or task not in decode_seen:
                # A canceled final request can emit a release without either
                # final timing line. That is UNKNOWN, not a completed decode.
                untimed_releases.add(task)
            released.add(task)
            active = None
    if not allow_empty and not launched:
        raise NativeJournalError("active native log contains no model requests")
    if (len(launched - released) > 1
            or (launched - released) != ({active} if active is not None else set())
            or len(untimed_releases) > 1
            or (untimed_releases and untimed_releases != {last_id})
            or not (released - untimed_releases).issubset(prompt_seen & decode_seen)
            or (prompt_seen | decode_seen) - launched):
        raise NativeJournalError("native request lifecycle incomplete beyond terminal tail")
    return {"launched": len(launched), "released": len(released),
            "unfinished_unknown": len(launched - released),
            "released_without_final_timings_unknown": len(untimed_releases),
            "fully_timed_releases": len(released - untimed_releases),
            "native_decode_tokens_from_timing": generation_tokens,
            "first_task_id": min(launched) if launched else None,
            "last_task_id": max(launched) if launched else None}


def reconcile_pair(native_raw: bytes, journal_raw: bytes) -> dict:
    native = parse_native(native_raw)
    journal = parse_journal(journal_raw)
    excess = native["launched"] - journal["requests"]
    # An extra native request may be a setup probe, but the historical logs
    # cannot establish that. Do not assign it to a scored item or certify it.
    if excess not in (0, 1):
        raise NativeJournalError("unexplained native/guard request count discrepancy")
    return {**native, **journal, "unmatched_native_launches": excess,
            "per_request_http_bytes_authenticated": False}


def verify_original_guardv3_sessions() -> dict:
    """Source-lock scorer and handle-lock historical F: evidence before parsing."""
    historical._require_assertions_enabled()
    with stable_snapshot(historical.ROOT, historical.REPO) as snapshot:
        owner = (snapshot / "owner.json").read_bytes()
        if _sha(owner) != historical.OWNER_SHA:
            raise NativeJournalError("historical guard.v3 owner SHA changed")
        runner, scorer = historical.historical_modules()
        scored = scorer.inspect(snapshot)
        if (scored.get("owner_sha256") != historical.OWNER_SHA
                or scored.get("completed_scored_tasks", 0) > 600
                or scored.get("frozen_gen0_status") != "BLOCKED"
                or scored.get("release_approved") is not False):
            raise NativeJournalError("historical source-locked scoring differs")
        native_names = {p.name for p in snapshot.glob("native-session-*.private.log")}
        journal_names = {p.name for p in snapshot.glob("guard-session-*.private.jsonl")}
        expected_native = {f"native-session-{native}.private.log" for native, _ in SESSIONS}
        expected_journal = {f"guard-session-{guard}.private.jsonl" for _, guard in SESSIONS}
        if (native_names != expected_native | {UNSERVED_LOG}
                or journal_names != expected_journal):
            raise NativeJournalError("original native/guard session inventory changed")
        unserved = (snapshot / UNSERVED_LOG).read_bytes()
        _require_pinned(unserved, UNSERVED_SHA, UNSERVED_LOG)
        if parse_native(unserved, allow_empty=True)["launched"] != 0:
            raise NativeJournalError("unserved native log contains unaccounted requests")
        results = []
        for native_id, guard_id in SESSIONS:
            native_file = snapshot / f"native-session-{native_id}.private.log"
            guard_file = snapshot / f"guard-session-{guard_id}.private.jsonl"
            native_raw, guard_raw = native_file.read_bytes(), guard_file.read_bytes()
            _require_pinned(native_raw, PINNED_NATIVE[native_id], native_file.name)
            _require_pinned(guard_raw, PINNED_GUARD[guard_id], guard_file.name)
            if (b"offloaded 41/41 layers to GPU" not in native_raw
                    or b"loading model 'G:\\reap128-v2-q4_k_m-20261025.gguf'" not in native_raw):
                raise NativeJournalError("native original Q4 model/offload witness absent")
            item = reconcile_pair(native_raw, guard_raw)
            results.append({"native_session": native_id, "guard_session": guard_id,
                            "native_log_sha256": _sha(native_raw),
                            "guard_journal_sha256": _sha(guard_raw), **item})
    return {
        "schema": SCHEMA, "historical_source_commit": historical.COMMIT,
        "historical_owner_sha256": historical.OWNER_SHA,
        "original_scored_tasks": scored["completed_scored_tasks"],
        "original_strict_passes": scored["totals"]["strict_passes"],
        "sessions": results,
        "native_launches": sum(item["launched"] for item in results),
        "native_releases": sum(item["released"] for item in results),
        "native_fully_timed_releases": sum(item["fully_timed_releases"] for item in results),
        "native_untimed_releases_unknown": sum(
            item["released_without_final_timings_unknown"] for item in results),
        "guard_requests": sum(item["requests"] for item in results),
        "unmatched_native_launches": sum(item["unmatched_native_launches"] for item in results),
        "unfinished_native_unknown": sum(item["unfinished_unknown"] for item in results),
        "unrepairable_guard_events": sum(
            item["events"]["TOOL_ARGUMENTS_UNREPAIRABLE"] for item in results),
        "session_accounting_sha256": _sha(json.dumps(
            results, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()),
        "native_http_request_response_authenticated": False,
        "original_frozen_four_slot_comparable": False,
        "complete_600_scored": scored["completed_scored_tasks"] == 600,
        "status": "RESEARCH_ONLY_SOURCE_LOCKED_SESSION_ACCOUNTING",
        "gen0_status": "BLOCKED", "chosen": None, "release_approved": False,
    }


def main() -> int:
    try:
        print(json.dumps(verify_original_guardv3_sessions(), indent=2, sort_keys=True))
        return 0
    except (NativeJournalError, ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "REFUSED_ORIGINAL_GUARDV3_NATIVE_JOURNAL",
                          "reason": str(exc)[:350], "gen0_status": "BLOCKED",
                          "chosen": None, "release_approved": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
