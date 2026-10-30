"""Cross-platform original GSM8K native prompt-length witness (research only).

Read the existing SHA-pinned original Windows Q4 and Linux Q4/Q8 logs and
response archives. Unique native timing fingerprints associate canonical row
IDs with server task IDs, allowing comparison of declared prompt-token counts
for the SAME 1,319 GSM8K items across three independently archived sessions.

Equal token lengths do not prove equal prompt bytes, model weights, or
authenticated HTTP response text. This never qualifies frozen Gen-0 gates.
"""

from __future__ import annotations

import hashlib
import json

from chowder_batch import reap_original_linux_gsm8k_independent_pair_20261029 as pair
from chowder_batch import reap_original_linux_native_log_witness_20261029 as native
from chowder_batch import reap_original_windows_q4_native_witness_20261029 as windows


N = 1319
EXPECTED_PROMPT_MAP_SHA256 = (
    "d148e024faa57d0df7dfb9b0e8cbdee024b68fdb9f6b505c89d989f4f2e251ef"
)
EXPECTED_ARMS = frozenset({"windows_q4", "linux_q4", "linux_q8"})
EXPECTED_IDS = frozenset(f"gsm8k-{i}" for i in range(N))


class CrossPlatformPromptError(ValueError):
    """Original per-item native prompt evidence is inconsistent or incomplete."""


def prompt_lengths_from_native(
    server_log: bytes, gsm8k_rows: bytes, humaneval_rows: bytes = b"",
    *, expected_native_requests: int,
) -> dict[str, int]:
    """Bind complete native tasks to canonical rows by unique timing evidence."""
    parsed = native.parse_native_lifecycles(
        server_log, expected_n=expected_native_requests
    )
    bound = native.bind_archived_rows_to_native_timings(
        (gsm8k_rows, humaneval_rows), parsed["per_task_eval_timing"]
    )
    mapping = bound["row_to_native_task"]
    if (set(mapping) != EXPECTED_IDS | (
        frozenset(f"HumanEval/{i}" for i in range(164))
        if humaneval_rows else frozenset()
    )):
        raise CrossPlatformPromptError("native task-to-canonical population mismatch")
    lengths = {
        ident: parsed["per_task_declared_prompt_tokens"][task]
        for ident, task in mapping.items() if ident in EXPECTED_IDS
    }
    if (set(lengths) != EXPECTED_IDS
            or any(type(value) is not int or value <= 0 for value in lengths.values())):
        raise CrossPlatformPromptError("invalid per-item original native prompt length")
    return lengths


def compare_original_prompt_maps(per_arm: dict[str, dict[str, int]]) -> dict:
    """Require exact per-ID equality, not only equal aggregate distributions."""
    if set(per_arm) != EXPECTED_ARMS:
        raise CrossPlatformPromptError("missing original Windows or Linux prompt arm")
    for arm, lengths in per_arm.items():
        if (type(lengths) is not dict or set(lengths) != EXPECTED_IDS
                or any(type(v) is not int or v <= 0 for v in lengths.values())):
            raise CrossPlatformPromptError(f"{arm}: incomplete native prompt population")
    win = per_arm["windows_q4"]
    if any(lengths != win for lengths in per_arm.values()):
        raise CrossPlatformPromptError("original cross-platform per-item prompts differ")
    raw = json.dumps(win, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_PROMPT_MAP_SHA256:
        raise CrossPlatformPromptError("original prompt-length mapping digest differs")
    return {"n": N, "prompt_length_map_sha256": digest,
            "declared_prompt_tokens_per_arm": sum(win.values())}


def verify_original_cross_platform_prompts() -> dict:
    """Reverify source-bound original receipts before comparing three sessions."""
    # These already independently regrade archived GSM8K and authenticate
    # native logs against their immutable SHA pins. No model is invoked.
    windows.verify_original_windows_q4_native()
    native.verify_original_native_log_pair()

    win_log = pair._read_pinned(
        windows.ARM / "server.log", windows.PINS["reap128_v2_q4/server.log"],
        windows.LIMITS["server.log"],
    )
    win_rows = pair._read_pinned(
        windows.ARM / "gsm8k.jsonl", windows.PINS["reap128_v2_q4/gsm8k.jsonl"],
        windows.LIMITS["gsm8k.jsonl"],
    )
    maps = {"windows_q4": prompt_lengths_from_native(
        win_log, win_rows, expected_native_requests=N,
    )}
    for arm in ("q4", "q8"):
        pin = native.ARMS[arm]
        source = pair.ARMS[arm]
        log = pair._read_pinned(pin["log"], pin["log_sha"], 32 * 1024 * 1024)
        gsm = pair._read_pinned(source["rows"], source["rows_sha"], 8 * 1024 * 1024)
        human = pair._read_pinned(pin["human"], pin["human_sha"], 2 * 1024 * 1024)
        maps[f"linux_{arm}"] = prompt_lengths_from_native(
            log, gsm, human, expected_native_requests=N + 164,
        )
    witness = compare_original_prompt_maps(maps)
    return {
        "schema": "chowder.reap-original-cross-platform-gsm8k-native-prompt-witness.v1",
        "original_sessions": {
            "windows_q4": windows.SESSION,
            "linux_q4": pair.ARMS["q4"]["session"],
            "linux_q8": pair.ARMS["q8"]["session"],
        },
        "cross_platform_same_canonical_gsm8k_prompt_lengths": True,
        "canonical_gsm8k_items_per_arm": witness["n"],
        "per_item_prompt_length_map_sha256": witness["prompt_length_map_sha256"],
        "declared_prompt_tokens_per_arm": witness["declared_prompt_tokens_per_arm"],
        "prompt_text_authenticated": False,
        "native_response_text_authenticated": False,
        "model_weights_authenticated_at_original_execution": False,
        "frozen_gate1_qualified": False,
        "frozen_gate2_qualified": False,
        "research_only": True,
        "status": "BLOCKED", "chosen": None, "release_approved": False,
    }


def main() -> int:
    try:
        print(json.dumps(verify_original_cross_platform_prompts(),
                         indent=2, sort_keys=True))
        return 0
    except (CrossPlatformPromptError, native.NativeWitnessError,
            windows.WindowsQ4WitnessError, pair.IndependentPairError,
            OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "REFUSED_CROSS_PLATFORM_PROMPT_WITNESS",
                          "reason": str(exc)[:300], "release_approved": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
