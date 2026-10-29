"""Research-only, independent canonical GSM8K source and row regrading.

This reads a NEW local copy of the already-frozen 1,319 original prompts and
answers, not an F: execution journal. Its content digest is the immutable
original evaluator's frozen source digest. The number extraction below is a
small independent implementation of the original ``final_number_match`` rule;
it does not import mutable production scoring code or trust recorded labels.

This does NOT authenticate native prompts, model weights, serving sessions,
HumanEval execution, or a frozen gate. Never treat it as release authority.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Mapping


FROZEN_GSM8K_SHA256 = "913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89"
FROZEN_GSM8K_N = 1319
# Original production scoring semantics, independently reimplemented below.
# This is the SHA of Git's immutable LF blob at historical guard.v3 commit,
# not the CRLF bytes of a Windows checkout. It is descriptive, not a signer.
ORIGINAL_SCORER_GIT_COMMIT = "84ff9703d6e2aba68eda8585ee461e3d84abec0c"
ORIGINAL_SCORER_GIT_SHA256 = "b9b31edb03bd1e49411d0d405ec55ec5e92e58a1650ddd6da9f011d1989ea896"
CANONICAL_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs/quals/reap-v2-original-gsm8k-canonical-items.json"
)
_NUMBER = re.compile(r"[-]?\d[\d,]*(?:\.\d+)?")
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class IndependentGradeError(ValueError):
    """Evidence is incomplete, ambiguous, or contradicts the frozen source."""


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise IndependentGradeError("duplicate canonical JSON object key")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise IndependentGradeError("nonfinite canonical JSON number: " + value)


def load_canonical(path: Path = CANONICAL_PATH) -> tuple[tuple[str, str, str], ...]:
    """Read one source snapshot and authenticate all 1,319 ordered items."""
    try:
        metadata = path.lstat()
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
                or getattr(metadata, "st_file_attributes", 0) & _REPARSE):
            raise IndependentGradeError("canonical GSM8K source is redirected or aliased")
        if not 0 < metadata.st_size <= 1024 * 1024:
            raise IndependentGradeError("canonical GSM8K source size invalid")
        raw = path.read_bytes()
        if len(raw) != metadata.st_size:
            raise IndependentGradeError("canonical GSM8K source changed during read")
        items = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                           parse_constant=_reject_nonfinite)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IndependentGradeError("canonical GSM8K source unavailable or invalid") from exc
    if type(items) is not list or len(items) != FROZEN_GSM8K_N:
        raise IndependentGradeError("canonical GSM8K population incomplete")
    ordered = []
    for i, item in enumerate(items):
        if (type(item) is not dict or set(item) != {"id", "prompt", "expected"}
                or item["id"] != f"gsm8k-{i}"
                or type(item["prompt"]) is not str or not item["prompt"]
                or type(item["expected"]) is not str or not item["expected"]):
            raise IndependentGradeError(f"canonical GSM8K item {i} invalid")
        ordered.append((item["id"], item["prompt"], item["expected"]))
    normalized = json.dumps(items, sort_keys=True, ensure_ascii=False,
                            separators=(",", ":"), allow_nan=False).encode("utf-8")
    if hashlib.sha256(normalized).hexdigest() != FROZEN_GSM8K_SHA256:
        raise IndependentGradeError("canonical GSM8K source SHA differs from frozen original")
    return tuple(ordered)


def _last_number(text: str) -> str | None:
    matches = _NUMBER.findall(text)
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    if raw.endswith(".0"):
        raw = raw[:-2]
    if raw.endswith("."):
        raw = raw[:-1]
    return raw or None


def independent_number_match(prediction: str, expected: str) -> bool:
    """Original final-answer extraction, independently implemented."""
    if "</think>" in prediction:
        answer = prediction.rsplit("</think>", 1)[1]
    elif "<think>" in prediction:
        answer = ""
    else:
        answer = prediction
    got = _last_number(answer)
    want = _last_number(expected)
    return got is not None and want is not None and got == want


def verify_rows(
    rows: Mapping[str, Mapping], canonical: tuple[tuple[str, str, str], ...],
) -> dict:
    """Recompute each original recorded correctness bit, never trust it.

    The caller must separately authenticate the native response-to-row binding.
    This function only checks canonical answers, scorer parity and population.
    """
    if (type(canonical) is not tuple or len(canonical) != FROZEN_GSM8K_N
            or any(type(item) is not tuple or len(item) != 3
                   or any(type(field) is not str for field in item)
                   or item[0] != f"gsm8k-{i}"
                   for i, item in enumerate(canonical))):
        raise IndependentGradeError("untrusted canonical item population")
    reconstructed = [
        {"id": item[0], "prompt": item[1], "expected": item[2]}
        for item in canonical
    ]
    digest = hashlib.sha256(json.dumps(
        reconstructed, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    if digest != FROZEN_GSM8K_SHA256:
        raise IndependentGradeError("untrusted canonical source SHA")
    n = len(rows)
    if n not in (200, FROZEN_GSM8K_N) or set(rows) != {
        f"gsm8k-{i}" for i in range(n)
    }:
        raise IndependentGradeError("historical GSM8K rows are not a canonical complete prefix")
    correct = 0
    naturally_terminated = 0
    for i in range(n):
        task_id, _, expected = canonical[i]
        row = rows[task_id]
        if (not isinstance(row, Mapping) or row.get("id") != task_id
                or type(row.get("text")) is not str
                or type(row.get("terminated")) is not bool
                or type(row.get("correct")) is not bool):
            raise IndependentGradeError(f"{task_id}: invalid historical scoring fields")
        terminated = row["terminated"]
        independent = terminated and independent_number_match(row["text"], expected)
        if row["correct"] is not independent:
            raise IndependentGradeError(f"{task_id}: historical correctness does not regrade")
        correct += int(independent)
        naturally_terminated += int(terminated)
    return {
        "schema": "chowder.reap-original-gsm8k-independent-canonical-grade.v1",
        "n": n,
        "correct": correct,
        "naturally_terminated": naturally_terminated,
        "canonical_source_sha256": FROZEN_GSM8K_SHA256,
        "native_request_and_response_lineage_authenticated": False,
        "research_only": True,
        "gen0_status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }
