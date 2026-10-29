"""Independent, read-only original Linux Q4/Q8 GSM8K1319 paired regrading.

Research-only historical evidence, NOT frozen Windows Gate1 or dense Gate2.
Never loads a model, executes generated code, or changes F: source artifacts.
Pinned raw-file digests and separate session identities are essential: a
matching historical receipt alone is not proof of live model execution.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import stat
from typing import Mapping

from chowder_batch import reap_original_gsm8k_independent_grade_20261029 as grade


ROOT = Path("F:/")
ARMS = {
    "q4": {
        "rows": ROOT / "reap-v2-kaggle-q4-full-complete-private-evidence-20261026"
        / "reap-v2-a2a6b83e345145839cc333561b2f3baf-gsm8k.jsonl",
        "rows_sha": "3b09d337f567729192e9c302ec698cf1e34d7451538847939e3bf9a36a9557db",
        "audit": ROOT / "reap-v2-sourcebound-q4-with-native-gpu-log-audit-20261026.json",
        "audit_sha": "163d2189f220fd04620f1d94a79e899624d43ef1e8bc6a5d462159fd2e178b6f",
        "session": "a2a6b83e345145839cc333561b2f3baf",
        "model": "reap128_v2_q4",
        "model_sha": "2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7",
        "correct": 1229,
        "terminated": 1281,
        "tokens": 456671,
    },
    "q8": {
        "rows": ROOT / "reap-v2-q8-postload-full-complete-original-evidence-20261026-phase4"
        / "reap-q8-reclaim-full-95876063255e45159f57e29306e0e885-gsm8k.jsonl",
        "rows_sha": "9ff2b418fecabdc419d94b2a06f6d9b115bbbd4bfba59ccae2d0d9f81e8ef749",
        "audit": ROOT / "reap-v2-q8-postload-original-independent-source-audit-20261026-phase4.json",
        "audit_sha": "bc7eb4ba83475db3b82f648a3ce50b8b7ef8ef4db030a285a227837e5dffdb48",
        "session": "95876063255e45159f57e29306e0e885",
        "model": "reap128_v2_q8",
        "model_sha": "89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6",
        "correct": 1220,
        "terminated": 1280,
        "tokens": 467560,
    },
}
PAIR = ROOT / "reap-v2-original-full-q8-vs-q4-gsm8k-paired-descriptive-20261026-phase4.json"
PAIR_SHA = "125df9301e222f3f9220742514f279b1b6ff4b31444b709afa0ab4bfe8e13b93"
REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
ROW_KEYS = frozenset({"id", "correct", "terminated", "trigram_ratio",
                      "completion_tokens", "seconds", "predicted_per_second",
                      "text", "eval_session_id"})


class IndependentPairError(ValueError):
    """Archived evidence is incomplete, malformed or not source-bound."""


def _read_pinned(path: Path, sha256: str, limit: int) -> bytes:
    try:
        before = path.lstat()
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or getattr(before, "st_file_attributes", 0) & REPARSE
                or not 0 < before.st_size <= limit):
            raise IndependentPairError("unsafe archived evidence file")
        raw = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise IndependentPairError("archived evidence unavailable") from exc
    if (len(raw) != before.st_size or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or hashlib.sha256(raw).hexdigest() != sha256):
        raise IndependentPairError("archived evidence changed or SHA differs")
    return raw


def _json(raw: bytes) -> dict:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=grade._unique_pairs,
                           parse_constant=grade._reject_nonfinite)
    except (ValueError, UnicodeError) as exc:
        raise IndependentPairError("invalid archived JSON") from exc
    if type(value) is not dict:
        raise IndependentPairError("archived JSON must be an object")
    return value


def _trigram(text: str) -> float:
    words = text.split()
    triples = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    return round(len(set(triples)) / len(triples), 4) if triples else 1.0


def regrade_rows(raw: bytes, canonical: tuple[tuple[str, str, str], ...],
                 session: str) -> dict:
    """Independent canonical answers, EOS, trigram and full session coverage."""
    if not raw.endswith(b"\n") or not 0 < len(raw) <= 8 * 1024 * 1024:
        raise IndependentPairError("incomplete archived JSONL")
    rows: dict[str, dict] = {}
    tokens = 0
    for line in raw.splitlines():
        row = _json(line)
        ident = row.get("id")
        if (set(row) != ROW_KEYS or type(ident) is not str or ident in rows
                or row.get("eval_session_id") != session
                or type(row.get("completion_tokens")) is not int
                or row["completion_tokens"] < 0
                or type(row.get("text")) is not str):
            raise IndependentPairError("invalid row identity, session or token count")
        for key in ("seconds", "predicted_per_second", "trigram_ratio"):
            value = row[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise IndependentPairError("invalid archived row timing or trigram")
        if row["trigram_ratio"] != _trigram(row["text"]):
            raise IndependentPairError("recorded trigram does not independently regrade")
        rows[ident] = row
        tokens += row["completion_tokens"]
    try:
        summary = grade.verify_rows(rows, canonical)
    except grade.IndependentGradeError as exc:
        raise IndependentPairError("independent canonical answer regrading failed") from exc
    return {**summary, "tokens": tokens,
            "correct_by_id": {ident: row["correct"] for ident, row in rows.items()}}


def verify_original_linux_pair() -> dict:
    """Verify existing original archived Linux arms; never certify frozen gates."""
    canonical = grade.load_canonical()
    results = {}
    for arm, expected in ARMS.items():
        raw = _read_pinned(expected["rows"], expected["rows_sha"], 8 * 1024 * 1024)
        scored = regrade_rows(raw, canonical, expected["session"])
        audit = _json(_read_pinned(expected["audit"], expected["audit_sha"], 2 * 1024 * 1024))
        suite = audit.get("suites", {}).get("gsm8k")
        if (audit.get("schema") != "chowder.reap-v2-original-population-external-independent-audit.v1"
                or audit.get("status") != "COMPLETE_LINUX_COHORT_VERIFIED_PROVISIONAL_NOT_WINDOWS_GATE"
                or audit.get("model") != expected["model"]
                or audit.get("model_sha256") != expected["model_sha"]
                or audit.get("session_id") != expected["session"]
                or audit.get("gen0_status") != "BLOCKED"
                or audit.get("chosen", "missing") is not None
                or audit.get("release_approved") is not False
                or audit.get("original_windows_gate_eligible") is not False
                or audit.get("speed_ratio_certified") is not False
                or not isinstance(suite, Mapping)
                or suite.get("row_sha256") != expected["rows_sha"]
                or suite.get("n") != 1319
                or suite.get("correct") != scored["correct"]
                or suite.get("terminated") != scored["naturally_terminated"]
                or suite.get("completion_tokens") != scored["tokens"]
                or suite.get("accuracy") != scored["correct"] / 1319
                or scored["correct"] != expected["correct"]
                or scored["naturally_terminated"] != expected["terminated"]
                or scored["tokens"] != expected["tokens"]):
            raise IndependentPairError(f"{arm} original audit disagrees with canonical rows")
        results[arm] = scored
    paired = {"both_correct": 0, "both_wrong": 0,
              "candidate_only_correct": 0, "reference_only_correct": 0}
    for ident in results["q4"]["correct_by_id"]:
        q4 = results["q4"]["correct_by_id"][ident]
        q8 = results["q8"]["correct_by_id"][ident]
        key = ("both_correct" if q4 and q8 else "both_wrong" if not q4 and not q8
               else "candidate_only_correct" if q8 else "reference_only_correct")
        paired[key] += 1
    certificate = _json(_read_pinned(PAIR, PAIR_SHA, 2 * 1024 * 1024))
    reported = certificate.get("gsm8k")
    if (certificate.get("schema") != "reap-v2-original-external-gsm8k-paired-private-descriptive.v1"
            or certificate.get("status") != "SOURCE_BOUND_COMPLETE_LINUX_GSM8K_PAIRED_DESCRIPTIVE_ONLY"
            or certificate.get("gen0_status") != "BLOCKED"
            or certificate.get("chosen", "missing") is not None
            or certificate.get("release_approved") is not False
            or certificate.get("source_row_sha256") != {k: v["rows_sha"] for k, v in ARMS.items()}
            or certificate.get("audit_sha256") != {k: v["audit_sha"] for k, v in ARMS.items()}
            or not isinstance(reported, Mapping)
            or any(reported.get(k) != v for k, v in paired.items())
            or reported.get("n") != 1319
            or reported.get("candidate_session") != ARMS["q8"]["session"]
            or reported.get("reference_session") != ARMS["q4"]["session"]
            or reported.get("candidate_model_sha256") != ARMS["q8"]["model_sha"]
            or reported.get("reference_model_sha256") != ARMS["q4"]["model_sha"]
            or reported.get("candidate_accuracy") != results["q8"]["correct"] / 1319
            or reported.get("reference_accuracy") != results["q4"]["correct"] / 1319
            or reported.get("delta_accuracy") != (
                paired["candidate_only_correct"] - paired["reference_only_correct"]) / 1319
            or reported.get("amendment4_eligibility") is not False
            or reported.get("paired_confidence_interval") is not None
            or reported.get("statistical_method") != "NONE_DESCRIPTIVE_ONLY_UNRESOLVED_MCNEMAR_VS_WALD"
            or reported.get("speed_ratio_certified") is not False):
        raise IndependentPairError("original descriptive paired receipt disagrees")
    return {"schema": "chowder.reap-original-linux-gsm8k-independent-pair.v1",
            "q4_correct": results["q4"]["correct"],
            "q8_correct": results["q8"]["correct"],
            "paired": paired, "source_sha256": grade.FROZEN_GSM8K_SHA256,
            "native_execution_independently_authenticated": False,
            "frozen_windows_gate1_qualified": False,
            "frozen_dense_gate2_qualified": False,
            "research_only": True, "status": "BLOCKED",
            "chosen": None, "release_approved": False}
