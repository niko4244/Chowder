"""Source-pinned, CPU-only cross-host original GSM8K1319 research pairing.

This pairs the already-completed Windows Q4 and *different-session Linux* Q8
original response rows. It does not rerun inference or claim a registered
Windows Q8-vs-dense frozen Gate1 comparison. Both files are immutable F:
research evidence, independently regraded against all 1,319 canonical items.
No HTTP request/response capture or model-session ownership is authenticated.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import stat

from chowder_batch import reap_original_gsm8k_frozen_scorer_20261029 as scorer
from chowder_batch import reap_original_gsm8k_independent_grade_20261029 as grade


SCHEMA = "chowder.reap-original-gsm8k-crosshost-q4-q8-research-pair.v1"
Q4_PATH = Path(
    r"F:\reap-v2-q4-windows-full1319-phase34-original-20261027-0152"
    r"\reap128_v2_q4\gsm8k.jsonl"
)
Q8_PATH = Path(
    r"F:\reap-v2-q8-postload-full-complete-original-evidence-20261026-phase4"
    r"\reap-q8-reclaim-full-95876063255e45159f57e29306e0e885-gsm8k.jsonl"
)
Q4_ROWS_SHA256 = "66d0f39545dd8769e837ed5772a265871cfe148bb9a529ea2188454d72c1243f"
Q8_ROWS_SHA256 = "9ff2b418fecabdc419d94b2a06f6d9b115bbbd4bfba59ccae2d0d9f81e8ef749"
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class OriginalPairedEvidenceError(ValueError):
    """Original research evidence cannot be paired without ambiguity."""


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise OriginalPairedEvidenceError("duplicate original JSON key")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise OriginalPairedEvidenceError("nonfinite original JSON: " + value)


def load_pinned_rows(path: Path, expected_sha256: str) -> dict[str, dict]:
    """Parse only exact pinned bytes, with complete canonical item inventory."""
    try:
        metadata = path.lstat()
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
                or getattr(metadata, "st_file_attributes", 0) & _REPARSE
                or not 0 < metadata.st_size <= 8 * 1024 * 1024):
            raise OriginalPairedEvidenceError("original row source not a regular bounded file")
        raw = path.read_bytes()
    except OSError as exc:
        raise OriginalPairedEvidenceError("original row source unavailable") from exc
    if (len(raw) != metadata.st_size
            or hashlib.sha256(raw).hexdigest() != expected_sha256
            or not raw.endswith(b"\n")):
        raise OriginalPairedEvidenceError("original row bytes changed or incomplete")
    lines = raw.splitlines(keepends=True)
    if len(lines) != 1319:
        raise OriginalPairedEvidenceError("original row population incomplete")
    rows = {}
    for line in lines:
        if not line.endswith(b"\n") or line.startswith(b"\xef\xbb\xbf"):
            raise OriginalPairedEvidenceError("original row framing invalid")
        try:
            row = json.loads(line.decode("utf-8"), object_pairs_hook=_unique,
                             parse_constant=_nonfinite)
        except (UnicodeError, ValueError) as exc:
            raise OriginalPairedEvidenceError("original row JSON invalid") from exc
        if (type(row) is not dict or type(row.get("id")) is not str
                or row["id"] in rows):
            raise OriginalPairedEvidenceError("original row ID missing or duplicated")
        rows[row["id"]] = row
    if set(rows) != {f"gsm8k-{i}" for i in range(1319)}:
        raise OriginalPairedEvidenceError("original row IDs not canonical")
    return rows


def paired_certificate(q4: dict, q8: dict) -> dict:
    """Independently regrade two complete cohorts and pair exact task IDs."""
    canonical = grade.load_canonical()
    q4_grade = grade.verify_rows(q4, canonical)
    q8_grade = grade.verify_rows(q8, canonical)
    # A separate source-SHA-pinned implementation must agree with the
    # independent canonical grader for every item, not merely the totals.
    pairs = []
    counts = {"both_correct": 0, "windows_q4_only": 0,
              "linux_q8_only": 0, "neither_correct": 0}
    for task_id, _, expected in canonical:
        a, b = q4[task_id], q8[task_id]
        if (scorer.score_original_gsm8k(a["text"], expected,
                                        terminated=a["terminated"]) is not a["correct"]
                or scorer.score_original_gsm8k(b["text"], expected,
                                                terminated=b["terminated"]) is not b["correct"]):
            raise OriginalPairedEvidenceError(task_id + ": independent graders disagree")
        category = ("both_correct" if a["correct"] and b["correct"]
                    else "windows_q4_only" if a["correct"]
                    else "linux_q8_only" if b["correct"] else "neither_correct")
        counts[category] += 1
        pairs.append((task_id, a["correct"], b["correct"]))
    if (sum(counts.values()) != 1319
            or q4_grade["correct"] != counts["both_correct"] + counts["windows_q4_only"]
            or q8_grade["correct"] != counts["both_correct"] + counts["linux_q8_only"]):
        raise OriginalPairedEvidenceError("paired and independently graded counts disagree")
    discordant = counts["windows_q4_only"] + counts["linux_q8_only"]
    smaller = min(counts["windows_q4_only"], counts["linux_q8_only"])
    numerator = min(2 ** discordant,
                    2 * sum(math.comb(discordant, k) for k in range(smaller + 1)))
    denominator = 2 ** discordant
    divisor = math.gcd(numerator, denominator)
    return {
        "schema": SCHEMA,
        "population": 1319,
        "canonical_sha256": grade.FROZEN_GSM8K_SHA256,
        "q4_correct": q4_grade["correct"],
        "q8_correct": q8_grade["correct"],
        "q4_terminated": q4_grade["naturally_terminated"],
        "q8_terminated": q8_grade["naturally_terminated"],
        "paired_counts": counts,
        "paired_per_item_sha256": hashlib.sha256(json.dumps(
            pairs, ensure_ascii=False, separators=(",", ":"),
            allow_nan=False).encode("utf-8")).hexdigest(),
        "exact_two_sided_mcnemar_p_numerator": numerator // divisor,
        "exact_two_sided_mcnemar_p_denominator": denominator // divisor,
        "native_http_lineage_authenticated": False,
        "same_host_frozen_comparison": False,
        "research_only": True,
        "status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }


def verify_original_research() -> dict:
    """Use only the two source-owned original F: SHA pins."""
    certificate = paired_certificate(load_pinned_rows(Q4_PATH, Q4_ROWS_SHA256),
                                     load_pinned_rows(Q8_PATH, Q8_ROWS_SHA256))
    # Only this path authenticates the actual F: bytes. The lower-level
    # pairing function is also used on synthetic test rows and must not
    # falsely attach the real original-file identities to those specimens.
    certificate["q4_rows_sha256"] = Q4_ROWS_SHA256
    certificate["q8_rows_sha256"] = Q8_ROWS_SHA256
    return certificate


if __name__ == "__main__":
    print(json.dumps(verify_original_research(), sort_keys=True, indent=2))
