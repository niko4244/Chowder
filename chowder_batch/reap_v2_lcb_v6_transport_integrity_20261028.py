"""Read-only source-locked audit of LiveCodeBench generation transport failures.

This does not execute generated Python, contact a model endpoint, repair a
journal, or certify an original serving session. Failed HTTP requests are
*missing generations*, never wrong model answers or completed benchmark cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from chowder_batch import reap_v2_lcb_v6_checkpoint_readonly_audit_20261028 as lcb

SCHEMA = "chowder.reap-v2-lcb-v6-transport-integrity.v1"
BASE_ROW_KEYS = frozenset({
    "question_id", "source_file", "task", "prompt_style", "max_tokens",
    "temperature", "top_p", "raw", "code", "error", "elapsed_s",
})
SUCCESS_ROW_KEYS = BASE_ROW_KEYS | {
    "finish_reason", "prompt_tokens", "completion_tokens",
}


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    """Do not let JSON's last-key-wins behavior conceal contradictory receipts."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON journal key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON journal constant: {value}")


def parse_complete_journal(raw: bytes) -> list[dict]:
    """Reject torn writes rather than interpreting an in-flight row as a case."""
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("generation journal is empty or has an in-flight partial row")
    rows = []
    for index, line in enumerate(raw.splitlines(), 1):
        if not line:
            raise ValueError(f"blank generation row {index}")
        try:
            row = json.loads(
                line, object_pairs_hook=_unique_object,
                parse_constant=_reject_nonfinite,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid generation row {index}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"non-object generation row {index}")
        rows.append(row)
    return rows


def _extract_original_openai_chat_code(raw: str) -> str:
    """Mirror pinned LCB extraction_utils.extract_code(OpenAIChat), without imports.

    The frozen generator stores this exact deterministic extraction of its
    response. Never execute the extracted text or import evaluator runtime code.
    The original implementation selects the final two lines containing ```.
    """
    lines = raw.split("\n")
    fences = [index for index, line in enumerate(lines) if "```" in line]
    if len(fences) < 2:
        return ""
    return "\n".join(lines[fences[-2] + 1:fences[-1]])


def classify(source: list[tuple[str, str]], rows: list[dict]) -> dict:
    """Classify original canonical generations without inferring any scores."""
    if not rows or len(rows) > len(source):
        raise ValueError("empty or excessive generation population")
    first_error = None
    errors = 0
    successful_after_error = 0
    nonerror_empty = 0
    transport_failures = 0
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or "error" not in row:
            raise ValueError(f"missing original transport status at {i}")
        error = row["error"]
        required_keys = SUCCESS_ROW_KEYS if error is None else BASE_ROW_KEYS
        if set(row) != required_keys:
            raise ValueError(f"original generation row field set mismatch at {i}")
        elapsed = row["elapsed_s"]
        if type(elapsed) is not float or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError(f"invalid original generation elapsed time at {i}")
        qid, filename = source[i]
        if (row.get("question_id") != qid
                or row.get("source_file") != filename
                or row.get("task") != "LCB_generation"
                or row.get("prompt_style") != "OpenAIChat"
                or type(row.get("max_tokens")) is not int
                or row["max_tokens"] != 4096
                or type(row.get("temperature")) is not float
                or row["temperature"] != 0.2
                or type(row.get("top_p")) is not float
                or row["top_p"] != 0.95
                or not isinstance(row.get("code"), str)
                or not isinstance(row.get("raw"), str)):
            raise ValueError(f"canonical source/prompt/parameters mismatch at {i}")
        if error is None:
            # The frozen generator stores exactly extract_code(raw, OpenAIChat).
            # A plausible-looking response is not evidence for arbitrary code.
            if row["code"] != _extract_original_openai_chat_code(row["raw"]):
                raise ValueError(f"candidate code disagrees with original extraction at {i}")
            for field in ("prompt_tokens", "completion_tokens"):
                value = row[field]
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError(f"invalid original model token usage at {i}")
            if first_error is not None:
                successful_after_error += 1
            if not row["code"].strip():
                nonerror_empty += 1
            if not isinstance(row.get("finish_reason"), (str, type(None))):
                raise ValueError(f"invalid model finish reason at {i}")
        elif isinstance(error, str) and error.strip():
            if row["raw"] or row["code"]:
                raise ValueError(f"transport error has candidate text at {i}")
            if first_error is None:
                first_error = i
            errors += 1
            if ("URLError" in error and "WinError 10061" in error
                    and "actively refused" in error):
                transport_failures += 1
        else:
            raise ValueError(f"invalid transport error encoding at {i}")
    return {
        "source_population": len(source),
        "journal_rows": len(rows),
        "successful_transport_rows": len(rows) - errors,
        "model_responses_with_empty_code": nonerror_empty,
        "failed_transport_rows_not_scored": errors,
        "connection_refused_rows": transport_failures,
        "first_failed_source_index_zero_based": first_error,
        "first_failed_question_id": source[first_error][0] if first_error is not None else None,
        "successes_after_first_failure": successful_after_error,
        "contiguous_successful_prefix": first_error if first_error is not None else len(rows),
        "unresolved_source_cases": len(source) - (len(rows) - errors),
        "retry_requires_new_source_bound_session_and_independent_admission": bool(errors),
        "original_model_serving_session_independently_attested": False,
        "candidate_code_execution_authorized": False,
        "official_complete_score": None,
        "status": "TRANSPORT_FAILURES_UNSCORED" if errors else "NO_TRANSPORT_FAILURES_OBSERVED",
        "frozen_gen0_status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }


def inspect() -> dict:
    # The existing independent source/metric auditor validates Git identity,
    # all six original dataset SHAs, frozen generator/evaluator SHAs and the
    # official scored prefix. It never runs generated programs.
    checkpoint = lcb.inspect()
    source = []
    for name in lcb.FILES:
        source.extend((row["question_id"], name)
                      for row in lcb.read_jsonl(lcb.DATA / name))
    raw = (lcb.OUT / "gen_q4.jsonl").read_bytes()
    result = classify(source, parse_complete_journal(raw))
    if result["contiguous_successful_prefix"] < checkpoint["scored_prefix"]:
        raise ValueError("official checkpoint includes failed transport cases")
    if lcb.sha(lcb.OUT / "metrics_q4_checkpoint.json") != checkpoint["checkpoint_sha256"]:
        raise ValueError("official checkpoint changed during transport audit")
    return {
        "schema": SCHEMA,
        **result,
        "generation_journal_sha256": hashlib.sha256(raw).hexdigest(),
        "source_dataset_sha256": checkpoint["source_dataset_sha256"],
        "source_git_commit": checkpoint["source_git_commit"],
        "generator_source_sha256": checkpoint["generator_source_sha256"],
        "evaluator_source_sha256": checkpoint["evaluator_source_sha256"],
        "previously_scored_prefix_internal_consistency_only": checkpoint["scored_prefix"],
        "previously_scored_prefix_passes_internal_consistency_only": checkpoint["correct"],
        "official_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "cpu_code_isolation_independently_attested": False,
        "research_only": True,
    }


def _verify_original_journal(journal: Path, expected_sha256: str) -> None:
    """Recheck original evidence bytes and file identity before publication."""
    if not journal.is_file() or journal.is_symlink():
        raise ValueError("original generation journal missing or linked")
    before = journal.stat()
    with journal.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        finished = os.fstat(stream.fileno())
    after = journal.stat()
    identities = {(s.st_dev, s.st_ino) for s in (before, opened, finished, after)}
    if len(identities) != 1 or digest.hexdigest() != expected_sha256:
        raise ValueError("original generation journal changed before evidence seal")


def _seal_exclusive_snapshot(target: Path, journal: Path, result: dict) -> None:
    """Publish a complete certificate atomically, never a torn public file."""
    expected = result["generation_journal_sha256"]
    _verify_original_journal(journal, expected)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.pending-",
            delete=False,
        ) as stream:
            pending = Path(stream.name)
            stream.write((json.dumps(result, sort_keys=True, indent=2) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        _verify_original_journal(journal, expected)
        os.link(pending, target)  # exclusive, atomic, never overwrites
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    try:
        result = inspect()
        if args.snapshot is not None:
            target = args.snapshot
            if (not target.is_absolute() or target.exists() or target.is_symlink()
                    or target.parent.resolve() != Path("F:/").resolve()):
                raise ValueError("new exclusive F: snapshot path required")
            _seal_exclusive_snapshot(target, lcb.OUT / "gen_q4.jsonl", result)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "REFUSED_LCB_TRANSPORT_AUDIT",
                          "reason": str(exc)[:350], "frozen_gen0_status": "BLOCKED"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
