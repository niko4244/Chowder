"""Read-only original LCB v6 generation-log/journal invocation-boundary audit.

The original generator resumes by skipping existing IDs without authenticating
their model or serving session. Correlating its log with the complete journal
can establish a prior-journal boundary, NOT homogeneous model provenance.
Never execute generated code, contact a model, or authorize score pooling.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from chowder_batch import reap_v2_lcb_v6_checkpoint_readonly_audit_20261028 as lcb
from chowder_batch import reap_v2_lcb_v6_recovery_manifest_20261028 as recovery
from chowder_batch import reap_v2_lcb_v6_transport_integrity_20261028 as transport

SCHEMA = "chowder.reap-v2-lcb-v6-invocation-lineage-research.v1"
LOG = Path("F:/lcb-gen-full.log")
HEADER = re.compile(
    r"^total_v6=(\d+) already=(\d+) settings temp=0\.2 top_p=0\.95 "
    r"max_tokens=4096 style=OpenAIChat$"
)
PROGRESS = re.compile(
    r"^\[(\d+)/(\d+)\] (\S+) finish=(\S+) code=(\d+) "
    r"([0-9]+(?:\.[0-9]+)?)s err=(.*)$"
)


def _original_framed_log_lines(log_bytes: bytes) -> list[str]:
    """Accept only consistently LF/CRLF-framed original UTF-8 log records.

    ``str.splitlines`` silently accepts bare CR, VT, FF and Unicode line
    separators. Those are not original generator log record boundaries and
    must not manufacture an apparently complete invocation lineage.
    """
    if (not log_bytes.endswith(b"\n") or log_bytes.startswith(b"\xef\xbb\xbf")):
        raise ValueError("original invocation log newline/UTF-8 framing invalid")
    style = None
    lines = []
    for framed in log_bytes.split(b"\n")[:-1]:
        current = "CRLF" if framed.endswith(b"\r") else "LF"
        if style is not None and current != style:
            raise ValueError("mixed original invocation log newline framing")
        style = current
        raw_line = framed[:-1] if current == "CRLF" else framed
        if (not raw_line or b"\r" in raw_line or b"\x0b" in raw_line
                or b"\x0c" in raw_line or b"\x00" in raw_line):
            raise ValueError("nonoriginal invocation log record framing")
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("generation log is not valid UTF-8") from exc
        if any(char in line for char in ("\x85", "\u2028", "\u2029")):
            raise ValueError("nonoriginal Unicode invocation log newline")
        lines.append(line)
    return lines


def audit_invocation_log(
    log_bytes: bytes, rows: list[dict], *, expected_population: int = 1055,
) -> dict:
    """Prove log-to-row agreement without treating preexisting rows as this run."""
    if not log_bytes or not rows:
        raise ValueError("complete log and original journal rows required")
    lines = _original_framed_log_lines(log_bytes)
    if len(lines) < 2 or lines[-1] != "GEN_DONE":
        raise ValueError("generation log lacks complete termination")
    header = HEADER.fullmatch(lines[0])
    if header is None:
        raise ValueError("original generation invocation header mismatch")
    total, already = (int(value) for value in header.groups())
    if total != len(rows) or total != expected_population or not 0 <= already < total:
        raise ValueError("invocation and complete canonical population disagree")
    if len(lines) != total - already + 2:
        raise ValueError("invocation log progress population mismatch")

    responded = failed = 0
    for index in range(already, total):
        match = PROGRESS.fullmatch(lines[index - already + 1])
        if match is None:
            raise ValueError(f"malformed invocation log at source index {index}")
        observed_index, observed_total, qid, finish, code_length, elapsed, error = (
            match.groups()
        )
        row = rows[index]
        if (int(observed_index) != index + 1 or int(observed_total) != total
                or qid != row.get("question_id")
                or finish != str(row.get("finish_reason"))
                or int(code_length) != len(row.get("code", ""))
                or elapsed != str(row.get("elapsed_s"))
                or error != str(row.get("error"))):
            raise ValueError(f"generation log/journal disagreement at source index {index}")
        if row["error"] is None:
            responded += 1
        else:
            failed += 1

    return {
        "schema": SCHEMA,
        "status": "INVOCATION_BOUNDARY_CORRELATED_RESEARCH_ONLY",
        "generation_log_sha256": hashlib.sha256(log_bytes).hexdigest(),
        "canonical_population": total,
        "preexisting_journal_rows_session_unattributed": already,
        "observed_invocation_appended_rows": total - already,
        "observed_invocation_model_responses_unattested": responded,
        "observed_invocation_transport_failures_unscored": failed,
        "preexisting_rows_same_model_or_session_proven": False,
        "invocation_model_session_independently_attested": False,
        "cross_session_pooling_allowed": False,
        "official_score": None,
        "frozen_gen0": {
            "status": "BLOCKED", "chosen": None, "release_approved": False,
        },
    }


def inspect() -> dict:
    """Independently recheck source, full journal, and original invocation log."""
    inventory = recovery.inspect()
    journal = lcb.OUT / "gen_q4.jsonl"
    raw = journal.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != inventory["journal_sha256"]:
        raise ValueError("original generation journal changed after source audit")
    rows = transport.parse_complete_journal(raw)
    log_bytes = LOG.read_bytes()
    result = audit_invocation_log(log_bytes, rows)
    if (result["observed_invocation_transport_failures_unscored"]
            + result["observed_invocation_model_responses_unattested"]
            + result["preexisting_journal_rows_session_unattributed"]
            != inventory["canonical_population"]):
        raise ValueError("generation invocation accounting mismatch")
    if (lcb.sha(journal) != digest
            or lcb.sha(LOG) != result["generation_log_sha256"]):
        raise ValueError("original generation evidence changed during lineage audit")
    return {
        **result,
        "original_journal_sha256": digest,
        "source_git_commit": inventory["source_git_commit"],
        "generator_source_sha256": inventory["generator_source_sha256"],
        "evaluator_source_sha256": inventory["evaluator_source_sha256"],
        "source_dataset_sha256": inventory["source_dataset_sha256"],
        "original_checkpoint_consistency_only": inventory[
            "checkpoint_consistency_only"],
    }


def main() -> int:
    try:
        print(json.dumps(inspect(), sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "REFUSED_LCB_INVOCATION_LINEAGE",
                          "reason": str(exc)[:350], "frozen_gen0_status": "BLOCKED"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
