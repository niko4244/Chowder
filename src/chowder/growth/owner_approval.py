"""Explicit, artifact-bound operator confirmation before a lineage promotion.

An algorithmic PROMOTED verdict means a candidate passed its machine gates;
it is NOT the owner's approval to advance the lineage. No environment variable
or non-interactive default is an authorization mechanism. A deployment wanting
headless signed approvals must add a pinned, independently verified owner key;
this module does not pretend a TTY proves the operator's real-world identity.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path

_SECRET = secrets.token_bytes(32)


def requires_owner_approval(model_path: str | Path) -> bool:
    """Recognize the REAP lineage from real checkpoint facts, not an env toggle.

    REAP-128/160 both inherit Qwen3.5-MoE's composite text config. The
    ``reap-prune.json`` marker is positive evidence, but its absence is *not* an
    opt-out: recognized 128/160-expert Qwen3.5-MoE bases also require approval.
    Other lineages retain their independently preregistered policies.
    """
    base = Path(model_path)
    if (base / "reap-prune.json").exists():
        return True
    config = base / "config.json"
    if not config.is_file():
        return False
    try:
        data = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return False
    if not isinstance(data, dict) or data.get("model_type") != "qwen3_5_moe":
        return False
    text = data.get("text_config")
    if not isinstance(text, dict):
        return False
    return text.get("num_experts") in (128, 160)


def judge_certifies_candidate(_run_root: Path) -> bool:
    """N3/N5 gate: no owner approval before the frozen REAP judge certifies.

    The current frozen judge requires campaign-run.json, which the current
    production runner publishes *after* finalization. There is no trustworthy
    pre-finalization judge receipt yet. This MUST fail closed rather than
    mistaking branch-protection certification for the distinct frozen judge.
    A future two-phase judge-then-release transaction can replace this seam.
    CPU fixtures patch it only when explicitly exercising the approved branch.
    """
    return False


@dataclass(frozen=True)
class ApprovalGrant:
    cycle_id: str
    candidate_version: str
    artifact_sha256: str
    judge_certified: bool
    mac: str


def _grant(
    cycle_id: str, candidate_version: str, artifact_sha256: str, *, judge_certified: bool = False
) -> ApprovalGrant:
    """Internal mint; tests inject a real token through their fixture seam."""
    payload = (
        f"{len(cycle_id)}:{cycle_id}:{len(candidate_version)}:{candidate_version}:"
        f"{artifact_sha256}:{int(judge_certified)}"
    )
    mac = hmac.new(_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return ApprovalGrant(cycle_id, candidate_version, artifact_sha256, judge_certified, mac)


def verify_grant(
    grant: ApprovalGrant | None, *, cycle_id: str, candidate_version: str, artifact_sha256: str
) -> bool:
    """Reject forgery, replay for another campaign, or another artifact."""
    if not isinstance(grant, ApprovalGrant) or not grant.judge_certified or len(artifact_sha256) != 64:
        return False
    if not all(ch in "0123456789abcdef" for ch in artifact_sha256):
        return False
    if (
        grant.cycle_id != cycle_id
        or grant.candidate_version != candidate_version
        or grant.artifact_sha256 != artifact_sha256
    ):
        return False
    expected = _grant(cycle_id, candidate_version, artifact_sha256, judge_certified=True)
    return hmac.compare_digest(expected.mac, grant.mac)


def authorize_promotion(
    *, cycle_id: str, candidate_version: str, artifact_sha256: str,
    judge_certified: bool = False,
) -> ApprovalGrant | None:
    """Ask the local human explicitly, after all machine gates have passed.

    In a headless service, EOF, or a non-TTY process, the answer is *no*.
    No CHOWDER_* env variable or pytest detection bypasses this requirement.
    """
    if not verify_grant(_grant(cycle_id, candidate_version, artifact_sha256,
                               judge_certified=judge_certified),
                        cycle_id=cycle_id, candidate_version=candidate_version,
                        artifact_sha256=artifact_sha256):
        return None
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    phrase = f"APPROVE {cycle_id} {candidate_version} {artifact_sha256}"
    sys.stdout.write(
        "\nHUMAN APPROVAL REQUIRED: the model passed mechanical promotion gates.\n"
        "This is a lineage advancement, not a candidate evaluation.\n"
        f"To authorize exactly this candidate, type:\n{phrase}\n> "
    )
    sys.stdout.flush()
    try:
        typed = sys.stdin.readline(len(phrase) + 3)
    except (OSError, UnicodeError):
        return None
    return (
        _grant(cycle_id, candidate_version, artifact_sha256, judge_certified=True)
        if typed.rstrip("\r\n") == phrase else None
    )
