"""What an attempt cost, what it claims, and whether the run may trust it.

``_verify_selected_artifact`` re-hashes the selected artifact so selection
cannot promote bytes it did not choose; ``_attempt_summary`` reduces an
attempt to the fields a human reads; ``_preflight_arms`` checks the declared
parent and ancestor arms exist before compute rather than after.

Extracted from ``campaign_runner``, which keeps the orchestration and the
two injection seams. Nothing here re-implements a decision another module owns.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from pathlib import Path
from chowder.evals.result import BenchmarkRun, EvalReport
from ..campaign import PROMOTION_POLICY_VERSION, CampaignManifest, CampaignManifestError, settle_campaign, settle_campaign_projection, stops_on_admission_refusal, stops_on_campaign_overrun
from ..certification import FAIL, PASS, UNKNOWN, Certification, CertificationRow, ProtectionPolicy, certify_run_root, digest_of
from ..compute_cost import ComputeCost, CycleCostLedger
from ..campaign_controllers.contracts import CampaignRunRefusal, FIELD_ENFORCEMENT, NON_BEHAVIORAL_FIELDS, assert_every_field_enforced
from ..campaign_controllers.declared import DECLARED_INPUT_REQUIREMENTS, _require_path, require_declared_inputs, undeclared_inputs

from ..candidate_search import CandidateSearchRefusal, CandidateSearchDeclaration, SearchPlan, plan_search, run_search
from ..contamination import ContaminationFirewall


def _attempt_cost(evidence: Mapping[str, Any]) -> ComputeCost:
    recorded = evidence.get("actual_cost")
    if isinstance(recorded, Mapping):
        return ComputeCost.from_dict(recorded)
    measured = evidence.get("measured_gpu_hours")
    return ComputeCost.from_wall_only(
        float(measured or 0.0), source=f"attempt:{evidence.get('attempt', 'unknown')}"
    )


def _attempt_summary(results: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    """The durable facts about what each attempt produced, refusal included."""
    return tuple(
        {
            "recipe_id": evidence.get("recipe_id"),
            "attempt": evidence.get("attempt"),
            "status": evidence.get("status"),
            "refused_by": evidence.get("refused_by"),
            "refusal_reason": evidence.get("refusal_reason"),
            "failure_reason": evidence.get("failure_reason"),
            "artifact_ref": evidence.get("artifact_ref"),
            "artifact_sha256": evidence.get("artifact_sha256"),
            "measured_gpu_hours": evidence.get("measured_gpu_hours"),
            # Absent for a single-pass campaign; the round an attempt belonged
            # to when a declared search ran it.
            "search_round": evidence.get("search_round"),
            "search_max_steps": evidence.get("search_max_steps"),
        }
        for evidence in results
    )


def _verify_selected_artifact(
    selected: Mapping[str, Any] | None, *, purpose: str
) -> None:
    """Re-read the selected artifact and require its bytes to be unchanged.

    A digest recorded at training time is a claim about bytes that may since
    have moved. Production re-derives it from disk at both boundaries that
    matter -- before the artifact is measured, and before a verdict is recorded
    -- so a run can never certify a digest the frozen judge would later
    recompute and reject.
    """
    from ..campaign_runner import CANDIDATE_ARTIFACT_DIGEST_STALE

    if not selected:
        return
    artifact_ref = str(selected.get("artifact_ref") or "")
    recorded = str(selected.get("artifact_sha256") or "")
    if not artifact_ref or not recorded:
        return
    path = Path(artifact_ref)
    if not path.exists():
        raise CampaignRunRefusal(
            f"{CANDIDATE_ARTIFACT_DIGEST_STALE}: the selected artifact "
            f"{artifact_ref!r} no longer exists, so there is nothing to verify "
            f"{purpose}"
        )
    observed = digest_of(path)
    if observed != recorded:
        raise CampaignRunRefusal(
            f"{CANDIDATE_ARTIFACT_DIGEST_STALE}: the selected artifact "
            f"{artifact_ref!r} hashes to {observed}, but the run recorded "
            f"{recorded}; the bytes changed, so nothing measured against the "
            f"recorded digest can be trusted {purpose}"
        )


