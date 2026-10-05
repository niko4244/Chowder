"""The inputs a run may read, and the refusals for the ones it may not.

A path nobody declared is not given a default, because a default is a policy
nobody preregistered. ``undeclared_inputs`` reports which declarations a phase
needs and the manifest does not make; ``require_declared_inputs`` raises. The
rest of the controllers resolve every path through ``_require_path`` so a
typo fails at the point of use, named.

Extracted from ``campaign_runner``, which keeps the orchestration and the
two injection seams. Nothing here re-implements a decision another module owns.
"""

from __future__ import annotations

from typing import Mapping

from pathlib import Path
from ..campaign_controllers.contracts import CampaignRunRefusal, FIELD_ENFORCEMENT, NON_BEHAVIORAL_FIELDS, assert_every_field_enforced


from ..campaign import PROMOTION_POLICY_VERSION, CampaignManifest, CampaignManifestError, settle_campaign, settle_campaign_projection, stops_on_admission_refusal, stops_on_campaign_overrun
from ..candidate_search import CandidateSearchRefusal, CandidateSearchDeclaration, SearchPlan, plan_search, run_search
from ..contamination import ContaminationFirewall



DECLARED_INPUT_REQUIREMENTS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "plan": (
        ("parent_profile_path", "a curriculum cannot be planned from nothing"),
        ("hardware_budget_path", "recipes are projected against measured hardware, never guesses"),
    ),
    "run": (
        ("project_template_path", "the executor has no project to compose"),
        ("training_material_path", "the executor writes the corpus this run trains on"),
        ("data_registry_path", "nothing may train on an unadmitted source"),
        ("hardware_budget_path", "recipes are projected against measured hardware, never guesses"),
        ("parent_profile_path", "a curriculum cannot be planned from nothing"),
        (
            "contamination_manifest_path",
            "rows cannot be bound against an unchecked firewall, so every row would bind UNKNOWN",
        ),
        (
            "evaluation_material_path",
            "the production evaluator has no data to measure the selected candidate "
            "on, so the run would spend its training compute and refuse afterwards",
        ),
        (
            "parent_eval_report_path",
            "the promotion rule compares the candidate against its parent, so a run "
            "without that arm can only reach INCONCLUSIVE",
        ),
    ),
}


def undeclared_inputs(manifest: CampaignManifest, *, phase: str) -> tuple[str, ...]:
    """Declared-input fields the phase needs that the manifest leaves empty."""
    return tuple(
        field
        for field, _why in DECLARED_INPUT_REQUIREMENTS[phase]
        if not str(getattr(manifest, field)).strip()
    )


def require_declared_inputs(manifest: CampaignManifest, *, phase: str) -> None:
    """Refuse with *every* undeclared input of the phase, not just the first.

    A pre-compute checklist is the honest form of this refusal: the operator
    learns what the campaign still has to declare before any compute starts.
    """
    missing = undeclared_inputs(manifest, phase=phase)
    if not missing:
        return
    reasons = "; ".join(
        f"{field} ({why})"
        for field, why in DECLARED_INPUT_REQUIREMENTS[phase]
        if field in missing
    )
    raise CampaignRunRefusal(
        f"the campaign leaves {len(missing)} declared input(s) the {phase} phase "
        f"requires undeclared: {reasons}; a campaign runs from fully declared "
        "inputs, so name them rather than letting the runner substitute a default"
    )


def _require_path(value: str, field: str, *, purpose: str) -> Path:
    if not value:
        raise CampaignRunRefusal(
            f"the manifest declares no {field}, so {purpose} cannot happen; "
            "declare the path rather than letting the runner guess one"
        )
    path = Path(value)
    if not path.exists():
        raise CampaignRunRefusal(f"{field} {value!r} does not exist ({purpose})")
    return path
