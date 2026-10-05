"""What a campaign declaration is *for*.

``FIELD_ENFORCEMENT`` is the table that makes a declaration non-decorative:
every field the manifest schema declares must name the behavior it drives,
and ``assert_every_field_enforced`` refuses to run if the schema and the table
diverge. ``CampaignRunRefusal`` is the one way a run stops: a missing input, an
unverifiable identity, or a declaration this code cannot honor.

Extracted from ``campaign_runner``, which keeps the orchestration and the
two injection seams. Nothing here re-implements a decision another module owns.
"""

from __future__ import annotations

from typing import Mapping

from ..campaign import PROMOTION_POLICY_VERSION, CampaignManifest, CampaignManifestError, settle_campaign, settle_campaign_projection, stops_on_admission_refusal, stops_on_campaign_overrun
from ..candidate_search import CandidateSearchRefusal, CandidateSearchDeclaration, SearchPlan, plan_search, run_search
from ..contamination import ContaminationFirewall



FIELD_ENFORCEMENT: Mapping[str, str] = {
    "cycle_id": "names the run, its durable record and its ledger entry",
    "parent_version": "the generation the candidate is compared against",
    "candidate_version": "the generation the verdict is recorded under (derived when empty)",
    "base_model_path": "hashed and compared with base_model_digest before any compute",
    "base_model_digest": "must equal the dense base tree's model-content digest (payload files only), else the run refuses",
    "parent_adapter_path": "hashed and compared with parent_adapter_digest when declared",
    "parent_adapter_digest": "must equal the parent adapter tree's directory digest, else the run refuses",
    "state_root": "attempts, registry, ledger, accounting, the judged evidence set and campaign-run.json live here",
    "target_benchmarks": "the cycle's target set: the improvement gate",
    "protected_benchmarks": "the cycle's protected set: the regression gate",
    "broad_benchmarks": "the cycle's broad battery: no material deterioration",
    "calibration_benchmarks": "the cycle's calibration set: hard gate when declared",
    "reliability_benchmarks": "the cycle's reliability set: hard gate when declared",
    "budget": "per-recipe ceilings become the executor's admission envelope; the campaign ceilings refuse a plan that does not fit and settle the actual cost after compute",
    "recipe_ids": "the recipe set: the planner proposes this many, and an unproposed id refuses",
    "candidate_selection_policy": "the selection order passed to select_candidate",
    "stopping_rules": "each rule must name a behavior that refuses: admission refusal ends the campaign before compute, an overrun rule stops once a ceiling is breached, the contamination rule is the firewall's own refusal, and the threshold rule is the code's frozen constants; any rule outside campaign.STOPPING_RULE_ENFORCEMENT refuses at load",
    "promotion_policy_version": "must be the policy this code implements",
    "contamination_manifest_path": "loaded into the MetricBinder; a declared-but-missing file refuses",
    "project_template_path": "the project template the executor composes each attempt from",
    "training_material_path": "curriculum item -> source and text material; a missing item refuses",
    "data_registry_path": "the admitted data sources the executor may draw on",
    "hardware_budget_path": "the measured local budget the recipe planner projects against",
    "parent_profile_path": "the parent capability profile the curriculum is planned from",
    "parent_eval_report_path": "the parent side of adjudication, and the parent arm of the judged evidence set",
    "evaluation_material_path": "the dataset, fields and scoring the production evaluator measures the selected candidate with; a declared-but-missing file, or one that names no dataset for a declared benchmark, refuses before compute",
    "baseline_eval_report_path": "the trusted-ancestor (gen0) arm of the judged evidence set: branch protection is judged against it, never against an unresolved parent",
    "protection": "the declared branch-protection policy (trusted ancestor version + slice regression tolerance) the certification gate applies before any lineage record is written",
    "evaluation_execution": "how many rows one generate call decodes, for every arm and for the candidate: the candidate evaluator and both arm measurements read this one value, and each records it in its own evidence; batching changes generated tokens (measured: 14/16 rows agree with a single-row pass), so the arms and the candidate must share it and never be chosen per path",
    "candidate_search": "the declared bounded candidate search: its rounds, starting step budget, multiplier and survival rule are preregistered here, its worst-case cost is projected before any compute and must fit its own declared device/wall envelope *and* the campaign's ceilings, and the runner refuses a declared search that does not. Absent (rounds=0) means one pass over the declared recipes, which is what every manifest predating it does",
    "retention_profile": "the campaign's preregistered promotion gates: the cycle evaluates every constraint fail-closed before anything promotes (a target win over a constrained regression is REJECTED, an unmeasured constraint is a violation), and a constraint naming a benchmark the declared sets never measure refuses at load",
    "eval_tier_policy": "the declared trust classification of the campaign's benchmarks: a retention constraint measured on search-readable evidence refuses — at load when both are declared, and again at promotion",
    "notes": "documentation only: it drives no behavior and gates nothing",
}


NON_BEHAVIORAL_FIELDS = frozenset({"notes"})


class CampaignRunRefusal(RuntimeError):
    """The campaign could not be executed as declared.

    Raised before or around compute for a missing input, an unverifiable
    identity, or a declaration the runner cannot honor. A refusal is never
    converted into a default.
    """


def assert_every_field_enforced(manifest_cls: type = CampaignManifest) -> None:
    """The schema and the enforcement table must describe the same fields."""
    schema = set(getattr(manifest_cls, "__dataclass_fields__"))
    declared = set(FIELD_ENFORCEMENT)
    if schema != declared:
        missing = sorted(schema - declared)
        extra = sorted(declared - schema)
        raise CampaignManifestError(
            "campaign manifest schema and FIELD_ENFORCEMENT disagree "
            f"(unlisted fields: {missing}; unknown fields: {extra}); a declared "
            "field the runner cannot act on would be silently ignored"
        )
    undocumented = sorted(set(FIELD_ENFORCEMENT) - NON_BEHAVIORAL_FIELDS)
    if not undocumented:
        raise CampaignManifestError("FIELD_ENFORCEMENT lists no behavioral fields")
