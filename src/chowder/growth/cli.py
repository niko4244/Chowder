"""CLI surface for the growth system.

Read-mostly by design: ``chowder eval catalog|scoreboard``,
``chowder data audit|contamination``, and ``chowder growth
profile|curriculum|status`` report on evidence already on disk or derived
mechanically from it. The one mutating path -- ``chowder data register`` --
requires every argument the data policy mandates and registers the source as
QUARANTINE, never directly trainable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .catalog import default_registry
from .data_registry import DataRegistry, DataSource, seed_registry


def _print_json(payload: Any) -> int:
    json.dump(payload, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


def _eval_catalog(_args: argparse.Namespace) -> int:
    registry = default_registry()
    rows = []
    for entry in registry:
        rows.append(
            {
                "benchmark": entry.qualified_id,
                "name": entry.name,
                "category": entry.category,
                "subcategory": entry.subcategory,
                "status": entry.status,
                "lifecycle": entry.lifecycle,
                "tier": entry.tier,
                "modality": entry.modality,
                "split_policy": entry.split_policy,
                "license": entry.license,
                "training_use_permitted": entry.training_use_permitted,
                "contamination_risk": entry.contamination_risk,
                "adapter": entry.adapter or None,
                "skills": list(entry.skills),
            }
        )
    rows.sort(key=lambda row: (row["category"], row["benchmark"]))
    return _print_json(
        {
            "count": len(rows),
            "runnable": len(registry.runnable()),
            "categories": list(registry.categories()),
            "benchmarks": rows,
        }
    )


def _eval_scoreboard(args: argparse.Namespace) -> int:
    from chowder.evals.result import EvalReport
    from chowder.evals.runner import Scoreboard

    report = EvalReport.load(Path(args.report))
    scoreboard = Scoreboard(default_registry())
    sys.stdout.write(scoreboard.render(report))
    return 0


def _load_data_registry(root: Path) -> DataRegistry:
    manifest = root / "growth" / "data-manifest.json"
    if not manifest.exists():
        return seed_registry()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    registry = DataRegistry()
    for row in data.get("sources", []):
        registry.register(DataSource(**row))
    return registry


def _save_data_registry(registry: DataRegistry, root: Path) -> None:
    manifest = root / "growth" / "data-manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {"sources": [s.to_dict() for s in registry.sources()]},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _data_audit(args: argparse.Namespace) -> int:
    registry = _load_data_registry(Path(args.root))
    rows = []
    for source in registry.sources():
        rows.append(
            {
                "source_id": source.source_id,
                "dataset_name": source.dataset_name,
                "revision": source.revision,
                "trust_class": source.trust_class,
                "verification": source.verification,
                "license": source.license,
                "permitted_training_use": source.permitted_training_use,
                "contamination": source.contamination_relationship,
                "inclusion_decision": source.inclusion_decision,
                "trainable": source.trainable,
            }
        )
    return _print_json(
        {
            "root": str(args.root),
            "total_sources": len(rows),
            "trainable_sources": [r["source_id"] for r in rows if r["trainable"]],
            "quarantine_count": sum(1 for r in rows if r["trust_class"] == "QUARANTINE"),
            "unknown_contamination": sum(
                1 for r in rows if r["contamination"] == "UNKNOWN"
            ),
            "sources": rows,
        }
    )


def _data_register(args: argparse.Namespace) -> int:
    """The single mutating data path. Registers as QUARANTINE."""
    from .discovery import Candidate, DataDiscovery

    if args.revision.strip().lower() in {"latest", "current", ""}:
        return _print_json(
            {
                "error": "revision must be pinned; 'latest' is refused by policy",
            }
        ) or 1
    registry = _load_data_registry(Path(args.root))
    discovery = DataDiscovery(registry=registry)
    candidate = Candidate(
        candidate_id=args.source_id,
        name=args.name,
        origin=args.origin,
        url=args.url,
        revision=args.revision,
        license_declared=args.license,
        domain=args.domain,
        language=tuple(lang.strip() for lang in args.language.split(",") if lang.strip()),
        source_type=args.source_type,
        example_count=args.examples,
        token_estimate=args.tokens,
    )
    discovery.discover(candidate)
    report = discovery.inspect(candidate.candidate_id)
    source = discovery.register(
        candidate.candidate_id,
        pii_reviewed=args.pii_reviewed,
        secrets_reviewed=args.secrets_reviewed,
        quality_score=args.quality,
        contamination_status="UNKNOWN",
        inclusion_reason=args.reason,
    )
    _save_data_registry(registry, Path(args.root))
    return _print_json(
        {
            "registered": source.source_id,
            "trust_class": source.trust_class,
            "inclusion_decision": source.inclusion_decision,
            "trainable": source.trainable,
            "inspection_blockers": report.blockers,
            "note": (
                "Registered as QUARANTINE/pending. Promote only through an explicit "
                "admit() decision backed by contamination and review evidence."
            ),
        }
    )


def _data_contamination(args: argparse.Namespace) -> int:
    from .contamination import ContaminationFirewall

    firewall = ContaminationFirewall()
    if args.manifest:
        for qualified_id, texts in json.loads(
            Path(args.manifest).read_text(encoding="utf-8")
        ).get("protected", {}).items():
            firewall.register_protected(qualified_id, texts)
    if args.text is not None:
        result = firewall.check_text(args.text)
        return _print_json(
            {
                "verdict": result.verdict,
                "clean": result.clean,
                "matches": [
                    {
                        "benchmark": m.benchmark_qualified_id,
                        "detector": m.detector,
                        "detail": m.detail,
                        "similarity": m.similarity,
                    }
                    for m in result.matches
                ],
            }
        )
    registry = _load_data_registry(Path(args.root))
    findings = {}
    for source in registry.sources():
        probe = f"{source.source_id} {source.dataset_name}"
        result = firewall.check_text(probe)
        findings[source.source_id] = {
            "verdict": result.verdict,
            "clean": result.clean,
        }
    return _print_json(
        {
            "root": str(args.root),
            "sources": findings,
            "not_clean": [sid for sid, f in findings.items() if not f["clean"]],
            "note": (
                "Registry names alone are a weak probe; the admission path checks "
                "real content samples through the same firewall before inclusion."
            ),
        }
    )


def _growth_profile(args: argparse.Namespace) -> int:
    from .capability import CapabilityProfile

    payload = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    profile = CapabilityProfile.from_dict(payload)
    return _print_json(profile.to_dict())

def _growth_curriculum(args: argparse.Namespace) -> int:
    from .capability import ALL_SKILLS, CapabilityProfile
    from .curriculum import CurriculumEngine

    payload = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    profile = CapabilityProfile.from_dict(payload)
    unknown = [s.skill for s in profile.skills if s.skill not in ALL_SKILLS]
    if unknown:
        return _print_json(
            {
                "error": "profile contains unknown skills",
                "unknown_skills": unknown,
                "note": "Skills must come from capability.ALL_SKILLS (the closed list).",
            }
        ) or 1
    engine = CurriculumEngine()
    plan = engine.plan(
        model_version=profile.model_version,
        profile=profile,
        protected_sets=tuple(args.protected.split(",") if args.protected else ()),
        budget_examples=args.budget_examples,
    )
    return _print_json(
        {
            "model_version": profile.model_version,
            "items": [item.to_dict() for item in plan],
            "note": (
                "Derived mechanically from the profile's measured evidence; "
                "every item records why it was selected."
            ),
        }
    )


def _growth_status(args: argparse.Namespace) -> int:
    outcome = json.loads(Path(args.ledger).read_text(encoding="utf-8"))
    summary = {
        "cycle_id": outcome.get("cycle_id"),
        "parent_version": outcome.get("parent_version"),
        "candidate_version": outcome.get("candidate_version"),
        "verdict": outcome.get("verdict"),
        "phases": outcome.get("phases"),
    }
    promotion = outcome.get("promotion") or {}
    summary["promotion_reasons"] = promotion.get("reasons")
    summary["checks"] = promotion.get("checks")
    return _print_json(summary)


def register_growth_subcommands(sub: argparse._SubParsersAction) -> None:
    """Attach eval/data/growth subcommands to the main chowder parser."""
    eval_parser = sub.add_parser("eval", help="Benchmark catalog and scoreboard tooling")
    eval_targets = eval_parser.add_subparsers(dest="eval_target", required=True)

    catalog = eval_targets.add_parser("catalog", help="List the pinned benchmark catalog")
    catalog.set_defaults(func=_eval_catalog)

    scoreboard = eval_targets.add_parser(
        "scoreboard", help="Render a scoreboard from an eval report"
    )
    scoreboard.add_argument("report", help="Path to an eval report JSON")
    scoreboard.set_defaults(func=_eval_scoreboard)

    data_parser = sub.add_parser("data", help="Training-data registry tooling")
    data_targets = data_parser.add_subparsers(dest="data_target", required=True)

    audit = data_targets.add_parser("audit", help="Audit registered data sources")
    audit.add_argument("--root", default=".", help="Workspace root holding growth/")
    audit.set_defaults(func=_data_audit)

    register = data_targets.add_parser(
        "register", help="Register a discovered source (enters as QUARANTINE)"
    )
    register.add_argument("--root", default=".", help="Workspace root holding growth/")
    register.add_argument("--source-id", required=True)
    register.add_argument("--name", required=True)
    register.add_argument(
        "--origin", required=True, help="huggingface|github|paper|manual"
    )
    register.add_argument("--url", required=True)
    register.add_argument(
        "--revision", required=True, help="Pinned revision; 'latest' refused"
    )
    register.add_argument("--license", required=True, dest="license")
    register.add_argument("--domain", required=True)
    register.add_argument(
        "--language", required=True, help="Comma-separated language tags"
    )
    register.add_argument(
        "--source-type",
        required=True,
        choices=("synthetic", "human", "web", "code", "paper", "mixed"),
    )
    register.add_argument("--examples", type=int, required=True)
    register.add_argument("--tokens", type=int, required=True)
    register.add_argument(
        "--quality", type=float, required=True, help="Quality score in [0,1]"
    )
    register.add_argument(
        "--reason", required=True, help="Why this source should be registered"
    )
    register.add_argument(
        "--pii-reviewed", action="store_true", help="PII review completed by a human"
    )
    register.add_argument(
        "--secrets-reviewed",
        action="store_true",
        help="Secrets review completed by a human",
    )
    register.set_defaults(func=_data_register)

    contamination = data_targets.add_parser(
        "contamination", help="Check data sources against the protected-set firewall"
    )
    contamination.add_argument("--root", default=".")
    contamination.add_argument(
        "--manifest", default=None, help="Protected-set manifest JSON to load"
    )
    contamination.add_argument(
        "--text", default=None, help="Check one free-text snippet instead of the registry"
    )
    contamination.set_defaults(func=_data_contamination)

    growth = sub.add_parser("growth", help="Model growth system status and planning")
    growth_targets = growth.add_subparsers(dest="growth_target", required=True)

    profile = growth_targets.add_parser("profile", help="Show a capability profile")
    profile.add_argument("profile", help="Path to a capability profile JSON")
    profile.set_defaults(func=_growth_profile)

    curriculum = growth_targets.add_parser(
        "curriculum", help="Derive a curriculum plan from a profile"
    )
    curriculum.add_argument("profile", help="Path to a capability profile JSON")
    curriculum.add_argument(
        "--protected",
        default="",
        help="Comma-separated protected benchmark ids carried as regression sets",
    )
    curriculum.add_argument("--budget-examples", type=int, default=20000)
    curriculum.set_defaults(func=_growth_curriculum)

    status = growth_targets.add_parser("status", help="Summarize a growth-cycle outcome")
    status.add_argument("ledger", help="Path to a cycle outcome JSON")
    status.set_defaults(func=_growth_status)

    campaign = growth_targets.add_parser(
        "campaign", help="Validate a campaign manifest and show admission/settlement"
    )
    campaign_targets = campaign.add_subparsers(dest="campaign_target", required=True)

    validate = campaign_targets.add_parser(
        "validate", help="Refuse a malformed manifest before any compute"
    )
    validate.add_argument("manifest", help="Path to the campaign manifest JSON")
    validate.set_defaults(func=_growth_campaign_validate)

    plan = campaign_targets.add_parser(
        "plan",
        help="Plan the declared campaign (curriculum + recipe ids) without compute",
    )
    plan.add_argument("manifest", help="Path to the campaign manifest JSON")
    plan.set_defaults(func=_growth_campaign_plan)

    readiness = campaign_targets.add_parser(
        "readiness",
        help="Check every pre-compute prerequisite without starting compute",
    )
    readiness.add_argument("manifest", help="Path to the campaign manifest JSON")
    readiness.set_defaults(func=_growth_campaign_readiness)

    prepare = campaign_targets.add_parser(
        "prepare",
        help="Emit the declaration's required inputs from durable evidence",
    )
    prepare.add_argument("manifest", help="Path to the campaign manifest JSON")
    prepare.add_argument(
        "--out-dir",
        required=True,
        help="Directory the prepared inputs are written into",
    )
    prepare.add_argument(
        "--parent-evidence",
        default="",
        help="The parent generation's durable run root (its candidate_evaluation.json)",
    )
    prepare.add_argument(
        "--write-declaration",
        default="",
        help="Write the declaration with the prepared inputs filled in to this path",
    )
    prepare.set_defaults(func=_growth_campaign_prepare)

    ancestor = campaign_targets.add_parser(
        "measure-ancestor",
        help="Measure the trusted-ancestor arm on the untouched dense base",
    )
    ancestor.add_argument("manifest", help="Path to the campaign manifest JSON")
    ancestor.add_argument(
        "--out",
        default="",
        help="Where to write the arm (default: the declared baseline_eval_report_path)",
    )
    ancestor.add_argument(
        "--placement",
        default="offload",
        help="Worker placement: resident or offload (default: offload)",
    )
    ancestor.set_defaults(func=_growth_campaign_measure_ancestor)

    parent_arm = campaign_targets.add_parser(
        "measure-parent",
        help="Measure the parent generation under this campaign's declared protocol",
    )
    parent_arm.add_argument("manifest", help="Path to the campaign manifest JSON")
    parent_arm.add_argument(
        "--out",
        default="",
        help="Where to write the arm (default: the declared parent_eval_report_path)",
    )
    parent_arm.add_argument(
        "--placement",
        default="offload",
        help="Worker placement: resident or offload (default: offload)",
    )
    parent_arm.set_defaults(func=_growth_campaign_measure_parent)

    run = campaign_targets.add_parser(
        "run", help="Execute the declared campaign through the real growth cycle"
    )
    run.add_argument("manifest", help="Path to the campaign manifest JSON")
    run.add_argument(
        "--state-root",
        default="",
        help="Override the manifest's state_root (testing/CI only)",
    )
    run.set_defaults(func=_growth_campaign_run)

    settle = campaign_targets.add_parser(
        "settle",
        help="Settle actual cycle cost from a compute-accounting artifact",
    )
    settle.add_argument("manifest", help="Path to the campaign manifest JSON")
    settle.add_argument(
        "accounting",
        help="Path to the cycle_compute_accounting.json produced by the cycle",
    )
    settle.set_defaults(func=_growth_campaign_settle)

    loop = growth_targets.add_parser(
        "loop",
        help="Advance generations unattended, one bounded campaign at a time",
    )
    loop_targets = loop.add_subparsers(dest="loop_target", required=True)

    loop_status = loop_targets.add_parser(
        "status", help="Show the durable state the loop decides from"
    )
    loop_status.add_argument(
        "state_root", help="The loop's durable state root (holds its JSONL history)"
    )
    loop_status.set_defaults(func=_growth_loop_status)

    loop_plan = loop_targets.add_parser(
        "plan", help="Show the target the loop would attempt next, without composing it"
    )
    _add_loop_arguments(loop_plan, parent_required=True)
    loop_plan.set_defaults(func=_growth_loop_plan)

    loop_run = loop_targets.add_parser(
        "run",
        help="Run generations until the loop reaches a terminal decision",
    )
    _add_loop_arguments(loop_run, parent_required=True)
    loop_run.set_defaults(func=_growth_loop_run)

    loop_resume = loop_targets.add_parser(
        "resume",
        help="Adopt the durable record of a session that already ended",
    )
    _add_loop_arguments(loop_resume, parent_required=True)
    loop_resume.set_defaults(func=_growth_loop_run, resume=True)


def _growth_campaign_validate(args: argparse.Namespace) -> int:
    """Load and validate a campaign manifest; print the declaration it pins."""
    from .campaign import CampaignManifest

    manifest = CampaignManifest.from_file(Path(args.manifest))
    return _print_json(
        {
            "status": "VALID",
            "cycle_id": manifest.cycle_id,
            "parent_version": manifest.parent_version,
            "target_benchmarks": list(manifest.target_benchmarks),
            "protected_benchmarks": list(manifest.protected_benchmarks),
            "broad_benchmarks": list(manifest.broad_benchmarks),
            "budget": {
                "device_gpu_hours_ceiling_per_recipe": manifest.budget.device_gpu_hours_ceiling_per_recipe,
                "wall_gpu_hours_ceiling_per_recipe": manifest.budget.wall_gpu_hours_ceiling_per_recipe,
                "device_gpu_hours_ceiling_campaign": manifest.budget.device_gpu_hours_ceiling_campaign,
                "wall_gpu_hours_ceiling_campaign": manifest.budget.wall_gpu_hours_ceiling_campaign,
            },
            "recipes": list(manifest.recipe_ids),
            "candidate_selection_policy": manifest.candidate_selection_policy,
            "promotion_policy_version": manifest.promotion_policy_version,
        }
    )


def _growth_campaign_plan(args: argparse.Namespace) -> int:
    """Print the recipes the declared inputs plan, so `recipes` can name them.

    A preregistration has to declare the recipe ids a run will execute, and the
    planner owns those ids; this is how a manifest author learns them without
    starting compute.
    """
    from .campaign import CampaignManifest
    from .campaign_runner import CampaignRunRefusal, plan_campaign

    manifest = CampaignManifest.from_file(Path(args.manifest))
    try:
        plan = plan_campaign(manifest)
    except CampaignRunRefusal as refusal:
        return _print_json(
            {
                "cycle_id": manifest.cycle_id,
                "status": "REFUSED",
                "refused_by": "campaign-plan",
                "refusal_reason": str(refusal),
            }
        ) or 1
    projected = sum(
        recipe.projected_wall_gpu_hours for recipe in plan.recipes
    )
    declared = set(manifest.recipe_ids)
    proposed = {recipe.recipe_id for recipe in plan.recipes}
    return _print_json(
        {
            "cycle_id": manifest.cycle_id,
            "status": "PLANNED",
            "curriculum_items": len(plan.items),
            "recipes": [recipe.recipe_id for recipe in plan.recipes],
            "declared_recipes": sorted(declared),
            "recipes_declared": sorted(declared) == sorted(proposed),
            "projected_wall_gpu_hours": projected,
            "wall_gpu_hours_ceiling_campaign": manifest.budget.wall_gpu_hours_ceiling_campaign,
            "plan": plan.to_dict(),
        }
    )


def _growth_campaign_readiness(args: argparse.Namespace) -> int:
    """Report every pre-compute prerequisite of a declared campaign.

    Zero compute by construction: it reads the declaration and the files it
    names and starts nothing. READY means a run will not refuse before it
    trains; the exit code is non-zero otherwise, so CI can gate on it.
    """
    from .campaign import CampaignManifest
    from .campaign_runner import check_campaign_readiness

    manifest = CampaignManifest.from_file(Path(args.manifest))
    report = check_campaign_readiness(manifest)
    _print_json(report.to_dict())
    return 0 if report.ready else 1


def _growth_campaign_prepare(args: argparse.Namespace) -> int:
    """Produce the declaration's required inputs from durable evidence.

    The seven inputs the run phase reads from disk (plus the contamination
    manifest) are emitted from evidence the repository already holds: a real
    device probe, the pinned dataset caches, the parent generation's own run
    root, and the production planner. Nothing is fabricated, so a prepared
    declaration's readiness stops reporting READINESS_DECLARED_INPUT for real.
    """
    from pathlib import Path as _Path

    from .campaign import CampaignManifest
    from .campaign_prepare import CampaignPrepareRefusal, prepare_campaign

    manifest = CampaignManifest.from_file(_Path(args.manifest))
    try:
        prepared = prepare_campaign(
            manifest,
            out_dir=args.out_dir,
            parent_evidence=args.parent_evidence or None,
        )
    except CampaignPrepareRefusal as refusal:
        return _print_json(
            {
                "cycle_id": manifest.cycle_id,
                "status": "REFUSED",
                "refused_by": "campaign-prepare",
                "refusal_reason": str(refusal),
            }
        ) or 1
    if args.write_declaration:
        document = json.loads(_Path(args.manifest).read_text(encoding="utf-8"))
        document.update(prepared.inputs)
        # The recipe ids depend on the measured hardware and the parent profile,
        # so they were not knowable when the declaration was written; the
        # planner's own proposals are what a run will execute.
        if prepared.recipe_ids:
            document["recipes"] = list(prepared.recipe_ids)
        _Path(args.write_declaration).write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return _print_json(
        {
            "cycle_id": prepared.cycle_id,
            "status": "PREPARED",
            "directory": str(prepared.directory),
            "inputs": dict(prepared.inputs),
            "recipe_ids": list(prepared.recipe_ids),
            "write_declaration": args.write_declaration,
            "notes": list(prepared.notes),
        }
    )


def _growth_campaign_measure_ancestor(args: argparse.Namespace) -> int:
    """Measure the trusted-ancestor (Gen-0) arm through the production worker.

    Real evaluation compute, not training: the untouched dense base is measured
    under the frozen 16-item protocol, and the arm is written to the declared
    baseline path with MEASURED_PARENT rows. Its cost is recorded as a prior
    evaluation job referenced at zero incremental campaign cost.
    """
    from pathlib import Path as _Path

    from .campaign import CampaignManifest
    from .campaign_prepare import CampaignPrepareRefusal, measure_ancestor_arm

    manifest = CampaignManifest.from_file(_Path(args.manifest))
    try:
        arm = measure_ancestor_arm(
            manifest,
            out_path=args.out or None,
            placement=args.placement,
        )
    except CampaignPrepareRefusal as refusal:
        return _print_json(
            {
                "cycle_id": manifest.cycle_id,
                "status": "REFUSED",
                "refused_by": "campaign-measure-ancestor",
                "refusal_reason": str(refusal),
            }
        ) or 1
    return _print_json({"cycle_id": manifest.cycle_id, "status": "MEASURED", **arm.to_dict()})


def _growth_campaign_measure_parent(args: argparse.Namespace) -> int:
    """Measure the parent generation under this campaign's declared protocol.

    The parent arm is a declared input read by adjudication. Measuring it here
    (the parent adapter over the declared base, under the declared instrument)
    is what gives the target comparison a real parent row; the rows are
    MEASURED_PARENT under the parent's own generation label.
    """
    from pathlib import Path as _Path

    from .campaign import CampaignManifest
    from .campaign_prepare import CampaignPrepareRefusal, measure_parent_arm

    manifest = CampaignManifest.from_file(_Path(args.manifest))
    try:
        arm = measure_parent_arm(
            manifest,
            out_path=args.out or None,
            placement=args.placement,
        )
    except CampaignPrepareRefusal as refusal:
        return _print_json(
            {
                "cycle_id": manifest.cycle_id,
                "status": "REFUSED",
                "refused_by": "campaign-measure-parent",
                "refusal_reason": str(refusal),
            }
        ) or 1
    return _print_json({"cycle_id": manifest.cycle_id, "status": "MEASURED", **arm.to_dict()})


def _growth_campaign_run(args: argparse.Namespace) -> int:
    """Run one manifest-driven campaign and print its mechanical outcome."""
    from .campaign import CampaignManifest
    from .campaign_runner import CampaignRunRefusal, run_campaign

    manifest = CampaignManifest.from_file(Path(args.manifest))
    try:
        run = run_campaign(
            manifest, state_root=args.state_root or manifest.state_root
        )
    except CampaignRunRefusal as refusal:
        return _print_json(
            {
                "cycle_id": manifest.cycle_id,
                "verdict": "REFUSED",
                "refused_by": "campaign-run",
                "refusal_reason": str(refusal),
            }
        ) or 1
    # A run that refused is a failure even though its record is durable: the
    # spend it may have made is written out, and the exit code still says the
    # campaign did not succeed.
    _print_json(run.to_dict())
    return 1 if run.verdict == "REFUSED" else 0


def _growth_campaign_settle(args: argparse.Namespace) -> int:
    """Settle the campaign's actual cost from its durable accounting artifact."""
    from .campaign import CampaignManifest, settle_campaign

    manifest = CampaignManifest.from_file(Path(args.manifest))
    document = json.loads(Path(args.accounting).read_text(encoding="utf-8"))
    totals = document.get("totals", {}).get("incremental", {})
    from .compute_cost import ComputeCost

    # The artifact says whether its device figure is a measurement; absent or
    # false means unmeasured, so a declared device ceiling cannot be settled
    # against a placeholder read back from disk.
    total = ComputeCost(
        device_gpu_hours=float(totals.get("device_gpu_hours", 0.0)),
        wall_gpu_hours=float(totals.get("wall_gpu_hours", 0.0)),
        source=str(args.accounting),
        device_measured=bool(totals.get("device_measured", False)),
    )
    verdict = settle_campaign(manifest, total=total)
    return _print_json(
        {
            "cycle_id": manifest.cycle_id,
            "budget_compliant": verdict.compliant,
            "budget_failure_reasons": list(verdict.failure_reasons),
            "actual": total.to_dict(),
            "ceilings": {
                "device_gpu_hours_ceiling_campaign": manifest.budget.device_gpu_hours_ceiling_campaign,
                "wall_gpu_hours_ceiling_campaign": manifest.budget.wall_gpu_hours_ceiling_campaign,
            },
        }
    )


# --------------------------------------------------------------------------
# the autonomous loop
# --------------------------------------------------------------------------

#: Terminal decisions that mean the loop stopped *correctly*: it either finished
#: improving, ran out of envelope, or stopped repeating itself. The remaining
#: terminals (UNCERTAIN, REVIEW, error states) are not success, and the exit code
#: has to say so -- a run that refused must not look like a run that promoted.
_LOOP_OK_DECISIONS = {"STOP_SUCCESS", "STOP_PLATEAU", "STOP_BUDGET"}


def _add_loop_arguments(parser: argparse.ArgumentParser, *, parent_required: bool) -> None:
    parser.add_argument("policy", help="Path to the loop policy JSON (the immutable envelope)")
    parser.add_argument(
        "--parent",
        required=parent_required,
        default="",
        help="The last trusted campaign declaration to advance from",
    )
    parser.add_argument(
        "--profile",
        default="",
        help="The parent's *measured* capability profile JSON",
    )
    parser.add_argument(
        "--parent-evidence",
        default="",
        help=(
            "The parent generation's durable run root; its own candidate "
            "evaluation is profiled rather than a profile file being trusted"
        ),
    )
    parser.add_argument(
        "--state-root",
        default="",
        help="The loop's durable state root (default: beside the parent's run root)",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=0,
        help="Advance at most this many generations (never more than the policy allows)",
    )


def _loop_state_root(args: argparse.Namespace, parent: Any) -> Path:
    if args.state_root:
        return Path(args.state_root)
    return Path(parent.state_root).parent / "growth-state"


def _loop_profile(args: argparse.Namespace) -> Any:
    """The parent's measured capability, from durable evidence or a stated file."""
    from .growth_loop import load_profile_from, profile_from_run_root

    if args.profile:
        return load_profile_from(args.profile)
    if args.parent_evidence:
        return profile_from_run_root(args.parent_evidence)
    return None


def _loop_refusal(reason: str) -> int:
    return _print_json(
        {"status": "REFUSED", "refused_by": "growth-loop", "refusal_reason": reason}
    ) or 1


def _growth_loop_status(args: argparse.Namespace) -> int:
    """Print what the loop would decide from, without deciding anything."""
    from .target_selection import GrowthState

    state = GrowthState(root=Path(args.state_root))
    rows = state.interventions()
    return _print_json(
        {
            "state_root": str(state.root),
            "stopping_state": state.stopping_state(),
            "generations_recorded": len(rows),
            "spent_wall_gpu_hours": sum(
                float(row.get("cost_gpu_hours", 0.0) or 0.0) for row in rows
            ),
            "targets": [
                {
                    "target_skill": row.get("target_skill", ""),
                    "training_type": row.get("training_type", ""),
                    "candidate_result": row.get("candidate_result", ""),
                    "measured_effect": row.get("measured_effect"),
                }
                for row in rows
            ],
            "banked_failures": len(state.failure_bank()),
            "repaired_classes": list(state.repaired_classes()),
        }
    )


def _growth_loop_plan(args: argparse.Namespace) -> int:
    """Print the target the loop would attempt next, and why -- no declaration written.

    The composition itself is owned by the loop (and its builder), so this prints
    the selector's proposal from the same evidence the loop would use rather than
    rebuilding a declaration here, where a second owner of the cycle identity
    would be able to disagree with the loop's.
    """
    from .campaign import CampaignManifest
    from .growth_loop import GrowthLoop
    from .next_campaign import LoopPolicy
    from .target_selection import GrowthState

    parent = CampaignManifest.from_file(Path(args.parent))
    profile = _loop_profile(args)
    if profile is None:
        return _loop_refusal(
            "no measured parent capability profile: pass --profile with the parent's "
            "profile JSON, or --parent-evidence pointing at a run root that holds a "
            "candidate_evaluation.json; a target chosen from an unmeasured profile is "
            "not evidence-backed"
        )
    state = GrowthState(root=_loop_state_root(args, parent))
    loop = GrowthLoop(
        policy=LoopPolicy.from_file(Path(args.policy)),
        state=state,
        parent_declaration=parent,
        parent_profile=profile,
    )
    proposal = loop.selector.propose(
        parent_version=parent.resolved_candidate_version(),
        profile=profile,
        state=state,
        known_skills=tuple(estimate.skill for estimate in profile.estimates),
    )
    return _print_json(
        {
            "parent_version": parent.resolved_candidate_version(),
            "proposal": proposal.to_dict(),
            "state_root": str(state.root),
        }
    )


def _growth_loop_run(args: argparse.Namespace) -> int:
    """Advance generations until the loop stops, and report how it stopped."""
    from .campaign import CampaignManifest
    from .growth_loop import GrowthLoop
    from .next_campaign import LoopPolicy
    from .target_selection import GrowthState

    parent = CampaignManifest.from_file(Path(args.parent))
    profile = _loop_profile(args)
    resume = bool(getattr(args, "resume", False))
    if profile is None and not resume:
        return _loop_refusal(
            "no measured parent capability profile: pass --profile with the parent's "
            "profile JSON, or --parent-evidence pointing at a run root that holds a "
            "candidate_evaluation.json; a target chosen from an unmeasured profile is "
            "not evidence-backed"
        )
    loop = GrowthLoop(
        policy=LoopPolicy.from_file(Path(args.policy)),
        state=GrowthState(root=_loop_state_root(args, parent)),
        parent_declaration=parent,
        parent_profile=profile,
    )
    report = loop.run(
        max_generations=args.max_generations or None,
        resume=resume,
    )
    payload = report.to_dict()
    payload["decision_ok"] = report.decision.action in _LOOP_OK_DECISIONS
    _print_json(payload)
    return 0 if payload["decision_ok"] else 1


__all__ = ["register_growth_subcommands"]
