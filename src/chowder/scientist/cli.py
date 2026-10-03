"""CLI for scientist mode: thin clients of ModelResearchService.

One service, one implementation — the (future) TUI research workspace will
call the same object, exactly as growth's TUI screen shares
AutonomousGrowthService with its CLI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .research_service import ModelResearchService, ResearchServiceError, default_provider
from .mission import ResearchMission


def _print_json(payload: Any) -> int:
    print(json.dumps(payload, indent=2, default=str))
    return 0


def _load_documents(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    mission_doc = json.loads(Path(args.mission).read_text(encoding="utf-8"))
    policy_doc = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    return mission_doc, policy_doc


def _build_service(args: argparse.Namespace) -> ModelResearchService:
    mission_doc, policy_doc = _load_documents(args)
    return ModelResearchService.from_policy(
        mission_document=mission_doc,
        scientist_policy=policy_doc,
        state_root=Path(args.state_root),
        run_exists=_registry_run_exists,
        run_complete=_registry_run_complete,
    )


def _registry_run_exists(run_id: str) -> bool:
    """Wire the resolvers to the production run registry when the run id
    points at an existing evidence root; otherwise refuse (fail-closed)."""
    from pathlib import Path as _P
    candidate = _P(str(run_id))
    return candidate.exists()


def _registry_run_complete(run_id: str) -> bool:
    from pathlib import Path as _P
    root = _P(str(run_id))
    if not root.exists():
        return False
    # A run root is 'complete' when its result record exists — the same
    # convention the repair pipeline uses for a settled run.
    return (root / "row_result.json").exists() or (root / "result.json").exists()


def _status(args: argparse.Namespace) -> int:
    service = _build_service(args)
    return _print_json(service.status().to_dict())


def _plan(args: argparse.Namespace) -> int:
    service = _build_service(args)
    portfolio = service.generate_portfolio(count=args.count)
    compiled = service.request_experiments()
    payload = {
        "hypotheses": [h.to_dict() for h in portfolio],
        "proposals_admitted": [p.to_dict() for p, _ in compiled],
        "experiments": [e.to_dict() for _, e in compiled],
        "next_decision": service.next_decision().to_dict(),
    }
    return _print_json(payload)


def _run(args: argparse.Namespace) -> int:
    service = _build_service(args)
    # The headless research session: portfolio -> admission -> (execution
    # happens through the production path; the session records observations
    # fed to it) -> findings -> next decision. Execution itself is NOT
    # triggered here: no GPU work starts from a CLI flag the policy did not
    # authorize.
    portfolio = service.generate_portfolio(count=args.count)
    compiled = service.request_experiments()
    decisions = []
    for _ in range(args.max_steps):
        decision = service.next_decision()
        decisions.append(decision.to_dict())
        if decision.terminal:
            break
    service.save()
    return _print_json({
        "hypotheses": [h.to_dict() for h in portfolio],
        "experiments": [e.to_dict() for _, e in compiled],
        "decisions": decisions,
    })


def _observe(args: argparse.Namespace) -> int:
    service = _build_service(args)
    payload = json.loads(Path(args.observation).read_text(encoding="utf-8"))
    from .observation import ExperimentObservation
    observation = ExperimentObservation.from_dict(payload)
    service.record_observation(observation)
    service.save()
    return _print_json({"recorded": observation.observation_id,
                        "run_id": observation.run_id})


def _findings(args: argparse.Namespace) -> int:
    service = _build_service(args)
    return _print_json({"findings": [f.to_dict() for f in service.memory.findings()]})


def _tree(args: argparse.Namespace) -> int:
    service = _build_service(args)
    return _print_json(service.tree.to_dict())


def register_scientist_subcommands(sub: argparse._SubParsersAction) -> None:
    scientist = sub.add_parser(
        "scientist", help="Research-director mode: hypothesis portfolios, typed experiments, findings"
    )
    scientist_sub = scientist.add_subparsers(dest="scientist_command", required=True)

    def _common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--mission", required=True, help="ResearchMission JSON document")
        p.add_argument("--policy", required=True,
                       help="scientist_policy JSON document (provider, budgets, treatments)")
        p.add_argument("--state-root", required=True,
                       help="durable state root for this research session")

    status = scientist_sub.add_parser("status", help="Mission status and next decision")
    _common(status)
    status.set_defaults(func=_status)

    plan = scientist_sub.add_parser("plan", help="Generate the hypothesis portfolio and admit experiments")
    _common(plan)
    plan.add_argument("--count", type=int, default=3)
    plan.set_defaults(func=_plan)

    run = scientist_sub.add_parser("run", help="Run the research session loop (no direct GPU work)")
    _common(run)
    run.add_argument("--count", type=int, default=3)
    run.add_argument("--max-steps", type=int, default=8)
    run.set_defaults(func=_run)

    observe = scientist_sub.add_parser("observe",
                                       help="Record a run-grounded ExperimentObservation JSON")
    _common(observe)
    observe.add_argument("--observation", required=True)
    observe.set_defaults(func=_observe)

    findings = scientist_sub.add_parser("findings", help="List persisted findings")
    _common(findings)
    findings.set_defaults(func=_findings)

    tree = scientist_sub.add_parser("tree", help="Show the research tree")
    _common(tree)
    tree.set_defaults(func=_tree)
