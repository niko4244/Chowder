"""Dry-run proof: the graduate CLI -> durable hand-off file -> the growth
loop's next generation, end to end, on the real screening fixtures — with no
compute touched (no Kaggle, no RunPod, no local training).

What it proves, in order:

 1. the screening candidates are admitted into durable memory through the
    ModelResearchService portfolio/admission path (the same durable state the
    plan CLI writes; the CLI's default replication policy admits one
    candidate, the direct path three — either way the lane is the same);
 2. the REAL `chowder scientist graduate` CLI drives the successive-halving
    lane (fake deterministic provider; the Kaggle provider spec is
    push=False and is never contacted) through
    submit -> `scientist observe` -> settle -> ... -> graduation, and then
    composes the growth loop's campaign draft for the final survivor;
 3. graduation writes the durable scientist-handoff.json into the shared
    state root — the auto-consume seam;
 4. a GrowthLoop pointed at the SAME state root adopts the survivor through
    its own gate chain (plan_next, read-only), then runs prepare_next:
    draft -> record target -> CLEAR the hand-off (one survivor per
    generation) -> prepare -> freeze;
 5. an executor that raises AssertionError proves the loop executed nothing
    and spent nothing.

Run:  python scripts/dryrun_survivor_handoff.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import chowder  # noqa: E402  (after the path pin)

assert str(ROOT) in str(Path(chowder.__file__).resolve()), (
    f"imported chowder from {chowder.__file__}, not the worktree {ROOT}"
)

from chowder.growth.campaign import CampaignManifest  # noqa: E402
from chowder.growth.growth_loop import GrowthLoop  # noqa: E402
from chowder.growth.next_campaign import LoopPolicy  # noqa: E402
from chowder.growth.scientist_handoff import (  # noqa: E402
    HANDOFF_FILENAME,
    load_graduated_survivor,
)
from chowder.growth.simulator import DEFAULT_PARENT_DECLARATION, skill_profile  # noqa: E402
from chowder.growth.target_selection import GrowthState  # noqa: E402
from chowder.scientist.cli import register_scientist_subcommands  # noqa: E402

RUN_ROOT = ROOT / "tmp" / "dryrun-survivor-handoff"
STATE_ROOT = RUN_ROOT / "state"

#: The screening outcome the operator's observations encode (same shape as the
#: lane tests): one loser, one middle survivor, one clear winner.
QUALITY = {
    "fake-prop-fake-hyp-000": 0.05,
    "fake-prop-fake-hyp-003": 0.60,
    "fake-prop-fake-hyp-004": 0.90,
}

#: Kaggle-spec-shaped config with push=False: builds the real provider seam,
#: never contacts Kaggle (the fake deterministic provider does the work).
KAGGLE_PUSH_FALSE = {"kind": "kaggle", "username": "u", "api_key": "k",
                     "weekly_gpu_hours": 50.0, "push": False}
SCHEDULE = {"initial_budget_gpu_hours": 0.05, "budget_cap_gpu_hours": 0.25,
            "step_multiplier": 2.0, "survival_fraction": 0.5,
            "min_survivors": 1, "max_rounds": 4}

#: A measured parent profile for the loop (the same shape the growth tests
#: use); its generation must match the parent declaration's candidate.
START_PROFILE = {"math.reasoning": 0.62, "instruction.formatting": 0.31,
                 "termination.control": 0.44}


def say(section: str, payload: Any = None) -> None:
    print(f"\n=== {section} " + "=" * max(0, 66 - len(section)))
    if payload is not None:
        print(json.dumps(payload, indent=2, default=str))


def run_cli(*argv: str) -> dict[str, Any]:
    """Run the production scientist CLI in-process and return its JSON."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="top", required=True)
    register_scientist_subcommands(sub)
    args = parser.parse_args(list(argv))
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = args.func(args)
    assert code == 0, f"CLI exit {code}: {buffer.getvalue()}"
    return json.loads(buffer.getvalue())


COMMON = ("--mission", str(RUN_ROOT / "mission.json"),
          "--policy", str(RUN_ROOT / "policy.json"),
          "--state-root", str(STATE_ROOT))


def write_fixtures() -> None:
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    STATE_ROOT.mkdir(parents=True)

    mission = {"mission_id": "m-screen", "objective": "improve reasoning",
               "priorities": {"reasoning": 1.0},
               "budget": {"max_gpu_hours": 8.0, "max_tree_nodes": 32,
                          "max_parallel_branches": 8}}
    (RUN_ROOT / "mission.json").write_text(json.dumps(mission), encoding="utf-8")

    policy = {"provider": "fake_deterministic", "provider_config": {},
              "compute": {"providers": [KAGGLE_PUSH_FALSE],
                          "screening": SCHEDULE}}
    (RUN_ROOT / "policy.json").write_text(json.dumps(policy), encoding="utf-8")

    # The loop policy document (operator's): math500 protected, mgsm a fair
    # target; same document the growth lane's CLI tests bind drafts with.
    loop_policy = {
        "maximum_generations": 3,
        "maximum_total_wall_gpu_hours": 6.0,
        "maximum_consecutive_non_promotions": 2,
        "maximum_same_target_attempts": 2,
        "maximum_candidates": 2,
        "plateau_epsilon": 0.01,
        "allowed_training_types": ["data", "targeted_repair", "sft"],
        "protected_benchmarks": ["math500@2024-04"],
        "broad_benchmarks": ["math500@2024-04", "mgsm@2022-11"],
        "calibration_benchmarks": [],
        "reliability_benchmarks": [],
        "campaign_budget": {
            "device_gpu_hours_ceiling_per_recipe": 0.9,
            "wall_gpu_hours_ceiling_per_recipe": 1.8,
            "device_gpu_hours_ceiling_campaign": 1.8,
            "wall_gpu_hours_ceiling_campaign": 3.6,
            "device_time_measured": True,
        },
        "protection": {
            "trusted_ancestor_version": "gen0",
            "slice_regression_max": 0.0625,
            "n_samples": 16,
            "seed": 1234,
            "shuffle": False,
            "decoding": {"temperature": 0.0, "do_sample": False,
                         "max_new_tokens": 512},
            "prompt_policy": "chat_template",
        },
        "evaluation_execution": {"batch_size": 16},
        "candidate_selection_policy": "first_successful",
        "stopping_rules": [],
        "promotion_policy_version": "promotion-policy-v2-provenance-settlement",
        "human_review_triggers": [],
    }
    (RUN_ROOT / "loop-policy.json").write_text(json.dumps(loop_policy),
                                               encoding="utf-8")

    document = json.loads(Path(DEFAULT_PARENT_DECLARATION).read_text(encoding="utf-8"))
    document["state_root"] = str(RUN_ROOT / "parent-run")
    (RUN_ROOT / "parent-campaign.json").write_text(json.dumps(document),
                                                   encoding="utf-8")


def forbidden_executor(frozen: Any) -> Any:
    raise AssertionError(
        f"DRY-RUN VIOLATION: the loop tried to execute campaign "
        f"{getattr(frozen, 'cycle_id', '?')} — a dry run must spend nothing")


def planned_recipes(draft: Any) -> Any:
    """Recording stand-in for the production preparation planner (the same
    seam shape the growth tests use): preparation reports its recipe set."""
    from chowder.growth.growth_loop import PreparationResult

    return PreparationResult(
        recipe_ids=tuple(
            f"recipe-{index:02d}-lr0.0001-r16-replay0.1"
            for index in range(len(draft.placeholder_recipe_ids))
        ),
        detail="dry-run preparation (recording stand-in, nothing executed)",
    )


def phase_a_plan() -> dict[str, str]:
    """Admit the screening candidates into durable memory (the portfolio/
    admission path the plan CLI shares), then report what is durably there."""
    from chowder.scientist import (
        MissionBudget,
        ResearchMission,
        ScreeningHalving,
        provider_from_config,
    )
    from chowder.scientist.compute import ExperimentScheduler
    from chowder.scientist.providers.fake import FakeDeterministicScientistProvider
    from chowder.scientist.research_service import ModelResearchService

    mission = ResearchMission.from_mapping(json.loads(
        (RUN_ROOT / "mission.json").read_text(encoding="utf-8")))
    service = ModelResearchService(
        mission=mission,
        provider=FakeDeterministicScientistProvider(hypotheses_per_round=5),
        state_root=STATE_ROOT,
        scheduler=ExperimentScheduler([provider_from_config(dict(KAGGLE_PUSH_FALSE))]),
        screening_schedule=ScreeningHalving(**dict(SCHEDULE)),
    )
    portfolio = service.generate_portfolio(count=5)
    compiled = service.request_experiments()
    service.save()
    mapping: dict[str, tuple[str, str]] = {
        experiment.experiment_id: (proposal.proposal_id, proposal.hypothesis_id)
        for proposal, experiment in compiled
    }
    say("Phase A: portfolio admitted into durable memory",
        {"hypotheses": len(portfolio), "admitted": len(mapping)})
    assert len(mapping) == 3, f"expected 3 admitted experiments, got {mapping}"
    for _eid, (pid, _hid) in mapping.items():
        assert pid in QUALITY, f"unexpected candidate {pid}"
    return mapping


def phase_b_lane_via_cli(mapping: dict[str, tuple[str, str]]) -> dict[str, Any]:
    """Drive the whole lane through `scientist graduate` + `scientist observe`
    until graduation, then let the same CLI call compose the drafts."""
    observations_written = 0
    for _step in range(10):
        step = run_cli("scientist", "graduate", *COMMON,
                       "--loop-policy", str(RUN_ROOT / "loop-policy.json"),
                       "--parent-manifest", str(RUN_ROOT / "parent-campaign.json"),
                       "--generation-root", str(RUN_ROOT / "generations"),
                       "--benchmark-map", "reasoning=mgsm@2022-11")
        phase = str(step.get("phase"))
        if phase in ("graduated", "complete"):
            say("Phase B: lane graduated via the graduate CLI",
                {"observations_recorded": observations_written,
                 "campaign_drafts": len(step.get("campaign_drafts", []))})
            return step
        inner = step.get("step", step)
        awaiting = list(inner.get("awaiting") or [])
        round_index = inner.get("round_index")
        say(f"Phase B: CLI step -> {phase}",
            {"round_index": round_index, "awaiting": awaiting,
             "refused": list(inner.get("refused") or [])})
        assert awaiting, f"step {phase} carries nothing to observe: {step}"
        for eid in awaiting:
            proposal_id, hypothesis_id = mapping[eid]
            run_dir = RUN_ROOT / "runs" / f"run-r{round_index}-{proposal_id}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "result.json").write_text(json.dumps(
                {"run_id": str(run_dir), "status": "complete"}), encoding="utf-8")
            observation = {
                "observation_id": f"obs-r{round_index}-{proposal_id}",
                "run_id": str(run_dir),
                "experiment_ref": eid,
                "proposal_id": proposal_id,
                "hypothesis_id": hypothesis_id,
                "measurements": [{"surface": "reasoning", "benchmark": "b",
                                  "value": QUALITY[proposal_id]}],
                "status": "complete",
                "wall_gpu_hours": 0.05,
            }
            path = RUN_ROOT / f"obs-r{round_index}-{proposal_id}.json"
            path.write_text(json.dumps(observation), encoding="utf-8")
            run_cli("scientist", "observe", *COMMON, "--observation", str(path))
            observations_written += 1
    raise AssertionError("the lane did not graduate within 10 CLI steps")


def phase_c_loop_dry_run(step: dict[str, Any]) -> dict[str, Any]:
    """The loop side: adopt the survivor through the loop's OWN gates, then
    prepare + freeze the next generation — and see the hand-off cleared."""
    drafts = step["campaign_drafts"]
    assert len(drafts) == 1, f"expected exactly one survivor draft: {drafts}"
    draft = drafts[0]
    assert draft["cycle_id"].startswith("gen3-a1-"), draft
    assert draft["draft"]["frozen"] is False
    assert Path(draft["directory"]).exists()
    say("Phase B result: composed draft",
        {"cycle_id": draft["cycle_id"], "directory": draft["directory"],
         "frozen": draft["draft"]["frozen"]})

    # 3. the durable hand-off the graduation wrote into the shared state root
    handoff_path = STATE_ROOT / HANDOFF_FILENAME
    assert handoff_path.exists(), f"expected {handoff_path}"
    loaded = load_graduated_survivor(STATE_ROOT)
    assert loaded is not None
    survivor, _path = loaded
    say("Hand-off file (auto-consume seam)",
        {"path": str(handoff_path),
         "target_skill": survivor.target_skill,
         "benchmark_map": survivor.benchmark_map,
         "survivor_score": survivor.survivor_score,
         "proposal_id": survivor.proposal_id,
         "screening_mission_id": survivor.screening_mission_id})

    # 4. the loop's dry run: same state root, its own gate chain
    parent = CampaignManifest.from_file(RUN_ROOT / "parent-campaign.json")
    policy = LoopPolicy.from_file(RUN_ROOT / "loop-policy.json")
    loop = GrowthLoop(
        policy=policy,
        state=GrowthState(root=STATE_ROOT),
        executor=forbidden_executor,
        parent_declaration=parent,
        parent_profile=skill_profile(START_PROFILE, generation="gen2"),
        prepare=planned_recipes,
        readiness=lambda frozen: True,
    )

    proposal = loop.plan_next()  # read-only dry run
    assert not hasattr(proposal, "action"), (
        f"the loop refused instead of adopting: {proposal.to_dict() if hasattr(proposal, 'to_dict') else proposal}")
    say("plan_next() adopts the survivor",
        {"target_skill": proposal.target_skill,
         "target_benchmarks": list(proposal.target_benchmarks),
         "factors_total": proposal.factors.total,
         "treatment": proposal.suggested_training_type,
         "treatment_reason": proposal.treatment_reason})
    assert tuple(proposal.target_benchmarks) == ("mgsm@2022-11",)
    assert proposal.factors.total > 0

    prepared = loop.prepare_next()  # draft -> record -> CLEAR -> prepare -> freeze
    assert hasattr(prepared, "frozen"), (
        f"prepare_next refused: {prepared.to_dict() if hasattr(prepared, 'to_dict') else prepared}")
    frozen = prepared.frozen
    assert tuple(frozen.target.target_benchmarks) == ("mgsm@2022-11",)
    say("prepare_next() froze the next generation",
        {"cycle_id": frozen.cycle_id,
         "frozen_directory": str(frozen.directory),
         "recipes": len(prepared.recipe_ids) if hasattr(prepared, "recipe_ids") else None})

    # 5. consumption cleared the file; one survivor per generation; nothing ran
    assert not handoff_path.exists(), "the consumed hand-off file should be cleared"
    assert load_graduated_survivor(STATE_ROOT) is None
    resumed = loop.plan_next()
    resumed_shape = (resumed.action if hasattr(resumed, "action")
                     else f"proposal:{resumed.target_skill}")
    say("After consumption",
        {"handoff_cleared": True,
         "executor_calls": "none (the executor raises if touched)",
         "next_plan_without_survivor": resumed_shape})
    return {"cycle_id": frozen.cycle_id, "resumed": resumed_shape}


def main() -> int:
    write_fixtures()
    mapping = phase_a_plan()
    step = phase_b_lane_via_cli(mapping)
    summary = phase_c_loop_dry_run(step)
    report = {"run_root": str(RUN_ROOT), "kaggle_touched": False,
              "chain": "plan CLI -> graduate CLI (lane + draft) -> handoff file "
                       "-> loop plan_next adoption -> prepare_next freeze -> clear",
              **summary}
    (RUN_ROOT / "dryrun-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    say("DRY-RUN PROOF COMPLETE", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
