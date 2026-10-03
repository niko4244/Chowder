"""ModelResearchService × screening lane: the resumable successive-halving
wiring (policy → compute providers → scheduler → durable session steps).

Pinned:
- a policy `compute` section builds the scheduler through the closed
  provider_from_config factory (unknown kind/key refuse by name);
- no compute section → advance_screening refuses (no default compute);
- advance_screening is a durable STEP machine: submit round → awaiting
  (spends nothing) → settle from the tree → submit next round at the
  multiplied budget → graduate final survivors;
- candidates come from ADMITTED proposals in durable memory, never re-asked
  from the provider;
- the session survives restart (screening-session.json round-trips);
- `scientist screen` runs the same machine through the CLI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chowder.scientist import (
    MissionBudget,
    ResearchMission,
    ScreeningHalving,
    provider_from_config,
)
from chowder.scientist.compute import (
    ExperimentScheduler,
    KaggleProvider,
    RunPodProvider,
    SchedulerRefusal,
)
from chowder.scientist.observation import ExperimentObservation, Measurement
from chowder.scientist.providers.fake import FakeDeterministicScientistProvider
from chowder.scientist.research_service import (
    ModelResearchService,
    ResearchServiceError,
    ScreeningSession,
)


# ---------------------------------------------------------------------------
# provider_from_config (closed set, closed keys)
# ---------------------------------------------------------------------------


def test_provider_from_config_builds_known_kinds() -> None:
    kaggle = provider_from_config({
        "kind": "kaggle", "username": "u", "api_key": "k",
        "weekly_gpu_hours": 12.0, "push": False,
    })
    assert isinstance(kaggle, KaggleProvider)
    assert kaggle.quota().weekly_gpu_hours == 12.0
    runpod = provider_from_config({
        "kind": "runpod", "api_key": "rk", "gpu_type_id": "NVIDIA A100",
        "image": "img", "command": "python run.py", "weekly_gpu_hours": 4.0,
    })
    assert isinstance(runpod, RunPodProvider)


def test_provider_from_config_refuses_unknown_kind_and_keys() -> None:
    with pytest.raises(SchedulerRefusal, match="UNKNOWN_PROVIDER_KIND"):
        provider_from_config({"kind": "lambda_labs"})
    with pytest.raises(SchedulerRefusal, match="UNKNOWN_PROVIDER_CONFIG_KEYS"):
        provider_from_config({"kind": "kaggle", "api_key": "k", "gpu_type_id": "x"})


# ---------------------------------------------------------------------------
# the service step machine
# ---------------------------------------------------------------------------


def _policy(providers: list[dict] | None = None,
            screening: dict | None = None) -> dict:
    policy: dict = {"provider": "fake_deterministic", "provider_config": {}}
    if providers is not None:
        compute: dict = {"providers": providers}
        if screening is not None:
            compute["screening"] = screening
        policy["compute"] = compute
    return policy


def _service(tmp_path: Path, policy: dict) -> ModelResearchService:
    return ModelResearchService.from_policy(
        mission_document={
            "mission_id": "m-screen",
            "objective": "improve reasoning",
            "priorities": {"reasoning": 1.0},
            "budget": MissionBudget(max_gpu_hours=8.0, max_tree_nodes=32,
                                    max_parallel_branches=8).to_dict(),
        },
        scientist_policy=policy,
        state_root=tmp_path / "state",
    )


def _admit_three(tmp_path: Path, policy: dict) -> ModelResearchService:
    """A service with three admitted proposals in durable memory, built the
    programmatic way (the from_policy path is covered by the refusal and CLI
    tests). The fake proposes one candidate per hypothesis with kind chosen
    by position (0 admissible, 1 over-budget, 2 architecture), so five
    hypotheses leave exactly three admitted candidates."""
    from chowder.scientist.research_service import ModelResearchService as _S
    mission = ResearchMission(
        mission_id="m-screen", objective="improve reasoning",
        priorities={"reasoning": 1.0},
        budget=MissionBudget(max_gpu_hours=8.0, max_tree_nodes=32,
                             max_parallel_branches=8),
    )
    provider = FakeDeterministicScientistProvider(hypotheses_per_round=5)
    compute = policy.get("compute") or {}
    scheduler = ExperimentScheduler(
        [provider_from_config(dict(spec)) for spec in compute.get("providers", [])]
    ) if compute else None
    schedule = ScreeningHalving(**dict(compute.get("screening") or {}))
    service = _S(mission=mission, provider=provider,
                 state_root=tmp_path / "state", scheduler=scheduler,
                 screening_schedule=schedule)
    service.generate_portfolio(count=5)
    service.request_experiments()
    return service


_KAGGLE = {"kind": "kaggle", "username": "u", "api_key": "k",
           "weekly_gpu_hours": 50.0, "push": False}
_SCHEDULE = {"initial_budget_gpu_hours": 0.05, "budget_cap_gpu_hours": 0.25,
             "step_multiplier": 2.0, "survival_fraction": 0.5,
             "min_survivors": 1, "max_rounds": 4}


def test_advance_refuses_without_compute_policy(tmp_path: Path) -> None:
    service = _admit_three(tmp_path, _policy())  # no compute section
    with pytest.raises(ResearchServiceError, match="SCREENING_NOT_CONFIGURED"):
        service.advance_screening()


def test_policy_declaring_compute_without_providers_refuses(tmp_path: Path) -> None:
    with pytest.raises(ResearchServiceError, match="COMPUTE_DECLARED_WITHOUT_PROVIDERS"):
        _service(tmp_path, {"provider": "fake_deterministic", "compute": {}})


def test_screening_step_machine_submits_then_awaits(tmp_path: Path) -> None:
    service = _admit_three(tmp_path, _policy([_KAGGLE], _SCHEDULE))
    step = service.advance_screening()
    assert step["phase"] == "submitted"
    assert step["round_index"] == 0
    assert step["budget_gpu_hours"] == pytest.approx(0.05)
    assert len(step["awaiting"]) == 3          # the three admitted proposals
    # durable: the session file exists and round-trips
    session = ScreeningSession.from_dict(json.loads(
        (tmp_path / "state" / "research" / "screening-session.json")
        .read_text(encoding="utf-8")))
    assert session.started and len(session.awaiting) == 3
    # advance again with NO observations: awaiting, and nothing new is spent
    kaggle = service.scheduler._providers["kaggle"]
    used = kaggle.quota().used_gpu_hours
    step2 = service.advance_screening()
    assert step2["phase"] == "awaiting"
    assert len(step2["missing_observations"]) == 3
    assert kaggle.quota().used_gpu_hours == used  # no compute behind the wait


def test_screening_step_machine_settles_and_graduates(tmp_path: Path) -> None:
    service = _admit_three(tmp_path, _policy([_KAGGLE], _SCHEDULE))

    def quality_of(proposal_id: str, round_index: int) -> float:
        # h4's candidate is the star, h3 decent, h0 weak — stable across rounds
        return {"fake-prop-fake-hyp-000": 0.05,
                "fake-prop-fake-hyp-003": 0.60,
                "fake-prop-fake-hyp-004": 0.90}[proposal_id] + round_index * 0.01

    def record_results(triples, round_index):
        # the production seam: observations flow through the director (which
        # grounds, journals AND grows the tree) — never straight into memory
        for i, (proposal, experiment, submission) in enumerate(triples):
            service.record_observation(ExperimentObservation(
                observation_id=f"obs-r{round_index}-{proposal.proposal_id}",
                run_id=f"run-r{round_index}-{proposal.proposal_id}",
                experiment_ref=experiment.experiment_id,
                proposal_id=proposal.proposal_id,
                hypothesis_id=proposal.hypothesis_id,
                measurements=(Measurement(surface="reasoning", benchmark="b",
                                          value=quality_of(proposal.proposal_id,
                                                           round_index)),),
                wall_gpu_hours=0.05,
                hardware_class=submission.hardware_class,
            ))

    first = service.advance_screening(record_results=record_results)
    assert first["phase"] == "submitted" and first["round_index"] == 0
    second = service.advance_screening(record_results=record_results)
    # 3 candidates → 2 survive round 0 at 2× budget
    assert second["phase"] == "submitted"
    assert second["round_index"] == 1
    assert second["budget_gpu_hours"] == pytest.approx(0.10)
    assert len(second["awaiting"]) == 2
    third = service.advance_screening(record_results=record_results)
    # 2 → 1 survivor = min_survivors: final round, graduates
    assert third["phase"] == "graduated"
    assert third["final_survivors"] == ["sciexp-fake-prop-fake-hyp-004"]
    assert len(third["rounds"]) == 2
    # every elimination journaled: 1 cutoff (round 0) + 1 cutoff (round 1)
    rows = (tmp_path / "state" / "research" / "refusals.jsonl").read_text(
        encoding="utf-8")
    assert rows.count("screening_eliminated") == 2
    # a completed session is stable: advancing again returns the outcome
    again = service.advance_screening()
    assert again["phase"] == "complete"
    assert again["final_survivors"] == ["sciexp-fake-prop-fake-hyp-004"]


def test_session_survives_a_restart_mid_schedule(tmp_path: Path) -> None:
    service = _admit_three(tmp_path, _policy([_KAGGLE], _SCHEDULE))
    service.advance_screening()  # round 0 submitted, nothing observed
    # a NEW service instance over the same state root resumes the schedule
    resumed = ModelResearchService.from_policy(
        mission_document={
            "mission_id": "m-screen", "objective": "improve reasoning",
            "priorities": {"reasoning": 1.0},
            "budget": {"max_gpu_hours": 8.0, "max_tree_nodes": 32,
                       "max_parallel_branches": 8},
        },
        scientist_policy=_policy([_KAGGLE], _SCHEDULE),
        state_root=tmp_path / "state",
    )
    step = resumed.advance_screening()
    assert step["phase"] == "awaiting"   # still the same round-0 submissions
    assert step["round_index"] == 0
    assert len(step["missing_observations"]) == 3


def test_graduated_pairs_can_be_rebuilt_for_substantial_runs(tmp_path: Path) -> None:
    service = _admit_three(tmp_path, _policy([_KAGGLE], _SCHEDULE))
    service.advance_screening()
    # fake the observations straight into memory so the lane can settle
    session = service._load_screening_session()
    for eid in session.awaiting:
        service.record_observation(ExperimentObservation(
            observation_id=f"obs-{eid}", run_id=f"run-{eid}",
            experiment_ref=eid, proposal_id=eid.removeprefix("sciexp-"),
            hypothesis_id="fake-hyp-000",
            measurements=(Measurement(surface="reasoning", benchmark="b",
                                      value=0.5),),
            wall_gpu_hours=0.05, hardware_class="kaggle_2x_t4_16gb",
        ))
    out = service.advance_screening()
    assert out["phase"] in ("submitted", "graduated")


# ---------------------------------------------------------------------------
# the CLI screen command
# ---------------------------------------------------------------------------


def _write_policy_files(tmp_path: Path, policy: dict) -> tuple[Path, Path]:
    mission = {
        "mission_id": "m-cli", "objective": "improve reasoning",
        "priorities": {"reasoning": 1.0},
        "budget": {"max_gpu_hours": 8.0, "max_tree_nodes": 32,
                   "max_parallel_branches": 8},
    }
    mission_path = tmp_path / "mission.json"
    mission_path.write_text(json.dumps(mission), encoding="utf-8")
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    return mission_path, policy_path


def test_cli_screen_advances_one_durable_step(tmp_path: Path, capsys) -> None:
    import argparse
    from chowder.scientist.cli import register_scientist_subcommands
    mission_path, policy_path = _write_policy_files(
        tmp_path, _policy([_KAGGLE], _SCHEDULE))
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="top", required=True)
    register_scientist_subcommands(sub)
    args = parser.parse_args([
        "scientist", "screen", "--mission", str(mission_path),
        "--policy", str(policy_path), "--state-root", str(tmp_path / "state"),
    ])
    # no admitted proposals yet → the step refuses with a named error (exit 1)
    assert args.func(args) == 1
    captured = json.loads(capsys.readouterr().out)
    assert "SCREENING_NO_CANDIDATES" in captured["error"]
    # admit a portfolio through the CLI's own plan command, then screen
    args = parser.parse_args([
        "scientist", "plan", "--mission", str(mission_path),
        "--policy", str(policy_path), "--state-root", str(tmp_path / "state"),
        "--count", "3",
    ])
    assert args.func(args) == 0
    capsys.readouterr()
    args = parser.parse_args([
        "scientist", "screen", "--mission", str(mission_path),
        "--policy", str(policy_path), "--state-root", str(tmp_path / "state"),
    ])
    assert args.func(args) == 0
    step = json.loads(capsys.readouterr().out)
    assert step["phase"] == "submitted"
    assert step["budget_gpu_hours"] == pytest.approx(0.05)
    # plan --count 3 leaves exactly one admissible candidate (fake provider
    # positions 1/2 are over-budget/architecture)
    assert len(step["awaiting"]) == 1
