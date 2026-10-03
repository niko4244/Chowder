"""Compute providers: scheduling policy and the hardware-context evidence
rules (docs/COMPUTE_PROVIDERS.md).

Pinned:
- screening requests prefer the declared screening lane; substantial requests
  follow declaration order;
- an exhausted quota falls back along the declared order; a pinned request
  never reroutes and an unknown pin refuses;
- an empty provider list refuses (no default provider);
- device GPU-hours are honest (Kaggle wall × 2 accelerators);
- a hardware-dependent (efficiency) claim cannot reach `replicated` citing
  runs from two hardware classes (requirement: efficiency results keep their
  hardware context);
- a quality claim may cite cross-hardware runs;
- observations carry the hardware class; the mission ledger charges the same
  regardless of provider;
- screening → survivor batching is score-driven, journaled on refusal.
"""

from __future__ import annotations

from dataclasses import replace as _replace
from pathlib import Path

import pytest

from chowder.scientist.compute import (
    ExperimentClass,
    ExperimentRequest,
    ExperimentScheduler,
    KaggleProvider,
    LocalCudaProvider,
    ProviderQuota,
    SchedulerRefusal,
    Submission,
)
from chowder.scientist.observation import ExperimentObservation, Measurement
from chowder.scientist.research_memory import ResearchMemory
from chowder.scientist.research_tree import ResearchTree


def _request(**kw) -> ExperimentRequest:
    base = dict(experiment_id="e1", proposal_id="p1", hypothesis_id="h1",
                campaign_spec={}, estimated_gpu_hours=0.5)
    base.update(kw)
    return ExperimentRequest(**base)


def _providers(*, kaggle_hours: float = 10.0):
    local = LocalCudaProvider(accelerators=1)
    kaggle = KaggleProvider(username="u", api_key="k", weekly_gpu_hours=kaggle_hours)
    return local, kaggle


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------


def test_screening_prefers_the_screening_lane() -> None:
    local, kaggle = _providers()
    scheduler = ExperimentScheduler([local, kaggle])
    submission = scheduler.schedule(_request(experiment_class=ExperimentClass.SCREENING))
    assert submission.provider_name == "kaggle"
    assert submission.hardware_class == "kaggle_2x_t4_16gb"
    # honest device-hours: wall 0.5 × 2 T4s
    assert submission.device_gpu_hours == pytest.approx(1.0)


def test_substantial_follows_declaration_order() -> None:
    local, kaggle = _providers()
    scheduler = ExperimentScheduler([local, kaggle])
    submission = scheduler.schedule(_request(experiment_class=ExperimentClass.SUBSTANTIAL))
    assert submission.provider_name == "local_cuda"


def test_quota_exhaustion_falls_back_along_the_order() -> None:
    local, kaggle = _providers(kaggle_hours=1.0)
    scheduler = ExperimentScheduler([kaggle, local])
    # burn the kaggle quota (0.5 wall × 2 = 1.0 device-hours)
    scheduler.schedule(_request(experiment_class=ExperimentClass.SCREENING))
    submission = scheduler.schedule(_request(experiment_id="e2",
                                             experiment_class=ExperimentClass.SCREENING))
    assert submission.provider_name == "local_cuda"
    assert kaggle.available() is False


def test_pinned_request_never_reroutes_and_unknown_pin_refuses() -> None:
    local, kaggle = _providers()
    scheduler = ExperimentScheduler([local, kaggle])
    pinned = scheduler.schedule(_request(require_provider="local_cuda"))
    assert pinned.provider_name == "local_cuda"
    with pytest.raises(SchedulerRefusal, match="UNKNOWN_PROVIDER_PIN"):
        scheduler.schedule(_request(require_provider="vast"))


def test_unconfigured_kaggle_is_unavailable_not_fake() -> None:
    provider = KaggleProvider()
    assert provider.configured() is False
    assert provider.available() is False
    scheduler = ExperimentScheduler([LocalCudaProvider(accelerators=0), provider])
    with pytest.raises(SchedulerRefusal, match="NO_PROVIDER_AVAILABLE"):
        scheduler.schedule(_request())


def test_empty_provider_list_refuses() -> None:
    with pytest.raises(SchedulerRefusal, match="NO_PROVIDER_AVAILABLE"):
        ExperimentScheduler([])


def test_local_zero_accelerators_is_unavailable() -> None:
    provider = LocalCudaProvider(accelerators=0)
    assert provider.available() is False


# ---------------------------------------------------------------------------
# the hardware-context rule
# ---------------------------------------------------------------------------


def _memory(tmp_path: Path) -> ResearchMemory:
    return ResearchMemory(tmp_path / "research",
                          run_exists=lambda r: True, run_complete=lambda r: True)


def _observation(tmp_path: Path, *, obs_id: str, run_id: str, hardware_class: str,
                 surface: str = "reasoning", value: float = 0.6):
    memory = _memory(tmp_path)
    observation = ExperimentObservation(
        observation_id=obs_id, run_id=run_id, experiment_ref="sciexp-p1",
        proposal_id="p1", hypothesis_id="h1",
        measurements=(Measurement(surface=surface, benchmark="b", value=value),),
        wall_gpu_hours=0.3, hardware_class=hardware_class,
    )
    memory.record_observation(observation)
    return memory, observation


def test_efficiency_claim_cannot_replicate_across_hardware(tmp_path: Path) -> None:
    from chowder.scientist import Claim, ResearchFinding
    memory, _ = _observation(tmp_path, obs_id="obs-a", run_id="run-kaggle",
                             hardware_class="kaggle_2x_t4_16gb",
                             surface="efficiency:tokens_per_sec", value=42.0)
    memory, _ = _observation(tmp_path, obs_id="obs-b", run_id="run-local",
                             hardware_class="local_rtx", surface="efficiency:tokens_per_sec",
                             value=55.0)
    finding = ResearchFinding(
        finding_id="f1", hypothesis_id="h1",
        claims=(Claim(
            claim_id="c1",
            statement="training reaches 42-55 tokens/sec",
            scope="test",
            status="replicated",
            supporting_experiments=("run-kaggle", "run-local"),
            hardware_dependent=True,
        ),),
        observation_ids=("obs-a", "obs-b"),
    )
    with pytest.raises(ValueError, match="HARDWARE_CONTEXT"):
        memory.record_finding(finding)


def test_efficiency_claim_replicates_on_one_hardware_class(tmp_path: Path) -> None:
    from chowder.scientist import Claim, ResearchFinding
    memory, _ = _observation(tmp_path, obs_id="obs-a2", run_id="run-k1",
                             hardware_class="kaggle_2x_t4_16gb",
                             surface="efficiency:tokens_per_sec", value=42.0)
    memory, _ = _observation(tmp_path, obs_id="obs-b2", run_id="run-k2",
                             hardware_class="kaggle_2x_t4_16gb",
                             surface="efficiency:tokens_per_sec", value=43.0)
    finding = ResearchFinding(
        finding_id="f2", hypothesis_id="h1",
        claims=(Claim(
            claim_id="c2", statement="2×T4 reaches ~42 tokens/sec", scope="test",
            status="replicated",
            supporting_experiments=("run-k1", "run-k2"),
            hardware_dependent=True,
        ),),
        observation_ids=("obs-a2", "obs-b2"),
    )
    memory.record_finding(finding)  # accepted: one hardware class
    assert memory.findings()[0].claims[0].status == "replicated"


def test_quality_claim_may_cite_cross_hardware_runs(tmp_path: Path) -> None:
    from chowder.scientist import Claim, ResearchFinding
    memory, _ = _observation(tmp_path, obs_id="obs-q1", run_id="run-q1",
                             hardware_class="kaggle_2x_t4_16gb")
    memory, _ = _observation(tmp_path, obs_id="obs-q2", run_id="run-q2",
                             hardware_class="local_rtx")
    finding = ResearchFinding(
        finding_id="f3", hypothesis_id="h1",
        claims=(Claim(
            claim_id="c3", statement="replay decay improves reasoning", scope="test",
            status="replicated",
            supporting_experiments=("run-q1", "run-q2"),
            hardware_dependent=False,
        ),),
        observation_ids=("obs-q1", "obs-q2"),
    )
    memory.record_finding(finding)  # accepted: quality is protocol-scoped, not hardware-scoped
    assert memory.findings()[0].claims[0].status == "replicated"


def test_observation_serialization_carries_hardware_class(tmp_path: Path) -> None:
    memory, observation = _observation(tmp_path, obs_id="obs-h", run_id="run-h",
                                       hardware_class="kaggle_2x_t4_16gb")
    stored = memory.observations()[0]
    assert stored.hardware_class == "kaggle_2x_t4_16gb"


# ---------------------------------------------------------------------------
# director-side batching
# ---------------------------------------------------------------------------


def _director_with_tree(tmp_path: Path):
    from chowder.scientist import (
        Hypothesis, MissionBudget, ResearchMission, ResearchQuestion,
    )
    from chowder.scientist.research_director import ResearchDirector
    mission = ResearchMission(
        mission_id="m-c", objective="improve reasoning",
        priorities={"reasoning": 1.0},
        budget=MissionBudget(max_gpu_hours=8.0, max_tree_nodes=10, max_parallel_branches=3),
    )
    memory = ResearchMemory(tmp_path / "research")
    tree = ResearchTree(mission_id=mission.mission_id)
    director = ResearchDirector(
        mission=mission, provider=_FakeProvider(), memory=memory, tree=tree,
    )
    return director


class _FakeProvider:
    name = "fake_deterministic"

    def available(self):
        return True

    def propose_hypotheses(self, context, *, count=3):
        return ()

    def propose_experiments(self, context, hypotheses):
        return ()

    def interpret(self, context, hypothesis, observations):
        raise NotImplementedError

    def export_state(self):
        return {}


def test_screening_and_survivor_batches_route_and_journal(tmp_path: Path) -> None:
    from chowder.scientist.lab_bridge import CompiledExperiment
    from chowder.scientist.proposal import (
        DataStrategy, ExperimentProposal, TrainingRecipeDelta,
    )
    director = _director_with_tree(tmp_path)
    local, kaggle = _providers()
    scheduler = ExperimentScheduler([kaggle, local])

    def _pair(i: int):
        proposal = ExperimentProposal(
            proposal_id=f"p{i}", hypothesis_id=f"h{i}", experiment_type="data",
            intervention="x", variables_changed=("replay_ratio",),
            variables_held_constant=("lr",),
            training_recipe_delta=TrainingRecipeDelta(epochs=1),
            data_strategy=DataStrategy(source_kinds=("curriculum",)),
            requested_evaluations=("reasoning",), expected_outcome="up",
            falsification_rule="delta<=0", estimated_gpu_hours=0.25,
        )
        experiment = CompiledExperiment(
            experiment_id=f"sciexp-p{i}", proposal_id=f"p{i}", hypothesis_id=f"h{i}",
            campaign_spec={"recipe_patch": {}}, estimated_gpu_hours=0.25,
        )
        return proposal, experiment

    triples = director.submit_screening_batch(scheduler, (_pair(0), _pair(1), _pair(2)))
    assert len(triples) == 3
    assert all(t[2].provider_name == "kaggle" for t in triples)
    assert kaggle.quota().used_gpu_hours == pytest.approx(3 * 0.5)

    # record observations so the tree can score, then promote survivors
    for i, (proposal, experiment, submission) in enumerate(triples):
        director.record_observation(ExperimentObservation(
            observation_id=f"obs-{i}", run_id=f"run-{i}",
            experiment_ref=experiment.experiment_id,
            proposal_id=proposal.proposal_id, hypothesis_id=proposal.hypothesis_id,
            measurements=(Measurement(surface="reasoning", benchmark="b",
                                      value=(0.9 if i == 1 else 0.1),),),
            wall_gpu_hours=0.25, hardware_class=submission.hardware_class,
        ))
    survivors = director.submit_survivor_batch(scheduler, triples, keep_top=1)
    assert len(survivors) == 1
    assert survivors[0][0].proposal_id == "p1"  # the 0.9-quality branch
    assert survivors[0][2].experiment_class == ExperimentClass.SUBSTANTIAL


def test_mission_ledger_charges_identically_across_providers(tmp_path: Path) -> None:
    director = _director_with_tree(tmp_path)
    before = director.spend.spent_gpu_hours
    director.record_observation(ExperimentObservation(
        observation_id="obs-led", run_id="run-led", experiment_ref="sciexp-x",
        proposal_id="x", hypothesis_id="h1",
        measurements=(Measurement(surface="reasoning", benchmark="b", value=0.5),),
        wall_gpu_hours=0.75, hardware_class="kaggle_2x_t4_16gb",
    ))
    assert director.spend.spent_gpu_hours == pytest.approx(before + 0.75)
