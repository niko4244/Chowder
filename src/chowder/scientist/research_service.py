"""ModelResearchService: the option-C composition coordinator.

One production service composing the two related-but-separate state machines:

    ModelResearchService
      ├── AutonomousGrowthService   (existing; owns growth state, unchanged)
      └── ResearchDirector          (new; owns the research mission state)

CLI and (future) TUI are thin clients of this service, exactly as growth's
CLI/TUI are thin clients of AutonomousGrowthService. The service owns
composition only — every decision belongs to the director, the loop, or the
production gates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .findings import ReplicationPolicy
from .lab_bridge import CompiledExperiment, ExperimentCompiler
from .mission import ResearchMission
from .observation import ExperimentObservation
from .proposal import ExperimentProposal
from .provider import ResearchContext, SkillSummary, ScientistProvider
from .research_decision import ResearchDecision
from .research_director import DirectorRefusal, DirectorStatus, ResearchDirector
from .research_memory import ResearchMemory
from .research_tree import ResearchTree


class ResearchServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class MissionView:
    """What `scientist status` shows; a read-only projection."""

    mission_id: str
    objective: str
    autonomy: str
    provider: str
    spent_gpu_hours: float
    remaining_gpu_hours: float
    tree_nodes: int
    active_branches: int
    hypotheses_open: int
    findings: int
    decision: str

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def default_provider(name: str, provider_config: dict[str, Any]) -> ScientistProvider:
    """Resolve a provider by name. No silent fallback: an unknown or
    unavailable provider raises instead of degrading to a fake."""
    if name == "fake_deterministic":
        from .providers.fake import FakeDeterministicScientistProvider
        return FakeDeterministicScientistProvider()
    if name in ("ai_scientist_v2", "ai-scientist-v2"):
        from .providers.ai_scientist_v2 import AIScientistV2Provider
        return AIScientistV2Provider.from_config(provider_config)
    raise ResearchServiceError(
        f"UNKNOWN_PROVIDER: {name!r} is not a known scientist provider; "
        "refusing rather than falling back to anything"
    )


class ModelResearchService:
    """Composes the growth service (untouched) and the research director."""

    def __init__(
        self,
        *,
        mission: ResearchMission,
        provider: ScientistProvider,
        state_root: Path,
        growth_service: Any | None = None,
        replication_policy: ReplicationPolicy | None = None,
        protected_benchmarks: tuple[str, ...] = (),
        run_exists: Callable[[str], bool] | None = None,
        run_complete: Callable[[str], bool] | None = None,
    ) -> None:
        self.mission = mission
        self.provider = provider
        self.growth_service = growth_service  # composition; owned elsewhere
        self.state_root = Path(state_root)
        self.memory = ResearchMemory(
            self.state_root / "research",
            run_exists=run_exists,
            run_complete=run_complete,
        )
        self.tree = self._load_or_init_tree()
        self.director = ResearchDirector(
            mission=mission,
            provider=provider,
            memory=self.memory,
            tree=self.tree,
            compiler=ExperimentCompiler(protected_benchmarks=tuple(protected_benchmarks)),
            replication_policy=replication_policy,
            protected_benchmarks=tuple(protected_benchmarks),
            run_exists=run_exists,
            run_complete=run_complete,
        )

    # -- mission lifecycle ------------------------------------------------------

    @classmethod
    def from_policy(
        cls,
        *,
        mission_document: dict[str, Any],
        scientist_policy: dict[str, Any],
        state_root: Path,
        growth_service: Any | None = None,
        run_exists: Callable[[str], bool] | None = None,
        run_complete: Callable[[str], bool] | None = None,
    ) -> "ModelResearchService":
        mission = ResearchMission.from_mapping(mission_document)
        provider_name = str(scientist_policy.get("provider", ""))
        provider = default_provider(provider_name,
                                    dict(scientist_policy.get("provider_config") or {}))
        if not provider.available():
            raise ResearchServiceError(
                f"PROVIDER_UNAVAILABLE: {provider_name} cannot run in this "
                "environment; refusing rather than faking research"
            )
        return cls(
            mission=mission,
            provider=provider,
            state_root=state_root,
            growth_service=growth_service,
            replication_policy=ReplicationPolicy.from_dict(
                dict(scientist_policy.get("replication_policy") or {})
            ),
            protected_benchmarks=tuple(
                str(b) for b in scientist_policy.get("protected_benchmarks", ())
            ),
            run_exists=run_exists,
            run_complete=run_complete,
        )

    def _load_or_init_tree(self) -> ResearchTree:
        path = self.state_root / "research" / "research-tree.json"
        if path.exists():
            return ResearchTree.load(path)
        return ResearchTree(mission_id=self.mission.mission_id)

    def save(self) -> None:
        (self.state_root / "research").mkdir(parents=True, exist_ok=True)
        self.tree.save(self.state_root / "research" / "research-tree.json")
        (self.state_root / "research" / "provider-state.json").write_text(
            json.dumps(self.provider.export_state(), indent=2), encoding="utf-8"
        )

    # -- research loop ----------------------------------------------------------

    def generate_portfolio(self, *, count: int = 3) -> tuple[Hypothesis, ...]:
        context = self.director.export_provider_context(
            skill_estimates=self._skill_summaries_from_growth(),
            attempted_mechanisms=tuple(self.memory.method_effects()),
        )
        return self.director.generate_portfolio(context, count=count)

    def request_experiments(self) -> tuple[tuple[ExperimentProposal, CompiledExperiment], ...]:
        context = self.director.export_provider_context(
            skill_estimates=self._skill_summaries_from_growth(),
            attempted_mechanisms=tuple(self.memory.method_effects()),
        )
        hypotheses = tuple(self.memory.hypotheses(status="open"))
        return self.director.request_experiments(context, hypotheses)

    def record_observation(self, observation: ExperimentObservation) -> ExperimentObservation:
        return self.director.record_observation(observation)

    def interpret_and_review(self, hypothesis_id: str) -> ResearchFinding:
        observations = tuple(
            o for o in self.memory.observations()
            if o.hypothesis_id == hypothesis_id
        )
        hypothesis = next(
            (h for h in self.memory.hypotheses() if h.hypothesis_id == hypothesis_id),
            None,
        )
        if hypothesis is None:
            raise ResearchServiceError(f"unknown hypothesis: {hypothesis_id!r}")
        context = self.director.export_provider_context()
        finding = self.provider.interpret(context, hypothesis, observations)
        return self.director.review_finding(finding)

    def next_decision(self) -> ResearchDecision:
        return self.director.next_decision()

    def status(self) -> MissionView:
        st = self._director_status()
        decision = self.next_decision()
        return MissionView(
            mission_id=self.mission.mission_id,
            objective=self.mission.objective,
            autonomy=self.mission.autonomy,
            provider=self.provider.name,
            spent_gpu_hours=self.director.spend.spent_gpu_hours,
            remaining_gpu_hours=self.director.spend.remaining_gpu_hours(),
            tree_nodes=st.tree_nodes,
            active_branches=st.active_branches,
            hypotheses_open=st.hypotheses_open,
            findings=st.findings,
            decision=decision.kind,
        )

    def _director_status(self) -> DirectorStatus:
        return DirectorStatus(
            mission_id=self.mission.mission_id,
            hypotheses_open=len(self.memory.hypotheses(status="open")),
            hypotheses_rejected=len(self.memory.hypotheses(status="rejected")),
            proposals_admitted=len([r for r in self._proposal_rows() if r.get("admitted")]),
            proposals_refused=len([r for r in self._proposal_rows() if not r.get("admitted")]),
            observations=len(self.memory.observations()),
            findings=len(self.memory.findings()),
            spent_gpu_hours=self.director.spend.spent_gpu_hours,
            remaining_gpu_hours=self.director.spend.remaining_gpu_hours(),
            active_branches=len(self.tree.active_branches()),
            tree_nodes=len(self.tree.nodes()),
        )

    def _proposal_rows(self) -> list[dict[str, Any]]:
        from .research_memory import _read_jsonl
        rows = _read_jsonl(self.memory.proposals_path)
        return rows

    def _skill_summaries_from_growth(self) -> tuple[SkillSummary, ...]:
        """Export skill estimates from the growth state when a growth service
        is composed; without one, the export is empty (the provider sees no
        invented numbers)."""
        if self.growth_service is None:
            return ()
        try:
            profile = self.growth_service.current_skill_profile()  # type: ignore[attr-defined]
        except AttributeError:
            return ()
        summaries: list[SkillSummary] = []
        for evidence in profile.measured() if hasattr(profile, "measured") else ():
            summaries.append(SkillSummary(
                skill=evidence.skill,
                estimate=evidence.estimate,
                confidence=float(getattr(evidence, "confidence", 0.0)),
                uncertainty=float(getattr(evidence, "uncertainty", 1.0)),
            ))
        return tuple(summaries)
