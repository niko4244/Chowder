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

from .compute import ExperimentScheduler, provider_from_config
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
from .screening_halving import (
    ScreeningHalving,
    screening_score_from_node,
    settle_screening_round,
)


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


@dataclass
class ScreeningSession:
    """Durable state of the screening lane's successive-halving schedule
    (`state_root/research/screening-session.json`). The schedule advances one
    step per `advance_screening` call; every field here survives a restart,
    so `scientist screen` is resumable across process invocations."""

    schedule: dict[str, Any]                 # ScreeningHalving fields
    round_index: int = 0
    started: bool = False
    #: experiment_id -> {"proposal": ..., "experiment": ...} for every
    #: candidate this session has ever considered (needed to resubmit
    #: survivors and to reconstruct the graduated pairs).
    candidates: dict[str, dict[str, Any]] = None  # type: ignore[assignment]
    active: tuple[str, ...] = ()             # candidates in the current round
    awaiting: tuple[str, ...] = ()           # submitted this round, not yet observed
    gate_eliminated: tuple[str, ...] = ()    # refused by the scheduler this round
    settled_rounds: tuple[dict[str, Any], ...] = ()
    final_survivors: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.candidates is None:
            self.candidates = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule": dict(self.schedule),
            "round_index": self.round_index,
            "started": self.started,
            "candidates": self.candidates,
            "active": list(self.active),
            "awaiting": list(self.awaiting),
            "gate_eliminated": list(self.gate_eliminated),
            "settled_rounds": list(self.settled_rounds),
            "final_survivors": (None if self.final_survivors is None
                                else list(self.final_survivors)),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScreeningSession":
        return cls(
            schedule=dict(data["schedule"]),
            round_index=int(data.get("round_index", 0)),
            started=bool(data.get("started", False)),
            candidates=dict(data.get("candidates") or {}),
            active=tuple(data.get("active") or ()),
            awaiting=tuple(data.get("awaiting") or ()),
            gate_eliminated=tuple(data.get("gate_eliminated") or ()),
            settled_rounds=tuple(data.get("settled_rounds") or ()),
            final_survivors=(None if data.get("final_survivors") is None
                             else tuple(data["final_survivors"])),
        )

    def schedule_obj(self) -> ScreeningHalving:
        return ScreeningHalving(**self.schedule)


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
        scheduler: Any | None = None,
        screening_schedule: ScreeningHalving | None = None,
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
        #: The compute layer (docs/COMPUTE_PROVIDERS.md): None unless the
        #: policy declares providers (or a scheduler is injected). Every
        #: screening advance refuses when this is absent — no default compute.
        self.scheduler = scheduler
        self.screening_schedule = screening_schedule or ScreeningHalving()

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
            **cls._compute_from_policy(scientist_policy),
        )

    @staticmethod
    def _compute_from_policy(scientist_policy: dict[str, Any]) -> dict[str, Any]:
        """The optional `compute` section of the scientist policy:
        {"providers": [<provider_from_config entries>], "screening":
        {<ScreeningHalving fields>}}. Absent → no compute layer (screening
        refuses); present with zero providers → a named error, never a
        silent default."""
        compute = scientist_policy.get("compute")
        if compute is None:
            return {}
        if not isinstance(compute, dict):
            raise ResearchServiceError(
                "COMPUTE_POLICY_INVALID: the `compute` section must be a mapping"
            )
        specs = compute.get("providers") or []
        if not specs:
            raise ResearchServiceError(
                "COMPUTE_DECLARED_WITHOUT_PROVIDERS: the policy declares a "
                "compute section but no providers; refusing rather than "
                "assuming local"
            )
        from .compute import ExperimentScheduler
        try:
            scheduler = ExperimentScheduler(
                [provider_from_config(dict(spec)) for spec in specs])
        except Exception as error:
            raise ResearchServiceError(
                f"COMPUTE_POLICY_INVALID: {error}") from error
        allowed = {
            "initial_budget_gpu_hours", "budget_cap_gpu_hours",
            "step_multiplier", "survival_fraction", "min_survivors", "max_rounds",
        }
        screening_cfg = dict(compute.get("screening") or {})
        unknown = sorted(set(screening_cfg) - allowed)
        if unknown:
            raise ResearchServiceError(
                f"COMPUTE_POLICY_INVALID: unknown screening schedule keys: {unknown}"
            )
        return {
            "scheduler": scheduler,
            "screening_schedule": ScreeningHalving(**screening_cfg),
        }

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

    # -- screening lane (successive halving, resumable) -------------------------

    @property
    def _screening_session_path(self) -> Path:
        return self.state_root / "research" / "screening-session.json"

    def _load_screening_session(self) -> ScreeningSession | None:
        if not self._screening_session_path.exists():
            return None
        return ScreeningSession.from_dict(
            json.loads(self._screening_session_path.read_text(encoding="utf-8")))

    def _save_screening_session(self, session: ScreeningSession) -> None:
        self._screening_session_path.parent.mkdir(parents=True, exist_ok=True)
        self._screening_session_path.write_text(
            json.dumps(session.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _compiled_from_dict(data: dict[str, Any]) -> CompiledExperiment:
        return CompiledExperiment(
            experiment_id=str(data["experiment_id"]),
            proposal_id=str(data["proposal_id"]),
            hypothesis_id=str(data["hypothesis_id"]),
            campaign_spec=dict(data.get("campaign_spec") or {}),
            estimated_gpu_hours=float(data.get("estimated_gpu_hours", 0.0)),
        )

    def screening_pairs_from_memory(self) -> tuple[tuple[ExperimentProposal, CompiledExperiment], ...]:
        """Rebuild (proposal, experiment) pairs from the ADMITTED proposals in
        durable memory — screening is always grounded in recorded admission,
        never re-asked from the provider. Admitted proposals the compiler
        cannot compile (e.g. architecture) are journaled and skipped."""
        from .lab_bridge import CompilationRefusal
        latest: dict[str, dict[str, Any]] = {}
        for row in self._proposal_rows():
            if row.get("admitted"):
                record = dict(row.get("record") or {})
                pid = str(record.get("proposal_id", ""))
                if pid:
                    latest[pid] = record
        out: list[tuple[ExperimentProposal, CompiledExperiment]] = []
        for pid, record in sorted(latest.items()):
            proposal = ExperimentProposal.from_dict(record)
            try:
                out.append((proposal, self.director.compiler.compile(proposal)))
            except CompilationRefusal as error:
                self.memory.record_refusal(
                    provider="screening_halving", kind="screening_pair_skipped",
                    detail=str(error), payload={"proposal_id": pid})
        return tuple(out)

    def _node_has_observations(self, experiment_id: str) -> bool:
        node_id = f"node-{experiment_id}"
        if not self.director.tree_has_node(node_id):
            return False
        return bool(self.tree.node(node_id).observation_ids)

    def _settle_screening_round(self, session: ScreeningSession,
                                schedule: ScreeningHalving) -> dict[str, Any]:
        """Settle the current round mechanically from the tree (gate vs
        cutoff), journal eliminations, and either graduate or set up the next
        round — the same rules as ResearchDirector.run_screening_halving."""
        scores: dict[str, float | None] = {}
        for experiment_id in session.awaiting:
            node_id = f"node-{experiment_id}"
            if not self.director.tree_has_node(node_id):
                scores[experiment_id] = None
                continue
            node = self.tree.node(node_id)
            scores[experiment_id] = screening_score_from_node(node, self.tree.score_node)
        survivors, by_gate, by_cutoff, ranked = settle_screening_round(scores, schedule)
        by_gate = tuple(sorted(set(by_gate) | set(session.gate_eliminated)))
        for experiment_id, reason in (
                [(eid, "gate: no usable observation") for eid in by_gate]
                + [(eid, "cutoff: scored below the survivor line") for eid in by_cutoff]):
            score = next((s for eid2, s in ranked if eid2 == experiment_id), None)
            self.memory.record_refusal(
                provider="screening_halving", kind="screening_eliminated",
                detail=f"round {session.round_index}: {reason}",
                payload={
                    "round_index": session.round_index,
                    "experiment_id": experiment_id,
                    "score": score,
                })
        settled = {
            "round_index": session.round_index,
            "submitted": list(session.awaiting),
            "survivors": list(survivors),
            "eliminated_by_gate": list(by_gate),
            "eliminated_by_cutoff": list(by_cutoff),
            "scores": [[eid, score] for eid, score in ranked],
        }
        session.settled_rounds = tuple([*session.settled_rounds, settled])
        if not survivors or schedule.is_final_round(session.round_index, len(survivors)):
            session.final_survivors = survivors
        else:
            session.round_index += 1
            session.active = survivors
        session.awaiting = ()
        session.gate_eliminated = ()
        return settled

    def advance_screening(
        self,
        *,
        pairs: tuple[tuple[ExperimentProposal, CompiledExperiment], ...] | None = None,
        record_results: Callable[..., None] | None = None,
    ) -> dict[str, Any]:
        """Advance the screening lane ONE durable step (resumable across
        processes; `scientist screen` calls this repeatedly):

        - not started → submit round 0 for every admitted pair at the
          schedule's round-0 budget;
        - awaiting observations → report what is missing, spend nothing;
        - observations in → settle the round (tree score, gate vs cutoff,
          journaled) and submit the next round for the survivors at the
          multiplied budget, or graduate the final survivors.

        `record_results(triples, round_index)` is the in-process seam: when
        given, it is invoked right after each submit (the research loop
        records run-grounded observations there). Without it the step
        returns `awaiting` and the observations arrive through
        `record_observation` (CLI `scientist observe`).
        """
        if self.scheduler is None:
            raise ResearchServiceError(
                "SCREENING_NOT_CONFIGURED: the scientist policy declares no "
                "compute providers; refusing rather than assuming local"
            )
        session = self._load_screening_session() or ScreeningSession(
            schedule=self.screening_schedule.to_dict())
        schedule = session.schedule_obj()
        if session.final_survivors is not None:
            return {"phase": "complete", "mission_id": self.mission.mission_id,
                    "final_survivors": list(session.final_survivors),
                    "rounds": list(session.settled_rounds)}

        # 1. settle a round whose observations have all landed
        if session.started and session.awaiting:
            missing = [eid for eid in session.awaiting
                       if not self._node_has_observations(eid)]
            if missing:
                return {"phase": "awaiting", "mission_id": self.mission.mission_id,
                        "round_index": session.round_index,
                        "awaiting": list(session.awaiting),
                        "missing_observations": missing}
            settled = self._settle_screening_round(session, schedule)
            if session.final_survivors is not None:
                self._save_screening_session(session)
                return {"phase": "graduated", "mission_id": self.mission.mission_id,
                        "settled_round": settled,
                        "final_survivors": list(session.final_survivors),
                        "rounds": list(session.settled_rounds)}

        # 2. submit the current round
        if not session.started:
            if pairs is None:
                pairs = self.screening_pairs_from_memory()
            if not pairs:
                raise ResearchServiceError(
                    "SCREENING_NO_CANDIDATES: no admitted proposals to screen; "
                    "run the portfolio/admission step first"
                )
            session.candidates = {
                experiment.experiment_id: {
                    "proposal": proposal.to_dict(),
                    "experiment": experiment.to_dict(),
                }
                for proposal, experiment in pairs
            }
            session.active = tuple(
                experiment.experiment_id for _, experiment in pairs)
            session.started = True
        budget = schedule.round_budget(session.round_index)
        current_pairs = tuple(
            (ExperimentProposal.from_dict(session.candidates[eid]["proposal"]),
             self._compiled_from_dict(session.candidates[eid]["experiment"]))
            for eid in session.active
        )
        triples = self.director.submit_screening_batch(
            self.scheduler, current_pairs, budget_gpu_hours=budget)
        submitted_ids = tuple(e.experiment_id for _, e, _ in triples)
        refused = tuple(eid for eid in session.active
                        if eid not in set(submitted_ids))
        session.awaiting = submitted_ids
        session.gate_eliminated = refused
        if triples and record_results is not None:
            record_results(triples, session.round_index)
        self._save_screening_session(session)
        if not submitted_ids:
            # nothing the scheduler could accept this round: the lane is
            # exhausted (honestly) rather than retried blindly
            self._settle_screening_round(session, schedule)
            self._save_screening_session(session)
            return {"phase": "exhausted", "mission_id": self.mission.mission_id,
                    "round_index": session.round_index,
                    "refused": list(refused),
                    "final_survivors": list(session.final_survivors or ())}
        return {"phase": "submitted", "mission_id": self.mission.mission_id,
                "round_index": session.round_index,
                "budget_gpu_hours": budget,
                "awaiting": list(submitted_ids),
                "refused": list(refused)}

    def screening_status(self) -> dict[str, Any] | None:
        """The durable screening session, for `scientist status` callers."""
        session = self._load_screening_session()
        return None if session is None else session.to_dict()

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
