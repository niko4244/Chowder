"""ResearchDirector: Chowder's native research control plane.

The director owns the research session: it exports the sanitized context the
provider may see, requests a hypothesis portfolio, admits or refuses each
proposal *before any compute*, records only run-grounded observations,
mechanically settles claim statuses against the replication policy, and
returns the next `ResearchDecision`. The provider proposes; the director
disposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .findings import Claim, ReplicationPolicy, ResearchFinding
from .hypothesis import Hypothesis
from .lab_bridge import CompiledExperiment, ExperimentCompiler
from .mission import MissionBudget, ResearchMission
from .observation import ExperimentObservation
from .proposal import ExperimentConstraint, ExperimentProposal
from .provider import ResearchContext, SkillSummary, ScientistProvider
from .research_decision import ResearchDecision
from .research_memory import ResearchMemory
from .research_tree import ResearchBranch, ResearchNode, ResearchTree


class DirectorRefusal(RuntimeError):
    """The director refused a provider submission. Fail-closed, journaled."""


@dataclass
class MissionSpend:
    """The mission's own ledger (the two-ledger rule: the growth envelope is
    the loop's; this is the research mission's, charged from measured costs)."""

    budget: MissionBudget
    spent_gpu_hours: float = 0.0
    spent_nodes: int = 0

    def remaining_gpu_hours(self) -> float:
        return max(0.0, self.budget.max_gpu_hours - self.spent_gpu_hours)

    def remaining_nodes(self) -> int:
        return max(0, self.budget.max_tree_nodes - self.spent_nodes)

    def admits(self, estimated_gpu_hours: float) -> tuple[bool, str]:
        if self.remaining_gpu_hours() < estimated_gpu_hours:
            return False, (
                f"MISSION_BUDGET_EXHAUSTED: estimated {estimated_gpu_hours} GPU-hours "
                f"but only {self.remaining_gpu_hours()} remain of "
                f"{self.budget.max_gpu_hours}"
            )
        if self.remaining_nodes() < 1:
            return False, (
                f"MISSION_NODE_BUDGET_EXHAUSTED: {self.spent_nodes}/"
                f"{self.budget.max_tree_nodes} tree nodes spent"
            )
        return True, ""

    def charge(self, *, gpu_hours: float, nodes: int = 1) -> None:
        self.spent_gpu_hours += max(0.0, gpu_hours)
        self.spent_nodes += nodes


@dataclass(frozen=True)
class DirectorStatus:
    mission_id: str
    hypotheses_open: int
    hypotheses_rejected: int
    proposals_admitted: int
    proposals_refused: int
    observations: int
    findings: int
    spent_gpu_hours: float
    remaining_gpu_hours: float
    active_branches: int
    tree_nodes: int

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class ResearchDirector:
    """One director drives one mission with one provider."""

    def __init__(
        self,
        *,
        mission: ResearchMission,
        provider: ScientistProvider,
        memory: ResearchMemory,
        tree: ResearchTree,
        compiler: ExperimentCompiler | None = None,
        replication_policy: ReplicationPolicy | None = None,
        protected_benchmarks: tuple[str, ...] = (),
        run_exists: Callable[[str], bool] | None = None,
        run_complete: Callable[[str], bool] | None = None,
    ) -> None:
        self.mission = mission
        self.provider = provider
        self.memory = memory
        self.tree = tree
        self.compiler = compiler or ExperimentCompiler(
            protected_benchmarks=tuple(protected_benchmarks),
        )
        self.replication_policy = replication_policy or ReplicationPolicy()
        self.spend = MissionSpend(budget=mission.budget)
        self.constraints = ExperimentConstraint(
            allowed_experiment_types=tuple(mission.allowed_experiment_types),
            max_gpu_hours_per_experiment=max(0.001, mission.budget.max_gpu_hours),
            protected_capabilities=tuple(mission.protected_capabilities),
        )
        #: run-registry resolvers for observation grounding; default accept
        #: keeps the director testable without a registry (tests pass explicit
        #: resolvers; production wires the registry).
        self._run_exists = run_exists or (lambda run_id: True)
        self._run_complete = run_complete or (lambda run_id: True)

    # -- context export ------------------------------------------------------

    def export_provider_context(
        self,
        *,
        skill_estimates: tuple[SkillSummary, ...] = (),
        open_failure_categories: tuple[dict[str, Any], ...] = (),
        attempted_mechanisms: tuple[dict[str, Any], ...] = (),
        carried_evidence: tuple[Any, ...] = (),
        architecture: str = "",
    ) -> ResearchContext:
        from .provider import CarriedEvidence
        carried = tuple(
            c if isinstance(c, CarriedEvidence)
            else CarriedEvidence(
                statement=str(c.get("statement", "")),
                source_run_ids=tuple(str(r) for r in c.get("source_run_ids", ())),
                capability=str(c.get("capability", "")),
            )
            for c in carried_evidence
        )
        return ResearchContext(
            mission=self.mission,
            skill_estimates=skill_estimates,
            open_failure_categories=open_failure_categories,
            attempted_mechanisms=attempted_mechanisms,
            carried_evidence=carried,
            remaining_gpu_hours=self.spend.remaining_gpu_hours(),
            architecture=architecture,
        )

    # -- portfolio -------------------------------------------------------------

    def generate_portfolio(self, context: ResearchContext, *,
                           count: int = 3) -> tuple[Hypothesis, ...]:
        if not self.provider.available():
            raise DirectorRefusal(
                f"PROVIDER_UNAVAILABLE: {self.provider.name} cannot run; "
                "refusing rather than fabricating research"
            )
        portfolio = self.provider.propose_hypotheses(context, count=count)
        stamped: list[Hypothesis] = []
        for hyp in portfolio:
            stamped.append(Hypothesis.from_dict({
                **hyp.to_dict(),
                "provider": self.provider.name,
                "mission_id": self.mission.mission_id,
            }))
            self.memory.record_hypothesis(stamped[-1])
        return tuple(stamped)

    def request_experiments(
        self,
        context: ResearchContext,
        hypotheses: tuple[Hypothesis, ...],
    ) -> tuple[tuple[ExperimentProposal, CompiledExperiment], ...]:
        """Admit + compile the provider's proposals. Refusals are journaled;
        compute is never spent on a refused proposal."""
        proposals = self.provider.propose_experiments(context, hypotheses)
        compiled: list[tuple[ExperimentProposal, CompiledExperiment]] = []
        for proposal in proposals:
            reasons = self.admit_proposal(proposal)
            if reasons:
                self.memory.record_proposal(proposal, admitted=False,
                                            refusal_reasons=reasons)
                self.memory.record_refusal(
                    provider=proposal.provider or self.provider.name,
                    kind="proposal_refused",
                    detail="; ".join(reasons),
                    payload={"proposal_id": proposal.proposal_id},
                )
                continue
            admitted = ExperimentProposal.from_dict({
                **proposal.to_dict(), "status": "admitted",
            })
            self.memory.record_proposal(admitted, admitted=True)
            experiment = self.compiler.compile(admitted)
            compiled.append((admitted, experiment))
        return tuple(compiled)

    def admit_proposal(self, proposal: ExperimentProposal) -> tuple[str, ...]:
        """Fail-closed admission: schema → constraints → mission budget.
        Zero compute before all three pass."""
        try:
            reasons = list(proposal.validate(self.constraints))
        except ValueError as error:
            return (f"SCHEMA_INVALID: {error}",)
        if reasons:
            return tuple(reasons)
        # budget admission (per-experiment ceiling already in constraints;
        # here against the mission's *remaining* envelope)
        affordable, reason = self.spend.admits(proposal.estimated_gpu_hours)
        if not affordable:
            reasons.append(reason)
        known = {h.hypothesis_id for h in self.memory.hypotheses()}
        if proposal.hypothesis_id not in known:
            reasons.append(
                f"HYPOTHESIS_UNKNOWN: {proposal.hypothesis_id!r} was never "
                "recorded; no orphan experiments"
            )
        return tuple(reasons)

    # -- execution results -------------------------------------------------------

    def start_experiment(self, proposal: ExperimentProposal,
                         experiment: CompiledExperiment) -> ResearchNode:
        affordable, reason = self.spend.admits(experiment.estimated_gpu_hours)
        if not affordable:
            raise DirectorRefusal(reason)
        node = ResearchNode(
            node_id=f"node-{experiment.experiment_id}",
            parent_id=None,
            hypothesis_id=experiment.hypothesis_id,
            proposal_id=experiment.proposal_id,
            experiment_ref=experiment.experiment_id,
            status="running",
        )
        self.tree.add_node(node)
        branch_id = f"branch-{experiment.hypothesis_id}"
        try:
            self.tree.branch(branch_id)
        except KeyError:
            self.tree.add_branch(ResearchBranch(
                branch_id=branch_id,
                hypothesis_id=experiment.hypothesis_id,
                root_node_id=node.node_id,
            ))
        return node

    def record_observation(self, observation: ExperimentObservation) -> ExperimentObservation:
        if observation.run_id and not self._run_exists(observation.run_id):
            raise DirectorRefusal(
                f"REFUSAL_RUN_UNKNOWN: {observation.observation_id} cites run "
                f"{observation.run_id!r}, unknown to the run registry"
            )
        if (observation.status == "complete" and observation.run_id
                and not self._run_complete(observation.run_id)):
            raise DirectorRefusal(
                f"REFUSAL_RUN_INCOMPLETE: run {observation.run_id!r} is not complete"
            )
        self.memory.record_observation(observation)
        self.spend.charge(gpu_hours=observation.wall_gpu_hours)
        node_id = f"node-sciexp-{observation.proposal_id}"
        try:
            node = self.tree.node(node_id)
        except KeyError:
            node = self.tree.add_node(ResearchNode(
                node_id=node_id,
                parent_id=None,
                hypothesis_id=observation.hypothesis_id,
                proposal_id=observation.proposal_id,
                experiment_ref=observation.experiment_ref,
                status="running",
            ))
        node.status = "observed"
        node.observation_ids = tuple({*node.observation_ids, observation.observation_id})
        node.evidence_refs = tuple({*node.evidence_refs, observation.run_id})
        node.measured_cost_gpu_hours += observation.wall_gpu_hours
        # The capability-delta feed takes quality deltas only, and only from
        # the mission's declared hardware context when one is established: a
        # cross-hardware efficiency number never masquerades as a capability
        # gain (docs/COMPUTE_PROVIDERS.md §3). The first observed hardware
        # class anchors the context; others are recorded but scored 0.
        quality = [m for m in observation.measurements
                   if not m.surface.startswith("efficiency:")]
        if quality and observation.status == "complete":
            if node.capability_delta is None:
                node.capability_delta = max(m.value for m in quality)
                self._node_hardware_class = observation.hardware_class
            elif observation.hardware_class == getattr(self, "_node_hardware_class", ""):
                node.capability_delta = max(node.capability_delta,
                                            max(m.value for m in quality))
        return observation

    def review_finding(self, finding: ResearchFinding) -> ResearchFinding:
        """The provider proposes a finding; Chowder settles the claim statuses
        mechanically from the replication policy and the cited observations.

        - supporting runs < replication seeds → provisional
        - transfer absent/failed → never "replicated", never a capability gain
        - zero successful runs → rejected
        """
        observations = {o.observation_id: o for o in self.memory.observations()}
        reviewed_claims: list[Claim] = []
        for claim in finding.claims:
            if not isinstance(claim, Claim):
                claim = Claim.from_dict(dict(claim))
            successful = sum(
                1 for run_id in claim.supporting_experiments
                if any(o.run_id == run_id and o.status == "complete"
                       for o in observations.values())
            )
            transfer_surfaces = {
                m.surface
                for o in observations.values()
                if o.status == "complete"
                for m in o.measurements
                if m.surface.startswith("transfer:")
            }
            transfer_supported: bool | None = None
            if claim.affected_capabilities:
                # a claim about capability X needs transfer:X measurements
                needed = [f"transfer:{c}" for c in claim.affected_capabilities
                          if f"transfer:{c}" in self._transfer_surfaces_declared()]
                if needed:
                    transfer_supported = all(s in transfer_surfaces for s in needed)
            status = self.replication_policy.claim_status_for(
                successful_runs=successful,
                transfer_supported=transfer_supported,
            )
            reviewed_claims.append(Claim(
                **{**claim.to_dict(),
                   "status": status,
                   "replication_count": successful},
            ))
        reviewed = ResearchFinding(
            finding_id=finding.finding_id,
            hypothesis_id=finding.hypothesis_id,
            claims=tuple(reviewed_claims),
            observation_ids=finding.observation_ids,
            provider=finding.provider,
            mission_id=finding.mission_id,
            note=finding.note,
        )
        self.memory.record_finding(reviewed)
        return reviewed

    def _transfer_surfaces_declared(self) -> tuple[str, ...]:
        declared: list[str] = []
        for proposal_row in self.memory.proposals_path.read_text().splitlines() \
                if self.memory.proposals_path.exists() else []:
            import json
            try:
                record = json.loads(proposal_row).get("record", {})
            except json.JSONDecodeError:
                continue
            declared.extend(f"transfer:{surface}"
                            for surface in record.get("transfer_evaluations", ()))
        return tuple(set(declared))

    # -- decisions ---------------------------------------------------------------

    def submit_screening_batch(
        self,
        scheduler: Any,
        pairs: tuple[tuple[ExperimentProposal, Any], ...],
    ) -> tuple[tuple[ExperimentProposal, Any, Any], ...]:
        """Route the screening stage: short/cheap runs for each candidate,
        preferably on the declared screening lane (Kaggle). Returns the
        (proposal, experiment, submission) triples that were admitted by the
        scheduler; refusals are journaled."""
        from .compute import ExperimentClass, ExperimentRequest
        out: list[tuple[ExperimentProposal, Any, Any]] = []
        for proposal, experiment in pairs:
            request = ExperimentRequest(
                experiment_id=experiment.experiment_id,
                proposal_id=proposal.proposal_id,
                hypothesis_id=proposal.hypothesis_id,
                campaign_spec=experiment.campaign_spec,
                experiment_class=ExperimentClass.SCREENING,
                estimated_gpu_hours=min(experiment.estimated_gpu_hours, 0.25),
            )
            try:
                submission = scheduler.schedule(request)
            except Exception as error:
                self.memory.record_refusal(
                    provider=getattr(scheduler, "name", "scheduler"),
                    kind="screening_refused",
                    detail=str(error),
                    payload={"experiment_id": experiment.experiment_id},
                )
                continue
            out.append((proposal, experiment, submission))
        return tuple(out)

    def submit_survivor_batch(
        self,
        scheduler: Any,
        triples: tuple[tuple[ExperimentProposal, Any, Any], ...],
        *,
        keep_top: int = 2,
    ) -> tuple[tuple[ExperimentProposal, Any, Any], ...]:
        """Promote the best `keep_top` screening survivors to substantial
        runs through the scheduler's preference order. Selection is the tree's
        deterministic branch score, not an LLM's opinion."""
        from .compute import ExperimentClass, ExperimentRequest
        scored = sorted(
            triples,
            key=lambda t: self.tree.score_node(
                self.tree.node(f"node-{t[1].experiment_id}"),
            ) if self.tree_has_node(f"node-{t[1].experiment_id}") else -1e9,
            reverse=True,
        )
        survivors = scored[:keep_top]
        out: list[tuple[ExperimentProposal, Any, Any]] = []
        for proposal, experiment, _screening in survivors:
            request = ExperimentRequest(
                experiment_id=experiment.experiment_id,
                proposal_id=proposal.proposal_id,
                hypothesis_id=proposal.hypothesis_id,
                campaign_spec=experiment.campaign_spec,
                experiment_class=ExperimentClass.SUBSTANTIAL,
                estimated_gpu_hours=experiment.estimated_gpu_hours,
            )
            try:
                submission = scheduler.schedule(request)
            except Exception as error:
                self.memory.record_refusal(
                    provider=getattr(scheduler, "name", "scheduler"),
                    kind="survivor_refused",
                    detail=str(error),
                    payload={"experiment_id": experiment.experiment_id},
                )
                continue
            out.append((proposal, experiment, submission))
        return tuple(out)

    def tree_has_node(self, node_id: str) -> bool:
        try:
            self.tree.node(node_id)
            return True
        except KeyError:
            return False

    def next_decision(self) -> ResearchDecision:
        """The research loop's next move, from durable state only."""
        pruned = self.tree.prune_repeated_failures()
        if self.tree.plateaued():
            return ResearchDecision(
                "stop_plateau", subject_id=self.mission.mission_id,
                reason="no capability movement across the plateau window",
            )
        affordable, reason = self.spend.admits(0.0)
        if not affordable or self.spend.remaining_nodes() < 1:
            return ResearchDecision(
                "stop_budget", subject_id=self.mission.mission_id, reason=reason,
            )
        open_hypotheses = [h for h in self.memory.hypotheses(status="open")]
        if not open_hypotheses:
            return ResearchDecision(
                "stop_no_admissible_hypothesis", subject_id=self.mission.mission_id,
                reason="no open hypothesis remains",
            )
        ranked = self.tree.rank_active_branches()
        if ranked:
            best_score, best_branch = ranked[0]
            return ResearchDecision(
                "expand_branch", subject_id=best_branch.branch_id,
                reason=f"highest-scoring active branch (score {best_score:.3f})",
            )
        return ResearchDecision(
            "human_review", subject_id=self.mission.mission_id,
            reason="open hypotheses exist but no active branch carries them",
        )
