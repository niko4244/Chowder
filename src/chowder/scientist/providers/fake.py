"""FakeDeterministicScientistProvider: the Phase-1 test double.

Tests-only, and named as such: it deterministically turns the exported context
into hypotheses/proposals/findings so the director, tree and memory can be
exercised without any LLM. It deliberately emits one over-budget proposal and
one architecture proposal among its candidates, so refusal paths get exercised
the same way the happy path does.
"""

from __future__ import annotations

from typing import Any

from ..findings import Claim, ResearchFinding
from ..hypothesis import Hypothesis, ResearchQuestion
from ..mission import ResearchMission
from ..observation import ExperimentObservation
from ..proposal import (
    DataStrategy,
    ExperimentProposal,
    TrainingRecipeDelta,
)
from ..provider import ResearchContext

TESTS_ONLY = True  # an import-time statement: never wire this in production


class FakeDeterministicScientistProvider:
    name = "fake_deterministic"

    def __init__(self, *, hypotheses_per_round: int = 3) -> None:
        self._hypotheses_per_round = hypotheses_per_round
        self._rounds = 0
        self._state: dict[str, Any] = {}

    def available(self) -> bool:
        return True

    def propose_hypotheses(self, context: ResearchContext,
                           *, count: int = 3) -> tuple[Hypothesis, ...]:
        out: list[Hypothesis] = []
        weakest = self._weakest_skill(context)
        for i in range(min(count, self._hypotheses_per_round)):
            n = self._rounds * self._hypotheses_per_round + i
            out.append(Hypothesis(
                hypothesis_id=f"fake-hyp-{n:03d}",
                research_question=ResearchQuestion(
                    text=f"does replay-ratio decay improve {weakest} transfer?",
                    capability=weakest,
                ),
                observation=(
                    f"{weakest} is the weakest measured skill "
                    f"(estimate {self._estimate_of(context, weakest)})"
                ),
                suspected_mechanism="replay material crowds new-skill gradient signal",
                predicted_effect=f"{weakest} transfer delta > 0 without protected regression",
                novelty_basis="not in attempted mechanisms" if n == 0 else "round-robin variant",
                uncertainty="medium",
                falsification_conditions=(
                    f"{weakest} delta <= 0 on two seeds",
                    "any protected capability regresses beyond tolerance",
                ),
                expected_information_gain=0.4,
            ))
        self._rounds += 1
        return tuple(out)

    def propose_experiments(
        self,
        context: ResearchContext,
        hypotheses: tuple[Hypothesis, ...],
    ) -> tuple[ExperimentProposal, ...]:
        out: list[ExperimentProposal] = []
        for i, hyp in enumerate(hypotheses):
            surface = hyp.research_question.capability
            # the fake's candidates deliberately include:
            #  - an over-budget proposal (refused: budget),
            #  - an architecture proposal (refused: type not allowed),
            #  - one admissible data-strategy proposal.
            if i == 1:
                out.append(self._proposal(hyp, surface, estimated=10_000.0))
            elif i == 2:
                proposal = self._proposal(hyp, surface, estimated=0.5)
                out.append(ExperimentProposal.from_dict({
                    **proposal.to_dict(), "experiment_type": "architecture",
                    "proposal_id": proposal.proposal_id + "-arch",
                }))
            else:
                out.append(self._proposal(hyp, surface, estimated=0.5))
        return tuple(out)

    def _proposal(self, hyp: Hypothesis, surface: str, *,
                  estimated: float) -> ExperimentProposal:
        return ExperimentProposal(
            proposal_id=f"fake-prop-{hyp.hypothesis_id}",
            hypothesis_id=hyp.hypothesis_id,
            experiment_type="data",
            intervention="decay replay ratio 0.25 -> 0.05 after convergence",
            variables_changed=("replay_ratio",),
            variables_held_constant=("learning_rate", "epochs", "lora_rank"),
            training_recipe_delta=TrainingRecipeDelta(epochs=2),
            data_strategy=DataStrategy(
                source_kinds=("curriculum", "replay"),
                replay_ratio=0.05,
            ),
            requested_evaluations=(surface,),
            transfer_evaluations=(f"{surface}-alt",),
            replication_plan="3 seeds; transfer stage on survivors",
            controls=("parent-adapter baseline",),
            expected_outcome=f"{surface} improves, protected flat",
            falsification_rule="transfer delta <= 0 on 2 of 3 seeds",
            estimated_gpu_hours=estimated,
            provider=self.name,
        )

    def interpret(
        self,
        context: ResearchContext,
        hypothesis: Hypothesis,
        observations: tuple[ExperimentObservation, ...],
    ) -> ResearchFinding:
        surface = hypothesis.research_question.capability
        supporting = tuple(
            o.run_id for o in observations if o.status == "complete"
        )
        claim = Claim(
            claim_id=f"fake-claim-{hypothesis.hypothesis_id}",
            statement=(
                f"replay-ratio decay improves {surface} transfer on this family"
            ),
            scope="test-scope",
            status="provisional",
            supporting_experiments=supporting,
            affected_capabilities=(surface,),
            conditions=("3 seeds", "transfer stage"),
        )
        return ResearchFinding(
            finding_id=f"fake-finding-{hypothesis.hypothesis_id}",
            hypothesis_id=hypothesis.hypothesis_id,
            claims=(claim,),
            observation_ids=tuple(o.observation_id for o in observations),
            provider=self.name,
        )

    def export_state(self) -> dict[str, Any]:
        return {"rounds": self._rounds}

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _weakest_skill(context: ResearchContext) -> str:
        measured = [s for s in context.skill_estimates if s.estimate is not None]
        if not measured:
            return "reasoning"
        return min(measured, key=lambda s: s.estimate).skill

    @staticmethod
    def _estimate_of(context: ResearchContext, skill: str) -> float | None:
        for s in context.skill_estimates:
            if s.skill == skill:
                return s.estimate
        return None
