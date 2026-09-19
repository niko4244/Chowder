"""The outer controller: generation N decides whether there is an N+1.

``campaign_runner`` runs one declared campaign and ``certification`` decides its
verdict. Neither of them knows when to stop, what to try next, or how much of
the session's budget is left -- and none of them may be talked into a second
generation. That authority lives here, in the one component that can see
durable state, actual spend and an immutable policy at the same time.

The loop is deliberately a *thin* orchestration layer with every heavy step
behind a seam:

* ``executor`` runs a frozen campaign and reports what it actually cost and
  actually did (production wires it to ``campaign_runner.run_campaign``; the
  fake-compute simulator wires it to a deterministic stand-in);
* ``prepare`` and ``readiness`` are the production preparation and zero-compute
  readiness gate by default;
* the selector, the campaign builder and the durable state are injected.

What the loop itself owns is the part that must not be delegated:

* **a finite stopping condition.** Every iteration either continues or reaches
  one of a closed set of terminal decisions, and the envelope is re-checked
  before each generation is allowed to spend anything;
* **budget from measurement, never from arithmetic on intent.** The remaining
  envelope is decremented by what the accounting artifact says was spent. A
  campaign that ran without reporting a measured cost stops the loop as
  UNCERTAIN -- it does not get charged an estimate, and it does not get treated
  as free;
* **no self-authorisation.** The policy supplies every ceiling, the protected
  set, the trusted ancestor and the allowed treatments, and the loop can only
  read them. A target the policy does not allow becomes human review, not a
  campaign.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .campaign import CampaignManifest
from .next_campaign import (
    FrozenCampaign,
    LoopPolicy,
    NextCampaignBuilder,
    NextCampaignRefusal,
)
from .target_selection import (
    GrowthState,
    NextTargetSelector,
    SkillProfile,
    TargetProposal,
    build_skill_profile,
)

# -- terminal and continuing decisions --------------------------------------

CONTINUE = "CONTINUE"
STOP_SUCCESS = "STOP_SUCCESS"
STOP_BUDGET = "STOP_BUDGET"
STOP_PLATEAU = "STOP_PLATEAU"
STOP_UNCERTAIN = "STOP_UNCERTAIN"
STOP_UNTRAINABLE = "STOP_UNTRAINABLE"
REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"

TERMINAL_DECISIONS = frozenset(
    {
        STOP_SUCCESS,
        STOP_BUDGET,
        STOP_PLATEAU,
        STOP_UNCERTAIN,
        STOP_UNTRAINABLE,
        REQUIRES_HUMAN_REVIEW,
    }
)

# Named reasons. A stop that cannot say why is not a decision.
NO_MEASURED_CAPABILITY = "NO_MEASURED_CAPABILITY"
NO_TARGET_AVAILABLE = "NO_TARGET_AVAILABLE"
TREATMENT_REQUIRES_REVIEW = "TREATMENT_REQUIRES_REVIEW"
REMAINING_ENVELOPE_TOO_SMALL = "REMAINING_ENVELOPE_TOO_SMALL"
PREPARATION_REFUSED = "PREPARATION_REFUSED"
READINESS_REFUSED = "READINESS_REFUSED"
CAMPAIGN_SPENT_NO_MEASURED_COST = "CAMPAIGN_SPENT_NO_MEASURED_COST"
PROMOTION_WITHOUT_IDENTITY = "PROMOTION_WITHOUT_IDENTITY"
PROMOTION_WITHOUT_PROFILE = "PROMOTION_WITHOUT_PROFILE"
TARGET_EXHAUSTED = "TARGET_EXHAUSTED"
PROMOTION_WITHOUT_DECLARATION = "PROMOTION_WITHOUT_DECLARATION"
CONSECUTIVE_NON_PROMOTIONS = "CONSECUTIVE_NON_PROMOTIONS"
GENERATION_LIMIT_REACHED = "GENERATION_LIMIT_REACHED"
NOTHING_LEFT_TO_IMPROVE = "NOTHING_LEFT_TO_IMPROVE"
PROFILE_GENERATION_MISMATCH = "PROFILE_GENERATION_MISMATCH"

#: Treatments that the loop will never start on its own, whatever the score.
#: They are not "bad targets": they are targets whose safety this loop cannot
#: establish, so a human decides.
REVIEW_TREATMENTS = frozenset(
    {
        "architecture_research",
        "evaluation_needed",
        "untrainable_with_current_path",
    }
)


@dataclass(frozen=True)
class LoopDecision:
    """What the loop decided to do next, and the evidence for it."""

    action: str
    reason: str
    reason_codes: tuple[str, ...] = ()

    @property
    def terminal(self) -> bool:
        return self.action in TERMINAL_DECISIONS

    @property
    def requires_human(self) -> bool:
        return self.action == REQUIRES_HUMAN_REVIEW

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "reason_codes": list(self.reason_codes),
            "terminal": self.terminal,
        }


@dataclass(frozen=True)
class CampaignOutcome:
    """What one executed campaign tells the loop. The executor's return value.

    ``wall_gpu_hours`` is *measured* spend read from the campaign's own
    accounting artifact. ``None`` means the campaign did not report a measured
    cost, which the loop treats as an unaccounted execution rather than as zero
    -- an unmeasured cost is not a discount.
    """

    verdict: str
    wall_gpu_hours: float | None
    device_gpu_hours: float | None = None
    measured_target_effect: float | None = None
    parent_identity: tuple[str, str] | None = None
    profile: Mapping[str, Any] | None = None
    #: Already-classified failure records, built by the executor from the
    #: campaign's own evidence. The loop persists them verbatim rather than
    #: rebuilding them from a lossy dict.
    failures: tuple[Any, ...] = ()
    regressions: tuple[str, ...] = ()
    run_root: str = ""
    reason: str = ""

    @property
    def promoted(self) -> bool:
        return str(self.verdict).upper() == "PROMOTED"

    @property
    def executed(self) -> bool:
        """Whether compute actually ran. A refusal before compute is not spend."""
        return self.verdict.upper() not in {"REFUSED", "NOT_RUN"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "wall_gpu_hours": self.wall_gpu_hours,
            "device_gpu_hours": self.device_gpu_hours,
            "measured_target_effect": self.measured_target_effect,
            "parent_identity": list(self.parent_identity) if self.parent_identity else None,
            "regressions": list(self.regressions),
            "run_root": self.run_root,
            "reason": self.reason,
        }


@dataclass
class LoopBudget:
    """The session envelope, decremented by measured spend.

    A campaign budget bounds one generation; this bounds the session, and the
    two are different controls. The loop refuses to start a generation whose
    *declared* campaign envelope no longer fits in what is left, so a loop
    cannot spend its way past its own limit one campaign at a time.
    """

    maximum_total_wall_gpu_hours: float
    spent_wall_gpu_hours: float = 0.0

    @property
    def remaining_wall_gpu_hours(self) -> float:
        return float(self.maximum_total_wall_gpu_hours) - float(self.spent_wall_gpu_hours)

    def charge(self, outcome: CampaignOutcome) -> None:
        """Add a campaign's *measured* spend. ``None`` is refused, not zeroed."""
        if outcome.wall_gpu_hours is None:
            raise LoopBudgetError(
                f"{CAMPAIGN_SPENT_NO_MEASURED_COST}: a campaign reporting no "
                "measured wall cost cannot be charged, and charging it zero would "
                "make unaccounted compute free"
            )
        self.spent_wall_gpu_hours += float(outcome.wall_gpu_hours)

    def restore(self, *, spent_wall_gpu_hours: float) -> None:
        """Adopt the spend a session already recorded durably.

        Only ever called with a figure summed from durable campaign records, so
        the envelope of a resumed session reflects measured spend rather than a
        fresh zero -- a resumed loop that forgot what it had spent would happily
        spend the whole budget twice.
        """
        spent = float(spent_wall_gpu_hours)
        if spent < 0:
            raise LoopBudgetError("a session cannot have spent negative GPU-hours")
        self.spent_wall_gpu_hours = spent

    def admits(self, *, declared_campaign_wall_gpu_hours: float) -> tuple[bool, str]:
        needed = float(declared_campaign_wall_gpu_hours)
        if needed > self.remaining_wall_gpu_hours + 1e-12:
            return False, (
                f"{REMAINING_ENVELOPE_TOO_SMALL}: {self.remaining_wall_gpu_hours:.6f} "
                f"wall GPU-hours remain and this campaign declares {needed:.6f}; no "
                "training is launched"
            )
        return True, ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximum_total_wall_gpu_hours": self.maximum_total_wall_gpu_hours,
            "spent_wall_gpu_hours": self.spent_wall_gpu_hours,
            "remaining_wall_gpu_hours": self.remaining_wall_gpu_hours,
        }


class LoopBudgetError(RuntimeError):
    """An accounting fact the loop refuses to invent."""


@dataclass(frozen=True)
class PlateauCheck:
    """Whether a target has stopped being worth another attempt."""

    plateaued: bool
    reason_code: str
    reason: str
    tried_treatments: tuple[str, ...]
    untried_treatments: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plateaued": self.plateaued,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "tried_treatments": list(self.tried_treatments),
            "untried_treatments": list(self.untried_treatments),
        }


def detect_plateau(
    *,
    state: GrowthState,
    target_skill: str,
    maximum_same_target_attempts: int,
    epsilon: float,
    allowed_treatments: Sequence[str],
) -> PlateauCheck:
    """Has this target stopped responding, and is there anything else to try?

    A target is not a plateau merely because it failed: it is a plateau when it
    has been attempted to the policy's limit, none of those attempts moved it by
    more than ``epsilon``, and every treatment class the policy allows has
    already been tried. Until then the honest next move is a *different*
    treatment, which is why the untried classes are reported rather than a bare
    boolean.
    """
    rows = state.interventions(target_skill=target_skill)
    tried = tuple(dict.fromkeys(str(row.get("training_type", "")) for row in rows))
    allowed = tuple(dict.fromkeys(str(value) for value in allowed_treatments))
    untried = tuple(value for value in allowed if value not in tried)
    if len(rows) < int(maximum_same_target_attempts):
        return PlateauCheck(
            False,
            "",
            f"{len(rows)} attempt(s) so far, the policy allows "
            f"{maximum_same_target_attempts} before this target is exhausted",
            tried,
            untried,
        )
    effects = [
        float(row["measured_effect"])
        for row in rows
        if isinstance(row.get("measured_effect"), (int, float))
    ]
    if not effects:
        return PlateauCheck(
            False,
            "",
            "no attempt on this target reported a measured effect, so whether it "
            "moved is unknown rather than zero",
            tried,
            untried,
        )
    if max(abs(effect) for effect in effects) > float(epsilon):
        return PlateauCheck(
            False,
            "",
            f"an attempt moved the target by {max(abs(effect) for effect in effects):.6f}, "
            f"above the {epsilon} epsilon",
            tried,
            untried,
        )
    if untried:
        return PlateauCheck(
            False,
            "",
            f"this target has not responded to {list(tried)}, and {list(untried)} "
            "remain untried",
            tried,
            untried,
        )
    return PlateauCheck(
        True,
        TARGET_EXHAUSTED,
        f"this target was attempted {len(rows)} time(s) with {list(tried)}, no "
        f"attempt moved it by more than {epsilon}, and no allowed treatment "
        "remains untried",
        tried,
        untried,
    )


@dataclass(frozen=True)
class GenerationRecord:
    """One iteration of the loop, as it will be reported and persisted."""

    index: int
    cycle_id: str
    candidate_version: str
    target_skill: str
    treatment: str
    verdict: str
    wall_gpu_hours: float | None
    measured_target_effect: float | None
    preregistration_digest: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "cycle_id": self.cycle_id,
            "candidate_version": self.candidate_version,
            "target_skill": self.target_skill,
            "treatment": self.treatment,
            "verdict": self.verdict,
            "wall_gpu_hours": self.wall_gpu_hours,
            "measured_target_effect": self.measured_target_effect,
            "preregistration_digest": self.preregistration_digest,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class LoopRunReport:
    """The whole session: what it decided and what it actually ran."""

    decision: LoopDecision
    generations: tuple[GenerationRecord, ...]
    budget: Mapping[str, Any]
    parent_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "generations": [record.to_dict() for record in self.generations],
            "budget": dict(self.budget),
            "parent_version": self.parent_version,
        }


class GrowthLoop:
    """Advance one generation at a time while the policy and the budget allow."""

    def __init__(
        self,
        *,
        policy: LoopPolicy,
        state: GrowthState,
        executor: Callable[[FrozenCampaign], CampaignOutcome] | None = None,
        parent_declaration: CampaignManifest,
        parent_profile: SkillProfile | Mapping[str, Any] | None = None,
        parent_identity: tuple[str, str] | None = None,
        selector: NextTargetSelector | None = None,
        builder: NextCampaignBuilder | None = None,
        prepare: Callable[[FrozenCampaign], Any] | None = None,
        readiness: Callable[[FrozenCampaign], Any] | None = None,
    ) -> None:
        self.policy = policy
        self.state = state
        # The default seam is the real campaign runner. A loop that had to be
        # handed an executor could be run against something that trains nothing
        # while still reporting a verdict, so "no executor" means "the real one",
        # never "skip the campaign".
        self.executor = executor or production_executor()
        self.parent_declaration = parent_declaration
        self.parent_identity = parent_identity
        self.selector = selector or NextTargetSelector(
            max_same_target_attempts=policy.maximum_same_target_attempts,
            structural_skills=policy.structural_skills,
        )
        self.builder = builder or NextCampaignBuilder(policy=policy)
        self._prepare = prepare or _production_prepare
        self._readiness = readiness or _production_readiness
        self._profile = _coerce_profile(parent_profile)
        self.budget = LoopBudget(
            maximum_total_wall_gpu_hours=policy.maximum_total_wall_gpu_hours
        )

    # -- the loop ----------------------------------------------------------

    def run(
        self,
        *,
        max_generations: int | None = None,
        resume: bool = False,
    ) -> LoopRunReport:
        """Run until a terminal decision. Never an unbounded loop.

        ``resume`` reads the session's durable record before it spends anything.
        A session the record says has already ended returns its stored decision
        and launches nothing, and a session that ended in a promotion restores
        the promoted declaration *and* adapter, so the next generation trains
        from the artifact the run actually produced rather than from the one this
        process happened to be holding. A promotion the record does not name is
        refused rather than guessed: an adapter the loop cannot identify is an
        adapter it must not train from.
        """
        limit = min(
            int(max_generations or self.policy.maximum_generations),
            int(self.policy.maximum_generations),
        )
        records: list[GenerationRecord] = []
        if resume:
            restored = self._restore_from_durable_state()
            if isinstance(restored, LoopDecision):
                self.state.set_stopping_state(restored.to_dict())
                return self._report(restored, [])
            records = list(restored)
            settled = self.state.stopping_state()
            if settled.get("terminal"):
                return self._report(
                    LoopDecision(
                        str(settled.get("action", STOP_UNCERTAIN)),
                        "resumed: the durable record already ended this session -- "
                        f"{settled.get('reason', '')}",
                        tuple(str(code) for code in (settled.get("reason_codes") or ())),
                    ),
                    records,
                )
        non_promotions = 0
        exhausted: set[str] = set()
        decision = LoopDecision(CONTINUE, "the loop has not attempted a generation yet")

        for index in range(1, limit + 1):
            step = self._one_generation(
                index=index,
                exhausted=exhausted,
            )
            if step.record is not None:
                records.append(step.record)
            if not step.ran_compute:
                self.state.set_stopping_state(step.decision.to_dict())
                return self._report(step.decision, records)

            outcome = step.outcome
            assert outcome is not None
            # 1. spend first: the envelope must reflect measured cost before any
            #    judgement about continuing is made. An absent figure is a
            #    terminal refusal *here*, before charging: charging it would be
            #    dishonest, and letting ``charge`` raise would end the session
            #    with an exception instead of a durable decision.
            if outcome.wall_gpu_hours is None:
                decision = LoopDecision(
                    STOP_UNCERTAIN,
                    f"{step.frozen.cycle_id} ran but reported no measured wall "
                    "cost, so the session's spend is not accounted; an "
                    "unmeasured cost is not a discount",
                    (CAMPAIGN_SPENT_NO_MEASURED_COST,),
                )
                self.state.set_stopping_state(decision.to_dict())
                return self._report(decision, records)
            self.budget.charge(outcome)
            # 2. learn, then advance the parent pointer only on a real promotion.
            learned = self._record_learning(step.frozen, outcome)
            if outcome.promoted:
                if outcome.parent_identity is None:
                    decision = LoopDecision(
                        STOP_UNCERTAIN,
                        "the campaign promoted and named no promoted identity, so the "
                        "loop cannot say what the next generation would train from",
                        (PROMOTION_WITHOUT_IDENTITY,),
                    )
                    self.state.set_stopping_state(decision.to_dict())
                    return self._report(decision, records)
                if learned is None:
                    # Not ``self._profile is None``: the loop always holds *some*
                    # profile by now, so that test could never fail and would be a
                    # refusal that looks enforced and is not. What matters is
                    # whether *this* promotion reported a measurement of the model
                    # it promoted -- otherwise the next target would be chosen
                    # from the parent's stale evidence and the generation that
                    # just advanced the lineage would have measured nothing.
                    decision = LoopDecision(
                        STOP_UNCERTAIN,
                        "the campaign promoted without reporting a measured "
                        "capability profile, so the next target would be chosen from "
                        "stale parent evidence rather than from what was promoted",
                        (PROMOTION_WITHOUT_PROFILE,),
                    )
                    self.state.set_stopping_state(decision.to_dict())
                    return self._report(decision, records)
                self.parent_declaration = step.frozen.manifest
                self.parent_identity = outcome.parent_identity
                non_promotions = 0
            else:
                non_promotions += 1

            decision = self._decide_after(
                index=index,
                limit=limit,
                outcome=outcome,
                frozen=step.frozen,
                non_promotions=non_promotions,
                exhausted=exhausted,
            )
            if decision.terminal:
                self.state.set_stopping_state(decision.to_dict())
                return self._report(decision, records)

        # The loop cannot fall out of the for-loop without a decision; the
        # generation limit is one, and it is decided above.
        if decision.action == CONTINUE:
            decision = LoopDecision(
                STOP_SUCCESS if records and records[-1].verdict.upper() == "PROMOTED" else STOP_PLATEAU,
                "the generation limit was reached",
                (GENERATION_LIMIT_REACHED,),
            )
        self.state.set_stopping_state(decision.to_dict())
        return self._report(decision, records)

    # -- one iteration -----------------------------------------------------

    def plan_next(self, *, exhausted: Iterable[str] = ()) -> Any:
        """What the loop would attempt next, decided by the loop's own gates.

        Returns the :class:`TargetProposal` it would build a campaign for, or the
        :class:`LoopDecision` it would stop on instead. Read-only: no declaration
        is frozen, no target is recorded, nothing is spent.

        This exists so a dry run cannot advertise something a run would refuse.
        The gates live in ``_proposal_or_decision`` and both callers go through
        them, rather than a plan re-deriving the selector call and quietly
        skipping the profile, treatment and envelope checks the run applies.
        """
        proposal, decision = self._proposal_or_decision(set(exhausted))
        return decision if decision is not None else proposal

    def _proposal_or_decision(
        self, exhausted: set[str]
    ) -> tuple[Any | None, LoopDecision | None]:
        """The gates that stand between the loop and spending, in one place.

        Returns either the proposal it may build a campaign for, or the decision
        it must take instead -- never both, and never neither.
        """
        if self._profile is None or not self._profile.measured:
            return None, LoopDecision(
                STOP_UNCERTAIN,
                "the parent has no measured capability profile, so no target can "
                "be chosen from evidence",
                (NO_MEASURED_CAPABILITY,),
            )
        # The profile must be a measurement of the generation it is used as the
        # parent *of*. It is handed in from outside (a run root, or a stated
        # file), so the loop is the only place that can pair it with the
        # declaration it will be attributed to -- and a loop given one
        # generation's arm while declaring another as the parent would choose
        # the next target from evidence about a different model, then record the
        # choice against this one. A mismatch is refused, never relabelled: the
        # two are independent claims and neither can be inferred from the other.
        declared = str(self.parent_declaration.resolved_candidate_version() or "")
        measured = str(getattr(self._profile, "generation", "") or "")
        if declared and measured != declared:
            return None, LoopDecision(
                STOP_UNCERTAIN,
                f"the capability profile measures {measured!r} and the parent "
                f"declaration's candidate is {declared!r}; planning the next "
                "target from it would choose from evidence about a different "
                "model while declaring this one as the parent",
                (PROFILE_GENERATION_MISMATCH,),
            )
        try:
            proposal = self.selector.propose(
                parent_version=self.parent_declaration.resolved_candidate_version(),
                profile=self._profile,
                state=self.state,
                category_counts=self._category_counts(),
                known_skills=self._known_skills(),
                exclude_skills=exhausted,
            )
        except ValueError as error:
            return None, LoopDecision(
                STOP_SUCCESS,
                f"no target this loop is allowed to improve remains: {error}",
                (NOTHING_LEFT_TO_IMPROVE,),
            )

        if str(proposal.suggested_training_type) in REVIEW_TREATMENTS:
            return None, LoopDecision(
                REQUIRES_HUMAN_REVIEW,
                f"the target {proposal.target_skill!r} needs "
                f"{proposal.suggested_training_type!r}: {proposal.treatment_reason}",
                (TREATMENT_REQUIRES_REVIEW,),
            )
        if not self.policy.allows_treatment(proposal.suggested_training_type):
            return None, LoopDecision(
                REQUIRES_HUMAN_REVIEW,
                f"the target {proposal.target_skill!r} wants "
                f"{proposal.suggested_training_type!r}, which this policy does "
                "not allow; widening the policy is a human decision",
                (TREATMENT_REQUIRES_REVIEW,),
            )

        # Admission *before* building or spending: the loop must be able to
        # accommodate the campaign it is about to declare.
        affordable, reason = self.budget.admits(
            declared_campaign_wall_gpu_hours=float(
                self.policy.campaign_budget.wall_gpu_hours_ceiling_campaign
            )
        )
        if not affordable:
            return None, LoopDecision(
                STOP_BUDGET, reason, (REMAINING_ENVELOPE_TOO_SMALL,)
            )
        return proposal, None

    def _one_generation(self, *, index: int, exhausted: set[str]) -> "_Step":
        proposal, refusal = self._proposal_or_decision(exhausted)
        if refusal is not None:
            return _Step(refusal)

        # Generations live beside the parent's run root. The *name* of this
        # attempt belongs to the builder, which is the only owner of the cycle
        # identity: an attempt the loop named separately could collide with a
        # frozen declaration that is still valid.
        generation_root = Path(self.parent_declaration.state_root).parent
        try:
            frozen = self.builder.build(
                parent=self.parent_declaration,
                target=proposal,
                generation_root=generation_root,
                attempt=self.state.attempts_on(proposal.target_skill) + 1,
                parent_identity=self.parent_identity,
            )
        except NextCampaignRefusal as error:
            # A policy-conformant declaration that cannot be composed is a
            # reviewable event, not a silent skip.
            return _Step(
                LoopDecision(
                    REQUIRES_HUMAN_REVIEW,
                    f"the next campaign could not be composed: {error}",
                    (TREATMENT_REQUIRES_REVIEW,),
                )
            )
        self.state.record_target(proposal)

        prepared, prepare_reason = self._run_prepare(frozen)
        if not prepared:
            return _Step(
                LoopDecision(STOP_UNCERTAIN, prepare_reason, (PREPARATION_REFUSED,)),
                frozen=frozen,
            )
        ready, readiness_reason = self._run_readiness(frozen)
        if not ready:
            return _Step(
                LoopDecision(STOP_UNCERTAIN, readiness_reason, (READINESS_REFUSED,)),
                frozen=frozen,
            )

        outcome = self.executor(frozen)
        if not outcome.executed:
            return _Step(
                LoopDecision(
                    STOP_UNCERTAIN,
                    f"the campaign refused before compute: {outcome.reason or outcome.verdict}",
                    (PREPARATION_REFUSED,),
                ),
                frozen=frozen,
                outcome=outcome,
            )
        record = GenerationRecord(
            index=index,
            cycle_id=frozen.cycle_id,
            candidate_version=frozen.candidate_version,
            target_skill=proposal.target_skill,
            treatment=str(proposal.suggested_training_type),
            verdict=str(outcome.verdict),
            wall_gpu_hours=outcome.wall_gpu_hours,
            measured_target_effect=outcome.measured_target_effect,
            preregistration_digest=frozen.digest,
            reason=str(outcome.reason),
        )
        return _Step(
            LoopDecision(CONTINUE, "the campaign ran; the loop decides from its outcome"),
            frozen=frozen,
            outcome=outcome,
            record=record,
            ran_compute=True,
        )

    # -- decisions ---------------------------------------------------------

    def _decide_after(
        self,
        *,
        index: int,
        limit: int,
        outcome: CampaignOutcome,
        frozen: FrozenCampaign,
        non_promotions: int,
        exhausted: set[str],
    ) -> LoopDecision:
        # No unmeasured-cost check here: ``run`` refuses that case before it
        # charges, so by the time a decision is being made the spend is a
        # measured figure. Keeping a second copy of the rule here would be a
        # second owner for it -- and it would be the unreachable one.
        plateau = detect_plateau(
            state=self.state,
            target_skill=frozen.target.target_skill,
            maximum_same_target_attempts=self.policy.maximum_same_target_attempts,
            epsilon=float(self.policy.plateau_epsilon),
            allowed_treatments=self.policy.allowed_training_types,
        )
        if plateau.plateaued:
            exhausted.add(frozen.target.target_skill)
            if self._another_target_exists(exhausted):
                return LoopDecision(
                    CONTINUE,
                    f"{plateau.reason}; another measured target remains, so the loop "
                    "switches rather than repeating it",
                    (TARGET_EXHAUSTED,),
                )
            return LoopDecision(STOP_PLATEAU, plateau.reason, (TARGET_EXHAUSTED,))
        if non_promotions >= int(self.policy.maximum_consecutive_non_promotions):
            return LoopDecision(
                STOP_PLATEAU,
                f"{non_promotions} consecutive generations failed to promote "
                f"(policy allows {self.policy.maximum_consecutive_non_promotions})",
                (CONSECUTIVE_NON_PROMOTIONS,),
            )
        if index >= limit:
            return LoopDecision(
                STOP_SUCCESS if outcome.promoted else STOP_PLATEAU,
                f"the generation limit ({limit}) was reached",
                (GENERATION_LIMIT_REACHED,),
            )
        return LoopDecision(
            CONTINUE,
            f"generation {index} is complete; the envelope allows another",
        )

    def _another_target_exists(self, exhausted: set[str]) -> bool:
        if self._profile is None:
            return False
        try:
            self.selector.propose(
                parent_version=self.parent_declaration.resolved_candidate_version(),
                profile=self._profile,
                state=self.state,
                known_skills=self._known_skills(),
                exclude_skills=exhausted,
            )
        except ValueError:
            return False
        return True

    # -- seams and state ---------------------------------------------------

    def _run_prepare(self, frozen: FrozenCampaign) -> tuple[bool, str]:
        try:
            self._prepare(frozen)
        except Exception as error:  # noqa: BLE001 - a refusal is a refusal
            return False, f"preparation refused for {frozen.cycle_id}: {error}"
        return True, ""

    def _run_readiness(self, frozen: FrozenCampaign) -> tuple[bool, str]:
        try:
            report = self._readiness(frozen)
        except Exception as error:  # noqa: BLE001
            return False, f"readiness could not be evaluated for {frozen.cycle_id}: {error}"
        ready = bool(getattr(report, "ready", report))
        if ready:
            return True, ""
        codes = tuple(getattr(report, "reason_codes", ()) or ())
        return False, (
            f"{frozen.cycle_id} is not READY ({list(codes) or 'no reason codes'}); "
            "no training is launched"
        )

    def _record_learning(
        self, frozen: FrozenCampaign, outcome: CampaignOutcome
    ) -> SkillProfile | None:
        """Write what this generation taught, durably, before deciding.

        Returns the profile this generation actually produced, or ``None`` when it
        reported none. Returning it (rather than re-reading ``self._profile``)
        keeps one owner for "what did this promotion measure", so the promotion
        gate cannot check a profile the parent contributed.
        """
        self.state.record_intervention(
            target_skill=frozen.target.target_skill,
            training_type=str(frozen.target.suggested_training_type),
            cycle_id=frozen.cycle_id,
            generation=frozen.candidate_version,
            # Strict, not ``or 0.0``: the loop refuses an unmeasured cost before
            # it gets here, and a silent zero would be exactly the "unmeasured
            # became zero" defect this record exists to prevent.
            cost_gpu_hours=float(outcome.wall_gpu_hours),
            candidate_result=str(outcome.verdict),
            promotion_result="promoted" if outcome.promoted else "not_promoted",
            promoted_identity=outcome.parent_identity if outcome.promoted else None,
            measured_effect=outcome.measured_target_effect,
            regressions=tuple(outcome.regressions),
        )
        if outcome.failures:
            self.state.persist_failures(outcome.failures)
        profile = _coerce_profile(outcome.profile)
        if profile is not None:
            self._profile = profile
            self.state.record_capability(profile)
        # Regressions are recorded with the intervention that caused them (in
        # record_intervention above), because a regression detached from the
        # treatment that produced it cannot tell the selector what to avoid.
        return profile

    def _restore_from_durable_state(self) -> list[GenerationRecord] | LoopDecision:
        """Rebuild the session's history from what it wrote, or refuse to.

        Everything here is read, never estimated: spend is the sum of the
        campaigns' recorded costs, the parent pointer is the last recorded
        promotion, and a promotion the record cannot name is a refusal. """
        generation_root = Path(self.parent_declaration.state_root).parent
        rows = self.state.interventions()
        self.budget.restore(
            spent_wall_gpu_hours=sum(
                float(row.get("cost_gpu_hours", 0.0) or 0.0) for row in rows
            )
        )
        records: list[GenerationRecord] = []
        for index, row in enumerate(rows, start=1):
            cycle_id = str(row.get("cycle_id", ""))
            if str(row.get("promotion_result", "")) == "promoted":
                identity = row.get("promoted_identity")
                if (
                    not isinstance(identity, (list, tuple))
                    or len(identity) != 2
                    or len(str(identity[1])) != 64
                ):
                    return LoopDecision(
                        STOP_UNCERTAIN,
                        f"cycle {cycle_id!r} promoted and the durable record does "
                        "not name the adapter it promoted, so resuming would train "
                        "the next generation from a parent the loop cannot identify",
                        (PROMOTION_WITHOUT_IDENTITY,),
                    )
                self.parent_identity = (str(identity[0]), str(identity[1]))
                declaration = _frozen_declaration(generation_root, cycle_id)
                if declaration is None:
                    return LoopDecision(
                        STOP_UNCERTAIN,
                        f"cycle {cycle_id!r} promoted but its frozen declaration is "
                        "absent, so the next generation could not carry its budget, "
                        "protection or trusted ancestor forward",
                        (PROMOTION_WITHOUT_DECLARATION,),
                    )
                self.parent_declaration = declaration
            records.append(
                GenerationRecord(
                    index=index,
                    cycle_id=cycle_id,
                    candidate_version=str(row.get("generation", "")),
                    target_skill=str(row.get("target_skill", "")),
                    treatment=str(row.get("training_type", "")),
                    verdict=str(row.get("candidate_result", "")),
                    wall_gpu_hours=float(row.get("cost_gpu_hours", 0.0) or 0.0),
                    measured_target_effect=row.get("measured_effect"),
                    preregistration_digest=_frozen_digest(generation_root, cycle_id),
                    reason="recovered from durable state on resume",
                )
            )
        return records

    def _category_counts(self) -> Mapping[str, int]:
        """Banked-failure counts, attributed to skills by *measured* support.

        A failure is counted against a skill only when the parent's own profile
        shows a benchmark supporting that skill which the failure was observed
        on. A failure nothing attributes is left out rather than charged to a
        skill the loop guessed at.
        """
        if self._profile is None:
            return {}
        owners: dict[str, tuple[str, ...]] = {
            estimate.skill: tuple(estimate.supporting_benchmarks)
            for estimate in self._profile.estimates
        }
        counts: dict[str, int] = {}
        for record in self.state.failure_bank():
            benchmark = str(getattr(record, "benchmark_qualified_id", "") or "")
            for skill, benchmarks in owners.items():
                if benchmark and benchmark in benchmarks:
                    counts[skill] = counts.get(skill, 0) + 1
        return counts

    def _known_skills(self) -> tuple[str, ...]:
        known = [estimate.skill for estimate in (self._profile.estimates if self._profile else ())]
        known.extend(
            str(row.get("target_skill", ""))
            for row in self.state.interventions()
            if row.get("target_skill")
        )
        return tuple(dict.fromkeys(value for value in known if value))

    def _report(
        self, decision: LoopDecision, records: list[GenerationRecord]
    ) -> LoopRunReport:
        return LoopRunReport(
            decision=decision,
            generations=tuple(records),
            budget=self.budget.to_dict(),
            parent_version=self.parent_declaration.resolved_candidate_version(),
        )


@dataclass
class _Step:
    """One iteration's outcome: what was decided, and what it ran."""

    decision: LoopDecision
    frozen: FrozenCampaign | None = None
    outcome: CampaignOutcome | None = None
    record: GenerationRecord | None = None
    ran_compute: bool = False


def _frozen_declaration(generation_root: Path, cycle_id: str) -> CampaignManifest | None:
    """The declaration a resumed generation was actually composed under."""
    path = Path(generation_root) / cycle_id / "campaign.json"
    if not path.is_file():
        return None
    try:
        return CampaignManifest.from_mapping(
            json.loads(path.read_text(encoding="utf-8")), source=f"resume:{cycle_id}"
        )
    except Exception:  # noqa: BLE001 - an unreadable declaration is an absent one
        return None


def _frozen_digest(generation_root: Path, cycle_id: str) -> str:
    """The frozen preregistration digest, or ``""`` when it cannot be read.

    Empty is honest: a resumed record may not invent a digest it cannot recompute
    from the bytes that were frozen.
    """
    path = Path(generation_root) / cycle_id / "preregistration.json"
    if not path.is_file():
        return ""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return ""
    digest = document.get("frozen_digest")
    return str(digest) if isinstance(digest, str) else ""


def _coerce_profile(value: SkillProfile | Mapping[str, Any] | None) -> SkillProfile | None:
    if value is None:
        return None
    if isinstance(value, SkillProfile):
        return value
    try:
        return SkillProfile.from_dict(dict(value))
    except Exception:  # noqa: BLE001 - an unreadable profile is no profile
        return None


def _production_prepare(frozen: FrozenCampaign) -> Any:
    """The real preparation step: the declaration's own inputs, from evidence."""
    from .campaign_prepare import prepare_campaign

    parent_evidence = Path(frozen.manifest.state_root).parent.parent
    return prepare_campaign(
        frozen.manifest,
        out_dir=frozen.directory,
        parent_evidence=parent_evidence,
    )


def _production_readiness(frozen: FrozenCampaign) -> Any:
    from .campaign_runner import check_campaign_readiness

    return check_campaign_readiness(frozen.manifest)


def production_executor() -> Callable[[FrozenCampaign], CampaignOutcome]:
    """The real executor seam: run the declared campaign, report what it measured.

    Lazily imports the campaign runner so the control plane (and the simulation)
    can be exercised without loading the training and evaluation stack.
    """

    def execute(frozen: FrozenCampaign) -> CampaignOutcome:
        from .campaign_runner import run_campaign

        return campaign_outcome(run_campaign(frozen.manifest))

    return execute


def campaign_outcome(run: Any) -> CampaignOutcome:  # noqa: ANN401 - a CampaignRun
    """A completed campaign's durable record, read as the loop consumes it.

    Every field is *read* from the run. An absent figure stays ``None`` so the
    loop refuses it rather than charging it zero, and the promoted identity is
    taken from the artifact the run selected -- including its digest -- so the
    next generation trains from bytes this run actually produced.
    """
    cost = dict(getattr(run, "cost", {}) or {})
    verdict = str(getattr(run, "verdict", ""))
    identity: tuple[str, str] | None = None
    if verdict.upper() == "PROMOTED":
        selection = dict(getattr(run, "selection", {}) or {})
        path = str(selection.get("artifact_ref") or "")
        digest = str(selection.get("artifact_sha256") or "")
        if path.strip() and len(digest) == 64:
            identity = (path, digest)

    root = _run_root(run)
    profile = profile_from_run_root(root) if root is not None else None
    promotion = dict(getattr(run, "promotion", {}) or {})
    deltas = {
        str(benchmark): float(delta)
        for benchmark, delta in dict(promotion.get("target_deltas", {}) or {}).items()
        if _finite(delta)
    }
    protected = dict(promotion.get("protected_deltas", {}) or {})
    return CampaignOutcome(
        verdict=verdict,
        wall_gpu_hours=_measured(cost.get("wall_gpu_hours")),
        device_gpu_hours=_measured(cost.get("device_gpu_hours")),
        # What the loop remembers as the target's movement is the mean of the
        # per-benchmark deltas the *promotion engine* computed from the measured
        # arms. No delta means no effect, which is None rather than 0.0 -- an
        # unmoved target and an unmeasured one are different facts.
        measured_target_effect=_mean(deltas.values()) if deltas else None,
        parent_identity=identity,
        profile=profile.to_dict() if profile is not None else None,
        failures=(),
        # Measured negative movement on a protected capability, named so the
        # selector can avoid the treatment that caused it. The tolerance is the
        # promotion engine's business; this is the observation, not the verdict.
        regressions=tuple(
            sorted(
                str(benchmark)
                for benchmark, delta in protected.items()
                if _finite(delta) and float(delta) < 0.0
            )
        ),
        run_root=str(root or ""),
        reason=_reason_of(run),
    )


def profile_from_run_root(root: str | Path) -> SkillProfile | None:
    """The measured capability of the model a run evaluated, from its own arm.

    Built from the run's ``candidate_evaluation.json`` -- the rows the run
    actually measured on the artifact it selected -- so a promoted generation's
    profile is attributable evidence rather than a summary of its parent's.
    ``None`` means the run produced no readable measured arm, and the loop stops
    on that rather than planning the next target from stale evidence.
    """
    from chowder.evals.result import EvalReport

    path = Path(root) / "candidate_evaluation.json"
    if not path.is_file():
        return None
    try:
        report = EvalReport.load(path)
        runs = tuple(run for run in report.runs if _finite(run.score))
        if not runs:
            return None
        return build_skill_profile(
            generation=str(report.generation_version or "unknown"), runs=runs
        )
    except Exception:  # noqa: BLE001 - unreadable evidence is absent evidence
        return None


def _run_root(run: Any) -> Path | None:  # noqa: ANN401 - a CampaignRun
    record = str(getattr(run, "record_path", "") or "")
    if not record.strip():
        return None
    path = Path(record)
    return path.parent if path.is_file() else path


def _reason_of(run: Any) -> str:  # noqa: ANN401 - a CampaignRun
    parts: list[str] = []
    certification = dict(getattr(run, "certification", {}) or {})
    parts.extend(str(reason) for reason in (certification.get("reasons") or ()))
    settlement = dict(getattr(run, "settlement", {}) or {})
    parts.extend(str(reason) for reason in (settlement.get("failure_reasons") or ()))
    return "; ".join(parts) or str(getattr(run, "verdict", ""))


def _measured(value: Any) -> float | None:  # noqa: ANN401 - a document field
    """A measured figure, or ``None``. Unmeasured is never reported as zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _finite(value: Any) -> bool:  # noqa: ANN401 - a document field
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _mean(values: Any) -> float | None:  # noqa: ANN401 - an iterable of floats
    numbers = [float(value) for value in values]
    return sum(numbers) / len(numbers) if numbers else None


def load_profile_from(path: str | Path) -> SkillProfile | None:
    """Read a parent capability profile from the declaration that names it."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return SkillProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, TypeError, KeyError):
        return None
