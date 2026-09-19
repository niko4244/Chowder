"""One autonomous-growth service, shared by every client that has one.

The control plane already owns the decisions: ``GrowthLoop`` selects targets,
composes and freezes declarations, gates on readiness, charges measured spend and
decides whether to continue. What it did not have was a *front door*, so each
client that wanted to drive it rebuilt the same wiring -- resolve a policy, load
the parent declaration, find or derive the parent's measured profile, open
durable state, construct the loop -- and every rebuild was a place the clients
could drift apart. ``chowder growth loop plan`` and ``chowder growth loop run``
already asked the loop itself so they could not disagree about the decision;
this module extends that to *all* of it, including the lineage and the history.

The service owns exactly two things the loop does not:

* **wiring.** Turning paths on disk into the objects the loop needs, refusing
  when one of them is missing rather than defaulting to something plausible;
* **read-only views.** ``inspect``, ``plan_next``, ``prepare_next``, ``status``
  and ``history`` project durable state and the loop's own decisions into shapes
  a CLI can print and a terminal UI can render, without either of them
  recomputing a decision.

Nothing here decides anything. Every method either delegates to the loop or
formats what the loop and the durable record already say.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .campaign import CampaignManifest
from .growth_loop import (
    TERMINAL_DECISIONS,
    CampaignOutcome,
    FrozenCampaign,
    GrowthLoop,
    LoopDecision,
    LoopRunReport,
    PreparedAttempt,
    load_profile_from,
    profile_from_run_root,
)
from .next_campaign import LoopPolicy, ParentEvidenceRef
from .target_selection import GrowthState, SkillProfile

#: Why the service will not open a session. Named, so a client can act on it.
SERVICE_POLICY_ABSENT = "SERVICE_POLICY_ABSENT"
SERVICE_PARENT_ABSENT = "SERVICE_PARENT_ABSENT"
#: A session that was asked to *plan* has no measured parent capability, so no
#: target can be chosen from evidence. Refused before any durable state exists,
#: because a refusal must not leave a session behind it.
SERVICE_PROFILE_ABSENT = "SERVICE_PROFILE_ABSENT"


class GrowthServiceRefusal(RuntimeError):
    """A session that cannot be opened, with the missing thing named."""


@dataclass(frozen=True)
class LineageView:
    """Which model the loop stands on, and what evidence says so."""

    generation: str
    base_model_path: str
    base_model_digest: str
    adapter_path: str
    adapter_digest: str
    run_root: str
    measured_arm_path: str
    trusted_ancestor: str
    trusted_ancestor_arm: str
    declared_protected: tuple[str, ...]
    protected_skills: tuple[str, ...]
    profile_generation: str
    profile_measured_skills: tuple[str, ...]
    policy_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "base_model_path": self.base_model_path,
            "base_model_digest": self.base_model_digest,
            "adapter_path": self.adapter_path,
            "adapter_digest": self.adapter_digest,
            "adapter_digest_prefix": self.adapter_digest[:12],
            "run_root": self.run_root,
            "measured_arm_path": self.measured_arm_path,
            "trusted_ancestor": self.trusted_ancestor,
            "trusted_ancestor_arm": self.trusted_ancestor_arm,
            "declared_protected": list(self.declared_protected),
            "protected_skills": list(self.protected_skills),
            "profile_generation": self.profile_generation,
            "profile_measured_skills": list(self.profile_measured_skills),
            "policy_digest": self.policy_digest,
        }


@dataclass(frozen=True)
class PlanningView:
    """What the loop would attempt next -- or the decision not to attempt it."""

    parent_version: str
    proposal: Mapping[str, Any] | None = None
    decision: Mapping[str, Any] | None = None

    @property
    def decided(self) -> bool:
        return self.decision is not None

    @property
    def action(self) -> str:
        return str((self.decision or {}).get("action", "PLAN"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_version": self.parent_version,
            "proposal": dict(self.proposal) if self.proposal else None,
            "decision": dict(self.decision) if self.decision else None,
        }


@dataclass(frozen=True)
class ReadinessView:
    """One readiness check, which is never collapsed into a boolean."""

    name: str
    status: str
    detail: str = ""
    reason_code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class PreparationView:
    """A planned, prepared and frozen attempt, plus its readiness verdict."""

    attempt: PreparedAttempt | None = None
    checks: tuple[ReadinessView, ...] = ()
    decision: Mapping[str, Any] | None = None

    @property
    def ready(self) -> bool:
        return bool(self.checks) and all(c.status == "ok" for c in self.checks)

    @property
    def refused(self) -> bool:
        return any(check.status == "refused" for check in self.checks)

    @property
    def reason_codes(self) -> tuple[str, ...]:
        if self.decision is not None:
            return tuple(str(code) for code in (self.decision.get("reason_codes") or ()))
        return tuple(
            check.reason_code
            for check in self.checks
            if check.status == "refused" and check.reason_code
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt.to_dict() if self.attempt else None,
            "ready": self.ready,
            "refused": self.refused,
            "checks": [check.to_dict() for check in self.checks],
            "decision": dict(self.decision) if self.decision else None,
        }


@dataclass(frozen=True)
class StatusView:
    """The live state a client displays while a session runs."""

    state_root: str
    generations_recorded: int
    current_parent: str
    spent_wall_gpu_hours: float
    remaining_wall_gpu_hours: float
    maximum_total_wall_gpu_hours: float
    last_decision: Mapping[str, Any] = field(default_factory=dict)
    operator_stop: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_root": self.state_root,
            "generations_recorded": self.generations_recorded,
            "current_parent": self.current_parent,
            "spent_wall_gpu_hours": self.spent_wall_gpu_hours,
            "remaining_wall_gpu_hours": self.remaining_wall_gpu_hours,
            "maximum_total_wall_gpu_hours": self.maximum_total_wall_gpu_hours,
            "last_decision": dict(self.last_decision),
            "operator_stop": dict(self.operator_stop),
        }


@dataclass(frozen=True)
class HistoryRow:
    """One generation as the durable record left it."""

    index: int
    cycle_id: str
    generation: str
    effective_generation: str
    target_skill: str
    treatment: str
    candidate_result: str
    promotion_result: str
    wall_gpu_hours: float
    measured_effect: float | None
    regressions: tuple[str, ...]

    @property
    def promoted(self) -> bool:
        return self.promotion_result.lower() == "promoted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "cycle_id": self.cycle_id,
            "generation": self.generation,
            "effective_generation": self.effective_generation,
            "target_skill": self.target_skill,
            "treatment": self.treatment,
            "candidate_result": self.candidate_result,
            "promotion_result": self.promotion_result,
            "wall_gpu_hours": self.wall_gpu_hours,
            "measured_effect": self.measured_effect,
            "regressions": list(self.regressions),
        }


def _readiness_views(report: Any) -> tuple[ReadinessView, ...]:  # noqa: ANN401 - a report
    return tuple(
        ReadinessView(
            name=str(getattr(check, "check", "") or getattr(check, "name", "")),
            status=str(getattr(check, "status", "")),
            detail=str(getattr(check, "detail", "")),
            reason_code=str(getattr(check, "reason_code", "") or ""),
        )
        for check in (getattr(report, "checks", ()) or ())
    )


class AutonomousGrowthService:
    """Advance one generation at a time, and report everything it decided.

    Construct one per session with :meth:`open_from_paths` (a CLI or a UI) or
    :meth:`open` (a test or an embedding caller), then drive it with
    ``inspect`` / ``plan_next`` / ``prepare_next`` / ``start`` / ``resume``.
    """

    def __init__(
        self,
        *,
        loop: GrowthLoop,
        state: GrowthState,
        policy: LoopPolicy,
        profile: SkillProfile | None,
    ) -> None:
        self.loop = loop
        self.state = state
        self.policy = policy
        self.profile = profile

    # -- construction ------------------------------------------------------

    @classmethod
    def open_from_paths(
        cls,
        *,
        policy_path: str | Path,
        parent_declaration_path: str | Path,
        state_root: str | Path | None = None,
        parent_profile_path: str | Path = "",
        parent_evidence_path: str | Path = "",
        parent_evidence_ref: ParentEvidenceRef | Mapping[str, Any] | None = None,
        require_profile: bool = False,
        executor: Callable[[FrozenCampaign], CampaignOutcome] | None = None,
        registry: Any = None,  # noqa: ANN401 - a benchmark registry
    ) -> "AutonomousGrowthService":
        """Wire a session from paths, refusing rather than guessing a missing one.

        ``parent_evidence_path`` names the parent's *measurement*: its run root
        (whose own candidate arm is profiled) or an arm report directly. It is
        not a directory search -- a session that cannot say which model it starts
        from is a session whose targets would be chosen from another model's
        evidence.
        """
        policy_file = Path(policy_path)
        if not policy_file.is_file():
            raise GrowthServiceRefusal(
                f"{SERVICE_POLICY_ABSENT}: the loop policy {policy_file} does not "
                "exist; the envelope is required and is never defaulted"
            )
        parent_file = Path(parent_declaration_path)
        if not parent_file.is_file():
            raise GrowthServiceRefusal(
                f"{SERVICE_PARENT_ABSENT}: the parent declaration {parent_file} does "
                "not exist; which generation to advance from is not inferable"
            )
        policy = LoopPolicy.from_file(policy_file)
        parent = CampaignManifest.from_file(parent_file)

        profile = None
        if str(parent_profile_path).strip():
            profile = load_profile_from(parent_profile_path)
        elif str(parent_evidence_path).strip():
            evidence = Path(parent_evidence_path)
            # A run root holds its own candidate arm; a file *is* the arm.
            profile = (
                profile_from_run_root(evidence)
                if evidence.is_dir()
                else _profile_from_arm_report(evidence)
            )
        if require_profile and profile is None:
            # Checked *before* the durable state is opened: a refusal must not
            # leave a session directory behind it for a later run to adopt.
            raise GrowthServiceRefusal(
                f"{SERVICE_PROFILE_ABSENT}: no measured parent capability profile -- "
                "pass the parent's profile JSON, or the run root or arm report its "
                "candidate evaluation is measured from; a target chosen from an "
                "unmeasured profile is not evidence-backed"
            )
        state = GrowthState(
            root=Path(state_root)
            if state_root
            else Path(parent.state_root).parent / "growth-state"
        )
        return cls.open(
            policy=policy,
            parent=parent,
            state=state,
            profile=profile,
            parent_evidence_ref=parent_evidence_ref,
            executor=executor,
            registry=registry,
        )

    @classmethod
    def open(
        cls,
        *,
        policy: LoopPolicy,
        parent: CampaignManifest,
        state: GrowthState,
        profile: SkillProfile | Mapping[str, Any] | None = None,
        parent_evidence_ref: ParentEvidenceRef | Mapping[str, Any] | None = None,
        executor: Callable[[FrozenCampaign], CampaignOutcome] | None = None,
        registry: Any = None,  # noqa: ANN401 - a benchmark registry
    ) -> "AutonomousGrowthService":
        loop = GrowthLoop(
            policy=policy,
            state=state,
            executor=executor,
            parent_declaration=parent,
            parent_profile=profile,
            parent_evidence=parent_evidence_ref,
            registry=registry,
        )
        return cls(loop=loop, state=state, policy=policy, profile=loop.profile)

    # -- read-only ---------------------------------------------------------

    def inspect(self) -> LineageView:
        """The lineage the session stands on, from stated evidence."""
        ref = self.loop.parent_evidence
        declaration = self.loop.parent_declaration
        profile = self.profile
        return LineageView(
            generation=ref.generation,
            base_model_path=ref.base_model_path,
            base_model_digest=ref.base_model_digest,
            adapter_path=ref.adapter_path,
            adapter_digest=ref.adapter_digest,
            run_root=ref.run_root,
            measured_arm_path=ref.measured_arm_path,
            trusted_ancestor=str(declaration.protection.trusted_ancestor_version),
            trusted_ancestor_arm=str(declaration.baseline_eval_report_path),
            declared_protected=tuple(declaration.protected_benchmarks),
            protected_skills=tuple(sorted(self.loop.protected_skills)),
            profile_generation=str(getattr(profile, "generation", "") or ""),
            profile_measured_skills=tuple(
                estimate.skill for estimate in (profile.measured if profile else ())
            ),
            policy_digest=self.policy.digest(),
        )

    def plan_next(self, *, exhausted: Iterable[str] = ()) -> PlanningView:
        """Ask the loop's own decision engine what it would attempt next."""
        planned = self.loop.plan_next(exhausted=exhausted)
        if isinstance(planned, LoopDecision):
            return PlanningView(
                parent_version=self.loop.parent_evidence.generation,
                decision=planned.to_dict(),
            )
        return PlanningView(
            parent_version=self.loop.parent_evidence.generation,
            proposal=planned.to_dict(),
        )

    def prepare_next(self, *, exhausted: Iterable[str] = ()) -> PreparationView:
        """Draft, prepare and freeze the next attempt, then gate it on readiness.

        Both halves of the front end, in the order a run performs them, so a
        preview cannot disagree with the run about either the recipe set or the
        readiness verdict.
        """
        prepared = self.loop.prepare_next(exhausted=exhausted)
        if isinstance(prepared, LoopDecision):
            return PreparationView(decision=prepared.to_dict())
        report = self.loop.readiness_of(prepared.frozen)
        return PreparationView(attempt=prepared, checks=_readiness_views(report))

    def status(self) -> StatusView:
        """Durable state: what has been spent, and what the loop last decided."""
        rows = self.state.interventions()
        spent = sum(float(row.get("cost_gpu_hours", 0.0) or 0.0) for row in rows)
        stopping = self.state.stopping_state()
        return StatusView(
            state_root=str(self.state.root),
            generations_recorded=len(rows),
            current_parent=self.loop.parent_evidence.generation,
            spent_wall_gpu_hours=spent,
            remaining_wall_gpu_hours=float(self.policy.maximum_total_wall_gpu_hours)
            - spent,
            maximum_total_wall_gpu_hours=float(self.policy.maximum_total_wall_gpu_hours),
            last_decision=(
                dict(stopping) if stopping.get("action") in TERMINAL_DECISIONS else {}
            ),
            operator_stop=self.state.operator_stop() or {},
        )

    def history(self) -> tuple[HistoryRow, ...]:
        """Every generation this session's durable record remembers."""
        return history_from_state(self.state)

    # -- actions -----------------------------------------------------------

    def start(
        self, *, max_generations: int | None = None, resume: bool = False
    ) -> LoopRunReport:
        """Run the loop. The one place a client may spend compute."""
        if not resume:
            # An operator asking to start or resume is overriding any earlier
            # stop request, so the durable flag cannot leave the session wedged.
            self.state.clear_stop()
        return self.loop.run(max_generations=max_generations, resume=resume)

    def request_stop(self, *, reason: str = "operator stop after the current campaign") -> None:
        """Ask the loop to finish the campaign it is running and start no more.

        Durable and honest: this stops at the next generation boundary, which is
        a boundary the loop already re-checks. It does not pretend to reach into
        a running trainer, because nothing in the campaign stack can be cancelled
        mid-kernel safely.
        """
        self.state.request_stop(reason=reason)


def history_from_state(state: GrowthState) -> tuple[HistoryRow, ...]:
    """Read a durable record into history rows. One owner of the row shape.

    ``effective_generation`` is what the lineage became: the candidate when the
    generation promoted, the parent when it did not. A reader that reported the
    candidate of a rejected generation as "the current model" would be claiming
    an advance that never happened.
    """
    rows: list[HistoryRow] = []
    for index, row in enumerate(state.interventions(), start=1):
        rows.append(
            HistoryRow(
                index=index,
                cycle_id=str(row.get("cycle_id", "")),
                generation=str(row.get("generation", "")),
                effective_generation=str(
                    row.get("effective_generation") or row.get("generation", "")
                ),
                target_skill=str(row.get("target_skill", "")),
                treatment=str(row.get("training_type", "")),
                candidate_result=str(row.get("candidate_result", "")),
                promotion_result=str(row.get("promotion_result", "")),
                wall_gpu_hours=float(row.get("cost_gpu_hours", 0.0) or 0.0),
                measured_effect=_optional_float(row.get("measured_effect")),
                regressions=tuple(str(r) for r in (row.get("regressions") or ())),
            )
        )
    return tuple(rows)


def _optional_float(value: Any) -> float | None:  # noqa: ANN401 - a document field
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _profile_from_arm_report(path: Path) -> SkillProfile | None:
    """A SkillProfile from an arm report file that *is* the measurement."""
    from chowder.evals.result import EvalReport

    from .target_selection import build_skill_profile

    if not path.is_file():
        return None
    try:
        report = EvalReport.load(path)
    except Exception:  # noqa: BLE001 - an unreadable measurement is an absent one
        return None
    runs = tuple(run for run in report.runs if run.score is not None)
    if not runs:
        return None
    return build_skill_profile(
        generation=str(report.generation_version or "unknown"), runs=runs
    )


def loop_run_was_successful(report: LoopRunReport) -> bool:
    """Whether a session stopped *correctly* (the exit-code rule, in one place).

    Finishing, exhausting the envelope or stopping at a plateau are successes.
    The refusals (UNCERTAIN, REQUIRES_HUMAN_REVIEW) are not, and a run that
    refused must not exit zero.
    """
    from .growth_loop import STOP_BUDGET, STOP_OPERATOR, STOP_PLATEAU, STOP_SUCCESS

    return report.decision.action in {
        STOP_SUCCESS,
        STOP_PLATEAU,
        STOP_BUDGET,
        STOP_OPERATOR,
    }


__all__ = [
    "AutonomousGrowthService",
    "GrowthServiceRefusal",
    "HistoryRow",
    "LineageView",
    "PlanningView",
    "PreparationView",
    "ReadinessView",
    "StatusView",
    "history_from_state",
    "loop_run_was_successful",
]
