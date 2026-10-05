"""The autonomous growth cycle: Model N → evaluation → curriculum → training →
verification → promotion → Model N+1.

Each phase produces a machine-readable record answering "why did Chowder do
this?": skill selection traces, mixture proportions, recipe projections,
statistical verdicts. The orchestrator owns sequencing and provenance only --
it never overrides hard constraints (contamination, budget, protected
regressions) and never invents evidence.

Integration seam with the production trainer: the caller supplies a
``TrainingFn`` that executes one recipe (typically through ``chowder
train`` / ``run_project``) and returns the artifact/evaluation evidence.
The orchestrator is trainer-agnostic by design and never invokes a trainer
itself.

Integration seam with evaluation: ``decide_promotion_from_runs`` consumes raw
``BenchmarkRun``s and delegates the arithmetic to ``metric_binding``, which
reads each metric's declared polarity and 0..1 scale out of the benchmark
registry. No phase body in this module converts a metric by hand -- a score
computed inline here would be a promotion scale nobody declared.

Recipe competition is bounded by this package's own planner and budget
envelope. Real successive-halving rounds are driven *above* this cycle, by
``chowder.growth.candidate_search``, which allocates each round's budget and
chooses survivors through ``successive_halving.HalvingSchedule`` -- one owner
of that policy, shared with the ``EvolutionEngine`` controller. Nothing here
invents a ``search`` project config for the cycle to target: ``run_project``
has no search section, so the rounds live in the campaign's declared search
and are executed through this cycle's own ``TrainingFn``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

from chowder.evals.result import BenchmarkRun

from .capability import CapabilityProfile, profile_delta
from .compute_cost import settlement_refusal
from .contamination import ContaminationFirewall
from .curriculum import CurriculumEngine, CurriculumItem
from .eval_tiers import EvalPlan, plan_eval_tier
from .eval_isolation import EvalTierPolicy, assert_promotion_gate_isolation
from .failure_bank import FailureBank
from .frontier_reference import FrontierSnapshot, SnapshotStore
from .lineage import GenerationLedger, RegressionMemory
from .metric_binding import MetricBinder, PromotionAssembly
from .promotion import PromotionDecision, PromotionInput, BenchmarkResult, evaluate_promotion
from .recipe_planner import RecipePlanner, TrainingRecipe
from .retention import RetentionProfile, evaluate_retention

# A training execution: recipe -> durable evidence ref (artifact/eval report).
TrainingFn = Callable[[TrainingRecipe, Sequence[CurriculumItem]], Mapping[str, Any]]

# Allowed candidate-selection evidence: the per-recipe training outcome and
# target-instrument smoke scores. Protected/broad/calibration/reliability
# rows are promotion GATES, not optimization targets -- selection code that
# could see them could pick the candidate that games the final battery, so
# the type system keeps them out of the function's reach.
SelectionEvidence = Mapping[str, Mapping[str, Any]]


def select_candidate(
    results: Sequence[Mapping[str, Any]],
    *,
    order: str = "first_successful",
) -> Mapping[str, Any] | None:
    """Deterministic candidate selection from allowed evidence only.

    Reads ONLY each attempt's own training evidence: ``status``,
    ``candidate_succeeded``, ``artifact_ref``/``artifact_sha256``, the
    recipe id, and the recorded train loss / smoke quality -- the fields the
    TrainingFn itself produced about its own run. It has no parameter that
    could accept protected or broad benchmark scores, so recipe selection
    cannot peek at final-gate evidence by construction.

    A settlement-refused attempt is not selectable either: settlement runs
    after ``candidate_succeeded`` is set, so an over-budget attempt can carry
    a success flag, and the attempt that gets measured for promotion must be
    one whose cost claims settled.

    Policies:
    - ``first_successful`` (default): the first recipe (in preregistered
      order) whose training succeeded and produced an artifact.
    - ``first_by_loss``: the first-preregistered-tie-break by lowest
      recorded training loss among successful attempts.

    Ties resolve by the preregistered recipe order. Protected benchmarks
    stay promotion gates; they are never selection inputs.
    """
    if order not in {"first_successful", "first_by_loss"}:
        raise ValueError(f"unknown candidate-selection policy {order!r}")
    successful = [
        r
        for r in results
        if (r.get("candidate_succeeded") is True or r.get("status") == "SUCCEEDED")
        and r.get("artifact_ref")
        and settlement_refusal(r) is None
    ]
    if not successful:
        return None
    if order == "first_by_loss":
        # Lowest recorded train loss among successful attempts; ties keep
        # preregistered order (stable min).
        return min(
            successful,
            key=lambda r: float(
                (r.get("candidate_metrics") or {}).get("train_loss", float("inf"))
            ),
        )
    # first_successful: the first recipe (in preregistered order) that
    # succeeded and produced an artifact.
    return successful[0]


def retention_values(
    profile: RetentionProfile,
    results: Mapping[str, BenchmarkResult],
    *,
    candidate_side: bool,
) -> dict[str, float]:
    """dimension -> measured score, from the benchmark each constraint names.

    Both sides count only earned measurements. The candidate side counts
    only rows measured on this generation (``gate_eligible``); the parent
    side counts rows measured on the parent arm (``parent_measured``) -- a
    carried reference is a quotation from history, not a baseline, so it
    reads as unmeasured. A missing, unmeasured, or unearned row is left
    out, so :func:`evaluate_retention` fails closed on it -- an unmeasured
    gate is not a passed gate.

    Public because it is the *one* provenance filter over declared-gate
    inputs: the promotion path calls it here, and the Gen-2 judge
    (``docs/gen2/judge_gen2.py``, prereg amendment 15) calls the same
    function so its own recomputation of a declared constraint cannot
    diverge from the run's on which rows count.
    """
    values: dict[str, float] = {}
    for constraint in profile.constraints:
        result = results.get(constraint.benchmark)
        if result is None:
            continue
        if candidate_side and not result.gate_eligible:
            continue
        if not candidate_side and not result.parent_measured:
            continue
        values[constraint.dimension] = float(result.score)
    return values


@dataclass(frozen=True)
class CycleConfig:
    cycle_id: str
    parent_version: str
    candidate_version: str
    device_gpu_hours_ceiling: float
    target_benchmarks: tuple[str, ...]
    protected_benchmarks: tuple[str, ...]
    broad_battery: tuple[str, ...]
    calibration_benchmarks: tuple[str, ...] = ()
    #: Pass@k-capable evals the cycle predeclares as reliability gates. The
    #: binder forwards them to promotion, where a regression past the declared
    #: tolerance is a hard rejection and a declared-but-unmeasured set is
    #: inconclusive rather than a free pass.
    reliability_benchmarks: tuple[str, ...] = ()
    min_target_improvement: float = 0.02
    max_protected_regression: float = 0.02
    budget_examples: int = 20000
    recipe_count: int = 4
    eval_budget_gpu_hours: float = 1.0
    #: The campaign's preregistered retention constraints. The promotion
    #: phase consults them before anything promotes: a candidate that wins
    #: its target while breaching a declared constraint is REJECTED, and an
    #: unmeasured constraint is a violation, not a pass. Optional only in the
    #: sense that a campaign must declare it for it to bind -- when it is
    #: declared, the gate is structural, not advisory.
    retention_profile: RetentionProfile | None = None
    #: The declared tier classification of the campaign's benchmarks. When
    #: declared alongside a retention profile, every constraint's benchmark
    #: must classify as promotion evidence; a constraint measured on
    #: search-readable evidence refuses outright (the search could otherwise
    #: shape its own gate).
    eval_tier_policy: EvalTierPolicy | None = None


@dataclass(frozen=True)
class CyclePhase:
    phase: str
    verdict: str
    detail: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "verdict": self.verdict,
            "detail": self.detail,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class CycleOutcome:
    cycle_id: str
    parent_version: str
    candidate_version: str
    verdict: str  # PROMOTED / REJECTED / INCONCLUSIVE / TAINTED / REFUSED
    phases: tuple[CyclePhase, ...]
    promotion: Mapping[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "parent_version": self.parent_version,
            "candidate_version": self.candidate_version,
            "verdict": self.verdict,
            "phases": [phase.to_dict() for phase in self.phases],
            "promotion": self.promotion,
        }


class GrowthCycle:
    """One Model N -> Model N+1 attempt, every decision recorded."""

    def __init__(
        self,
        config: CycleConfig,
        *,
        curriculum: CurriculumEngine,
        planner: RecipePlanner,
        failure_bank: FailureBank,
        firewall: ContaminationFirewall,
        ledger: GenerationLedger,
        regression_memory: RegressionMemory,
        snapshots: SnapshotStore,
        train_fn: TrainingFn,
    ) -> None:
        self.config = config
        self.curriculum = curriculum
        self.planner = planner
        self.failure_bank = failure_bank
        self.firewall = firewall
        self.ledger = ledger
        self.regression_memory = regression_memory
        self.snapshots = snapshots
        self.train_fn = train_fn

    # ---------------- phases ----------------

    def plan_curriculum(
        self,
        profile: CapabilityProfile,
        *,
        frontier_skills: Mapping[str, float] | None = None,
    ) -> tuple[CurriculumItem, ...]:
        """Phase: capability -> curriculum. Refuses to plan from nothing."""
        if not profile.skills:
            return ()
        protected = tuple(self.config.protected_benchmarks)
        items = self.curriculum.plan(
            model_version=self.config.parent_version,
            profile=profile,
            protected_sets=protected,
            frontier=frontier_skills,
            budget_examples=self.config.budget_examples,
        )
        return items

    def plan_recipes(
        self, items: Sequence[CurriculumItem]
    ) -> tuple[TrainingRecipe, ...]:
        """Phase: curriculum -> bounded competing recipes."""
        return self.planner.propose(items, count=self.config.recipe_count)

    def plan_eval(self, *, stage: str, tiers: Mapping[str, Sequence[str]],
                  costs: Mapping[str, float], targeted: Sequence[str] = ()) -> EvalPlan:
        """Phase: which evaluation tier this stage deserves."""
        return plan_eval_tier(
            stage=stage,
            benchmarks_by_tier=tiers,
            gpu_hours_per_benchmark=costs,
            budget_gpu_hours=self.config.eval_budget_gpu_hours,
            targeted_benchmarks=targeted,
        )

    def train_candidates(
        self,
        items: Sequence[CurriculumItem],
        recipes: Sequence[TrainingRecipe],
        *,
        contamination_samples: Mapping[str, Sequence[str]] | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        """Phase: train every recipe through the injected TrainingFn, with the
        contamination gate applied to the curriculum's material first."""
        if contamination_samples:
            for source_id, samples in contamination_samples.items():
                self.firewall.check_source(source_id=source_id, samples=list(samples))
        results: list[Mapping[str, Any]] = []
        for recipe in recipes:
            evidence = dict(self.train_fn(recipe, items))
            evidence["recipe_id"] = recipe.recipe_id
            evidence["projected_device_gpu_hours"] = recipe.projected_device_gpu_hours
            results.append(evidence)
        return tuple(results)

    def select_candidate(
        self,
        results: Sequence[Mapping[str, Any]],
        *,
        order: str = "first_successful",
    ) -> Mapping[str, Any] | None:
        """Phase: pick the candidate from allowed evidence only.

        Delegates to :func:`select_candidate`, which structurally cannot see
        protected/broad scores: selection happens BEFORE the final
        evaluation phase and from training-side evidence only.
        """
        return select_candidate(results, order=order)

    # ---------------- promotion gates ----------------

    def _apply_promotion_gates(
        self,
        decision: PromotionDecision,
        *,
        candidate_results: Mapping[str, BenchmarkResult],
        parent_results: Mapping[str, BenchmarkResult],
    ) -> PromotionDecision:
        """The gates the promotion path consults before anything promotes.

        The predeclared promotion rule keeps its protected-benchmark
        arithmetic; the campaign's preregistered retention profile is
        evaluated on top of it, so a candidate that wins its target while
        breaching a declared constraint is REJECTED with the violation named
        -- not a promotion with a footnote. Unmeasured constraints fail
        closed: an unmeasured gate is not a passed gate.

        When the campaign also declares a tier classification, every
        constraint's benchmark must be promotion evidence. A constraint
        measured on search-readable evidence would let the search shape its
        own gate, so that is refused outright -- it is wiring, not a
        measured outcome.

        Every violation's reason is recorded whatever verdict the predeclared
        rule already reached. A declared gate that fires on a candidate the
        protected-benchmark arithmetic rejected anyway is still a fact about
        that candidate, and a record that names it only when it was the sole
        cause teaches the wrong lesson from the same run. The verdict is
        only tightened: PROMOTED becomes REJECTED, and an already REJECTED,
        TAINTED or INCONCLUSIVE decision keeps the verdict the predeclared
        rule earned -- a declared breach does not manufacture a stronger
        verdict out of evidence the rule found too thin to decide.
        """
        profile = self.config.retention_profile
        if profile is None:
            return decision
        if self.config.eval_tier_policy is not None:
            assert_promotion_gate_isolation(
                policy=self.config.eval_tier_policy,
                retention_profile=profile,
            )
        violations = evaluate_retention(
            profile,
            parent_values=retention_values(profile, parent_results, candidate_side=False),
            candidate_values=retention_values(profile, candidate_results, candidate_side=True),
        )
        if not violations:
            return decision
        return replace(
            decision,
            verdict="REJECTED" if decision.verdict == "PROMOTED" else decision.verdict,
            reasons=tuple(decision.reasons)
            + tuple(violation.reason for violation in violations),
        )

    def decide_promotion(
        self,
        *,
        candidate_results: Mapping[str, BenchmarkResult],
        parent_results: Mapping[str, BenchmarkResult],
        device_gpu_hours: float,
        actual_wall_gpu_hours: float | None = None,
        actual_device_gpu_hours: float | None = None,
        wall_gpu_hours_ceiling: float | None = None,
    ) -> PromotionDecision:
        """Phase: the single predeclared promotion rule, then the declared
        retention and isolation gates."""
        decision = evaluate_promotion(
            PromotionInput(
                candidate_version=self.config.candidate_version,
                parent_version=self.config.parent_version,
                target_benchmarks=self.config.target_benchmarks,
                candidate_results=candidate_results,
                parent_results=parent_results,
                protected_benchmarks=self.config.protected_benchmarks,
                broad_battery_benchmarks=self.config.broad_battery,
                calibration_benchmarks=self.config.calibration_benchmarks,
                reliability_benchmarks=self.config.reliability_benchmarks,
                min_target_improvement=self.config.min_target_improvement,
                max_protected_regression=self.config.max_protected_regression,
                device_gpu_hours=device_gpu_hours,
                device_gpu_hours_ceiling=self.config.device_gpu_hours_ceiling,
                actual_wall_gpu_hours=actual_wall_gpu_hours,
                actual_device_gpu_hours=actual_device_gpu_hours,
                wall_gpu_hours_ceiling=wall_gpu_hours_ceiling,
            )
        )
        return self._apply_promotion_gates(
            decision,
            candidate_results=candidate_results,
            parent_results=parent_results,
        )

    def decide_promotion_from_runs(
        self,
        binder: MetricBinder,
        *,
        candidate_runs: Sequence[BenchmarkRun],
        parent_runs: Sequence[BenchmarkRun],
        device_gpu_hours: float = 0.0,
        actual_wall_gpu_hours: float | None = None,
        wall_gpu_hours_ceiling: float | None = None,
    ) -> PromotionAssembly:
        """Phase: measured runs -> the single predeclared promotion rule.

        The cycle owns the promotion *sets* (which benchmarks are targets,
        which are protected, which form the broad battery) and the declared
        tolerances; the binder owns turning raw metrics into declared 0..1
        scores. Splitting it this way is what keeps the arithmetic reviewable:
        no conversion happens in a phase body, and a benchmark the cycle names
        but the registry does not declare refuses rather than quietly
        disappearing from the comparison. The declared retention and
        isolation gates are applied to the bound decision before it returns.
        """
        assembly = binder.promotion_input(
            candidate_version=self.config.candidate_version,
            parent_version=self.config.parent_version,
            candidate_runs=candidate_runs,
            parent_runs=parent_runs,
            target_benchmarks=self.config.target_benchmarks,
            protected_benchmarks=self.config.protected_benchmarks,
            broad_battery_benchmarks=self.config.broad_battery,
            calibration_benchmarks=self.config.calibration_benchmarks,
            reliability_benchmarks=self.config.reliability_benchmarks,
            min_target_improvement=self.config.min_target_improvement,
            max_protected_regression=self.config.max_protected_regression,
            device_gpu_hours=device_gpu_hours,
            device_gpu_hours_ceiling=self.config.device_gpu_hours_ceiling,
            actual_wall_gpu_hours=actual_wall_gpu_hours,
            wall_gpu_hours_ceiling=wall_gpu_hours_ceiling,
        )
        decision = self._apply_promotion_gates(
            assembly.decision,
            candidate_results=assembly.promotion_input.candidate_results,
            parent_results=assembly.promotion_input.parent_results,
        )
        if decision is assembly.decision:
            return assembly
        return replace(assembly, decision=decision)

    def finalize(
        self,
        decision: PromotionDecision,
        *,
        base_model: Mapping[str, Any],
        dataset_manifest_ref: str,
        curriculum_manifest_ref: str,
        recipe: Mapping[str, Any],
        training_evidence_ref: str,
        evaluation_report_ref: str,
        frontier_snapshot_id: str | None = None,
        notes: str = "",
    ) -> CycleOutcome:
        """Record the outcome in the lineage ledger (PROMOTED and REJECTED
        both get recorded -- rejected candidates are evidence too)."""
        phases = (CyclePhase(
            phase="promotion",
            verdict=decision.verdict,
            detail="; ".join(decision.reasons) or "predeclared rule",
            provenance={"checks": dict(decision.checks)},
        ),)
        promoted = decision.verdict == "PROMOTED"
        if promoted:
            self.ledger.record(
                version=self.config.candidate_version,
                parent_version=self.config.parent_version,
                cycle_id=self.config.cycle_id,
                base_model=base_model,
                dataset_manifest_ref=dataset_manifest_ref,
                curriculum_manifest_ref=curriculum_manifest_ref,
                recipe=recipe,
                training_evidence_ref=training_evidence_ref,
                evaluation_report_ref=evaluation_report_ref,
                promotion=decision,
                required_probes=self.regression_memory.protected_benchmark_ids(),
                frontier_snapshot_id=frontier_snapshot_id,
                notes=notes,
            )
        return CycleOutcome(
            cycle_id=self.config.cycle_id,
            parent_version=self.config.parent_version,
            candidate_version=self.config.candidate_version,
            verdict=decision.verdict,
            phases=phases,
            promotion=decision.to_dict(),
        )

    def frontier_snapshot_for_cycle(self) -> FrontierSnapshot | None:
        """The frozen frontier this cycle was judged against, if declared."""
        try:
            return self.snapshots.get(f"{self.config.cycle_id}-frontier")
        except KeyError:
            return None
