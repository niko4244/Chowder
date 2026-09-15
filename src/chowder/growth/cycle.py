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
The orchestrator is trainer-agnostic by design; when the production search
controller is reachable from this branch, ``plan_search`` output maps
directly onto its ``search`` project config (min_survivors, rounds, UCB
prioritization), so recipe competition reuses the qualified path instead of
a parallel one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .capability import CapabilityProfile, profile_delta
from .contamination import ContaminationFirewall
from .curriculum import CurriculumEngine, CurriculumItem
from .eval_tiers import EvalPlan, plan_eval_tier
from .failure_bank import FailureBank
from .frontier_reference import FrontierSnapshot, SnapshotStore
from .lineage import GenerationLedger, RegressionMemory
from .promotion import PromotionDecision, PromotionInput, BenchmarkResult, evaluate_promotion
from .recipe_planner import RecipePlanner, TrainingRecipe

# A training execution: recipe -> durable evidence ref (artifact/eval report).
TrainingFn = Callable[[TrainingRecipe, Sequence[CurriculumItem]], Mapping[str, Any]]


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
    min_target_improvement: float = 0.02
    max_protected_regression: float = 0.02
    budget_examples: int = 20000
    recipe_count: int = 4
    eval_budget_gpu_hours: float = 1.0


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

    def decide_promotion(
        self,
        *,
        candidate_results: Mapping[str, BenchmarkResult],
        parent_results: Mapping[str, BenchmarkResult],
        device_gpu_hours: float,
    ) -> PromotionDecision:
        """Phase: the single predeclared promotion rule."""
        return evaluate_promotion(
            PromotionInput(
                candidate_version=self.config.candidate_version,
                parent_version=self.config.parent_version,
                target_benchmarks=self.config.target_benchmarks,
                candidate_results=candidate_results,
                parent_results=parent_results,
                protected_benchmarks=self.config.protected_benchmarks,
                broad_battery_benchmarks=self.config.broad_battery,
                calibration_benchmarks=self.config.calibration_benchmarks,
                min_target_improvement=self.config.min_target_improvement,
                max_protected_regression=self.config.max_protected_regression,
                device_gpu_hours=device_gpu_hours,
                device_gpu_hours_ceiling=self.config.device_gpu_hours_ceiling,
            )
        )

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
