"""ResearchMission: the highest-level object in scientist mode.

A mission states what to improve *in capability terms* — never "raise benchmark
X". Benchmarks are sensors of capabilities, not the objective; the mission
therefore carries capability priorities, the capabilities that must be
protected rather than optimized, the two research/compute budgets, and the
stop conditions under which the research session ends.

Mirrors the growth policy idiom: a frozen dataclass constructed through
:meth:`from_mapping` with a **closed key set** — an unknown key is a refused
limit that looks enforced and is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

#: Experiment families a proposal may name. Architecture-mutating families are
#: opt-in per mission; data/optimization/adapter/strategy families are the
#: default research surface. System-efficiency proposals are admissible but
#: never alter model behavior, so they sit behind no extra gate.
EXPERIMENT_TYPES: tuple[str, ...] = (
    "data",
    "optimization",
    "adapter",
    "training_strategy",
    "architecture",
    "system_efficiency",
    "measurement",
)

#: The architecture family is the one an LLM must never reach by default.
ARCHITECTURE_EXPERIMENT_TYPE = "architecture"

AUTONOMY_LEVELS: tuple[str, ...] = ("low", "medium", "high")


@dataclass(frozen=True)
class MissionBudget:
    """The two ledgers a mission owns (see the two-ledger rule in
    docs/SCIENTIST_MODE.md § threat model)."""

    max_gpu_hours: float = 0.0
    max_tree_nodes: int = 0
    max_parallel_branches: int = 1
    max_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.max_gpu_hours < 0 or self.max_cost_usd < 0:
            raise ValueError("mission budget ceilings must be non-negative")
        if self.max_tree_nodes < 0 or self.max_parallel_branches < 1:
            raise ValueError("mission budget node/branch ceilings must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_gpu_hours": self.max_gpu_hours,
            "max_tree_nodes": self.max_tree_nodes,
            "max_parallel_branches": self.max_parallel_branches,
            "max_cost_usd": self.max_cost_usd,
        }


@dataclass(frozen=True)
class ResearchMission:
    """What the research session is for, declared before any research runs."""

    mission_id: str
    objective: str
    priorities: Mapping[str, float]
    protected_capabilities: tuple[str, ...] = ()
    budget: MissionBudget = field(default_factory=MissionBudget)
    autonomy: str = "low"
    allowed_experiment_types: tuple[str, ...] = (
        "data",
        "optimization",
        "adapter",
        "training_strategy",
        "measurement",
    )
    stop_conditions: tuple[str, ...] = (
        "compute_budget_exhausted",
        "no_admissible_hypothesis",
        "plateau_across_rounds",
        "evidence_fails_to_support",
        "human_review_required",
    )

    def __post_init__(self) -> None:
        if not self.mission_id:
            raise ValueError("mission_id is required")
        if not self.objective.strip():
            raise ValueError("an objective is required; a mission without a question is not research")
        if not self.priorities:
            raise ValueError("capability priorities are required; do not reduce a mission to one metric")
        negative = [k for k, v in self.priorities.items() if v < 0]
        if negative:
            raise ValueError(f"capability priorities must be non-negative: {negative}")
        total = sum(self.priorities.values())
        if total <= 0:
            raise ValueError("capability priorities must sum to a positive weight")
        if self.autonomy not in AUTONOMY_LEVELS:
            raise ValueError(f"unknown autonomy level: {self.autonomy}")
        overlap = sorted(set(self.protected_capabilities) & set(self.priorities))
        if overlap:
            raise ValueError(
                f"capabilities are either protected or optimized, never both: {overlap}"
            )
        unknown = [t for t in self.allowed_experiment_types if t not in EXPERIMENT_TYPES]
        if unknown:
            raise ValueError(f"unknown experiment types in mission: {unknown}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "objective": self.objective,
            "priorities": dict(self.priorities),
            "protected_capabilities": list(self.protected_capabilities),
            "budget": self.budget.to_dict(),
            "autonomy": self.autonomy,
            "allowed_experiment_types": list(self.allowed_experiment_types),
            "stop_conditions": list(self.stop_conditions),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ResearchMission":
        _KEYS = (
            "mission_id",
            "objective",
            "priorities",
            "protected_capabilities",
            "budget",
            "autonomy",
            "allowed_experiment_types",
            "stop_conditions",
        )
        unknown = sorted(set(data) - set(_KEYS))
        if unknown:
            raise ValueError(f"unknown mission keys (fail-closed): {unknown}")
        missing = [k for k in ("mission_id", "objective", "priorities") if k not in data]
        if missing:
            raise ValueError(f"mission is missing required keys: {missing}")
        budget_data = dict(data.get("budget") or {})
        _BUDGET_KEYS = ("max_gpu_hours", "max_tree_nodes", "max_parallel_branches", "max_cost_usd")
        unknown_budget = sorted(set(budget_data) - set(_BUDGET_KEYS))
        if unknown_budget:
            raise ValueError(f"unknown mission budget keys (fail-closed): {unknown_budget}")
        return cls(
            mission_id=str(data["mission_id"]),
            objective=str(data["objective"]),
            priorities={str(k): float(v) for k, v in dict(data["priorities"]).items()},
            protected_capabilities=tuple(str(c) for c in data.get("protected_capabilities") or ()),
            budget=MissionBudget(
                max_gpu_hours=float(budget_data.get("max_gpu_hours", 0.0)),
                max_tree_nodes=int(budget_data.get("max_tree_nodes", 0)),
                max_parallel_branches=int(budget_data.get("max_parallel_branches", 1)),
                max_cost_usd=float(budget_data.get("max_cost_usd", 0.0)),
            ),
            autonomy=str(data.get("autonomy", "low")),
            allowed_experiment_types=tuple(
                str(t) for t in data.get("allowed_experiment_types") or (
                    "data", "optimization", "adapter", "training_strategy", "measurement",
                )
            ),
            stop_conditions=tuple(
                str(s) for s in data.get("stop_conditions") or (
                    "compute_budget_exhausted",
                    "no_admissible_hypothesis",
                    "plateau_across_rounds",
                    "evidence_fails_to_support",
                    "human_review_required",
                )
            ),
        )
