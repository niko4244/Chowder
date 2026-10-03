"""ExperimentProposal: the strict typed boundary every requested experiment
crosses.

A provider's proposal is *not executable*. It becomes executable only by
passing admission (schema → policy → budget) and then compilation
(`lab_bridge.ExperimentCompiler`). The proposal states what to change, what to
hold constant, what evidence would falsify it, and what it will cost — a
proposal without a falsification rule is refused, because an experiment that
cannot fail proves nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .hypothesis import Hypothesis
from .mission import EXPERIMENT_TYPES


@dataclass(frozen=True)
class ExperimentConstraint:
    """What the mission and policy allow; constructed by Chowder, never by the
    provider (the provider never writes its own permission slip)."""

    allowed_experiment_types: tuple[str, ...]
    max_gpu_hours_per_experiment: float
    protected_capabilities: tuple[str, ...] = ()
    require_falsification_rule: bool = True
    require_replication_plan: bool = False

    def __post_init__(self) -> None:
        unknown = [t for t in self.allowed_experiment_types if t not in EXPERIMENT_TYPES]
        if unknown:
            raise ValueError(f"constraints name unknown experiment types: {unknown}")
        if self.max_gpu_hours_per_experiment <= 0:
            raise ValueError("constraint ceiling must be positive")


@dataclass(frozen=True)
class DataStrategy:
    """Where training material comes from and how contamination is handled.

    The provider *requests* a data strategy; admission still routes it through
    Chowder's own registry/contamination firewall — this object cannot admit a
    source by itself."""

    source_kinds: tuple[str, ...] = ()          # registry | synthetic | replay | curriculum
    source_ids: tuple[str, ...] = ()            # registered source ids, if named
    replay_ratio: float | None = None
    contamination_policy: str = "firewall_default"

    def __post_init__(self) -> None:
        known = ("registry", "synthetic", "replay", "curriculum")
        bad = [s for s in self.source_kinds if s not in known]
        if bad:
            raise ValueError(f"unknown data source kinds: {bad}; known: {known}")
        if self.replay_ratio is not None and not (0.0 <= self.replay_ratio <= 1.0):
            raise ValueError("replay_ratio must be within [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_kinds": list(self.source_kinds),
            "source_ids": list(self.source_ids),
            "replay_ratio": self.replay_ratio,
            "contamination_policy": self.contamination_policy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataStrategy":
        unknown = sorted(set(data) - {"source_kinds", "source_ids", "replay_ratio",
                                      "contamination_policy"})
        if unknown:
            raise ValueError(f"unknown data-strategy keys: {unknown}")
        return cls(
            source_kinds=tuple(str(s) for s in data.get("source_kinds", ())),
            source_ids=tuple(str(s) for s in data.get("source_ids", ())),
            replay_ratio=(None if data.get("replay_ratio") is None
                          else float(data["replay_ratio"])),
            contamination_policy=str(data.get("contamination_policy", "firewall_default")),
        )


@dataclass(frozen=True)
class TrainingRecipeDelta:
    """The *change* to the training recipe; everything not named is held
    constant by construction."""

    learning_rate: float | None = None
    scheduler: str | None = None
    warmup_ratio: float | None = None
    weight_decay: float | None = None
    gradient_clipping: float | None = None
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_target_preset: str | None = None
    epochs: int | None = None
    batch_size: int | None = None
    grad_accumulation: int | None = None
    max_length: int | None = None
    training_type: str | None = None            # sft | continued_pretrain | preference | repair

    _KNOWN = (
        "learning_rate", "scheduler", "warmup_ratio", "weight_decay",
        "gradient_clipping", "lora_rank", "lora_alpha", "lora_target_preset",
        "epochs", "batch_size", "grad_accumulation", "max_length", "training_type",
    )

    def changed(self) -> tuple[str, ...]:
        return tuple(k for k in self._KNOWN if getattr(self, k) is not None)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self._KNOWN if getattr(self, k) is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingRecipeDelta":
        unknown = sorted(set(data) - set(cls._KNOWN))
        if unknown:
            raise ValueError(f"unknown recipe-delta keys: {unknown}")
        return cls(**{k: v for k, v in data.items() if k in cls._KNOWN})


@dataclass(frozen=True)
class ExperimentProposal:
    """A requested experiment. Data, not code; never self-executing."""

    proposal_id: str
    hypothesis_id: str
    experiment_type: str                        # one of EXPERIMENT_TYPES
    intervention: str
    variables_changed: tuple[str, ...]
    variables_held_constant: tuple[str, ...]
    training_recipe_delta: TrainingRecipeDelta
    data_strategy: DataStrategy
    requested_evaluations: tuple[str, ...]      # capability/skill names to measure
    transfer_evaluations: tuple[str, ...] = ()  # independent representations
    replication_plan: str = ""                  # seeds/stages, "" = none requested
    controls: tuple[str, ...] = ()
    expected_outcome: str = ""
    falsification_rule: str = ""
    estimated_gpu_hours: float = 0.0
    safety_requirements: tuple[str, ...] = ()
    provider: str = ""
    mission_id: str = ""
    status: str = "proposed"                    # proposed | admitted | refused | compiled | executed

    _KNOWN = (
        "proposal_id", "hypothesis_id", "experiment_type", "intervention",
        "variables_changed", "variables_held_constant", "training_recipe_delta",
        "data_strategy", "requested_evaluations", "transfer_evaluations",
        "replication_plan", "controls", "expected_outcome", "falsification_rule",
        "estimated_gpu_hours", "safety_requirements", "provider", "mission_id",
        "status",
    )

    def __post_init__(self) -> None:
        if not self.proposal_id:
            raise ValueError("proposal_id is required")
        if not self.hypothesis_id:
            raise ValueError("a proposal must name its hypothesis; no orphan experiments")
        if self.experiment_type not in EXPERIMENT_TYPES:
            raise ValueError(
                f"unknown experiment_type: {self.experiment_type}; known: {EXPERIMENT_TYPES}"
            )
        if not self.intervention.strip():
            raise ValueError("a proposal must state its intervention")
        if not self.variables_changed:
            raise ValueError("a proposal must name the variables it changes")
        if not self.requested_evaluations:
            raise ValueError("a proposal must name the evaluations it requests")
        if not self.expected_outcome.strip():
            raise ValueError("a proposal must state its expected outcome")

    def validate(self, constraints: ExperimentConstraint) -> tuple[str, ...]:
        """Return refusal reasons (empty = admissible under the constraints)."""
        reasons: list[str] = []
        if self.experiment_type not in constraints.allowed_experiment_types:
            reasons.append(
                f"EXPERIMENT_TYPE_NOT_ALLOWED: {self.experiment_type!r} is not in this "
                "mission's allowed experiment types"
            )
        # Architecture-mutating proposals are refused unless the mission
        # explicitly allowlists them (allowed_experiment_types above). The
        # default mission constraint excludes "architecture", so an LLM
        # suggesting an architecture change is refused here without any
        # additional policy work — and an operator who genuinely wants one
        # must name it in the mission, which is the explicit permission the
        # threat model requires.
        if constraints.require_falsification_rule and not self.falsification_rule.strip():
            reasons.append("FALSIFICATION_RULE_REQUIRED: state what result would refute the hypothesis")
        if constraints.require_replication_plan and not self.replication_plan.strip():
            reasons.append("REPLICATION_PLAN_REQUIRED: state the seed/replication plan")
        if self.estimated_gpu_hours > constraints.max_gpu_hours_per_experiment:
            reasons.append(
                f"EXPERIMENT_TOO_EXPENSIVE: estimated {self.estimated_gpu_hours} GPU-hours "
                f"exceeds the per-experiment ceiling {constraints.max_gpu_hours_per_experiment}"
            )
        overlap = sorted(set(self.variables_changed) & set(self.variables_held_constant))
        if overlap:
            reasons.append(f"VARIABLE_BOTH_CHANGED_AND_HELD: {overlap}")
        protected_overlap = sorted(
            set(self.requested_evaluations) & set(constraints.protected_capabilities)
        )
        if protected_overlap:
            reasons.append(
                f"PROTECTED_CAPABILITY_REQUESTED_AS_TARGET: {protected_overlap} are "
                "protected; they are measured as gates, never optimized"
            )
        return tuple(reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "hypothesis_id": self.hypothesis_id,
            "experiment_type": self.experiment_type,
            "intervention": self.intervention,
            "variables_changed": list(self.variables_changed),
            "variables_held_constant": list(self.variables_held_constant),
            "training_recipe_delta": self.training_recipe_delta.to_dict(),
            "data_strategy": self.data_strategy.to_dict(),
            "requested_evaluations": list(self.requested_evaluations),
            "transfer_evaluations": list(self.transfer_evaluations),
            "replication_plan": self.replication_plan,
            "controls": list(self.controls),
            "expected_outcome": self.expected_outcome,
            "falsification_rule": self.falsification_rule,
            "estimated_gpu_hours": self.estimated_gpu_hours,
            "safety_requirements": list(self.safety_requirements),
            "provider": self.provider,
            "mission_id": self.mission_id,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentProposal":
        unknown = sorted(set(data) - set(cls._KNOWN))
        if unknown:
            raise ValueError(f"unknown proposal keys (fail-closed): {unknown}")
        missing = [k for k in ("proposal_id", "hypothesis_id", "experiment_type",
                               "intervention", "variables_changed",
                               "requested_evaluations") if k not in data]
        if missing:
            raise ValueError(f"proposal is missing required keys: {missing}")
        recipe = TrainingRecipeDelta.from_dict(dict(data.get("training_recipe_delta") or {}))
        data_strategy = DataStrategy.from_dict(dict(data.get("data_strategy") or {}))
        return cls(
            proposal_id=str(data["proposal_id"]),
            hypothesis_id=str(data["hypothesis_id"]),
            experiment_type=str(data["experiment_type"]),
            intervention=str(data["intervention"]),
            variables_changed=tuple(str(v) for v in data["variables_changed"]),
            variables_held_constant=tuple(
                str(v) for v in data.get("variables_held_constant", ())
            ),
            training_recipe_delta=recipe,
            data_strategy=data_strategy,
            requested_evaluations=tuple(str(e) for e in data["requested_evaluations"]),
            transfer_evaluations=tuple(str(e) for e in data.get("transfer_evaluations", ())),
            replication_plan=str(data.get("replication_plan", "")),
            controls=tuple(str(c) for c in data.get("controls", ())),
            expected_outcome=str(data.get("expected_outcome", "")),
            falsification_rule=str(data.get("falsification_rule", "")),
            estimated_gpu_hours=float(data.get("estimated_gpu_hours", 0.0)),
            safety_requirements=tuple(str(s) for s in data.get("safety_requirements", ())),
            provider=str(data.get("provider", "")),
            mission_id=str(data.get("mission_id", "")),
            status=str(data.get("status", "proposed")),
        )
