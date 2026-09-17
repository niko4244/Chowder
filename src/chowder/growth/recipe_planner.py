"""Training-recipe planner: bounded candidate recipes for one cycle.

Given the curriculum plan and measured hardware reality (the device
preflight numbers Chowder's routers/PEFT backends already measure), propose
a small set of competing recipes whose projected cost fits the preregistered
budget. The planner proposes; selection is whatever the caller's `TrainingFn`
and the cycle's predeclared promotion rule decide. Chowder's successive-halving
controller would be the qualified selector, but it has no production caller
yet and `run_project` has no `search` config for it to read
(`docs/ROADMAP.md`), so nothing here may assume that interface exists.

Variables stay inside the currently qualified training path: LoRA-family
post-training with measured memory/step/load budgets. Architecture changes
are a separate declared campaign, never a silent planner decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .curriculum import CurriculumItem


@dataclass(frozen=True)
class HardwareBudget:
    """Measured local reality (from device_preflight runs, not guesses)."""

    gpu_name: str
    vram_gb: float
    measured_step_seconds_at_seq: Mapping[int, float]  # seq_len -> s/step
    measured_load_seconds: float
    wall_multiplier: float = 3.5  # M = wall/device, measured rung-3c value

    def step_seconds(self, seq_len: int) -> float:
        if seq_len in self.measured_step_seconds_at_seq:
            return self.measured_step_seconds_at_seq[seq_len]
        known = sorted(self.measured_step_seconds_at_seq)
        if not known:
            raise ValueError("no measured step costs recorded")
        nearest = min(known, key=lambda s: abs(s - seq_len))
        return self.measured_step_seconds_at_seq[nearest] * (seq_len / nearest) ** 1.2


@dataclass(frozen=True)
class TrainingRecipe:
    """One bounded post-training recipe candidate."""

    recipe_id: str
    curriculum_item_ids: tuple[str, ...]
    mixture: Mapping[str, float]  # role -> share
    learning_rate: float
    scheduler: str
    warmup_steps: int
    lora_rank: int
    lora_alpha: int
    target_modules: tuple[str, ...]
    seq_len: int
    batch_size: int
    gradient_accumulation: int
    max_steps: int
    objective: str  # sft | dpo | continued_pretrain
    replay_rate: float
    dataset_manifest: Mapping[str, Any]
    projected_device_gpu_hours: float
    projected_wall_gpu_hours: float
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serializable provenance record for the lineage ledger."""
        return {
            "recipe_id": self.recipe_id,
            "curriculum_item_ids": list(self.curriculum_item_ids),
            "mixture": dict(self.mixture),
            "learning_rate": self.learning_rate,
            "scheduler": self.scheduler,
            "warmup_steps": self.warmup_steps,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "target_modules": list(self.target_modules),
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "max_steps": self.max_steps,
            "objective": self.objective,
            "replay_rate": self.replay_rate,
            "dataset_manifest": dict(self.dataset_manifest),
            "projected_device_gpu_hours": self.projected_device_gpu_hours,
            "projected_wall_gpu_hours": self.projected_wall_gpu_hours,
            "notes": self.notes,
        }

    def to_config_patch(self, *, backend_type: str = "router-healing") -> dict[str, Any]:
        """The nested ``config_patch`` this recipe contributes to one experiment.

        Only knobs the target backend's validator actually reads are emitted;
        the shape follows the project's ``backend.type`` (the binding passes
        the type of the template it is composing, so a recipe never invents a
        namespace the target backend would refuse):

        - ``router-healing``: the router engine's knobs.
        - ``transformers-peft``: the same training knobs in the peft
          validator's namespace (``backend.training.*`` plus ``max_length``).
          Note the peft path has no ``seq_len``: sequence length is the
          backend-level ``max_length``.

        ``ExperimentGraph`` resolves a patch by merging it with
        ``deep_merge_config``, so any other key -- bookkeeping included --
        would silently land inside a qualified configuration. An unknown
        backend type raises: emitting nothing silently would drop the
        recipe's knobs; emitting router-healing keys into another backend
        would fail validation anyway (both discovered on real runs).

        LoRA rank/alpha, and anything else the target path names inside its
        own spec, stay proposed-but-unmapped: inventing a namespace for them
        here would be drift, not integration. Recipe identity, mixture, and
        projections travel in ``to_dict()``, which is what the cycle ledger
        records.

        There is no project-level ``search`` config to target: ``run_project``
        has no search section and Chowder's successive-halving controller has
        no production caller yet (``docs/ROADMAP.md``).
        """
        if backend_type == "router-healing":
            return {
                "backend": {
                    "router_healing": {
                        "max_steps": self.max_steps,
                        "learning_rate": self.learning_rate,
                        "seq_len": self.seq_len,
                    }
                }
            }
        if backend_type == "transformers-peft":
            return {
                "backend": {
                    "max_length": self.seq_len,
                    "training": {
                        "max_steps": self.max_steps,
                        "learning_rate": self.learning_rate,
                        "lr_scheduler_type": self.scheduler,
                        "warmup_steps": self.warmup_steps,
                    },
                }
            }
        raise ValueError(
            f"recipe knobs have no mapping for backend type {backend_type!r}; "
            "a patch for an unknown backend would either be dropped or refused "
            "by the backend's validator -- name the backend or extend the mapper"
        )


class RecipePlanner:
    """Proposes N bounded recipes around the curriculum plan."""

    def __init__(
        self,
        *,
        budget: HardwareBudget,
        max_device_gpu_hours: float,
        max_wall_gpu_hours: float,
    ) -> None:
        self.budget = budget
        self.max_device = max_device_gpu_hours
        self.max_wall = max_wall_gpu_hours

    def project_cost(self, *, seq_len: int, max_steps: int) -> tuple[float, float]:
        """(device GPU-h, wall GPU-h) from measured step/load costs.
        Load cost is counted once per recipe (the paired/eval amortization
        is applied downstream); wall projection uses the measured multiplier."""
        step_seconds = self.budget.step_seconds(seq_len)
        device_seconds = max_steps * step_seconds + self.budget.measured_load_seconds
        device_gpu_hours = device_seconds / 3600.0
        wall_gpu_hours = device_gpu_hours * self.budget.wall_multiplier
        return (device_gpu_hours, wall_gpu_hours)

    def propose(
        self,
        items: Sequence[CurriculumItem],
        *,
        count: int = 4,
        base_examples: int = 3000,
    ) -> tuple[TrainingRecipe, ...]:
        """A deterministic spread of recipes around evidence-based defaults.

        The spread is over the highest-leverage hyperparameters for small
        LoRA post-training (LR x rank x replay), each projected against the
        measured budget; recipes that bust either ceiling are refused, not
        silently included.
        """
        if not items:
            return ()
        item_ids = tuple(item.item_id for item in items)
        dataset_manifest = {
            "items": [item.to_dict() for item in items],
            "total_examples": sum(item.example_count for item in items),
            "protected_regression_sets": sorted(
                {s for item in items for s in item.protected_regression_set}
            ),
        }
        total_examples = sum(item.example_count for item in items) or base_examples
        steps_estimate = max(12, min(400, total_examples // 30))

        # LR x rank x replay grid: 2 x 2 x 2 candidates, truncated to `count`.
        grid = [
            (lr, rank, replay)
            for lr in (1e-4, 2e-4)
            for rank in (16, 32)
            for replay in (0.1, 0.25)
        ]
        recipes: list[TrainingRecipe] = []
        for index, (lr, rank, replay) in enumerate(grid[:count]):
            seq_len = 2048
            device, wall = self.project_cost(seq_len=seq_len, max_steps=steps_estimate)
            if device > self.max_device or wall > self.max_wall:
                # Scale steps down to fit, floor at a minimum useful run.
                while steps_estimate > 12:
                    steps_estimate = max(12, int(steps_estimate * 0.8))
                    device, wall = self.project_cost(seq_len=seq_len, max_steps=steps_estimate)
                    if device <= self.max_device and wall <= self.max_wall:
                        break
                if device > self.max_device or wall > self.max_wall:
                    raise ValueError(
                        "no bounded recipe fits the declared budget: "
                        f"smallest candidate projects {device:.4f} device / "
                        f"{wall:.4f} wall GPU-h vs ceilings {self.max_device:.4f}/"
                        f"{self.max_wall:.4f}"
                    )
            recipes.append(
                TrainingRecipe(
                    recipe_id=f"recipe-{index:02d}-lr{lr:g}-r{rank}-replay{replay:g}",
                    curriculum_item_ids=item_ids,
                    mixture={"TARGET": 0.5, "PRESERVE": 0.15, "GENERAL": 0.2, "REPLAY": replay, "STRETCH": 0.1},
                    learning_rate=lr,
                    scheduler="cosine",
                    warmup_steps=max(2, steps_estimate // 10),
                    lora_rank=rank,
                    lora_alpha=rank * 2,
                    target_modules=("router.gate",) if any("router" in i.skill for i in items) else ("q_proj", "v_proj"),
                    seq_len=seq_len,
                    batch_size=2,
                    gradient_accumulation=4,
                    max_steps=steps_estimate,
                    objective="sft",
                    replay_rate=replay,
                    dataset_manifest=dataset_manifest,
                    projected_device_gpu_hours=device,
                    projected_wall_gpu_hours=wall,
                    notes=f"projected from measured step cost {self.budget.step_seconds(seq_len):.3f}s/step @ {seq_len}",
                )
            )
        return tuple(recipes)
