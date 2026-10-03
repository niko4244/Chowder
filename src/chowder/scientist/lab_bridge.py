"""The lab bridge: Chowder compiles proposals into its own experiments.

`ExperimentCompiler` turns an *admitted* `ExperimentProposal` into a
Chowder-native experiment specification — the growth-campaign recipe shape the
production training/evaluation bindings already execute — and returns it for
execution through the production path. The provider never sees the compiled
form and never executes anything.

Honest status (docs/SCIENTIST_MODE.md §6): the compiler emits the campaign
spec; executing that spec end-to-end on real hardware through the research
tree is demonstrated only via the growth loop's own production path. The
spec shape is the binding seam, deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .observation import ExperimentObservation, Measurement
from .proposal import ExperimentProposal


class CompilationRefusal(RuntimeError):
    """Raised when an admitted proposal cannot be compiled. Admission and
    compilation are separate gates: admission says the *idea* is allowed;
    compilation says *this lab* can run it."""


@dataclass(frozen=True)
class CompiledExperiment:
    """A Chowder-native experiment spec, ready for the production path.

    `campaign_spec` mirrors the growth campaign draft fields
    (`NextCampaignBuilder.CampaignDraft`): recipe dict, data strategy,
    evaluation surfaces. It is a declaration, not a schedule."""

    experiment_id: str
    proposal_id: str
    hypothesis_id: str
    campaign_spec: dict[str, Any]
    estimated_gpu_hours: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "proposal_id": self.proposal_id,
            "hypothesis_id": self.hypothesis_id,
            "campaign_spec": self.campaign_spec,
            "estimated_gpu_hours": self.estimated_gpu_hours,
        }


def _recipe_delta_to_patch(delta: Any) -> dict[str, Any]:
    """Map a proposal's recipe delta onto the project-config keys the
    production training executor reads (`backend.*` / `training` fields)."""
    patch: dict[str, Any] = {}
    if delta.learning_rate is not None:
        patch["backend.training.learning_rate"] = delta.learning_rate
    if delta.scheduler is not None:
        patch["backend.training.lr_scheduler_type"] = delta.scheduler
    if delta.warmup_ratio is not None:
        patch["backend.training.warmup_ratio"] = delta.warmup_ratio
    if delta.weight_decay is not None:
        patch["backend.training.weight_decay"] = delta.weight_decay
    if delta.gradient_clipping is not None:
        patch["backend.training.max_grad_norm"] = delta.gradient_clipping
    if delta.lora_rank is not None:
        patch["backend.lora.r"] = delta.lora_rank
    if delta.lora_alpha is not None:
        patch["backend.lora.alpha"] = delta.lora_alpha
    if delta.lora_target_preset is not None:
        patch["backend.lora.target_preset"] = delta.lora_target_preset
    if delta.epochs is not None:
        patch["backend.training.epochs"] = delta.epochs
    if delta.batch_size is not None:
        patch["backend.training.batch_size"] = delta.batch_size
    if delta.grad_accumulation is not None:
        patch["backend.training.grad_accumulation"] = delta.grad_accumulation
    if delta.max_length is not None:
        patch["backend.training.max_length"] = delta.max_length
    if delta.training_type is not None:
        patch["training_type"] = delta.training_type
    return patch


class ExperimentCompiler:
    """Compiles admitted proposals into production-shaped experiment specs."""

    def __init__(self, *, protected_benchmarks: tuple[str, ...] = ()) -> None:
        self._protected_benchmarks = tuple(protected_benchmarks)

    def compile(self, proposal: ExperimentProposal, *, seed: int = 2026) -> CompiledExperiment:
        if proposal.status != "admitted":
            raise CompilationRefusal(
                f"REFUSAL_NOT_ADMITTED: {proposal.proposal_id} is "
                f"{proposal.status!r}; only admitted proposals compile"
            )
        if proposal.experiment_type == "architecture":
            raise CompilationRefusal(
                "REFUSAL_ARCHITECTURE_NOT_COMPILABLE: the lab bridge does not "
                "compile architecture-mutating experiments in this release; "
                "admission may allowlist the type, the compiler still refuses it"
            )
        recipe_patch = _recipe_delta_to_patch(proposal.training_recipe_delta)
        campaign_spec: dict[str, Any] = {
            "recipe_patch": recipe_patch,
            "data": {
                "source_kinds": list(proposal.data_strategy.source_kinds),
                "source_ids": list(proposal.data_strategy.source_ids),
                "replay_ratio": proposal.data_strategy.replay_ratio,
                "contamination_policy": proposal.data_strategy.contamination_policy,
            },
            "evaluations": {
                "target_surfaces": list(proposal.requested_evaluations),
                "transfer_surfaces": list(proposal.transfer_evaluations),
                "protected_benchmarks": list(self._protected_benchmarks),
            },
            "replication": {
                "plan": proposal.replication_plan,
                "seed": seed,
            },
            "controls": list(proposal.controls),
            "falsification_rule": proposal.falsification_rule,
        }
        experiment_id = f"sciexp-{proposal.proposal_id}"
        return CompiledExperiment(
            experiment_id=experiment_id,
            proposal_id=proposal.proposal_id,
            hypothesis_id=proposal.hypothesis_id,
            campaign_spec=campaign_spec,
            estimated_gpu_hours=proposal.estimated_gpu_hours,
        )

    def observation_for(
        self,
        experiment: CompiledExperiment,
        *,
        observation_id: str,
        run_id: str,
        metrics: dict[str, float],
        surface_benchmarks: dict[str, str] | None = None,
        wall_gpu_hours: float = 0.0,
        status: str = "complete",
        notes: str = "",
        provider: str = "",
        mission_id: str = "",
    ) -> ExperimentObservation:
        """Build the run-grounded observation for a compiled experiment.

        The caller must supply the *registry's* run id; recording still goes
        through `ResearchMemory.record_observation`, which re-verifies the run
        exists and is complete. Metrics arrive as `{surface: value}`;
        `surface_benchmarks` optionally names the benchmark per surface."""
        surface_benchmarks = surface_benchmarks or {}
        measurements = tuple(
            Measurement(
                surface=surface,
                benchmark=surface_benchmarks.get(surface, f"surface:{surface}"),
                value=float(value),
            )
            for surface, value in metrics.items()
        )
        return ExperimentObservation(
            observation_id=observation_id,
            run_id=run_id,
            experiment_ref=experiment.experiment_id,
            proposal_id=experiment.proposal_id,
            hypothesis_id=experiment.hypothesis_id,
            measurements=measurements,
            status=status,
            wall_gpu_hours=wall_gpu_hours,
            notes=notes,
            provider=provider,
            mission_id=mission_id,
        )
