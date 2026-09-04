from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .activation_offload import run_activation_offload_experiment
from .backends.transformers_peft import TransformersPeftExecutor
from .executors import ExecutionContext
from .frozen_layer_streaming import run_frozen_layer_streaming_experiment
from .memory_preflight import _SAFETY_MARGIN_GB, estimate_memory_requirements
from .optimizer_tiering import run_optimizer_tiering_experiment

_MECHANISM_NAMES = ("activation_offload", "optimizer_tiering", "frozen_layer_streaming")
# A documented starting point, not a claimed-optimal constant, matching
# every other threshold in this codebase's placement mechanisms. Real
# measured vram_saved_gb is rarely exactly 0.0 -- allocator noise/
# rounding on a workload with genuinely negligible activation/state
# pressure can still measure as a tiny positive number (confirmed for
# real: a tiny smoke-test model measured "savings" that print as 0.00 GB
# at 2 decimal places but are not literally zero). Without this floor, a
# mechanism offering no real benefit would still get folded into the
# "best" combination purely because summing a near-zero number produced
# a technically-smaller estimate.
_MEANINGFUL_SAVINGS_GB = 0.01


@dataclass(frozen=True)
class PlacementPlan:
    """A first, deterministic, evidence-based Memory Fabric placement
    plan (Phase 7E): given a recipe that does not fit under normal
    resident training, decides which combination of activation_offload/
    optimizer_tiering/frozen_layer_streaming to enable, using the real
    predicted savings each mechanism's own independent experiment
    already measures -- something none of them do alone today, since
    each only decides whether *it alone* is worth enabling.

    combined-effect caveat: a chosen combination of 2+ mechanisms is only
    ever considered when a REAL, empirically-measured
    `combined_mechanism_experiment.CombinedMechanismExperiment` for that
    exact combination+model+recipe+hardware has been cached (see
    `combination_validated` and `read_validated_combination`) -- an
    unvalidated multi-mechanism combination is never auto-selected, only
    reported as informational context in `reasoning`. Single mechanisms
    remain always-eligible, since each is already independently real-
    measured by its own always-run experiment. This is a real, deliberate
    safety choice, not an oversight: this module's own development found
    the naive additive assumption can be meaningfully wrong (a real
    combined run can show far less benefit than summing each mechanism's
    own isolated savings implies) -- see combined_mechanism_experiment.py's
    module docstring for the measured finding.

    Auto-applied by `TransformersPeftExecutor.resolved_activation_offload`/
    `resolved_optimizer_tiering`/`resolved_frozen_layer_streaming` when a
    mechanism's own config value is `"auto"` -- an explicit `"always"`/
    `"off"`/boolean on any individual mechanism always wins regardless of
    what this plan would have chosen for it, since those staticmethods
    only consult this plan inside their own `"auto"` branch.
    """

    fits_without_intervention: bool
    ddp_active: bool
    active_accelerator_count: int
    baseline_estimate_gb: float
    per_rank_available_gb: float
    enable_activation_offload: bool
    enable_optimizer_tiering: bool
    enable_frozen_layer_streaming: bool
    predicted_combined_estimate_gb: float | None
    fits_with_plan: bool
    combination_validated: bool
    reasoning: tuple[str, ...]


def _mechanism_savings_gb(
    *, activation_offload_exp, optimizer_tiering_exp, frozen_layer_streaming_exp
) -> dict[str, float]:
    """Real, measured (or honestly-approximated) VRAM this codebase's
    own experiments say each mechanism could free, keyed by mechanism
    name. 0.0 for an unavailable mechanism (never a candidate to enable).

    optimizer_tiering has no clean vram_saved_gb field the way the other
    two do -- bitsandbytes' paged optimizer does not reduce optimizer-
    state size, it makes the *entire* state pageable to host RAM under
    pressure. Its real, measured baseline AdamW state_bytes is used here
    as the upper bound of what paging could free -- an honest
    approximation (paging happens only under real pressure, not a
    guaranteed permanent reduction), not a claim that this is identical
    in kind to activation_offload/frozen_layer_streaming's real, direct
    peak-VRAM deltas.
    """
    savings = {name: 0.0 for name in _MECHANISM_NAMES}
    if activation_offload_exp.available:
        savings["activation_offload"] = max(0.0, activation_offload_exp.vram_saved_gb)
    if optimizer_tiering_exp.available:
        baseline_variant = optimizer_tiering_exp.variant("adamw")
        if baseline_variant is not None:
            savings["optimizer_tiering"] = baseline_variant.state_bytes / (1024**3)
    if frozen_layer_streaming_exp.available:
        savings["frozen_layer_streaming"] = max(0.0, frozen_layer_streaming_exp.vram_saved_gb)
    return savings


def _mechanism_penalty_ratios(
    *, activation_offload_exp, optimizer_tiering_exp, frozen_layer_streaming_exp
) -> dict[str, float]:
    return {
        "activation_offload": activation_offload_exp.wall_time_penalty_ratio,
        "optimizer_tiering": optimizer_tiering_exp.wall_time_penalty_ratio,
        "frozen_layer_streaming": frozen_layer_streaming_exp.wall_time_penalty_ratio,
    }


def build_placement_plan(
    *,
    resolved_config: Mapping[str, Any],
    context: ExecutionContext,
    work_dir: str | Path,
) -> PlacementPlan:
    """Build a first, deterministic Memory Fabric placement plan for the
    given recipe+hardware. See PlacementPlan's own docstring for the
    combined-effect caveat and why this is informational, not yet
    auto-applied.

    Real experiments are only run when actually needed: if the baseline
    recipe already fits, or the run is multi-GPU DDP (where every
    mechanism is independently, explicitly rejected -- see each
    executor's own DDP-rejection check), this returns immediately
    without spawning any of the three real experiment subprocesses.
    """
    baseline = estimate_memory_requirements(
        resolved_config=resolved_config, context=context, work_dir=work_dir
    )
    available_gb = baseline.per_rank_available_gb

    if baseline.fits:
        return PlacementPlan(
            fits_without_intervention=True,
            ddp_active=False,
            active_accelerator_count=1,
            baseline_estimate_gb=baseline.estimated_peak_gb,
            per_rank_available_gb=available_gb,
            enable_activation_offload=False,
            enable_optimizer_tiering=False,
            enable_frozen_layer_streaming=False,
            predicted_combined_estimate_gb=baseline.estimated_peak_gb,
            fits_with_plan=True,
            combination_validated=True,
            reasoning=(
                f"baseline recipe already fits: estimated "
                f"{baseline.estimated_peak_gb:.2f} GB vs {available_gb:.2f} GB "
                "available -- no Memory Fabric intervention needed",
            ),
        )

    active_accelerator_count = TransformersPeftExecutor._active_accelerator_count(context)
    if active_accelerator_count > 1:
        return PlacementPlan(
            fits_without_intervention=False,
            ddp_active=True,
            active_accelerator_count=active_accelerator_count,
            baseline_estimate_gb=baseline.estimated_peak_gb,
            per_rank_available_gb=available_gb,
            enable_activation_offload=False,
            enable_optimizer_tiering=False,
            enable_frozen_layer_streaming=False,
            predicted_combined_estimate_gb=None,
            fits_with_plan=False,
            combination_validated=True,
            reasoning=(
                f"baseline recipe does not fit ({baseline.estimated_peak_gb:.2f} GB "
                f"estimated vs {available_gb:.2f} GB available per device), but "
                f"active_accelerator_count={active_accelerator_count} -- every Memory "
                "Fabric mechanism explicitly rejects multi-GPU DDP as unverified, so "
                "no placement plan is possible here",
            ),
        )

    activation_offload_exp = run_activation_offload_experiment(
        resolved_config=resolved_config, context=context, work_dir=work_dir
    )
    optimizer_tiering_exp = run_optimizer_tiering_experiment(
        resolved_config=resolved_config, context=context, work_dir=work_dir
    )
    frozen_layer_streaming_exp = run_frozen_layer_streaming_experiment(
        resolved_config=resolved_config, context=context, work_dir=work_dir
    )
    savings = _mechanism_savings_gb(
        activation_offload_exp=activation_offload_exp,
        optimizer_tiering_exp=optimizer_tiering_exp,
        frozen_layer_streaming_exp=frozen_layer_streaming_exp,
    )
    penalty_ratios = _mechanism_penalty_ratios(
        activation_offload_exp=activation_offload_exp,
        optimizer_tiering_exp=optimizer_tiering_exp,
        frozen_layer_streaming_exp=frozen_layer_streaming_exp,
    )
    available_mechanisms = [
        name for name in _MECHANISM_NAMES if savings[name] >= _MEANINGFUL_SAVINGS_GB
    ]

    # A validated multi-mechanism combo uses its REAL measured
    # actual_combined_peak_vram_gb instead of the naive additive sum.
    # Unvalidated multi-mechanism combos are excluded entirely from
    # consideration -- "do not auto-apply an unvalidated combination" --
    # single mechanisms remain always-eligible since each is already
    # independently real-measured. Imported locally to avoid a module
    # import cycle (combined_mechanism_experiment.py imports from this
    # module for _MECHANISM_NAMES/_mechanism_savings_gb).
    from .combined_mechanism_experiment import read_validated_combination

    skipped_unvalidated: list[tuple[str, ...]] = []
    fitting_combinations: list[tuple[str, ...]] = []
    combination_estimates: dict[tuple[str, ...], float] = {}
    best_combination: tuple[str, ...] = ()
    best_combination_estimate = baseline.estimated_peak_gb
    for size in range(len(available_mechanisms) + 1):
        for combo in itertools.combinations(available_mechanisms, size):
            if size >= 2:
                validated = read_validated_combination(
                    mechanisms=combo, resolved_config=resolved_config, context=context, work_dir=work_dir,
                )
                if validated is None:
                    skipped_unvalidated.append(combo)
                    continue
                estimate = validated.actual_combined_peak_vram_gb
            else:
                estimate = baseline.estimated_peak_gb - sum(savings[name] for name in combo)
            combination_estimates[combo] = estimate
            if estimate < best_combination_estimate:
                best_combination_estimate = estimate
                best_combination = combo
            if estimate <= available_gb - _SAFETY_MARGIN_GB:
                fitting_combinations.append(combo)

    reasoning: list[str] = [
        f"baseline recipe does not fit: estimated {baseline.estimated_peak_gb:.2f} GB "
        f"vs {available_gb:.2f} GB available per device",
    ]
    for name in _MECHANISM_NAMES:
        reasoning.append(
            f"{name}: real measured savings {savings[name]:.2f} GB, "
            f"penalty ratio {penalty_ratios[name]:.2f}x"
            if savings[name] >= _MEANINGFUL_SAVINGS_GB
            else f"{name}: unavailable or no meaningful real measured savings on this hardware"
        )
    for combo in skipped_unvalidated:
        reasoning.append(
            f"{'+'.join(combo)}: no real combined-mechanism measurement cached -- excluded "
            "from auto-selection (run combined_mechanism_experiment.run_combined_mechanism_"
            "experiment to validate this combination before it can be auto-applied)"
        )

    if fitting_combinations:
        # Fewest mechanisms enabled first (least overhead/complexity for
        # no benefit); tie-break by the lowest worst-case penalty ratio
        # among the enabled mechanisms, since penalties are not assumed
        # to compound in any specific, unverified way.
        chosen = min(
            fitting_combinations,
            key=lambda combo: (
                len(combo),
                max((penalty_ratios[name] for name in combo), default=0.0),
            ),
        )
        chosen_estimate = combination_estimates[chosen]
        validated_note = "real measured" if len(chosen) >= 2 else "predicted"
        reasoning.append(
            f"chosen combination {chosen or '(none)'}: {validated_note} combined estimate "
            f"{chosen_estimate:.2f} GB, fits within {available_gb:.2f} GB available"
        )
        return PlacementPlan(
            fits_without_intervention=False,
            ddp_active=False,
            active_accelerator_count=active_accelerator_count,
            baseline_estimate_gb=baseline.estimated_peak_gb,
            per_rank_available_gb=available_gb,
            enable_activation_offload="activation_offload" in chosen,
            enable_optimizer_tiering="optimizer_tiering" in chosen,
            enable_frozen_layer_streaming="frozen_layer_streaming" in chosen,
            predicted_combined_estimate_gb=chosen_estimate,
            fits_with_plan=True,
            combination_validated=True,
            reasoning=tuple(reasoning),
        )

    reasoning.append(
        f"no combination of available mechanisms is predicted to fit -- best case "
        f"{best_combination or '(none)'} still estimates {best_combination_estimate:.2f} GB "
        f"vs {available_gb:.2f} GB available; recommending the closest combination anyway "
        "as free insurance, but this recipe is not expected to fit even with every "
        "Memory Fabric mechanism enabled"
    )
    return PlacementPlan(
        fits_without_intervention=False,
        ddp_active=False,
        active_accelerator_count=active_accelerator_count,
        baseline_estimate_gb=baseline.estimated_peak_gb,
        per_rank_available_gb=available_gb,
        enable_activation_offload="activation_offload" in best_combination,
        enable_optimizer_tiering="optimizer_tiering" in best_combination,
        enable_frozen_layer_streaming="frozen_layer_streaming" in best_combination,
        predicted_combined_estimate_gb=best_combination_estimate,
        fits_with_plan=False,
        combination_validated=True,
        reasoning=tuple(reasoning),
    )
