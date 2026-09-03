from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from chowder.activation_offload import ActivationOffloadExperiment
from chowder.executors import ExecutionContext
from chowder.frozen_layer_streaming import FrozenLayerStreamingExperiment
from chowder.memory import HardwareProfile
from chowder.memory_preflight import MemoryEstimate
from chowder.optimizer_tiering import OptimizerTieringExperiment, OptimizerVariantMeasurement
from chowder.placement_policy import PlacementPlan, build_placement_plan

_REAL_ML_SMOKE = pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
_TINY_MODEL = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"


def _config(**backend_overrides):
    backend = {
        "type": "transformers-peft",
        "base_model": "org/model",
        "dataset": "train.jsonl",
        "max_length": 64,
        "quantization": "none",
        "precision": "fp32",
        "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
    }
    backend.update(backend_overrides)
    return {"backend": backend}


def _hardware(vram_gb=16.0, pools=(16.0,)):
    return HardwareProfile(vram_gb, 64.0, 500.0, 12.0, 40.0, 3.0, accelerator_vram_gb=pools)


def _context(hardware=None, resolved_config=None):
    return ExecutionContext(hardware or _hardware(), ".", seed=1, resolved_config=resolved_config)


def _estimate(**overrides):
    fields = dict(
        device="cuda",
        frozen_params=1_000_000,
        trainable_params=1000,
        max_length=64,
        measured_peak_gb_at_batch_1=8.0,
        measured_peak_gb_at_batch_2=9.0,
        per_example_activation_gb=1.0,
        configured_batch_size=2,
        estimated_peak_gb=20.0,
        per_rank_available_gb=16.0,
        fits=False,
        from_cache=True,
    )
    fields.update(overrides)
    return MemoryEstimate(**fields)


def _offload_exp(**overrides):
    fields = dict(
        device="cuda", available=True, batch_size=2, max_length=64,
        baseline_peak_vram_gb=20.0, offload_peak_vram_gb=15.0, vram_saved_gb=5.0,
        baseline_wall_seconds=0.1, offload_wall_seconds=0.12,
        wall_time_penalty_ratio=1.2, per_rank_available_gb=16.0,
        required=False, recommended=True,
    )
    fields.update(overrides)
    return ActivationOffloadExperiment(**fields)


def _tiering_exp(**overrides):
    variants = overrides.pop(
        "variants",
        (
            OptimizerVariantMeasurement(name="adamw", step_seconds=0.05, state_bytes=3 * 1024**3),
            OptimizerVariantMeasurement(name="paged_adamw", step_seconds=0.06, state_bytes=3 * 1024**3),
        ),
    )
    fields = dict(
        device="cuda", available=True, batch_size=2, max_length=64,
        variants=variants, model_peak_vram_gb=17.0, per_rank_available_gb=16.0,
        wall_time_penalty_ratio=1.1, required=False, recommended=True,
    )
    fields.update(overrides)
    return OptimizerTieringExperiment(**fields)


def _streaming_exp(**overrides):
    fields = dict(
        device="cuda", available=True, batch_size=2, max_length=64,
        baseline_peak_vram_gb=20.0, streamed_peak_vram_gb=18.0, vram_saved_gb=2.0,
        baseline_wall_seconds=0.1, streamed_wall_seconds=0.13,
        wall_time_penalty_ratio=1.3, bytes_transferred_per_step=1000,
        per_rank_available_gb=16.0, required=False, recommended=True,
    )
    fields.update(overrides)
    return FrozenLayerStreamingExperiment(**fields)


def _patch_experiments(offload=None, tiering=None, streaming=None):
    return (
        patch(
            "chowder.placement_policy.run_activation_offload_experiment",
            return_value=offload or _offload_exp(),
        ),
        patch(
            "chowder.placement_policy.run_optimizer_tiering_experiment",
            return_value=tiering or _tiering_exp(),
        ),
        patch(
            "chowder.placement_policy.run_frozen_layer_streaming_experiment",
            return_value=streaming or _streaming_exp(),
        ),
    )


# --- baseline already fits / DDP short-circuits ------------------------------


def test_fits_without_intervention_when_baseline_already_fits():
    with patch(
        "chowder.placement_policy.estimate_memory_requirements",
        return_value=_estimate(estimated_peak_gb=4.0, per_rank_available_gb=16.0, fits=True),
    ):
        plan = build_placement_plan(resolved_config=_config(), context=_context(), work_dir=".")
    assert plan.fits_without_intervention is True
    assert plan.fits_with_plan is True
    assert not plan.enable_activation_offload
    assert not plan.enable_optimizer_tiering
    assert not plan.enable_frozen_layer_streaming


def test_never_calls_any_experiment_when_baseline_already_fits():
    p1, p2, p3 = _patch_experiments()
    with p1 as mock_offload, p2 as mock_tiering, p3 as mock_streaming, patch(
        "chowder.placement_policy.estimate_memory_requirements",
        return_value=_estimate(fits=True),
    ):
        build_placement_plan(resolved_config=_config(), context=_context(), work_dir=".")
    mock_offload.assert_not_called()
    mock_tiering.assert_not_called()
    mock_streaming.assert_not_called()


def test_ddp_short_circuits_with_no_plan_and_no_experiment_calls():
    config = _config(runtime={"active_accelerator_count": 2})
    context = _context(_hardware(vram_gb=16.0, pools=(16.0, 16.0)), resolved_config=config)
    p1, p2, p3 = _patch_experiments()
    with p1 as mock_offload, p2 as mock_tiering, p3 as mock_streaming, patch(
        "chowder.placement_policy.estimate_memory_requirements",
        return_value=_estimate(fits=False),
    ):
        plan = build_placement_plan(resolved_config=config, context=context, work_dir=".")
    assert plan.ddp_active is True
    assert plan.active_accelerator_count == 2
    assert plan.fits_with_plan is False
    assert not plan.enable_activation_offload
    assert not plan.enable_optimizer_tiering
    assert not plan.enable_frozen_layer_streaming
    mock_offload.assert_not_called()
    mock_tiering.assert_not_called()
    mock_streaming.assert_not_called()


# --- combination search -------------------------------------------------------


def test_recommends_the_single_mechanism_that_alone_makes_it_fit():
    """estimated=20, available=16 (fits <= 15.5 w/ 0.5 safety margin).
    activation_offload alone saves 5 -> 15, fits. Cheapest (fewest
    mechanisms) fitting combination must be chosen over larger ones."""
    with patch(
        "chowder.placement_policy.estimate_memory_requirements",
        return_value=_estimate(estimated_peak_gb=20.0, per_rank_available_gb=16.0, fits=False),
    ):
        p1, p2, p3 = _patch_experiments()
        with p1, p2, p3:
            plan = build_placement_plan(resolved_config=_config(), context=_context(), work_dir=".")
    assert plan.fits_with_plan is True
    assert plan.enable_activation_offload is True
    assert plan.enable_optimizer_tiering is False
    assert plan.enable_frozen_layer_streaming is False
    assert plan.predicted_combined_estimate_gb == pytest.approx(15.0)


def test_combines_two_mechanisms_when_neither_alone_is_enough():
    """Each mechanism alone saves only 2.5 GB (17.5 GB, still over the
    15.5 fit boundary) -- only a combination of two reaches 15 GB,
    fitting."""
    offload = _offload_exp(baseline_peak_vram_gb=20.0, offload_peak_vram_gb=17.5, vram_saved_gb=2.5)
    streaming = _streaming_exp(baseline_peak_vram_gb=20.0, streamed_peak_vram_gb=17.5, vram_saved_gb=2.5)
    tiering = _tiering_exp(
        variants=(
            OptimizerVariantMeasurement(name="adamw", step_seconds=0.05, state_bytes=int(0.001 * 1024**3)),
            OptimizerVariantMeasurement(name="paged_adamw", step_seconds=0.06, state_bytes=int(0.001 * 1024**3)),
        )
    )
    with patch(
        "chowder.placement_policy.estimate_memory_requirements",
        return_value=_estimate(estimated_peak_gb=20.0, per_rank_available_gb=16.0, fits=False),
    ):
        p1, p2, p3 = _patch_experiments(offload=offload, tiering=tiering, streaming=streaming)
        with p1, p2, p3:
            plan = build_placement_plan(resolved_config=_config(), context=_context(), work_dir=".")
    assert plan.fits_with_plan is True
    assert plan.enable_activation_offload is True
    assert plan.enable_frozen_layer_streaming is True
    assert plan.enable_optimizer_tiering is False  # negligible real savings, excluded
    assert plan.predicted_combined_estimate_gb == pytest.approx(15.0)


def test_prefers_fewer_mechanisms_when_multiple_combinations_fit():
    """activation_offload alone (saves 5) already fits -- must not also
    enable optimizer_tiering (saves an additional 3) just because the
    bigger combination also fits; fewer mechanisms wins."""
    offload = _offload_exp(vram_saved_gb=5.0)
    tiering = _tiering_exp(
        variants=(
            OptimizerVariantMeasurement(name="adamw", step_seconds=0.05, state_bytes=3 * 1024**3),
            OptimizerVariantMeasurement(name="paged_adamw", step_seconds=0.06, state_bytes=3 * 1024**3),
        )
    )
    with patch(
        "chowder.placement_policy.estimate_memory_requirements",
        return_value=_estimate(estimated_peak_gb=20.0, per_rank_available_gb=16.0, fits=False),
    ):
        p1, p2, p3 = _patch_experiments(offload=offload, tiering=tiering)
        with p1, p2, p3:
            plan = build_placement_plan(resolved_config=_config(), context=_context(), work_dir=".")
    assert plan.fits_with_plan is True
    assert plan.enable_activation_offload is True
    assert plan.enable_optimizer_tiering is False


def test_breaks_ties_between_equally_sized_combinations_by_lowest_penalty_ratio():
    """Both activation_offload alone and frozen_layer_streaming alone
    save enough to fit -- the one with the lower measured wall-time
    penalty ratio must be chosen."""
    offload = _offload_exp(vram_saved_gb=6.0, wall_time_penalty_ratio=3.0)
    streaming = _streaming_exp(vram_saved_gb=6.0, wall_time_penalty_ratio=1.1)
    with patch(
        "chowder.placement_policy.estimate_memory_requirements",
        return_value=_estimate(estimated_peak_gb=20.0, per_rank_available_gb=16.0, fits=False),
    ):
        p1, p2, p3 = _patch_experiments(offload=offload, streaming=streaming)
        with p1, p2, p3:
            plan = build_placement_plan(resolved_config=_config(), context=_context(), work_dir=".")
    assert plan.fits_with_plan is True
    assert plan.enable_frozen_layer_streaming is True
    assert plan.enable_activation_offload is False


def test_no_combination_fits_reports_the_closest_one_honestly():
    """Every mechanism's real measured savings, even combined, is not
    enough -- fits_with_plan must be False, and the plan should still
    report the best (maximum-savings) combination as "free insurance"
    rather than silently recommending nothing."""
    offload = _offload_exp(vram_saved_gb=1.0)
    tiering = _tiering_exp(
        variants=(
            OptimizerVariantMeasurement(name="adamw", step_seconds=0.05, state_bytes=int(0.5 * 1024**3)),
            OptimizerVariantMeasurement(name="paged_adamw", step_seconds=0.06, state_bytes=int(0.5 * 1024**3)),
        )
    )
    streaming = _streaming_exp(vram_saved_gb=1.0)
    with patch(
        "chowder.placement_policy.estimate_memory_requirements",
        return_value=_estimate(estimated_peak_gb=30.0, per_rank_available_gb=16.0, fits=False),
    ):
        p1, p2, p3 = _patch_experiments(offload=offload, tiering=tiering, streaming=streaming)
        with p1, p2, p3:
            plan = build_placement_plan(resolved_config=_config(), context=_context(), work_dir=".")
    assert plan.fits_with_plan is False
    assert plan.enable_activation_offload is True
    assert plan.enable_optimizer_tiering is True
    assert plan.enable_frozen_layer_streaming is True
    assert plan.predicted_combined_estimate_gb == pytest.approx(27.5)


def test_excludes_unavailable_mechanisms_from_the_combination_search():
    offload = _offload_exp(available=False, vram_saved_gb=0.0)
    with patch(
        "chowder.placement_policy.estimate_memory_requirements",
        return_value=_estimate(estimated_peak_gb=20.0, per_rank_available_gb=16.0, fits=False),
    ):
        p1, p2, p3 = _patch_experiments(offload=offload)
        with p1, p2, p3:
            plan = build_placement_plan(resolved_config=_config(), context=_context(), work_dir=".")
    assert plan.enable_activation_offload is False


def test_excludes_negligible_savings_that_round_to_zero():
    """A real measurement can report a tiny positive vram_saved_gb from
    allocator noise on a workload with genuinely no real pressure to
    relieve -- this must not be treated as a meaningful candidate."""
    offload = _offload_exp(vram_saved_gb=0.0001)
    with patch(
        "chowder.placement_policy.estimate_memory_requirements",
        return_value=_estimate(estimated_peak_gb=20.0, per_rank_available_gb=16.0, fits=False),
    ):
        p1, p2, p3 = _patch_experiments(offload=offload)
        with p1, p2, p3:
            plan = build_placement_plan(resolved_config=_config(), context=_context(), work_dir=".")
    assert plan.enable_activation_offload is False


def test_reasoning_is_non_empty_and_mentions_every_mechanism():
    with patch(
        "chowder.placement_policy.estimate_memory_requirements",
        return_value=_estimate(estimated_peak_gb=20.0, per_rank_available_gb=16.0, fits=False),
    ):
        p1, p2, p3 = _patch_experiments()
        with p1, p2, p3:
            plan = build_placement_plan(resolved_config=_config(), context=_context(), work_dir=".")
    reasoning_text = " ".join(plan.reasoning)
    assert "activation_offload" in reasoning_text
    assert "optimizer_tiering" in reasoning_text
    assert "frozen_layer_streaming" in reasoning_text
    assert len(plan.reasoning) >= 2


# --- real end-to-end (actual model load + real experiments) -----------------


@_REAL_ML_SMOKE
def test_real_placement_plan_fits_without_intervention_on_real_hardware(tmp_path):
    from chowder.hardware import detect_hardware
    from chowder.project_runner import hardware_profile_from_snapshot

    hardware = hardware_profile_from_snapshot(detect_hardware(str(tmp_path)))
    context = ExecutionContext(hardware, str(tmp_path), seed=7)
    config = {
        "backend": {
            "type": "transformers-peft",
            "base_model": _TINY_MODEL,
            "dataset": "unused.jsonl",
            "max_length": 64,
            "quantization": "none",
            "precision": "fp32",
            "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
            "training": {"batch_size": 2},
        }
    }
    plan = build_placement_plan(resolved_config=config, context=context, work_dir=str(tmp_path))
    assert isinstance(plan, PlacementPlan)
    # estimate_memory_requirements measures 0.0 GB on CPU-only hardware
    # (nothing to measure -- there is no VRAM allocator to query) and
    # its own established convention treats that as correctly fitting,
    # not a real positive estimate -- only assert a real positive
    # baseline when CUDA is actually present.
    import torch

    if torch.cuda.is_available():
        assert plan.baseline_estimate_gb > 0
    # A tiny smoke-test model has negligible real memory pressure on any
    # real GPU worth running this suite on -- the honest, correct
    # conclusion here is "no intervention needed", the same real
    # conclusion each individual mechanism's own real tests reach.
    assert plan.fits_without_intervention is True


@_REAL_ML_SMOKE
def test_real_placement_plan_runs_real_experiments_when_baseline_does_not_fit(tmp_path):
    """Artificially constrains available VRAM so the baseline estimate
    (from a real dry run) does not fit, forcing the real combination
    search to actually run all three real experiments."""
    from chowder.hardware import detect_hardware
    from chowder.project_runner import hardware_profile_from_snapshot

    hardware = hardware_profile_from_snapshot(detect_hardware(str(tmp_path)))
    if hardware.accelerator_vram_gb == () and hardware.vram_gb <= 0:
        pytest.skip("no CUDA device available for a real placement-plan comparison")
    tiny_hardware = HardwareProfile(
        0.05, hardware.ram_gb, hardware.nvme_gb, hardware.pcie_gbps,
        hardware.ram_gbps, hardware.nvme_gbps, accelerator_vram_gb=(0.05,),
    )
    context = ExecutionContext(tiny_hardware, str(tmp_path), seed=7)
    config = {
        "backend": {
            "type": "transformers-peft",
            "base_model": _TINY_MODEL,
            "dataset": "unused.jsonl",
            "max_length": 64,
            "quantization": "none",
            "precision": "fp32",
            "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
            "training": {"batch_size": 2},
        }
    }
    plan = build_placement_plan(resolved_config=config, context=context, work_dir=str(tmp_path))
    assert plan.fits_without_intervention is False
    assert plan.ddp_active is False
    assert plan.predicted_combined_estimate_gb is not None
    assert len(plan.reasoning) >= 2
