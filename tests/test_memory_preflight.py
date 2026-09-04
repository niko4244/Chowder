from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from chowder.backends.transformers_peft import TransformersPeftRunSpec
from chowder.executors import ExecutionContext
from chowder.memory import HardwareProfile
from chowder.memory_preflight import (
    _ALLOWED_MEMORY_PREFLIGHT_MODES,
    _calibration_key,
    _hardware_signature,
    _read_cache_entry,
    _recommendations,
    _resolve_memory_preflight_policy,
    _write_cache_entry,
    estimate_memory_requirements,
    resolve_memory_fit,
)

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


def _context(hardware=None):
    return ExecutionContext(hardware or _hardware(), ".", seed=1)


def _spec(**overrides):
    fields = dict(
        base_model="org/model",
        dataset="train.jsonl",
        output_dir="/tmp/out",
        max_length=64,
        quantization="none",
        precision="fp32",
        lora_r=4,
        lora_alpha=8,
        target_modules=("q_proj", "v_proj"),
        batch_size=1,
        gradient_checkpointing=True,
    )
    fields.update(overrides)
    return TransformersPeftRunSpec(**fields)


# --- hardware signature / calibration key ------------------------------------


def test_hardware_signature_sorts_pools_deterministically():
    a = _hardware_signature(_context(_hardware(pools=(16.0, 8.0))))
    b = _hardware_signature(_context(_hardware(pools=(8.0, 16.0))))
    assert a == b == "8.00|16.00"


def test_hardware_signature_falls_back_to_cpu_with_no_pools():
    context = _context(HardwareProfile(0.0, 64.0, 500.0, 12.0, 40.0, 3.0))
    assert _hardware_signature(context) == "cpu"


def test_calibration_key_is_stable_for_identical_inputs():
    spec = _spec()
    context = _context()
    assert _calibration_key(spec=spec, context=context) == _calibration_key(spec=spec, context=context)


def test_calibration_key_changes_when_base_model_changes():
    context = _context()
    a = _calibration_key(spec=_spec(base_model="org/model-a"), context=context)
    b = _calibration_key(spec=_spec(base_model="org/model-b"), context=context)
    assert a != b


def test_calibration_key_changes_when_hardware_changes():
    spec = _spec()
    a = _calibration_key(spec=spec, context=_context(_hardware(pools=(16.0,))))
    b = _calibration_key(spec=spec, context=_context(_hardware(vram_gb=8.0, pools=(8.0,))))
    assert a != b


def test_calibration_key_is_unaffected_by_batch_size():
    """batch_size only affects the extrapolation the caller does afterward,
    not what the worker measures -- two configs differing only in
    batch_size should share a cache entry rather than each paying for a
    separate real dry run."""
    context = _context()
    a = _calibration_key(spec=_spec(batch_size=1), context=context)
    b = _calibration_key(spec=_spec(batch_size=4), context=context)
    assert a == b


# --- cache read/write ---------------------------------------------------------


def test_write_then_read_cache_entry_round_trips(tmp_path):
    _write_cache_entry(tmp_path, "key1", {"peak_vram_gb_bs1": 1.0})
    entry = _read_cache_entry(tmp_path, "key1")
    assert entry == {"peak_vram_gb_bs1": 1.0}


def test_read_cache_entry_returns_none_when_file_missing(tmp_path):
    assert _read_cache_entry(tmp_path, "key1") is None


def test_read_cache_entry_returns_none_for_unknown_key(tmp_path):
    _write_cache_entry(tmp_path, "key1", {"a": 1})
    assert _read_cache_entry(tmp_path, "key2") is None


def test_write_cache_entry_preserves_other_keys(tmp_path):
    _write_cache_entry(tmp_path, "key1", {"a": 1})
    _write_cache_entry(tmp_path, "key2", {"b": 2})
    assert _read_cache_entry(tmp_path, "key1") == {"a": 1}
    assert _read_cache_entry(tmp_path, "key2") == {"b": 2}


def test_read_cache_entry_tolerates_malformed_json(tmp_path):
    cache_path = tmp_path / ".chowder" / "memory_calibration.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("not json{{{", encoding="utf-8")
    assert _read_cache_entry(tmp_path, "key1") is None


# --- recommendations ----------------------------------------------------------


def test_recommendations_empty_when_it_fits():
    assert _recommendations(estimate_gb=1.0, available_gb=16.0, spec=_spec()) == ()


def test_recommendations_suggest_smaller_batch_when_batch_size_over_one():
    recs = _recommendations(estimate_gb=20.0, available_gb=16.0, spec=_spec(batch_size=4))
    assert any("batch_size" in r for r in recs)


def test_recommendations_omit_batch_size_suggestion_at_batch_size_one():
    recs = _recommendations(estimate_gb=20.0, available_gb=16.0, spec=_spec(batch_size=1))
    assert not any("batch_size" in r for r in recs)


def test_recommendations_suggest_gradient_checkpointing_when_disabled():
    recs = _recommendations(
        estimate_gb=20.0, available_gb=16.0, spec=_spec(gradient_checkpointing=False)
    )
    assert any("gradient_checkpointing" in r for r in recs)


def test_recommendations_omit_gradient_checkpointing_when_already_enabled():
    recs = _recommendations(
        estimate_gb=20.0, available_gb=16.0, spec=_spec(gradient_checkpointing=True)
    )
    assert not any("gradient_checkpointing" in r for r in recs)


def test_recommendations_suggest_quantization_when_not_already_4bit():
    recs = _recommendations(estimate_gb=20.0, available_gb=16.0, spec=_spec(quantization="none"))
    assert any("4bit" in r for r in recs)


def test_recommendations_omit_quantization_suggestion_when_already_4bit():
    recs = _recommendations(estimate_gb=20.0, available_gb=16.0, spec=_spec(quantization="4bit"))
    assert not any("quantization" in r for r in recs)


def test_recommendations_always_state_the_overage():
    recs = _recommendations(estimate_gb=20.0, available_gb=16.0, spec=_spec())
    assert any("4.00 GB" in r for r in recs)


# --- estimate_memory_requirements orchestration (mocked worker) --------------


def _fake_measured(**overrides):
    measured = {
        "device": "cuda",
        "frozen_params": 1000,
        "trainable_params": 10,
        "max_length": 64,
        "peak_vram_gb_bs1": 1.0,
        "peak_vram_gb_bs2": 1.5,
    }
    measured.update(overrides)
    return measured


def test_estimate_extrapolates_linearly_from_two_real_measurements(tmp_path):
    with patch(
        "chowder.memory_preflight._run_dry_run_worker", return_value=_fake_measured()
    ) as mock_worker:
        config = _config(training={"batch_size": 5})
        context = _context()
        estimate = estimate_memory_requirements(
            resolved_config=config, context=context, work_dir=tmp_path, use_cache=False
        )
    mock_worker.assert_called_once()
    # per_example = 1.5 - 1.0 = 0.5; estimated = 1.0 + 0.5 * (5 - 1) = 3.0
    assert estimate.per_example_activation_gb == pytest.approx(0.5)
    assert estimate.estimated_peak_gb == pytest.approx(3.0)
    assert estimate.configured_batch_size == 5
    assert estimate.from_cache is False


def test_estimate_prefers_a_direct_measurement_over_extrapolation(tmp_path):
    """A real, measured finding: extrapolating linearly from two tiny
    points to a much larger configured batch size can be badly wrong
    (confirmed directly against real hardware -- see MemoryEstimate's own
    docstring). When the worker took a real, direct measurement at the
    configured batch size, it must be used as-is, not overridden by the
    extrapolation formula."""
    with patch(
        "chowder.memory_preflight._run_dry_run_worker",
        return_value=_fake_measured(
            peak_vram_gb_bs1=1.0, peak_vram_gb_bs2=1.5,
            peak_vram_gb_at_configured_batch_size=9.5,  # real number, not 1.0 + 0.5*4=3.0
        ),
    ) as mock_worker:
        config = _config(training={"batch_size": 5})
        estimate = estimate_memory_requirements(
            resolved_config=config, context=_context(), work_dir=tmp_path, use_cache=False
        )
    mock_worker.assert_called_once()
    assert estimate.estimated_peak_gb == pytest.approx(9.5)
    assert estimate.measured_peak_gb_at_configured_batch_size == pytest.approx(9.5)
    assert estimate.configured_batch_size_confirmed_oom is False


def test_estimate_falls_back_to_extrapolation_when_no_direct_measurement_present(tmp_path):
    """An older cache entry (written before the direct-measurement fix
    existed) has neither new key -- must not KeyError, must fall back to
    the pre-existing linear-extrapolation behavior exactly."""
    with patch(
        "chowder.memory_preflight._run_dry_run_worker", return_value=_fake_measured()
    ):
        config = _config(training={"batch_size": 5})
        estimate = estimate_memory_requirements(
            resolved_config=config, context=_context(), work_dir=tmp_path, use_cache=False
        )
    assert estimate.estimated_peak_gb == pytest.approx(3.0)
    assert estimate.measured_peak_gb_at_configured_batch_size is None
    assert estimate.configured_batch_size_confirmed_oom is False


def test_estimate_confirmed_oom_at_configured_batch_size_forces_does_not_fit(tmp_path):
    """A real, direct measurement at the configured batch size genuinely
    CUDA-OOM'd -- this must force fits=False unconditionally, even if the
    (necessarily incomplete) extrapolation would have technically guessed
    it fits."""
    with patch(
        "chowder.memory_preflight._run_dry_run_worker",
        return_value=_fake_measured(
            peak_vram_gb_bs1=0.1, peak_vram_gb_bs2=0.1,  # extrapolation alone would say "fits"
            peak_vram_gb_at_configured_batch_size=None,
            configured_batch_size_oom=True,
        ),
    ):
        config = _config(training={"batch_size": 5})
        context = _context(_hardware(vram_gb=16.0, pools=(16.0,)))
        estimate = estimate_memory_requirements(
            resolved_config=config, context=context, work_dir=tmp_path, use_cache=False
        )
    assert estimate.fits is False
    assert estimate.configured_batch_size_confirmed_oom is True


def test_estimate_at_batch_size_one_uses_the_bs1_measurement_directly(tmp_path):
    with patch("chowder.memory_preflight._run_dry_run_worker", return_value=_fake_measured()):
        config = _config(training={"batch_size": 1})
        estimate = estimate_memory_requirements(
            resolved_config=config, context=_context(), work_dir=tmp_path, use_cache=False
        )
    assert estimate.estimated_peak_gb == pytest.approx(1.0)


def test_estimate_uses_the_cache_on_a_second_call_with_the_same_key(tmp_path):
    config = _config(training={"batch_size": 2})
    context = _context()
    with patch(
        "chowder.memory_preflight._run_dry_run_worker", return_value=_fake_measured()
    ) as mock_worker:
        first = estimate_memory_requirements(
            resolved_config=config, context=context, work_dir=tmp_path
        )
        second = estimate_memory_requirements(
            resolved_config=config, context=context, work_dir=tmp_path
        )
    mock_worker.assert_called_once()  # only the first call actually ran the worker
    assert first.from_cache is False
    assert second.from_cache is True
    assert second.estimated_peak_gb == first.estimated_peak_gb


def test_estimate_use_cache_false_always_calls_the_worker(tmp_path):
    config = _config()
    context = _context()
    with patch(
        "chowder.memory_preflight._run_dry_run_worker", return_value=_fake_measured()
    ) as mock_worker:
        estimate_memory_requirements(
            resolved_config=config, context=context, work_dir=tmp_path, use_cache=False
        )
        estimate_memory_requirements(
            resolved_config=config, context=context, work_dir=tmp_path, use_cache=False
        )
    assert mock_worker.call_count == 2


def test_estimate_reports_fits_true_when_within_available_vram(tmp_path):
    with patch(
        "chowder.memory_preflight._run_dry_run_worker",
        return_value=_fake_measured(peak_vram_gb_bs1=1.0, peak_vram_gb_bs2=1.2),
    ):
        config = _config(training={"batch_size": 1})
        estimate = estimate_memory_requirements(
            resolved_config=config,
            context=_context(_hardware(vram_gb=16.0, pools=(16.0,))),
            work_dir=tmp_path,
            use_cache=False,
        )
    assert estimate.fits is True
    assert estimate.recommendations == ()


def test_estimate_reports_fits_false_and_recommendations_when_over_capacity(tmp_path):
    with patch(
        "chowder.memory_preflight._run_dry_run_worker",
        return_value=_fake_measured(peak_vram_gb_bs1=10.0, peak_vram_gb_bs2=12.0),
    ):
        config = _config(training={"batch_size": 4})
        estimate = estimate_memory_requirements(
            resolved_config=config,
            context=_context(_hardware(vram_gb=8.0, pools=(8.0,))),
            work_dir=tmp_path,
            use_cache=False,
        )
    assert estimate.fits is False
    assert len(estimate.recommendations) > 0


def test_estimate_per_rank_available_uses_the_smallest_accelerator_pool(tmp_path):
    """Under DDP each rank holds a full replica -- comparing against the
    aggregate (sum) of multiple pools would be the wrong number."""
    with patch("chowder.memory_preflight._run_dry_run_worker", return_value=_fake_measured()):
        config = _config()
        estimate = estimate_memory_requirements(
            resolved_config=config,
            context=_context(_hardware(vram_gb=8.0, pools=(16.0, 8.0))),
            work_dir=tmp_path,
            use_cache=False,
        )
    assert estimate.per_rank_available_gb == 8.0


def test_estimate_cpu_device_always_fits(tmp_path):
    """No real VRAM figure exists on CPU -- must never be reported as not
    fitting just because the measured GB values happen to be 0.0."""
    with patch(
        "chowder.memory_preflight._run_dry_run_worker",
        return_value=_fake_measured(device="cpu", peak_vram_gb_bs1=0.0, peak_vram_gb_bs2=0.0),
    ):
        config = _config()
        estimate = estimate_memory_requirements(
            resolved_config=config,
            context=_context(HardwareProfile(0.0, 64.0, 500.0, 12.0, 40.0, 3.0)),
            work_dir=tmp_path,
            use_cache=False,
        )
    assert estimate.device == "cpu"
    assert estimate.fits is True


# --- memory_preflight policy (auto | always | cached | off) -----------------


def test_resolve_memory_preflight_policy_defaults_to_off_when_unset():
    """An existing project.json with no memory_preflight key must train
    exactly as it always has -- this check did not exist at all before
    this policy was added (estimate_memory_requirements was previously
    only reachable from the TUI's manual button), so anything other than
    "off" as the default would newly reject candidates a config never
    opted into being rejected by."""
    assert _resolve_memory_preflight_policy({}) == "off"


def test_resolve_memory_preflight_policy_accepts_each_allowed_mode():
    for mode in _ALLOWED_MEMORY_PREFLIGHT_MODES:
        assert _resolve_memory_preflight_policy({"memory_preflight": mode}) == mode


def test_resolve_memory_preflight_policy_is_case_insensitive():
    assert _resolve_memory_preflight_policy({"memory_preflight": "ALWAYS"}) == "always"


def test_resolve_memory_preflight_policy_rejects_unknown_values():
    with pytest.raises(ValueError, match="memory_preflight"):
        _resolve_memory_preflight_policy({"memory_preflight": "sometimes"})


def test_resolve_memory_fit_off_returns_none_without_calling_the_worker(tmp_path):
    with patch("chowder.memory_preflight._run_dry_run_worker") as mock_worker:
        result = resolve_memory_fit(
            resolved_config=_config(memory_preflight="off"), context=_context(), work_dir=tmp_path
        )
    mock_worker.assert_not_called()
    assert result is None


def test_resolve_memory_fit_always_ignores_an_existing_cache(tmp_path):
    config = _config(memory_preflight="always")
    context = _context()
    with patch(
        "chowder.memory_preflight._run_dry_run_worker", return_value=_fake_measured()
    ) as mock_worker:
        resolve_memory_fit(resolved_config=config, context=context, work_dir=tmp_path)
        resolve_memory_fit(resolved_config=config, context=context, work_dir=tmp_path)
    assert mock_worker.call_count == 2


def test_resolve_memory_fit_cached_reuses_an_existing_cache(tmp_path):
    config = _config(memory_preflight="cached")
    context = _context()
    with patch(
        "chowder.memory_preflight._run_dry_run_worker", return_value=_fake_measured()
    ) as mock_worker:
        first = resolve_memory_fit(resolved_config=config, context=context, work_dir=tmp_path)
        second = resolve_memory_fit(resolved_config=config, context=context, work_dir=tmp_path)
    mock_worker.assert_called_once()
    assert first is not None and first.from_cache is False
    assert second is not None and second.from_cache is True


def test_resolve_memory_fit_auto_trusts_a_cache_hit_with_comfortable_headroom(tmp_path):
    """Cached estimate using well under the pressure threshold of
    available VRAM -- auto must trust it, not pay for a fresh
    measurement."""
    config = _config(memory_preflight="auto", training={"batch_size": 1})
    context = _context(_hardware(vram_gb=16.0, pools=(16.0,)))
    with patch(
        "chowder.memory_preflight._run_dry_run_worker",
        return_value=_fake_measured(peak_vram_gb_bs1=1.0, peak_vram_gb_bs2=1.5),
    ) as mock_worker:
        first = resolve_memory_fit(resolved_config=config, context=context, work_dir=tmp_path)
        second = resolve_memory_fit(resolved_config=config, context=context, work_dir=tmp_path)
    mock_worker.assert_called_once()  # only the first call (the cache miss) ran the worker
    assert first is not None and second is not None
    assert second.from_cache is True


def test_resolve_memory_fit_auto_refreshes_a_cache_hit_under_pressure(tmp_path):
    """Cached estimate using nearly all of available VRAM -- auto must
    not trust a possibly-stale number that close to the fit boundary,
    and pays for a fresh measurement instead."""
    config = _config(memory_preflight="auto", training={"batch_size": 1})
    # available=16.0, estimated_peak=15.0 -> 15.0 / 16.0 = 93.75%, over the
    # 90% pressure threshold.
    context = _context(_hardware(vram_gb=16.0, pools=(16.0,)))
    with patch(
        "chowder.memory_preflight._run_dry_run_worker",
        return_value=_fake_measured(peak_vram_gb_bs1=15.0, peak_vram_gb_bs2=15.0),
    ) as mock_worker:
        resolve_memory_fit(resolved_config=config, context=context, work_dir=tmp_path)
        resolve_memory_fit(resolved_config=config, context=context, work_dir=tmp_path)
    # First call: cache miss, real measurement. Second call: cache hit,
    # but under pressure -- another real measurement, not trusted as-is.
    assert mock_worker.call_count == 2


def test_resolve_memory_fit_auto_measures_fresh_on_a_cache_miss(tmp_path):
    config = _config(memory_preflight="auto")
    context = _context()
    with patch(
        "chowder.memory_preflight._run_dry_run_worker", return_value=_fake_measured()
    ) as mock_worker:
        result = resolve_memory_fit(resolved_config=config, context=context, work_dir=tmp_path)
    mock_worker.assert_called_once()
    assert result is not None and result.from_cache is False


# --- real end-to-end (actual model load + forward/backward in a subprocess) --


@_REAL_ML_SMOKE
def test_real_dry_run_measures_actual_peak_memory_for_a_tiny_model():
    from chowder.backends.memory_preflight_worker import dry_run

    spec = _spec(base_model=_TINY_MODEL, dataset="unused.jsonl", max_length=32)
    result = dry_run(spec)

    assert result["device"] in ("cuda", "cpu")
    assert result["frozen_params"] > 0
    assert result["trainable_params"] > 0
    assert result["max_length"] == 32
    # A batch of 2 must never measure less real memory than a batch of 1.
    assert result["peak_vram_gb_bs2"] >= result["peak_vram_gb_bs1"]


@_REAL_ML_SMOKE
def test_real_estimate_memory_requirements_spawns_worker_and_caches(tmp_path):
    from chowder.hardware import detect_hardware
    from chowder.project_runner import hardware_profile_from_snapshot

    hardware = hardware_profile_from_snapshot(detect_hardware(str(tmp_path)))
    context = ExecutionContext(hardware, str(tmp_path), seed=7)
    config = {
        "backend": {
            "type": "transformers-peft",
            "base_model": _TINY_MODEL,
            "dataset": "unused.jsonl",
            "max_length": 32,
            "quantization": "none",
            "precision": "fp32",
            "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
            "training": {"batch_size": 2},
        }
    }

    first = estimate_memory_requirements(resolved_config=config, context=context, work_dir=tmp_path)
    assert first.from_cache is False
    assert first.frozen_params > 0
    assert first.measured_peak_gb_at_batch_2 >= first.measured_peak_gb_at_batch_1
    assert first.fits is True  # a few-hundred-KB tiny model always fits real hardware

    second = estimate_memory_requirements(resolved_config=config, context=context, work_dir=tmp_path)
    assert second.from_cache is True
    assert second.estimated_peak_gb == first.estimated_peak_gb
    assert second.frozen_params == first.frozen_params


@_REAL_ML_SMOKE
def test_real_dry_run_measures_directly_at_a_configured_batch_size_other_than_one_or_two():
    """The real fix this module's own docstring documents: when the
    configured batch size differs from the two always-measured tiny
    points, dry_run() must take one more real measurement AT that batch
    size rather than leaving the caller to extrapolate from batch=1/2 --
    proven for real, not just asserted against a mocked worker."""
    from chowder.backends.memory_preflight_worker import dry_run

    spec = _spec(base_model=_TINY_MODEL, dataset="unused.jsonl", max_length=32, batch_size=4)
    result = dry_run(spec)

    assert result["peak_vram_gb_at_configured_batch_size"] is not None
    assert result["configured_batch_size_oom"] is False
    # The real, directly measured batch=4 peak must be at least as large as
    # the real batch=2 measurement -- more real activation memory, never less.
    assert result["peak_vram_gb_at_configured_batch_size"] >= result["peak_vram_gb_bs2"]


@_REAL_ML_SMOKE
def test_real_estimate_uses_the_real_direct_measurement_end_to_end(tmp_path):
    from chowder.hardware import detect_hardware
    from chowder.project_runner import hardware_profile_from_snapshot

    hardware = hardware_profile_from_snapshot(detect_hardware(str(tmp_path)))
    context = ExecutionContext(hardware, str(tmp_path), seed=7)
    config = {
        "backend": {
            "type": "transformers-peft",
            "base_model": _TINY_MODEL,
            "dataset": "unused.jsonl",
            "max_length": 32,
            "quantization": "none",
            "precision": "fp32",
            "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
            "training": {"batch_size": 4},
        }
    }

    estimate = estimate_memory_requirements(resolved_config=config, context=context, work_dir=tmp_path)
    assert estimate.measured_peak_gb_at_configured_batch_size is not None
    assert estimate.configured_batch_size_confirmed_oom is False
    assert estimate.estimated_peak_gb == pytest.approx(estimate.measured_peak_gb_at_configured_batch_size)
    assert estimate.fits is True  # a tiny model always fits real hardware
