"""P6: an unmeasured phase is unknown, never zero.

The completed GSM8K rerun cost 3.463 GPU-hours and its artifacts could not say
which part was load, which was the 500 optimizer steps, and which was
generation. These tests pin the accounting rules that make that question
answerable, and the refusals that stop an unmeasured run from *looking* cheap:

* ``None`` is a distinct value from ``0.0`` and is carried in ``unmeasured``.
* A phase required for a claim that was never measured refuses.
* A duration says whether it was synchronized (async accelerator work timed
  without synchronization measures submission, not work).
* A forecast states what it does not know instead of summing unknowns as zero.
"""

from __future__ import annotations

import math

import pytest

from chowder.lifecycle import (
    LIFECYCLE_PHASES,
    PHASE_BASELINE_GENERATION,
    PHASE_CANDIDATE_GENERATION,
    PHASE_CHECKPOINT_PUBLICATION,
    PHASE_CLOSEOUT,
    PHASE_FIRST_BACKWARD,
    PHASE_FIRST_FORWARD,
    PHASE_FIRST_UPDATE,
    PHASE_MODEL_LOAD,
    PHASE_RELOAD,
    PHASE_STEADY_STEPS,
    REQUIRED_FOR_EVALUATION,
    REQUIRED_FOR_TRAINING,
    LifecycleAccountingError,
    LifecycleForecast,
    LifecycleLedger,
    PhaseEstimate,
    PhaseTimer,
    cuda_synchronize,
    evaluation_lifecycle_ledger,
    quantization_reality_report,
    sampling_device,
    tensor_inventory,
    training_lifecycle_ledger,
)


# ---------------------------------------------------------------------------
# the ledger
# ---------------------------------------------------------------------------


def test_lifecycle_phases_are_declared_in_execution_order():
    assert LIFECYCLE_PHASES[0] == PHASE_MODEL_LOAD
    assert LIFECYCLE_PHASES[-1] == PHASE_CLOSEOUT
    assert PHASE_STEADY_STEPS in LIFECYCLE_PHASES
    assert PHASE_BASELINE_GENERATION in LIFECYCLE_PHASES
    assert PHASE_CANDIDATE_GENERATION in LIFECYCLE_PHASES


def test_unmeasured_phase_is_unknown_not_zero():
    ledger = LifecycleLedger(accelerator_count=1)
    ledger.record_unavailable(PHASE_RELOAD, "not applicable on a fresh run")
    measured = ledger.record(PHASE_MODEL_LOAD, 12.5)

    assert measured.measured is True
    assert measured.seconds == pytest.approx(12.5)
    assert ledger.phases[PHASE_RELOAD].measured is False
    assert ledger.phases[PHASE_RELOAD].seconds is None
    # The distinction that matters: the total is the measured part only, and the
    # unknown part is still named.
    assert ledger.measured_seconds == pytest.approx(12.5)
    assert ledger.measured_gpu_hours == pytest.approx(12.5 / 3600.0)
    assert ledger.unmeasured == {PHASE_RELOAD: "not applicable on a fresh run"}


def test_measured_zero_is_a_measurement_and_not_an_unknown():
    ledger = LifecycleLedger(accelerator_count=1)
    ledger.record(PHASE_RELOAD, 0.0)
    assert ledger.phases[PHASE_RELOAD].measured is True
    assert ledger.unmeasured == {}
    ledger.require([PHASE_RELOAD])


def test_gpu_hours_scale_with_the_accelerator_count():
    ledger = LifecycleLedger(accelerator_count=2)
    ledger.record(PHASE_STEADY_STEPS, 1800.0)
    assert ledger.measured_gpu_hours == pytest.approx(1.0)
    assert ledger.phase_gpu_hours()[PHASE_STEADY_STEPS] == pytest.approx(1.0)


def test_a_phase_can_be_charged_to_a_different_accelerator_count():
    ledger = LifecycleLedger(accelerator_count=1)
    measurement = ledger.record(PHASE_BASELINE_GENERATION, 3600.0, accelerator_count=0)
    assert measurement.gpu_hours == pytest.approx(0.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_nonfinite_and_negative_durations_are_refused(value):
    ledger = LifecycleLedger(accelerator_count=1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        ledger.record(PHASE_MODEL_LOAD, value)


def test_accelerator_count_must_be_a_real_non_negative_integer():
    with pytest.raises(TypeError):
        LifecycleLedger(accelerator_count=True)  # bool is not an accelerator count
    with pytest.raises(ValueError):
        LifecycleLedger(accelerator_count=-1)


def test_require_refuses_and_names_what_was_never_measured():
    ledger = LifecycleLedger(accelerator_count=1)
    ledger.record(PHASE_MODEL_LOAD, 3.0)
    ledger.record_unavailable(PHASE_STEADY_STEPS, "worker died before the first step")

    with pytest.raises(LifecycleAccountingError) as excinfo:
        ledger.require(REQUIRED_FOR_TRAINING, purpose="training qualification")

    message = str(excinfo.value)
    assert PHASE_STEADY_STEPS in message
    assert "worker died before the first step" in message
    assert "UNKNOWN" in message


def test_require_passes_once_the_required_phases_are_measured():
    ledger = LifecycleLedger(accelerator_count=1)
    ledger.record(PHASE_MODEL_LOAD, 3.0)
    ledger.record(PHASE_STEADY_STEPS, 300.0)
    ledger.require(REQUIRED_FOR_TRAINING)


def test_evaluation_qualification_requires_both_generation_arms():
    ledger = LifecycleLedger(accelerator_count=1)
    ledger.record(PHASE_BASELINE_GENERATION, 100.0)
    with pytest.raises(LifecycleAccountingError) as excinfo:
        ledger.require(REQUIRED_FOR_EVALUATION, purpose="evaluation")
    assert PHASE_CANDIDATE_GENERATION in str(excinfo.value)

    ledger.record(PHASE_CANDIDATE_GENERATION, 120.0)
    ledger.require(REQUIRED_FOR_EVALUATION)


def test_assert_within_budget_reports_the_comparison_and_refuses_overflow():
    ledger = LifecycleLedger(accelerator_count=1)
    ledger.record(PHASE_MODEL_LOAD, 60.0)
    ledger.record(PHASE_STEADY_STEPS, 3540.0)

    report = ledger.assert_within_budget(1.0, required=REQUIRED_FOR_TRAINING)
    assert report["within_budget"] is True
    assert report["measured_gpu_hours"] == pytest.approx(1.0)
    assert report["unmeasured"] == {}

    with pytest.raises(LifecycleAccountingError, match="exceeds the"):
        ledger.assert_within_budget(0.5)


def test_an_unknown_required_phase_refuses_before_the_budget_comparison():
    ledger = LifecycleLedger(accelerator_count=1)
    ledger.record(PHASE_MODEL_LOAD, 1.0)
    # A total that silently omits the unmeasured steps is not a smaller cost.
    with pytest.raises(LifecycleAccountingError, match="UNKNOWN, not zero"):
        ledger.assert_within_budget(1000.0, required=REQUIRED_FOR_TRAINING)


@pytest.mark.parametrize("budget", [float("nan"), float("inf"), -1.0])
def test_a_nonfinite_budget_is_refused(budget):
    ledger = LifecycleLedger(accelerator_count=1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        ledger.assert_within_budget(budget)


def test_estimator_versus_actual_is_durable_and_unknown_is_not_zero():
    ledger = LifecycleLedger(accelerator_count=1)
    ledger.record(PHASE_MODEL_LOAD, 30.0)
    ledger.record_unavailable(PHASE_RELOAD, "no reload on this path")

    rows = ledger.compare_to_estimate(
        {PHASE_MODEL_LOAD: 25.0, PHASE_RELOAD: 10.0, PHASE_CLOSEOUT: 1.0}
    )

    assert rows[PHASE_MODEL_LOAD]["state"] == "diverged"
    assert rows[PHASE_MODEL_LOAD]["delta_seconds"] == pytest.approx(5.0)
    # Measured but never estimated: the difference is unknown, not "no change".
    assert rows[PHASE_RELOAD]["delta_seconds"] is None
    assert rows[PHASE_RELOAD]["state"] == "unknown"
    # Estimated but never measured: same rule in the other direction.
    assert rows[PHASE_CLOSEOUT]["measured_seconds"] is None
    assert rows[PHASE_CLOSEOUT]["delta_seconds"] is None


def test_ledger_serializes_unknown_as_none_not_zero():
    ledger = LifecycleLedger(accelerator_count=1)
    ledger.record_unavailable(PHASE_CANDIDATE_GENERATION, "not run")
    payload = ledger.to_dict()
    assert payload["phases"][PHASE_CANDIDATE_GENERATION]["seconds"] is None
    assert payload["phases"][PHASE_CANDIDATE_GENERATION]["measured"] is False
    assert payload["phases"][PHASE_CANDIDATE_GENERATION]["gpu_hours"] is None


# ---------------------------------------------------------------------------
# the timer
# ---------------------------------------------------------------------------


def test_phase_timer_measures_and_records_sync_overhead_separately():
    calls: list[str] = []

    def synchronize() -> None:
        calls.append("sync")

    timer = PhaseTimer(synchronize=synchronize)
    with timer:
        pass

    assert calls == ["sync", "sync"]
    ledger = LifecycleLedger(accelerator_count=1)
    measurement = ledger.record_timer(PHASE_STEADY_STEPS, timer)
    assert timer.seconds is not None and timer.seconds >= 0.0
    assert measurement.synchronized is True
    assert measurement.sync_overhead_seconds >= 0.0
    # The instrumentation's own cost is stated rather than hidden inside seconds.
    assert measurement.to_dict()["sync_overhead_seconds"] is not None


def test_an_unsynchronized_timer_says_so():
    timer = PhaseTimer()
    with timer:
        pass
    # None, not False: "we did not synchronize" and "synchronization failed"
    # are different facts, and neither may be read as a synchronized number.
    assert timer.synchronized is None
    ledger = LifecycleLedger(accelerator_count=1)
    assert ledger.record_timer(PHASE_STEADY_STEPS, timer).synchronized is None


def test_a_failing_synchronize_never_kills_the_measurement():
    def synchronize() -> None:
        raise RuntimeError("no CUDA context")

    timer = PhaseTimer(synchronize=synchronize)
    with timer:
        pass

    assert timer.seconds is not None
    assert timer.synchronized is False
    assert "no CUDA context" in (timer.sync_failure or "")

    ledger = LifecycleLedger(accelerator_count=1)
    measurement = ledger.record_timer(PHASE_MODEL_LOAD, timer)
    assert measurement.measured is True
    assert measurement.synchronized is False
    assert "no CUDA context" in (measurement.note or "")


# ---------------------------------------------------------------------------
# what is actually loaded
# ---------------------------------------------------------------------------


class _Param:
    def __init__(self, dtype, numel, *, device="cuda:0", requires_grad=False):
        self.dtype = dtype
        self._numel = numel
        self.device = device
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._numel


class _FakeModel:
    def __init__(self, parameters, config=None):
        self._parameters = parameters
        self.config = config

    def named_parameters(self):
        return list(self._parameters.items())


def test_tensor_inventory_reports_storage_dtype_not_the_request():
    model = _FakeModel(
        {
            "model.embed_tokens.weight": _Param("torch.uint8", 1000),
            "model.layers.0.mlp.experts.gate_up_proj": _Param("torch.bfloat16", 2000),
            "model.layers.0.mlp.gate.weight": _Param(
                "torch.bfloat16", 8, requires_grad=True
            ),
        }
    )
    inventory = tensor_inventory(model)

    assert inventory["total_elements"] == 3008
    assert inventory["elements_by_dtype"] == {"uint8": 1000, "bfloat16": 2008}
    assert inventory["trainable_elements"] == 8
    assert inventory["frozen_elements"] == 3000
    # int8/uint8 storage packs two 4-bit values per byte; an element count read
    # as "one value per element" would misreport the model's real size.
    assert inventory["packed_quantized_storage_elements"] == 1000
    assert "PACKED" in inventory["packed_quantized_note"]


def test_tensor_inventory_reports_unavailable_quantization_metadata_honestly():
    inventory = tensor_inventory(_FakeModel({"w": _Param("torch.float32", 4)}))

    assert inventory["quantization"]["declared_available"] is False
    assert inventory["quantization"]["declared"] is None
    assert inventory["packed_quantized_storage_elements"] == 0
    assert inventory["packed_quantized_note"] is None


def test_quantization_reality_exposes_raw_experts_left_unquantized():
    # The audit's shape: a 4-bit loader that skips the raw expert parameters.
    model = _FakeModel(
        {
            "model.layers.0.mlp.experts.gate_up_proj": _Param("torch.bfloat16", 500),
            "model.layers.0.self_attn.q_proj.weight": _Param("torch.uint8", 250),
        }
    )
    report = quantization_reality_report(model, requested="4bit")

    assert report["matches_request"] is False
    assert report["expert_elements_by_dtype"] == {"bfloat16": 500}
    assert "bfloat16" in (report["note"] or "")
    assert "4-bit loader" in (report["note"] or "")


def test_quantization_reality_matches_when_everything_is_packed():
    model = _FakeModel({"w": _Param("torch.uint8", 10)})
    report = quantization_reality_report(model, requested="4bit")
    assert report["matches_request"] is True
    assert report["note"] is None


def test_quantization_reality_flags_unpacked_storage_when_none_was_requested():
    model = _FakeModel({"w": _Param("torch.uint8", 10)})
    report = quantization_reality_report(model, requested="none")
    assert report["matches_request"] is False


def test_full_precision_request_satisfied_by_full_precision_storage():
    model = _FakeModel({"w": _Param("torch.bfloat16", 10)})
    report = quantization_reality_report(model, requested="none")
    assert report["matches_request"] is True


# ---------------------------------------------------------------------------
# the forecast: what it does not know, stated
# ---------------------------------------------------------------------------


def test_forecast_keeps_unknown_phases_out_of_the_known_total():
    forecast = LifecycleForecast.from_terms(
        accelerator_count=1,
        terms={
            PHASE_MODEL_LOAD: (60.0, "measured", "from the previous run's ledger"),
            PHASE_STEADY_STEPS: (3600.0, "derived", "500 steps x 7.2s"),
            PHASE_CANDIDATE_GENERATION: (None, "unknown", "no measured generation profile"),
        },
    )

    assert forecast.known_gpu_hours == pytest.approx(1.0 + 60.0 / 3600.0)
    assert forecast.unknown_phases == (PHASE_CANDIDATE_GENERATION,)
    payload = forecast.to_dict()
    assert payload["phases"][PHASE_CANDIDATE_GENERATION]["seconds"] is None
    assert payload["phases"][PHASE_CANDIDATE_GENERATION]["basis"] == "unknown"


def test_forecast_refuses_a_claim_over_a_required_unknown():
    forecast = LifecycleForecast.from_terms(
        accelerator_count=1,
        terms={
            PHASE_MODEL_LOAD: (10.0, "measured", None),
            PHASE_STEADY_STEPS: (100.0, "derived", None),
            PHASE_BASELINE_GENERATION: (None, "unknown", "no evaluation estimate"),
            PHASE_CANDIDATE_GENERATION: (None, "unknown", "no evaluation estimate"),
        },
    )
    forecast.require_estimated(REQUIRED_FOR_TRAINING)
    with pytest.raises(LifecycleAccountingError) as excinfo:
        forecast.require_estimated(REQUIRED_FOR_EVALUATION, purpose="reservation")
    assert "independent evaluation" in str(excinfo.value) or "generation" in str(excinfo.value)


def test_forecast_confidence_reflects_the_weakest_basis_it_used():
    weak = LifecycleForecast.from_terms(
        accelerator_count=1,
        terms={
            PHASE_MODEL_LOAD: (10.0, "measured", None),
            PHASE_STEADY_STEPS: (100.0, "declared", None),
        },
    )
    strong = LifecycleForecast.from_terms(
        accelerator_count=1,
        terms={
            PHASE_MODEL_LOAD: (10.0, "measured", None),
            PHASE_STEADY_STEPS: (100.0, "measured", None),
        },
    )
    assert weak.confidence < strong.confidence
    assert strong.confidence == pytest.approx(1.0)
    # An unknown required phase cannot carry a confident forecast at all.
    assert weak.confidence_for((PHASE_MODEL_LOAD, PHASE_CANDIDATE_GENERATION)) == 0.0


def test_forecast_rejects_unlabelled_or_nonfinite_terms():
    with pytest.raises(ValueError, match="basis"):
        LifecycleForecast.from_terms(
            accelerator_count=1, terms={PHASE_MODEL_LOAD: (10.0, "vibes", None)}
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        LifecycleForecast.from_terms(
            accelerator_count=1, terms={PHASE_MODEL_LOAD: (-1.0, "measured", None)}
        )


def test_forecast_versus_actual_is_a_durable_comparison():
    forecast = LifecycleForecast.from_terms(
        accelerator_count=1,
        terms={
            PHASE_MODEL_LOAD: (60.0, "measured", None),
            PHASE_STEADY_STEPS: (3600.0, "derived", None),
            PHASE_CANDIDATE_GENERATION: (None, "unknown", "no profile"),
        },
    )
    ledger = LifecycleLedger(accelerator_count=1)
    ledger.record(PHASE_MODEL_LOAD, 90.0)
    ledger.record(PHASE_STEADY_STEPS, 1800.0)
    ledger.record(PHASE_CANDIDATE_GENERATION, 400.0)

    rows = forecast.compare_to(ledger)
    assert rows[PHASE_MODEL_LOAD]["state"] == "diverged"
    assert rows[PHASE_MODEL_LOAD]["delta_seconds"] == pytest.approx(30.0)
    # Estimated-unknown but measured: the phase that broke the earlier forecast
    # is the one the record has to keep visible.
    assert rows[PHASE_CANDIDATE_GENERATION]["estimated_seconds"] is None
    assert rows[PHASE_CANDIDATE_GENERATION]["measured_seconds"] == pytest.approx(400.0)
    assert rows[PHASE_CANDIDATE_GENERATION]["state"] == "unknown"


def test_phase_estimate_refuses_a_basis_it_cannot_justify():
    with pytest.raises(ValueError, match="basis"):
        PhaseEstimate(phase=PHASE_MODEL_LOAD, seconds=1.0, basis="probably")


def test_forecast_accelerator_hours_use_the_declared_count():
    forecast = LifecycleForecast.from_terms(
        accelerator_count=2,
        terms={PHASE_STEADY_STEPS: (1800.0, "measured", None)},
    )
    assert forecast.known_gpu_hours == pytest.approx(1.0)
    assert math.isclose(forecast.to_dict()["accelerator_count"], 2)


# ---------------------------------------------------------------------------
# what a training worker can and cannot measure
# ---------------------------------------------------------------------------


def test_training_ledger_records_the_phases_a_trainer_really_timed():
    load = PhaseTimer()
    with load:
        pass
    publication = PhaseTimer()
    with publication:
        pass

    ledger = training_lifecycle_ledger(
        accelerator_count=1,
        model_load=load,
        checkpoint_publication=publication,
        steady_state_steps_seconds=3600.0,
    )

    assert ledger.phases[PHASE_STEADY_STEPS].seconds == pytest.approx(3600.0)
    assert ledger.phases[PHASE_MODEL_LOAD].measured is True
    assert ledger.phases[PHASE_CHECKPOINT_PUBLICATION].measured is True
    ledger.require(REQUIRED_FOR_TRAINING)


def test_training_ledger_says_why_the_first_step_and_generation_are_unknown():
    ledger = training_lifecycle_ledger(
        accelerator_count=1,
        model_load=None,
        steady_state_steps_seconds=10.0,
        detailed_timing_enabled=False,
    )

    # A phase that is not measured because instrumenting it costs real wall time
    # says so, rather than being silently absent or silently zero.
    reason = ledger.unmeasured[PHASE_FIRST_FORWARD]
    assert "detailed_timing_telemetry" in reason
    assert PHASE_BASELINE_GENERATION in ledger.unmeasured
    assert ledger.phases[PHASE_STEADY_STEPS].seconds == pytest.approx(10.0)
    # The load was never timed at all -- still unknown, with its own reason.
    assert ledger.phases[PHASE_MODEL_LOAD].seconds is None


def test_training_ledger_records_the_first_step_when_detailed_timing_is_on():
    ledger = training_lifecycle_ledger(
        accelerator_count=1,
        model_load=None,
        steady_state_steps_seconds=60.0,
        detailed_timing_enabled=True,
        first_forward_seconds=3.0,
        first_backward_seconds=4.0,
        first_update_seconds=1.0,
    )

    assert ledger.phases[PHASE_FIRST_FORWARD].seconds == pytest.approx(3.0)
    assert ledger.phases[PHASE_FIRST_BACKWARD].seconds == pytest.approx(4.0)
    assert ledger.phases[PHASE_FIRST_UPDATE].seconds == pytest.approx(1.0)


def test_a_resumed_run_still_cannot_claim_a_separately_timed_reload():
    ledger = training_lifecycle_ledger(
        accelerator_count=1,
        model_load=None,
        steady_state_steps_seconds=5.0,
        resumed_from_checkpoint=True,
    )
    assert PHASE_RELOAD in ledger.unmeasured
    assert "Trainer" in ledger.unmeasured[PHASE_RELOAD]


def test_an_evaluation_ledger_measures_its_own_arm_and_names_the_other():
    baseline = evaluation_lifecycle_ledger(
        accelerator_count=1, arm="baseline", generation_seconds=120.0
    )
    assert baseline.phases[PHASE_BASELINE_GENERATION].seconds == pytest.approx(120.0)
    baseline.require([PHASE_BASELINE_GENERATION])
    assert PHASE_CANDIDATE_GENERATION in baseline.unmeasured
    with pytest.raises(LifecycleAccountingError):
        baseline.require(REQUIRED_FOR_EVALUATION)

    candidate = evaluation_lifecycle_ledger(
        accelerator_count=1, arm="candidate", generation_seconds=150.0
    )
    assert candidate.phases[PHASE_CANDIDATE_GENERATION].seconds == pytest.approx(150.0)
    assert PHASE_BASELINE_GENERATION in candidate.unmeasured


def test_an_evaluation_ledger_refuses_an_unknown_arm():
    with pytest.raises(ValueError, match="arm"):
        evaluation_lifecycle_ledger(accelerator_count=1, arm="vibes", generation_seconds=1.0)


def test_an_evaluation_ledger_can_report_an_unmeasured_generation_phase():
    ledger = evaluation_lifecycle_ledger(
        accelerator_count=0, arm="baseline", generation_seconds=None
    )
    assert ledger.phases[PHASE_BASELINE_GENERATION].seconds is None
    assert "not measured" in ledger.unmeasured[PHASE_BASELINE_GENERATION]


# ---------------------------------------------------------------------------
# the shared synchronization / device helpers every worker uses
# ---------------------------------------------------------------------------


class _FakeCuda:
    def __init__(self, available, index=0):
        self._available = available
        self._index = index

    def is_available(self):
        return self._available

    def current_device(self):
        return self._index

    def synchronize(self):
        return "synchronized"


class _FakeTorch:
    def __init__(self, available, index=0):
        self.cuda = _FakeCuda(available, index)


class _ExplodingTorch:
    @property
    def cuda(self):
        raise RuntimeError("no CUDA runtime")


def test_cuda_synchronize_is_none_without_a_device():
    # None, not a no-op: an unsynchronized duration must not look synchronized.
    assert cuda_synchronize(_FakeTorch(available=False)) is None
    assert cuda_synchronize(_ExplodingTorch()) is None
    assert callable(cuda_synchronize(_FakeTorch(available=True)))


def test_sampling_device_names_the_current_cuda_device_or_cpu():
    assert sampling_device(_FakeTorch(available=True, index=1)) == "cuda:1"
    assert sampling_device(_FakeTorch(available=False)) == "cpu"
    assert sampling_device(_ExplodingTorch()) == "cpu"
