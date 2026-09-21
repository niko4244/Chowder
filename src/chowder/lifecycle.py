"""Where the time and the GPU-hours actually went, phase by phase.

The completed GSM8K rerun cost 3.463 GPU-hours and its artifacts could not say
how much was model load, how much was the 500 optimizer steps, and how much was
baseline plus candidate generation. The plan's P6 records the answer: generation
dominated it (1.16 + 1.86 against 0.44 for training), so a budget derived from
optimizer steps alone was wrong by roughly seven times.

Two rules in here matter more than the arithmetic:

* **An unmeasured phase is unknown, never zero.** ``None`` is a distinct value
  from ``0.0``, it is listed in ``unmeasured``, and a phase required for
  qualification that was never measured refuses rather than quietly reducing the
  reported total. A run must not be able to look cheap by not measuring.
* **A duration must say how it was taken.** Wall-clock timing of asynchronous
  accelerator work without synchronization measures *submission*, not work. A
  synchronized measurement records its synchronization overhead; an unsynchronized
  one is labelled as such so it is never compared with a synchronized number as
  though they were the same quantity.

`profile()` discipline: this module measures nothing on its own and is cheap to
construct. Model loading, warmup and step sampling happen inside an accountable
execution boundary (the worker), and their measurements are recorded here.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

PHASE_MODEL_LOAD = "model_load"
PHASE_FIRST_FORWARD = "first_forward"
PHASE_FIRST_BACKWARD = "first_backward"
PHASE_FIRST_UPDATE = "first_update"
PHASE_STEADY_STEPS = "steady_state_steps"
PHASE_CHECKPOINT_PUBLICATION = "checkpoint_publication"
PHASE_RELOAD = "reload"
PHASE_BASELINE_GENERATION = "baseline_generation"
PHASE_CANDIDATE_GENERATION = "candidate_generation"
PHASE_CLOSEOUT = "closeout"

#: Every phase a complete lifecycle can account for, in execution order.
LIFECYCLE_PHASES: tuple[str, ...] = (
    PHASE_MODEL_LOAD,
    PHASE_FIRST_FORWARD,
    PHASE_FIRST_BACKWARD,
    PHASE_FIRST_UPDATE,
    PHASE_STEADY_STEPS,
    PHASE_CHECKPOINT_PUBLICATION,
    PHASE_RELOAD,
    PHASE_BASELINE_GENERATION,
    PHASE_CANDIDATE_GENERATION,
    PHASE_CLOSEOUT,
)

#: Phases a *training* run cannot qualify without: if the model load or the
#: steady-state step cost was never measured, the run cannot say what it spent,
#: and a forecast built on top of it would be fiction.
REQUIRED_FOR_TRAINING: tuple[str, ...] = (PHASE_MODEL_LOAD, PHASE_STEADY_STEPS)

#: Phases an evaluation cannot qualify without. Both arms are required: the
#: completed rerun's cost was dominated by generation, and one arm alone cannot
#: say what the comparison cost.
REQUIRED_FOR_EVALUATION: tuple[str, ...] = (
    PHASE_BASELINE_GENERATION,
    PHASE_CANDIDATE_GENERATION,
)


class LifecycleAccountingError(RuntimeError):
    """A lifecycle claim cannot be made from the measurements available."""


def cuda_synchronize(backend: Any) -> Callable[[], Any] | None:
    """``backend.cuda.synchronize`` when a real device exists, else None.

    One shared implementation for every worker (trainer and both evaluators),
    for the same reason the renderer is shared: two copies drift, and a drifted
    timing rule silently changes what a number means. None rather than a no-op,
    so an unsynchronized duration is labelled unsynchronized instead of being
    compared with a synchronized one. Never raises. ``backend`` is the imported
    ``torch`` module, passed in so this module needs no torch dependency.
    """
    try:
        if backend.cuda.is_available():
            return backend.cuda.synchronize
    except Exception:
        return None
    return None


def sampling_device(backend: Any) -> str:
    """The device a worker's own memory readings belong to (``cpu`` if unknown)."""
    try:
        if backend.cuda.is_available():
            return f"cuda:{int(backend.cuda.current_device())}"
    except Exception:
        pass
    return "cpu"


def measurement_state(value: Any) -> str:
    """``"unknown"`` for None/missing, ``"measured"`` otherwise.

    A measured zero is a real measurement; an absent one is not a zero. This is
    the same distinction ``target_coverage`` and ``adapter_guard`` make.
    """
    return "unknown" if value is None else "measured"


@dataclass(frozen=True)
class PhaseMeasurement:
    """One phase's duration, or an explicit statement that it was not measured."""

    phase: str
    seconds: float | None
    accelerator_count: int
    synchronized: bool | None = None
    sync_overhead_seconds: float = 0.0
    note: str | None = None

    @property
    def measured(self) -> bool:
        return self.seconds is not None

    @property
    def gpu_hours(self) -> float | None:
        """Attributable accelerator hours, or None when the phase is unknown."""
        if self.seconds is None:
            return None
        return (self.seconds * self.accelerator_count) / 3600.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "seconds": self.seconds,
            "measured": self.measured,
            "gpu_hours": self.gpu_hours,
            "accelerator_count": self.accelerator_count,
            "synchronized": self.synchronized,
            "sync_overhead_seconds": self.sync_overhead_seconds,
            "note": self.note,
        }


class PhaseTimer:
    """Time one phase, optionally bracketing it with accelerator synchronization.

    `synchronize` is a callable (e.g. ``torch.cuda.synchronize``). When it is
    provided it is called before and after the body, and the time spent inside
    those calls is recorded as `sync_overhead_seconds` -- the instrumentation's
    own cost, stated rather than hidden. A synchronization failure never kills
    the measurement: it is recorded, and the timing is marked unsynchronized so
    nobody compares it with a synchronized one.
    """

    def __init__(self, *, synchronize: Callable[[], Any] | None = None) -> None:
        self._synchronize = synchronize
        self.seconds: float | None = None
        self.sync_overhead_seconds = 0.0
        self.synchronized: bool | None = None if synchronize is None else True
        self.sync_failure: str | None = None

    def __enter__(self) -> "PhaseTimer":
        self._sync()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self._sync()
        self.seconds = time.perf_counter() - self._start
        return False

    def _sync(self) -> None:
        if self._synchronize is None:
            return
        started = time.perf_counter()
        try:
            self._synchronize()
        except Exception as exc:  # telemetry is never fatal
            self.synchronized = False
            self.sync_failure = f"{type(exc).__name__}: {exc}"
        self.sync_overhead_seconds += time.perf_counter() - started


@dataclass
class LifecycleLedger:
    """Measured, per-phase accounting for one run's whole lifecycle."""

    accelerator_count: int
    phases: dict[str, PhaseMeasurement] = field(default_factory=dict)
    unmeasured: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.accelerator_count, bool) or not isinstance(self.accelerator_count, int):
            raise TypeError("accelerator_count must be an int")
        if self.accelerator_count < 0:
            raise ValueError("accelerator_count must be non-negative")

    # -- recording ---------------------------------------------------------

    def record(
        self,
        phase: str,
        seconds: float | None,
        *,
        synchronized: bool | None = None,
        sync_overhead_seconds: float = 0.0,
        note: str | None = None,
        accelerator_count: int | None = None,
    ) -> PhaseMeasurement:
        """Record a phase duration. ``seconds=None`` records it as unmeasured."""
        if seconds is not None:
            number = float(seconds)
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"phase {phase!r} seconds must be finite and non-negative")
        count = self.accelerator_count if accelerator_count is None else int(accelerator_count)
        measurement = PhaseMeasurement(
            phase=str(phase),
            seconds=None if seconds is None else float(seconds),
            accelerator_count=count,
            synchronized=synchronized,
            sync_overhead_seconds=float(sync_overhead_seconds),
            note=note,
        )
        self.phases[str(phase)] = measurement
        if measurement.measured:
            self.unmeasured.pop(str(phase), None)
        else:
            self.unmeasured[str(phase)] = note or "not measured"
        return measurement

    def record_unavailable(self, phase: str, reason: str) -> PhaseMeasurement:
        """Record that a phase could not be measured, and why."""
        return self.record(phase, None, note=str(reason))

    def record_timer(self, phase: str, timer: PhaseTimer, *, note: str | None = None) -> PhaseMeasurement:
        """Record a completed `PhaseTimer`."""
        return self.record(
            phase,
            timer.seconds,
            synchronized=timer.synchronized,
            sync_overhead_seconds=timer.sync_overhead_seconds,
            note=note or timer.sync_failure,
        )

    # -- derived totals ---------------------------------------------------

    @property
    def measured_seconds(self) -> float:
        return sum(p.seconds for p in self.phases.values() if p.seconds is not None)

    @property
    def measured_gpu_hours(self) -> float:
        return sum(value for value in self.phase_gpu_hours().values() if value is not None)

    def phase_gpu_hours(self) -> dict[str, float | None]:
        return {name: p.gpu_hours for name, p in self.phases.items()}

    def missing(self, required: Iterable[str]) -> tuple[str, ...]:
        return tuple(
            phase
            for phase in required
            if phase not in self.phases or not self.phases[phase].measured
        )

    # -- refusals ---------------------------------------------------------

    def require(self, required: Sequence[str], *, purpose: str = "qualification") -> None:
        """Refuse when a phase required for `purpose` was never measured."""
        missing = self.missing(required)
        if not missing:
            return
        detail = ", ".join(f"{phase} ({self.unmeasured.get(phase, 'no measurement')})" for phase in missing)
        raise LifecycleAccountingError(
            f"cannot make a {purpose} claim: {len(missing)} required phase(s) were "
            f"never measured: {detail}. An unmeasured phase is UNKNOWN, not zero -- "
            "reporting it as zero would make an unmeasured run look cheaper than a "
            "measured one. Measure the phase, or record explicitly why it cannot be."
        )

    def assert_within_budget(
        self, budget_gpu_hours: float, *, required: Sequence[str] = ()
    ) -> dict[str, Any]:
        """Refuse when required phases are unknown or the measured total overflows.

        Only *measured* time is compared: an unmeasured required phase refuses
        first, because a total that silently omits it is not a smaller cost, it is
        an unknown one. The reported comparison is durable evidence either way.
        """
        if required:
            self.require(required, purpose="budget")
        budget = float(budget_gpu_hours)
        if not math.isfinite(budget) or budget < 0:
            raise ValueError("budget_gpu_hours must be finite and non-negative")
        measured = self.measured_gpu_hours
        report = {
            "budget_gpu_hours": budget,
            "measured_gpu_hours": measured,
            "measured_seconds": self.measured_seconds,
            "phase_gpu_hours": self.phase_gpu_hours(),
            "unmeasured": dict(self.unmeasured),
            "within_budget": measured <= budget + 1e-12,
        }
        if not report["within_budget"]:
            raise LifecycleAccountingError(
                f"measured lifecycle cost {measured:.6g} GPU-hours exceeds the "
                f"reserved budget {budget:.6g} GPU-hours "
                f"({len(self.unmeasured)} phase(s) still unmeasured: "
                f"{sorted(self.unmeasured)})."
            )
        return report

    def record_forecast_comparison(
        self, forecast: "LifecycleForecast"
    ) -> dict[str, Any]:
        """The durable estimate-versus-actual record for this ledger (see below)."""
        return forecast.compare_to(self)

    def compare_to_estimate(self, estimate_by_phase: Mapping[str, float | None]) -> dict[str, Any]:
        """Durable estimated-versus-measured comparison, per phase.

        A phase with no estimate, or no measurement, reports `delta: None` -- the
        difference is unknown, not zero. This is the record that lets a later run
        stop repeating a wrong estimate.
        """
        rows: dict[str, Any] = {}
        for phase, estimate in estimate_by_phase.items():
            measured = self.phases.get(str(phase))
            seconds = None if measured is None else measured.seconds
            delta = (
                None
                if seconds is None or estimate is None
                else float(seconds) - float(estimate)
            )
            rows[str(phase)] = {
                "estimated_seconds": None if estimate is None else float(estimate),
                "measured_seconds": seconds,
                "delta_seconds": delta,
                "state": (
                    "unknown"
                    if delta is None
                    else ("matched" if abs(delta) < 1e-9 else "diverged")
                ),
            }
        for phase, measurement in self.phases.items():
            if phase not in rows:
                rows[phase] = {
                    "estimated_seconds": None,
                    "measured_seconds": measurement.seconds,
                    "delta_seconds": None,
                    "state": "unknown",
                }
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "accelerator_count": self.accelerator_count,
            "phases": {name: p.to_dict() for name, p in self.phases.items()},
            "measured_seconds": self.measured_seconds,
            "measured_gpu_hours": self.measured_gpu_hours,
            "unmeasured": dict(self.unmeasured),
        }


# ---------------------------------------------------------------------------
# what is actually loaded: tensor classes, storage, and quantization reality
# ---------------------------------------------------------------------------

#: Storage dtypes that mean "packed quantized values", not one value per element.
_PACKED_STORAGE_DTYPES = frozenset({"uint8", "int8"})


def _dtype_name(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")


# ---------------------------------------------------------------------------
# the forecast: derived from measurements, and stating what it cannot know
# ---------------------------------------------------------------------------

#: How much a phase's estimate is worth, by where it came from. A "declared"
#: number is a human's guess; a "derived" one is arithmetic over measured
#: quantities (steps x measured step time); "measured" is a prior real run.
BASIS_WEIGHTS: dict[str, float] = {
    "measured": 1.0,
    "derived": 0.75,
    "declared": 0.25,
    "unknown": 0.0,
}


@dataclass(frozen=True)
class PhaseEstimate:
    """An estimated duration for one phase, with the basis that justifies it."""

    phase: str
    seconds: float | None
    basis: str
    note: str | None = None

    def __post_init__(self) -> None:
        if self.basis not in BASIS_WEIGHTS:
            raise ValueError(
                f"phase {self.phase!r} estimate basis {self.basis!r} is not one of "
                f"{sorted(BASIS_WEIGHTS)} -- an unlabelled estimate is not evidence"
            )
        if self.seconds is None:
            if self.basis != "unknown":
                raise ValueError(
                    f"phase {self.phase!r} has no estimate but basis {self.basis!r}; "
                    "only 'unknown' can carry no number"
                )
            return
        number = float(self.seconds)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"phase {self.phase!r} seconds must be finite and non-negative")
        if self.basis == "unknown":
            raise ValueError(
                f"phase {self.phase!r} has an estimate of {number} but basis 'unknown'"
            )

    @property
    def known(self) -> bool:
        return self.seconds is not None

    @property
    def weight(self) -> float:
        return BASIS_WEIGHTS[self.basis]

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "seconds": self.seconds,
            "basis": self.basis,
            "note": self.note,
        }


class LifecycleForecast:
    """A whole-lifecycle estimate that names its own unknowns.

    The lesson this exists to encode: a forecast built from optimizer steps
    alone under-predicted the completed rerun by roughly seven times, because
    baseline and candidate generation cost 1.16 + 1.86 GPU-hours against 0.44
    for the 500 training steps. So the forecast keeps per-phase estimates, keeps
    the phases it has no estimate for as *unknown* rather than zero, and refuses
    to present a total as though the unknowns had been counted.
    """

    def __init__(
        self, *, accelerator_count: int, estimates: Mapping[str, PhaseEstimate]
    ) -> None:
        if isinstance(accelerator_count, bool) or not isinstance(accelerator_count, int):
            raise TypeError("accelerator_count must be an int")
        if accelerator_count < 0:
            raise ValueError("accelerator_count must be non-negative")
        self.accelerator_count = accelerator_count
        self.estimates: dict[str, PhaseEstimate] = {
            str(phase): estimate for phase, estimate in estimates.items()
        }

    @classmethod
    def from_terms(
        cls,
        *,
        accelerator_count: int,
        terms: Mapping[str, PhaseEstimate | tuple[float | None, str, str | None]],
    ) -> "LifecycleForecast":
        """Build from ``phase -> (seconds, basis, note)`` or `PhaseEstimate`.

        A missing phase is simply absent (unknown); passing ``(None, "unknown",
        reason)`` records *why* it is unknown, which is the difference between an
        acknowledged gap and an oversight.
        """
        estimates: dict[str, PhaseEstimate] = {}
        for phase, term in terms.items():
            if isinstance(term, PhaseEstimate):
                estimate = term
            elif isinstance(term, tuple) and len(term) == 3:
                seconds, basis, note = term
                estimate = PhaseEstimate(
                    phase=str(phase), seconds=seconds, basis=basis, note=note
                )
            else:
                raise TypeError(
                    f"phase {phase!r} estimate must be a PhaseEstimate or a "
                    "(seconds, basis, note) triple"
                )
            estimates[str(phase)] = estimate
        return cls(accelerator_count=accelerator_count, estimates=estimates)

    # -- what is known ----------------------------------------------------

    @property
    def known_seconds(self) -> float:
        """Sum over estimated phases only. Unknown phases are excluded, not zeroed."""
        return sum(e.seconds for e in self.estimates.values() if e.seconds is not None)

    @property
    def known_gpu_hours(self) -> float:
        return (self.known_seconds * self.accelerator_count) / 3600.0

    @property
    def unknown_phases(self) -> tuple[str, ...]:
        return tuple(sorted(p for p, e in self.estimates.items() if not e.known))

    @property
    def confidence(self) -> float:
        """The weakest basis the forecast leaned on (1.0 when nothing is estimated)."""
        weights = [e.weight for e in self.estimates.values() if e.known]
        return min(weights) if weights else 1.0

    def confidence_for(self, required: Iterable[str]) -> float:
        """Confidence in a claim about `required` phases: 0.0 if any is unknown."""
        required = tuple(str(phase) for phase in required)
        if self.missing_estimates(required):
            return 0.0
        weights = [self.estimates[phase].weight for phase in required if phase in self.estimates]
        return min(weights) if weights else 1.0

    def missing_estimates(self, required: Iterable[str]) -> tuple[str, ...]:
        return tuple(
            str(phase)
            for phase in required
            if str(phase) not in self.estimates or not self.estimates[str(phase)].known
        )

    # -- refusals ---------------------------------------------------------

    def require_estimated(self, required: Iterable[str], *, purpose: str = "forecast") -> None:
        missing = self.missing_estimates(required)
        if not missing:
            return
        generation_legs = {PHASE_BASELINE_GENERATION, PHASE_CANDIDATE_GENERATION}
        detail = ", ".join(
            f"{phase} ({self.estimates[phase].note if phase in self.estimates else 'no estimate'})"
            for phase in missing
        )
        extra = (
            " Baseline and candidate generation are separate measured legs, and the "
            "completed rerun's cost was dominated by them -- a reservation that "
            "omits an independent evaluation leg is not smaller, it is unknown."
            if generation_legs.intersection(missing)
            else ""
        )
        raise LifecycleAccountingError(
            f"cannot make a {purpose} claim: {len(missing)} required phase(s) have "
            f"no estimate: {detail}. No basis means UNKNOWN, not free.{extra}"
        )

    # -- the durable record -----------------------------------------------

    def compare_to(self, ledger: LifecycleLedger) -> dict[str, Any]:
        """Estimated versus measured, per phase -- durable, never invented.

        A phase measured but never estimated (and vice versa) reports
        ``delta_seconds: None``: the difference is unknown, not zero.
        """
        rows = ledger.compare_to_estimate(
            {phase: estimate.seconds for phase, estimate in self.estimates.items()}
        )
        for phase, row in rows.items():
            estimate = self.estimates.get(phase)
            row["estimated_basis"] = None if estimate is None else estimate.basis
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "accelerator_count": self.accelerator_count,
            "phases": {phase: estimate.to_dict() for phase, estimate in self.estimates.items()},
            "known_seconds": self.known_seconds,
            "known_gpu_hours": self.known_gpu_hours,
            "unknown_phases": list(self.unknown_phases),
            "confidence": self.confidence,
        }


def tensor_inventory(model: Any) -> dict[str, Any]:
    """What the loaded model actually holds, measured rather than requested.

    Reports per-dtype element counts, the trainable/frozen split, devices, and the
    model's declared quantization metadata. A parameter stored as `uint8` is
    packed quantized data (4-bit packs two values per byte) and is labelled as
    such, so it is never read as one value per element.
    """
    parameters = [(str(name), param) for name, param in model.named_parameters()]
    by_dtype: dict[str, int] = {}
    by_device: dict[str, int] = {}
    packed_quantized_elements = 0
    trainable_elements = 0
    frozen_elements = 0
    for _, param in parameters:
        dtype = _dtype_name(getattr(param, "dtype", "unknown"))
        elements = int(param.numel())
        by_dtype[dtype] = by_dtype.get(dtype, 0) + elements
        device = str(getattr(param, "device", "unknown"))
        by_device[device] = by_device.get(device, 0) + elements
        if dtype in _PACKED_STORAGE_DTYPES:
            packed_quantized_elements += elements
        if getattr(param, "requires_grad", False):
            trainable_elements += elements
        else:
            frozen_elements += elements

    config = getattr(model, "config", None)
    quantization_config = getattr(config, "quantization_config", None) if config is not None else None
    metadata: dict[str, Any] = {
        "declared": None if quantization_config is None else dict(quantization_config),
        "declared_available": quantization_config is not None,
    }
    return {
        "parameter_count": len(parameters),
        "total_elements": sum(by_dtype.values()),
        "elements_by_dtype": by_dtype,
        "elements_by_device": by_device,
        "packed_quantized_storage_elements": packed_quantized_elements,
        "packed_quantized_note": (
            "parameters stored as int8/uint8 hold PACKED quantized values "
            "(two 4-bit values per byte); element counts are storage elements"
            if packed_quantized_elements
            else None
        ),
        "trainable_elements": trainable_elements,
        "frozen_elements": frozen_elements,
        "quantization": metadata,
    }


def quantization_reality_report(
    model: Any, *, requested: str, expert_marker: str = "experts"
) -> dict[str, Any]:
    """Does the loaded model's storage match the requested quantization?

    The audit's shape: a loader asked for 4-bit, and the expert tensors stayed
    bfloat16 (the raw expert parameters are skipped by the quantizer). Reporting
    the request would have been a lie; this reports the measurement.
    """
    inventory = tensor_inventory(model)
    observed = inventory["elements_by_dtype"]
    expert_dtypes: dict[str, int] = {}
    for name, param in model.named_parameters():
        if expert_marker in str(name):
            dtype = _dtype_name(getattr(param, "dtype", "unknown"))
            expert_dtypes[dtype] = expert_dtypes.get(dtype, 0) + int(param.numel())
    requested_normalized = str(requested).lower()
    quantized_requested = requested_normalized not in {"none", "fp32", "float32"}
    unpacked_present = sorted(
        dtype
        for dtype in observed
        if quantized_requested and dtype not in _PACKED_STORAGE_DTYPES
    )
    matches = (not quantized_requested and not inventory["packed_quantized_storage_elements"]) or (
        quantized_requested and not unpacked_present
    )
    note = None
    if not matches:
        note = (
            f"requested {requested!r} but these tensors are stored unquantized: "
            f"{unpacked_present}. Raw expert parameters are the usual case -- a "
            "4-bit loader that skips them leaves them in their original dtype, and "
            "the budget must be computed from what is present, not from the request."
        )
    return {
        "requested": requested,
        "elements_by_dtype": observed,
        "expert_elements_by_dtype": expert_dtypes,
        "packed_quantized_storage_elements": inventory["packed_quantized_storage_elements"],
        "matches_request": matches,
        "note": note,
    }


# ---------------------------------------------------------------------------
# the ledgers a real worker can and cannot fill in
# ---------------------------------------------------------------------------

_FIRST_STEP_PHASES: tuple[tuple[str, str], ...] = (
    (PHASE_FIRST_FORWARD, "forward"),
    (PHASE_FIRST_BACKWARD, "backward"),
    (PHASE_FIRST_UPDATE, "optimizer update"),
)

_DETAILED_TIMING_OFF = (
    "detailed_timing_telemetry was not enabled: synchronizing around each "
    "phase costs real wall time (a measured ~17% on a real run), so the first "
    "step is unknown rather than guessed"
)


def training_lifecycle_ledger(
    *,
    accelerator_count: int,
    model_load: PhaseTimer | None = None,
    checkpoint_publication: PhaseTimer | None = None,
    steady_state_steps_seconds: float | None = None,
    detailed_timing_enabled: bool = False,
    first_forward_seconds: float | None = None,
    first_backward_seconds: float | None = None,
    first_update_seconds: float | None = None,
    resumed_from_checkpoint: bool = False,
    closeout_seconds: float | None = None,
) -> LifecycleLedger:
    """The phase ledger a training worker can honestly produce.

    Every phase the worker did not measure is recorded with the reason it was
    not measured. This is the deliberate opposite of a zero: an unmeasured
    phase that reads as ``0.0`` makes an unmeasured run look cheaper than a
    measured one, which is exactly how a 3.463 GPU-hour run's cost became
    unexplainable from its own artifacts.
    """
    ledger = LifecycleLedger(accelerator_count=accelerator_count)

    if model_load is not None:
        ledger.record_timer(PHASE_MODEL_LOAD, model_load)
    else:
        ledger.record_unavailable(PHASE_MODEL_LOAD, "the worker did not time the model load")

    if steady_state_steps_seconds is None:
        ledger.record_unavailable(
            PHASE_STEADY_STEPS, "the training loop did not complete"
        )
    else:
        # Wall time over the whole train() call, deliberately *unsynchronized*:
        # it is a real elapsed duration, and labelling it synchronized would
        # imply a per-phase sync bracket it never had.
        ledger.record(PHASE_STEADY_STEPS, steady_state_steps_seconds, synchronized=False)

    first_seconds = {
        PHASE_FIRST_FORWARD: first_forward_seconds,
        PHASE_FIRST_BACKWARD: first_backward_seconds,
        PHASE_FIRST_UPDATE: first_update_seconds,
    }
    names = dict(_FIRST_STEP_PHASES)
    for phase, seconds in first_seconds.items():
        if seconds is not None:
            ledger.record(phase, seconds, synchronized=True)
        elif not detailed_timing_enabled:
            ledger.record_unavailable(phase, _DETAILED_TIMING_OFF)
        else:
            ledger.record_unavailable(
                phase, f"the first {names[phase]} was never observed"
            )

    if checkpoint_publication is not None:
        ledger.record_timer(PHASE_CHECKPOINT_PUBLICATION, checkpoint_publication)
    else:
        ledger.record_unavailable(
            PHASE_CHECKPOINT_PUBLICATION, "this run published no checkpoint"
        )

    if resumed_from_checkpoint:
        ledger.record_unavailable(
            PHASE_RELOAD,
            "the restore happens inside Trainer.train(resume_from_checkpoint=...), "
            "which the worker cannot time as a separate phase",
        )
    else:
        ledger.record_unavailable(
            PHASE_RELOAD, "this run did not resume from a checkpoint"
        )

    for phase in (PHASE_BASELINE_GENERATION, PHASE_CANDIDATE_GENERATION):
        ledger.record_unavailable(
            phase,
            "generation is performed by the independent evaluator process, which "
            "reports its own ledger",
        )

    if closeout_seconds is None:
        ledger.record_unavailable(PHASE_CLOSEOUT, "closeout was not separately timed")
    else:
        ledger.record(PHASE_CLOSEOUT, closeout_seconds)

    return ledger


def ledger_from_payload(payload: Mapping[str, Any]) -> LifecycleLedger:
    """Rebuild a ledger from a worker's serialized payload.

    A parent must not guess at a child's numbers. Anything unparseable -- a
    phase list that is not a mapping, a duration that is not a number -- is
    refused rather than coerced, because an artifact whose cost breakdown is
    wrong-looking-but-parsed is worse than one that admits it has none.
    """
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"worker lifecycle ledger is invalid: {type(payload).__name__}")
    phases = payload.get("phases", {})
    if not isinstance(phases, Mapping):
        raise RuntimeError(
            "worker lifecycle ledger is invalid: 'phases' is not a mapping of "
            f"phase -> measurement ({type(phases).__name__})"
        )
    unmeasured = payload.get("unmeasured", {})
    if not isinstance(unmeasured, Mapping):
        raise RuntimeError("worker lifecycle ledger is invalid: 'unmeasured' is not a mapping")
    accelerator_count = payload.get("accelerator_count", 0)
    if isinstance(accelerator_count, bool) or not isinstance(accelerator_count, int):
        raise RuntimeError(
            "worker lifecycle ledger is invalid: 'accelerator_count' is not an integer"
        )

    ledger = LifecycleLedger(accelerator_count=accelerator_count)
    for phase, measurement in phases.items():
        if not isinstance(measurement, Mapping):
            raise RuntimeError(
                f"worker lifecycle ledger is invalid: phase {phase!r} is not a mapping"
            )
        seconds = measurement.get("seconds")
        overhead = measurement.get("sync_overhead_seconds", 0.0)
        if isinstance(overhead, bool) or not isinstance(overhead, (int, float)):
            raise RuntimeError(
                f"worker lifecycle ledger is invalid: phase {phase!r} sync overhead "
                "is not a number"
            )
        try:
            ledger.record(
                str(phase),
                seconds,
                synchronized=measurement.get("synchronized"),
                sync_overhead_seconds=float(overhead),
                note=measurement.get("note"),
                accelerator_count=measurement.get("accelerator_count", accelerator_count),
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"worker lifecycle ledger is invalid: phase {phase!r}: {exc}"
            ) from exc
    for phase, reason in unmeasured.items():
        if phase in ledger.phases and ledger.phases[phase].measured:
            continue
        ledger.record_unavailable(str(phase), str(reason))
    return ledger


def evaluation_lifecycle_evidence(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """A parent's view of one evaluation arm's reported lifecycle.

    An arm that reported nothing leaves the ledger ``None`` with a stated
    reason -- the same rule as the trainer. A *malformed* report is refused: a
    stored cost breakdown that does not parse is worse than an absent one,
    because it looks like evidence.
    """
    if not isinstance(runtime, Mapping):
        raise RuntimeError("evaluation runtime payload is not a mapping")
    sampling = runtime.get("memory_sampling")
    if sampling is not None and not isinstance(sampling, Mapping):
        raise RuntimeError("evaluator reported an invalid memory_sampling payload")
    payload = runtime.get("lifecycle")
    if payload is None:
        return {
            "phase_ledger": None,
            "state": "unknown",
            "reason": "the evaluator did not report a lifecycle ledger",
            "memory_sampling": None if sampling is None else dict(sampling),
        }
    ledger = ledger_from_payload(payload)
    return {
        "phase_ledger": ledger.to_dict(),
        "state": "measured",
        "reason": None,
        "memory_sampling": None if sampling is None else dict(sampling),
    }


_LEDGER_ARM_PHASES: dict[str, str] = {
    "baseline": PHASE_BASELINE_GENERATION,
    "candidate": PHASE_CANDIDATE_GENERATION,
}


def evaluation_lifecycle_ledger(
    *,
    accelerator_count: int,
    arm: str,
    generation_seconds: float | None = None,
    model_load_seconds: float | None = None,
) -> LifecycleLedger:
    """One evaluation arm's ledger: its own generation, and the other arm's absence.

    Each arm runs in its own worker process, so an arm cannot measure the other
    one; it records that as unknown rather than inheriting a number it never saw.
    The comparison that needs both legs joins the two ledgers.
    """
    if arm not in _LEDGER_ARM_PHASES:
        raise ValueError(
            f"unknown evaluation arm {arm!r}; expected one of {sorted(_LEDGER_ARM_PHASES)}"
        )
    ledger = LifecycleLedger(accelerator_count=accelerator_count)
    if model_load_seconds is None:
        ledger.record_unavailable(
            PHASE_MODEL_LOAD, "the evaluator did not time the model load"
        )
    else:
        ledger.record(PHASE_MODEL_LOAD, model_load_seconds, synchronized=False)

    own_phase = _LEDGER_ARM_PHASES[arm]
    if generation_seconds is None:
        ledger.record_unavailable(own_phase, "generation was not measured")
    else:
        ledger.record(own_phase, generation_seconds, synchronized=False)

    for other_arm, other_phase in _LEDGER_ARM_PHASES.items():
        if other_phase != own_phase:
            ledger.record_unavailable(
                other_phase,
                f"the {other_arm} arm runs in a separate worker process",
            )
    return ledger
