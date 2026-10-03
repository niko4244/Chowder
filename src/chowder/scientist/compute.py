"""Compute providers: where a compiled experiment runs.

The ResearchDirector does not care where an experiment runs; the scheduler
decides, from explicit provider availability and quota models. The mission's
hardware-context rule lives here too: quality findings may compare across
hardware when the protocol is identical; efficiency findings never leave their
hardware context. See docs/COMPUTE_PROVIDERS.md (audit + honest status: the
Kaggle *submission* path is scaffolded, the scheduling/evidence rules are
proven).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class SchedulerRefusal(RuntimeError):
    """Fail-closed scheduling refusal (no provider, quota, unknown pin)."""


class ExperimentClass(str, Enum):
    SCREENING = "screening"
    EXPLORATORY = "exploratory"
    SUBSTANTIAL = "substantial"
    REPLICATION = "replication"


@dataclass(frozen=True)
class ExperimentRequest:
    """What the research layer asks the scheduler to run."""

    experiment_id: str
    proposal_id: str
    hypothesis_id: str
    campaign_spec: dict[str, Any]
    experiment_class: ExperimentClass = ExperimentClass.EXPLORATORY
    estimated_gpu_hours: float = 0.0
    #: Pins the provider by name (replication-on-other-hardware does this).
    require_provider: str | None = None
    #: Claims about speed/VRAM/wall-time must run where the claim lives.
    hardware_dependent: bool = False

    def __post_init__(self) -> None:
        if not self.experiment_id or not self.proposal_id or not self.hypothesis_id:
            raise ValueError("a request names its experiment, proposal and hypothesis")
        if self.estimated_gpu_hours < 0:
            raise ValueError("estimated_gpu_hours must be non-negative")


@dataclass(frozen=True)
class ProviderQuota:
    """A provider's declared, declining budget model."""

    weekly_gpu_hours: float
    used_gpu_hours: float = 0.0
    concurrent_jobs: int = 1

    def remaining_gpu_hours(self) -> float:
        return max(0.0, self.weekly_gpu_hours - self.used_gpu_hours)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weekly_gpu_hours": self.weekly_gpu_hours,
            "used_gpu_hours": self.used_gpu_hours,
            "remaining_gpu_hours": self.remaining_gpu_hours(),
            "concurrent_jobs": self.concurrent_jobs,
        }


@dataclass
class Submission:
    """A job handed to a provider. `queued` providers (like Kaggle in this
    pass) carry the job spec; executing providers run it inline."""

    submission_id: str
    request: ExperimentRequest
    provider_name: str
    status: str = "queued"              # queued | running | complete | failed
    hardware_class: str = ""
    device_gpu_hours: float = 0.0
    experiment_class: ExperimentClass = ExperimentClass.EXPLORATORY
    result: dict[str, Any] | None = None
    environment_fingerprint: dict[str, Any] | None = None


@runtime_checkable
class ComputeProvider(Protocol):
    """One external/attached compute pool. Returns typed data; never touches
    policy, thresholds, or the protected set."""

    name: str
    hardware_class: str

    def available(self) -> bool:
        """Whether this provider can accept work right now."""
        ...

    def quota(self) -> ProviderQuota:
        ...

    def estimate_cost(self, request: ExperimentRequest) -> float:
        """Device GPU-hours (wall × active accelerators), not a promise."""
        ...

    def submit(self, request: ExperimentRequest) -> Submission:
        ...

    def poll(self, submission: Submission) -> Submission:
        ...


# ---------------------------------------------------------------------------
# local provider
# ---------------------------------------------------------------------------


def _local_hardware_class() -> str:
    try:
        from ..hardware import detect_hardware
        snapshot = detect_hardware()
        if snapshot.accelerators:
            first = snapshot.accelerators[0]
            total = snapshot.topology.total_accelerator_memory_gb
            count = len(snapshot.accelerators)
            name = (getattr(first, "name", "") or "gpu").lower().replace(" ", "_")
            return f"local_{count}x_{name}_{total:.0f}gb"
    except Exception:
        pass
    return "local_cpu_only"


class LocalCudaProvider:
    """The machine Chowder runs on. Executes nothing here — the production
    training/evaluation bindings already own local execution; this provider
    exists so the scheduler treats 'local' as one choice among many, with the
    same evidence discipline."""

    name = "local_cuda"

    def __init__(self, *, accelerators: int | None = None,
                 weekly_gpu_hours: float = 1e9) -> None:
        self._hardware_class = _local_hardware_class()
        if accelerators is None:
            accelerators = self._detected_accelerators()
        if accelerators < 0:
            raise ValueError("accelerators must be non-negative")
        self._accelerators = accelerators
        self._quota = ProviderQuota(weekly_gpu_hours=weekly_gpu_hours)

    def _detected_accelerators(self) -> int:
        try:
            from ..hardware import detect_hardware
            snapshot = detect_hardware()
            return len(snapshot.accelerators)
        except Exception:
            return 0

    @property
    def hardware_class(self) -> str:
        return self._hardware_class

    def available(self) -> bool:
        return self._accelerators > 0

    def quota(self) -> ProviderQuota:
        return self._quota

    def estimate_cost(self, request: ExperimentRequest) -> float:
        return request.estimated_gpu_hours * max(1, self._accelerators)

    def submit(self, request: ExperimentRequest) -> Submission:
        if not self.available():
            raise SchedulerRefusal("LOCAL_UNAVAILABLE: no accelerator detected")
        affordable = request.estimated_gpu_hours <= self._quota.remaining_gpu_hours()
        if not affordable:
            raise SchedulerRefusal("QUOTA_EXHAUSTED: local weekly budget exhausted")
        self._quota = replace(
            self._quota, used_gpu_hours=self._quota.used_gpu_hours + request.estimated_gpu_hours
        )
        return Submission(
            submission_id=f"sub-local-{request.experiment_id}",
            request=request,
            provider_name=self.name,
            status="queued",
            hardware_class=self.hardware_class,
            device_gpu_hours=self.estimate_cost(request),
            experiment_class=request.experiment_class,
        )

    def poll(self, submission: Submission) -> Submission:
        # Local execution happens through the production bindings; the
        # scheduler observes the resulting run, so a submission stays queued
        # until the observation is recorded. Honest: nothing pretends to run.
        return submission


# ---------------------------------------------------------------------------
# kaggle provider
# ---------------------------------------------------------------------------


class KaggleProvider:
    """Opportunistic Kaggle notebook capacity (T4×2 today, 12 h sessions,
    weekly GPU quota that varies with demand).

    Honest scope (docs/COMPUTE_PROVIDERS.md §6): submission constructs the job
    spec and records it queued; the kernel-push/session management is NOT
    implemented in this pass. Quota is a declared, declining model — the
    provider refuses when the operator's declared budget is out, rather than
    promising capacity Kaggle may not grant.
    """

    name = "kaggle"
    hardware_class = "kaggle_2x_t4_16gb"
    screening = True  # the designated screening lane

    #: Two T4s per notebook (Kaggle's current GPU shape); device-hours =
    #: wall × 2, matching the growth accounting's accelerator-seconds rule.
    ACCELERATORS_PER_NOTEBOOK = 2

    def __init__(self, *, username: str = "", api_key: str = "",
                 weekly_gpu_hours: float = 20.0, screening_lane: bool = True) -> None:
        self._username = username
        self._api_key = api_key
        self._quota = ProviderQuota(
            weekly_gpu_hours=weekly_gpu_hours,
            concurrent_jobs=2,
        )
        self.screening_lane = screening_lane

    def configured(self) -> bool:
        return bool(self._username) and bool(self._api_key)

    def available(self) -> bool:
        return self.configured() and self._quota.remaining_gpu_hours() > 0

    def quota(self) -> ProviderQuota:
        return self._quota

    def estimate_cost(self, request: ExperimentRequest) -> float:
        return request.estimated_gpu_hours * self.ACCELERATORS_PER_NOTEBOOK

    def submit(self, request: ExperimentRequest) -> Submission:
        if not self.configured():
            raise SchedulerRefusal(
                "KAGGLE_NOT_CONFIGURED: set the Kaggle username/API key to use "
                "this provider; refusing rather than pretending"
            )
        cost = self.estimate_cost(request)
        if cost > self._quota.remaining_gpu_hours():
            raise SchedulerRefusal(
                f"QUOTA_EXHAUSTED: needs {cost} device GPU-hours, "
                f"{self._quota.remaining_gpu_hours()} remain of "
                f"{self._quota.weekly_gpu_hours}"
            )
        self._quota = replace(self._quota, used_gpu_hours=self._quota.used_gpu_hours + cost)
        return Submission(
            submission_id=f"sub-kaggle-{request.experiment_id}",
            request=request,
            provider_name=self.name,
            status="queued",
            hardware_class=self.hardware_class,
            device_gpu_hours=cost,
            experiment_class=request.experiment_class,
        )

    def poll(self, submission: Submission) -> Submission:
        # Kernel-push/session management is deliberately not implemented in
        # this pass (docs/COMPUTE_PROVIDERS.md §6): a queued submission stays
        # queued. Nothing fabricates a result.
        return submission


# ---------------------------------------------------------------------------
# scheduler
# ---------------------------------------------------------------------------


class ExperimentScheduler:
    """Chooses where an experiment runs, fail-closed."""

    def __init__(self, providers: list[Any]) -> None:
        if not providers:
            raise SchedulerRefusal(
                "NO_PROVIDER_AVAILABLE: the scheduler was constructed with no "
                "providers; refusing rather than assuming local"
            )
        self._providers: dict[str, Any] = {}
        self._order: list[str] = []
        for provider in providers:
            if provider.name in self._providers:
                raise SchedulerRefusal(f"DUPLICATE_PROVIDER: {provider.name}")
            self._providers[provider.name] = provider
            self._order.append(provider.name)

    def providers(self) -> tuple[str, ...]:
        return tuple(self._order)

    def schedule(self, request: ExperimentRequest) -> Submission:
        """Route one request. Screening requests prefer a declared screening
        lane (before the general preference order); everything else follows
        declaration order. A pinned request runs nowhere else: an unknown pin
        refuses by name, and a pin the scheduler cannot honor right now
        refuses rather than rerouting (replication-on-different-hardware must
        not silently become replication-on-the-same-machine)."""
        if request.require_provider is not None:
            provider = self._providers.get(request.require_provider)
            if provider is None:
                raise SchedulerRefusal(
                    f"UNKNOWN_PROVIDER_PIN: {request.require_provider!r} is not a "
                    "declared provider; refusing rather than rerouting"
                )
            if not provider.available():
                raise SchedulerRefusal(
                    f"PINNED_PROVIDER_UNAVAILABLE: {request.require_provider!r} "
                    "cannot accept work now"
                )
            return provider.submit(request)
        candidates: list[str] = list(self._order)
        if request.experiment_class == ExperimentClass.SCREENING:
            screening_first = [n for n in self._order
                               if getattr(self._providers[n], "screening", False)]
            candidates = screening_first + [n for n in self._order
                                            if n not in screening_first]
        for name in candidates:
            provider = self._providers[name]
            if not provider.available():
                continue
            cost = provider.estimate_cost(request)
            if cost > provider.quota().remaining_gpu_hours():
                continue
            try:
                return provider.submit(request)
            except SchedulerRefusal:
                continue
        raise SchedulerRefusal(
            "NO_PROVIDER_AVAILABLE: every provider refused this request "
            "(unavailable or over quota)"
        )
