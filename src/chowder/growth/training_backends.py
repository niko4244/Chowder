"""Declared training backends: one execution surface, three providers.

A campaign declares *where and how* it executes with a single
``training_backend`` object::

    "training_backend": {"provider": "local", "config": {"device": "auto"}}

Downstream code never branches on the backend. It asks the declared provider
for a panel (what hardware exists), an estimate (what one recipe would cost in
memory), an admission verdict (whether the recipe fits the growth envelope),
and a ``TrainingFn`` (the execution seam the cycle already uses). The three
providers wrap the executors that already exist -- they do not re-implement
training:

* ``local``   -- the production CLI through
  :class:`chowder.growth.training_binding.SubprocessTrainingFn`.
* ``unsloth`` -- the same production CLI, whose project template selects the
  isolated Unsloth engine; the provider adds the capability matrix and the
  unsupported-combination refusals the executor enforces.
* ``kaggle``  -- :class:`chowder.growth.kaggle_campaign.KaggleTrainingFn` over a
  :class:`chowder.growth.compute_backend.ComputeBackend`, re-exposed rather
  than re-implemented.

``auto`` is a thin policy: it may only choose a provider whose preflight
passes, and it records the chosen provider and the reason before compute.

Everything a backend observes is rendered through one vocabulary:
:class:`UniformAttemptOutcome` wraps
:class:`chowder.growth.compute_backend.AttemptOutcome` (same statuses, same
settlement, same evidence keys), so classification, settlement and promotion
need no per-backend branch. No backend is ever inferred from an installed
package: an undeclared campaign is ``local``, and an unknown provider refuses.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (
    Any,
    Callable,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

from .attempt_failure import classify_failure
from .compute_backend import (
    OUTCOME_STATUSES,
    OUTCOME_SUCCEEDED,
    ArtifactEntry,
    AttemptOutcome,
    SourceBinding,
    file_digest,
)
from .compute_cost import ComputeCost, SettlementVerdict
from .recipe_planner import TrainingRecipe
from .training_binding import EVIDENCE_FILE, GrowthEnvelope, check_growth_envelope

__all__ = [
    "PROVIDER_LOCAL",
    "PROVIDER_UNSLOTH",
    "PROVIDER_KAGGLE",
    "PROVIDER_AUTO",
    "PROVIDERS",
    "AUTO_DEFAULT_CANDIDATES",
    "TRAINING_BACKEND_SCHEMA",
    "TRAINING_BACKEND_UNKNOWN_PROVIDER",
    "TRAINING_BACKEND_UNSUPPORTED_CONFIG",
    "STRUCTURAL_PREFLIGHT_CODES",
    "TRAINING_BACKEND_AUTO_UNRESOLVED",
    "TRAINING_BACKEND_NO_DEVICE",
    "TRAINING_BACKEND_TEMPLATE_UNDECLARED",
    "TRAINING_BACKEND_TEMPLATE_INVALID",
    "TRAINING_BACKEND_TRAINER_MISMATCH",
    "TRAINING_BACKEND_UNSUPPORTED_CAPABILITY",
    "TRAINING_BACKEND_EVIDENCE_MISSING",
    "TRAINING_BACKEND_EVIDENCE_INVALID",
    "TRAINING_BACKEND_EVIDENCE_STATUS_UNKNOWN",
    "TRAINING_BACKEND_ARTIFACT_DIGEST_MISMATCH",
    "TRAINING_BACKEND_RESUME_CHECKPOINT_MISSING",
    "TRAINING_BACKEND_RESUME_CHECKPOINT_UNDECLARED",
    "KAGGLE_BACKEND_UNAVAILABLE",
    "KAGGLE_CONFIG_INCOMPLETE",
    "CAPABILITY_SUPPORTED",
    "CAPABILITY_DATA_LAYER",
    "CAPABILITY_MODEL_HARDWARE_DEPENDENT",
    "CAPABILITY_VERIFY",
    "CAPABILITY_CAPABILITY_DEPENDENT",
    "CAPABILITY_REFUSED",
    "CAPABILITY_STATUSES",
    "TrainingBackendRefusal",
    "TrainingBackendDeclaration",
    "DevicePanel",
    "PreflightPanel",
    "PreflightResult",
    "BackendEstimate",
    "CapabilityRow",
    "BackendCapabilities",
    "UniformAttemptOutcome",
    "uniform_outcome_from_evidence",
    "TrainingBackend",
    "LocalTrainingBackend",
    "UnslothTrainingBackend",
    "KaggleTrainingBackend",
    "AutoSelection",
    "OverheadReport",
    "preflight_report",
    "chowder_version",
    "backend_declaration",
    "backend_for_provider",
    "choose_auto_backend",
    "resolve_training_backend",
    "probe_local_panel",
    "collect_attempt_evidence",
    "verify_attempt_evidence",
]

# --------------------------------------------------------------------------
# providers and refusal vocabulary
# --------------------------------------------------------------------------

PROVIDER_LOCAL = "local"
PROVIDER_UNSLOTH = "unsloth"
PROVIDER_KAGGLE = "kaggle"
PROVIDER_AUTO = "auto"

#: The closed provider enum. Unknown providers refuse at load time.
PROVIDERS = frozenset({PROVIDER_LOCAL, PROVIDER_UNSLOTH, PROVIDER_KAGGLE, PROVIDER_AUTO})

#: What ``auto`` tries, in order, when the declaration names no candidates.
AUTO_DEFAULT_CANDIDATES = (PROVIDER_LOCAL, PROVIDER_UNSLOTH, PROVIDER_KAGGLE)

DEVICE_CHOICES = ("auto", "cuda", "cpu")

#: Machine-readable refusal identifiers. A consumer branches on these, never
#: on prose.
TRAINING_BACKEND_SCHEMA = "TRAINING_BACKEND_SCHEMA"
TRAINING_BACKEND_UNKNOWN_PROVIDER = "TRAINING_BACKEND_UNKNOWN_PROVIDER"
TRAINING_BACKEND_UNSUPPORTED_CONFIG = "TRAINING_BACKEND_UNSUPPORTED_CONFIG"
TRAINING_BACKEND_AUTO_UNRESOLVED = "TRAINING_BACKEND_AUTO_UNRESOLVED"
TRAINING_BACKEND_NO_DEVICE = "TRAINING_BACKEND_NO_DEVICE"
TRAINING_BACKEND_TEMPLATE_UNDECLARED = "TRAINING_BACKEND_TEMPLATE_UNDECLARED"
TRAINING_BACKEND_TEMPLATE_INVALID = "TRAINING_BACKEND_TEMPLATE_INVALID"
TRAINING_BACKEND_TRAINER_MISMATCH = "TRAINING_BACKEND_TRAINER_MISMATCH"
TRAINING_BACKEND_UNSUPPORTED_CAPABILITY = "TRAINING_BACKEND_UNSUPPORTED_CAPABILITY"
TRAINING_BACKEND_EVIDENCE_MISSING = "TRAINING_BACKEND_EVIDENCE_MISSING"
TRAINING_BACKEND_EVIDENCE_INVALID = "TRAINING_BACKEND_EVIDENCE_INVALID"
TRAINING_BACKEND_EVIDENCE_STATUS_UNKNOWN = "TRAINING_BACKEND_EVIDENCE_STATUS_UNKNOWN"
TRAINING_BACKEND_ARTIFACT_DIGEST_MISMATCH = "TRAINING_BACKEND_ARTIFACT_DIGEST_MISMATCH"
TRAINING_BACKEND_RESUME_CHECKPOINT_MISSING = "TRAINING_BACKEND_RESUME_CHECKPOINT_MISSING"
TRAINING_BACKEND_RESUME_CHECKPOINT_UNDECLARED = (
    "TRAINING_BACKEND_RESUME_CHECKPOINT_UNDECLARED"
)
KAGGLE_BACKEND_UNAVAILABLE = "KAGGLE_BACKEND_UNAVAILABLE"
KAGGLE_CONFIG_INCOMPLETE = "KAGGLE_CONFIG_INCOMPLETE"

#: Preflight refusals that are *declaration* errors -- the campaign and the
#: material it names disagree, so nothing may execute. These stop the campaign
#: at the backend boundary, before any compute. A hardware fact (no accelerator
#: visible) is deliberately not in this set: the panel reports it, and it is
#: enforced where the declaration demands it (``config.device``) or by
#: ``auto``, which only chooses providers whose preflight admitted.
STRUCTURAL_PREFLIGHT_CODES = frozenset(
    {
        TRAINING_BACKEND_UNKNOWN_PROVIDER,
        TRAINING_BACKEND_UNSUPPORTED_CONFIG,
        TRAINING_BACKEND_TEMPLATE_UNDECLARED,
        TRAINING_BACKEND_TEMPLATE_INVALID,
        TRAINING_BACKEND_TRAINER_MISMATCH,
        TRAINING_BACKEND_UNSUPPORTED_CAPABILITY,
        KAGGLE_CONFIG_INCOMPLETE,
    }
)

#: Capability statuses. ``refused`` is the one that stops a run; the others
#: tell an operator where the decision actually lives.
CAPABILITY_SUPPORTED = "supported"
CAPABILITY_DATA_LAYER = "data-layer"
CAPABILITY_MODEL_HARDWARE_DEPENDENT = "model-hardware-dependent"
CAPABILITY_VERIFY = "verify"
CAPABILITY_CAPABILITY_DEPENDENT = "capability-dependent"
CAPABILITY_REFUSED = "refused"
CAPABILITY_STATUSES = frozenset(
    {
        CAPABILITY_SUPPORTED,
        CAPABILITY_DATA_LAYER,
        CAPABILITY_MODEL_HARDWARE_DEPENDENT,
        CAPABILITY_VERIFY,
        CAPABILITY_CAPABILITY_DEPENDENT,
        CAPABILITY_REFUSED,
    }
)

#: The knobs Chowder's own mechanisms expose and Unsloth's engine refuses
#: outright (verified against Unsloth's patched model/attention path is
#: outstanding), mirroring ``UnslothPeftRunSpec.from_resolved_config``.
UNSLOTH_REFUSED_KNOBS = (
    "activation_offload",
    "optimizer_tiering",
    "frozen_layer_streaming",
)

_LOCAL_CONFIG_KEYS = frozenset({"device"})
_UNSLOTH_CONFIG_KEYS = frozenset({"device"})
_AUTO_CONFIG_KEYS = frozenset({"candidates"})

KAGGLE_REQUIRED_CONFIG_KEYS = (
    "repository",
    "commit_sha",
    "entry_point",
    "mounts",
    "attempts_root",
    "timeout_seconds",
    "accelerator",
    # The kernel has to be told what to run; a remote attempt without a payload
    # declaration is a job that would have to invent its own entry point.
    "payload",
)
KAGGLE_OPTIONAL_CONFIG_KEYS = (
    #: A hand-written prepared-campaign document. Optional: without it the
    #: provider assembles the prepared campaign from the manifest's own declared
    #: inputs, so a campaign can be dispatched from its declaration alone.
    "prepared_path",
    "owner",
    "input_paths",
    "model_commit",
    "pip_extras",
    "projection_tolerance",
    "declared_quota_ceiling_gpu_hours",
)
_KAGGLE_CONFIG_KEYS = frozenset(KAGGLE_REQUIRED_CONFIG_KEYS + KAGGLE_OPTIONAL_CONFIG_KEYS)

_CONFIG_KEYS_BY_PROVIDER: Mapping[str, frozenset[str]] = {
    PROVIDER_LOCAL: _LOCAL_CONFIG_KEYS,
    PROVIDER_UNSLOTH: _UNSLOTH_CONFIG_KEYS,
    PROVIDER_KAGGLE: _KAGGLE_CONFIG_KEYS,
    PROVIDER_AUTO: _AUTO_CONFIG_KEYS,
}


class TrainingBackendRefusal(RuntimeError):
    """A declared backend cannot be honored. The code comes first."""

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(f"{code}: {reason}")
        self.code = code
        self.reason = reason


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingBackendRefusal(
            TRAINING_BACKEND_SCHEMA, f"{label} must be a non-empty string, got {value!r}"
        )
    return value


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingBackendRefusal(
            TRAINING_BACKEND_SCHEMA, f"{label} must be a number, got {value!r}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise TrainingBackendRefusal(
            TRAINING_BACKEND_SCHEMA, f"{label} must be finite, got {value!r}"
        )
    if minimum is not None and number < minimum:
        raise TrainingBackendRefusal(
            TRAINING_BACKEND_SCHEMA, f"{label} cannot be below {minimum}, got {number!r}"
        )
    return number


# --------------------------------------------------------------------------
# the declaration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingBackendDeclaration:
    """One campaign's declared execution backend, and its backend-specific config.

    Distinct from the project template's ``backend.type``: the template selects
    the *trainer engine inside* a backend, this declares *where and how* the
    campaign executes. Provider values are closed; unknown providers and
    unknown config keys both refuse, because a declaration nothing reads is the
    same defect class as a declaration that silently defaults.
    """

    provider: str = PROVIDER_LOCAL
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._checked(self.provider, self.config, source="<training_backend>")
        if self.provider == PROVIDER_AUTO:
            # Fail closed at load: an auto declaration whose candidate list is
            # empty or meaningless could never choose a backend, and refusing it
            # at the first attempt would be a late way of learning that.
            self.candidate_order()

    @staticmethod
    def _checked(
        provider: Any, config: Any, *, source: str
    ) -> dict[str, Any]:
        if not isinstance(provider, str) or provider not in PROVIDERS:
            raise TrainingBackendRefusal(
                TRAINING_BACKEND_UNKNOWN_PROVIDER,
                f"{source}: provider {provider!r} is not one of "
                f"{sorted(PROVIDERS)}; a backend is declared, never inferred from "
                "what happens to be installed",
            )
        if config is None:
            config = {}
        if not isinstance(config, Mapping):
            raise TrainingBackendRefusal(
                TRAINING_BACKEND_SCHEMA,
                f"{source}: config must be an object, got {type(config).__name__}",
            )
        allowed = _CONFIG_KEYS_BY_PROVIDER[provider]
        unknown = sorted(str(key) for key in config if str(key) not in allowed)
        if unknown:
            raise TrainingBackendRefusal(
                TRAINING_BACKEND_UNSUPPORTED_CONFIG,
                f"{source}: provider {provider!r} does not declare {unknown}; "
                f"known keys are {sorted(allowed)}. An unsupported key would "
                "silently change what the backend is told to do",
            )
        return {str(key): value for key, value in config.items()}

    @classmethod
    def from_mapping(
        cls, document: Any, *, source: str = "<memory>"
    ) -> "TrainingBackendDeclaration":
        if document is None:
            return cls()
        if not isinstance(document, Mapping):
            raise TrainingBackendRefusal(
                TRAINING_BACKEND_SCHEMA,
                f"{source}: training_backend must be an object with a provider, "
                f"got {type(document).__name__}",
            )
        unknown = sorted(str(key) for key in document if key not in {"provider", "config"})
        if unknown:
            raise TrainingBackendRefusal(
                TRAINING_BACKEND_SCHEMA,
                f"{source}: unknown training_backend fields {unknown}; provider-"
                "specific settings belong under 'config'",
            )
        provider = document.get("provider", PROVIDER_LOCAL)
        config = cls._checked(provider, document.get("config", {}), source=source)
        return cls(provider=str(provider), config=config)

    def to_dict(self) -> dict[str, Any]:
        return {"provider": self.provider, "config": dict(self.config)}

    def candidate_order(self) -> tuple[str, ...]:
        """The providers ``auto`` tries, in the declared order."""
        raw = self.config.get("candidates")
        if raw is None:
            return AUTO_DEFAULT_CANDIDATES
        if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
            raise TrainingBackendRefusal(
                TRAINING_BACKEND_SCHEMA,
                "training_backend.config.candidates must be a list of providers, "
                f"got {raw!r}",
            )
        order: list[str] = []
        for entry in raw:
            if not isinstance(entry, str) or entry not in PROVIDERS - {PROVIDER_AUTO}:
                raise TrainingBackendRefusal(
                    TRAINING_BACKEND_UNKNOWN_PROVIDER,
                    f"training_backend.config.candidates names {entry!r}, which is "
                    f"not one of {sorted(PROVIDERS - {PROVIDER_AUTO})}",
                )
            if entry not in order:
                order.append(entry)
        if not order:
            raise TrainingBackendRefusal(
                TRAINING_BACKEND_SCHEMA,
                "training_backend.config.candidates is empty; an auto declaration "
                "with no candidates can never choose a backend",
            )
        return tuple(order)


def backend_declaration(manifest: Any) -> TrainingBackendDeclaration:
    """The manifest's declared backend, or the historical default (``local``)."""
    declaration = getattr(manifest, "training_backend", None)
    if declaration is None:
        return TrainingBackendDeclaration()
    if not isinstance(declaration, TrainingBackendDeclaration):
        raise TrainingBackendRefusal(
            TRAINING_BACKEND_SCHEMA,
            "manifest.training_backend must be a TrainingBackendDeclaration, got "
            f"{type(declaration).__name__}",
        )
    return declaration


# --------------------------------------------------------------------------
# the preflight panel
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DevicePanel:
    """One visible accelerator, as the device itself reported it."""

    index: int
    name: str
    compute_capability: str
    vram_gb: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "compute_capability": self.compute_capability,
            "vram_gb": self.vram_gb,
        }


@dataclass(frozen=True)
class PreflightPanel:
    """What the declared backend can see before any compute is admitted.

    Device identity/VRAM come from the framework; system and available RAM and
    the free space on the state root's volume come from the operating system.
    Step timings are deliberately *not* here: those are
    :func:`chowder.growth.campaign_prepare.probe_hardware`'s measurement, and
    labelling a device probe as a step measurement would be the same class of
    dishonesty as a copied benchmark row.
    """

    provider: str
    devices: tuple[DevicePanel, ...] = ()
    cuda_available: bool = False
    cuda_runtime: str | None = None
    framework_version: str | None = None
    system_ram_gb: float | None = None
    available_ram_gb: float | None = None
    disk_path: str | None = None
    disk_free_gb: float | None = None
    measurement_method: str = ""
    warnings: tuple[str, ...] = ()

    @property
    def device_count(self) -> int:
        return len(self.devices)

    @property
    def total_vram_gb(self) -> float:
        return round(sum(device.vram_gb for device in self.devices), 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "devices": [device.to_dict() for device in self.devices],
            "device_count": self.device_count,
            "total_vram_gb": self.total_vram_gb,
            "cuda_available": self.cuda_available,
            "cuda_runtime": self.cuda_runtime,
            "framework_version": self.framework_version,
            "system_ram_gb": self.system_ram_gb,
            "available_ram_gb": self.available_ram_gb,
            "disk_path": self.disk_path,
            "disk_free_gb": self.disk_free_gb,
            "measurement_method": self.measurement_method,
            "warnings": list(self.warnings),
        }


_MEMORY_STATUSEX_FIELDS = (
    ("dwLength", ctypes.c_ulong),
    ("dwMemoryLoad", ctypes.c_ulong),
    ("ullTotalPhys", ctypes.c_ulonglong),
    ("ullAvailPhys", ctypes.c_ulonglong),
    ("ullTotalPageFile", ctypes.c_ulonglong),
    ("ullAvailPageFile", ctypes.c_ulonglong),
    ("ullTotalVirtual", ctypes.c_ulonglong),
    ("ullAvailVirtual", ctypes.c_ulonglong),
    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
)


class _MemoryStatusEx(ctypes.Structure):
    """Windows' MEMORYSTATUSEX, so system RAM needs no third-party dependency."""

    _fields_ = list(_MEMORY_STATUSEX_FIELDS)


def _windows_memory_bytes() -> tuple[float | None, float | None]:
    try:
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None, None
        return float(status.ullTotalPhys), float(status.ullAvailPhys)
    except Exception:
        return None, None


def _posix_memory_bytes() -> tuple[float | None, float | None]:
    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        return float(total), float(available)
    except (AttributeError, ValueError, OSError):
        return None, None


def _system_memory_bytes() -> tuple[float | None, float | None]:
    """(total, available) system RAM in bytes, or ``(None, None)`` unreadable."""
    if sys.platform.startswith("win"):
        return _windows_memory_bytes()
    return _posix_memory_bytes()


def _disk_free_bytes(path: str | Path | None) -> tuple[str | None, float | None]:
    """(existing ancestor, free bytes) for a path, or ``(None, None)``."""
    candidate = Path(path) if path is not None else Path.cwd()
    try:
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        usage = shutil.disk_usage(str(candidate))
    except OSError:
        return None, None
    return str(candidate), float(usage.free)


_UNSET = object()


def probe_local_panel(
    *,
    disk_path: str | Path | None = None,
    torch_module: Any = _UNSET,
    memory_reader: Callable[[], tuple[float | None, float | None]] | None = None,
    disk_reader: Callable[[str | Path | None], tuple[str | None, float | None]] | None = None,
) -> PreflightPanel:
    """Build the local device panel from the framework and the OS.

    Every seam is injectable (the framework module, the memory reader, the disk
    reader), so the panel's shape is provable on a machine with no accelerator
    at all. Nothing here refuses: an empty device list is a fact the *preflight
    verdict* acts on, and reporting it is what lets an operator see why.
    """
    if torch_module is _UNSET:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except Exception as error:  # pragma: no cover - depends on the environment
            torch_module = None
            unimportable = error
        else:
            unimportable = None
    else:
        unimportable = None

    warnings: list[str] = []
    devices: list[DevicePanel] = []
    cuda_available = False
    cuda_runtime: str | None = None
    framework_version: str | None = None

    if torch_module is None:
        warnings.append(
            f"the framework is not importable ({unimportable}), so no device "
            "could be probed"
        )
    else:
        framework_version = str(getattr(torch_module, "__version__", "") or "") or None
        version_module = getattr(torch_module, "version", None)
        runtime = getattr(version_module, "cuda", None) if version_module else None
        cuda_runtime = str(runtime) if runtime else None
        try:
            cuda_available = bool(torch_module.cuda.is_available())
        except Exception as error:  # pragma: no cover - depends on the environment
            cuda_available = False
            warnings.append(f"the framework's device probe raised: {error}")
        if cuda_available:
            count = int(torch_module.cuda.device_count())
            for index in range(count):
                properties = torch_module.cuda.get_device_properties(index)
                total = float(getattr(properties, "total_memory", 0.0))
                major = getattr(properties, "major", None)
                minor = getattr(properties, "minor", None)
                capability = (
                    f"{int(major)}.{int(minor)}"
                    if major is not None and minor is not None
                    else ""
                )
                devices.append(
                    DevicePanel(
                        index=index,
                        name=str(getattr(properties, "name", f"device-{index}")),
                        compute_capability=capability,
                        vram_gb=round(total / (1024**3), 3),
                    )
                )
            if not devices:
                warnings.append(
                    "the framework reports CUDA available but enumerated no device"
                )
        else:
            warnings.append(
                "no CUDA device is visible to the framework; only an explicit "
                "operator override can admit a campaign on this machine"
            )

    read_memory = memory_reader or _system_memory_bytes
    total_ram, available_ram = read_memory()
    if total_ram is None:
        warnings.append("system RAM could not be read from the operating system")

    read_disk = disk_reader or _disk_free_bytes
    disk_path_text, disk_free = read_disk(disk_path)
    if disk_free is None:
        warnings.append("free disk space could not be read on the state root's volume")

    return PreflightPanel(
        provider=PROVIDER_LOCAL,
        devices=tuple(devices),
        cuda_available=cuda_available,
        cuda_runtime=cuda_runtime,
        framework_version=framework_version,
        system_ram_gb=(
            round(total_ram / (1024**3), 3) if total_ram is not None else None
        ),
        available_ram_gb=(
            round(available_ram / (1024**3), 3) if available_ram is not None else None
        ),
        disk_path=disk_path_text,
        disk_free_gb=(
            round(disk_free / (1024**3), 3) if disk_free is not None else None
        ),
        measurement_method=(
            "chowder.growth.training_backends.local-panel.v1: accelerator "
            "identity and VRAM from the framework, system RAM and free disk "
            "from the operating system; not a step-timing measurement"
        ),
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class PreflightResult:
    """Whether a provider may run this campaign, and what it saw."""

    provider: str
    admitted: bool
    code: str | None
    reason: str
    panel: PreflightPanel
    strategy: str = "undeclared"
    overrides: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def refusal(self) -> tuple[str, str] | None:
        if self.admitted:
            return None
        return (self.code or TRAINING_BACKEND_SCHEMA, self.reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "admitted": self.admitted,
            "refusal_code": self.code,
            "refusal_reason": self.reason,
            "strategy": self.strategy,
            "overrides": list(self.overrides),
            "notes": list(self.notes),
            "panel": self.panel.to_dict(),
        }


# --------------------------------------------------------------------------
# the per-recipe estimate
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendEstimate:
    """What one recipe would need on this backend, or why that is unknown.

    ``available=False`` is an answer, not a failure: an estimate that could not
    be computed from declared material is reported as unavailable rather than
    filled with an invented number.
    """

    provider: str
    recipe_id: str
    strategy: str
    available: bool
    reason: str = ""
    model_bytes: int | None = None
    adapter_bytes: int | None = None
    optimizer_bytes: int | None = None
    activation_bytes: int | None = None
    total_bytes: int | None = None
    total_vram_bytes: int | None = None
    safe_sequence_length: int | None = None
    requires_offload: bool | None = None
    assumptions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "recipe_id": self.recipe_id,
            "strategy": self.strategy,
            "available": self.available,
            "reason": self.reason,
            "model_bytes": self.model_bytes,
            "adapter_bytes": self.adapter_bytes,
            "optimizer_bytes": self.optimizer_bytes,
            "activation_bytes": self.activation_bytes,
            "total_bytes": self.total_bytes,
            "total_vram_bytes": self.total_vram_bytes,
            "safe_sequence_length": self.safe_sequence_length,
            "requires_offload": self.requires_offload,
            "assumptions": list(self.assumptions),
        }


# --------------------------------------------------------------------------
# capability matrix
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CapabilityRow:
    """One declared capability and where its decision actually lives."""

    capability: str
    status: str
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in CAPABILITY_STATUSES:
            raise TrainingBackendRefusal(
                TRAINING_BACKEND_SCHEMA,
                f"capability {self.capability!r} has unknown status {self.status!r}; "
                f"known statuses are {sorted(CAPABILITY_STATUSES)}",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BackendCapabilities:
    """What one backend can do, declared in code rather than discovered at run."""

    provider: str
    trainer: str
    rows: tuple[CapabilityRow, ...] = ()

    def status(self, capability: str) -> str | None:
        for row in self.rows:
            if row.capability == capability:
                return row.status
        return None

    def refused_rows(self) -> tuple[CapabilityRow, ...]:
        return tuple(row for row in self.rows if row.status == CAPABILITY_REFUSED)

    def refusal(self, capability: str) -> tuple[str, str] | None:
        """``(code, reason)`` when the capability is refused, else ``None``."""
        status = self.status(capability)
        if status != CAPABILITY_REFUSED:
            return None
        return (
            TRAINING_BACKEND_UNSUPPORTED_CAPABILITY,
            f"{self.provider} refuses {capability!r}: it is declared unsupported "
            "for this backend, and a silent no-op would misrepresent what ran",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "trainer": self.trainer,
            "rows": [row.to_dict() for row in self.rows],
        }


# --------------------------------------------------------------------------
# the uniform outcome
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class UniformAttemptOutcome:
    """One attempt's canonical outcome, whatever backend produced it.

    Wraps :class:`AttemptOutcome` (attempt/candidate/intervention identity,
    source commit, artifact + digest, measured compute, settlement, status) and
    adds the fields the spec names beyond it (backend and version, wall time,
    warnings, training metrics, evidence references). ``to_evidence`` renders
    the whole thing in the growth loop's existing vocabulary, so no consumer
    branches on the backend.
    """

    backend: str
    backend_version: str
    outcome: AttemptOutcome
    candidate_id: str = ""
    intervention_id: str = ""
    parent_checkpoint: str | None = None
    training_metrics: Mapping[str, Any] = field(default_factory=dict)
    wall_seconds: float | None = None
    warnings: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        return self.outcome.status

    @property
    def failure_class(self) -> str | None:
        return self.outcome.failure_class

    @property
    def attempt_id(self) -> str:
        return self.outcome.source.attempt_id

    def to_evidence(self) -> dict[str, Any]:
        evidence = self.outcome.to_evidence()
        evidence["backend"] = {
            "provider": self.backend,
            "version": self.backend_version,
            "parent_checkpoint": self.parent_checkpoint,
        }
        evidence["candidate_id"] = self.candidate_id
        evidence["intervention_id"] = self.intervention_id
        evidence["training_metrics"] = dict(self.training_metrics)
        evidence["wall_seconds"] = self.wall_seconds
        evidence["warnings"] = list(self.warnings)
        evidence["evidence_refs"] = list(self.evidence_refs)
        return evidence

    def to_dict(self) -> dict[str, Any]:
        return self.to_evidence()


def _settlement_from_evidence(document: Any) -> SettlementVerdict | None:
    if not isinstance(document, Mapping):
        return None
    return SettlementVerdict(
        compliant=bool(document.get("budget_compliant", False)),
        failure_reasons=tuple(
            str(reason) for reason in document.get("budget_failure_reasons", ()) or ()
        ),
    )


def _artifacts_from_evidence(document: Mapping[str, Any]) -> tuple[ArtifactEntry, ...]:
    entries: list[ArtifactEntry] = []
    manifest = document.get("artifact_manifest")
    if isinstance(manifest, Sequence) and not isinstance(manifest, (str, bytes)):
        for entry in manifest:
            if not isinstance(entry, Mapping):
                continue
            path = entry.get("path")
            sha256 = entry.get("sha256")
            if isinstance(path, str) and path and isinstance(sha256, str) and sha256:
                entries.append(
                    ArtifactEntry(
                        path=path, sha256=sha256, bytes=int(entry.get("bytes", 0))
                    )
                )
    if not entries:
        reference = document.get("artifact_ref")
        digest = document.get("artifact_sha256")
        if isinstance(reference, str) and reference and isinstance(digest, str) and digest:
            entries.append(ArtifactEntry(path=reference, sha256=digest, bytes=0))
    return tuple(entries)


def uniform_outcome_from_evidence(
    evidence: Mapping[str, Any],
    *,
    source: SourceBinding,
    backend: str,
    backend_version: str,
    candidate_id: str = "",
    intervention_id: str = "",
    parent_checkpoint: str | None = None,
    wall_seconds: float | None = None,
    warnings: Sequence[str] = (),
    evidence_refs: Sequence[str] = (),
) -> UniformAttemptOutcome:
    """Map a ``TrainingFn``'s evidence mapping onto the uniform outcome.

    The status vocabulary, the artifacts, the cost and the settlement are the
    ones the executor already wrote; nothing here re-measures anything. A
    failure is classified with :func:`attempt_failure.classify_failure`, the
    same function the cycle uses, so a backend cannot classify its own
    failures differently.
    """
    status = evidence.get("status")
    if status not in OUTCOME_STATUSES:
        raise TrainingBackendRefusal(
            TRAINING_BACKEND_EVIDENCE_STATUS_UNKNOWN,
            f"attempt evidence declares status {status!r}, which is not one of "
            f"{sorted(OUTCOME_STATUSES)}; an outcome with no status cannot settle, "
            "classify or advance",
        )
    cost_document = evidence.get("compute_cost")
    if isinstance(cost_document, Mapping):
        cost: ComputeCost | None = ComputeCost.from_dict(cost_document)
    else:
        measured = evidence.get("measured_gpu_hours")
        cost = (
            ComputeCost.from_wall_only(float(measured), source="executor-summary")
            if isinstance(measured, (int, float)) and not isinstance(measured, bool)
            else None
        )
    failure_class: str | None = None
    if status != OUTCOME_SUCCEEDED:
        failure_class = classify_failure(evidence).failure_class.value
    environment = evidence.get("environment")
    training_metrics = evidence.get("candidate_metrics")
    refusal_code = evidence.get("refused_by")
    outcome = AttemptOutcome(
        source=source,
        status=str(status),
        artifacts=_artifacts_from_evidence(evidence),
        cost=cost,
        settlement=_settlement_from_evidence(evidence.get("budget_settlement")),
        failure_class=failure_class,
        refusal_code=(
            str(refusal_code)
            if isinstance(refusal_code, str) and refusal_code
            else None
        ),
        refusal_reason=str(evidence.get("refusal_reason") or ""),
        environment=dict(environment) if isinstance(environment, Mapping) else {},
        resume_state=str(evidence.get("resume_state") or "not-applicable"),
        job_id=str(evidence.get("job_id") or ""),
        source_commit_sha=str(evidence.get("source_commit_sha") or ""),
    )
    return UniformAttemptOutcome(
        backend=backend,
        backend_version=backend_version,
        outcome=outcome,
        candidate_id=candidate_id,
        intervention_id=intervention_id,
        parent_checkpoint=parent_checkpoint,
        training_metrics=(
            dict(training_metrics) if isinstance(training_metrics, Mapping) else {}
        ),
        wall_seconds=wall_seconds,
        warnings=tuple(str(warning) for warning in warnings),
        evidence_refs=tuple(str(reference) for reference in evidence_refs),
    )


# --------------------------------------------------------------------------
# shared attempt-lifecycle helpers
# --------------------------------------------------------------------------


def collect_attempt_evidence(manifest: Any, attempt_dir: str | Path) -> Mapping[str, Any]:
    """Read one attempt's evidence document, or refuse it could not be read."""
    path = Path(attempt_dir) / EVIDENCE_FILE
    if not path.is_file():
        raise TrainingBackendRefusal(
            TRAINING_BACKEND_EVIDENCE_MISSING,
            f"attempt directory {attempt_dir} has no {EVIDENCE_FILE}; an attempt "
            "that wrote no evidence produced no recordable result",
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise TrainingBackendRefusal(
            TRAINING_BACKEND_EVIDENCE_INVALID,
            f"evidence at {path} could not be read: {error}",
        ) from error
    if not isinstance(document, Mapping):
        raise TrainingBackendRefusal(
            TRAINING_BACKEND_EVIDENCE_INVALID,
            f"evidence at {path} is not a JSON object",
        )
    return dict(document)


def verify_attempt_evidence(
    manifest: Any, attempt_dir: str | Path
) -> tuple[str, str] | None:
    """Re-verify one attempt's evidence before a consumer references it.

    Checks that the evidence exists, parses, carries a known status, and -- when
    it names an artifact with a digest and that artifact is present locally --
    that the bytes still hash to the declared digest. An artifact that lives
    only on the remote side is left to the transport's own admission, which is
    where its bytes are verified.
    """
    try:
        document = collect_attempt_evidence(manifest, attempt_dir)
    except TrainingBackendRefusal as refusal:
        return refusal.code, refusal.reason
    status = document.get("status")
    if status not in OUTCOME_STATUSES:
        return (
            TRAINING_BACKEND_EVIDENCE_STATUS_UNKNOWN,
            f"evidence at {attempt_dir} declares status {status!r}, which is not "
            f"one of {sorted(OUTCOME_STATUSES)}",
        )
    reference = document.get("artifact_ref")
    digest = document.get("artifact_sha256")
    if isinstance(reference, str) and reference and isinstance(digest, str) and digest:
        candidate = Path(attempt_dir) / reference
        if candidate.is_file():
            actual, _size = file_digest(candidate)
            if actual != digest.lower():
                return (
                    TRAINING_BACKEND_ARTIFACT_DIGEST_MISMATCH,
                    f"artifact {reference!r} hashes {actual}, but the evidence "
                    f"declares {digest}; a moved artifact is not the artifact the "
                    "attempt produced",
                )
    return None


def _resume_recipe(
    manifest: Any, recipe: TrainingRecipe, checkpoint: str | Path, *, paths_are_local: bool
) -> TrainingRecipe:
    """The recipe that continues from ``checkpoint``, or a named refusal."""
    text = str(checkpoint).strip()
    if not text:
        raise TrainingBackendRefusal(
            TRAINING_BACKEND_RESUME_CHECKPOINT_UNDECLARED,
            "resume requires the checkpoint identity to continue from; an empty "
            "checkpoint would silently restart the run",
        )
    if paths_are_local:
        path = Path(text)
        if not path.is_dir():
            raise TrainingBackendRefusal(
                TRAINING_BACKEND_RESUME_CHECKPOINT_MISSING,
                f"checkpoint {path} is not a directory on this machine; the "
                "continuation cannot be verified, so it is refused rather than "
                "started from the wrong place",
            )
        text = str(path)
    return replace(recipe, resume_from_checkpoint=text)


def _load_template(manifest: Any) -> Mapping[str, Any] | None:
    """The declared project template, or ``None`` when it is not declared."""
    raw = str(getattr(manifest, "project_template_path", "") or "").strip()
    if not raw:
        return None
    try:
        document = json.loads(Path(raw).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise TrainingBackendRefusal(
            TRAINING_BACKEND_TEMPLATE_INVALID,
            f"project template {raw} could not be read: {error}",
        ) from error
    if not isinstance(document, Mapping):
        raise TrainingBackendRefusal(
            TRAINING_BACKEND_TEMPLATE_INVALID,
            f"project template {raw} is not a JSON object",
        )
    return document


def _template_backend(template: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(template, Mapping):
        return {}
    backend = template.get("backend")
    return backend if isinstance(backend, Mapping) else {}


def _training_strategy(template: Mapping[str, Any] | None) -> str:
    """The expected strategy, read from the declared template -- never assumed."""
    backend = _template_backend(template)
    if not backend:
        return "undeclared"
    backend_type = str(backend.get("type", "") or "")
    engine = str(backend.get("engine", "") or "")
    quantization = str(backend.get("quantization", "") or "")
    parts = [part for part in (backend_type, engine) if part]
    if quantization:
        parts.append(quantization)
    return "/".join(parts) if parts else "undeclared"


def _requested_knobs(template: Mapping[str, Any] | None) -> tuple[str, ...]:
    """The Chowder mechanisms the declared template turns on, if any."""
    backend = _template_backend(template)
    training = backend.get("training")
    training = training if isinstance(training, Mapping) else {}
    requested: list[str] = []
    for knob in UNSLOTH_REFUSED_KNOBS:
        raw = training.get(knob, "off")
        if isinstance(raw, bool):
            on = raw
        else:
            on = str(raw).strip().lower() not in {"", "off", "false", "none"}
        if on:
            requested.append(knob)
    return tuple(requested)


def _envelope(manifest: Any) -> GrowthEnvelope:
    """The manifest's per-recipe ceilings -- one owner of that arithmetic."""
    from .campaign_runner import envelope_for

    return envelope_for(manifest)


def _device_override(declaration: TrainingBackendDeclaration) -> str:
    raw = declaration.config.get("device", "auto")
    if not isinstance(raw, str) or raw.strip().lower() not in DEVICE_CHOICES:
        raise TrainingBackendRefusal(
            TRAINING_BACKEND_UNSUPPORTED_CONFIG,
            f"training_backend.config.device must be one of {list(DEVICE_CHOICES)}, "
            f"got {raw!r}",
        )
    return raw.strip().lower()


# --------------------------------------------------------------------------
# memory estimation from declared material
# --------------------------------------------------------------------------

_ESTIMATE_ASSUMPTIONS = (
    "parameter count from the base model's own config.json using the standard "
    "decoder layout (vocab embedding, per-layer attention projections and a "
    "gated MLP); an architecture that differs from that layout will differ from "
    "this estimate",
    "activations are approximated as batch x sequence x hidden x layers x 2 "
    "bytes, doubled for the backward pass",
    "LoRA adapter parameters are approximated as 2 x rank x hidden x targets x "
    "layers (the A and B matrices), with fp32 optimizer state for the adapter",
)

_SAFE_SEQUENCE_LENGTHS = (512, 1024, 2048, 4096, 8192, 16384)


def _model_config(manifest: Any) -> Mapping[str, Any] | None:
    base = str(getattr(manifest, "base_model_path", "") or "").strip()
    if not base:
        return None
    candidate = Path(base)
    config_path = candidate / "config.json" if candidate.is_dir() else candidate
    if not config_path.is_file():
        return None
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, Mapping) else None


def _parameter_count(config: Mapping[str, Any]) -> tuple[int, int, int, int] | None:
    """(params, hidden, layers, targets-placeholder) from a declared config."""
    try:
        hidden = int(config["hidden_size"])
        layers = int(config["num_hidden_layers"])
        vocab = int(config.get("vocab_size", 0))
        intermediate = int(config.get("intermediate_size", 4 * hidden))
    except (KeyError, TypeError, ValueError):
        return None
    if hidden <= 0 or layers <= 0:
        return None
    if intermediate <= 0:
        intermediate = 4 * hidden
    per_layer = 4 * hidden * hidden + 2 * hidden * intermediate
    params = vocab * hidden + layers * per_layer
    return params, hidden, layers, intermediate


def _bytes_per_parameter(template: Mapping[str, Any] | None) -> float:
    backend = _template_backend(template)
    quantization = str(backend.get("quantization", "") or "").strip().lower()
    if quantization in {"4bit", "nf4", "int4"}:
        return 0.5
    return 2.0


def _estimate_local(
    manifest: Any,
    recipe: TrainingRecipe,
    panel: PreflightPanel,
    *,
    template: Mapping[str, Any] | None,
) -> BackendEstimate:
    strategy = _training_strategy(template)
    config = _model_config(manifest)
    if config is None:
        return BackendEstimate(
            provider=PROVIDER_LOCAL,
            recipe_id=recipe.recipe_id,
            strategy=strategy,
            available=False,
            reason=(
                "the base model path declares no readable config.json, so the "
                "model, optimizer, adapter and activation estimates would be "
                "invented numbers; declare the model or measure it separately"
            ),
        )
    counted = _parameter_count(config)
    if counted is None:
        return BackendEstimate(
            provider=PROVIDER_LOCAL,
            recipe_id=recipe.recipe_id,
            strategy=strategy,
            available=False,
            reason=(
                "the base model's config.json declares no hidden_size/"
                "num_hidden_layers, so the parameter count cannot be derived"
            ),
        )
    params, hidden, layers, _intermediate = counted
    bytes_per_parameter = _bytes_per_parameter(template)

    model_bytes = int(params * bytes_per_parameter)
    if recipe.lora_rank > 0:
        targets = max(1, len(recipe.target_modules))
        adapter_params = int(2 * recipe.lora_rank * hidden * targets * layers)
        adapter_bytes = int(adapter_params * 2)
        optimizer_bytes = int(adapter_params * 8)
    else:
        adapter_bytes = 0
        optimizer_bytes = int(params * 12)
    base_strategy = "lora" if recipe.lora_rank > 0 else "full-finetune"
    if bytes_per_parameter < 2.0 and recipe.lora_rank > 0:
        base_strategy = "qlora"
    strategy = (
        f"{base_strategy} ({strategy})" if strategy != "undeclared" else base_strategy
    )

    def activation_bytes_for(sequence_length: int) -> int:
        return int(
            max(1, recipe.batch_size)
            * sequence_length
            * hidden
            * layers
            * 2
            * 2
        )

    activation_bytes = activation_bytes_for(recipe.seq_len)
    total_bytes = model_bytes + adapter_bytes + optimizer_bytes + activation_bytes

    vram_bytes = int(panel.total_vram_gb * (1024**3)) if panel.devices else 0
    safe_sequence_length: int | None = None
    requires_offload: bool | None = None
    if panel.devices:
        requires_offload = total_bytes > vram_bytes
        headroom = int(vram_bytes * 0.9)
        for sequence_length in _SAFE_SEQUENCE_LENGTHS:
            candidate_total = (
                model_bytes
                + adapter_bytes
                + optimizer_bytes
                + activation_bytes_for(sequence_length)
            )
            if candidate_total <= headroom:
                safe_sequence_length = sequence_length
            else:
                break
    return BackendEstimate(
        provider=PROVIDER_LOCAL,
        recipe_id=recipe.recipe_id,
        strategy=strategy,
        available=True,
        model_bytes=model_bytes,
        adapter_bytes=adapter_bytes,
        optimizer_bytes=optimizer_bytes,
        activation_bytes=activation_bytes,
        total_bytes=total_bytes,
        total_vram_bytes=vram_bytes or None,
        safe_sequence_length=safe_sequence_length,
        requires_offload=requires_offload,
        assumptions=_ESTIMATE_ASSUMPTIONS,
    )


# --------------------------------------------------------------------------
# the provider protocol
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OverheadReport:
    """A provider's declared attach overhead, and what that number is worth.

    ``hours`` is the launch/attach cost the provider itself can account for
    (a local subprocess starts in place; a remote kernel has to be pushed,
    installed, polled and pulled). ``None`` means *not measured* -- and an
    unmeasured overhead is deliberately not ranked as if it were zero, because
    that would make the cheapest-looking provider the one nobody measured.
    """

    hours: float | None
    basis: str

    def to_dict(self) -> dict[str, Any]:
        return {"hours": self.hours, "basis": self.basis}


@runtime_checkable
class TrainingBackend(Protocol):
    """The declared execution backend the campaign runner dispatches through.

    Every method answers a question the campaign already asks, so no consumer
    branches on *which* backend answered it: capabilities (what this backend
    supports), preflight (what it can see), estimate (what one recipe needs),
    admit (whether the growth envelope admits it), build_training_fn (the
    execution seam), and the attempt lifecycle (resume, collect, verify).
    """

    provider: str
    trainer: str

    def capabilities(self) -> BackendCapabilities:
        """What this backend supports, and where each decision lives."""

    def preflight(self, manifest: Any) -> PreflightResult:
        """Whether this backend may run the campaign, before any compute."""

    def estimate(self, manifest: Any, recipe: TrainingRecipe) -> BackendEstimate:
        """What one recipe would need on this backend."""

    def projected_overhead(self, manifest: Any) -> OverheadReport:
        """This provider's declared attach overhead, with its basis."""

    def admit(self, manifest: Any, recipe: TrainingRecipe) -> tuple[str, str] | None:
        """``None`` when the recipe is admitted, else ``(kind, reason)``."""

    def build_training_fn(
        self, manifest: Any, *, state_root: Any = None, runner: Any = None
    ) -> Any:
        """The ``TrainingFn`` the cycle executes through."""

    def resume(
        self, manifest: Any, recipe: TrainingRecipe, checkpoint: str | Path
    ) -> TrainingRecipe:
        """The recipe that continues from ``checkpoint``, or a named refusal."""

    def collect(self, manifest: Any, attempt_dir: str | Path) -> Mapping[str, Any]:
        """One attempt's evidence document."""

    def verify(
        self, manifest: Any, attempt_dir: str | Path
    ) -> tuple[str, str] | None:
        """``None`` when the attempt's evidence verifies, else ``(code, reason)``."""


class _SharedAttemptLifecycle:
    """The lifecycle helpers every provider shares, unchanged."""

    provider: str = ""
    trainer: str = ""

    def collect(self, manifest: Any, attempt_dir: str | Path) -> Mapping[str, Any]:
        return collect_attempt_evidence(manifest, attempt_dir)

    def verify(self, manifest: Any, attempt_dir: str | Path) -> tuple[str, str] | None:
        return verify_attempt_evidence(manifest, attempt_dir)


# --------------------------------------------------------------------------
# LOCAL
# --------------------------------------------------------------------------


class LocalTrainingBackend(_SharedAttemptLifecycle):
    """The production CLI on this machine, through the growth binding."""

    provider = PROVIDER_LOCAL
    trainer = "transformers-peft"

    def __init__(
        self,
        *,
        probe: Callable[..., PreflightPanel] | None = None,
    ) -> None:
        #: The panel probe is injectable so admission on a machine with no
        #: accelerator is provable without one.
        self.probe = probe or probe_local_panel

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            provider=self.provider,
            trainer=self.trainer,
            rows=(
                CapabilityRow(
                    "lora",
                    CAPABILITY_SUPPORTED,
                    "PEFT LoRA through the production transformers executor",
                ),
                CapabilityRow(
                    "qlora",
                    CAPABILITY_SUPPORTED,
                    "4-bit base weights; requires the qlora extra",
                ),
                CapabilityRow(
                    "bf16_fp16",
                    CAPABILITY_SUPPORTED,
                    "resolved from the declared template",
                ),
                CapabilityRow(
                    "gradient_accumulation",
                    CAPABILITY_SUPPORTED,
                    "a recipe field the template consumes",
                ),
                CapabilityRow(
                    "gradient_checkpointing",
                    CAPABILITY_SUPPORTED,
                    "template training knob",
                ),
                CapabilityRow(
                    "checkpoint_resume",
                    CAPABILITY_SUPPORTED,
                    "resume_from_checkpoint travels in the recipe and the "
                    "transformers checkpoint manifest verifies it",
                ),
                CapabilityRow(
                    "activation_offload",
                    CAPABILITY_SUPPORTED,
                    "chowder's own mechanism, resolved by the transformers executor",
                ),
                CapabilityRow(
                    "optimizer_tiering",
                    CAPABILITY_SUPPORTED,
                    "chowder's own mechanism, resolved by the transformers executor",
                ),
                CapabilityRow(
                    "frozen_layer_streaming",
                    CAPABILITY_SUPPORTED,
                    "chowder's own mechanism, resolved by the transformers executor",
                ),
                CapabilityRow(
                    "successive_halving",
                    CAPABILITY_SUPPORTED,
                    "executed by the campaign runner over the declared recipes",
                ),
                CapabilityRow(
                    "search",
                    CAPABILITY_SUPPORTED,
                    "the declared bounded candidate search",
                ),
            ),
        )

    def preflight(self, manifest: Any) -> PreflightResult:
        panel = self.probe(disk_path=getattr(manifest, "state_root", None))
        template = _load_template(manifest)
        strategy = _training_strategy(template)
        override = _device_override(backend_declaration(manifest))
        overrides: list[str] = []
        if override == "cpu":
            overrides.append("device=cpu")
            return PreflightResult(
                provider=self.provider,
                admitted=True,
                code=None,
                reason=(
                    "the campaign declares device=cpu, so the accelerator "
                    "requirement is deliberately overridden by the operator"
                ),
                panel=panel,
                strategy=strategy,
                overrides=tuple(overrides),
                notes=(
                    "device time cannot be measured on a CPU path, so a declared "
                    "device ceiling is unsettled rather than satisfied",
                ),
            )
        if override == "cuda" and not panel.cuda_available:
            return PreflightResult(
                provider=self.provider,
                admitted=False,
                code=TRAINING_BACKEND_NO_DEVICE,
                reason=(
                    "the declaration requires device=cuda, but the framework "
                    "reports no CUDA device"
                ),
                panel=panel,
                strategy=strategy,
            )
        if not panel.devices:
            return PreflightResult(
                provider=self.provider,
                admitted=False,
                code=TRAINING_BACKEND_NO_DEVICE,
                reason=(
                    "no accelerator is visible to the framework; recipes are "
                    "projected against measured hardware, never guesses. Declare "
                    "training_backend.config.device=cpu to override this "
                    "deliberately, or use a machine with a device"
                ),
                panel=panel,
                strategy=strategy,
            )
        return PreflightResult(
            provider=self.provider,
            admitted=True,
            code=None,
            reason=(
                f"{panel.device_count} device(s) visible, "
                f"{panel.total_vram_gb:.3f} GB total VRAM"
            ),
            panel=panel,
            strategy=strategy,
            overrides=tuple(overrides),
        )

    def estimate(self, manifest: Any, recipe: TrainingRecipe) -> BackendEstimate:
        panel = self.probe(disk_path=getattr(manifest, "state_root", None))
        return _estimate_local(
            manifest, recipe, panel, template=_load_template(manifest)
        )

    def projected_overhead(self, manifest: Any) -> OverheadReport:
        return OverheadReport(
            hours=0.0,
            basis=(
                "the executor is a local subprocess that starts in place, so there "
                "is no remote attach cost to pay"
            ),
        )

    def admit(self, manifest: Any, recipe: TrainingRecipe) -> tuple[str, str] | None:
        return check_growth_envelope(recipe, _envelope(manifest))

    def build_training_fn(
        self, manifest: Any, *, state_root: Any = None, runner: Any = None
    ) -> Any:
        from .campaign_runner import build_local_training_fn

        return build_local_training_fn(manifest, state_root=state_root, runner=runner)

    def resume(
        self, manifest: Any, recipe: TrainingRecipe, checkpoint: str | Path
    ) -> TrainingRecipe:
        return _resume_recipe(manifest, recipe, checkpoint, paths_are_local=True)


# --------------------------------------------------------------------------
# UNSLOTH
# --------------------------------------------------------------------------


class UnslothTrainingBackend(_SharedAttemptLifecycle):
    """The isolated Unsloth engine, declared first-class.

    Execution is the production CLI through the same binding as ``local``: the
    project template selects the engine and the executor runs in its own
    isolated environment. What this provider adds is the capability matrix and
    the refusals that stop an unsupported combination *before* training.
    """

    provider = PROVIDER_UNSLOTH
    trainer = "unsloth-peft"

    def __init__(
        self,
        *,
        probe: Callable[..., PreflightPanel] | None = None,
    ) -> None:
        self.probe = probe or probe_local_panel

    def capabilities(self) -> BackendCapabilities:
        supported = (
            "learning_rate",
            "lora_rank",
            "lora_alpha",
            "target_modules",
            "gradient_accumulation",
            "quantization",
            "max_steps",
            "batch_size",
            "lr_scheduler_type",
            "warmup_steps",
            "warmup_ratio",
            "max_length",
            "seed",
        )
        rows = [
            CapabilityRow(
                capability,
                CAPABILITY_SUPPORTED,
                "read by UnslothPeftRunSpec.from_resolved_config",
            )
            for capability in supported
        ]
        rows.extend(
            [
                CapabilityRow(
                    "replay_mix",
                    CAPABILITY_DATA_LAYER,
                    "a data-layer responsibility, declared through backend.replay",
                ),
                CapabilityRow(
                    "full_finetune",
                    CAPABILITY_MODEL_HARDWARE_DEPENDENT,
                    "refused where the Unsloth fast-path LoRA/QLoRA contract does "
                    "not cover the declared model",
                ),
                CapabilityRow(
                    "checkpoint_resume",
                    CAPABILITY_SUPPORTED,
                    "resume_from_checkpoint is resolved into UnslothPeftRunSpec and "
                    "the executor refuses a missing checkpoint, a changed dataset or "
                    "a changed environment against its own "
                    "chowder-unsloth-checkpoint-manifest.json (acceptance covered by "
                    "tests/test_unsloth_peft.py's mocked-worker resume suite; "
                    "real-CUDA acceptance is still outstanding)",
                ),
                CapabilityRow(
                    "custom_objective",
                    CAPABILITY_CAPABILITY_DEPENDENT,
                    "an objective the isolated worker does not execute is refused",
                ),
            ]
        )
        rows.extend(
            CapabilityRow(
                knob,
                CAPABILITY_REFUSED,
                "unverified against Unsloth's patched model/attention "
                "implementation; a silent no-op would misrepresent what ran",
            )
            for knob in UNSLOTH_REFUSED_KNOBS
        )
        return BackendCapabilities(
            provider=self.provider, trainer=self.trainer, rows=tuple(rows)
        )

    def preflight(self, manifest: Any) -> PreflightResult:
        template = _load_template(manifest)
        if template is None:
            return PreflightResult(
                provider=self.provider,
                admitted=False,
                code=TRAINING_BACKEND_TEMPLATE_UNDECLARED,
                reason=(
                    "the campaign declares no project_template_path, so the "
                    "trainer engine cannot be confirmed; a backend is declared, "
                    "never inferred"
                ),
                panel=self.probe(disk_path=getattr(manifest, "state_root", None)),
            )
        strategy = _training_strategy(template)
        panel = self.probe(disk_path=getattr(manifest, "state_root", None))
        engine = str(_template_backend(template).get("engine", "") or "").strip().lower()
        if engine != "unsloth":
            return PreflightResult(
                provider=self.provider,
                admitted=False,
                code=TRAINING_BACKEND_TRAINER_MISMATCH,
                reason=(
                    f"the campaign declares provider 'unsloth' but its project "
                    f"template declares engine {engine or '<undeclared>'!r}; the "
                    "declaration and the template must describe one trainer "
                    "(backend.type: peft, engine: unsloth)"
                ),
                panel=panel,
                strategy=strategy,
            )
        requested = _requested_knobs(template)
        if requested:
            return PreflightResult(
                provider=self.provider,
                admitted=False,
                code=TRAINING_BACKEND_UNSUPPORTED_CAPABILITY,
                reason=(
                    f"the declared template turns on {list(requested)}, which the "
                    "engine='unsloth' path refuses outright (unverified against "
                    "Unsloth's patched model/attention implementation); set them "
                    "to 'off' or declare provider 'local'"
                ),
                panel=panel,
                strategy=strategy,
            )
        override = _device_override(backend_declaration(manifest))
        if override == "cpu":
            return PreflightResult(
                provider=self.provider,
                admitted=False,
                code=TRAINING_BACKEND_NO_DEVICE,
                reason=(
                    "the Unsloth engine requires an accelerator; "
                    "training_backend.config.device=cpu cannot be honored here"
                ),
                panel=panel,
                strategy=strategy,
            )
        if not panel.devices:
            return PreflightResult(
                provider=self.provider,
                admitted=False,
                code=TRAINING_BACKEND_NO_DEVICE,
                reason=(
                    "no accelerator is visible to the framework; the isolated "
                    "Unsloth engine cannot run without one"
                ),
                panel=panel,
                strategy=strategy,
            )
        return PreflightResult(
            provider=self.provider,
            admitted=True,
            code=None,
            reason=(
                "the declared template selects the isolated Unsloth engine and no "
                "refused knob is requested"
            ),
            panel=panel,
            strategy=strategy,
            notes=(
                "the isolated environment itself is verified when the executor "
                "resolves it, not here",
            ),
        )

    def estimate(self, manifest: Any, recipe: TrainingRecipe) -> BackendEstimate:
        return BackendEstimate(
            provider=self.provider,
            recipe_id=recipe.recipe_id,
            strategy=_training_strategy(_load_template(manifest)),
            available=False,
            reason=(
                "this build measures no Unsloth panel; the isolated executor's own "
                "resolution decides at run time rather than reporting a guessed "
                "memory estimate"
            ),
            assumptions=(
                "the executor refuses an unsupported Unsloth combination before "
                "training; that refusal is the control, not this estimate",
            ),
        )

    def projected_overhead(self, manifest: Any) -> OverheadReport:
        return OverheadReport(
            hours=0.0,
            basis=(
                "the executor is a local subprocess in an isolated environment, so "
                "there is no remote attach cost to pay"
            ),
        )

    def admit(self, manifest: Any, recipe: TrainingRecipe) -> tuple[str, str] | None:
        return check_growth_envelope(recipe, _envelope(manifest))

    def build_training_fn(
        self, manifest: Any, *, state_root: Any = None, runner: Any = None
    ) -> Any:
        # One execution path: the production CLI composes the declared template,
        # whose engine resolves to the isolated Unsloth executor.
        from .campaign_runner import build_local_training_fn

        return build_local_training_fn(manifest, state_root=state_root, runner=runner)

    def resume(
        self, manifest: Any, recipe: TrainingRecipe, checkpoint: str | Path
    ) -> TrainingRecipe:
        return _resume_recipe(manifest, recipe, checkpoint, paths_are_local=True)


# --------------------------------------------------------------------------
# KAGGLE
# --------------------------------------------------------------------------


def _normalize_kaggle_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """The declared Kaggle wiring, validated, or a named refusal."""
    missing = [
        key
        for key in KAGGLE_REQUIRED_CONFIG_KEYS
        if str(config.get(key, "")).strip() == ""
    ]
    if missing:
        raise TrainingBackendRefusal(
            KAGGLE_CONFIG_INCOMPLETE,
            f"provider 'kaggle' requires {sorted(KAGGLE_REQUIRED_CONFIG_KEYS)}; "
            f"the declaration omits {missing}. A remote attempt ships exactly "
            "what the campaign declares",
        )
    normalized: dict[str, Any] = {}
    if "prepared_path" in config:
        normalized["prepared_path"] = _text(config["prepared_path"], "prepared_path")
    normalized["repository"] = _text(config["repository"], "repository")
    commit = _text(config["commit_sha"], "commit_sha").strip().lower()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise TrainingBackendRefusal(
            TRAINING_BACKEND_SCHEMA,
            f"commit_sha must be a full 40-character commit sha, got {commit!r}",
        )
    normalized["commit_sha"] = commit
    normalized["entry_point"] = _text(config["entry_point"], "entry_point")
    normalized["attempts_root"] = _text(config["attempts_root"], "attempts_root")
    normalized["accelerator"] = _text(config["accelerator"], "accelerator")
    normalized["timeout_seconds"] = _finite(
        config["timeout_seconds"], "timeout_seconds", minimum=1e-9
    )
    mounts = config["mounts"]
    if isinstance(mounts, str) or not isinstance(mounts, (list, tuple)) or not mounts:
        raise TrainingBackendRefusal(
            TRAINING_BACKEND_SCHEMA,
            f"mounts must be a non-empty list of dataset mounts, got {mounts!r}",
        )
    normalized["mounts"] = tuple(_text(mount, "mount") for mount in mounts)
    if "owner" in config:
        normalized["owner"] = _text(config["owner"], "owner")
    if "input_paths" in config:
        paths = config["input_paths"]
        if not isinstance(paths, Mapping):
            raise TrainingBackendRefusal(
                TRAINING_BACKEND_SCHEMA,
                f"input_paths must be an object of name -> path(s), got {paths!r}",
            )
        normalized["input_paths"] = dict(paths)
    if "model_commit" in config:
        normalized["model_commit"] = _text(config["model_commit"], "model_commit")
    if "pip_extras" in config:
        extras = config["pip_extras"]
        if isinstance(extras, str) or not isinstance(extras, (list, tuple)):
            raise TrainingBackendRefusal(
                TRAINING_BACKEND_SCHEMA,
                f"pip_extras must be a list of extras, got {extras!r}",
            )
        normalized["pip_extras"] = tuple(_text(extra, "pip_extras entry") for extra in extras)
    if "projection_tolerance" in config:
        normalized["projection_tolerance"] = _finite(
            config["projection_tolerance"], "projection_tolerance", minimum=0.0
        )
    if "declared_quota_ceiling_gpu_hours" in config:
        normalized["declared_quota_ceiling_gpu_hours"] = _finite(
            config["declared_quota_ceiling_gpu_hours"],
            "declared_quota_ceiling_gpu_hours",
            minimum=0.0,
        )
    normalized["payload"] = _validate_kaggle_payload(config["payload"])
    return normalized


def _declared_prepared_inputs(manifest: Any) -> tuple[dict[str, str], list[str]]:
    """The prepared inputs this manifest already declares, and the ones it does not."""
    from .kaggle_campaign import DECLARED_PREPARED_FIELDS

    present: dict[str, str] = {}
    missing: list[str] = []
    for name in DECLARED_PREPARED_FIELDS:
        value = str(getattr(manifest, name, "") or "").strip()
        if value:
            present[name] = value
        else:
            missing.append(name)
    return present, missing


def _prepared_campaign(manifest: Any, config: Mapping[str, Any]) -> Any:
    """One attempt's prepared campaign: the declared document, or the manifest's own.

    A campaign that declares ``config.prepared_path`` points at a prepared
    document. Absent one, the prepared inputs are assembled from the manifest's
    *own* declared input fields -- the same fields
    :func:`chowder.growth.campaign_prepare.prepare_campaign` writes, and the same
    ones the run phase already requires -- so dispatching to Kaggle is one
    declaration rather than two documents that can disagree.
    """
    declared_path = str(config.get("prepared_path", "") or "").strip()
    if declared_path:
        return _load_prepared_campaign(declared_path)
    from .campaign_prepare import PreparedCampaign

    inputs, missing = _declared_prepared_inputs(manifest)
    if missing:
        raise TrainingBackendRefusal(
            KAGGLE_CONFIG_INCOMPLETE,
            f"the campaign declares no prepared document and no {missing}; a "
            "remote attempt ships exactly the inputs the campaign declares, so "
            "declare them on the manifest (chowder growth campaign prepare "
            "writes them) or declare config.prepared_path",
        )
    return PreparedCampaign(
        cycle_id=str(getattr(manifest, "cycle_id", "")),
        directory=Path(str(getattr(manifest, "state_root", "."))),
        inputs=inputs,
        recipe_ids=tuple(str(entry) for entry in getattr(manifest, "recipe_ids", ())),
        notes=(
            "assembled from the manifest's declared inputs; the campaign declared "
            "no prepared document",
        ),
    )


def _validate_kaggle_payload(document: Any) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        raise TrainingBackendRefusal(
            TRAINING_BACKEND_SCHEMA,
            f"payload must be an object, got {type(document).__name__}",
        )
    kind = str(document.get("kind", "") or "").strip()
    if kind == "corpus-training":
        budget = document.get("project_gpu_hour_budget")
        _finite(budget, "payload.project_gpu_hour_budget", minimum=0.0)
        python = document.get("python", "python")
        return {
            "kind": kind,
            "project_gpu_hour_budget": float(budget),
            "python": _text(python, "payload.python"),
        }
    if kind == "command":
        command = document.get("command")
        if (
            isinstance(command, str)
            or not isinstance(command, (list, tuple))
            or not command
        ):
            raise TrainingBackendRefusal(
                TRAINING_BACKEND_SCHEMA,
                "payload.kind == 'command' requires a non-empty command list",
            )
        return {"kind": kind, "command": [_text(part, "command part") for part in command]}
    raise TrainingBackendRefusal(
        TRAINING_BACKEND_UNSUPPORTED_CONFIG,
        f"payload.kind {kind!r} is not one this wiring implements (known: "
        "'corpus-training', 'command')",
    )


def _load_prepared_campaign(path: str | Path) -> Any:
    from .campaign_prepare import PreparedCampaign

    location = Path(path)
    try:
        document = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise TrainingBackendRefusal(
            KAGGLE_CONFIG_INCOMPLETE,
            f"prepared campaign {location} could not be read: {error}",
        ) from error
    if not isinstance(document, Mapping):
        raise TrainingBackendRefusal(
            KAGGLE_CONFIG_INCOMPLETE,
            f"prepared campaign {location} is not a JSON object",
        )
    try:
        return PreparedCampaign(
            cycle_id=_text(document["cycle_id"], "prepared.cycle_id"),
            directory=Path(str(document["directory"])),
            inputs=dict(document["inputs"]),
            recipe_ids=tuple(str(entry) for entry in document["recipe_ids"]),
            notes=tuple(str(entry) for entry in document.get("notes", ())),
        )
    except (KeyError, TypeError) as error:
        raise TrainingBackendRefusal(
            KAGGLE_CONFIG_INCOMPLETE,
            f"prepared campaign {location} is missing required fields: {error}",
        ) from error


class KaggleTrainingBackend(_SharedAttemptLifecycle):
    """The remote kernel path, re-exposed through the unified surface.

    The transform/PAYLOAD/kernel code is the already-committed
    ``kaggle_campaign`` / ``kaggle_kernel`` / ``kaggle_payload`` wiring; this
    provider declares the wiring, proves it before compute, and maps the
    remote attempt onto the uniform outcome. A blocked credential or quota
    refuses here rather than pretending a dispatch happened.
    """

    provider = PROVIDER_KAGGLE
    trainer = "kaggle-kernel"

    def __init__(
        self,
        *,
        backend: Any = None,
        transport: Any = None,
        cli_probe: Callable[[str], str | None] | None = None,
    ) -> None:
        #: An injected ComputeBackend (or transport) is how a test -- or an
        #: operator with a pre-built backend -- avoids the CLI check.
        self.backend = backend
        self.transport = transport
        self.cli_probe = cli_probe or shutil.which

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            provider=self.provider,
            trainer=self.trainer,
            rows=(
                CapabilityRow(
                    "kernel_dispatch",
                    CAPABILITY_SUPPORTED,
                    "KaggleComputeBackend pushes, polls and pulls one kernel",
                ),
                CapabilityRow(
                    "declared_inputs",
                    CAPABILITY_SUPPORTED,
                    "every input is hashed before push and re-verified in the kernel",
                ),
                CapabilityRow(
                    "artifact_manifest",
                    CAPABILITY_SUPPORTED,
                    "the returned artifact is verified against the kernel's manifest",
                ),
                CapabilityRow(
                    "cost_settlement",
                    CAPABILITY_SUPPORTED,
                    "remote spend settles through the same settle_cost as local",
                ),
                CapabilityRow(
                    "checkpoint_resume",
                    CAPABILITY_SUPPORTED,
                    "resume_state travels in the request and the kernel's own "
                    "resume marker reports it",
                ),
                CapabilityRow(
                    "quota_exemption",
                    CAPABILITY_REFUSED,
                    "there is no quota exemption: a blocked quota refuses rather "
                    "than dispatching",
                ),
                CapabilityRow(
                    "interactive_session",
                    CAPABILITY_REFUSED,
                    "the backend dispatches batches; it does not hold a session",
                ),
            ),
        )

    def preflight(self, manifest: Any) -> PreflightResult:
        declaration = backend_declaration(manifest)
        panel = PreflightPanel(
            provider=self.provider,
            measurement_method=(
                "not applicable: the remote device is resolved on the Kaggle "
                "machine, and this loop never pretends to have probed it"
            ),
        )
        try:
            config = _normalize_kaggle_config(declaration.config)
        except TrainingBackendRefusal as refusal:
            return PreflightResult(
                provider=self.provider,
                admitted=False,
                code=refusal.code,
                reason=refusal.reason,
                panel=panel,
                strategy="kernel-dispatch",
            )
        if "prepared_path" not in config:
            inputs, missing = _declared_prepared_inputs(manifest)
            if missing:
                return PreflightResult(
                    provider=self.provider,
                    admitted=False,
                    code=KAGGLE_CONFIG_INCOMPLETE,
                    reason=(
                        f"the campaign declares no prepared document and no "
                        f"{missing}; a remote attempt ships exactly the inputs the "
                        "campaign declares, so declare them on the manifest "
                        "(chowder growth campaign prepare writes them) or declare "
                        "config.prepared_path"
                    ),
                    panel=panel,
                    strategy="kernel-dispatch",
                )
            covered = {str(name) for name in (config.get("input_paths") or {})}
            uncovered = sorted(name for name in inputs if name not in covered)
            if uncovered:
                return PreflightResult(
                    provider=self.provider,
                    admitted=False,
                    code=TRAINING_BACKEND_UNSUPPORTED_CONFIG,
                    reason=(
                        f"config.input_paths does not cover {uncovered}; every "
                        "declared input needs the kernel path it will be read from"
                    ),
                    panel=panel,
                    strategy="kernel-dispatch",
                )
        if self.backend is not None or self.transport is not None:
            return PreflightResult(
                provider=self.provider,
                admitted=True,
                code=None,
                reason=(
                    "a ComputeBackend/transport is injected, so the remote wiring "
                    "is available without a CLI; quota is verified at dispatch"
                ),
                panel=panel,
                strategy="kernel-dispatch",
            )
        owner = config.get("owner")
        if owner and self.cli_probe("kaggle"):
            return PreflightResult(
                provider=self.provider,
                admitted=True,
                code=None,
                reason=(
                    f"the kaggle CLI is present for owner {owner!r}; credentials "
                    "and quota are verified when the transport is called"
                ),
                panel=panel,
                strategy="kernel-dispatch",
                notes=(
                    "credentials/quota are checked at dispatch, not at preflight",
                ),
            )
        return PreflightResult(
            provider=self.provider,
            admitted=False,
            code=KAGGLE_BACKEND_UNAVAILABLE,
            reason=(
                "IMPLEMENTED / LOCAL_VERIFICATION: PASS / "
                "REMOTE_PRODUCTION_VERIFICATION: "
                "BLOCKED_BY_OPERATOR_CREDENTIAL_OR_QUOTA -- no injected backend "
                "and no usable kaggle CLI (declare config.owner, or inject a "
                "ComputeBackend/transport)"
            ),
            panel=panel,
            strategy="kernel-dispatch",
        )

    def estimate(self, manifest: Any, recipe: TrainingRecipe) -> BackendEstimate:
        return BackendEstimate(
            provider=self.provider,
            recipe_id=recipe.recipe_id,
            strategy="kernel-dispatch",
            available=False,
            reason=(
                "the remote device is resolved on the Kaggle machine; the "
                "declared recipe projection is what admission settles, and "
                "inventing a remote memory panel here would be a guess"
            ),
        )

    def projected_overhead(self, manifest: Any) -> OverheadReport:
        return OverheadReport(
            hours=None,
            basis=(
                "the kernel is pushed, installed, polled and pulled, and this build "
                "measures none of that; ranking it as zero would make the provider "
                "nobody measured look like the cheapest"
            ),
        )

    def admit(self, manifest: Any, recipe: TrainingRecipe) -> tuple[str, str] | None:
        return check_growth_envelope(recipe, _envelope(manifest))

    def build_training_fn(
        self, manifest: Any, *, state_root: Any = None, runner: Any = None
    ) -> Any:
        from .kaggle_campaign import KaggleTrainingFn
        from .kaggle_compute import KaggleComputeBackend
        from .kaggle_payload import CorpusTraining

        config = _normalize_kaggle_config(backend_declaration(manifest).config)
        backend = self.backend
        if backend is None:
            transport = self.transport
            if transport is None:
                from .kaggle_cli_transport import KaggleCliTransport

                transport = KaggleCliTransport(owner=str(config["owner"]))
            backend = KaggleComputeBackend(
                transport,
                accelerator=str(config["accelerator"]),
                declared_quota_ceiling_gpu_hours=config.get(
                    "declared_quota_ceiling_gpu_hours"
                ),
                projection_tolerance=float(config.get("projection_tolerance", 0.25)),
            )
        payload_config = config["payload"]
        declared_payload = None
        command: Sequence[str] | None = None
        if payload_config["kind"] == "corpus-training":
            declared_payload = CorpusTraining(
                project_gpu_hour_budget=float(payload_config["project_gpu_hour_budget"]),
                python=str(payload_config["python"]),
            )
        else:
            command = tuple(payload_config["command"])
        return KaggleTrainingFn(
            backend,
            prepared=_prepared_campaign(manifest, config),
            envelope=_envelope(manifest),
            repository=str(config["repository"]),
            commit_sha=str(config["commit_sha"]),
            chowder_version=chowder_version(),
            entry_point=str(config["entry_point"]),
            command=command,
            input_paths=config.get("input_paths", {}),
            mounts=tuple(config["mounts"]),
            attempts_root=(
                Path(state_root) if state_root is not None else Path(config["attempts_root"])
            ),
            timeout_seconds=float(config["timeout_seconds"]),
            declared_payload=declared_payload,
            model_commit=config.get("model_commit"),
            pip_extras=tuple(config.get("pip_extras", ("train",))),
            projection_tolerance=float(config.get("projection_tolerance", 0.25)),
        )

    def resume(
        self, manifest: Any, recipe: TrainingRecipe, checkpoint: str | Path
    ) -> TrainingRecipe:
        # The checkpoint lives in the kernel's own state, so its identity is
        # verified remotely; only its declaration is checked here.
        return _resume_recipe(manifest, recipe, checkpoint, paths_are_local=False)


def chowder_version() -> str:
    """The chowder version an attempt's evidence names as its backend version."""
    import importlib.metadata

    try:
        return importlib.metadata.version("chowder-ai")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - source tree
        import chowder

        return str(getattr(chowder, "__version__", "0.0.0"))


# --------------------------------------------------------------------------
# registry and AUTO
# --------------------------------------------------------------------------


def backend_for_provider(
    provider: str,
    *,
    probes: Mapping[str, Callable[..., PreflightPanel]] | None = None,
    kaggle_backend: Any = None,
    kaggle_transport: Any = None,
    kaggle_cli_probe: Callable[[str], str | None] | None = None,
) -> TrainingBackend:
    """The provider instance for a declared provider name, or a refusal.

    ``probes`` injects a panel probe per provider; production always uses the
    real device probe, and the injection exists so admission is provable on a
    machine with no accelerator.
    """
    panels = dict(probes or {})
    if provider == PROVIDER_LOCAL:
        return LocalTrainingBackend(probe=panels.get(PROVIDER_LOCAL))
    if provider == PROVIDER_UNSLOTH:
        return UnslothTrainingBackend(probe=panels.get(PROVIDER_UNSLOTH))
    if provider == PROVIDER_KAGGLE:
        return KaggleTrainingBackend(
            backend=kaggle_backend, transport=kaggle_transport, cli_probe=kaggle_cli_probe
        )
    raise TrainingBackendRefusal(
        TRAINING_BACKEND_UNKNOWN_PROVIDER,
        f"provider {provider!r} is not one of {sorted(PROVIDERS)}; a backend is "
        "declared, never inferred",
    )


#: What an auto comparison does and does not claim. No measured per-backend
#: throughput exists in this build, so the comparison is over each provider's
#: own declared attach overhead -- never over a fabricated cost model.
_AUTO_COST_BASIS = (
    "compared the declared attach overhead of every preflight-passable candidate; "
    "this is not a measured end-to-end cost, and a provider that reports no "
    "measured overhead is not ranked as if it reported zero"
)


@dataclass(frozen=True)
class AutoSelection:
    """The record ``auto`` writes *before* compute: what it chose and why."""

    provider: str
    reason: str
    refused: tuple[tuple[str, str, str], ...] = ()
    #: One entry per preflight-passable candidate: its declared attach overhead
    #: and the basis for it. Empty when nothing was passable.
    cost_comparison: tuple[Mapping[str, Any], ...] = ()
    cost_basis: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "reason": self.reason,
            "refused": [
                {"provider": name, "code": code, "reason": reason}
                for name, code, reason in self.refused
            ],
            "cost_comparison": [dict(entry) for entry in self.cost_comparison],
            "cost_basis": self.cost_basis,
        }


def choose_auto_backend(
    declaration: TrainingBackendDeclaration,
    manifest: Any,
    *,
    providers: Sequence[TrainingBackend] | None = None,
    probes: Mapping[str, Callable[..., PreflightPanel]] | None = None,
    kaggle_backend: Any = None,
    kaggle_transport: Any = None,
    kaggle_cli_probe: Callable[[str], str | None] | None = None,
) -> tuple[TrainingBackend | None, AutoSelection]:
    """Choose the first declared candidate whose preflight passes.

    Only preflight-passable providers are eligible; every refusal is recorded
    with its code and reason, so the choice is auditable before any compute.
    """
    candidates = declaration.candidate_order()
    passable: list[tuple[str, TrainingBackend, PreflightResult]] = []
    available: dict[str, TrainingBackend] = {}
    if providers is None:
        for candidate in candidates:
            available[candidate] = backend_for_provider(
                candidate,
                probes=probes,
                kaggle_backend=kaggle_backend,
                kaggle_transport=kaggle_transport,
                kaggle_cli_probe=kaggle_cli_probe,
            )
    else:
        for instance in providers:
            available[instance.provider] = instance

    refused: list[tuple[str, str, str]] = []
    for candidate in candidates:
        instance = available.get(candidate)
        if instance is None:
            refused.append(
                (
                    candidate,
                    TRAINING_BACKEND_UNKNOWN_PROVIDER,
                    "the declared candidate is not among the supplied providers",
                )
            )
            continue
        panel = instance.preflight(manifest)
        if panel.admitted:
            passable.append((candidate, instance, panel))
        else:
            refused.append(
                (candidate, panel.code or TRAINING_BACKEND_SCHEMA, panel.reason)
            )
    if not passable:
        return None, AutoSelection(
            provider="",
            reason=(
                "auto resolved to no provider: every declared candidate failed "
                "preflight, so the campaign refuses rather than dispatching anything"
            ),
            refused=tuple(refused),
        )

    comparison: list[Mapping[str, Any]] = []
    for candidate, instance, _panel in passable:
        overhead = instance.projected_overhead(manifest)
        comparison.append(
            {
                "provider": candidate,
                "overhead_hours": overhead.hours,
                "basis": overhead.basis,
            }
        )
    measured = [entry for entry in comparison if entry["overhead_hours"] is not None]
    if measured:
        cheapest = min(
            measured,
            key=lambda entry: (
                float(entry["overhead_hours"]),
                candidates.index(str(entry["provider"])),
            ),
        )
    else:
        # Nothing measured: the declared order decides, and the record says so
        # rather than presenting an unmeasured field as a comparison.
        cheapest = comparison[0]
    chosen_name = str(cheapest["provider"])
    chosen = next(entry for entry in passable if entry[0] == chosen_name)
    reason = (
        f"auto chose {chosen_name}: {chosen[2].reason}"
        + (
            f"; cheapest known attach overhead of {len(passable)} passable "
            f"candidate(s): {cheapest['overhead_hours']} h "
            f"({cheapest['basis']})"
            if measured
            else "; no passable candidate reports a measured attach overhead, so "
            "the declared candidate order decided"
        )
        + (f" ({len(refused)} candidate(s) refused)" if refused else "")
    )
    return chosen[1], AutoSelection(
        provider=chosen_name,
        reason=reason,
        refused=tuple(refused),
        cost_comparison=tuple(comparison),
        cost_basis=_AUTO_COST_BASIS,
    )


def resolve_training_backend(
    declaration: TrainingBackendDeclaration,
    manifest: Any,
    *,
    providers: Sequence[TrainingBackend] | None = None,
    probes: Mapping[str, Callable[..., PreflightPanel]] | None = None,
    kaggle_backend: Any = None,
    kaggle_transport: Any = None,
    kaggle_cli_probe: Callable[[str], str | None] | None = None,
) -> tuple[TrainingBackend, AutoSelection | None]:
    """The backend the declaration names, with the auto record when auto chose.

    A manual declaration is returned as declared: preflight is reported, not
    silently vetoed. ``auto``, by construction, returns only a provider whose
    preflight passed, and the :class:`AutoSelection` carries the record.
    """
    if declaration.provider == PROVIDER_AUTO:
        chosen, selection = choose_auto_backend(
            declaration,
            manifest,
            providers=providers,
            probes=probes,
            kaggle_backend=kaggle_backend,
            kaggle_transport=kaggle_transport,
            kaggle_cli_probe=kaggle_cli_probe,
        )
        if chosen is None:
            raise TrainingBackendRefusal(
                TRAINING_BACKEND_AUTO_UNRESOLVED, selection.reason
            )
        return chosen, selection
    return (
        backend_for_provider(
            declaration.provider,
            probes=probes,
            kaggle_backend=kaggle_backend,
            kaggle_transport=kaggle_transport,
            kaggle_cli_probe=kaggle_cli_probe,
        ),
        None,
    )


def preflight_report(
    manifest: Any,
    *,
    recipes: Sequence[TrainingRecipe] = (),
    recipes_unavailable: str = "",
    providers: Sequence[TrainingBackend] | None = None,
    probes: Mapping[str, Callable[..., PreflightPanel]] | None = None,
    kaggle_backend: Any = None,
    kaggle_transport: Any = None,
    kaggle_cli_probe: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """The declared backend's preflight, as one JSON-serializable document.

    Everything an operator needs to decide *before* spending anything: the
    declaration, the auto record when auto chose, the panel, the capability
    matrix, each declared recipe's estimate, and any refusal -- with
    ``stops_the_run`` saying whether the refusal is one the runner enforces
    (a declaration error) or a hardware fact it only reports.

    ``recipes`` are the planned recipes to estimate; the caller plans them (the
    plan is the campaign runner's business, not the backend's), and
    ``recipes_unavailable`` carries why they could not be planned rather than
    pretending an empty estimate list meant "nothing to estimate".
    """
    declaration = backend_declaration(manifest)
    report: dict[str, Any] = {
        "cycle_id": str(getattr(manifest, "cycle_id", "")),
        "declared": declaration.to_dict(),
        "recipes": [recipe.recipe_id for recipe in recipes],
        "recipes_unavailable": recipes_unavailable or None,
    }
    seams: dict[str, Any] = {
        "providers": providers,
        "probes": probes,
        "kaggle_backend": kaggle_backend,
        "kaggle_transport": kaggle_transport,
        "kaggle_cli_probe": kaggle_cli_probe,
    }
    try:
        provider, selection = resolve_training_backend(declaration, manifest, **seams)
        preflight = provider.preflight(manifest)
        capabilities = provider.capabilities().to_dict()
        overhead = provider.projected_overhead(manifest).to_dict()
    except TrainingBackendRefusal as refusal:
        report.update(
            {
                "status": "REFUSED",
                "provider": declaration.provider,
                "trainer": "",
                "strategy": "undeclared",
                "refused_by": refusal.code,
                "refusal_reason": refusal.reason,
                "stops_the_run": True,
                "panel": {
                    "provider": declaration.provider,
                    "measurement_method": (
                        "not reached: the declaration did not resolve to a provider"
                    ),
                },
                "capabilities": {},
                "estimates": [],
                "overrides": [],
                "notes": [],
                "auto_selection": None,
                "overhead": None,
            }
        )
        return report
    estimates = [provider.estimate(manifest, recipe).to_dict() for recipe in recipes]
    report.update(
        {
            "status": "ADMITTED" if preflight.admitted else "REFUSED",
            "provider": provider.provider,
            "trainer": provider.trainer,
            "strategy": preflight.strategy,
            "refused_by": preflight.code if not preflight.admitted else None,
            "refusal_reason": preflight.reason,
            # A declaration error stops the run; a hardware fact is reported
            # here and enforced where the declaration demands it (or by auto).
            "stops_the_run": (
                not preflight.admitted
                and preflight.code in STRUCTURAL_PREFLIGHT_CODES
            ),
            "panel": preflight.panel.to_dict(),
            "capabilities": capabilities,
            "estimates": estimates,
            "overrides": list(preflight.overrides),
            "notes": list(preflight.notes),
            "auto_selection": selection.to_dict() if selection is not None else None,
            "overhead": overhead,
        }
    )
    return report
