from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Tier(str, Enum):
    VRAM = "vram"
    RAM = "ram"
    NVME = "nvme"


@dataclass(frozen=True)
class HardwareProfile:
    """Memory-planning profile for one active device plus host tiers.

    ``vram_gb`` is the contiguous VRAM pool used by this plan. Additional
    accelerator pools are recorded separately in ``accelerator_vram_gb`` and are
    never summed into ``vram_gb`` implicitly.
    """

    vram_gb: float
    ram_gb: float
    nvme_gb: float
    pcie_gbps: float
    ram_gbps: float
    nvme_gbps: float
    reserve_vram_gb: float = 1.0
    reserve_ram_fraction: float = 0.15
    accelerator_vram_gb: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for label, value in (
            ("vram_gb", self.vram_gb),
            ("ram_gb", self.ram_gb),
            ("nvme_gb", self.nvme_gb),
            ("pcie_gbps", self.pcie_gbps),
            ("ram_gbps", self.ram_gbps),
            ("nvme_gbps", self.nvme_gbps),
            ("reserve_vram_gb", self.reserve_vram_gb),
        ):
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        fraction = float(self.reserve_ram_fraction)
        if not math.isfinite(fraction) or not 0 <= fraction < 1:
            raise ValueError("reserve_ram_fraction must be finite and in [0, 1)")
        for pool in self.accelerator_vram_gb:
            number = float(pool)
            if not math.isfinite(number) or number < 0:
                raise ValueError("accelerator VRAM pools must be finite and non-negative")
        if self.accelerator_vram_gb and self.vram_gb > max(self.accelerator_vram_gb) + 1e-12:
            raise ValueError(
                "vram_gb cannot exceed the largest declared accelerator VRAM pool"
            )

    @property
    def vram_pools_gb(self) -> tuple[float, ...]:
        return self.accelerator_vram_gb or ((self.vram_gb,) if self.vram_gb else ())

    @property
    def total_accelerator_vram_gb(self) -> float:
        return sum(self.vram_pools_gb)


@dataclass(frozen=True)
class WorkloadProfile:
    frozen_weights_gb: float
    trainable_gb: float
    activation_gb: float
    optimizer_gb: float
    workspace_gb: float

    def __post_init__(self) -> None:
        for label, value in (
            ("frozen_weights_gb", self.frozen_weights_gb),
            ("trainable_gb", self.trainable_gb),
            ("activation_gb", self.activation_gb),
            ("optimizer_gb", self.optimizer_gb),
            ("workspace_gb", self.workspace_gb),
        ):
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"{label} must be finite and non-negative")


@dataclass(frozen=True)
class MemoryPlan:
    vram: dict[str, float]
    ram: dict[str, float]
    nvme: dict[str, float]
    stream_frozen_weights: bool
    offload_activations: bool
    offload_optimizer: bool
    bottleneck: str

    @property
    def vram_total(self) -> float:
        return sum(self.vram.values())


def _place(amount: float, capacity: float) -> tuple[float, float]:
    placed = min(max(amount, 0.0), max(capacity, 0.0))
    return placed, max(0.0, amount - placed)


def plan_memory(hardware: HardwareProfile, workload: WorkloadProfile) -> MemoryPlan:
    """Create a conservative single-device tensor-residency plan.

    Multiple accelerator pools may exist, but this planner never treats their
    aggregate capacity as one allocation domain. Multi-device sharding belongs to
    a separate explicit placement planner.
    """
    usable_vram = max(0.0, hardware.vram_gb - hardware.reserve_vram_gb)
    mandatory = workload.trainable_gb + workload.workspace_gb
    if mandatory > usable_vram:
        raise ValueError(
            f"trainable state + workspace ({mandatory:.2f} GB) exceeds usable VRAM "
            f"({usable_vram:.2f} GB)"
        )

    vram = {"trainable": workload.trainable_gb, "workspace": workload.workspace_gb}
    remaining = usable_vram - mandatory

    vram["activations"], activation_offload = _place(workload.activation_gb, remaining)
    remaining -= vram["activations"]

    vram["optimizer"], optimizer_offload = _place(workload.optimizer_gb, remaining)
    remaining -= vram["optimizer"]

    vram["frozen_hot_cache"], frozen_offload = _place(workload.frozen_weights_gb, remaining)

    ram_capacity = hardware.ram_gb * (1.0 - hardware.reserve_ram_fraction)
    ram: dict[str, float] = {}
    nvme: dict[str, float] = {}
    remaining_ram = ram_capacity

    for name, amount in (
        ("optimizer", optimizer_offload),
        ("activations", activation_offload),
        ("frozen_weights", frozen_offload),
    ):
        in_ram, on_nvme = _place(amount, remaining_ram)
        if in_ram:
            ram[name] = in_ram
        if on_nvme:
            nvme[name] = on_nvme
        remaining_ram -= in_ram

    if sum(nvme.values()) > hardware.nvme_gb:
        raise ValueError("workload cannot fit across VRAM, RAM, and NVMe tiers")

    stream = frozen_offload > 0
    if nvme.get("activations", 0.0) or nvme.get("optimizer", 0.0):
        bottleneck = "unsupported_latency_critical_nvme_offload"
    elif nvme.get("frozen_weights", 0.0) > 0:
        bottleneck = "nvme_bandwidth"
    elif stream and hardware.pcie_gbps < hardware.ram_gbps / 2:
        bottleneck = "pcie_bandwidth"
    elif activation_offload > 0:
        bottleneck = "activation_transfer"
    elif optimizer_offload > 0:
        bottleneck = "optimizer_transfer"
    else:
        bottleneck = "compute_or_kernel"

    return MemoryPlan(
        vram=vram,
        ram=ram,
        nvme=nvme,
        stream_frozen_weights=stream,
        offload_activations=activation_offload > 0,
        offload_optimizer=optimizer_offload > 0,
        bottleneck=bottleneck,
    )
