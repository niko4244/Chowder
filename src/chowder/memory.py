from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Tier(str, Enum):
    VRAM = "vram"
    RAM = "ram"
    NVME = "nvme"


@dataclass(frozen=True)
class HardwareProfile:
    vram_gb: float
    ram_gb: float
    nvme_gb: float
    pcie_gbps: float
    ram_gbps: float
    nvme_gbps: float
    reserve_vram_gb: float = 1.0
    reserve_ram_fraction: float = 0.15


@dataclass(frozen=True)
class WorkloadProfile:
    frozen_weights_gb: float
    trainable_gb: float
    activation_gb: float
    optimizer_gb: float
    workspace_gb: float


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
    """Create a conservative tensor-residency plan.

    Policy intentionally favors high-frequency state in VRAM:
    trainable/workspace -> activations -> optimizer -> frozen hot cache.
    Cold frozen weights are the first tensors demoted to RAM/NVMe.
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

    # Keep state with tighter latency requirements in RAM before cold weights.
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
