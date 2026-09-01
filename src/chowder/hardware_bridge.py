from __future__ import annotations

from .hardware import HardwareSnapshot
from .memory import HardwareProfile


def memory_profile_from_snapshot(
    snapshot: HardwareSnapshot,
    *,
    pcie_gbps: float,
    ram_gbps: float,
    nvme_gbps: float,
    reserve_vram_gb: float = 1.0,
    reserve_ram_fraction: float = 0.15,
) -> HardwareProfile:
    """Build a memory profile without pooling separate accelerator VRAM.

    The active contiguous pool is the largest individual accelerator. All pools
    are retained in ``accelerator_vram_gb`` for later sharding/topology-aware
    planners. This is intentionally conservative for machines such as Kaggle's
    T4x2: 2x16 GB is represented as two pools, not one 32 GB device.
    """

    pools = snapshot.topology.memory_pools_gb
    return HardwareProfile(
        vram_gb=max(pools, default=0.0),
        ram_gb=snapshot.ram_gb,
        nvme_gb=snapshot.storage_free_gb,
        pcie_gbps=pcie_gbps,
        ram_gbps=ram_gbps,
        nvme_gbps=nvme_gbps,
        reserve_vram_gb=reserve_vram_gb,
        reserve_ram_fraction=reserve_ram_fraction,
        accelerator_vram_gb=pools,
    )
