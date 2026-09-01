import pytest

from chowder.hardware import AcceleratorProfile, HardwareSnapshot, HardwareTopology
from chowder.hardware_bridge import memory_profile_from_snapshot
from chowder.memory import WorkloadProfile, plan_memory


def _kaggle_t4x2() -> HardwareSnapshot:
    return HardwareSnapshot(
        platform="Linux Kaggle",
        cpu_count=4,
        ram_gb=29.0,
        storage_total_gb=100.0,
        storage_free_gb=80.0,
        accelerators=(
            AcceleratorProfile(
                "nvidia", "Tesla T4", 16.0, "0000:00:04.0", index=0, compute_capability="7.5"
            ),
            AcceleratorProfile(
                "nvidia", "Tesla T4", 16.0, "0000:00:05.0", index=1, compute_capability="7.5"
            ),
        ),
    )


def test_kaggle_t4x2_is_two_memory_pools_not_one_32gb_pool():
    topology = _kaggle_t4x2().topology
    assert topology.memory_pools_gb == (16.0, 16.0)
    assert topology.total_accelerator_memory_gb == pytest.approx(32.0)
    assert topology.max_contiguous_accelerator_memory_gb == pytest.approx(16.0)
    assert topology.can_fit_single_device(15.5)
    assert not topology.can_fit_single_device(20.0)


def test_memory_profile_bridge_uses_largest_device_not_aggregate_capacity():
    profile = memory_profile_from_snapshot(
        _kaggle_t4x2(),
        pcie_gbps=12.0,
        ram_gbps=40.0,
        nvme_gbps=3.0,
    )
    assert profile.vram_gb == pytest.approx(16.0)
    assert profile.accelerator_vram_gb == (16.0, 16.0)
    assert profile.total_accelerator_vram_gb == pytest.approx(32.0)


def test_single_device_memory_plan_cannot_spend_second_t4_pool_implicitly():
    profile = memory_profile_from_snapshot(
        _kaggle_t4x2(),
        pcie_gbps=12.0,
        ram_gbps=40.0,
        nvme_gbps=3.0,
        reserve_vram_gb=1.0,
    )
    workload = WorkloadProfile(
        frozen_weights_gb=0,
        trainable_gb=14.0,
        activation_gb=0,
        optimizer_gb=0,
        workspace_gb=2.0,
    )
    with pytest.raises(ValueError, match="exceeds usable VRAM"):
        plan_memory(profile, workload)


def test_compute_capability_exposes_sm_marker_for_investigator_use():
    accelerator = _kaggle_t4x2().accelerators[0]
    assert accelerator.sm_marker == "sm_75"


def test_topology_rejects_link_to_unknown_gpu():
    from chowder.hardware import AcceleratorLink

    with pytest.raises(ValueError, match="unknown accelerator"):
        HardwareTopology(
            accelerators=(AcceleratorProfile("nvidia", "T4", 16.0, index=0),),
            links=(AcceleratorLink(0, 1),),
        )
