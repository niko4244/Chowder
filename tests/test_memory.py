import pytest

from chowder.memory import HardwareProfile, WorkloadProfile, plan_memory


def test_streams_weights_when_vram_is_constrained():
    plan = plan_memory(
        HardwareProfile(16, 64, 1000, 24, 50, 5),
        WorkloadProfile(24, 1, 6, 2, 2),
    )
    assert plan.stream_frozen_weights
    assert plan.vram_total <= 15
    assert plan.ram or plan.nvme


def test_rejects_mandatory_state_that_cannot_fit_vram():
    with pytest.raises(ValueError):
        plan_memory(
            HardwareProfile(8, 64, 1000, 24, 50, 5),
            WorkloadProfile(4, 6, 2, 2, 2),
        )
