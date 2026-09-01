import pytest

from chowder.backends.transformers_peft import TransformersPeftExecutor
from chowder.executors import ExecutionContext
from chowder.memory import HardwareProfile
from chowder.models import Experiment, Hypothesis


def _hardware():
    return HardwareProfile(
        vram_gb=16,
        ram_gb=32,
        nvme_gb=100,
        pcie_gbps=12,
        ram_gbps=40,
        nvme_gbps=3,
    )


def _experiment():
    return Experiment(
        "x",
        None,
        Hypothesis("obs", "cause", "fix"),
        {},
        0.5,
    )


def test_profile_multiplies_measured_wall_time_by_active_accelerators(tmp_path):
    executor = TransformersPeftExecutor()
    context = ExecutionContext(
        hardware=_hardware(),
        work_dir=str(tmp_path),
        seed=1,
        resolved_config={
            "backend": {
                "profile": {
                    "estimated_steps": 100,
                    "seconds_per_step": 36,
                    "active_accelerator_count": 2,
                    "source": "measured",
                }
            }
        },
    )
    estimate = executor.profile(_experiment(), context)
    assert estimate.gpu_hours == pytest.approx(2.0)
    assert "2 active accelerator" in estimate.notes[0]


def test_worker_resource_payload_charges_two_t4s_as_two_gpu_hours():
    usage = TransformersPeftExecutor._resource_usage_from_worker(
        {
            "resource_usage": {
                "active_accelerator_count": 2,
                "visible_accelerator_count": 2,
                "peak_vram_gb_by_accelerator": {
                    "cuda:0": 15.4,
                    "cuda:1": 15.3,
                },
            }
        },
        wall_seconds=3600,
        fallback_gpu=True,
    )
    assert usage.gpu_hours == pytest.approx(2.0)
    assert usage.active_accelerator_count == 2
    assert usage.visible_accelerator_count == 2
    assert usage.peak_vram_gb_by_accelerator["cuda:0"] == pytest.approx(15.4)
    assert usage.peak_vram_gb_by_accelerator["cuda:1"] == pytest.approx(15.3)
