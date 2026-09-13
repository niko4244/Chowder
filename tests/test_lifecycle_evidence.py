"""P6: the ledger a worker reports must reach the durable artifact evidence.

Measuring phases inside the worker is only half the job. The completed GSM8K
rerun's 3.463 GPU-hours were unexplainable from its own artifacts, so these
tests pin that the phase ledger, the storage reality, the sampled headroom, and
the estimator-versus-actual comparison all survive into the artifact a parent
controller writes -- and that a worker which reports nothing leaves *unknown*,
not zeros.

A worker that reports a malformed ledger is refused: an artifact whose "cost
breakdown" is unparseable is worse than one that admits it has none.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    return Experiment("x", None, Hypothesis("obs", "cause", "fix"), {}, 0.5)


def _config(dataset: str, *, profile: dict | None = None):
    config = {
        "seed": 17,
        "backend": {
            "type": "transformers-peft",
            "base_model": "example/model",
            "dataset": dataset,
            "max_length": 256,
            "quantization": "4bit",
            "training": {"learning_rate": 1e-4, "epochs": 2},
            "lora": {"r": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]},
        },
    }
    if profile is not None:
        config["backend"]["profile"] = profile
    return config


def _fake_process_factory(telemetry_extra: dict, *, omit_lifecycle: bool = False):
    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            spec_path = Path(command[command.index("--spec") + 1])
            result_path = Path(command[command.index("--result") + 1])
            spec = json.loads(spec_path.read_text())
            output = Path(spec["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            (output / "adapter_model.safetensors").write_bytes(b"adapter")
            telemetry = {"train_loss": 0.25, "global_step": 3}
            if not omit_lifecycle:
                telemetry.update(telemetry_extra)
            result_path.write_text(
                json.dumps(
                    {
                        "telemetry": telemetry,
                        "versions": {"transformers": "5.test"},
                        "provenance": {"resolved_model_commit": "abc123"},
                        "data_provenance": {"primary_rows": 1, "replay_selected_rows": 0},
                    }
                )
            )

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    return FakeProcess


#: A realistic worker payload: load measured, steps measured, the first step
#: unknown because detailed timing was off, generation owned by the evaluator.
#: The shape mirrors what the worker writes into `telemetry`.
_LEDGER = {
    "lifecycle": {
        "accelerator_count": 1,
        "phases": {
            "model_load": {
                "phase": "model_load",
                "seconds": 60.0,
                "measured": True,
                "gpu_hours": 60.0 / 3600.0,
                "accelerator_count": 1,
                "synchronized": True,
                "sync_overhead_seconds": 0.02,
                "note": None,
            },
            "steady_state_steps": {
                "phase": "steady_state_steps",
                "seconds": 3600.0,
                "measured": True,
                "gpu_hours": 1.0,
                "accelerator_count": 1,
                "synchronized": False,
                "sync_overhead_seconds": 0.0,
                "note": None,
            },
            "first_forward": {
                "phase": "first_forward",
                "seconds": None,
                "measured": False,
                "gpu_hours": None,
                "accelerator_count": 1,
                "synchronized": None,
                "sync_overhead_seconds": 0.0,
                "note": "detailed_timing_telemetry was not enabled",
            },
        },
        "measured_seconds": 3660.0,
        "measured_gpu_hours": 61.0 / 60.0,
        "unmeasured": {
            "first_forward": "detailed_timing_telemetry was not enabled",
            "baseline_generation": "generation belongs to the evaluator",
        },
    },
    "tensor_inventory": {
        "parameter_count": 2,
        "total_elements": 3008,
        "elements_by_dtype": {"uint8": 1000, "bfloat16": 2008},
        "elements_by_device": {"cuda:0": 3008},
        "packed_quantized_storage_elements": 1000,
        "trainable_elements": 8,
        "frozen_elements": 3000,
        "quantization": {"declared_available": True, "declared": {"load_in_4bit": True}},
    },
    "quantization_reality": {
        "requested": "4bit",
        "elements_by_dtype": {"uint8": 1000, "bfloat16": 2008},
        "expert_elements_by_dtype": {"bfloat16": 2008},
        "packed_quantized_storage_elements": 1000,
        "matches_request": False,
        "note": "requested '4bit' but these tensors are stored unquantized: ['bfloat16']",
    },
    "memory_sampling": {
        "samples": 120,
        "cadence_seconds": 0.5,
        "span_seconds": 60.0,
        "min_free_device_bytes": 3 * 1024**3,
        "max_used_device_bytes": 5 * 1024**3,
        "min_host_rss_bytes": None,
        "max_host_commit_bytes": None,
        "unavailable_fields": ["host_rss_bytes", "host_commit_bytes"],
        "note": "sampled at the recorded cadence",
        "device": "cuda:0",
    },
}


def _run(tmp_path, monkeypatch, *, telemetry_extra=None, omit_lifecycle=False, **config_kwargs):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    context = ExecutionContext(
        _hardware(),
        str(tmp_path),
        1,
        resolved_config=_config(str(data), **config_kwargs),
    )
    monkeypatch.setattr(
        "chowder.backends.transformers_peft.subprocess.Popen",
        _fake_process_factory(telemetry_extra or {}, omit_lifecycle=omit_lifecycle),
    )
    return TransformersPeftExecutor().run(_experiment(), context)


def test_reported_phase_ledger_reaches_the_artifact_evidence(tmp_path, monkeypatch):
    artifact = _run(
        tmp_path,
        monkeypatch,
        telemetry_extra=dict(_LEDGER),
        profile={"estimated_steps": 100, "seconds_per_step": 36.0, "source": "measured"},
    )
    lifecycle = artifact.evidence["lifecycle"]

    ledger = lifecycle["phase_ledger"]
    assert lifecycle["reservation_basis"].startswith("backend step profile")
    assert ledger["phases"]["steady_state_steps"]["seconds"] == pytest.approx(3600.0)
    assert ledger["phases"]["model_load"]["seconds"] == pytest.approx(60.0)
    # An unmeasured phase stays null in the durable record, with its reason.
    assert ledger["phases"]["first_forward"]["seconds"] is None
    assert "detailed_timing_telemetry" in ledger["unmeasured"]["first_forward"]
    assert lifecycle["state"] == "measured"


def test_estimator_versus_actual_is_recorded_per_phase(tmp_path, monkeypatch):
    artifact = _run(
        tmp_path,
        monkeypatch,
        telemetry_extra=dict(_LEDGER),
        # 100 steps x 36s = 3600s estimated, exactly what the worker measured.
        profile={"estimated_steps": 100, "seconds_per_step": 36.0, "source": "measured"},
    )
    rows = artifact.evidence["lifecycle"]["forecast_comparison"]

    assert rows["steady_state_steps"]["estimated_seconds"] == pytest.approx(3600.0)
    assert rows["steady_state_steps"]["measured_seconds"] == pytest.approx(3600.0)
    assert rows["steady_state_steps"]["state"] == "matched"
    assert rows["steady_state_steps"]["estimated_basis"] == "derived"
    # A phase measured but never estimated must not read as a zero difference.
    assert rows["model_load"]["estimated_seconds"] is None
    assert rows["model_load"]["delta_seconds"] is None
    assert rows["model_load"]["state"] == "unknown"


def test_a_diverged_estimate_is_visible_in_the_artifact(tmp_path, monkeypatch):
    artifact = _run(
        tmp_path,
        monkeypatch,
        telemetry_extra=dict(_LEDGER),
        # 100 x 7.2s = 720s estimated against 3600s measured: the exact shape of
        # the completed rerun's roughly seven-fold underestimate.
        profile={"estimated_steps": 100, "seconds_per_step": 7.2, "source": "measured"},
    )
    row = artifact.evidence["lifecycle"]["forecast_comparison"]["steady_state_steps"]
    assert row["state"] == "diverged"
    assert row["delta_seconds"] == pytest.approx(2880.0)


def test_storage_reality_and_sampled_headroom_reach_the_evidence(tmp_path, monkeypatch):
    artifact = _run(tmp_path, monkeypatch, telemetry_extra=dict(_LEDGER))

    inventory = artifact.evidence["tensor_inventory"]
    assert inventory["packed_quantized_storage_elements"] == 1000
    assert inventory["trainable_elements"] == 8

    reality = artifact.evidence["quantization_reality"]
    # The loader asked for 4-bit; the expert tensors stayed bfloat16.
    assert reality["matches_request"] is False
    assert reality["expert_elements_by_dtype"] == {"bfloat16": 2008}

    sampling = artifact.evidence["memory_sampling"]
    assert sampling["min_free_device_bytes"] == 3 * 1024**3
    assert sampling["cadence_seconds"] == pytest.approx(0.5)
    assert "host_rss_bytes" in sampling["unavailable_fields"]


def test_a_worker_without_a_ledger_reports_unknown_not_zero(tmp_path, monkeypatch):
    artifact = _run(tmp_path, monkeypatch, omit_lifecycle=True)
    lifecycle = artifact.evidence["lifecycle"]

    assert lifecycle["phase_ledger"] is None
    assert lifecycle["state"] == "unknown"
    assert "did not report" in lifecycle["reason"]
    # Absent measurements stay absent: no invented zeros anywhere.
    assert artifact.evidence["tensor_inventory"] is None
    assert artifact.evidence["quantization_reality"] is None
    assert artifact.evidence["memory_sampling"] is None


def test_a_malformed_ledger_is_refused_rather_than_stored(tmp_path, monkeypatch):
    bad = dict(_LEDGER)
    bad["lifecycle"] = {"phases": "not a mapping"}
    with pytest.raises(RuntimeError, match="lifecycle"):
        _run(tmp_path, monkeypatch, telemetry_extra=bad)


def test_a_ledger_with_non_numeric_durations_is_refused(tmp_path, monkeypatch):
    bad = dict(_LEDGER)
    bad["lifecycle"] = {
        "phases": {"model_load": {"seconds": "sixty", "measured": True}},
        "unmeasured": {},
    }
    with pytest.raises(RuntimeError, match="lifecycle"):
        _run(tmp_path, monkeypatch, telemetry_extra=bad)


def test_a_forecast_is_built_even_when_no_profile_was_configured(tmp_path, monkeypatch):
    artifact = _run(tmp_path, monkeypatch, telemetry_extra=dict(_LEDGER))
    lifecycle = artifact.evidence["lifecycle"]

    # With no configured step profile there is nothing to compare against, and
    # the comparison says so phase by phase rather than inventing an estimate.
    rows = lifecycle["forecast_comparison"]
    assert rows["steady_state_steps"]["estimated_seconds"] is None
    assert rows["steady_state_steps"]["state"] == "unknown"
    assert rows["steady_state_steps"]["measured_seconds"] == pytest.approx(3600.0)
    assert lifecycle["reservation_basis"] == "no configured step profile"
