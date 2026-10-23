"""No network/model/GPU. Exercise generated Kaggle notebooks and admission truth."""

import json
from pathlib import Path

import pytest

from chowder_batch import build_kaggle_hardware_probe as builder
from chowder_batch import kaggle_hardware_probe as probe


def test_t4_emulated_bf16_cannot_claim_registered_native_bf16():
    details = {"devices": [
        {"name": "Tesla T4", "sm": [7, 5], "native_bf16": False},
        {"name": "Tesla T4", "sm": [7, 5], "native_bf16": False},
    ]}
    result = probe.classify("gpu", details, model_staged=True, calibration_staged=True)
    assert result["hardware"] == "GPU_REJECTED_NATIVE_BF16_UNAVAILABLE"
    assert not result["qualified_to_run_registered_observer"]
    assert "FULL_FROZEN_704_ROW_D0_NOT_RUN" in result["unmet_scientific_requirements"]


def test_ampere_is_hardware_candidate_not_d0():
    result = probe.classify(
        "gpu", {"devices": [{"sm": [8, 6], "native_bf16": True}]},
        model_staged=True, calibration_staged=True,
    )
    assert result["hardware"] == "GPU_NATIVE_BF16_CANDIDATE_ONLY"
    assert not result["qualified_to_run_registered_observer"]


def test_tpu_never_promotes_unvalidated_pytorch_xla_port():
    good = {"available": True, "jax_bfloat16_dtype": "bfloat16",
            "bf16_matmul_32x32": "PASS"}
    result = probe.classify("tpu", good, model_staged=True, calibration_staged=True)
    assert result["hardware"] == "TPU_NATIVE_BF16_CANDIDATE_ONLY"
    assert "QWEN3_5_MOE_LAYER_EXACT_PYTORCH_TO_TPU_PARITY_NOT_PROVEN" in (
        result["unmet_scientific_requirements"])
    assert not result["qualified_to_run_registered_observer"]
    rejected = probe.classify("tpu", {"available": False},
                              model_staged=True, calibration_staged=True)
    assert rejected["hardware"] == "TPU_REJECTED_RUNTIME_NOT_VERIFIED"


def test_no_source_identity_from_directory_presence():
    result = probe.classify("cpu", {}, model_staged=False, calibration_staged=False)
    assert len(result["unmet_scientific_requirements"]) == 3
    assert "MODEL_NOT" in result["unmet_scientific_requirements"][0]


@pytest.mark.parametrize("accelerator,expected", [
    ("cpu", {"enable_gpu": False, "enable_tpu": False, "machine_shape": "None"}),
    ("gpu", {"enable_gpu": True, "enable_tpu": False, "machine_shape": "NvidiaTeslaT4"}),
    ("tpu", {"enable_gpu": False, "enable_tpu": True, "machine_shape": "TpuV5E8"}),
])
def test_generated_private_notebook_runnable_and_accelerator_specific(tmp_path, accelerator, expected):
    dest = tmp_path / "private"
    result = builder.generate(dest=dest, kernel_id="nikmarco/chowder-hardware-probe",
                              accelerator=accelerator)
    assert result["remote_job_started"] is False
    meta = json.loads((dest / "kernel-metadata.json").read_text())
    assert meta["is_private"] is True
    assert meta["title"] == "chowder-hardware-probe"
    assert meta["enable_internet"] is False
    for key, value in expected.items():
        assert meta[key] == value
    nb = json.loads((dest / "chowder-hardware-probe.ipynb").read_text())
    assert len(nb["cells"]) == 2
    source = "".join(nb["cells"][1]["source"])
    compile(source, "kaggle-generated.ipynb.py", "exec")
    assert source.count("raise SystemExit(main(") == 1
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        builder.generate(dest=dest, kernel_id="nikmarco/chowder-hardware-probe",
                         accelerator=accelerator)


@pytest.mark.parametrize("bad", ["", "test", "a/b/c", "a/b c", "a/../b"])
def test_bad_kernel_ids_refused_without_writes(tmp_path, bad):
    dest = tmp_path / "output"
    with pytest.raises(ValueError, match="kernel id"):
        builder.generate(dest=dest, kernel_id=bad, accelerator="cpu")
    assert not dest.exists()


def test_actual_cpu_probe_requires_unoccupied_exclusive_receipt(tmp_path):
    outfile = tmp_path / "receipt.json"
    assert probe.main(["--accelerator", "cpu", "--out", str(outfile)]) == 0
    report = json.loads(outfile.read_text())
    assert report["status"] == "HARDWARE_ONLY_NOT_D0"
    assert report["model_loaded"] is False
    assert report["gate_d0"] == "NOT_MEASURED"
    assert report["expected_original_checkpoint_bytes"] == 71_903_776_776
    assert not report["admission"]["qualified_to_run_registered_observer"]
    with pytest.raises(SystemExit):
        probe.main(["--accelerator", "cpu", "--out", str(outfile)])


def test_missing_parent_not_created(tmp_path):
    outfile = tmp_path / "not-created" / "receipt.json"
    with pytest.raises(SystemExit):
        probe.main(["--out", str(outfile)])
    assert not outfile.parent.exists()
