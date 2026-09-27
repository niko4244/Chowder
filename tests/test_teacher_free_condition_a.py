"""Condition A's completed run must stay re-checkable, and stay honest.

The run happened once, on one machine's disk: a 25.7 MB adapter, the launcher's
run record, the worker's result and a recovered loss history. These tests pin
the record/verify contract — a byte that changes, a claim that no longer
re-derives, or a record that quietly gains an "evaluated" flag must all fail.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments" / "teacher_free_distill"
sys.path.insert(0, str(REPO / "src"))

LOGGED_STEPS = [
    {"step": 10, "loss": 2.3453, "learning_rate": 0.0002},
    {"step": 20, "loss": 1.8798, "learning_rate": 0.0001993},
    {"step": 280, "loss": 1.6871, "learning_rate": 1.630896073864352e-07},
]


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


verify_tool = load("verify_condition_a")


def fake_artifacts(tmp_path: Path) -> Path:
    """A miniature stand-in for the published Condition A directory."""
    root = tmp_path / "cond_a"
    run_dir = root / ".chowder" / "runs" / "cond-a-supervised-distillation-pilot-abc123"
    (root / "adapter").mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (root / "loss_history.json").write_text(json.dumps(LOGGED_STEPS), encoding="utf-8")
    (run_dir / "run-spec.json").write_text(json.dumps({
        "base_model": "Qwen/Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "dataset": "D:/pilot/train.jsonl", "dataset_sha256": "a" * 64,
        "epochs": 2.0, "batch_size": 1, "gradient_accumulation_steps": 32,
        "learning_rate": 0.0002, "lr_scheduler_type": "cosine", "warmup_ratio": 0.03,
        "max_length": 2048, "seed": 2026, "precision": "bf16",
        "gradient_checkpointing": True, "lora_r": 16, "lora_alpha": 32,
        "lora_dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }), encoding="utf-8")
    (root / "worker-result.json").write_text(json.dumps({
        "versions": {"torch": "2.11.0+cu128", "peft": "0.20.0"},
        "resource_usage": {"peak_vram_gb_by_accelerator": {"cuda:0": 7.034}},
        "telemetry": {
            "global_step": 284, "train_loss": 1.7276, "training_rows": 4529,
            "peak_vram_gb": 7.034, "train_runtime_seconds": 17046.19,
            "step_log": {"entries": LOGGED_STEPS, "total_entries": 3, "truncated": False},
            "lifecycle": {"measured_gpu_hours": 4.7363, "measured_seconds": 17050.69},
        },
    }), encoding="utf-8")
    (root / "run_record.json").write_text(json.dumps({
        "experiment_id": "cond-a-supervised-distillation-pilot",
        "condition": "A_supervised_distillation",
        "run_dir": str(run_dir), "wall_seconds": 17152.7,
    }), encoding="utf-8")
    (root / "adapter" / "adapter_model.safetensors").write_bytes(b"lora-weights")
    (root / "adapter" / "adapter_config.json").write_bytes(b'{"r": 16}')
    (root / "adapter" / "tokenizer.json").write_bytes(b"{}")
    (root / "adapter" / "tokenizer_config.json").write_bytes(b"{}")
    (root / "adapter" / "chat_template.jinja").write_bytes(b"{{ }}")
    return root


def write_record(root: Path, out: Path) -> dict:
    record = verify_tool.build_record(root)
    out.write_text(json.dumps(record), encoding="utf-8")
    return record


def test_record_pins_every_artifact_and_reverifies(tmp_path):
    root = fake_artifacts(tmp_path)
    record = write_record(root, tmp_path / "record.json")
    pinned = {entry["path"] for entry in record["files"]}
    assert {"adapter/adapter_model.safetensors", "run_record.json", "worker-result.json",
            "loss_history.json"} <= pinned
    assert any(path.endswith("run-spec.json") for path in pinned)
    for entry in record["files"]:
        assert len(entry["sha256"]) == 64 and entry["bytes"] > 0

    result = verify_tool.verify_record(record)
    assert result["ok"] and result["state"] == "verified"
    assert result["failures"] == []
    # 4529 rows / (1 x 32) per step x 2 epochs = 284 steps, the worker's count.
    assert result["derived"]["steps_total"] == 284
    assert result["derived"]["mean_train_loss"] == 1.7276


def test_record_refuses_to_overwrite_an_existing_record(tmp_path, monkeypatch, capsys):
    root = fake_artifacts(tmp_path)
    out = tmp_path / "record.json"
    out.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["verify_condition_a.py", "record",
                                      "--artifacts", str(root), "--out", str(out)])
    assert verify_tool.main() == 2
    assert "refusing to overwrite" in capsys.readouterr().err
    assert out.read_text() == "{}\n"


def test_verify_detects_a_tampered_adapter_byte(tmp_path):
    root = fake_artifacts(tmp_path)
    record = write_record(root, tmp_path / "record.json")
    (root / "adapter" / "adapter_model.safetensors").write_bytes(b"lora-weightz")
    result = verify_tool.verify_record(record)
    assert not result["ok"] and result["state"] == "mismatch"
    assert result["mismatched_files"][0]["path"] == "adapter/adapter_model.safetensors"


def test_verify_reports_absent_artifacts_as_unverifiable(tmp_path):
    root = fake_artifacts(tmp_path)
    record = write_record(root, tmp_path / "record.json")
    record["artifacts_root"] = str(tmp_path / "another-host")
    result = verify_tool.verify_record(record)
    assert not result["ok"] and result["state"] == "unverifiable"
    assert "adapter/adapter_model.safetensors" in result["missing_files"]
    assert result["failures"] == []  # preserved, not disproven


def test_verify_detects_a_claim_that_no_longer_derives(tmp_path):
    """A tampered measurement with a refreshed digest is still a failure: the
    record's own numbers must re-derive from the artifacts."""
    root = fake_artifacts(tmp_path)
    record = write_record(root, tmp_path / "record.json")
    worker = json.loads((root / "worker-result.json").read_text())
    worker["telemetry"]["train_loss"] = 0.5
    (root / "worker-result.json").write_text(json.dumps(worker), encoding="utf-8")
    for entry in record["files"]:
        if entry["path"] == "worker-result.json":
            entry["sha256"] = verify_tool.sha256_file(root / "worker-result.json")
            entry["bytes"] = (root / "worker-result.json").stat().st_size
    result = verify_tool.verify_record(record)
    assert not result["ok"] and result["state"] == "mismatch"
    assert any("mean_train_loss disagrees" in failure for failure in result["failures"])


def test_verify_refuses_a_claim_without_evidence(tmp_path):
    """`evaluated: true` is only accepted when the record pins a readable
    comparison artifact with the operator-review decision and a recorded
    leakage section; anything less stays a refusal."""
    root = fake_artifacts(tmp_path)
    record = write_record(root, tmp_path / "record.json")
    record["evaluation"]["evaluated"] = True
    result = verify_tool.verify_record(record)
    assert not result["ok"]
    assert any("claims a run but pins no readable" in failure
               for failure in result["failures"])

    comparison = tmp_path / "comparison_paired_final.json"
    comparison.write_text(json.dumps({"decision": "requires_operator_review",
                                      "leakage": {"dev_final_separation": {"ok": True}}}),
                          encoding="utf-8")
    record["evaluation"]["results"] = {
        "artifact": comparison.name, "paired_mean_delta": 0.114}
    result = verify_tool.verify_record(record)
    assert result["ok"], result["failures"]


def test_verify_cross_checks_the_pinned_dataset_digest(tmp_path):
    root = fake_artifacts(tmp_path)
    record = write_record(root, tmp_path / "record.json")
    result = verify_tool.verify_record(record, manifest_train_sha256="b" * 64)
    assert not result["ok"]
    assert any("differs from the pinned manifest" in failure for failure in result["failures"])


def test_committed_record_is_structurally_valid_without_the_artifacts():
    """This must hold on any host, including CI where the run is not present."""
    record = json.loads((EXP / "condition_a_artifacts.json").read_text(encoding="utf-8"))
    assert record["format"] == verify_tool.CONDITION_A_FORMAT
    run = record["run"]
    assert run["steps_completed"] == run["steps_total"] > 0
    assert len(run["dataset_sha256"]) == 64
    assert run["base_model"] == "Qwen/Qwen3-1.7B"
    assert run["resolved"]["max_length"] == 2048
    assert record["loss_history"]["entries"] > 0
    assert record["evaluation"]["evaluated"] is True
    # The evaluation claim must stay evidence-backed on every host: the
    # comparison artifact lives beside the record in the same directory.
    assert record["evaluation"]["results"]["artifact"] == "comparison_paired_final.json"
    assert (EXP / record["evaluation"]["results"]["artifact"]).is_file()
    assert {entry["path"] for entry in record["files"]} >= set(verify_tool.PINNED_FILES)


@pytest.mark.skipif(not Path(
    json.loads((EXP / "condition_a_artifacts.json").read_text(encoding="utf-8"))["artifacts_root"]
).is_dir(), reason="Condition A artifacts are not on this host")
def test_committed_record_reverifies_against_the_real_artifacts():
    record = json.loads((EXP / "condition_a_artifacts.json").read_text(encoding="utf-8"))
    result = verify_tool.verify_record(record)
    assert result["ok"] and result["state"] == "verified", result["failures"]
    assert result["derived"]["steps_completed"] == 284
