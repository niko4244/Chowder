"""Focused tests for the Condition A launcher's logic (no GPU required).

These cover the invariants that made the supervised pilot runs correct and
reviewable: the student alias must resolve to the pinned HF repo, operator
memory overrides must not silently change the effective batch, the dataset is
SHA-pinned at launch, and device exclusivity must fail closed while still
tolerating (and recording) unrelated inference servers.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments" / "teacher_free_distill"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train_pilot = load("train_pilot")


def recipe(student="qwen3-1.7b", **training):
    t = {
        "seed": 2026, "micro_batch": 4, "gradient_accumulation": 8,
        "max_length": 2048, "dtype": "bfloat16", "lora_r": 16, "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "epochs": 2.0, "learning_rate": 2e-4, "scheduler": "cosine",
        "warmup_ratio": 0.03, "gradient_checkpointing": True,
    }
    t.update(training)
    return {"student": student, "student_revision": "rev-pinned", "training": t}


def data_dir(tmp_path, rows=2):
    path = tmp_path / "data"
    path.mkdir(exist_ok=True)
    (path / "train.jsonl").write_text(
        "".join(json.dumps({"messages": []}) + "\n" for _ in range(rows)),
        encoding="utf8")
    return path


def test_recipe_alias_resolves_to_pinned_hf_repo(tmp_path):
    cfg = train_pilot.build_resolved_config(recipe(), data_dir(tmp_path))
    assert cfg["backend"]["base_model"] == "Qwen/Qwen3-1.7B"
    assert cfg["backend"]["revision"] == "rev-pinned"
    assert cfg["backend"]["type"] == "transformers-peft"


def test_unknown_student_id_passes_through(tmp_path):
    cfg = train_pilot.build_resolved_config(recipe(student="some/Other-1B"), data_dir(tmp_path))
    assert cfg["backend"]["base_model"] == "some/Other-1B"


def test_memory_override_preserves_effective_batch(tmp_path):
    """micro-batch / grad-accum may be traded against each other to fit the
    device, but their product must stay the recipe's effective batch."""
    effective = 4 * 8
    cfg = train_pilot.build_resolved_config(recipe(), data_dir(tmp_path),
                                            micro_batch=1, grad_accum=32)
    training = cfg["backend"]["training"]
    assert training["batch_size"] == 1
    assert training["gradient_accumulation_steps"] == 32
    assert training["batch_size"] * training["gradient_accumulation_steps"] == effective
    # Only memory knobs move: the rest of the frozen recipe is untouched.
    assert training["learning_rate"] == 2e-4 and training["epochs"] == 2.0
    assert cfg["backend"]["lora"]["r"] == 16


def test_dataset_hash_pins_the_actual_file(tmp_path):
    data = data_dir(tmp_path)
    cfg = train_pilot.build_resolved_config(recipe(), data)
    expected = hashlib.sha256((data / "train.jsonl").read_bytes()).hexdigest()
    assert cfg["backend"]["dataset_sha256"] == expected


def test_missing_train_file_fails_closed(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        train_pilot.build_resolved_config(recipe(), empty)


def test_exclusivity_blocks_when_chowder_holds_the_device(monkeypatch):
    monkeypatch.setattr(train_pilot, "gpu_compute_apps",
                        lambda: "1234, chowder-worker, 900 MiB")
    monkeypatch.setattr(train_pilot, "gpu_query",
                        lambda: "0, RTX 5060 Ti, 3000 MiB, 16311 MiB")
    with pytest.raises(RuntimeError, match="exclusivity"):
        train_pilot.check_device_exclusivity(0)


def test_exclusivity_enforces_free_vram_floor(monkeypatch):
    monkeypatch.setattr(train_pilot, "gpu_compute_apps", lambda: "")
    monkeypatch.setattr(train_pilot, "gpu_query",
                        lambda: "0, RTX 2060, 5000 MiB, 6144 MiB")
    with pytest.raises(RuntimeError, match="floor"):
        train_pilot.check_device_exclusivity(0)


def test_exclusivity_records_foreign_servers_as_contention_not_failure(monkeypatch):
    monkeypatch.setattr(train_pilot, "gpu_compute_apps",
                        lambda: "28400, llama-server.exe, 512 MiB")
    monkeypatch.setattr(train_pilot, "gpu_query",
                        lambda: "0, NVIDIA GeForce RTX 5060 Ti, 3880 MiB, 16311 MiB")
    info = train_pilot.check_device_exclusivity(0)
    assert info["chowder_processes_on_device"] == 0
    assert info["free_vram_gb"] == round((16311 - 3880) / 1024.0, 2)
    assert "contention risk" in info["contention_note"]
