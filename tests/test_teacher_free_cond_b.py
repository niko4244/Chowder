"""Condition B (repair-specialized continuation) launcher logic tests.

Condition B continues LoRA training from the Condition A adapter on
sandbox-verified repair rows mixed with general replay, so the launcher must
bind the parent adapter by directory digest (the worker re-verifies it with
chowder.provenance.sha256_directory before load) and translate the recipe's
final-mix fraction into the executor's replay ratio. These tests pin that
arithmetic and the fail-closed guards; no GPU is required.
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


def b_recipe(fraction=0.2):
    return {
        "condition": "B_repair_specialized",
        "student": "qwen3-1.7b",
        "student_revision": "rev-pinned",
        "data": {"general_mix_fraction": fraction},
        "training": {
            "seed": 2026, "micro_batch": 1, "gradient_accumulation": 32,
            "max_length": 4096, "dtype": "bfloat16", "lora_r": 16,
            "lora_alpha": 32, "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "epochs": 2, "learning_rate": 1e-4, "scheduler": "cosine",
            "warmup_ratio": 0.03, "gradient_checkpointing": True,
        },
    }


def adapter_dir(tmp_path):
    d = tmp_path / "adapter"
    (d / "trainer").mkdir(parents=True)
    (d / "adapter_config.json").write_text("{\"r\": 16}", encoding="utf8")
    (d / "adapter_model.safetensors").write_bytes(b"\x00\x01\x02fake-weights")
    (d / "trainer" / "state.json").write_text("{}", encoding="utf8")
    return d


def mix_file(tmp_path, rows=4):
    p = tmp_path / "general.jsonl"
    p.write_text("".join(
        json.dumps({"messages": [{"role": "user", "content": f"q{i}"},
                                 {"role": "assistant", "content": f"a{i}"}]}) + "\n"
        for i in range(rows)), encoding="utf8")
    return p


def test_sha256_directory_matches_provenance(tmp_path):
    """The inline digest must equal the worker's verifier byte-for-byte,
    including subdirectories, or the worker would refuse the parent adapter
    the launcher just pinned."""
    from chowder.provenance import sha256_directory

    d = adapter_dir(tmp_path)
    assert train_pilot.sha256_directory(d) == sha256_directory(d)


def test_bindings_pin_parent_and_translate_mix_ratio(tmp_path):
    """0.2 final-mix fraction -> 0.25 raw replay ratio (ceil(primary*ratio)
    rows sampled from the mix => 20% of the final concatenation)."""
    d = adapter_dir(tmp_path)
    mix = mix_file(tmp_path)
    b = train_pilot.condition_b_bindings(
        b_recipe(0.2), parent_adapter=d, general_mix=mix)
    assert b["parent_adapter"]["path"] == str(d)
    assert b["parent_adapter"]["sha256"] == train_pilot.sha256_directory(d)
    assert b["replay"]["dataset"] == str(mix)
    assert b["replay"]["sha256"] == hashlib.sha256(mix.read_bytes()).hexdigest()
    assert b["replay"]["ratio"] == pytest.approx(0.25)


def test_bindings_zero_fraction_omits_replay(tmp_path):
    b = train_pilot.condition_b_bindings(
        b_recipe(0.0), parent_adapter=adapter_dir(tmp_path),
        general_mix=mix_file(tmp_path))
    assert "replay" not in b
    assert "parent_adapter" in b


def test_bindings_fail_closed_on_missing_inputs(tmp_path):
    with pytest.raises(SystemExit):
        train_pilot.condition_b_bindings(
            b_recipe(), parent_adapter=tmp_path / "nope",
            general_mix=mix_file(tmp_path))
    with pytest.raises(SystemExit):
        train_pilot.condition_b_bindings(
            b_recipe(), parent_adapter=adapter_dir(tmp_path),
            general_mix=tmp_path / "nope.jsonl")


def test_bindings_reject_out_of_range_fraction(tmp_path):
    with pytest.raises(SystemExit):
        train_pilot.condition_b_bindings(
            b_recipe(1.2), parent_adapter=adapter_dir(tmp_path),
            general_mix=mix_file(tmp_path))


def test_bindings_reject_non_b_recipe(tmp_path):
    with pytest.raises(ValueError):
        train_pilot.condition_b_bindings(
            {"condition": "A_supervised_distillation"},
            parent_adapter=adapter_dir(tmp_path),
            general_mix=mix_file(tmp_path))


def test_resolved_config_places_fragments_inside_backend(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text(
        json.dumps({"messages": []}) + "\n", encoding="utf8")
    parent = {"path": "X", "sha256": "a" * 64}
    replay = {"dataset": "Y", "sha256": "b" * 64, "ratio": 0.25}
    cfg = train_pilot.build_resolved_config(
        b_recipe(), data, replay=replay, parent_adapter=parent)
    assert cfg["backend"]["parent_adapter"] == parent
    assert cfg["backend"]["replay"] == replay
    # Condition A path stays clean of the continuation fragments.
    a_cfg = train_pilot.build_resolved_config(
        {"condition": "A_supervised_distillation", "student": "qwen3-1.7b",
         "student_revision": "rev-pinned",
         "training": {"seed": 1, "micro_batch": 4, "gradient_accumulation": 8,
                      "max_length": 2048, "dtype": "bfloat16", "lora_r": 16,
                      "lora_alpha": 32, "lora_dropout": 0.05,
                      "target_modules": ["q_proj"], "epochs": 1,
                      "learning_rate": 2e-4, "scheduler": "cosine",
                      "warmup_ratio": 0.0, "gradient_checkpointing": True}},
        data)
    assert "parent_adapter" not in a_cfg["backend"]
    assert "replay" not in a_cfg["backend"]
