"""CPU-only four independent executable Gen-1 arm admission/provenance tests."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from chowder.project import load_project
from chowder.project_runner import _legacy_unbounded_lifecycle

from chowder_batch import moe_gen1_four_arm
from chowder_batch.moe_gen1_four_arm import FourArmRefusal, prepare_four_arms


def _write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    mix = tmp_path / "mix"
    mix.mkdir()
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
    (tokenizer / "tokenizer_config.json").write_text(
        '{"chat_template":"{{ messages }}"}', encoding="utf-8"
    )
    base = tmp_path / "base-model"
    base.mkdir()
    (base / "config.json").write_text('{"model_type":"synthetic"}', encoding="utf-8")
    (base / "model.safetensors.index.json").write_text(
        '{"weight_map":{"model.embed_tokens.weight":"model-00001-of-00001.safetensors"}}',
        encoding="utf-8",
    )
    for asset in ("tokenizer.json", "tokenizer_config.json"):
        (base / asset).write_bytes((tokenizer / asset).read_bytes())
    shared = [
        {"messages": [{"role": "user", "content": f"shared {i}"},
                      {"role": "assistant", "content": f"answer {i}"}], "tools": ""}
        for i in range(3)
    ]
    fable = {"messages": [{"role": "user", "content": "Fable specific"},
                          {"role": "assistant", "content": "Fable result"}], "tools": ""}
    fable_rows = [shared[0], fable, *shared[1:]]
    files = {"train-fable.jsonl": fable_rows, "train-control.jsonl": shared}
    outputs = {}
    for filename, rows in files.items():
        path = mix / filename
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        treatment = filename == "train-fable.jsonl"
        outputs[filename] = {"sha256": _sha(path), "rows": len(rows),
                             "fable_declared": treatment,
                             "sources": (["math", "fable_tool_use"] if treatment else ["math"])}
    _write(mix / "manifest.json", {
        "builder": "chowder_batch/build_moe_gen1_mix.py",
        "template_validated": True, "tool_use_instrument": "present", "outputs": outputs,
        "tokenizer": {"path": str(tokenizer), "loaded": True},
    })
    receipt = tmp_path / "train-preflight.json"
    _write(receipt, {
        "ready": True, "max_length": 2048, "manifest_sha256": _sha(mix / "manifest.json"),
        "tokenizer_manifest_path": str(tokenizer),
        "tokenizer_json_sha256": _sha(tokenizer / "tokenizer.json"),
        "tokenizer_config_sha256": _sha(tokenizer / "tokenizer_config.json"),
        "tokenizer_assets_sha256": {
            name: _sha(tokenizer / name)
            for name in ("tokenizer.json", "tokenizer_config.json")
        },
        "fable_only_rows": 1,
        "arms": {filename: {"rows": len(rows), "over_length": 0, "invalid_chat": 0,
                            "answer_span_truncated": 0, "answer_span_completely_removed": 0}
                 for filename, rows in files.items()},
    })

    eval_path = tmp_path / "prepared-eval.jsonl"
    eval_path.write_text('{"prompt":"hi","expected":"yes"}\n', encoding="utf-8")
    timing_receipt = tmp_path / "measured-optimizer-steps.json"
    _write(timing_receipt, {
        "schema": "chowder.moe-gen1-step-timing.v1", "base_model": str(base),
        "placement": "cpu-transient-grad", "gpu": "synthetic test GPU", "seq": 2048,
        "batch_size": 1, "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True, "load_seconds": 500.0,
        "step_seconds": [6.0, 6.25, 6.5, 6.0],
        "base_config_sha256": _sha(base / "config.json"),
        "base_index_sha256": _sha(base / "model.safetensors.index.json"),
    })
    project_path = tmp_path / "prepared" / "project-template.json"
    _write(project_path, {
        "schema_version": 1, "name": "gen1-reap-moe", "seed": 7,
        "goal": {"metrics": [{"name": "tool-use", "direction": "maximize", "minimum": 0.5}],
                 "gpu_hour_budget": 2.5, "max_parallel_candidates": 1,
                 "minimum_promotion_gain": 0, "require_protocol_match": False},
        "baseline": {"mode": "fixed", "experiment_id": "baseline",
                     "metrics": {"tool-use": 0.42}, "gpu_hours": 0.0},
        "experiment": {"experiment_id": "gen1-reap-moe", "estimated_gpu_hours": 2.25,
                       "hypothesis": {"observation": "measured failure",
                                      "suspected_cause": "tool calls", "intervention": "LoRA"}},
        "config": {
            "goal_lifecycle": {"mode": "bounded"},
            "seed": 7, "backend": {
                "schema_version": 1, "type": "transformers-peft",
                "base_model": str(base), "dataset": "{corpus}",
                "dataset_format": "chat", "messages_field": "messages", "tools_field": "tools",
                "max_length": 2048, "precision": "bf16", "quantization": "none",
                "trust_remote_code": False, "expert_placement": "cpu-transient-grad",
                "training": {"epochs": 1.0, "max_steps": 1000, "learning_rate": 2e-4,
                             "lr_scheduler_type": "cosine", "batch_size": 1,
                             "gradient_accumulation_steps": 1,
                         "gradient_checkpointing": True},
                "runtime": {"timeout_seconds": 7200},
                "lora": {"r": 16, "alpha": 32, "dropout": 0.05,
                         "target_preset": "moe_frozen_experts"},
            },
            "evaluation": {"type": "transformers-text", "precision": "bf16",
                           "estimated_gpu_hours": 0.1,
                           "quantization": "none", "placement": "experts-cpu",
                           "suites": [{"name": "tool-use", "dataset": str(eval_path),
                                       "prompt_field": "prompt", "expected_field": "expected",
                                       "scoring": "normalized_exact_match"}]},
        },
    })
    return {"mix": mix, "project_template": project_path,
            "train_preflight": receipt, "timing_receipt": timing_receipt,
            "out_dir": tmp_path / "four-arms"}


def _prepare(paths):
    return prepare_four_arms(**paths)


def test_four_distinct_executable_production_projects_have_bound_source_and_lr(tmp_path):
    paths = _fixture(tmp_path)
    record = _prepare(paths)
    assert record["status"] == "PREPARED_ONLY_NOT_EXECUTED"
    assert record["measured_cost"]["registered_steps"] == 1000
    assert record["measured_cost"]["projected_training_gpu_hours"] == pytest.approx(7000/3600)
    assert set(record["measured_cost"]["model_metadata_sha256"]) == {
        "config.json", "model.safetensors.index.json",
    }
    assert len(record["arms"]) == 4
    assert [(arm["dataset_arm"], arm["learning_rate"]) for arm in record["arms"]] == [
        ("fable", 5e-5), ("fable", 1e-4), ("control", 5e-5), ("control", 1e-4),
    ]
    assert len({arm["project_sha256"] for arm in record["arms"]}) == 4
    assert len({arm["id"] for arm in record["arms"]}) == 4
    assert record["limitation"].startswith("Four independent")
    for arm in record["arms"]:
        path = Path(arm["project_path"])
        raw = json.loads(path.read_text())
        spec = load_project(path, validate_files=True)  # actual production project loader
        backend = raw["config"]["backend"]
        assert backend["dataset_sha256"] == arm["data_sha256"]
        assert "fable_tool_use" in arm["sources"] if arm["fable_declared"] else "fable_tool_use" not in arm["sources"]
        assert backend["dataset"] == str(paths["mix"] / f"train-{arm['dataset_arm']}.jsonl")
        assert backend["training"]["learning_rate"] == arm["learning_rate"]
        assert backend["max_length"] == 2048
        assert backend["training"]["max_steps"] == 1000
        assert backend["tools_field"] == "tools"
        assert raw["goal"]["gpu_hour_budget"] == 2.5
        assert raw["goal"]["metrics"][0]["minimum"] == 0.5
        assert raw["config"]["goal_lifecycle"] == {"mode": "bounded"}
        assert raw["experiment"]["estimated_gpu_hours"] == 2.25
        assert _legacy_unbounded_lifecycle(spec) is False
        assert spec.experiment.experiment_id == arm["id"]
        assert arm["validate_command"][-1] == str(path)
        assert arm["train_command"][-1] == str(path)
    assert len({json.loads(Path(arm["project_path"]).read_text())["registry_path"]
                for arm in record["arms"]}) == 4
    assert not any((Path(arm["project_path"]).parent / "runs.db").exists()
                   for arm in record["arms"])


def test_explicit_direction_only_lifecycle_keeps_native_semantics_but_still_caps_gpu(tmp_path):
    paths = _fixture(tmp_path)
    template = json.loads(paths["project_template"].read_text(encoding="utf-8"))
    template["config"]["goal_lifecycle"] = {"mode": "legacy_unbounded"}
    template["goal"]["metrics"][0].pop("minimum")
    _write(paths["project_template"], template)

    plan = _prepare(paths)
    assert len(plan["arms"]) == 4
    for arm in plan["arms"]:
        project_path = Path(arm["project_path"])
        result = json.loads(project_path.read_text(encoding="utf-8"))
        assert result["config"]["goal_lifecycle"] == {"mode": "legacy_unbounded"}
        assert result["goal"]["metrics"][0].get("minimum") is None
        assert result["goal"]["gpu_hour_budget"] == 2.5
        assert result["experiment"]["estimated_gpu_hours"] == 2.25
        assert _legacy_unbounded_lifecycle(load_project(project_path)) is True


@pytest.mark.parametrize("damage", [
    "not_ready", "wrong_length", "screened_tails", "changed_manifest",
    "changed_fable", "changed_control", "control_not_subset", "untrusted_tool_instrument",
    "wrong_chat", "no_tools", "wrong_preset", "wrong_max_length", "missing_budget",
    "mixed_dataset", "existing_output", "no_fable_only", "missing_source",
    "other_tokenizer", "tokenizer_changed", "over_budget", "nan_budget", "infinite_budget",
    "boolean_budget", "estimate_over_budget", "missing_lifecycle",
    "unknown_lifecycle", "bad_lifecycle", "direction_only_goal", "nan_goal_target",
    "legacy_with_numeric_target", "different_training_tokenizer", "changed_chat_template",
    "different_training_chat_template",
    "extra_training_tokenizer_file", "missing_tokenizer_assets_receipt",
    "missing_training_model",
    "wrong_steps", "missing_steps", "boolean_steps", "overlong_steps",
    "slow_timing", "wrong_timing_base", "wrong_timing_placement", "missing_timing",
    "wrong_timing_seq", "bogus_timing_steps", "nan_timing", "low_estimate",
    "missing_evaluation_reserve", "reserve_over_budget", "short_training_timeout",
    "wrong_batch_size", "changed_gradient_checkpointing",
    "missing_model_digest", "changed_model_index", "changed_model_config",
])
def test_four_arm_preparation_refuses_changed_inputs_before_writing(tmp_path, damage):
    paths = _fixture(tmp_path)
    manifest_path = paths["mix"] / "manifest.json"
    receipt_path = paths["train_preflight"]
    template_path = paths["project_template"]
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    template = json.loads(template_path.read_text())
    timing_path = paths["timing_receipt"]
    timing = json.loads(timing_path.read_text())
    if damage == "not_ready":
        receipt["ready"] = False
    elif damage == "wrong_length":
        receipt["max_length"] = 4096
    elif damage == "screened_tails":
        receipt["arms"]["train-fable.jsonl"]["over_length"] = 1
    elif damage == "changed_manifest":
        manifest["seed"] = 500
    elif damage in ("changed_fable", "changed_control"):
        filename = "train-fable.jsonl" if damage == "changed_fable" else "train-control.jsonl"
        with (paths["mix"] / filename).open("a", encoding="utf-8") as handle:
            handle.write('{"messages":[]}\n')
    elif damage == "control_not_subset":
        control = paths["mix"] / "train-control.jsonl"
        rows = [json.loads(line) for line in control.read_text().splitlines()]
        rows[0]["messages"][0]["content"] = "NOT IN TREATMENT"
        control.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest["outputs"]["train-control.jsonl"]["sha256"] = _sha(control)
        _write(manifest_path, manifest)
        receipt["manifest_sha256"] = _sha(manifest_path)
    elif damage == "untrusted_tool_instrument":
        manifest["tool_use_instrument"] = "not present"
    elif damage == "wrong_chat":
        template["config"]["backend"]["dataset_format"] = "text"
    elif damage == "no_tools":
        template["config"]["backend"].pop("tools_field")
    elif damage == "wrong_preset":
        template["config"]["backend"]["lora"]["target_preset"] = "attention_and_mlp"
    elif damage == "wrong_max_length":
        template["config"]["backend"]["max_length"] = 4096
    elif damage == "missing_budget":
        template["goal"]["gpu_hour_budget"] = 0
    elif damage == "over_budget":
        template["goal"]["gpu_hour_budget"] = 2.500001
    elif damage == "nan_budget":
        template["goal"]["gpu_hour_budget"] = float("nan")
    elif damage == "infinite_budget":
        template["goal"]["gpu_hour_budget"] = float("inf")
    elif damage == "boolean_budget":
        template["goal"]["gpu_hour_budget"] = True
    elif damage == "estimate_over_budget":
        template["experiment"]["estimated_gpu_hours"] = 2.500001
    elif damage == "missing_lifecycle":
        template["config"].pop("goal_lifecycle")
    elif damage == "legacy_with_numeric_target":
        template["config"]["goal_lifecycle"] = {"mode": "legacy_unbounded"}
    elif damage == "unknown_lifecycle":
        template["config"]["goal_lifecycle"] = {"mode": "other"}
    elif damage == "bad_lifecycle":
        template["config"]["goal_lifecycle"] = {"mode": "bounded", "legacy_unbounded": True}
    elif damage == "direction_only_goal":
        template["goal"]["metrics"][0].pop("minimum")
    elif damage == "nan_goal_target":
        template["goal"]["metrics"][0]["minimum"] = float("nan")
    elif damage == "mixed_dataset":
        template["config"]["backend"]["dataset"] = str(paths["mix"] / "train-fable.jsonl")
    elif damage == "existing_output":
        paths["out_dir"].mkdir()
    elif damage == "no_fable_only":
        receipt["fable_only_rows"] = 0
    elif damage == "missing_source":
        manifest["outputs"]["train-control.jsonl"].pop("sources")
    elif damage == "other_tokenizer":
        receipt["tokenizer_manifest_path"] = str(tmp_path / "wrong-tokenizer")
    elif damage == "tokenizer_changed":
        (tmp_path / "tokenizer" / "tokenizer.json").write_text('{"version":"changed"}', encoding="utf-8")
    elif damage == "changed_chat_template":
        (tmp_path / "tokenizer" / "tokenizer_config.json").write_text(
            '{"chat_template":"changed"}', encoding="utf-8"
        )
    elif damage == "different_training_tokenizer":
        (tmp_path / "base-model" / "tokenizer.json").write_text(
            '{"version":"different"}', encoding="utf-8"
        )
    elif damage == "different_training_chat_template":
        (tmp_path / "base-model" / "tokenizer_config.json").write_text(
            '{"chat_template":"{{ changed }}"}', encoding="utf-8"
        )
    elif damage == "extra_training_tokenizer_file":
        (tmp_path / "base-model" / "chat_template.jinja").write_text(
            "{{ changed }}", encoding="utf-8"
        )
    elif damage == "missing_tokenizer_assets_receipt":
        receipt.pop("tokenizer_assets_sha256")
    elif damage == "missing_training_model":
        template["config"]["backend"]["base_model"] = str(tmp_path / "not-present")
    elif damage == "wrong_steps":
        template["config"]["backend"]["training"]["max_steps"] = 20
    elif damage == "missing_steps":
        template["config"]["backend"]["training"].pop("max_steps")
    elif damage == "boolean_steps":
        template["config"]["backend"]["training"]["max_steps"] = True
    elif damage == "overlong_steps":
        template["config"]["backend"]["training"]["max_steps"] = 1001
    elif damage == "slow_timing":
        timing["step_seconds"] = [27.48] * 4
    elif damage == "wrong_timing_base":
        timing["base_model"] = str(tmp_path / "different-base")
    elif damage == "wrong_timing_placement":
        timing["placement"] = "cpu-transient-grad-int8"
    elif damage == "missing_timing":
        timing_path.unlink()
    elif damage == "wrong_timing_seq":
        timing["seq"] = 1024
    elif damage == "bogus_timing_steps":
        timing["step_seconds"] = [6.5]
    elif damage == "nan_timing":
        timing["step_seconds"] = [6.5, float("nan"), 6.0]
    elif damage == "low_estimate":
        template["experiment"]["estimated_gpu_hours"] = 1.8
    elif damage == "missing_evaluation_reserve":
        template["config"]["evaluation"].pop("estimated_gpu_hours")
    elif damage == "reserve_over_budget":
        template["config"]["evaluation"]["estimated_gpu_hours"] = 0.4
    elif damage == "short_training_timeout":
        template["config"]["backend"]["runtime"]["timeout_seconds"] = 6900
    elif damage == "wrong_batch_size":
        template["config"]["backend"]["training"]["batch_size"] = 2
    elif damage == "changed_gradient_checkpointing":
        template["config"]["backend"]["training"]["gradient_checkpointing"] = False
    elif damage == "missing_model_digest":
        timing.pop("base_index_sha256")
    elif damage == "changed_model_index":
        (tmp_path / "base-model" / "model.safetensors.index.json").write_text(
            '{"weight_map":{"changed":"model-00001-of-00001.safetensors"}}',
            encoding="utf-8",
        )
    elif damage == "changed_model_config":
        (tmp_path / "base-model" / "config.json").write_text(
            '{"model_type":"modified"}', encoding="utf-8",
        )
    if manifest != json.loads(manifest_path.read_text()):
        _write(manifest_path, manifest)
    if receipt != json.loads(receipt_path.read_text()):
        _write(receipt_path, receipt)
    if template != json.loads(template_path.read_text()):
        _write(template_path, template)
    if timing_path.exists() and timing != json.loads(timing_path.read_text()):
        _write(timing_path, timing)
    with pytest.raises((FourArmRefusal, ValueError)):
        _prepare(paths)
    if damage != "existing_output":
        assert not paths["out_dir"].exists()


def test_direct_script_cli_from_repository_root_uses_no_pythonpath_or_gpu(tmp_path):
    paths = _fixture(tmp_path)
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-B", "chowder_batch/moe_gen1_four_arm.py",
         "--project-template", str(paths["project_template"]),
         "--mix", str(paths["mix"]), "--train-preflight", str(paths["train_preflight"]),
         "--timing-receipt", str(paths["timing_receipt"]),
         "--out", str(paths["out_dir"])],
        cwd=repo, env=env, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "PREPARED_ONLY_NOT_EXECUTED"
    assert (paths["out_dir"] / "four-arm-plan.json").is_file()


def test_int8_placement_must_pass_authoritative_gate_before_projects_are_written(tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    template = json.loads(paths["project_template"].read_text())
    template["config"]["backend"]["expert_placement"] = "cpu-transient-grad-int8"
    _write(paths["project_template"], template)
    seen = []

    def refuse(*, expert_placement, where, base_model):
        seen.append((expert_placement, where, base_model))
        raise moe_gen1_four_arm.CampaignPrepareRefusal("required WDDM spill gate still unmeasured")

    monkeypatch.setattr(moe_gen1_four_arm, "placement_gate_refusal", refuse)
    with pytest.raises(FourArmRefusal, match="WDDM spill gate"):
        _prepare(paths)
    assert len(seen) == 1 and seen[0][0] == "cpu-transient-grad-int8"
    assert not paths["out_dir"].exists()


def test_real_int8_ab_receipt_shape_refuses_measured_1000_step_budget(tmp_path):
    """A matching 27.48s/step measurement cannot admit a 2.5 GPU-hour arm."""
    paths = _fixture(tmp_path)
    template = json.loads(paths["project_template"].read_text())
    backend = template["config"]["backend"]
    backend["expert_placement"] = "cpu-transient-grad-int8"
    base = Path(backend["base_model"])
    timing = {
        "base": str(base), "gpu": "synthetic 5060 Ti test GPU", "seq": 2048,
        "steps_per_phase": 4, "load_seconds": 1277.8,
        "load_census": {"expert_int8_buffers": 160},
        "base_config_sha256": _sha(base / "config.json"),
        "base_index_sha256": _sha(base / "model.safetensors.index.json"),
        "phases": [{"name": "loop", "grouped": False, "retain": False,
                    "step_seconds": [32.63, 32.4, 32.3, 32.0]},
                   {"name": "grouped+retain", "grouped": True, "retain": True,
                    "step_seconds": [27.57, 27.49, 27.20, 27.75]}],
    }
    _write(paths["timing_receipt"], timing)
    with pytest.raises(FourArmRefusal, match="measured 1000-step projection"):
        moe_gen1_four_arm._admit_measured_cost(
            backend, template["config"]["evaluation"], paths["timing_receipt"],
            2.5, 2.25,
        )
