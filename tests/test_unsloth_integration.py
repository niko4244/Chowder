"""PR F of the Unsloth integration plan: prove the full lifecycle --
Unsloth executor -> TrainingArtifact -> independent evaluator -> gate ->
registry/promotion -- holds together end to end, using the real,
unmodified project_runner.run_project()/TransformersTextEvaluator/gate/
registry pipeline exactly the way tests/test_real_ml_training.py already
proves it for the Transformers engine.

Real Unsloth/CUDA cannot run in ordinary CI (no isolated environment, no
GPU) -- training is mocked here, but ONLY the Unsloth subprocess launch
itself. The mock still performs real CPU training (plain transformers +
peft, the same tiny model/LoRA config test_real_ml_training.py's own real
tests use) so it produces a genuine, independently-loadable PEFT adapter,
not a placeholder. Evaluation, the hard gate, and registry persistence
are entirely real and unmodified -- this is the actual proof that
"training backend doesn't control evaluation" holds for a second backend,
not merely a Popen call-count assertion.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from chowder.project import write_project
from chowder.project_runner import run_project
from chowder.registry import RunRegistry
from chowder.unsloth_env import unsloth_env_dir, unsloth_python

_REAL_ML_SMOKE = pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
_TINY_MODEL = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"


def _fake_isolated_python(work_dir: Path) -> None:
    env_dir = unsloth_env_dir(work_dir)
    python_path = unsloth_python(env_dir)
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_bytes(b"")


def _make_fake_unsloth_popen(target_modules):
    """A surgical Popen replacement, monkeypatched onto
    chowder.backends.unsloth_peft.subprocess.Popen. Since `import
    subprocess` binds every caller's `subprocess` name to the exact same
    module object, patching `Popen` there patches it process-wide, not
    just for calls "through" unsloth_peft.py -- this must delegate to the
    real, unpatched Popen for every command that isn't genuinely this
    module's own worker invocation (unsloth_worker.py --spec ... --result
    ...), or it would also swallow chowder.hardware's real `nvidia-smi`
    call and the real evaluator's own real subprocess launch, both of
    which this test needs to actually run for real.

    For the one command it *does* recognize, it performs real CPU
    training (plain transformers + peft standing in for Unsloth's own
    FastLanguageModel, which requires real CUDA and the isolated
    environment) and writes a real, independently-loadable PEFT adapter
    plus an Unsloth-worker-shaped result manifest -- the two things
    UnslothPeftExecutor.run() actually depends on from the subprocess.
    """
    real_popen = subprocess.Popen

    class _FakeUnslothWorkerProcess:
        returncode = 0

        def __init__(self, spec_path: Path, result_path: Path) -> None:
            spec = json.loads(spec_path.read_text())

            import torch
            from datasets import load_dataset
            from peft import LoraConfig, get_peft_model
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                DataCollatorForLanguageModeling,
                Trainer,
                TrainingArguments,
            )

            tokenizer = AutoTokenizer.from_pretrained(spec["base_model"])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(spec["base_model"])
            model = get_peft_model(
                model,
                LoraConfig(
                    r=spec["lora_r"],
                    lora_alpha=spec["lora_alpha"],
                    lora_dropout=0.0,
                    target_modules=list(target_modules),
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )
            resolved_target_modules = sorted(
                model.peft_config[model.active_adapter].target_modules
            )

            dataset = load_dataset("json", data_files=spec["dataset"], split="train")
            dataset = dataset.select_columns([spec["text_field"]])

            def tokenize(batch):
                return tokenizer(
                    batch[spec["text_field"]],
                    truncation=True,
                    max_length=spec["max_length"],
                    padding=False,
                )

            tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
            output_dir = Path(spec["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=str(output_dir / "trainer"),
                num_train_epochs=spec["epochs"],
                max_steps=spec.get("max_steps", -1) if spec.get("max_steps", -1) > 0 else -1,
                per_device_train_batch_size=spec["batch_size"],
                gradient_accumulation_steps=spec["gradient_accumulation_steps"],
                learning_rate=spec["learning_rate"],
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                seed=spec["seed"],
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized,
                data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            )
            train_output = trainer.train()

            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            result_path.write_text(
                json.dumps(
                    {
                        "telemetry": {
                            "train_loss": float(train_output.training_loss),
                            "global_step": int(trainer.state.global_step),
                            "train_runtime_seconds": 1.0,
                            "peak_vram_gb": 0.0,
                            "training_rows": len(dataset),
                        },
                        "resolved_target_modules": resolved_target_modules,
                        "resource_usage": {
                            "active_accelerator_count": 0,
                            "visible_accelerator_count": 0,
                            "peak_vram_gb_by_accelerator": {},
                        },
                        "model_provenance": {
                            "requested_base_model": spec["base_model"],
                            "requested_revision": spec.get("revision"),
                        },
                        "versions": {"unsloth": "fake-cpu-stand-in", "torch": torch.__version__},
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

    def fake_popen(command, **kwargs):
        # Every Chowder worker (transformers_worker.py, unsloth_worker.py,
        # transformers_text_worker.py, ...) shares the same --spec/--result
        # CLI convention, so those flags alone don't identify *which*
        # worker this is -- the evaluator's own real subprocess (which
        # this test needs to actually run) uses them too. The script path
        # (command[1]: <python> <script.py> --spec ... --result ...) is
        # what actually distinguishes this module's own worker.
        if len(command) > 1 and Path(str(command[1])).name == "unsloth_worker.py":
            spec_path = Path(command[command.index("--spec") + 1])
            result_path = Path(command[command.index("--result") + 1])
            return _FakeUnslothWorkerProcess(spec_path, result_path)
        return real_popen(command, **kwargs)

    return fake_popen


@_REAL_ML_SMOKE
def test_real_unsloth_trained_adapter_flows_through_the_real_evaluator_gate_and_registry(
    tmp_path: Path, monkeypatch
):
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                {"text": "Question: What token comes after alpha? Answer: beta"},
                {"text": "Question: What token comes after red? Answer: blue"},
            ]
        ),
        encoding="utf-8",
    )
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"prompt": "Question: What token comes after alpha? Answer:", "expected": "beta"})
        + "\n",
        encoding="utf-8",
    )

    _fake_isolated_python(tmp_path)
    target_modules = ["q_proj", "v_proj"]
    monkeypatch.setattr(
        "chowder.backends.unsloth_peft.subprocess.Popen",
        _make_fake_unsloth_popen(target_modules),
    )

    project_path = tmp_path / "project.json"
    write_project(
        project_path,
        {
            "schema_version": 1,
            "name": "unsloth integration smoke",
            "work_dir": str(tmp_path),
            "registry_path": ".chowder/runs.db",
            "seed": 123,
            "goal": {
                "metrics": [
                    {
                        "name": "quality",
                        "minimum": 0.0,
                        "direction": "maximize",
                        "regression_tolerance": 1.0,
                    }
                ],
                "gpu_hour_budget": 2.0,
                "max_parallel_candidates": 1,
                "minimum_promotion_gain": 1.0,
                "require_protocol_match": False,
            },
            "baseline": {
                "experiment_id": "baseline",
                "metrics": {"quality": 0.0},
                "gpu_hours": 0.0,
            },
            "experiment": {
                "experiment_id": "real-unsloth-sft",
                "estimated_gpu_hours": 0.25,
                "hypothesis": {
                    "observation": "tiny model is unadapted",
                    "suspected_cause": "target examples are unseen",
                    "intervention": "one small LoRA SFT run via the Unsloth engine",
                    "expected_deltas": {"quality": 0.0},
                },
                "config_patch": {},
                "tags": ["integration", "real-ml", "unsloth"],
            },
            "config": {
                "seed": 123,
                "backend": {
                    "schema_version": 1,
                    "type": "peft",
                    "engine": "unsloth",
                    "base_model": _TINY_MODEL,
                    "dataset": "train.jsonl",
                    "text_field": "text",
                    "max_length": 64,
                    "quantization": "none",
                    "lora": {"r": 4, "alpha": 8, "target_modules": target_modules},
                    "training": {
                        "epochs": 1.0,
                        "learning_rate": 0.001,
                        "batch_size": 1,
                        "gradient_accumulation_steps": 1,
                        "logging_steps": 1,
                    },
                    "runtime": {"timeout_seconds": 180.0},
                },
                "evaluation": {
                    "type": "transformers-text",
                    "estimated_gpu_hours": 0.05,
                    "precision": "fp32",
                    "quantization": "none",
                    "device": "cpu",
                    "trust_remote_code": False,
                    "runtime": {"timeout_seconds": 180.0},
                    "suites": [
                        {
                            "name": "quality",
                            "dataset": "eval.jsonl",
                            "prompt_field": "prompt",
                            "expected_field": "expected",
                            "scoring": "normalized_exact_match",
                            "max_new_tokens": 2,
                            "use_chat_template": False,
                        }
                    ],
                },
            },
        },
    )

    events = []
    outcome = run_project(project_path, on_event=events.append)

    assert len(outcome.generation.candidates) == 1
    candidate = outcome.generation.candidates[0]
    assert candidate.error is None, candidate.error
    assert candidate.artifact is not None
    assert candidate.evaluation is not None
    assert candidate.result is not None

    # The Unsloth executor really ran (not silently swapped for
    # Transformers by normalize_training_config_for_executor, which only
    # touches engine='transformers' configs).
    assert candidate.artifact.evidence["backend"] == "unsloth-peft"
    assert candidate.artifact.evidence["engine"] == "unsloth"
    assert candidate.artifact.evidence["resolved_target_modules"] == sorted(target_modules)

    adapter_dir = Path(candidate.artifact.artifact_ref)
    assert adapter_dir.is_dir()
    assert (adapter_dir / "adapter_config.json").is_file()
    assert any(path.name.startswith("adapter_model") for path in adapter_dir.iterdir())

    # The independent evaluator is the real, entirely unmodified
    # transformers-text evaluator -- it never knows or cares that this
    # adapter came from the Unsloth engine rather than Transformers.
    assert candidate.evaluation.evidence["evaluator"] == "transformers-text"
    assert len(candidate.evaluation.evidence["protocol_sha256"]) == 64
    assert candidate.evaluation.source_artifact_ref == str(adapter_dir)
    assert set(candidate.result.metrics) == {"quality"}

    registry_path = tmp_path / ".chowder" / "runs.db"
    with RunRegistry(registry_path) as registry:
        artifacts = list(registry.list_training_artifacts())
        evaluations = list(registry.list_evaluation_outcomes())
        results = list(registry.list_results())
        assert len(artifacts) == 1
        assert len(evaluations) == 1
        assert len(results) == 1
        assert artifacts[0].artifact_ref == str(adapter_dir)
        assert artifacts[0].evidence["engine"] == "unsloth"
        assert evaluations[0].source_artifact_ref == str(adapter_dir)
        assert results[0].artifact_ref == str(adapter_dir)
