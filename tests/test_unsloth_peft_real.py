"""Real-hardware acceptance test for UnslothPeftExecutor -- genuine
Unsloth + CUDA training, not a mocked subprocess.

Separate from tests/test_unsloth_peft.py (which mocks the subprocess and
runs in ordinary CI) and gated behind its own CHOWDER_REAL_UNSLOTH_SMOKE
flag, distinct from CHOWDER_REAL_ML_SMOKE: this needs a real, already-set-up
isolated Unsloth environment (`chowder setup unsloth`, see docs/UNSLOTH.md),
which ordinary CI does not have and this test does not provision itself --
matching this project's existing convention for hardware it cannot
provision in CI (e.g. tests/test_ddp_acceptance.py's real 2xGPU
requirement).

This test exists because a real bug (Unsloth's FastLanguageModel.
get_peft_model does not accept target_modules=None for auto-detection,
unlike plain PEFT's LoraConfig -- it raises `TypeError: 'NoneType' object
is not iterable`) was only caught by running a real end-to-end training
run on real hardware; the mocked-subprocess tests in test_unsloth_peft.py
cannot catch it because they never execute unsloth_worker.py for real.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from chowder.backends.unsloth_peft import UnslothPeftExecutor
from chowder.executors import ExecutionContext
from chowder.memory import HardwareProfile
from chowder.models import Experiment, Hypothesis
from chowder.unsloth_env import unsloth_env_dir, unsloth_python

_REAL_UNSLOTH_SMOKE = pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_UNSLOTH_SMOKE") != "1",
    reason="real Unsloth smoke requires CHOWDER_REAL_UNSLOTH_SMOKE=1 and a "
    "real isolated Unsloth environment (see docs/UNSLOTH.md)",
)
_TINY_MODEL = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"


@_REAL_UNSLOTH_SMOKE
def test_real_unsloth_training_with_default_target_modules_produces_a_peft_adapter(tmp_path):
    """The exact real-hardware path that broke before the fix: a recipe
    that does not specify backend.lora.target_modules at all (the common
    case) must still resolve to a real, non-empty target module list and
    produce a genuine, loadable PEFT adapter -- not crash inside Unsloth's
    own get_peft_model."""
    env_dir = unsloth_env_dir(tmp_path)
    python_executable = unsloth_python(env_dir)
    if not python_executable.is_file():
        pytest.skip(
            f"no isolated Unsloth environment at {env_dir}; run "
            "`chowder setup unsloth` in this workspace first"
        )

    data_path = tmp_path / "train.jsonl"
    rows = [
        {"text": "Question: What token comes after alpha? Answer: beta"},
        {"text": "Question: What token comes after red? Answer: blue"},
        {"text": "Question: What token comes after one? Answer: two"},
        {"text": "Question: What token comes after up? Answer: down"},
    ]
    data_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    config = {
        "backend": {
            "type": "peft",
            "engine": "unsloth",
            "base_model": _TINY_MODEL,
            "dataset": str(data_path),
            "max_length": 64,
            "quantization": "none",
            # No lora.target_modules -- the default path that crashed.
            "lora": {"r": 4, "alpha": 8},
            "training": {
                "epochs": 1.0,
                "max_steps": 4,
                "learning_rate": 1e-3,
                "batch_size": 2,
                "gradient_accumulation_steps": 1,
                "logging_steps": 1,
            },
            "runtime": {"timeout_seconds": 300.0},
        }
    }
    hardware = HardwareProfile(16, 64, 500, 12, 40, 3)
    context = ExecutionContext(hardware, str(tmp_path), 7, resolved_config=config)
    experiment = Experiment("real-e1", None, Hypothesis("obs", "cause", "fix"), {}, 0.1)

    artifact = UnslothPeftExecutor().run(experiment, context)

    assert artifact.telemetry["global_step"] == 4
    assert artifact.evidence["resolved_target_modules"]
    adapter_dir = Path(artifact.artifact_ref)
    assert (adapter_dir / "adapter_config.json").is_file()
    assert any(adapter_dir.glob("adapter_model.*"))
    assert artifact.resource_usage is not None
    assert artifact.resource_usage.active_accelerator_count == 1
