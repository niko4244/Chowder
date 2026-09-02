from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from chowder.backends.transformers_peft import TransformersPeftExecutor, TransformersPeftRunSpec
from chowder.checkpoint_discovery import discover_checkpoints
from chowder.executors import ExecutionContext
from chowder.memory import HardwareProfile
from chowder.provenance import sha256_file


def _hardware():
    return HardwareProfile(16, 64, 500, 12, 40, 3)


def _config(dataset: str, *, base_model: str = "example/model"):
    return {
        "seed": 17,
        "backend": {
            "type": "transformers-peft",
            "base_model": base_model,
            "dataset": dataset,
            "max_length": 256,
            "quantization": "4bit",
            "training": {"learning_rate": 1e-4, "epochs": 2},
            "lora": {"r": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]},
        },
    }


def _context(tmp_path, config):
    return ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)


def _write_checkpoint(
    tmp_path: Path, *, run_id: str, step: int, manifest_config: dict, work_dir: Path
) -> Path:
    trainer_dir = tmp_path / ".chowder" / "runs" / run_id / "adapter" / "trainer"
    checkpoint_dir = trainer_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True)
    spec = TransformersPeftRunSpec.from_resolved_config(
        manifest_config,
        work_dir=work_dir,
        output_dir=tmp_path / "unused",
        seed=1,
        hardware=_hardware(),  # a real run always resolves with real hardware -- see
        # _spec_for, which always passes context.hardware; a manifest written
        # without it would bake in a different hardware-aware gradient_checkpointing
        # default than discovery (which correctly uses context.hardware) computes.
    )
    if spec.dataset_sha256 is None:
        # Matches _spec_for's own order of operations exactly: the real
        # dataset hash is resolved onto the spec BEFORE _bound_inputs is
        # computed, because checkpoint_recipe_sha256's digest is derived
        # from spec.to_dict() and dataset_sha256 is one of its fields --
        # patching the returned dict afterward would leave the digest
        # computed against a None hash a real run's manifest never has.
        spec = replace(spec, dataset_sha256=sha256_file(spec.dataset))
    bound_inputs = dict(TransformersPeftExecutor._bound_inputs(spec))
    (trainer_dir / "chowder-checkpoint-manifest.json").write_text(
        json.dumps(bound_inputs), encoding="utf-8"
    )
    return checkpoint_dir


def test_no_runs_directory_returns_empty(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    result = discover_checkpoints(
        work_dir=tmp_path, resolved_config=_config(str(data)), context=_context(tmp_path, {})
    )
    assert result == ()


def test_ignores_a_trainer_directory_with_no_manifest(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    trainer_dir = tmp_path / ".chowder" / "runs" / "e1-abc" / "adapter" / "trainer"
    (trainer_dir / "checkpoint-10").mkdir(parents=True)  # no manifest alongside it
    result = discover_checkpoints(
        work_dir=tmp_path, resolved_config=_config(str(data)), context=_context(tmp_path, {})
    )
    assert result == ()


def test_discovers_a_compatible_checkpoint_even_without_a_declared_dataset_hash(tmp_path):
    """The config used both to write the manifest and to run discovery
    declares no dataset_sha256 (the common case -- a TUI-generated project
    never does) -- discovery must still recognize a match by hashing the
    real file, not by comparing None to None coincidentally."""
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    _write_checkpoint(tmp_path, run_id="e1-abc", step=50, manifest_config=config, work_dir=tmp_path)

    result = discover_checkpoints(
        work_dir=tmp_path, resolved_config=config, context=_context(tmp_path, config)
    )
    assert len(result) == 1
    checkpoint = result[0]
    assert checkpoint.valid is True
    assert checkpoint.step == 50
    assert checkpoint.mismatches == {}
    assert "compatible" in checkpoint.reason


def test_discovers_an_incompatible_checkpoint_and_names_the_mismatch(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    recorded_config = _config(str(data), base_model="org/original-model")
    _write_checkpoint(
        tmp_path, run_id="e1-abc", step=50, manifest_config=recorded_config, work_dir=tmp_path
    )

    current_config = _config(str(data), base_model="org/different-model")
    result = discover_checkpoints(
        work_dir=tmp_path,
        resolved_config=current_config,
        context=_context(tmp_path, current_config),
    )
    assert len(result) == 1
    checkpoint = result[0]
    assert checkpoint.valid is False
    assert "base_model" in checkpoint.mismatches
    assert checkpoint.mismatches["base_model"] == {
        "checkpoint": "org/original-model",
        "requested": "org/different-model",
    }
    assert "base_model" in checkpoint.reason


def test_sorts_valid_checkpoints_first_then_by_step_descending(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    current_config = _config(str(data), base_model="org/current")
    mismatched_config = _config(str(data), base_model="org/stale")

    _write_checkpoint(
        tmp_path, run_id="run-a", step=100, manifest_config=current_config, work_dir=tmp_path
    )
    _write_checkpoint(
        tmp_path, run_id="run-b", step=50, manifest_config=current_config, work_dir=tmp_path
    )
    _write_checkpoint(
        tmp_path, run_id="run-c", step=200, manifest_config=mismatched_config, work_dir=tmp_path
    )

    result = discover_checkpoints(
        work_dir=tmp_path,
        resolved_config=current_config,
        context=_context(tmp_path, current_config),
    )
    assert [(c.valid, c.step) for c in result] == [
        (True, 100),
        (True, 50),
        (False, 200),
    ]


def test_reports_a_config_error_rather_than_raising(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    config = _config(str(data))
    _write_checkpoint(tmp_path, run_id="e1-abc", step=50, manifest_config=config, work_dir=tmp_path)

    broken_config = {"backend": {"type": "transformers-peft"}}  # missing base_model/dataset
    result = discover_checkpoints(
        work_dir=tmp_path,
        resolved_config=broken_config,
        context=_context(tmp_path, broken_config),
    )
    assert len(result) == 1
    assert result[0].valid is False
    assert "config" in result[0].mismatches
