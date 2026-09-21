"""Real tests for Unsloth parent-adapter continuation (Track C).

Mirrors transformers_peft.py's own parent_adapter contract exactly: same
config shape (backend.parent_adapter = {"path": ..., "sha256": ...}), same
both-or-neither/64-char-digest validation, same _verify_bound_adapter
tamper check before spending GPU-hours, same checkpoint-manifest binding
(a resume with a different parent adapter than what produced the checkpoint
fails closed). Uses the existing mocked-subprocess pattern from
test_unsloth_peft.py -- no real Unsloth/CUDA needed to prove the wiring.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from chowder.backends.unsloth_peft import (
    UnslothPeftExecutor,
    UnslothPeftRunSpec,
)
from chowder.executors import ExecutionContext
from chowder.memory import HardwareProfile
from chowder.models import Experiment, Hypothesis
from chowder.provenance import sha256_directory, sha256_file
from chowder.unsloth_env import unsloth_env_dir, unsloth_python


def _hardware():
    return HardwareProfile(16, 64, 500, 12, 40, 3)


def _experiment():
    return Experiment("e1", None, Hypothesis("obs", "cause", "fix"), {}, 2.0)


def _config(dataset: str, **backend_overrides):
    backend = {
        "type": "peft",
        "engine": "unsloth",
        "base_model": "org/model",
        "dataset": dataset,
        "max_length": 256,
        "lora": {"r": 8, "alpha": 16},
        "training": {"learning_rate": 1e-4, "epochs": 1.0},
    }
    backend.update(backend_overrides)
    return {"backend": backend}


def _fake_isolated_python(work_dir: Path) -> Path:
    env_dir = unsloth_env_dir(work_dir)
    python_path = unsloth_python(env_dir)
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_bytes(b"")
    return python_path


def _make_parent_adapter(root: Path, *, content: bytes = b"parent weights") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
    (root / "adapter_model.safetensors").write_bytes(content)
    return root


class _FakeProcess:
    returncode = 0

    def __init__(self, command, **kwargs):
        self.command = command
        spec_path = Path(command[command.index("--spec") + 1])
        result_path = Path(command[command.index("--result") + 1])
        self.observed_spec = json.loads(spec_path.read_text())
        output = Path(self.observed_spec["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "adapter_model.safetensors").write_bytes(b"adapter")
        result_path.write_text(
            json.dumps(
                {
                    "telemetry": {"train_loss": 0.5, "global_step": 3},
                    "resolved_target_modules": ["q_proj"],
                    "resource_usage": {
                        "active_accelerator_count": 1,
                        "visible_accelerator_count": 1,
                        "peak_vram_gb_by_accelerator": {"cuda:0": 4.2},
                    },
                    "model_provenance": {
                        "requested_base_model": self.observed_spec["base_model"],
                        "continued_from_parent_adapter": self.observed_spec.get("parent_adapter")
                        is not None,
                        "parent_adapter_sha256": self.observed_spec.get("parent_adapter_sha256"),
                    },
                    "versions": {"unsloth": "test", "torch": "test"},
                }
            )
        )

    def wait(self, timeout=None):
        return 0


# --- UnslothPeftRunSpec validation ------------------------------------------


def test_spec_requires_parent_adapter_path_and_sha_together(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    with pytest.raises(ValueError, match="path and SHA must be supplied together"):
        UnslothPeftRunSpec(
            base_model="org/model",
            dataset=str(data),
            output_dir=str(tmp_path / "out"),
            parent_adapter=str(tmp_path / "parent"),
        )


def test_spec_rejects_a_malformed_parent_adapter_sha(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    with pytest.raises(ValueError, match="must be a SHA-256 digest"):
        UnslothPeftRunSpec(
            base_model="org/model",
            dataset=str(data),
            output_dir=str(tmp_path / "out"),
            parent_adapter=str(tmp_path / "parent"),
            parent_adapter_sha256="not-a-real-digest",
        )


def test_from_resolved_config_parses_the_parent_adapter_section(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    parent_dir = _make_parent_adapter(tmp_path / "parent")
    parent_sha = sha256_directory(parent_dir)
    config = _config(
        "train.jsonl", parent_adapter={"path": str(parent_dir), "sha256": parent_sha}
    )
    spec = UnslothPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "adapter", seed=1
    )
    assert spec.parent_adapter == str(parent_dir.resolve())
    assert spec.parent_adapter_sha256 == parent_sha


# --- executor: tamper detection before spending GPU-hours -------------------


def test_executor_rejects_a_parent_adapter_that_changed_after_proposal(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    parent_dir = _make_parent_adapter(tmp_path / "parent")
    real_sha = sha256_directory(parent_dir)
    _fake_isolated_python(tmp_path)

    config = _config(
        "train.jsonl", parent_adapter={"path": str(parent_dir), "sha256": real_sha}
    )
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)

    # Tamper with the adapter after "proposal" (config resolution), before run().
    (parent_dir / "adapter_model.safetensors").write_bytes(b"tampered weights")

    def should_not_launch(*args, **kwargs):
        raise AssertionError("must not launch a worker for a tampered parent adapter")

    monkeypatch.setattr("chowder.backends.unsloth_peft.subprocess.Popen", should_not_launch)
    with pytest.raises(RuntimeError, match="parent adapter digest changed"):
        UnslothPeftExecutor().run(_experiment(), context)


# --- executor: real wiring through to the worker ----------------------------


def test_executor_hands_the_worker_the_verified_parent_adapter(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    parent_dir = _make_parent_adapter(tmp_path / "parent")
    parent_sha = sha256_directory(parent_dir)
    _fake_isolated_python(tmp_path)

    config = _config(
        "train.jsonl", parent_adapter={"path": str(parent_dir), "sha256": parent_sha}
    )
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)

    captured = {}

    class Recording(_FakeProcess):
        def __init__(self, command, **kwargs):
            super().__init__(command, **kwargs)
            captured["process"] = self

    monkeypatch.setattr("chowder.backends.unsloth_peft.subprocess.Popen", Recording)
    artifact = UnslothPeftExecutor().run(_experiment(), context)

    observed = captured["process"].observed_spec
    assert observed["parent_adapter"] == str(parent_dir.resolve())
    assert observed["parent_adapter_sha256"] == parent_sha
    assert artifact.evidence["parent_adapter_sha256"] == parent_sha
    assert artifact.evidence["continued_from_parent_adapter"] is True


def test_fresh_start_run_records_no_continuation(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    _fake_isolated_python(tmp_path)
    context = ExecutionContext(
        _hardware(), str(tmp_path), 1, resolved_config=_config("train.jsonl")
    )
    monkeypatch.setattr("chowder.backends.unsloth_peft.subprocess.Popen", _FakeProcess)
    artifact = UnslothPeftExecutor().run(_experiment(), context)
    assert artifact.evidence["parent_adapter_sha256"] is None
    assert artifact.evidence["continued_from_parent_adapter"] is False


# --- checkpoint/resume binding ----------------------------------------------


def test_resume_is_rejected_when_parent_adapter_differs_from_the_checkpoint(tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello"}\n')
    dataset_sha = sha256_file(data)
    parent_dir = _make_parent_adapter(tmp_path / "parent")
    parent_sha = sha256_directory(parent_dir)
    other_parent_dir = _make_parent_adapter(tmp_path / "other-parent", content=b"a different lineage")
    other_parent_sha = sha256_directory(other_parent_dir)
    assert other_parent_sha != parent_sha, "test setup: the two parent adapters must actually differ"
    _fake_isolated_python(tmp_path)

    checkpoint_trainer_dir = tmp_path / "prior" / "trainer"
    checkpoint_dir = checkpoint_trainer_dir / "checkpoint-50"
    checkpoint_dir.mkdir(parents=True)

    config = _config(
        "train.jsonl",
        dataset_sha256=dataset_sha,
        parent_adapter={"path": str(parent_dir), "sha256": parent_sha},
    )
    spec_for_manifest = UnslothPeftRunSpec.from_resolved_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "prior", seed=1
    )
    bound_inputs = UnslothPeftExecutor._bound_inputs(
        spec_for_manifest, environment_manifest_sha256=None
    )
    (checkpoint_trainer_dir / "chowder-unsloth-checkpoint-manifest.json").write_text(
        json.dumps(bound_inputs)
    )

    resume_config = _config(
        "train.jsonl",
        dataset_sha256=dataset_sha,
        parent_adapter={"path": str(other_parent_dir), "sha256": other_parent_sha},
        resume_from_checkpoint=str(checkpoint_dir),
    )
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=resume_config)

    def should_not_launch(*args, **kwargs):
        raise AssertionError("must not launch a worker for a rejected resume")

    monkeypatch.setattr("chowder.backends.unsloth_peft.subprocess.Popen", should_not_launch)
    with pytest.raises(ValueError, match="bound training input"):
        UnslothPeftExecutor().run(_experiment(), context)


# --- isolated worker: local hashing must match the shared implementation ---


def test_isolated_worker_directory_hash_matches_chowder_provenance(tmp_path):
    """unsloth_worker.py is deliberately self-contained (no chowder imports
    in the isolated env) and duplicates sha256_directory's logic locally --
    this proves that duplicate stays byte-for-byte identical to the real
    chowder.provenance.sha256_directory it mirrors, so a parent adapter
    verified by the controller and re-verified by the isolated worker agree
    on what "unchanged" means. unsloth_worker.py's own top-level imports are
    stdlib-only, so it's importable directly without the isolated env."""
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src" / "chowder" / "backends"))
    import unsloth_worker

    adapter = _make_parent_adapter(tmp_path / "parent")
    (adapter / "subdir").mkdir()
    (adapter / "subdir" / "extra.bin").write_bytes(b"more content")

    assert unsloth_worker._sha256_directory(adapter) == sha256_directory(adapter)

    real_sha = sha256_directory(adapter)
    assert unsloth_worker._verify_bound_adapter(str(adapter), real_sha) == real_sha
    with pytest.raises(RuntimeError, match="parent adapter digest changed"):
        unsloth_worker._verify_bound_adapter(str(adapter), "0" * 64)
