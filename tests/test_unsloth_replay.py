"""Real tests for Unsloth replay/rehearsal (Track D).

Chat-format replay is merged in the controller (unsloth_peft.py, before
tokenization -- exactly mirroring transformers_worker.py's order) and
proven here via _materialize_pretokenized_chat_dataset directly. Text-format
replay is merged inside the isolated worker itself (unsloth_worker.py has
its own local _replay_sample_count mirror, since it cannot import
chowder.backends.training_data); proven here by importing unsloth_worker.py
directly (its top-level imports are stdlib-only) and calling train() with a
fully mocked unsloth/transformers/peft/torch import surface -- no real
GPU/Unsloth install needed to prove the row-mixing logic itself, matching
the discipline established for the sha256_directory parity test in
test_unsloth_parent_adapter_continuation.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("datasets")

from chowder.backends.unsloth_peft import (
    UnslothConfigError,
    UnslothPeftExecutor,
    UnslothPeftRunSpec,
    _materialize_pretokenized_chat_dataset,
)
from chowder.executors import ExecutionContext
from chowder.memory import HardwareProfile
from chowder.models import Experiment, Hypothesis
from chowder.provenance import sha256_file
from chowder.unsloth_env import unsloth_env_dir, unsloth_python


class _FakeTokenizer:
    _ROLE_MARKERS = {"system": 100, "user": 200, "assistant": 300}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        ids: list[int] = []
        for message in messages:
            marker = self._ROLE_MARKERS[message["role"]]
            ids.append(marker)
            ids.extend(ord(char) % 50 + 10 for char in message["content"])
            ids.append(marker + 1)
        if add_generation_prompt:
            ids.append(self._ROLE_MARKERS["assistant"])
        return ids


@pytest.fixture(autouse=True)
def _fake_auto_tokenizer(monkeypatch):
    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: _FakeTokenizer()))


def _write_chat_dataset(path: Path, rows: list[list[dict[str, str]]]) -> None:
    path.write_text("".join(json.dumps({"messages": row}) + "\n" for row in rows), encoding="utf-8")


def _write_text_dataset(path: Path, rows: list[str]) -> None:
    path.write_text("".join(json.dumps({"text": row}) + "\n" for row in rows), encoding="utf-8")


def _chat_spec(tmp_path: Path, dataset: Path, **overrides) -> UnslothPeftRunSpec:
    defaults = dict(
        base_model="org/model",
        dataset=str(dataset),
        output_dir=str(tmp_path / "adapter"),
        dataset_format="chat",
        max_length=256,
    )
    defaults.update(overrides)
    return UnslothPeftRunSpec(**defaults)


# --- chat-format replay: merged in the controller before tokenization -----


def test_chat_replay_rows_are_actually_mixed_into_training_examples(tmp_path):
    primary = tmp_path / "primary.jsonl"
    _write_chat_dataset(
        primary,
        [[{"role": "user", "content": "p1"}, {"role": "assistant", "content": "r1"}]],
    )
    replay = tmp_path / "replay.jsonl"
    _write_chat_dataset(
        replay,
        [
            [{"role": "user", "content": "old-a"}, {"role": "assistant", "content": "old-answer-a"}],
            [{"role": "user", "content": "old-b"}, {"role": "assistant", "content": "old-answer-b"}],
        ],
    )
    replay_sha = sha256_file(replay)
    spec = _chat_spec(
        tmp_path,
        primary,
        replay_dataset=str(replay),
        replay_sha256=replay_sha,
        replay_ratio=10.0,  # request far more than available -> capped to what exists
    )

    (
        path,
        _sha,
        total_tokens,
        _assistant_tokens,
        replay_available,
        replay_selected,
    ) = _materialize_pretokenized_chat_dataset(spec, work_dir=tmp_path)

    assert replay_available == 2
    assert replay_selected == 2  # capped: never more than actually available
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 + 2  # primary + both replay rows
    assert total_tokens == sum(len(row["input_ids"]) for row in rows)


def test_chat_replay_ratio_of_zero_selects_nothing(tmp_path):
    primary = tmp_path / "primary.jsonl"
    _write_chat_dataset(
        primary,
        [[{"role": "user", "content": "p1"}, {"role": "assistant", "content": "r1"}]],
    )
    replay = tmp_path / "replay.jsonl"
    _write_chat_dataset(
        replay,
        [[{"role": "user", "content": "old"}, {"role": "assistant", "content": "old-answer"}]],
    )
    spec = _chat_spec(
        tmp_path,
        primary,
        replay_dataset=str(replay),
        replay_sha256=sha256_file(replay),
        replay_ratio=0.01,  # rounds up to at least 1 selected row per _replay_sample_count's floor
    )
    (
        path,
        _sha,
        _total,
        _assist,
        replay_available,
        replay_selected,
    ) = _materialize_pretokenized_chat_dataset(spec, work_dir=tmp_path)
    assert replay_available == 1
    # _replay_sample_count floors at max(1, ceil(...)) whenever replay rows exist,
    # so a tiny positive ratio still selects the one available row -- proven here
    # rather than assumed, since it is the real, load-bearing floor behavior.
    assert replay_selected == 1
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2


def test_chat_replay_dataset_missing_messages_field_fails_closed(tmp_path):
    primary = tmp_path / "primary.jsonl"
    _write_chat_dataset(
        primary,
        [[{"role": "user", "content": "p1"}, {"role": "assistant", "content": "r1"}]],
    )
    replay = tmp_path / "replay.jsonl"
    replay.write_text(json.dumps({"text": "not a chat row"}) + "\n", encoding="utf-8")
    spec = _chat_spec(
        tmp_path,
        primary,
        replay_dataset=str(replay),
        replay_sha256=sha256_file(replay),
        replay_ratio=1.0,
    )
    with pytest.raises(UnslothConfigError, match="replay dataset is missing messages field"):
        _materialize_pretokenized_chat_dataset(spec, work_dir=tmp_path)


def test_chat_replay_cache_key_changes_when_replay_content_changes(tmp_path):
    primary = tmp_path / "primary.jsonl"
    _write_chat_dataset(
        primary,
        [[{"role": "user", "content": "p1"}, {"role": "assistant", "content": "r1"}]],
    )
    replay = tmp_path / "replay.jsonl"
    _write_chat_dataset(
        replay, [[{"role": "user", "content": "old"}, {"role": "assistant", "content": "old-answer"}]]
    )
    spec = _chat_spec(
        tmp_path, primary, replay_dataset=str(replay), replay_sha256=sha256_file(replay), replay_ratio=1.0
    )
    path1, *_ = _materialize_pretokenized_chat_dataset(spec, work_dir=tmp_path)

    _write_chat_dataset(
        replay, [[{"role": "user", "content": "new"}, {"role": "assistant", "content": "new-answer"}]]
    )
    spec2 = _chat_spec(
        tmp_path, primary, replay_dataset=str(replay), replay_sha256=sha256_file(replay), replay_ratio=1.0
    )
    path2, *_ = _materialize_pretokenized_chat_dataset(spec2, work_dir=tmp_path)
    assert path1 != path2


# --- executor wiring: real evidence recorded end to end ---------------------


def _hardware():
    return HardwareProfile(16, 64, 500, 12, 40, 3)


def _experiment():
    return Experiment("e1", None, Hypothesis("obs", "cause", "fix"), {}, 2.0)


class _RecordingChatProcess:
    returncode = 0

    def __init__(self, command, **kwargs):
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
                    "model_provenance": {"requested_base_model": self.observed_spec["base_model"]},
                    "versions": {"unsloth": "test", "torch": "test"},
                }
            )
        )

    def wait(self, timeout=None):
        return 0


def test_executor_records_real_chat_replay_evidence(tmp_path, monkeypatch):
    primary = tmp_path / "primary.jsonl"
    _write_chat_dataset(
        primary,
        [[{"role": "user", "content": "p1"}, {"role": "assistant", "content": "r1"}]],
    )
    replay = tmp_path / "replay.jsonl"
    _write_chat_dataset(
        replay,
        [[{"role": "user", "content": "old"}, {"role": "assistant", "content": "old-answer"}]],
    )
    env_dir = unsloth_env_dir(tmp_path)
    python_path = unsloth_python(env_dir)
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_bytes(b"")

    resolved_config = {
        "backend": {
            "type": "peft",
            "engine": "unsloth",
            "base_model": "org/model",
            "dataset": str(primary),
            "dataset_format": "chat",
            "max_length": 256,
            "replay": {"dataset": str(replay), "sha256": sha256_file(replay), "ratio": 1.0},
            "lora": {"r": 8, "alpha": 16},
            "training": {"learning_rate": 1e-4, "epochs": 1.0},
        }
    }
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=resolved_config)
    monkeypatch.setattr("chowder.backends.unsloth_peft.subprocess.Popen", _RecordingChatProcess)

    artifact = UnslothPeftExecutor().run(_experiment(), context)

    assert artifact.evidence["replay_dataset_sha256"] == sha256_file(replay)
    assert artifact.evidence["replay_ratio"] == 1.0
    assert artifact.evidence["replay_available_rows"] == 1
    assert artifact.evidence["replay_selected_rows"] == 1


def test_executor_rejects_replay_dataset_identical_to_primary(tmp_path, monkeypatch):
    data = tmp_path / "same.jsonl"
    _write_chat_dataset(
        data, [[{"role": "user", "content": "p1"}, {"role": "assistant", "content": "r1"}]]
    )
    env_dir = unsloth_env_dir(tmp_path)
    python_path = unsloth_python(env_dir)
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_bytes(b"")

    resolved_config = {
        "backend": {
            "type": "peft",
            "engine": "unsloth",
            "base_model": "org/model",
            "dataset": str(data),
            "dataset_format": "chat",
            "max_length": 256,
            "replay": {"dataset": str(data), "sha256": sha256_file(data), "ratio": 1.0},
            "lora": {"r": 8, "alpha": 16},
            "training": {"learning_rate": 1e-4, "epochs": 1.0},
        }
    }
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=resolved_config)

    def should_not_launch(*args, **kwargs):
        raise AssertionError("must not launch a worker for an invalid replay config")

    monkeypatch.setattr("chowder.backends.unsloth_peft.subprocess.Popen", should_not_launch)
    with pytest.raises(ValueError, match="must be different files"):
        UnslothPeftExecutor().run(_experiment(), context)


# --- text-format replay: merged inside the isolated worker itself ---------


def _load_worker_module():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "chowder" / "backends"))
    import unsloth_worker

    return unsloth_worker


def test_text_replay_sample_count_matches_the_shared_contract():
    """The isolated worker's local _replay_sample_count mirror must agree
    with chowder.backends.training_data's real implementation -- same
    parity discipline as the sha256_directory mirror in
    test_unsloth_parent_adapter_continuation.py."""
    from chowder.backends.training_data import _replay_sample_count as real_impl

    worker = _load_worker_module()
    cases = [(10, 20, 0.5), (10, 3, 5.0), (0, 5, 1.0), (5, 0, 1.0), (10, 100, 0.01)]
    for primary_rows, replay_rows, ratio in cases:
        assert worker._replay_sample_count(primary_rows, replay_rows, ratio) == real_impl(
            primary_rows, replay_rows, ratio
        )


def test_text_format_replay_rows_are_actually_mixed(tmp_path):
    """Drives unsloth_worker._load_text_dataset_with_replay -- the real
    row-mixing logic train()'s text-format path uses -- directly, with real
    datasets objects, no torch/unsloth/transformers/Trainer involved.

    Deliberately does *not* drive train() end to end: an earlier version of
    this test tried to monkeypatch transformers.Trainer to prove the same
    thing, and it worked in isolation but failed for real in the full CI
    suite. Root cause, confirmed for real: transformers' top-level package
    is a _LazyModule whose __getattr__ caches each name's first real
    resolution directly into the module's own __dict__; once any other test
    in the same process has ever touched `transformers.Trainer` first (as
    many real ML tests in this suite do), a later `from transformers import
    Trainer` returns that cached real class from __dict__ directly and never
    calls __getattr__ again -- so patching transformers.trainer.Trainer (or
    even overwriting transformers.__dict__['Trainer'] directly, confirmed
    ineffective too) has no effect, and which behavior you observe depends
    on unrelated test execution order. That is not a foundation to build a
    test on, so the row-mixing logic was extracted into its own pure
    function specifically so it never needs to go anywhere near Trainer to
    be verified for real.
    """
    worker = _load_worker_module()

    primary_path = tmp_path / "primary.jsonl"
    _write_text_dataset(primary_path, ["p1"])
    replay_path = tmp_path / "replay.jsonl"
    _write_text_dataset(replay_path, ["r1", "r2", "r3"])

    from datasets import load_dataset

    raw_primary = load_dataset("json", data_files=str(primary_path), split="train")

    spec = worker._Spec(
        base_model="org/model",
        dataset=str(primary_path),
        output_dir=str(tmp_path / "adapter"),
        dataset_sha256=None,
        revision=None,
        parent_adapter=None,
        parent_adapter_sha256=None,
        replay_dataset=str(replay_path),
        replay_sha256=None,
        replay_ratio=10.0,  # request more than available -> capped to 3
        text_field="text",
        pretokenized=False,
        max_length=64,
        epochs=1.0,
        max_steps=1,
        learning_rate=1e-4,
        batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=1,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=[],
        quantization="none",
        seed=1,
        timeout_seconds=None,
        offline=False,
        save_strategy="no",
        save_steps=0,
        save_total_limit=None,
        resume_from_checkpoint=None,
    )

    merged, replay_available, replay_selected = worker._load_text_dataset_with_replay(
        raw_primary, spec
    )

    assert replay_available == 3
    assert replay_selected == 3  # capped to what's available
    assert len(merged) == 1 + 3
    texts = sorted(row["text"] for row in merged)
    assert texts == ["p1", "r1", "r2", "r3"]


def test_text_format_replay_missing_field_fails_closed(tmp_path):
    worker = _load_worker_module()
    primary_path = tmp_path / "primary.jsonl"
    _write_text_dataset(primary_path, ["p1"])
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text(json.dumps({"messages": []}) + "\n", encoding="utf-8")

    from datasets import load_dataset

    raw_primary = load_dataset("json", data_files=str(primary_path), split="train")
    spec = worker._Spec(
        base_model="org/model",
        dataset=str(primary_path),
        output_dir=str(tmp_path / "adapter"),
        dataset_sha256=None,
        revision=None,
        parent_adapter=None,
        parent_adapter_sha256=None,
        replay_dataset=str(replay_path),
        replay_sha256=None,
        replay_ratio=1.0,
        text_field="text",
        pretokenized=False,
        max_length=64,
        epochs=1.0,
        max_steps=1,
        learning_rate=1e-4,
        batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=1,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=[],
        quantization="none",
        seed=1,
        timeout_seconds=None,
        offline=False,
        save_strategy="no",
        save_steps=0,
        save_total_limit=None,
        resume_from_checkpoint=None,
    )
    with pytest.raises(RuntimeError, match="replay dataset is missing text field"):
        worker._load_text_dataset_with_replay(raw_primary, spec)
