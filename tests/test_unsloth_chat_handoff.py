"""Real tests for the Unsloth chat-format deterministic handoff.

`_materialize_pretokenized_chat_dataset` (unsloth_peft.py, controller-side)
renders every chat row through the exact same shared contract
(`chowder.backends.training_data._validate_chat_messages`/
`_build_chat_example`) that `transformers_worker.py` uses, then writes a
plain {input_ids, attention_mask, labels} JSONL the isolated Unsloth worker
loads with zero chat-template/masking logic of its own. These tests prove:
the materialized rows are byte-identical to what the shared contract
produces directly (parity by construction, verified for real); the cache
is content-addressed (same inputs reuse it, any real input change
invalidates it); structural errors (malformed role, no assistant turn)
surface at materialization time, before any subprocess/GPU work; and the
full executor wiring (mocked subprocess, no real Unsloth/CUDA) hands the
worker a pretokenized dataset and records real evidence.

Uses the same deterministic fake chat template as test_training_data.py
(no network/real tokenizer download) -- only AutoTokenizer.from_pretrained
is monkeypatched; _build_chat_example/_validate_chat_messages run for real.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Every test in this file exercises real chat materialization, which needs
# transformers + datasets in the *controller* process (see
# _materialize_pretokenized_chat_dataset's docstring). The base CI jobs
# install only chowder[dev], not chowder[train], so importing these
# unconditionally at module level would break collection for the whole
# file there -- matches this repo's existing convention (see
# test_moe_instrumentation.py) for a torch/transformers-needing test file.
pytest.importorskip("transformers")
pytest.importorskip("datasets")

from chowder.backends.unsloth_peft import (
    UnslothConfigError,
    UnslothPeftExecutor,
    UnslothPeftRunSpec,
    _materialize_pretokenized_chat_dataset,
)
from chowder.backends.training_data import _build_chat_example
from chowder.executors import ExecutionContext
from chowder.memory import HardwareProfile
from chowder.models import Experiment, Hypothesis


class _FakeTokenizer:
    """Same deterministic, prefix-consistent fake chat template as
    test_training_data.py's own _FakeTokenizer -- see that file for the
    exact rationale. Duplicated locally rather than imported across test
    modules, matching this repo's test-file convention."""

    _ROLE_MARKERS = {"system": 100, "user": 200, "assistant": 300}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
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
    path.write_text(
        "".join(json.dumps({"messages": row}) + "\n" for row in rows), encoding="utf-8"
    )


def _spec(tmp_path: Path, dataset: Path, **overrides) -> UnslothPeftRunSpec:
    defaults = dict(
        base_model="org/model",
        dataset=str(dataset),
        output_dir=str(tmp_path / "adapter"),
        dataset_format="chat",
        max_length=256,
    )
    defaults.update(overrides)
    return UnslothPeftRunSpec(**defaults)


# --- _materialize_pretokenized_chat_dataset: parity + correctness ---------


def test_materialized_rows_are_byte_identical_to_the_shared_contract(tmp_path):
    rows = [
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"},
        ],
    ]
    dataset = tmp_path / "chat.jsonl"
    _write_chat_dataset(dataset, rows)
    spec = _spec(tmp_path, dataset)

    path, sha, total_tokens, assistant_tokens = _materialize_pretokenized_chat_dataset(
        spec, work_dir=tmp_path
    )

    materialized = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    tokenizer = _FakeTokenizer()
    expected = [
        _build_chat_example(tokenizer, row, max_length=spec.max_length, row_index=i)
        for i, row in enumerate(rows)
    ]
    assert materialized == expected
    assert total_tokens == sum(len(row["input_ids"]) for row in expected)
    assert assistant_tokens == sum(
        sum(1 for label in row["labels"] if label != -100) for row in expected
    )
    from chowder.provenance import sha256_file

    assert sha == sha256_file(path)


def test_materialization_is_content_addressed_and_reused(tmp_path):
    rows = [[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]]
    dataset = tmp_path / "chat.jsonl"
    _write_chat_dataset(dataset, rows)
    spec = _spec(tmp_path, dataset)

    path1, sha1, _, _ = _materialize_pretokenized_chat_dataset(spec, work_dir=tmp_path)
    path2, sha2, _, _ = _materialize_pretokenized_chat_dataset(spec, work_dir=tmp_path)
    assert path1 == path2
    assert sha1 == sha2


@pytest.mark.parametrize(
    "mutate",
    [
        lambda spec: spec.__class__(**{**spec.to_dict(), "max_length": 64}),
        lambda spec: spec.__class__(**{**spec.to_dict(), "base_model": "org/other-model"}),
    ],
)
def test_materialization_cache_key_changes_with_real_inputs(tmp_path, mutate):
    rows = [[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]]
    dataset = tmp_path / "chat.jsonl"
    _write_chat_dataset(dataset, rows)
    spec = _spec(tmp_path, dataset)
    path1, _, _, _ = _materialize_pretokenized_chat_dataset(spec, work_dir=tmp_path)
    path2, _, _, _ = _materialize_pretokenized_chat_dataset(mutate(spec), work_dir=tmp_path)
    assert path1 != path2


def test_changed_dataset_content_invalidates_the_cache(tmp_path):
    dataset = tmp_path / "chat.jsonl"
    _write_chat_dataset(
        dataset, [[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]]
    )
    spec = _spec(tmp_path, dataset)
    path1, sha1, _, _ = _materialize_pretokenized_chat_dataset(spec, work_dir=tmp_path)

    _write_chat_dataset(
        dataset, [[{"role": "user", "content": "bye"}, {"role": "assistant", "content": "later"}]]
    )
    path2, sha2, _, _ = _materialize_pretokenized_chat_dataset(spec, work_dir=tmp_path)
    assert path1 != path2
    assert sha1 != sha2


# --- Regression cases named by the Qwen3.8 program directive --------------


def test_multi_turn_multiple_assistant_turns_and_system_prompt(tmp_path):
    rows = [
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        ]
    ]
    dataset = tmp_path / "chat.jsonl"
    _write_chat_dataset(dataset, rows)
    spec = _spec(tmp_path, dataset)
    path, _, _, assistant_tokens = _materialize_pretokenized_chat_dataset(spec, work_dir=tmp_path)
    row = json.loads(Path(path).read_text(encoding="utf-8"))
    labeled = [label for label in row["labels"] if label != -100]
    assert labeled, "no assistant tokens labeled across two assistant turns"
    assert assistant_tokens == len(labeled)
    # System and user tokens must not receive training labels.
    label_positions = {i for i, label in enumerate(row["labels"]) if label != -100}
    for i in label_positions:
        assert row["input_ids"][i] == row["labels"][i]


def test_unicode_content_round_trips(tmp_path):
    rows = [[{"role": "user", "content": "héllo 世界"}, {"role": "assistant", "content": "🎉ok"}]]
    dataset = tmp_path / "chat.jsonl"
    _write_chat_dataset(dataset, rows)
    spec = _spec(tmp_path, dataset)
    path, _, _, assistant_tokens = _materialize_pretokenized_chat_dataset(spec, work_dir=tmp_path)
    row = json.loads(Path(path).read_text(encoding="utf-8"))
    assert assistant_tokens > 0
    assert any(label != -100 for label in row["labels"])


def test_truncation_before_assistant_response_raises_at_materialization_time(tmp_path):
    """max_length so small the assistant turn is truncated away entirely
    must fail loudly (nothing to train on), not silently train on zero
    labeled tokens."""
    rows = [[{"role": "user", "content": "hello there"}, {"role": "assistant", "content": "hi"}]]
    dataset = tmp_path / "chat.jsonl"
    _write_chat_dataset(dataset, rows)
    spec = _spec(tmp_path, dataset, max_length=1)
    with pytest.raises(RuntimeError, match="no assistant tokens remain"):
        _materialize_pretokenized_chat_dataset(spec, work_dir=tmp_path)


def test_truncation_inside_assistant_response_keeps_a_labeled_prefix(tmp_path):
    rows = [[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "a longer answer"}]]
    dataset = tmp_path / "chat.jsonl"
    _write_chat_dataset(dataset, rows)
    full_spec = _spec(tmp_path, dataset, max_length=256)
    full_path, _, _, _ = _materialize_pretokenized_chat_dataset(full_spec, work_dir=tmp_path)
    full_row = json.loads(Path(full_path).read_text(encoding="utf-8"))

    truncated_len = len(full_row["input_ids"]) - 2
    truncated_spec = _spec(tmp_path, dataset, max_length=truncated_len)
    truncated_path, _, _, _ = _materialize_pretokenized_chat_dataset(
        truncated_spec, work_dir=tmp_path
    )
    truncated_row = json.loads(Path(truncated_path).read_text(encoding="utf-8"))
    assert len(truncated_row["input_ids"]) == truncated_len
    assert truncated_row["input_ids"] == full_row["input_ids"][:truncated_len]
    assert any(label != -100 for label in truncated_row["labels"])


def test_empty_assistant_content_still_labels_the_real_turn_markers(tmp_path):
    """An assistant turn with empty *content* is not the same as "no
    assistant tokens": a real chat template still wraps even an empty
    completion in real turn-boundary tokens (e.g. `<|assistant|>...<|eot|>`),
    so those tokens are real, correctly-labeled training signal, not
    nothing. Only a template producing a genuinely empty completion span
    entirely (see the truncation test) should raise."""
    rows = [[{"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}]]
    dataset = tmp_path / "chat.jsonl"
    _write_chat_dataset(dataset, rows)
    spec = _spec(tmp_path, dataset)
    path, _, _, assistant_tokens = _materialize_pretokenized_chat_dataset(spec, work_dir=tmp_path)
    row = json.loads(Path(path).read_text(encoding="utf-8"))
    assert assistant_tokens > 0
    assert any(label != -100 for label in row["labels"])


def test_malformed_role_sequence_fails_before_any_subprocess_work(tmp_path):
    dataset = tmp_path / "chat.jsonl"
    dataset.write_text(
        json.dumps({"messages": [{"role": "narrator", "content": "once upon a time"}]}) + "\n",
        encoding="utf-8",
    )
    spec = _spec(tmp_path, dataset)
    with pytest.raises(RuntimeError, match="unsupported message role"):
        _materialize_pretokenized_chat_dataset(spec, work_dir=tmp_path)


def test_no_assistant_turn_at_all_fails_closed(tmp_path):
    dataset = tmp_path / "chat.jsonl"
    dataset.write_text(
        json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n", encoding="utf-8"
    )
    spec = _spec(tmp_path, dataset)
    with pytest.raises(RuntimeError, match="no assistant turn"):
        _materialize_pretokenized_chat_dataset(spec, work_dir=tmp_path)


def test_missing_messages_field_raises_unsloth_config_error(tmp_path):
    dataset = tmp_path / "chat.jsonl"
    dataset.write_text(json.dumps({"text": "hello"}) + "\n", encoding="utf-8")
    spec = _spec(tmp_path, dataset)
    with pytest.raises(UnslothConfigError, match="missing messages field"):
        _materialize_pretokenized_chat_dataset(spec, work_dir=tmp_path)


def test_long_conversation_all_assistant_turns_get_real_labels(tmp_path):
    conversation = [{"role": "user", "content": "start"}]
    for i in range(8):
        conversation.append({"role": "assistant", "content": f"reply {i}"})
        conversation.append({"role": "user", "content": f"follow-up {i}"})
    rows = [conversation]
    dataset = tmp_path / "chat.jsonl"
    _write_chat_dataset(dataset, rows)
    spec = _spec(tmp_path, dataset, max_length=4096)
    _, _, _, assistant_tokens = _materialize_pretokenized_chat_dataset(spec, work_dir=tmp_path)
    assert assistant_tokens > 8  # at least one real token per assistant turn


# --- Full executor wiring (mocked subprocess, no real Unsloth/CUDA) -------


def _hardware():
    return HardwareProfile(16, 64, 500, 12, 40, 3)


def _experiment():
    return Experiment("e1", None, Hypothesis("obs", "cause", "fix"), {}, 2.0)


class _RecordingChatProcess:
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
                    "model_provenance": {"requested_base_model": self.observed_spec["base_model"]},
                    "versions": {"unsloth": "test", "torch": "test"},
                }
            )
        )

    def wait(self, timeout=None):
        return 0


def test_executor_hands_the_worker_a_pretokenized_chat_dataset(tmp_path, monkeypatch):
    from chowder.unsloth_env import unsloth_env_dir, unsloth_python

    dataset = tmp_path / "chat.jsonl"
    _write_chat_dataset(
        dataset,
        [[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello there"}]],
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
            "dataset": str(dataset),
            "dataset_format": "chat",
            "max_length": 256,
            "lora": {"r": 8, "alpha": 16},
            "training": {"learning_rate": 1e-4, "epochs": 1.0},
        }
    }
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=resolved_config)

    captured = {}

    class Recording(_RecordingChatProcess):
        def __init__(self, command, **kwargs):
            super().__init__(command, **kwargs)
            captured["process"] = self

    monkeypatch.setattr("chowder.backends.unsloth_peft.subprocess.Popen", Recording)

    artifact = UnslothPeftExecutor().run(_experiment(), context)

    observed = captured["process"].observed_spec
    assert observed["pretokenized"] is True
    assert observed["dataset_format"] == "chat"
    materialized_rows = [
        json.loads(line) for line in Path(observed["dataset"]).read_text(encoding="utf-8").splitlines()
    ]
    assert set(materialized_rows[0]) == {"input_ids", "attention_mask", "labels"}

    assert artifact.evidence["dataset_format"] == "chat"
    assert artifact.evidence["pretokenized"] is True
    assert artifact.evidence["chat_assistant_token_count"] > 0
    assert artifact.evidence["chat_total_token_count"] >= artifact.evidence["chat_assistant_token_count"]
