"""P7: a real save -> checkpoint -> fresh-process resume, compared to a control.

This file runs the real worker on a tiny real model on CPU and asks:

    does a resumed run end up where the uninterrupted run ended up?

The control keeps the same total scheduler horizon. A run that trains 4 steps
with ``max_steps=4`` and is then continued with ``max_steps=8`` is a *different
schedule* (the linear decay is computed over a different total), so it could not
prove exact resume even if it looked right. Here the control trains all 8 steps
with ``max_steps=8`` and checkpoints at step 4 mid-flight; the resumed run is a
separate interpreter from that same checkpoint. Distinct tokenized inputs,
matching losses/LRs at steps 5-8, and matching final adapters test continuation
equivalence for this fixture. This is not a forced-interruption test or a
qualification of the actual GPU/router workload.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("datasets")
pytest.importorskip("tokenizers")
pytest.importorskip("safetensors")

from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from chowder.backends.transformers_peft import TransformersPeftRunSpec  # noqa: E402
from chowder.backends.transformers_worker import train  # noqa: E402
from chowder.resume_state import (  # noqa: E402
    IncompleteCheckpointError,
    assert_resumable,
    inventory_checkpoint,
)
from chowder.worker_env import chowder_source_identity, worker_env  # noqa: E402

TOTAL_STEPS = 8
CHECKPOINT_STEP = 4
ADAPTER_TOLERANCE = 1e-6
# Fixed CPU test tolerances, not research/promotion thresholds. Restored history
# and integer counters remain exact; new float computations use zero relative tolerance.
LOSS_ABS_TOLERANCE = 1e-6
LR_ABS_TOLERANCE = 1e-12


def _bytes_to_unicode() -> dict[int, str]:
    """The GPT2 byte<->unicode alphabet, written out rather than imported.

    Transformers has moved this helper between modules across versions, and the
    CI job installs whatever `.[train]` resolves to; a private import would turn
    a version bump into a collection error instead of a skip.
    """
    printable = list(range(ord("!"), ord("~") + 1)) + list(
        range(ord("¡"), ord("¬") + 1)
    ) + list(range(ord("®"), ord("ÿ") + 1))
    codepoints = list(printable)
    offset = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codepoints.append(256 + offset)
            offset += 1
    return {byte: chr(codepoint) for byte, codepoint in zip(printable, codepoints)}


def _write_tiny_base(directory: Path) -> Path:
    """A real (if tiny) causal LM plus a real byte-level BPE tokenizer.

    Both files must sit in one directory because the worker loads the tokenizer
    and the model from the same ``base_model`` path, and the model's ``config.json``
    decides which tokenizer class ``AutoTokenizer`` resolves to -- so the
    tokenizer has to be in the format that class actually reads. A hand-built
    ``tokenizer.json`` is silently ignored by the resolved class and yields
    *empty* training examples rather than an error, which is how this fixture
    first failed.
    """
    directory.mkdir(parents=True, exist_ok=True)
    vocab = {
        token: index for index, token in enumerate(_bytes_to_unicode().values())
    }
    vocab.update(
        {"<pad>": len(vocab), "<unk>": len(vocab) + 1, "</s>": len(vocab) + 2}
    )
    (directory / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (directory / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "Qwen2Tokenizer",
                "model_max_length": 64,
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "eos_token": "</s>",
                "add_bos_token": False,
                "add_eos_token": False,
                "clean_up_tokenization_spaces": False,
            }
        ),
        encoding="utf-8",
    )
    (directory / "special_tokens_map.json").write_text(
        json.dumps({"unk_token": "<unk>", "pad_token": "<pad>", "eos_token": "</s>"}),
        encoding="utf-8",
    )
    torch.manual_seed(0)
    Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(vocab),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
    ).save_pretrained(directory, safe_serialization=True)
    return directory


def _write_dataset(path: Path) -> Path:
    words = ["one", "two", "three", "four", "five", "six", "seven", "eight"]
    path.write_text(
        "".join(
            json.dumps({"text": f"{word}: the cat sat on the mat"}) + "\n"
            for word in words
        ),
        encoding="utf-8",
    )
    return path


def test_fixture_preserves_distinct_samples_after_training_truncation(tmp_path):
    from transformers import AutoTokenizer

    base = _write_tiny_base(tmp_path / "base")
    dataset = _write_dataset(tmp_path / "train.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    texts = [json.loads(row)["text"] for row in dataset.read_text().splitlines()]
    rows = tokenizer(texts, truncation=True, max_length=16)["input_ids"]
    assert len({tuple(row) for row in rows}) == len(texts) == 8


def _spec(base: Path, dataset: Path, output: Path, **overrides) -> TransformersPeftRunSpec:
    kwargs: dict = {
        "base_model": str(base),
        "dataset": str(dataset),
        "output_dir": str(output),
        "quantization": "none",
        "precision": "fp32",
        "max_length": 16,
        "epochs": 1,
        "max_steps": TOTAL_STEPS,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-2,
        "lr_scheduler_type": "linear",
        "warmup_steps": 0,
        "logging_steps": 1,
        "lora_r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "target_modules": ("q_proj", "v_proj"),
        "target_preset": "auto",
        "gradient_checkpointing": False,
        "seed": 7,
        "save_strategy": "steps",
        "save_steps": CHECKPOINT_STEP,
    }
    kwargs.update(overrides)
    return TransformersPeftRunSpec(**kwargs)


@pytest.fixture(scope="module")
def continuation(tmp_path_factory):
    """An 8-step control and a checkpoint-4 resume in separate interpreters."""
    root = tmp_path_factory.mktemp("p7-resume")
    base = _write_tiny_base(root / "base")
    dataset = _write_dataset(root / "train.jsonl")
    identity_path = root / "source-identity.json"
    identity_path.write_text(json.dumps(chowder_source_identity()), encoding="utf-8")

    def run_worker(name, **overrides):
        spec = _spec(base, dataset, root / name, **overrides)
        spec_path, result_path = root / f"{name}-spec.json", root / f"{name}-result.json"
        spec_path.write_text(spec.canonical_json(), encoding="utf-8")
        process = subprocess.run(
            [sys.executable, "-m", "chowder.backends.transformers_worker",
             "--spec", str(spec_path), "--result", str(result_path),
             "--chowder-identity", str(identity_path)],
            env=worker_env({
                "CUDA_VISIBLE_DEVICES": "-1", "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
            }),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=180, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        assert process.returncode == 0, process.stdout + process.stderr
        return json.loads(result_path.read_text(encoding="utf-8"))

    control = run_worker("control")
    checkpoint = root / "control" / "trainer" / f"checkpoint-{CHECKPOINT_STEP}"
    resumed = run_worker("resumed", resume_from_checkpoint=str(checkpoint))
    return {
        "root": root,
        "control": control,
        "resumed": resumed,
        "checkpoint": checkpoint,
    }


def _entries(result) -> list[dict]:
    return result["telemetry"]["step_log"]["entries"]


def test_the_checkpoint_holds_the_state_a_resume_needs(continuation):
    inventory = inventory_checkpoint(continuation["checkpoint"])

    assert inventory.state == "complete"
    assert inventory.global_step == CHECKPOINT_STEP
    assert inventory.missing_required == ()
    # Require recorded RNG state as well as optimizer/scheduler state. Equivalence
    # below remains specific to this deterministic, no-dropout fixture.
    assert "rng_state" in inventory.present
    assert_resumable(inventory, require_rng=True)


def test_the_resume_advanced_past_the_restore_point(continuation):
    witness = continuation["resumed"]["telemetry"]["resume"]

    assert witness is not None
    assert witness["restored_global_step"] == CHECKPOINT_STEP
    assert witness["final_global_step"] == TOTAL_STEPS
    assert witness["steps_executed"] == TOTAL_STEPS - CHECKPOINT_STEP
    assert witness["optimizer_state_present"] is True
    assert witness["matched"] is True
    assert witness["reason"] is None
    assert witness["progress_state"] == "advanced"
    # Same declared horizon as the checkpoint recorded, so this is an exact
    # resume rather than a longer continuation with a recomputed schedule.
    assert witness["restored_max_steps"] == TOTAL_STEPS
    assert witness["horizon_changed"] is False


def test_the_resumed_run_reproduces_the_control_loss_and_lr_sequence(continuation):
    control_entries = _entries(continuation["control"])
    resumed_entries = _entries(continuation["resumed"])

    assert [entry["step"] for entry in control_entries] == list(
        range(1, TOTAL_STEPS + 1)
    )
    assert [entry["step"] for entry in resumed_entries] == list(
        range(1, TOTAL_STEPS + 1)
    )
    # Include newly executed steps, not just history copied from the checkpoint.
    # The separate fixture check ensures truncation did not collapse the inputs.
    assert [entry["loss"] for entry in resumed_entries[CHECKPOINT_STEP:]] == pytest.approx(
        [entry["loss"] for entry in control_entries[CHECKPOINT_STEP:]],
        rel=0, abs=LOSS_ABS_TOLERANCE,
    )
    # And the scheduler continued rather than restarting: the LR at the step
    # after the restore point must be the next step of the same decay, not a
    # fresh maximum.
    assert [entry["learning_rate"] for entry in resumed_entries[CHECKPOINT_STEP:]] == pytest.approx(
        [entry["learning_rate"] for entry in control_entries[CHECKPOINT_STEP:]],
        rel=0, abs=LR_ABS_TOLERANCE,
    )


def test_the_resumed_run_restored_the_recorded_history(continuation):
    resumed_entries = _entries(continuation["resumed"])
    control_entries = _entries(continuation["control"])

    # The prefix came *from the checkpoint*, not from a rerun of steps 1-4:
    # this is the same run reporting its restored history plus its new steps.
    assert resumed_entries[:CHECKPOINT_STEP] == control_entries[:CHECKPOINT_STEP]
    assert (
        continuation["resumed"]["telemetry"]["global_step"]
        == continuation["control"]["telemetry"]["global_step"]
        == TOTAL_STEPS
    )


def test_the_final_adapters_are_the_same_weights(continuation):
    from safetensors.torch import load_file

    root = continuation["root"]
    control = load_file(str(root / "control" / "adapter_model.safetensors"))
    resumed = load_file(str(root / "resumed" / "adapter_model.safetensors"))

    assert sorted(control) == sorted(resumed)
    worst = 0.0
    for key in control:
        worst = max(
            worst,
            float((control[key].float() - resumed[key].float()).abs().max().item()),
        )
    assert worst <= ADAPTER_TOLERANCE, f"resumed adapter diverged by {worst}"


def test_the_worker_refuses_a_checkpoint_without_optimizer_state(tmp_path):
    """The refusal costs nothing: it happens before the model is loaded.

    A directory holding only a trainer-state file is exactly what a process
    killed mid-save leaves behind, and Transformers would happily restore the
    weights from it while starting the optimizer over.
    """
    dataset = _write_dataset(tmp_path / "train.jsonl")
    checkpoint = tmp_path / "checkpoint-4"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 4}), encoding="utf-8"
    )

    spec = _spec(
        tmp_path / "base-that-is-never-loaded",
        dataset,
        tmp_path / "out",
        resume_from_checkpoint=str(checkpoint),
    )
    with pytest.raises(IncompleteCheckpointError, match="optimizer"):
        train(spec)


def test_partial_save_inventory_does_not_invalidate_last_complete_checkpoint(tmp_path):
    """Synthetic inventory control, not a real cancellation/worker-death test.

    A partial newer directory must not invalidate the older complete checkpoint,
    or be silently accepted as resumable itself. Payload integrity is not tested.
    """
    complete = tmp_path / "checkpoint-4"
    complete.mkdir()
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (complete / name).write_bytes(b"state")
    (complete / "trainer_state.json").write_text(
        json.dumps({"global_step": 4}), encoding="utf-8"
    )

    partial = tmp_path / "checkpoint-8"
    partial.mkdir()
    # The save reached training_args.bin and the step file, then the process died.
    (partial / "training_args.bin").write_bytes(b"args")
    (partial / "trainer_state.json").write_text(
        json.dumps({"global_step": 8}), encoding="utf-8"
    )

    assert inventory_checkpoint(complete).is_complete is True
    assert_resumable(inventory_checkpoint(complete), require_rng=True)

    partial_inventory = inventory_checkpoint(partial)
    assert partial_inventory.state == "partial"
    # Step 8 is *recorded* -- and that is exactly why it must not be trusted: the
    # optimizer state that step 8 belongs to is not there.
    assert partial_inventory.global_step == 8
    with pytest.raises(IncompleteCheckpointError, match="optimizer"):
        assert_resumable(partial_inventory)
