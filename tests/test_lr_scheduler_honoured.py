"""A recipe's learning-rate schedule must actually reach the trainer.

The Unsloth engine silently ignored it. `lr_scheduler_type` was declared on the
shared spec, validated against `_ALLOWED_LR_SCHEDULER_TYPES`, and honoured by
`transformers_worker` -- but `unsloth_worker` never read it and never put it in
`TrainingArguments`, so every Unsloth run got the trainer's default **linear**
schedule no matter what the recipe asked for, and said nothing.

It cost a real experiment. `docs/PRUNED_9B_REAL_TRAINING_PREREG.md` pre-registered
"lr 2e-4 cosine" and the run trained on linear. A config key that validates and is
then dropped is worse than one that is rejected: the run looks compliant.

Same for `warmup_ratio` and `warmup_steps`, which were dropped the same way.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chowder.backends.unsloth_peft import UnslothPeftRunSpec


def _config(dataset: str, **training_overrides):
    training = {"learning_rate": 2e-4, "epochs": 1.0}
    training.update(training_overrides)
    return {
        "backend": {
            "type": "peft",
            "engine": "unsloth",
            "base_model": "org/model",
            "dataset": dataset,
            "max_length": 256,
            "lora": {"r": 8, "alpha": 16},
            "training": training,
        }
    }


def _spec(tmp_path: Path, **training_overrides) -> UnslothPeftRunSpec:
    (tmp_path / "train.jsonl").write_text('{"text":"hello"}\n', encoding="utf-8")
    return UnslothPeftRunSpec.from_resolved_config(
        _config("train.jsonl", **training_overrides),
        work_dir=tmp_path,
        output_dir=tmp_path / "adapter",
        seed=7,
    )


# ---- the spec carries it, and carries it onto the wire --------------------------


def test_cosine_survives_into_the_payload_the_worker_receives(tmp_path) -> None:
    """to_dict() IS the wire format: the worker reads these keys off it."""
    spec = _spec(tmp_path, lr_scheduler_type="cosine", warmup_ratio=0.03)
    assert spec.lr_scheduler_type == "cosine"
    payload = spec.to_dict()
    assert payload["lr_scheduler_type"] == "cosine"
    assert payload["warmup_ratio"] == pytest.approx(0.03)
    assert "warmup_steps" in payload


def test_the_default_is_still_linear(tmp_path) -> None:
    spec = _spec(tmp_path)
    assert spec.lr_scheduler_type == "linear"
    assert spec.warmup_ratio == 0.0
    assert spec.warmup_steps == 0


def test_the_schedule_is_part_of_the_spec_digest(tmp_path) -> None:
    """Otherwise two runs on different schedules would be indistinguishable to
    protocol binding."""
    assert _spec(tmp_path).digest() != _spec(tmp_path, lr_scheduler_type="cosine").digest()


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"lr_scheduler_type": "vibes"}, "unsupported lr_scheduler_type"),
        ({"warmup_ratio": 1.0}, r"warmup_ratio must be finite and in \[0, 1\)"),
        ({"warmup_ratio": -0.1}, r"warmup_ratio must be finite and in \[0, 1\)"),
        ({"warmup_steps": -1}, "warmup_steps cannot be negative"),
    ],
)
def test_invalid_schedules_are_refused_not_silently_defaulted(
    tmp_path, overrides, message
) -> None:
    with pytest.raises(ValueError, match=message):
        _spec(tmp_path, **overrides)


def test_every_allowed_scheduler_is_accepted(tmp_path) -> None:
    from chowder.backends.transformers_peft import _ALLOWED_LR_SCHEDULER_TYPES

    for name in sorted(_ALLOWED_LR_SCHEDULER_TYPES):
        assert _spec(tmp_path, lr_scheduler_type=name).lr_scheduler_type == name


def test_both_engines_validate_against_the_same_allowed_set() -> None:
    """One list, so the engines cannot disagree about what a valid schedule is."""
    from chowder.backends import unsloth_peft
    from chowder.backends.transformers_peft import _ALLOWED_LR_SCHEDULER_TYPES

    assert unsloth_peft._ALLOWED_LR_SCHEDULER_TYPES is _ALLOWED_LR_SCHEDULER_TYPES


# ---- and the worker actually hands it to the trainer ----------------------------


def test_both_workers_pass_the_schedule_into_training_arguments() -> None:
    """The defect was here and nowhere else: the spec was fine, the worker dropped
    it. Checked by source because constructing a real Unsloth trainer needs a GPU
    and a model download."""
    import chowder

    backends = Path(chowder.__file__).resolve().parent / "backends"
    for name in ("unsloth_worker.py", "transformers_worker.py"):
        source = (backends / name).read_text(encoding="utf-8")
        assert '"lr_scheduler_type": spec.lr_scheduler_type' in source, (
            f"{name} does not pass lr_scheduler_type to the trainer -- "
            "a recipe asking for cosine would silently get linear"
        )

    unsloth = (backends / "unsloth_worker.py").read_text(encoding="utf-8")
    for key in ('"warmup_ratio": spec.warmup_ratio', '"warmup_steps": spec.warmup_steps'):
        assert key in unsloth, f"unsloth_worker drops {key}"
