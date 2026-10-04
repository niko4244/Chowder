"""The search-axis contract: declaration -> config patch -> backend reader.

Every field the recipe mapper emits must be a key the real backend reader
reads, and every field the consumption table claims must be emitted by the
mapper. A knob that fails either direction is inert: candidates would differ
in name and not in the run. These tests read the backend readers themselves
(``TransformersPeftRunSpec.from_config`` for peft; the router engine's
settings parser surface for router-healing), not a second hand-copied list,
so a renamed backend key fails here instead of silently desyncing the mapper.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from chowder.growth.recipe_planner import (
    ALL_RECIPE_FIELDS,
    CONSUMED_RECIPE_FIELDS,
    PLANNED_UNMAPPED_FIELDS,
    SEARCH_AXES,
    UNSAFE_SEARCH_FIELDS,
    assert_search_axes_consumed,
    classify_recipe_fields,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "chowder"


def _recipe_fields(**overrides):
    from chowder.growth.recipe_planner import TrainingRecipe

    fields: dict[str, object] = {
        "recipe_id": "contract",
        "curriculum_item_ids": ("item-1",),
        "mixture": {"TARGET": 1.0},
        "learning_rate": 1e-4,
        "scheduler": "cosine",
        "warmup_steps": 2,
        "lora_rank": 16,
        "lora_alpha": 32,
        "target_modules": ("q_proj", "v_proj"),
        "seq_len": 2048,
        "batch_size": 2,
        "gradient_accumulation": 4,
        "max_steps": 20,
        "objective": "sft",
        "replay_rate": 0.1,
        "dataset_manifest": {},
        "projected_device_gpu_hours": 0.01,
        "projected_wall_gpu_hours": 0.035,
        "notes": "contract recipe",
    }
    fields.update(overrides)
    return TrainingRecipe(**fields)


# --------------------------------------------------------------------------
# the classification: every field lands in exactly one bucket
# --------------------------------------------------------------------------


def test_every_recipe_field_is_classified_exactly_once_per_backend() -> None:
    for backend_type in sorted(CONSUMED_RECIPE_FIELDS):
        classification = classify_recipe_fields(backend_type)
        assert set(classification) == set(ALL_RECIPE_FIELDS), (
            f"{backend_type}: classification must cover every recipe field"
        )
        buckets = set(classification.values())
        assert buckets <= {"consumed", "recorded-only", "planned-unmapped", "unsafe"}
        # A field classified consumed must actually be claimed by the table,
        # and vice versa -- the two views cannot disagree.
        for field_name, bucket in classification.items():
            if bucket == "consumed":
                assert field_name in CONSUMED_RECIPE_FIELDS[backend_type]
            else:
                assert field_name not in CONSUMED_RECIPE_FIELDS[backend_type]


def test_the_unsafe_bucket_has_a_stated_reason_for_every_field() -> None:
    assert set(UNSAFE_SEARCH_FIELDS) <= set(ALL_RECIPE_FIELDS)
    for field_name, reason in UNSAFE_SEARCH_FIELDS.items():
        assert isinstance(reason, str) and len(reason) > 10, (
            f"{field_name}: an unsafe classification without a stated reason "
            "is just a vibe"
        )


def test_the_planned_unmapped_bucket_names_the_missing_wiring() -> None:
    for backend_type, fields in PLANNED_UNMAPPED_FIELDS.items():
        assert backend_type in CONSUMED_RECIPE_FIELDS
        for field_name, wiring in fields.items():
            assert field_name in ALL_RECIPE_FIELDS
            assert field_name not in CONSUMED_RECIPE_FIELDS[backend_type], (
                f"{backend_type}.{field_name}: cannot be both consumed and "
                "planned-unmapped"
            )
            assert isinstance(wiring, str) and len(wiring) > 10


# --------------------------------------------------------------------------
# mapper -> backend reader: the emitted keys are the read keys
# --------------------------------------------------------------------------


def _keys_the_peft_reader_reads() -> set[str]:
    """The backend keys ``TransformersPeftRunSpec.from_config`` reads, parsed
    from the reader's own source -- the runtime behavior lives there."""
    source = (SRC / "backends" / "transformers_peft.py").read_text(encoding="utf-8")
    keys: set[str] = set()
    for match in re.finditer(r'backend\.get\("([^"]+)"', source):
        keys.add(match.group(1))
    for match in re.finditer(r'training\.get\("([^"]+)"', source):
        keys.add(f"training.{match.group(1)}")
    for match in re.finditer(r'lora\.get\("([^"]+)"', source):
        keys.add(f"lora.{match.group(1)}")
    return keys


def _keys_the_router_reader_reads() -> set[str]:
    """The router-engine settings keys the spec builder reads, from source."""
    source = (SRC / "backends" / "router_healing.py").read_text(encoding="utf-8")
    keys: set[str] = set()
    for match in re.finditer(r'settings\.get\("([^"]+)"', source):
        keys.add(match.group(1))
    for match in re.finditer(r'settings\["([^"]+)"\]', source):
        keys.add(match.group(1))
    return keys


def test_every_key_the_peft_mapper_emits_is_read_by_the_peft_backend() -> None:
    recipe = _recipe_fields()
    patch = recipe.to_config_patch(backend_type="transformers-peft")
    backend_section = patch["backend"]
    read_keys = _keys_the_peft_reader_reads()

    emitted: set[str] = set()
    for key, value in backend_section.items():
        if isinstance(value, dict):
            for sub_key in value:
                emitted.add(f"{key}.{sub_key}")
        else:
            emitted.add(key)

    read_names = read_keys
    for emitted_key in sorted(emitted):
        # top-level backend keys are read via backend.get("<key>"); nested
        # ones via their section's get.
        if "." in emitted_key:
            section, sub = emitted_key.split(".", 1)
            assert f"{section}.{sub}" in read_names or sub in read_names, (
                f"the peft mapper emits {emitted_key} but the backend reader "
                "never reads it -- an inert knob that only changes the name "
                "of a candidate"
            )
        else:
            assert emitted_key in read_names, (
                f"the peft mapper emits backend.{emitted_key} but the backend "
                "reader never reads it"
            )


def test_every_key_the_router_mapper_emits_is_read_by_the_router_engine() -> None:
    recipe = _recipe_fields()
    patch = recipe.to_config_patch(backend_type="router-healing")
    section = patch["backend"]["router_healing"]
    read_keys = _keys_the_router_reader_reads()

    for key in sorted(section):
        assert key in read_keys, (
            f"the router-healing mapper emits router_healing.{key} but the "
            "engine's spec builder never reads it -- an inert knob"
        )


def test_every_consumed_recipe_field_is_emitted_by_the_mapper() -> None:
    """The table claims consumption; the mapper must actually emit it.

    A field marked consumed but never emitted would make the consumption
    table (and any axis built on it) lie.
    """
    recipe = _recipe_fields(resume_from_checkpoint=None)
    for backend_type in sorted(CONSUMED_RECIPE_FIELDS):
        patch = recipe.to_config_patch(backend_type=backend_type)
        backend_section = patch["backend"]
        flattened: set[str] = set()
        for key, value in backend_section.items():
            if isinstance(value, dict):
                for sub_key in value:
                    flattened.add(sub_key)
            else:
                flattened.add(key)
        for field_name in sorted(CONSUMED_RECIPE_FIELDS[backend_type]):
            # The mapper emits the field under its documented namespace name;
            # the pairs are part of the contract itself.
            emitted_name = {
                "seq_len": "max_length" if backend_type == "transformers-peft" else "seq_len",
                "lora_rank": "r" if backend_type == "transformers-peft" else "lora_rank",
                "lora_alpha": "alpha" if backend_type == "transformers-peft" else "lora_alpha",
                "gradient_accumulation": (
                    "gradient_accumulation_steps"
                    if backend_type == "transformers-peft"
                    else "gradient_accumulation"
                ),
                "scheduler": (
                    "lr_scheduler_type" if backend_type == "transformers-peft" else "scheduler"
                ),
            }.get(field_name, field_name)
            if field_name == "resume_from_checkpoint":
                continue  # emitted only when a continuation is declared
            assert emitted_name in flattened, (
                f"{backend_type}: recipe field {field_name!r} is claimed as "
                f"consumed but to_config_patch never emits {emitted_name!r}"
            )


# --------------------------------------------------------------------------
# the real runtime reader: the patch lands in the spec the engine runs
# --------------------------------------------------------------------------


def test_the_peft_spec_constructor_reads_the_recipe_patch(tmp_path) -> None:
    """The strongest offline proof of the path's last link: the real
    ``TransformersPeftRunSpec.from_config`` reads the composed config and
    every recipe field arrives as the value the run would train at."""
    pytest.importorskip("transformers")
    from chowder.backends.transformers_peft import TransformersPeftRunSpec

    recipe = _recipe_fields(
        learning_rate=3.5e-4,
        lora_rank=64,
        lora_alpha=96,
        target_modules=("q_proj", "o_proj"),
        batch_size=3,
        gradient_accumulation=7,
        warmup_steps=9,
        seq_len=1024,
        max_steps=31,
        resume_from_checkpoint=str(tmp_path / "checkpoint-12"),
    )
    patch = recipe.to_config_patch(backend_type="transformers-peft")
    config = {
        "backend": {
            **patch["backend"],
            "type": "transformers-peft",
            "base_model": str(tmp_path / "base"),
            "dataset": str(tmp_path / "dataset.txt"),
        },
    }
    spec = TransformersPeftRunSpec.from_resolved_config(
        config,
        work_dir=tmp_path,
        output_dir=tmp_path / "out",
        seed=1,
    )
    assert spec.learning_rate == 3.5e-4
    assert spec.lr_scheduler_type == recipe.scheduler
    assert spec.warmup_steps == 9
    assert spec.max_steps == 31
    assert spec.max_length == 1024
    assert spec.lora_r == 64
    assert spec.lora_alpha == 96
    assert spec.target_modules == ("q_proj", "o_proj")
    assert spec.batch_size == 3
    assert spec.gradient_accumulation_steps == 7
    assert spec.resume_from_checkpoint == str(tmp_path / "checkpoint-12")


def test_a_changed_recipe_field_changes_the_composed_spec(tmp_path) -> None:
    """Identity, not just presence: two candidates differing in one consumed
    field must compose two different run specs, and two candidates differing
    only in a recorded-only field must compose the same run spec."""
    pytest.importorskip("transformers")
    from chowder.backends.transformers_peft import TransformersPeftRunSpec

    def _spec(recipe):
        config = {
            "backend": {
                **recipe.to_config_patch(backend_type="transformers-peft")["backend"],
                "type": "transformers-peft",
                "base_model": str(tmp_path / "base"),
                "dataset": str(tmp_path / "dataset.txt"),
            },
        }
        return TransformersPeftRunSpec.from_resolved_config(
            config,
            work_dir=tmp_path,
            output_dir=tmp_path / "out",
            seed=1,
        )

    base = _recipe_fields(learning_rate=1e-4)
    other_lr = _recipe_fields(learning_rate=2e-4)
    notes_only = _recipe_fields(learning_rate=1e-4, notes="a different note")

    assert _spec(base).learning_rate == 1e-4
    assert _spec(other_lr).learning_rate == 2e-4
    # recorded-only provenance does not leak into the run's identity
    assert _spec(base) == _spec(notes_only)


# --------------------------------------------------------------------------
# the guard: unsafe axes are refused even when a backend reads them
# --------------------------------------------------------------------------


def test_an_unsafe_field_is_refused_as_a_search_axis_even_if_consumed() -> None:
    # max_steps is consumed by both backends -- and still refused, because
    # the halving schedule owns it.
    assert "max_steps" in CONSUMED_RECIPE_FIELDS["transformers-peft"]
    with pytest.raises(ValueError, match="unsafe"):
        assert_search_axes_consumed(["learning_rate", "max_steps"])


def test_search_axes_stay_inside_the_classified_consumed_set() -> None:
    # The planner's own guard must pass for the axes it actually ships.
    assert_search_axes_consumed()
    for axis in SEARCH_AXES:
        classification = classify_recipe_fields("transformers-peft")
        assert classification[axis] == "consumed"
