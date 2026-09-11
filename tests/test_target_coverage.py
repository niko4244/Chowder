"""A requested target module that adapted nothing must be loud.

The case encoded here was measured: an Unsloth run on the hybrid 9B, given an
explicit ten-name target list, adapted 128 modules instead of 200 -- every
`linear_attn` module (`in_proj_qkv`, `in_proj_z`, `out_proj`) silently skipped,
because Unsloth converts the list into a regex that missed them. The adapter was
live, the metric moved, and the gate promoted a candidate that never touched 24 of
the 32 layers' attention.

No torch: the model is a stub exposing `named_modules()`.
"""

import pytest

from chowder.target_coverage import (
    TargetCoverageError,
    adapted_modules_by_leaf,
    assert_targets_covered,
    coverage_report,
)

# The real requested list for the hybrid qwen3_5.
HYBRID = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "out_proj",
    "gate_proj", "up_proj", "down_proj",
]
# What the Transformers engine actually adapted (200) and Unsloth did (128).
FULL = {"q_proj": 8, "k_proj": 8, "v_proj": 8, "o_proj": 8,
        "in_proj_qkv": 24, "in_proj_z": 24, "out_proj": 24,
        "gate_proj": 32, "up_proj": 32, "down_proj": 32}
UNSLOTH_PARTIAL = {"q_proj": 8, "k_proj": 8, "v_proj": 8, "o_proj": 8,
                   "gate_proj": 32, "up_proj": 32, "down_proj": 32}


class _Model:
    def __init__(self, names):
        self._names = names

    def named_modules(self):
        return [(n, object()) for n in self._names]


def test_counts_adapted_modules_by_leaf_from_a_peft_model():
    model = _Model([
        "",
        "base_model.model.model.layers.0.self_attn.q_proj",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default",
        "base_model.model.model.layers.1.linear_attn.in_proj_qkv.lora_A.default",
    ])
    # q_proj is ONE adapted module even though both `lora_A` and `lora_A.default`
    # appear as module names -- counting every match would double every total.
    assert adapted_modules_by_leaf(model) == {"q_proj": 1, "in_proj_qkv": 1}


def test_works_for_a_non_default_adapter_name():
    model = _Model(["base.model.layers.0.self_attn.q_proj.lora_A.my_adapter"])
    assert adapted_modules_by_leaf(model) == {"q_proj": 1}


def test_full_coverage_passes():
    report = assert_targets_covered(HYBRID, FULL)
    assert report["unmatched"] == []
    assert report["adapted_modules_total"] == 200


def test_the_real_unsloth_partial_coverage_is_refused():
    with pytest.raises(TargetCoverageError) as excinfo:
        assert_targets_covered(HYBRID, UNSLOTH_PARTIAL)
    message = str(excinfo.value)
    # names the exact modules that were skipped
    for skipped in ("in_proj_qkv", "in_proj_z", "out_proj"):
        assert skipped in message
    assert "adapted NOTHING" in message
    assert "128" in message, "the achieved total should be visible"
    # and says how to proceed deliberately
    assert "allow_unmatched_target_modules" in message
    assert "regex" in message


def test_partial_counts_within_a_name_are_fine():
    """A name present in only some layers is legitimate -- only zero is a defect."""
    report = assert_targets_covered(["q_proj", "gate_proj"], {"q_proj": 8, "gate_proj": 32})
    assert report["unmatched"] == []


def test_an_empty_request_is_not_policed():
    """No explicit list means PEFT or a preset decided; there is no stated intent."""
    report = assert_targets_covered([], {"q_proj": 8})
    assert report["unmatched"] == []
    assert report["requested_count"] == 0


def test_the_escape_hatch_records_rather_than_hides():
    report = assert_targets_covered(HYBRID, UNSLOTH_PARTIAL, allow_unmatched=True)
    assert report["allow_unmatched"] is True
    # the gap is still reported, not swallowed
    assert report["unmatched"] == ["in_proj_qkv", "in_proj_z", "out_proj"]
    assert report["adapted_modules_total"] == 128


def test_report_surfaces_modules_adapted_without_being_requested():
    report = coverage_report(["q_proj"], {"q_proj": 8, "k_proj": 8})
    assert report["adapted_not_requested"] == ["k_proj"]
    assert report["matched_by_name"] == {"q_proj": 8}


def test_both_training_backends_enforce_coverage():
    """Two engines train; a third added later must not skip the check."""
    from pathlib import Path

    import chowder

    src = Path(chowder.__file__).resolve().parent
    controllers = ["backends/transformers_peft.py", "backends/unsloth_peft.py"]
    missing = [
        rel for rel in controllers
        if "assert_targets_covered" not in (src / rel).read_text(encoding="utf-8")
    ]
    assert not missing, f"training controller without a coverage check: {missing}"


def test_an_unreported_measurement_is_unknown_not_zero():
    """A worker that did not report what it adapted leaves coverage UNKNOWN.
    Treating that as zero would fail every run whose worker predates the report --
    the same rule adapter_guard follows for unreadable weights."""
    report = assert_targets_covered(HYBRID, None)
    assert report["status"] == "not_reported"
    assert "unmatched" not in report
    assert "unknown, not zero" in report["note"]


def test_a_measured_report_is_labelled_as_such():
    report = assert_targets_covered(HYBRID, FULL)
    assert report["status"] == "measured"
