"""Tests for `dense_to_moe.set_experts_per_token`.

The property under protection: lowering `top_k` is an explicit,
provenance-recorded act that breaks the conversion's output-exactness, and
it can never silently exceed the expert count. No checkpoint or GPU needed
-- a config.json plus a provenance sidecar is the whole contract.
"""
from __future__ import annotations

import json

import pytest

from chowder.dense_to_moe import DenseToMoeError, set_experts_per_token


def _write_checkpoint(tmp_path, *, num_experts: int = 16, top_k: int | None = None, provenance: bool = True):
    top_k = num_experts if top_k is None else top_k
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "num_experts": num_experts,
                    "num_experts_per_tok": top_k,
                    "moe_intermediate_size": 17408 // num_experts,
                },
            }
        ),
        encoding="utf-8",
    )
    if provenance:
        (tmp_path / "conversion.provenance.json").write_text(
            json.dumps({"num_experts": num_experts, "scheme_digest": "c8ad351a"}), encoding="utf-8"
        )
    return tmp_path


def _top_k_on_disk(path) -> int:
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    return config["text_config"]["num_experts_per_tok"]


def test_lowering_top_k_updates_config_and_reports_the_rescale(tmp_path):
    root = _write_checkpoint(tmp_path, num_experts=16)
    change = set_experts_per_token(root, 3)
    assert _top_k_on_disk(root) == 3
    assert change["previous_num_experts_per_tok"] == 16
    assert change["exact_at_init"] is False
    assert change["active_expert_fraction"] == pytest.approx(3 / 16)
    # the (E/k) factor the router's unconditional renormalise produces
    assert change["routed_rescale_factor"] == pytest.approx(16 / 3)


def test_top_k_equal_to_num_experts_is_still_exact(tmp_path):
    root = _write_checkpoint(tmp_path, num_experts=16, top_k=3)
    change = set_experts_per_token(root, 16)
    assert change["exact_at_init"] is True
    assert change["routed_rescale_factor"] == pytest.approx(1.0)


def test_breaking_exactness_is_recorded_in_provenance(tmp_path):
    root = _write_checkpoint(tmp_path, num_experts=16)
    set_experts_per_token(root, 4)
    provenance = json.loads((root / "conversion.provenance.json").read_text(encoding="utf-8"))
    assert provenance["num_experts_per_tok"] == 4
    assert len(provenance["exactness_broken_by"]) == 1
    assert provenance["exactness_broken_by"][0]["num_experts_per_tok"] == 4

    # a second reduction appends rather than overwriting the history
    set_experts_per_token(root, 2)
    provenance = json.loads((root / "conversion.provenance.json").read_text(encoding="utf-8"))
    assert len(provenance["exactness_broken_by"]) == 2


def test_restoring_exactness_does_not_append_a_break_record(tmp_path):
    root = _write_checkpoint(tmp_path, num_experts=16)
    set_experts_per_token(root, 4)
    set_experts_per_token(root, 16)
    provenance = json.loads((root / "conversion.provenance.json").read_text(encoding="utf-8"))
    assert provenance["num_experts_per_tok"] == 16
    assert len(provenance["exactness_broken_by"]) == 1  # only the reduction


def test_top_k_above_num_experts_is_refused(tmp_path):
    root = _write_checkpoint(tmp_path, num_experts=16)
    with pytest.raises(DenseToMoeError):
        set_experts_per_token(root, 17)
    assert _top_k_on_disk(root) == 16  # unchanged


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "3"])
def test_non_positive_or_non_integer_top_k_is_refused(tmp_path, bad):
    root = _write_checkpoint(tmp_path, num_experts=16)
    with pytest.raises(DenseToMoeError):
        set_experts_per_token(root, bad)
    assert _top_k_on_disk(root) == 16


def test_non_moe_checkpoint_is_refused(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "text_config": {"intermediate_size": 17408}}),
        encoding="utf-8",
    )
    with pytest.raises(DenseToMoeError):
        set_experts_per_token(tmp_path, 4)


def test_missing_config_is_refused(tmp_path):
    with pytest.raises(DenseToMoeError):
        set_experts_per_token(tmp_path, 4)


def test_works_without_a_provenance_sidecar(tmp_path):
    root = _write_checkpoint(tmp_path, num_experts=8, provenance=False)
    change = set_experts_per_token(root, 2)
    assert _top_k_on_disk(root) == 2
    assert change["exact_at_init"] is False
    assert not (root / "conversion.provenance.json").exists()
