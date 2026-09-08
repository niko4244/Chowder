"""Tests for Phase 9 hierarchical active-parameter accounting.

Builds on the Phase 11 fixtures: the same synthetic sparse model
directory (real safetensors headers, no torch) is first accounted by
``account_parameters``, then extended hierarchically. The census-evidence
path is exercised with synthetic per-layer dReLU sparsities; every
fail-closed rule gets its own test — assuming sparsity without a census
is the specific scientific failure this module exists to prevent.
"""

from __future__ import annotations

import json
import struct

import pytest

from chowder.parameter_accounting import (
    ParameterAccountingError,
    account_parameters,
)
from chowder.sparse_accounting import (
    EFFECTIVE_DEFINITION_ID,
    HierarchicalActiveBreakdown,
    hierarchical_active_breakdown,
    write_hierarchical_accounting,
)

_DTYPE_BYTES = {"F32": 4}


def _prod(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


def _write_shard(path, tensors, fill=0):
    header = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        nbytes = _prod(shape) * _DTYPE_BYTES[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    raw = json.dumps(header).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.write(bytes([fill]) * offset)


def _write_sparse_model(root, *, layers=2, experts=4, moe_int=2, hidden=8, top_k=2):
    """The verified qwen3_5_moe fused layout (mirrors test_parameter_accounting)."""
    root.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for layer in range(layers):
        tensors[f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj.weight"] = (
            "F32",
            [experts, 2 * moe_int, hidden],
        )
        tensors[f"model.language_model.layers.{layer}.mlp.experts.down_proj.weight"] = (
            "F32",
            [experts, hidden, moe_int],
        )
        tensors[f"model.language_model.layers.{layer}.mlp.gate.weight"] = (
            "F32",
            [experts, hidden],
        )
        tensors[f"model.language_model.layers.{layer}.mlp.shared_expert.gate_proj.weight"] = (
            "F32",
            [2, hidden],
        )
        tensors[f"model.language_model.layers.{layer}.mlp.shared_expert.up_proj.weight"] = (
            "F32",
            [2, hidden],
        )
        tensors[f"model.language_model.layers.{layer}.mlp.shared_expert.down_proj.weight"] = (
            "F32",
            [hidden, 2],
        )
        tensors[f"model.language_model.layers.{layer}.mlp.shared_expert_gate.weight"] = (
            "F32",
            [1, hidden],
        )
        tensors[f"model.language_model.layers.{layer}.self_attn.q_proj.weight"] = (
            "F32",
            [hidden, hidden],
        )
        tensors[f"model.language_model.layers.{layer}.input_layernorm.weight"] = (
            "F32",
            [hidden],
        )
    tensors["model.language_model.embed_tokens.weight"] = ("F32", [16, hidden])
    tensors["lm_head.weight"] = ("F32", [16, hidden])
    _write_shard(root / "model-00001-of-00001.safetensors", tensors)
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "text_config": {
                    "hidden_size": hidden,
                    "num_hidden_layers": layers,
                    "num_experts_per_tok": top_k,
                    "num_routed_experts": experts,
                },
            }
        ),
        encoding="utf-8",
    )
    return root


@pytest.fixture()
def sparse_accounting(tmp_path):
    from chowder.parameter_accounting import account_parameters as ap

    return ap(_write_sparse_model(tmp_path / "sparse-model"))


def test_hierarchy_topk_only_composition_matches_base(sparse_accounting):
    b = hierarchical_active_breakdown(sparse_accounting)
    # routed capacity = per layer (4 experts x 2*moe_int x H + 4 x H x moe_int)
    #   = 4*4*8 + 4*8*2 = 128 + 64 = 192; two layers -> 384
    assert b.routed_expert_capacity == 384
    assert b.num_experts == 4 and b.top_k == 2
    assert b.routed_active_parameters == 384 * 2 // 4
    # always-on: attn (2x64) + norms (2x8) + embed (128) + lm_head (128)
    #   + router (2x32) = 128+16+128+128+64 = 464
    assert b.always_on_parameters == 464
    # dense/shared: shared_expert (3 x 2 x 8 = 48/layer... gate+up 2x8x2=32,
    #   down 8x2=16 -> 48/layer) + shared_expert_gate (8/layer) -> 112
    assert b.dense_shared_ffn_parameters == 112
    # base active = total - routed - router; our composition must agree
    assert (
        b.always_on_parameters + b.dense_shared_ffn_parameters + b.routed_active_parameters
        == sparse_accounting.active_parameters
    )
    # no census evidence -> intra-expert fields absent, label refuses
    assert b.census_digest is None and b.total_effective_active is None
    with pytest.raises(ParameterAccountingError):
        b.effective_label()


def test_hierarchy_with_census_evidence(sparse_accounting):
    b = hierarchical_active_breakdown(
        sparse_accounting,
        census_digest="a" * 64,
        per_layer_drelu_sparsity=[0.5, 0.75],
    )
    assert b.neuron_sparsity_measured == pytest.approx(0.625)
    assert b.neuron_sparsity_min == 0.5 and b.neuron_sparsity_max == 0.75
    expected_neuron = round(b.routed_active_parameters * 0.375)
    assert b.neuron_active_parameters == expected_neuron
    assert b.total_effective_active == (
        b.always_on_parameters + b.dense_shared_ffn_parameters + expected_neuron
    )
    assert 0 < b.total_effective_active < b.routed_active_parameters + b.always_on_parameters
    label = b.effective_label()
    assert "effective" in label and "top-2 of 4" in label and ("a" * 12) in label
    formulas = b.formula_document()
    assert formulas["definition_id"] == EFFECTIVE_DEFINITION_ID
    assert "neuron_active" in formulas and "total_effective_active" in formulas


def test_fail_closed_matrix(sparse_accounting):
    # census digest without sparsities -> refused
    with pytest.raises(ParameterAccountingError):
        hierarchical_active_breakdown(sparse_accounting, census_digest="a" * 64)
    # sparsities without digest -> refused
    with pytest.raises(ParameterAccountingError):
        hierarchical_active_breakdown(
            sparse_accounting, per_layer_drelu_sparsity=[0.5, 0.5]
        )
    # empty sparsity list -> refused
    with pytest.raises(ParameterAccountingError):
        hierarchical_active_breakdown(
            sparse_accounting,
            census_digest="a" * 64,
            per_layer_drelu_sparsity=[],
        )
    # out-of-range sparsity -> refused (sparsity of exactly 1.0 is absurd)
    with pytest.raises(ParameterAccountingError):
        hierarchical_active_breakdown(
            sparse_accounting,
            census_digest="a" * 64,
            per_layer_drelu_sparsity=[1.0, 0.5],
        )
    # truncated digest -> refused
    with pytest.raises(ParameterAccountingError):
        hierarchical_active_breakdown(
            sparse_accounting,
            census_digest="short",
            per_layer_drelu_sparsity=[0.5, 0.5],
        )


def test_dense_model_refused(tmp_path):
    from chowder.parameter_accounting import account_parameters as ap

    root = tmp_path / "dense"
    root.mkdir()
    tensors = {
        "model.language_model.layers.0.mlp.gate_proj.weight": ("F32", [4, 8]),
        "model.language_model.layers.0.mlp.up_proj.weight": ("F32", [4, 8]),
        "model.language_model.layers.0.mlp.down_proj.weight": ("F32", [8, 4]),
    }
    _write_shard(root / "model-00001-of-00001.safetensors", tensors)
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "text_config": {}}), encoding="utf-8"
    )
    with pytest.raises(ParameterAccountingError):
        hierarchical_active_breakdown(ap(root))


def test_round_trip_and_atomic_write(sparse_accounting, tmp_path):
    b = hierarchical_active_breakdown(
        sparse_accounting,
        census_digest="b" * 64,
        per_layer_drelu_sparsity=[0.4, 0.6],
    )
    out = tmp_path / "hier" / "hierarchical_accounting.json"
    written = write_hierarchical_accounting(b, out)
    assert written == str(out)
    assert not out.with_suffix(".json.tmp").exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["formulas"]["definition_id"] == EFFECTIVE_DEFINITION_ID
    restored = HierarchicalActiveBreakdown.from_dict(data)
    assert restored == b
