"""Tests for Phase 11 active-parameter accounting.

Offline and deterministic: synthetic model directories whose shards are
real safetensors-format files (8-byte header length + JSON header + raw
bytes) written by a helper here — no torch, no safetensors library, no
downloads. Dense classification tests mirror parent A's real namespaces
(self_attn/linear_attn/mlp/visual/mtp); sparse tests mirror the verified
qwen3_5_moe fused layout; the failure tests pin every fail-closed rule
the module claims.
"""

import json
import struct

import pytest

from chowder.cli import build_parser
from chowder.parameter_accounting import (
    CategoryTotals,
    ParameterAccountingError,
    RouterGeometry,
    account_parameters,
    read_safetensors_header,
)

_DTYPE_BYTES = {"F32": 4, "BF16": 2, "I64": 8}


def _prod(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


def _write_shard(path, tensors, fill=0):
    """Write a real safetensors-format file: header-length prefix, JSON
    header (with data_offsets), and filler bytes of the right total size."""
    header = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        nbytes = _prod(shape) * _DTYPE_BYTES[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    raw = json.dumps(header).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.write(bytes([fill]) * offset)


def _dense_tensors(layers=2):
    tensors = {}
    for layer in range(layers):
        tensors[f"model.language_model.layers.{layer}.self_attn.q_proj.weight"] = ("F32", [8, 8])
        tensors[f"model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight"] = ("F32", [8, 8])
        tensors[f"model.language_model.layers.{layer}.mlp.gate_proj.weight"] = ("F32", [4, 8])
        tensors[f"model.language_model.layers.{layer}.mlp.up_proj.weight"] = ("F32", [4, 8])
        tensors[f"model.language_model.layers.{layer}.mlp.down_proj.weight"] = ("F32", [8, 4])
        tensors[f"model.language_model.layers.{layer}.input_layernorm.weight"] = ("F32", [8])
    tensors["model.language_model.embed_tokens.weight"] = ("F32", [16, 8])
    tensors["lm_head.weight"] = ("F32", [16, 8])
    tensors["mtp.fc.weight"] = ("F32", [8, 8])
    tensors["model.visual.patch_embed.proj.weight"] = ("F32", [4, 8, 1, 1])
    return tensors


def _write_dense_model(root, *, layers=2, with_index=True, config_overrides=None):
    root.mkdir(parents=True, exist_ok=True)
    tensors = _dense_tensors(layers)
    _write_shard(root / "model-00001-of-00001.safetensors", tensors)
    config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {"hidden_size": 8, "num_hidden_layers": layers},
    }
    config.update(config_overrides or {})
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if with_index:
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {name: "model-00001-of-00001.safetensors" for name in tensors}}),
            encoding="utf-8",
        )
    return root


def _sparse_tensors(layers=2, experts=4, moe_int=2, hidden=8):
    tensors = {}
    for layer in range(layers):
        tensors[f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj.weight"] = ("F32", [experts, 2 * moe_int, hidden])
        tensors[f"model.language_model.layers.{layer}.mlp.experts.down_proj.weight"] = ("F32", [experts, hidden, moe_int])
        tensors[f"model.language_model.layers.{layer}.mlp.gate.weight"] = ("F32", [experts, hidden])
        tensors[f"model.language_model.layers.{layer}.mlp.shared_expert.gate_proj.weight"] = ("F32", [2, hidden])
        tensors[f"model.language_model.layers.{layer}.mlp.shared_expert.up_proj.weight"] = ("F32", [2, hidden])
        tensors[f"model.language_model.layers.{layer}.mlp.shared_expert.down_proj.weight"] = ("F32", [hidden, 2])
        tensors[f"model.language_model.layers.{layer}.mlp.shared_expert_gate.weight"] = ("F32", [1, hidden])
        tensors[f"model.language_model.layers.{layer}.self_attn.q_proj.weight"] = ("F32", [hidden, hidden])
        tensors[f"model.language_model.layers.{layer}.input_layernorm.weight"] = ("F32", [hidden])
    tensors["model.language_model.embed_tokens.weight"] = ("F32", [16, hidden])
    tensors["lm_head.weight"] = ("F32", [16, hidden])
    return tensors


def _write_sparse_model(root, *, layers=2, experts=4, moe_int=2, hidden=8, config_overrides=None, tensors_override=None):
    root.mkdir(parents=True, exist_ok=True)
    tensors = tensors_override if tensors_override is not None else _sparse_tensors(layers, experts, moe_int, hidden)
    _write_shard(root / "model-00001-of-00001.safetensors", tensors)
    config = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "hidden_size": hidden,
            "num_experts": experts,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": moe_int,
        },
    }
    config["text_config"].update(config_overrides or {})
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Dense accounting (parent A's shape)
# ---------------------------------------------------------------------------


def test_dense_classification_and_total(tmp_path):
    root = _write_dense_model(tmp_path / "dense")
    accounting = account_parameters(root)

    assert accounting.model_type == "qwen3_5"
    assert accounting.num_tensors == 16
    assert accounting.is_sparse is False
    categories = accounting.categories
    assert categories["embedding"].tensors == 2
    assert categories["embedding"].parameters == 256
    assert categories["attention_and_deltanet"].tensors == 4  # self_attn + linear_attn both counted
    assert categories["attention_and_deltanet"].parameters == 256
    assert categories["dense_ffn"].tensors == 6
    assert categories["dense_ffn"].parameters == 192
    assert categories["layernorm"].parameters == 16
    assert categories["mtp"].parameters == 64
    assert categories["vision"].parameters == 32
    assert "routed_expert" not in categories

    # the census sums exactly
    assert accounting.total_parameters == 816
    assert accounting.total_bytes == 816 * 4
    assert accounting.active_parameters == accounting.total_parameters
    assert "dense model" in accounting.active_definition


def test_dense_a_label_refuses(tmp_path):
    accounting = account_parameters(_write_dense_model(tmp_path / "dense"))
    with pytest.raises(ParameterAccountingError, match="no a-label is constructible"):
        accounting.a_label()


def test_dense_report_shape(tmp_path):
    report = account_parameters(_write_dense_model(tmp_path / "dense")).format_phase11_report()
    assert "Total parameters:         816" in report
    assert "Active/token:             816" in report
    assert "none constructible" in report


# ---------------------------------------------------------------------------
# Sparse accounting (verified qwen3_5_moe fused layout)
# ---------------------------------------------------------------------------


def test_sparse_measured_geometry_and_active_split(tmp_path):
    root = _write_sparse_model(tmp_path / "sparse")
    accounting = account_parameters(root)

    assert accounting.is_sparse is True
    geometry = accounting.router_geometry
    assert geometry == RouterGeometry(num_experts=4, moe_intermediate_size=2, top_k=2)

    categories = accounting.categories
    routed_params = categories["routed_expert"].parameters
    router_params = categories["router"].parameters
    assert categories["routed_expert"].tensors == 4  # 2 layers x 2 fused tensors
    assert categories["shared_expert"].tensors == 8  # 2 layers x 4 tensors
    assert categories["shared_expert"].parameters > 0

    # active = total - routed * (1 - top_k/num_experts): the top-k routed
    # share runs every token; the router itself also runs every token.
    # (2026-09-08 fix: the old `total - routed - router` formula excluded
    # the computed expert share and the router entirely.)
    assert accounting.active_parameters == (
        accounting.total_parameters
        - routed_params
        + routed_params * 2 // 4
    )
    assert "routed active" in accounting.active_definition

    label = accounting.a_label()
    assert label.startswith("A0.0B")
    assert "top-2 of 4" in label


def test_sparse_report_includes_gate_lines(tmp_path):
    report = account_parameters(_write_sparse_model(tmp_path / "sparse")).format_phase11_report()
    assert "Routing geometry:         top-2 of 4 experts" in report
    assert "A-label:                  A0.0B" in report
    assert "routed_expert:" in report
    assert "shared_expert:" in report


def test_shared_expert_and_router_category_boundaries(tmp_path):
    accounting = account_parameters(_write_sparse_model(tmp_path / "sparse"))
    # per layer: shared expert = (2*8 + 2*8 + 8*2)*4 bytes=160B->40 params... compute exactly:
    # gate_proj (2,8)=16, up_proj 16, down_proj (8,2)=16, gate (1,8)=8 → 56 params/layer? no:
    # gate(1,8)=8 params. So 16+16+16+8 = 56 per layer, 112 total.
    assert accounting.categories["shared_expert"].parameters == 112
    # router: (4,8) per layer = 32 params x 2 layers
    assert accounting.categories["router"].parameters == 64
    # routed: gate_up (4,4,8)=128 + down (4,8,2)=64 → 192/layer, 384 total
    assert accounting.categories["routed_expert"].parameters == 384


# ---------------------------------------------------------------------------
# Fail-closed rules
# ---------------------------------------------------------------------------


def test_sparse_without_top_k_in_config_fails_closed(tmp_path):
    root = _write_sparse_model(tmp_path / "sparse", config_overrides={"num_experts_per_tok": None})
    with pytest.raises(ParameterAccountingError, match="num_experts_per_tok"):
        account_parameters(root)


def test_inconsistent_expert_shapes_fail(tmp_path):
    tensors = _sparse_tensors(layers=1)
    tensors["model.language_model.layers.0.mlp.experts.gate_up_proj.weight"] = ("F32", [4, 4, 8])
    extra = dict(tensors)
    extra["model.language_model.layers.1.mlp.experts.gate_up_proj.weight"] = ("F32", [4, 6, 8])
    root = _write_sparse_model(tmp_path / "sparse", tensors_override=extra)
    with pytest.raises(ParameterAccountingError, match="differ across layers"):
        account_parameters(root)


def test_router_dimension_mismatch_fails(tmp_path):
    tensors = _sparse_tensors(layers=1)
    tensors["model.language_model.layers.0.mlp.gate.weight"] = ("F32", [8, 8])  # says E=8, experts say 4
    root = _write_sparse_model(tmp_path / "sparse", tensors_override=tensors)
    with pytest.raises(ParameterAccountingError, match="router weight first dim"):
        account_parameters(root)


def test_config_num_experts_mismatch_fails(tmp_path):
    root = _write_sparse_model(tmp_path / "sparse", config_overrides={"num_experts": 9})
    with pytest.raises(ParameterAccountingError, match="num_experts 9"):
        account_parameters(root)


def test_config_moe_intermediate_mismatch_fails(tmp_path):
    root = _write_sparse_model(tmp_path / "sparse", config_overrides={"moe_intermediate_size": 7})
    with pytest.raises(ParameterAccountingError, match="moe_intermediate_size 7"):
        account_parameters(root)


def test_top_k_above_experts_fails(tmp_path):
    root = _write_sparse_model(tmp_path / "sparse", config_overrides={"num_experts_per_tok": 5})
    with pytest.raises(ParameterAccountingError, match="outside"):
        account_parameters(root)


def test_duplicate_tensor_across_shards_fails(tmp_path):
    root = _write_dense_model(tmp_path / "model", with_index=False)
    _write_shard(root / "model-00002-of-00002.safetensors", {"lm_head.weight": ("F32", [16, 8])})
    with pytest.raises(ParameterAccountingError, match="duplicated tensor names"):
        account_parameters(root)


def test_index_shard_mismatch_fails(tmp_path):
    root = _write_dense_model(tmp_path / "model")
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    index["weight_map"]["ghost.tensor.weight"] = "model-00001-of-00001.safetensors"
    (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ParameterAccountingError, match="index/shard mismatch"):
        account_parameters(root)


def test_unknown_dtype_fails_closed(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    with open(root / "model.safetensors", "wb") as handle:
        raw = json.dumps({"x.weight": {"dtype": "COMPLEX64", "shape": [2, 2], "data_offsets": [0, 4]}}).encode()
        raw += b" " * ((8 - len(raw) % 8) % 8)
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.write(b"\0" * 16)
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")
    with pytest.raises(ParameterAccountingError, match="unknown safetensors dtype"):
        account_parameters(root)


def test_truncated_header_fails(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", 9999) + b"{")
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")
    with pytest.raises(ParameterAccountingError, match="truncated header"):
        account_parameters(root)


def test_directory_and_content_requirements(tmp_path):
    with pytest.raises(ParameterAccountingError, match="not an existing directory"):
        account_parameters(tmp_path / "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")
    with pytest.raises(ParameterAccountingError, match="no .safetensors shards"):
        account_parameters(empty)
    noconfig = tmp_path / "noconfig"
    noconfig.mkdir()
    with pytest.raises(ParameterAccountingError, match="config.json is required"):
        account_parameters(noconfig)


def test_bad_shape_dimension_fails(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    with open(root / "model.safetensors", "wb") as handle:
        raw = json.dumps({"x.weight": {"dtype": "F32", "shape": [2, -1], "data_offsets": [0, 4]}}).encode()
        raw += b" " * ((8 - len(raw) % 8) % 8)
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.write(b"\0" * 8)
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")
    with pytest.raises(ParameterAccountingError, match="invalid tensor dimension"):
        account_parameters(root)


# ---------------------------------------------------------------------------
# Header reader and round-trips
# ---------------------------------------------------------------------------


def test_header_reader_skips_metadata_and_reports_dtype_shape(tmp_path):
    path = tmp_path / "shard.safetensors"
    _write_shard(path, {"a.weight": ("BF16", [3, 4])})
    header = read_safetensors_header(path)
    assert header == {"a.weight": {"dtype": "BF16", "shape": [3, 4]}}


def test_accounting_round_trips_through_dict(tmp_path):
    accounting = account_parameters(_write_sparse_model(tmp_path / "sparse"))
    rebuilt = type(accounting).from_dict(accounting.to_dict())
    assert rebuilt.to_dict() == accounting.to_dict()
    assert rebuilt.a_label() == accounting.a_label()


def test_category_totals_validate():
    with pytest.raises(ValueError, match="non-negative"):
        CategoryTotals(tensors=-1, parameters=0, bytes=0)
    with pytest.raises(ValueError, match="positive int"):
        RouterGeometry(num_experts=0, moe_intermediate_size=2, top_k=1)
    with pytest.raises(ValueError, match="exceed"):
        RouterGeometry(num_experts=2, moe_intermediate_size=2, top_k=3)


def test_accounting_sums_are_inconstructibly_wrong(tmp_path):
    root = _write_dense_model(tmp_path / "dense")
    accounting = account_parameters(root)
    payload = accounting.to_dict()
    payload["total_parameters"] += 1
    with pytest.raises(ParameterAccountingError, match="sum to"):
        type(accounting).from_dict(payload)


# ---------------------------------------------------------------------------
# `chowder moe account-parameters` CLI wiring
# ---------------------------------------------------------------------------


def _run_account_parameters(args):
    from chowder.cli import main

    argv = ["chowder", "moe", "account-parameters", *args]
    parser = build_parser()
    parsed = parser.parse_args(argv[1:])
    assert parsed.func is not None
    import contextlib
    import io as _io

    buffer = _io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = parsed.func(parsed)
    return code, buffer.getvalue()


def test_cli_account_parameters_dense_model(tmp_path):
    """The command accounts a dense model, writes hash-recorded evidence,
    and honestly reports the absent a-label with the module's reason."""
    root = _write_dense_model(tmp_path / "dense")
    out = tmp_path / "evidence" / "accounting.json"
    code, printed = _run_account_parameters(
        ["--model", str(root), "--output", str(out)]
    )
    assert code == 0
    summary = json.loads(printed)
    assert summary["model_type"] == "qwen3_5"
    assert summary["total_parameters"] > 0
    assert summary["num_tensors"] > 0
    assert summary["is_sparse"] is False
    assert summary["active_parameters"] == summary["total_parameters"]
    assert summary["routing_geometry"] is None
    assert summary["accounting_path"] == str(out)
    assert len(summary["accounting_sha256"]) == 64
    assert summary["a_label"] is None
    assert "no a-label is constructible" in summary["a_label_error"]

    # the evidence file exists and its sha256 matches the summary's record
    evidence = json.loads(out.read_text(encoding="utf-8"))
    assert evidence["total_parameters"] == summary["total_parameters"]
    import hashlib

    assert hashlib.sha256(out.read_bytes()).hexdigest() == summary["accounting_sha256"]


def test_cli_account_parameters_sparse_model(tmp_path):
    """A sparse model's evidence carries measured routing geometry and a
    real a-label — the Phase 11 gate working through the CLI."""
    root = _write_sparse_model(tmp_path / "sparse")
    out = tmp_path / "sparse_accounting.json"
    code, printed = _run_account_parameters(
        ["--model", str(root), "--output", str(out)]
    )
    assert code == 0
    summary = json.loads(printed)
    assert summary["is_sparse"] is True
    assert summary["routing_geometry"] == {
        "num_experts": 4,
        "moe_intermediate_size": 2,
        "top_k": 2,
    }
    assert summary["active_parameters"] < summary["total_parameters"]
    assert summary["a_label"].startswith("A0.0B")
    assert "top-2 of 4" in summary["a_label"]
    assert "a_label_error" not in summary
    assert out.is_file()


def test_cli_account_parameters_fails_closed_without_writing(tmp_path):
    """A bad directory must exit non-zero and leave no evidence file —
    a failed accounting invents nothing."""
    out = tmp_path / "never.json"
    with pytest.raises(ParameterAccountingError):
        _run_account_parameters(["--model", str(tmp_path / "missing"), "--output", str(out)])
    assert not out.exists()


def test_cli_account_parameters_requires_arguments():
    """The subcommand exists under `moe` with the documented arguments."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["moe", "account-parameters"])
    parsed = parser.parse_args(
        ["moe", "account-parameters", "--model", "m", "--output", "o.json"]
    )
    assert parsed.model == "m" and parsed.output == "o.json"
