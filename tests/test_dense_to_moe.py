"""Tests for the stdlib partition-converter (dense_to_moe).

Offline and deterministic: synthetic dense qwen3_5-style directories
built from real safetensors-format shards. The exact-scaling edge cases
are pinned bit-for-bit; the fusion layout is checked against the
verified qwen3_5_moe fused shapes; every plan-time refusal is exercised.
"""

import json
import math
import struct

import pytest

from chowder.dense_to_moe import (
    DenseToMoeError,
    PartitionScheme,
    _scale_bytes_exact,
    convert_checkpoint,
    plan_conversion,
)
from chowder.local_model_manifest import build_local_model_manifest, verify_local_model_manifest
from chowder.parameter_accounting import read_safetensors_header

H = 8  # hidden
I = 32  # intermediate (divisible by 8/16/32)


def _bf16_bytes(value: float) -> bytes:
    """Truncate an f32 to bf16: bf16 is the top half of the f32 word."""
    f32 = struct.pack("<f", value)
    return f32[2:4]


def _bf16_value(raw: bytes) -> float:
    return struct.unpack("<f", raw + b"\x00\x00")[0]


def _write_shard(path, tensors):
    widths = {"BF16": 2, "F32": 4, "F8_E4M3": 1}
    header = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        count = math.prod(shape)
        nbytes = count * widths[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    raw = json.dumps(header).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.write(b"\x00" * offset)


def _dense_dir(root, *, layers=2, dtype="BF16", single_shard=True):
    root.mkdir(parents=True, exist_ok=True)
    width = 2 if dtype == "BF16" else 4
    tensors = {}
    for layer in range(layers):
        tensors[f"model.language_model.layers.{layer}.mlp.gate_proj.weight"] = (dtype, [I, H])
        tensors[f"model.language_model.layers.{layer}.mlp.up_proj.weight"] = (dtype, [I, H])
        tensors[f"model.language_model.layers.{layer}.mlp.down_proj.weight"] = (dtype, [H, I])
        tensors[f"model.language_model.layers.{layer}.self_attn.q_proj.weight"] = (dtype, [H, H])
        tensors[f"model.language_model.layers.{layer}.input_layernorm.weight"] = (dtype, [H])
    tensors["model.language_model.embed_tokens.weight"] = (dtype, [16, H])
    tensors["lm_head.weight"] = (dtype, [16, H])
    tensors["mtp.fc.weight"] = (dtype, [H, H])
    tensors["model.visual.patch_embed.proj.weight"] = (dtype, [4, H, 1, 1])
    if single_shard:
        _write_shard(root / "model-00001-of-00001.safetensors", tensors)
    else:
        # split: layer 0 mlp into shard 1, layer 1 mlp into shard 2
        part1, part2 = {}, {}
        for name, spec in tensors.items():
            (part1 if "layers.0." in name or "layers." not in name else part2)[name] = spec
        _write_shard(root / "model-00001-of-00002.safetensors", part1)
        _write_shard(root / "model-00002-of-00002.safetensors", part2)
    config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {"hidden_size": H, "intermediate_size": I, "num_hidden_layers": layers},
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Exact scaling edge cases (bf16 raw-bit path)
# ---------------------------------------------------------------------------


def test_bf16_scale_one_is_bit_exact():
    raw = _bf16_bytes(1.0) + _bf16_bytes(-2.5) + _bf16_bytes(0.0) + _bf16_bytes(-0.0)
    scaled = _scale_bytes_exact(raw, "BF16", 8)
    for before, after in zip([raw[i:i+2] for i in range(0, 8, 2)], [scaled[i:i+2] for i in range(0, 8, 2)]):
        b = struct.unpack("<f", b"\x00\x00" + before)[0]
        a = struct.unpack("<f", b"\x00\x00" + after)[0]
        assert a == b * 8 or (math.isnan(b) and math.isnan(a))


def test_bf16_scale_promotes_subnormals_like_a_real_multiply():
    # smallest bf16 subnormal is 2^-133; x8 promotes it up the subnormal range
    tiny = struct.pack("<f", 2.0 ** -133)[2:4]
    scaled = _scale_bytes_exact(tiny, "BF16", 8)
    a = struct.unpack("<f", b"\x00\x00" + scaled)[0]
    assert a == 2.0 ** -130


def test_bf16_scale_overflow_becomes_signed_infinity():
    # bf16 max finite is 0x7F7F = (2 - 2**-7) * 2**127; x8 overflows the exponent
    for big in (b"\x7f\x7f", b"\x7f\xff"):
        scaled = _scale_bytes_exact(big, "BF16", 8)
        word = scaled[0] | (scaled[1] << 8)
        assert (word >> 7) & 0xFF == 0xFF          # exponent all ones -> infinity
        assert scaled[0] & 0x7F == 0               # zero mantissa -> inf, not nan
        assert scaled[1] & 0x80 == big[1] & 0x80   # sign preserved


def test_bf16_scale_passes_inf_and_nan_verbatim():
    inf = b"\x80\x7f"      # +inf: exponent all ones, zero mantissa
    neg_inf = b"\x80\xff"  # -inf
    qnan = b"\xc0\x7f"     # quiet NaN
    snan = b"\xff\xff"     # negative sNaN: exponent all ones, msb of mantissa clear
    raw = inf + neg_inf + qnan + snan
    assert _scale_bytes_exact(raw, "BF16", 8) == raw


def test_f32_scale_is_exact_value_multiply():
    import array

    values = array.array("f", [0.1, -1.5, 3.25])
    scaled = _scale_bytes_exact(values.tobytes(), "F32", 16)
    out = array.array("f")
    out.frombytes(scaled)
    assert list(out) == [
        struct.unpack("<f", struct.pack("<f", v * 16))[0]
        for v in (0.1, -1.5, 3.25)
    ]


def test_non_float_dtype_refused():
    with pytest.raises(DenseToMoeError, match="fail closed"):
        _scale_bytes_exact(b"\x00\x00\x00\x00", "I32", 8)


# ---------------------------------------------------------------------------
# Partition scheme
# ---------------------------------------------------------------------------


def test_scheme_requires_power_of_two():
    with pytest.raises(DenseToMoeError, match="power of two"):
        PartitionScheme.contiguous_scheme(12, 96)
    with pytest.raises(DenseToMoeError, match="power of two"):
        PartitionScheme.contiguous_scheme(6, 96)


def test_scheme_requires_divisibility():
    with pytest.raises(DenseToMoeError, match="divisible"):
        PartitionScheme.contiguous_scheme(8, 100)


def test_contiguous_scheme_digest_is_stable():
    a = PartitionScheme.contiguous_scheme(8, 32)
    b = PartitionScheme.contiguous_scheme(8, 32)
    assert a.digest() == b.digest()
    assert a.to_dict()["assignment_sha256"]


# ---------------------------------------------------------------------------
# Planning refusals (profile-only mode's whole truth)
# ---------------------------------------------------------------------------


def test_plan_refuses_wrong_model_type(tmp_path):
    root = _dense_dir(tmp_path / "m")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config["model_type"] = "qwen3_moe"
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(DenseToMoeError, match="model_type"):
        plan_conversion(root, 8)


def test_plan_refuses_non_divisible(tmp_path):
    root = _dense_dir(tmp_path / "m")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config["text_config"]["intermediate_size"] = 30
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(DenseToMoeError, match="not divisible"):
        plan_conversion(root, 8)


def test_plan_refuses_non_power_of_two_experts(tmp_path):
    root = _dense_dir(tmp_path / "m")
    with pytest.raises(DenseToMoeError, match="power of two"):
        plan_conversion(root, 12)


def test_straddling_mlp_shards_convert_identically(tmp_path):
    """A layer's gate/up/down may live in different shards (measured on
    parent A: 63/64 layers co-locate, layer 15 does not). The converted
    output must be byte-identical regardless of source shard layout."""
    colocated = _dense_dir(tmp_path / "colocated", layers=2, single_shard=True)
    straddling = _dense_dir(tmp_path / "straddling", layers=2, single_shard=False)
    plan_a = plan_conversion(colocated, 8)
    plan_b = plan_conversion(straddling, 8)
    assert plan_a.storage_delta_bytes == plan_b.storage_delta_bytes
    assert plan_a.scheme_digest == plan_b.scheme_digest
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    convert_checkpoint(colocated, out_a, 8)
    convert_checkpoint(straddling, out_b, 8)

    def tensor_map(out_dir):
        tensors = {}
        for shard in sorted(out_dir.glob("*.safetensors")):
            raw = shard.read_bytes()
            (hlen,) = struct.unpack("<Q", raw[:8])
            header = json.loads(raw[8 : 8 + hlen])
            header.pop("__metadata__", None)
            data = 8 + hlen
            for name, entry in header.items():
                begin, end = entry["data_offsets"]
                tensors[name] = raw[data + begin : data + end]
        return tensors

    ta, tb = tensor_map(out_a), tensor_map(out_b)
    assert set(ta) == set(tb)
    for name in sorted(ta):
        assert ta[name] == tb[name], name
    # and the straddling conversion really did span two source shards
    assert (straddling / "model-00002-of-00002.safetensors").is_file()
    assert (out_b / "model-00002-of-00002.safetensors").is_file()


def test_plan_refuses_inexact_mlp_dtype(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    _write_shard(root / "model.safetensors", {
        "model.language_model.layers.0.mlp.gate_proj.weight": ("F8_E4M3", [I, H]),
        "model.language_model.layers.0.mlp.up_proj.weight": ("F8_E4M3", [I, H]),
        "model.language_model.layers.0.mlp.down_proj.weight": ("F8_E4M3", [H, I]),
    })
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "text_config": {"hidden_size": H, "intermediate_size": I, "num_hidden_layers": 1}}),
        encoding="utf-8",
    )
    with pytest.raises(DenseToMoeError, match="scaled exactly"):
        plan_conversion(root, 8)


def test_plan_measures_storage_delta_correctly(tmp_path):
    root = _dense_dir(tmp_path / "m", layers=2)
    plan = plan_conversion(root, 8)
    # routed 3T -> 3T (disjoint slices); delta is router + shared expert + gate
    expected_per_layer = (8 + 3 * 512 + 1) * H * 2
    assert plan.storage_delta_bytes == expected_per_layer * 2
    assert plan.estimated_output_bytes == plan.source_bytes + plan.storage_delta_bytes
    assert plan.weight_dtype == "BF16"
    assert plan.num_layers_converted == 2
    assert plan.routed_source_tensors == 6
    assert plan.passthrough_tensors == 8  # attn x2, norm x2, embed, lm_head, mtp, visual


# ---------------------------------------------------------------------------
# Conversion output invariants
# ---------------------------------------------------------------------------


@pytest.fixture()
def converted(tmp_path):
    source = _dense_dir(tmp_path / "src", layers=1)
    out = tmp_path / "out"
    provenance = convert_checkpoint(source, out, 8)
    return source, out, provenance


def test_converted_shapes_and_zeros(tmp_path):
    source = _dense_dir(tmp_path / "src", layers=1)
    out = tmp_path / "out"
    convert_checkpoint(source, out, 8)
    header = read_safetensors_header(out / "model-00001-of-00001.safetensors")

    fused_gu = header["model.language_model.layers.0.mlp.experts.gate_up_proj"]
    assert fused_gu["shape"] == [8, 2 * 4, H]  # moe_int = 32/8 = 4
    fused_down = header["model.language_model.layers.0.mlp.experts.down_proj"]
    assert fused_down["shape"] == [8, H, 4]
    for extra in (
        "mlp.gate.weight",
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight",
        "mlp.shared_expert_gate.weight",
    ):
        assert f"model.language_model.layers.0.{extra}" in header

    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["model_type"] == "qwen3_5_moe"
    text = config["text_config"]
    assert text["num_experts"] == 8 and text["num_experts_per_tok"] == 8
    assert text["moe_intermediate_size"] == 4
    assert config["architectures"] == ["Qwen3_5MoeForConditionalGeneration"]


def test_fusion_layout_is_exact_data_movement(tmp_path):
    """gate/up rows: expert e owns verbatim dense channel rows; down: strided
    columns of the xE-scaled tensor."""
    source = _dense_dir(tmp_path / "src", layers=1)
    out = tmp_path / "out"
    convert_checkpoint(source, out, 8)

    def _raw_header(path):
        raw = path.read_bytes()
        (hlen,) = struct.unpack("<Q", raw[:8])
        header = json.loads(raw[8 : 8 + hlen])
        header.pop("__metadata__", None)
        return header

    src_header = _raw_header(source / "model-00001-of-00001.safetensors")
    out_header = _raw_header(out / "model-00001-of-00001.safetensors")

    def tensor_bytes(path, header, name):
        raw = path.read_bytes()
        (hlen,) = struct.unpack("<Q", raw[:8])
        entry = header[name]
        begin, end = entry["data_offsets"]
        return raw[8 + hlen + begin : 8 + hlen + end]



    gate = tensor_bytes(source / "model-00001-of-00001.safetensors", src_header,
                        "model.language_model.layers.0.mlp.gate_proj.weight")
    fused_gu = tensor_bytes(out / "model-00001-of-00001.safetensors", out_header,
                            "model.language_model.layers.0.mlp.experts.gate_up_proj")
    moe_int = 4
    row_bytes = H * 2
    # expert 3's gate rows are the dense gate rows 12..15, VERBATIM
    # (silu is not positively homogeneous: the gate factor is never scaled)
    for i, channel in enumerate(range(12, 16)):
        dense_row = gate[channel * row_bytes : (channel + 1) * row_bytes]
        fused_offset = (3 * 2 * moe_int + i) * row_bytes
        assert fused_gu[fused_offset : fused_offset + row_bytes] == dense_row
    # and the up part follows after moe_int gate rows, also verbatim
    up = tensor_bytes(source / "model-00001-of-00001.safetensors", src_header,
                      "model.language_model.layers.0.mlp.up_proj.weight")
    fused_up_row0 = 3 * 2 * moe_int + moe_int
    assert fused_gu[fused_up_row0 * row_bytes : (fused_up_row0 + 1) * row_bytes] == (
        up[12 * row_bytes : 13 * row_bytes]
    )

    down = tensor_bytes(source / "model-00001-of-00001.safetensors", src_header,
                        "model.language_model.layers.0.mlp.down_proj.weight")
    fused_down = tensor_bytes(out / "model-00001-of-00001.safetensors", out_header,
                              "model.language_model.layers.0.mlp.experts.down_proj")
    # down (hidden, intermediate): fused(e, r, j) == scaled_down(r, e*4 + j)
    scaled_down = _scale_bytes_exact(down, "BF16", 8)
    in_row_bytes = I * 2
    out_row_bytes = 4 * 2
    for expert in (0, 5, 7):
        col_base = expert * 4 * 2
        for r in (0, 3, 7):
            start = r * in_row_bytes + col_base
            fused_offset = (expert * H + r) * out_row_bytes
            assert fused_down[fused_offset : fused_offset + out_row_bytes] == scaled_down[start : start + out_row_bytes]


def test_passthrough_tensors_are_byte_identical(tmp_path):
    source = _dense_dir(tmp_path / "src", layers=1)
    out = tmp_path / "out"
    convert_checkpoint(source, out, 8)
    for name in ("model.language_model.embed_tokens.weight", "lm_head.weight", "mtp.fc.weight"):
        src_header = read_safetensors_header(source / "model-00001-of-00001.safetensors")
        out_header = read_safetensors_header(out / "model-00001-of-00001.safetensors")
        assert src_header[name] == out_header[name]
        assert (source / "model-00001-of-00001.safetensors").read_bytes().count(name.encode()) >= 0
    # tokenizer companion file copied
    assert (out / "tokenizer_config.json").read_text(encoding="utf-8") == "{}"


def test_index_regenerated_and_complete(tmp_path):
    source = _dense_dir(tmp_path / "src", layers=1)
    out = tmp_path / "out"
    convert_checkpoint(source, out, 8)
    index = json.loads((out / "model.safetensors.index.json").read_text(encoding="utf-8"))
    header = read_safetensors_header(out / "model-00001-of-00001.safetensors")
    assert set(index["weight_map"]) == set(header)
    # dense mlp tensors are gone from the converted checkpoint
    assert not any("mlp.gate_proj" in name for name in header)


def test_provenance_and_output_manifest(tmp_path):
    source = _dense_dir(tmp_path / "src", layers=1)
    out = tmp_path / "out"
    provenance = convert_checkpoint(source, out, 8)
    assert provenance["num_experts"] == 8
    assert provenance["scheme"]["num_experts"] == 8
    assert "top_k == num_experts" in provenance["exactness_contract"]
    assert (out / "conversion.provenance.json").is_file()

    manifest = json.loads((out / "conversion.manifest.json").read_text(encoding="utf-8"))
    rebuilt = build_local_model_manifest(out, mode="full")
    # manifest json records the manifest taken at conversion time
    assert manifest["manifest_sha256"]
    verification = verify_local_model_manifest(rebuilt, out, rehash_weights=False)
    assert verification.clean is True


def test_refuses_nonempty_output_dir(tmp_path):
    source = _dense_dir(tmp_path / "src", layers=1)
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale.txt").write_text("x", encoding="utf-8")
    with pytest.raises(DenseToMoeError, match="not empty"):
        convert_checkpoint(source, out, 8)

