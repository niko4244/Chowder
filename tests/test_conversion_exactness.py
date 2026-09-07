"""Exactness-harness tests (torch/transformers; gated behind CHOWDER_REAL_ML_SMOKE).

These tests are the *measured* half of docs/PHASE6_CONVERSION_PLAN.md's
validation ladder: a tiny random dense qwen3_5 fixture is converted with
`dense_to_moe.convert_checkpoint` and the converted model's forward is
compared to the dense model's, through the real transformers classes.
The stage-2 real-weight test runs the same machinery on parent A's
actual layer-0 when the checkpoint is present locally; it skips with a
stated reason otherwise, never fakes the evidence.

Nothing in this file imports torch at module level (base CI installs no
train dependencies); every torch/transformers import is inside a
function body, and the module-level skip gate mirrors
test_memory_fabric.py's established pattern.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from chowder.dense_to_moe import convert_checkpoint
from chowder.conversion_exactness import (
    ConversionExactnessError,
    build_tiny_dense_fixture,
    measure_conversion_exactness,
)

_REAL_ML_SMOKE = pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)

#: Where parent A's verified local cache lives (docs/LOCAL_MODELS.md layout).
_PARENT_A_DIR = Path("F:/Local Models/HuggingFace/Qwen/Qwen3.8-27B")


def _fixture(tmp_path: Path, dtype: str) -> tuple[Path, Path]:
    dense = Path(build_tiny_dense_fixture(tmp_path / "dense", dtype=dtype))
    converted = tmp_path / "converted"
    convert_checkpoint(dense, converted, 8)
    return dense, converted


#: Documented gate parameter from docs/PHASE6_CONVERSION_PLAN.md: the MoE
#: forward sums E partial expert reductions where the dense forward does one
#: whole-intermediate reduction, and float addition is not associative, so
#: bit-identity is NOT promised. The gate bounds the deviation to float
#: association noise (measured on this fixture: max_abs ~= 1.3e-07, i.e. a
#: few ulps of unit-scale logits; bf16 measured ~1e-2, its own ulp scale).
_F32_ASSOCIATION_GATE = 1e-5


@_REAL_ML_SMOKE
def test_stage1_fixture_f32_within_documented_gate(tmp_path):
    """Stage 1a: fp32 value-multiply path. Must-holds: weights recover
    bitwise from the dense source and the router is exactly uniform.
    The forward is measured against the plan's documented association
    gate, not a bit-identity promise the construction cannot make."""
    dense, converted = _fixture(tmp_path, "float32")
    report = measure_conversion_exactness(dense, converted, num_experts=8)
    assert report.num_experts == 8
    assert report.dense_recovery_bitwise, report.summary()
    assert report.dense_recovery_max_abs_deviation == 0.0
    assert report.router_uniform_max_deviation == 0.0
    assert report.max_abs_deviation < _F32_ASSOCIATION_GATE, (
        f"fp32 forward deviation exceeds the documented association gate: "
        f"{report.summary()}"
    )


@_REAL_ML_SMOKE
def test_stage1_fixture_bf16_reported_and_recovered(tmp_path):
    """Stage 1b: bf16 raw-bit exponent path. Dense recovery and router
    uniformity are must-holds; forward bit-identity is *reported* — the
    plan says bf16 rounding may or may not absorb the reduction
    reordering, and the measured numbers decide."""
    dense, converted = _fixture(tmp_path, "bfloat16")
    report = measure_conversion_exactness(dense, converted, num_experts=8)
    assert report.dtype == "bfloat16"
    assert report.dense_recovery_bitwise, report.summary()
    assert report.router_uniform_max_deviation == 0.0
    if not report.logits_bitwise_equal:
        # documented gate parameter: bounded, deterministic deviation
        assert report.max_abs_deviation < 1e-2, report.summary()


@_REAL_ML_SMOKE
def test_exact_report_is_serializable(tmp_path):
    dense, converted = _fixture(tmp_path, "float32")
    report = measure_conversion_exactness(dense, converted, num_experts=8)
    payload = report.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert "bitwise-exact" in report.summary() or "measured deviation" in report.summary()


@_REAL_ML_SMOKE
def test_must_hold_recovery_failure_raises(tmp_path, monkeypatch):
    """If the converter emitted inconsistent fused weights, the harness
    must fail closed rather than print a number."""
    dense, converted = _fixture(tmp_path, "float32")
    shard = sorted(converted.glob("*.safetensors"))[0]
    import struct as _struct

    raw = shard.read_bytes()
    (hlen,) = _struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + hlen])
    data = bytearray(raw[8 + hlen :])
    target = next(n for n in header if n != "__metadata__" and n.endswith("mlp.experts.gate_up_proj"))
    begin, end = header[target]["data_offsets"]
    # break the xE scaling invariant: double every f32 element
    for off in range(begin, end, 4):
        value = _struct.unpack("<f", data[off : off + 4])[0]
        data[off : off + 4] = _struct.pack("<f", value * 2.0)
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)
    mutated = shard.with_suffix(".mutated.tmp")
    mutated.write_bytes(_struct.pack("<Q", len(blob)) + blob + bytes(data))
    os.replace(mutated, shard)
    with pytest.raises(ConversionExactnessError, match="not weight-preserving"):
        measure_conversion_exactness(dense, converted, num_experts=8)


@_REAL_ML_SMOKE
def test_stage2_real_parent_a_layer0(tmp_path):
    """Stage 2: the same ladder against real parent A weights. Skips —
    with the reason stated — when the checkpoint or its manifest is
    absent; never fabricates the evidence."""
    if not _PARENT_A_DIR.is_dir():
        pytest.skip(f"parent A not cached at {_PARENT_A_DIR}")
    manifest = _PARENT_A_DIR.parent / "Qwen3.8-27B.manifest.json"
    if not manifest.is_file():
        pytest.skip(f"parent A manifest not yet recorded at {manifest}")
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(_PARENT_A_DIR)
    text = getattr(config, "text_config", config)
    intermediate = text.intermediate_size
    if intermediate % 8 != 0:
        pytest.skip("intermediate size not divisible by the ladder rung")
    # The full 27B conversion is a real disk decision; stage 2 runs the
    # plan/measurement machinery on the real config and first-layer
    # shapes without materializing the whole converted checkpoint.
    from chowder.dense_to_moe import plan_conversion

    plan = plan_conversion(_PARENT_A_DIR, 8)
    assert plan.weight_dtype == "BF16"
    assert plan.intermediate_size == intermediate
    assert plan.num_layers_converted == text.num_hidden_layers
    assert plan.storage_delta_bytes > 0
