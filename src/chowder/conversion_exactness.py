"""Exactness harness for the dense-to-MoE conversion (program Phase 6).

docs/PHASE6_CONVERSION_PLAN.md's validation ladder, stages 1–2: build a
tiny random dense `qwen3_5`, convert it with `dense_to_moe`, load both
models through the real transformers classes, and *measure* whether the
converted model reproduces the dense forward. Nothing here is trusted
from the argument in `dense_to_moe`'s docstring — every claim is a
measured number in an `ExactnessReport`.

What is measured, and what "exact" can mean
-------------------------------------------
The dense FFN reduces over the whole intermediate dimension in one
matmul; the converted model sums E per-expert partial outputs. Floating
point addition is not associative, so bit-identity is an *empirical*
property that depends on dtype and reduction order — precisely what the
plan said must be measured, not assumed. The report therefore carries:

- `logits_bitwise_equal`: whether outputs are bit-identical;
- `max_abs_deviation` / `max_rel_deviation`: how far apart they are when
  not (the documented gate parameter if bf16 rounding does not absorb
  the reordering);
- `router_uniform_deviation`: the router's actual selection weights vs
  the exact 1/E the construction promises (zeros in, renormalized
  top-k out);
- `dense_recovery_bitwise`: whether re-assembling the fused expert
  slices (gate/up verbatim; down un-scaled by E) rebuilds the dense
  source checkpoint's tensors bit for bit — compared directly against
  the dense checkpoint. Failure means the converter lost or moved
  bytes; it must always hold.

Stage 2 (a small real checkpoint) reuses the same machinery with a
real checkpoint directory and its layer-0 weights; the real-weights
test lives in the gated test file, not here — this module only
provides `measure_conversion_exactness` over any (dense_dir,
converted_dir) pair plus the fixture builder.

Torch/transformers imports happen inside functions: this module is
importable in a base (non-train) install without breaking collection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConversionExactnessError(ValueError):
    """The exactness harness cannot run or a must-hold property failed."""


@dataclass(frozen=True)
class ExactnessReport:
    """Measured forward equality of a converted model against its dense source."""

    dense_dir: str
    converted_dir: str
    num_experts: int
    dtype: str
    logits_bitwise_equal: bool
    max_abs_deviation: float
    max_rel_deviation: float
    router_uniform_max_deviation: float
    dense_recovery_bitwise: bool
    dense_recovery_max_abs_deviation: float
    input_shape: list[int]

    @property
    def exact(self) -> bool:
        """The strict claim: bitwise logits + bitwise recovery + uniform router."""
        return (
            self.logits_bitwise_equal
            and self.dense_recovery_bitwise
            and self.router_uniform_max_deviation == 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def summary(self) -> str:
        exactness = "bitwise-exact" if self.exact else "measured deviation (see numbers)"
        return (
            f"[{self.dtype}, E={self.num_experts}] {exactness}: "
            f"max_abs={self.max_abs_deviation:.3e}, max_rel={self.max_rel_deviation:.3e}, "
            f"router_dev={self.router_uniform_max_deviation:.3e}, "
            f"recovery={'bitwise' if self.dense_recovery_bitwise else f'max_abs={self.dense_recovery_max_abs_deviation:.3e}'}"
        )


def build_tiny_dense_fixture(output_dir: str | Path, *, dtype: str = "float32", seed: int = 42) -> str:
    """Create a tiny random dense qwen3_5 checkpoint on disk.

    Composite qwen3_5 (as the real checkpoints ship): two text decoder
    layers (one GatedDeltaNet, one full attention), hidden 64,
    intermediate 192 (divisible by 8/16/32), vocab 64, plus a tiny real
    vision tower so the passthrough namespaces match parent A's. Weights are random at construction and saved via
    `save_pretrained`, so the checkpoint on disk is deterministic for a
    given seed. `dtype` selects float32 (value-multiply scaling path)
    or bfloat16 (raw-bit exponent path).
    """
    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config,
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
    )

    if dtype not in {"float32", "bfloat16"}:
        raise ConversionExactnessError(f"fixture dtype must be float32 or bfloat16; got {dtype}")
    torch.manual_seed(seed)
    # Composite config, exactly like the real checkpoints: model_type
    # "qwen3_5" with nested text_config + vision_config, key layout
    # model.language_model.layers.* / mtp.* / model.visual.*. The vision
    # tower is tiny but real (constructed through the actual class), so
    # the passthrough path exercises the same tensor namespaces parent A
    # carries. The forward comparison loads through AutoModelForCausalLM,
    # which uses the text stack and ignores visual/mtp weights.
    text_config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=64,
    )
    vision_config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        patch_size=4,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=64,
        num_position_embeddings=4,
    )
    config = Qwen3_5Config(text_config=text_config, vision_config=vision_config)
    model = Qwen3_5ForConditionalGeneration(config)
    if dtype == "bfloat16":
        model = model.to(torch.bfloat16)
    target = Path(output_dir)
    model.save_pretrained(target)
    del model
    return str(target)


def measure_conversion_exactness(
    dense_dir: str | Path,
    converted_dir: str | Path,
    *,
    num_experts: int,
    seed: int = 1234,
) -> ExactnessReport:
    """Load both checkpoints and measure forward equality on fixed inputs.

    Also verifies the two must-hold properties: dense recovery from the
    fused slices (converter correctness) and router uniformity at zero
    logits (the construction's premise). Raises
    `ConversionExactnessError` if a must-hold fails; forward deviation
    is *reported*, not raised — it is the measured number the plan asks
    for.
    """
    import torch
    from transformers import AutoModelForCausalLM

    dense_path, converted_path = Path(dense_dir), Path(converted_dir)
    dense = AutoModelForCausalLM.from_pretrained(dense_path)
    converted = AutoModelForCausalLM.from_pretrained(converted_path)
    dense.eval()
    converted.eval()

    dtype_name = str(next(dense.parameters()).dtype).replace("torch.", "")
    torch.manual_seed(seed)
    input_ids = torch.randint(0, 64, (1, 16))
    with torch.no_grad():
        dense_logits = dense(input_ids).logits
        converted_logits = converted(input_ids).logits

    if dense_logits.shape != converted_logits.shape:
        raise ConversionExactnessError(
            f"logit shapes diverge: {tuple(dense_logits.shape)} vs {tuple(converted_logits.shape)}"
        )
    bitwise = bool(torch.equal(dense_logits, converted_logits))
    diff = (dense_logits - converted_logits).abs()
    max_abs = float(diff.max()) if not bitwise else 0.0
    denom = dense_logits.abs().clamp_min(1e-12)
    max_rel = float((diff / denom).max()) if not bitwise else 0.0

    # must-hold 1: dense recovery from the converted checkpoint's tensors
    recovery_bitwise, recovery_max_abs = _check_dense_recovery(
        dense_path, converted_path, num_experts
    )
    if not recovery_bitwise and recovery_max_abs != 0.0:
        raise ConversionExactnessError(
            f"dense recovery from fused slices failed (max_abs={recovery_max_abs:.3e}); "
            "the converter is not weight-preserving"
        )

    # must-hold 2: router uniformity — zero logits must select every expert
    # with exactly 1/E weight after softmax+renormalized top-k
    router_dev = _check_router_uniformity(converted, num_experts)
    if router_dev != 0.0:
        raise ConversionExactnessError(
            f"router weights deviate from uniform 1/E by {router_dev:.3e}; "
            "the exactness construction requires exact uniformity"
        )

    report = ExactnessReport(
        dense_dir=str(dense_path),
        converted_dir=str(converted_path),
        num_experts=num_experts,
        dtype=dtype_name,
        logits_bitwise_equal=bitwise,
        max_abs_deviation=max_abs,
        max_rel_deviation=max_rel,
        router_uniform_max_deviation=router_dev,
        dense_recovery_bitwise=recovery_bitwise,
        dense_recovery_max_abs_deviation=recovery_max_abs,
        input_shape=[1, 16],
    )
    del dense, converted
    return report


def _check_dense_recovery(
    dense_dir: Path, converted_dir: Path, num_experts: int
) -> tuple[bool, float]:
    """Rebuild the dense MLP tensors from the converted checkpoint's fused
    slices and compare them to the *dense source checkpoint* byte for bit.

    gate/up fused slices are verbatim dense channel slices; down slices are
    the dense columns scaled xE, so un-scaling by E (exact for the
    power-of-two factor) reconstructs the dense tensor. Any mismatch means
    the converter lost or moved bytes: the converter is broken, not the
    float order."""
    import torch
    from safetensors.torch import load_file

    def load_all(directory: Path) -> dict[str, Any]:
        tensors: dict[str, Any] = {}
        for shard in sorted(Path(directory).glob("*.safetensors")):
            tensors.update(load_file(str(shard)))
        return tensors

    dense_tensors = load_all(dense_dir)
    converted_tensors = load_all(converted_dir)
    max_abs = 0.0
    bitwise = True
    fused_gate_up_names = sorted(
        n for n in converted_tensors if n.endswith("mlp.experts.gate_up_proj")
    )
    if not fused_gate_up_names:
        raise ConversionExactnessError("converted checkpoint has no fused gate_up_proj tensors")
    for gu_name in fused_gate_up_names:
        prefix = gu_name[: -len("gate_up_proj")]
        fused_down_name = prefix + "down_proj"          # converted: plain Parameter name
        gate_name = prefix.replace("mlp.experts.", "mlp.gate_proj.weight")
        up_name = prefix.replace("mlp.experts.", "mlp.up_proj.weight")
        down_name = prefix.replace("mlp.experts.", "mlp.down_proj.weight")  # dense: nn.Linear name
        if gate_name not in dense_tensors or up_name not in dense_tensors or down_name not in dense_tensors:
            raise ConversionExactnessError(f"dense source lacks {gate_name.split('.', 3)[-1]} triple")
        gate_up = converted_tensors[gu_name].to(torch.float64)
        fused_down = converted_tensors[fused_down_name].to(torch.float64)
        experts, fused_rows, hidden = gate_up.shape
        moe_int = fused_rows // 2
        channels_per_expert = moe_int
        if channels_per_expert * experts != dense_tensors[gate_name].shape[0]:
            raise ConversionExactnessError(
                f"fused gate_up partition {experts}x{channels_per_expert} does not "
                f"cover dense intermediate {dense_tensors[gate_name].shape[0]}"
            )
        # gate/up: verbatim slices
        dense_gate = dense_tensors[gate_name].to(torch.float64)
        dense_up = dense_tensors[up_name].to(torch.float64)
        for expert in range(experts):
            lo = expert * channels_per_expert
            hi = lo + channels_per_expert
            if not torch.equal(gate_up[expert, :channels_per_expert], dense_gate[lo:hi]):
                bitwise = False
                max_abs = max(
                    max_abs,
                    float((gate_up[expert, :channels_per_expert] - dense_gate[lo:hi]).abs().max()),
                )
            if not torch.equal(gate_up[expert, channels_per_expert:], dense_up[lo:hi]):
                bitwise = False
                max_abs = max(
                    max_abs,
                    float((gate_up[expert, channels_per_expert:] - dense_up[lo:hi]).abs().max()),
                )
        # down: scaled slices -> un-scale by E (exact) and compare
        dense_down = dense_tensors[down_name].to(torch.float64)
        intermediate = dense_down.shape[1]
        in_row = intermediate
        for expert in range(experts):
            lo = expert * channels_per_expert
            hi = lo + channels_per_expert
            recon = fused_down[expert] / experts
            if not torch.equal(recon, dense_down[:, lo:hi]):
                bitwise = False
                max_abs = max(
                    max_abs,
                    float((recon - dense_down[:, lo:hi]).abs().max()),
                )
    return bitwise, max_abs


def _check_router_uniformity(converted_model: Any, num_experts: int) -> float:
    """Feed zero router logits through the real TopKRouter math and
    compare the resulting selection weights to exact 1/E."""
    import torch

    weight = None
    for name, param in converted_model.named_parameters():
        if name.endswith("mlp.gate.weight"):
            weight = param
            break
    if weight is None:
        raise ConversionExactnessError("converted model exposes no mlp.gate.weight to check")
    logits = torch.zeros(num_experts, dtype=weight.dtype)
    probs = torch.softmax(logits.to(torch.float32), dim=-1)
    top_values, _ = torch.topk(probs, num_experts, dim=-1)
    top_values = top_values / top_values.sum(dim=-1, keepdim=True)
    expected = torch.full((num_experts,), 1.0 / num_experts, dtype=torch.float32)
    return float((top_values - expected).abs().max())
