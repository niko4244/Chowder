"""Phase 6 router healing (docs/PHASE6_CONVERSION_PLAN.md sec 6).

After dense->MoE conversion, every expert fires for every token
(num_experts_per_tok == num_experts) by the conversion's own
exactness-preserving design -- there is no compute saving yet. Healing's
job is letting token-conditional routing emerge by training ONLY the
router (`mlp.gate.weight`) and the shared-expert gate
(`mlp.shared_expert_gate.weight`) per layer, with every other parameter
-- including the routed experts themselves -- frozen. This isolates the
routing problem from the expert-weight problem, exactly as the plan
requires.

Real module shapes (verified from the installed transformers source,
`transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`):
`Qwen3_5MoeExperts.gate_up_proj`/`.down_proj` and the router's own
`.weight` are raw `nn.Parameter` tensors, not `nn.Linear` -- bitsandbytes'
4-bit auto-replace only swaps `nn.Linear` modules, so these are never
quantized regardless of `load_in_4bit`; they load as plain BF16 and need
no special handling to stay trainable. `shared_expert_gate` IS a real
`nn.Linear(hidden_size, 1, bias=False)` and WOULD be auto-quantized to
4-bit unless excluded -- `QUANTIZATION_SKIP_MODULES` exists for exactly
that, passed to `BitsAndBytesConfig(llm_int8_skip_modules=...)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The only module name bitsandbytes' 4-bit auto-replace would otherwise
#: quantize among the new MoE-specific tensors (see module docstring).
QUANTIZATION_SKIP_MODULES: tuple[str, ...] = ("shared_expert_gate",)

#: Parameter name suffixes selected for training. Everything else is frozen.
_TRAINABLE_SUFFIXES: tuple[str, ...] = ("mlp.gate.weight", "mlp.shared_expert_gate.weight")


class RouterHealingError(ValueError):
    """Router healing cannot proceed honestly."""


@dataclass(frozen=True)
class RouterHealingFreezeSummary:
    """What `freeze_for_router_healing` actually did -- measured, not assumed."""

    trainable_param_names: tuple[str, ...]
    trainable_param_count: int
    frozen_param_count: int
    layers_with_trainable_gate: int
    layers_with_trainable_shared_expert_gate: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "trainable_param_names": list(self.trainable_param_names),
            "trainable_param_count": self.trainable_param_count,
            "frozen_param_count": self.frozen_param_count,
            "layers_with_trainable_gate": self.layers_with_trainable_gate,
            "layers_with_trainable_shared_expert_gate": self.layers_with_trainable_shared_expert_gate,
        }


def select_trainable_parameter_names(named_parameters: Any) -> list[str]:
    """Pick exactly the router + shared-expert-gate parameter names.

    `named_parameters` is anything iterable of (name, param) pairs (a real
    `model.named_parameters()`, or a fake for tests) -- kept untyped so this
    function needs no torch import and is fast/CPU to unit test.
    """
    return [name for name, _ in named_parameters if name.endswith(_TRAINABLE_SUFFIXES)]


def freeze_for_router_healing(model: Any) -> RouterHealingFreezeSummary:
    """Freeze everything except the router + shared-expert gates.

    Hard-stops (no guessing a different module layout) if nothing matched --
    mirrors `audit_moe_architecture`'s refusal discipline. Call
    `audit_moe_architecture(model)` first; this function trusts that the
    model already has the verified MoE shape.
    """
    named = list(model.named_parameters())
    trainable_names = select_trainable_parameter_names(named)
    if not trainable_names:
        raise RouterHealingError(
            "no parameter matched the router/shared-expert-gate suffixes "
            f"{_TRAINABLE_SUFFIXES}; refusing to guess a different module "
            "layout for this model"
        )
    trainable_set = set(trainable_names)
    trainable_count = 0
    frozen_count = 0
    for name, param in named:
        is_trainable = name in trainable_set
        param.requires_grad_(is_trainable)
        numel = param.numel()
        if is_trainable:
            trainable_count += numel
        else:
            frozen_count += numel

    gate_layers = sum(1 for n in trainable_names if n.endswith("mlp.gate.weight"))
    shared_gate_layers = sum(1 for n in trainable_names if n.endswith("mlp.shared_expert_gate.weight"))
    return RouterHealingFreezeSummary(
        trainable_param_names=tuple(sorted(trainable_names)),
        trainable_param_count=trainable_count,
        frozen_param_count=frozen_count,
        layers_with_trainable_gate=gate_layers,
        layers_with_trainable_shared_expert_gate=shared_gate_layers,
    )
