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


#: Router only. The honest selection when the shared expert is zero-init and
#: frozen, which makes its gate provably unlearnable -- better to declare that
#: narrower scope than to designate a dead tensor and have the reachability
#: check refuse the whole run.
ROUTER_ONLY_SUFFIXES: tuple[str, ...] = ("mlp.gate.weight",)


def select_trainable_parameter_names(
    named_parameters: Any, suffixes: tuple[str, ...] | None = None
) -> list[str]:
    """Pick exactly the designated trainable parameter names.

    `named_parameters` is anything iterable of (name, param) pairs (a real
    `model.named_parameters()`, or a fake for tests) -- kept untyped so this
    function needs no torch import and is fast/CPU to unit test.
    `suffixes` defaults to router + shared-expert gate; pass
    `ROUTER_ONLY_SUFFIXES` to scope a run to the router alone.
    """
    wanted = _TRAINABLE_SUFFIXES if suffixes is None else tuple(suffixes)
    return [name for name, _ in named_parameters if name.endswith(wanted)]


def assert_trainable_gradients_reachable(model: Any, trainable_names: Any) -> dict[str, Any]:
    """Refuse a freeze plan whose 'trainable' tensors cannot receive gradient.

    This exists because a real defect shipped past review without it. The
    conversion writes the shared expert's gate/up/down projections as ZEROS
    (exactness: a zero shared expert contributes nothing at init), and the
    block computes ``sigmoid(shared_expert_gate(x)) * shared_expert(x)``. With
    `shared_expert(x)` identically zero AND its projections frozen, the
    derivative w.r.t. `shared_expert_gate` is exactly zero -- forever. So
    64 tensors were marked trainable, consumed optimizer state, and could
    never learn. Verified by CPU probe: grad 0.0 as converted vs 0.16 with a
    non-zero shared expert.

    The original pilot missed it because its proof-of-life was a single
    *global* grad-norm plus the router's own weight norm, both dominated by
    `mlp.gate.weight`, which does learn. A per-tensor reachability check is
    the thing that would have caught it, so it is now a precondition rather
    than an observation.

    Structural, not a forward pass: a tensor multiplied by an all-zero frozen
    factor is unreachable by construction, and detecting that costs a weight
    read instead of a 27B backward. Returns the reachability report; raises
    `RouterHealingError` when a designated-trainable tensor is dead.
    """
    names = list(trainable_names)
    modules = dict(model.named_modules())
    dead: list[dict[str, Any]] = []

    for name in names:
        if not name.endswith("mlp.shared_expert_gate.weight"):
            continue
        # the sibling shared expert whose output this gate scales
        prefix = name[: -len("shared_expert_gate.weight")]
        blockers = []
        for leaf in ("gate_proj", "up_proj", "down_proj"):
            mod = modules.get(f"{prefix}shared_expert.{leaf}")
            weight = getattr(mod, "weight", None) if mod is not None else None
            if weight is None:
                continue
            try:
                all_zero = bool(weight.detach().eq(0).all().item())
            except Exception:  # pragma: no cover - exotic/quantized storage
                all_zero = False
            if all_zero and not weight.requires_grad:
                blockers.append(f"shared_expert.{leaf} is all-zero and frozen")
        if blockers:
            dead.append({"parameter": name, "reasons": blockers})

    if dead:
        raise RouterHealingError(
            "these parameters are marked trainable but cannot receive gradient, "
            "so training them would burn compute and optimizer state for nothing:\n  "
            + "\n  ".join(f"{d['parameter']}: {'; '.join(d['reasons'])}" for d in dead)
            + "\nFix the init or stop designating them trainable -- do not train a "
            "dead path. A zero shared expert multiplied by its gate has zero "
            "derivative w.r.t. that gate."
        )
    return {"checked": len(names), "dead": []}


def freeze_for_router_healing(
    model: Any,
    *,
    require_reachable: bool = True,
    suffixes: tuple[str, ...] | None = None,
) -> RouterHealingFreezeSummary:
    """Freeze everything except the router + shared-expert gates.

    Hard-stops (no guessing a different module layout) if nothing matched --
    mirrors `audit_moe_architecture`'s refusal discipline. Call
    `audit_moe_architecture(model)` first; this function trusts that the
    model already has the verified MoE shape.
    """
    named = list(model.named_parameters())
    trainable_names = select_trainable_parameter_names(named, suffixes)
    if not trainable_names:
        raise RouterHealingError(
            "no parameter matched the requested trainable suffixes "
            f"{_TRAINABLE_SUFFIXES if suffixes is None else tuple(suffixes)}; "
            "refusing to guess a different module "
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

    if require_reachable:
        assert_trainable_gradients_reachable(model, trainable_names)

    gate_layers = sum(1 for n in trainable_names if n.endswith("mlp.gate.weight"))
    shared_gate_layers = sum(1 for n in trainable_names if n.endswith("mlp.shared_expert_gate.weight"))
    return RouterHealingFreezeSummary(
        trainable_param_names=tuple(sorted(trainable_names)),
        trainable_param_count=trainable_count,
        frozen_param_count=frozen_count,
        layers_with_trainable_gate=gate_layers,
        layers_with_trainable_shared_expert_gate=shared_gate_layers,
    )
