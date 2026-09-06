"""Real router/expert instrumentation for Chowder's MoE downsizing program.

Implements the "First implementation slice" of docs/MOE_DOWNSIZING.md Phase A/B:
verify the exact MoE module shape before touching anything, then capture
per-(layer, expert) usage statistics over a calibration corpus.

Module-shape assumptions here are not guesses. They were verified by reading
the actual installed transformers==5.16.1 source for
``Qwen3MoeSparseMoeBlock``, ``Qwen3_5MoeSparseMoeBlock``, and
``OlmoeSparseMoeBlock``: all three share one fused-batched-expert design --
a router (``mlp.gate``, a ``*TopKRouter`` with a single ``.weight`` matrix)
plus one fused expert module (``mlp.experts``, a ``*Experts`` module holding
single ``gate_up_proj``/``down_proj`` tensors indexed by expert id inside a
Python loop) -- rather than one ``nn.Module`` per expert. Per Phase A's own
rule, failure to identify this shape on a given model is a hard stop, not a
reason to guess a different attribute path.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .moe_planning import ExpertImportance

# A minimal starting calibration set touching each Phase B category (code
# generation, debugging, repo navigation, tool calling, planning, long-context
# reasoning, appliance diagnostics, document extraction, general reasoning).
# docs/MOE_DOWNSIZING.md expects this corpus to grow; callers may pass their
# own texts to run_calibration instead.
DEFAULT_CALIBRATION_TEXTS: tuple[str, ...] = (
    "def add(a, b):\n    return a + b",
    "Traceback (most recent call last): ValueError: invalid literal",
    "git diff shows three files changed in src/chowder/backends",
    "Call the search tool, then read the top result before answering",
    "Step 1: gather requirements. Step 2: design. Step 3: implement.",
    "The context window holds up to 128k tokens for this configuration",
    "The refrigerator compressor cycles on for two minutes then stops",
    "Extract the total from the invoice PDF and return it as JSON",
    "Hello, how has your week been going so far?",
)


class MoeArchitectureAuditError(RuntimeError):
    """Raised when the verified MoE module shape cannot be located.

    Per docs/MOE_DOWNSIZING.md Phase A: failure to identify a tensor/module
    is a hard stop, not a reason to guess a name.
    """


@dataclass(frozen=True)
class MoeLayerStructure:
    layer: int
    num_experts: int
    num_experts_per_tok: int
    hidden_dim: int


@dataclass(frozen=True)
class MoeArchitectureAudit:
    model_type: str
    num_hidden_layers: int
    moe_layers: tuple[MoeLayerStructure, ...]
    dense_layer_indices: tuple[int, ...]

    @property
    def is_uniform(self) -> bool:
        return len(self.dense_layer_indices) == 0


def _decoder_layers(model: Any) -> list[Any]:
    """Locate the model's decoder layer list.

    Verified against Qwen3Moe/Qwen3_5Moe/Olmoe, which all expose
    ``model.model.layers``. A model that does not match is a hard stop, not
    a guess at an alternate attribute path.
    """
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is None:
        raise MoeArchitectureAuditError(
            "cannot locate model.model.layers on this model; refusing to "
            "guess an alternate attribute path for an unverified architecture"
        )
    return list(layers)


def _moe_block(layer: Any) -> Any | None:
    """Return ``layer.mlp`` if it matches the verified fused-expert shape.

    Verified shape (transformers 5.x Qwen3Moe/Qwen3_5Moe/Olmoe): ``mlp.gate``
    has ``.weight``/``.num_experts``/``.top_k``; ``mlp.experts`` has
    ``.gate_up_proj``/``.down_proj``/``.num_experts``/``.act_fn``. Returns
    None for a legitimate dense layer (Qwen3Moe interleaves dense layers via
    ``decoder_sparse_step``) rather than raising -- absence on one layer is
    not evidence the whole model lacks the shape.
    """
    block = getattr(layer, "mlp", None)
    if block is None:
        return None
    gate = getattr(block, "gate", None)
    experts = getattr(block, "experts", None)
    if gate is None or experts is None:
        return None
    has_router_shape = hasattr(gate, "weight") and hasattr(gate, "num_experts") and hasattr(gate, "top_k")
    has_experts_shape = (
        hasattr(experts, "gate_up_proj")
        and hasattr(experts, "down_proj")
        and hasattr(experts, "num_experts")
        and hasattr(experts, "act_fn")
    )
    if not (has_router_shape and has_experts_shape):
        return None
    return block


def audit_moe_architecture(model: Any) -> MoeArchitectureAudit:
    """Verify the real module structure of ``model`` before any instrumentation.

    Raises MoeArchitectureAuditError if no layer matches the verified shape.
    """
    layers = _decoder_layers(model)
    moe_layers: list[MoeLayerStructure] = []
    dense_layer_indices: list[int] = []
    for index, layer in enumerate(layers):
        block = _moe_block(layer)
        if block is None:
            dense_layer_indices.append(index)
            continue
        moe_layers.append(
            MoeLayerStructure(
                layer=index,
                num_experts=int(block.experts.num_experts),
                num_experts_per_tok=int(block.gate.top_k),
                hidden_dim=int(block.gate.weight.shape[1]),
            )
        )
    if not moe_layers:
        raise MoeArchitectureAuditError(
            "no decoder layer matched the verified router+experts MoE shape "
            "(mlp.gate.{weight,num_experts,top_k} and "
            "mlp.experts.{gate_up_proj,down_proj,num_experts,act_fn}); "
            "refusing to guess a different module layout for this model"
        )
    model_type = getattr(getattr(model, "config", None), "model_type", type(model).__name__)
    return MoeArchitectureAudit(
        model_type=str(model_type),
        num_hidden_layers=len(layers),
        moe_layers=tuple(moe_layers),
        dense_layer_indices=tuple(dense_layer_indices),
    )


@dataclass
class _Accumulator:
    selected_tokens: int = 0
    router_mass: float = 0.0
    gated_activation: float = 0.0
    output_norm: float = 0.0


class MoeCalibrationRecorder:
    """Captures exact per-(layer, expert) usage statistics via forward hooks.

    Gated-activation and output-norm are not proxies: they re-derive the same
    per-expert computation ``*Experts.forward`` performs internally
    (``act_fn(gate) * up`` then ``down_proj``) using the live model
    parameters and the real tokens routed to that expert in this batch. This
    side computation is required because experts are stored as one fused
    batched tensor indexed inside a Python loop, not as separate hookable
    submodules -- there is no per-expert ``nn.Module`` to attach a hook to.
    """

    def __init__(self, model: Any, audit: MoeArchitectureAudit) -> None:
        self._model = model
        self._audit = audit
        self._accumulators: dict[tuple[int, int], _Accumulator] = {}
        self._handles: list[Any] = []
        for structure in audit.moe_layers:
            for expert in range(structure.num_experts):
                self._accumulators[(structure.layer, expert)] = _Accumulator()

    def __enter__(self) -> "MoeCalibrationRecorder":
        import torch  # heavy ML dependency; only needed while recording

        layers = _decoder_layers(self._model)
        for structure in self._audit.moe_layers:
            block = _moe_block(layers[structure.layer])
            if block is None:
                raise MoeArchitectureAuditError(
                    f"layer {structure.layer} no longer matches the audited MoE "
                    "shape; refusing to record against a model that changed "
                    "since it was audited"
                )
            handle = block.gate.register_forward_hook(
                self._make_hook(structure.layer, block.experts, torch)
            )
            self._handles.append(handle)
        return self

    def __exit__(self, *exc_info: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, layer_index: int, experts_module: Any, torch_module: Any):
        def hook(module: Any, args: tuple[Any, ...], output: tuple[Any, Any, Any]) -> None:
            hidden_states = args[0].detach()
            _, router_scores, router_indices = output
            router_scores = router_scores.detach()
            router_indices = router_indices.detach()
            with torch_module.no_grad():
                for expert_idx in range(experts_module.num_experts):
                    mask = router_indices == expert_idx
                    if not bool(mask.any()):
                        continue
                    token_idx, slot_idx = torch_module.where(mask)
                    weights_for_expert = router_scores[token_idx, slot_idx]
                    current_state = hidden_states[token_idx]
                    gate, up = torch_module.nn.functional.linear(
                        current_state, experts_module.gate_up_proj[expert_idx]
                    ).chunk(2, dim=-1)
                    gated = experts_module.act_fn(gate) * up
                    expert_output = torch_module.nn.functional.linear(
                        gated, experts_module.down_proj[expert_idx]
                    )
                    accumulator = self._accumulators[(layer_index, int(expert_idx))]
                    accumulator.selected_tokens += int(token_idx.numel())
                    accumulator.router_mass += float(weights_for_expert.sum().item())
                    accumulator.gated_activation += float(gated.norm(dim=-1).sum().item())
                    accumulator.output_norm += float(expert_output.norm(dim=-1).sum().item())

        return hook

    def importance_records(self) -> list[ExpertImportance]:
        return [
            ExpertImportance(
                layer=layer,
                expert=expert,
                selected_tokens=accumulator.selected_tokens,
                router_mass=accumulator.router_mass,
                gated_activation=accumulator.gated_activation,
                output_norm=accumulator.output_norm,
            )
            for (layer, expert), accumulator in sorted(self._accumulators.items())
        ]


def run_calibration(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    device: str = "cpu",
    max_length: int = 512,
) -> tuple[MoeArchitectureAudit, list[ExpertImportance]]:
    """Audit ``model``'s MoE structure, then run it over ``texts`` recording
    real per-(layer, expert) usage. Performs no model surgery."""
    import torch

    audit = audit_moe_architecture(model)
    model.eval()
    with MoeCalibrationRecorder(model, audit) as recorder:
        for text in texts:
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                model(**encoded)
        records = recorder.importance_records()
    return audit, records


def write_expert_importance_jsonl(
    path: str | Path,
    *,
    audit: MoeArchitectureAudit,
    records: Sequence[ExpertImportance],
    model_source: str,
    calibration_texts: Sequence[str],
    transformers_version: str,
) -> None:
    """Emit a versioned expert_importance.jsonl with model/data provenance."""
    provenance = {
        "schema_version": 1,
        "model_source": str(model_source),
        "model_type": audit.model_type,
        "num_hidden_layers": audit.num_hidden_layers,
        "moe_layer_count": len(audit.moe_layers),
        "dense_layer_indices": list(audit.dense_layer_indices),
        "transformers_version": transformers_version,
        "python_version": platform.python_version(),
        "calibration_corpus_sha256": hashlib.sha256(
            "\n".join(calibration_texts).encode("utf-8")
        ).hexdigest(),
        "calibration_example_count": len(calibration_texts),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"provenance": provenance}) + "\n")
        for record in records:
            handle.write(
                json.dumps(
                    {
                        "layer": record.layer,
                        "expert": record.expert,
                        "selected_tokens": record.selected_tokens,
                        "router_mass": record.router_mass,
                        "gated_activation": record.gated_activation,
                        "output_norm": record.output_norm,
                    }
                )
                + "\n"
            )
