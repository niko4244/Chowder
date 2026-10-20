"""int8 frozen experts for the 16 GB MoE training path (option B).

``cpu-transient-grad-bulk`` keeps the routed experts in host RAM as bf16 and
copies one layer's matrices to the training device per pass. That works while
the host has room: for REAP-160 Qwen3.6 (160 experts/layer, 42.9 GB checkpoint)
the bf16 experts are ~40 GB, and on a 64 GB machine with ~24-33 GB available the
training step pages to disk.

This module is option B: keep the same frozen experts in RAM as **int8 with one
per-output-row scale**, so the resident expert bytes halve (int8 weights plus an
fp32 scale per output row is 8.5 bits/weight, ~20 GB for the same artifact),
copy the int8 layer to the device per pass and dequantize *there*. The placement
is ``cpu-transient-grad-int8``; the forward/backward contract is the bulk one
(one copy per layer per pass, recompute in backward, nothing but activations and
routing saved), so gradients still reach the hidden states -- and every LoRA
leaf below the layer -- and ``top_k_weights``.

Two things this module exists to guarantee:

* **Peak host RAM never holds the bf16 experts.** ``load_model_int8_experts``
  builds the model skeleton on ``meta``, then streams the safetensors shards
  tensor by tensor: an expert tensor is quantized as soon as it is read and the
  bf16 values are dropped before the next one is touched, so at most one expert
  tensor is live at a time. Everything else is loaded to the device.
* **Nothing bf16 is left for the experts.** The expert parameters are removed
  from each experts module (before the skeleton is materialised, so they are
  never allocated) and replaced by four CPU buffers: ``gate_up_q``,
  ``gate_up_scale``, ``down_q``, ``down_scale``.

Numerically this is not the bulk path: each expert weight row is rounded to a
127-level grid, so an expert's output carries up to ~0.4% relative error per
element (half a step of ``absmax/127``). The router and every LoRA leaf still see
gradients computed through the dequantized weights, exactly as the bulk path
computes them through the bf16 weights.

Design: docs/DESIGN-MOE-16GB-TRAINING-BACKEND-2026-10-18.md (option B). This
module is the mechanism only; the placement's preregistered gates must be
re-measured on the real artifact before any campaign declares it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors import safe_open

from .moe_transient_grad import (
    EXPERTS_CLASSES,
    EXPERT_PARAM_INFIX,
    INT8_PLACEMENT,
    _layer_moe_forward,
)

#: The placement name for this path (also in ``moe_transient_grad.EXPERT_PLACEMENTS``).
INT8_PLACEMENTS = (INT8_PLACEMENT,)

#: The four CPU buffers an int8 experts module carries instead of its bf16 params.
INT8_EXPERT_BUFFERS = ("gate_up_q", "gate_up_scale", "down_q", "down_scale")

#: The quantisation grid: ``max |w_row| / 127`` -- the largest magnitude a signed
#: int8 row can express, so the worst per-element error is half of one step.
INT8_MAX = 127


def quantize_rows(weight: Any, *, scale_dtype: Any = None) -> tuple[Any, Any]:
    """Symmetric per-output-row int8 quantisation of a weight matrix.

    The last dimension is the *input* dimension (``F.linear`` contracts it), so
    each row over the last dim is one output feature: ``scale = max|row| / 127``
    and ``q = round(w / scale)`` clamped to ``[-127, 127]``.

    A row that is entirely zero keeps a scale of 1.0 rather than 0.0: dividing by
    zero would produce NaNs, while an all-zero int8 row dequantizes back to 0
    with either scale.

    Returns ``(q[int8], scale)`` with the scale in ``scale_dtype`` (fp32 by
    default: it is one value per output row, 0.5 byte per 4 weights).
    """
    import torch

    if scale_dtype is None:
        scale_dtype = torch.float32
    detached = weight.detach()
    absmax = detached.abs().amax(dim=-1, keepdim=True).float()
    scale = torch.where(absmax > 0, absmax / float(INT8_MAX), torch.ones_like(absmax))
    q = torch.round(detached.float() / scale).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return q, scale.to(scale_dtype)


#: Dequantisation works through row blocks of at most this many fp32 elements
#: (16 MB). The fp32 product of a whole expert matrix would be twice its bf16
#: bytes on the card, which the 16 GB budget cannot spend on top of the copy.
DEQUANT_ELEMENTS_PER_BLOCK = 1 << 22


def dequantize_rows(q: Any, scale: Any, dtype: Any = None) -> Any:
    """``q * scale`` with the multiply in fp32, returned in ``dtype``.

    Works on any device: the arithmetic happens wherever ``q`` lives, so a
    caller that copied the int8 weights to the card dequantizes on the card.
    The product is accumulated in row blocks (``DEQUANT_ELEMENTS_PER_BLOCK``),
    so a layer's worth of fp32 never exists at once.
    """
    import torch

    out_dtype = torch.float32 if dtype is None else dtype
    out = torch.empty(q.shape, dtype=out_dtype, device=q.device)
    if q.numel() == 0:
        return out
    row_elements = max(1, q[0].numel())
    block = max(1, DEQUANT_ELEMENTS_PER_BLOCK // row_elements)
    for start in range(0, q.shape[0], block):
        rows = slice(start, start + block)
        out[rows] = (q[rows].to(torch.float32) * scale[rows].to(torch.float32)).to(out_dtype)
    return out


def resolve_device(device: Any = 0) -> Any:
    """``device=0`` means "the training card" -- except where there is none.

    ``CHOWDER_NO_GPU`` (or no CUDA) resolves to the CPU so the CPU tests and any
    no-GPU inspection run the same loader instead of skipping it.
    """
    import os

    import torch

    disabled = os.environ.get("CHOWDER_NO_GPU", "").strip().lower() not in ("", "0", "false")
    if device is None:
        resolved = torch.device("cuda", 0)
    elif isinstance(device, int):
        resolved = torch.device("cuda", device)
    else:
        resolved = torch.device(device)
    if resolved.type != "cuda":
        return resolved
    if disabled or not torch.cuda.is_available():
        return torch.device("cpu")
    return resolved


def _dtype_from_config(config: Any) -> Any:
    """The dtype ``from_pretrained`` would use: the config's, else fp32."""
    import torch

    dtype = getattr(config, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        mapped = getattr(torch, dtype, None)
        if isinstance(mapped, torch.dtype):
            return mapped
    return torch.float32


def _experts_modules(model: Any) -> list[tuple[str, Any]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if type(module).__name__ in EXPERTS_CLASSES
    ]


def _shard_files(base: Path) -> list[Path]:
    """Every safetensors shard of a local checkpoint, in index order when there
    is an index (a sharded checkpoint's order is the index's, not the name's)."""
    index_path = base / "model.safetensors.index.json"
    shards: list[Path] = []
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for name in (index.get("weight_map") or {}).values():
            path = base / str(name)
            if path not in shards:
                shards.append(path)
    if not shards:
        shards = sorted(base.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards under {str(base)!r}")
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"the safetensors index names shards that are missing: {missing[:3]}")
    return shards


def _empty_int8_experts(module: Any, *, scale_dtype: Any) -> dict[str, Any]:
    """The int8 storage for one layer, shaped from the skeleton's own params."""
    import torch

    gate_up = module.gate_up_proj
    down = module.down_proj
    return {
        "gate_up_q": torch.empty(gate_up.shape, dtype=torch.int8),
        "gate_up_scale": torch.empty((*gate_up.shape[:-1], 1), dtype=scale_dtype),
        "down_q": torch.empty(down.shape, dtype=torch.int8),
        "down_scale": torch.empty((*down.shape[:-1], 1), dtype=scale_dtype),
    }


def _quantize_into(store: dict[str, Any], which: str, rows: Any, expert: int | None = None, offset: int = 0) -> None:
    """Quantize ``rows`` and write them into the layer's int8 buffers.

    ``expert`` selects the 3-D slice, ``offset`` the row block inside it: a
    legacy checkpoint's ``gate_proj`` and ``up_proj`` are the two halves of the
    fused ``gate_up_proj`` (``chunk(2, dim=-1)`` -> gate first), so they land in
    rows ``[0:inter]`` and ``[inter:2*inter]`` of the fused matrix.
    """
    q, scale = quantize_rows(rows)
    if expert is None:
        store[f"{which}_q"].copy_(q)
        store[f"{which}_scale"].copy_(scale)
        return
    q_target = store[f"{which}_q"][expert]
    s_target = store[f"{which}_scale"][expert]
    rows_in = q.shape[0]
    q_target[offset:offset + rows_in].copy_(q)
    s_target[offset:offset + rows_in].copy_(scale)


#: Checkpoint key prefixes transformers' own loader rewrites and this streaming
#: loader must rewrite itself. A Qwen3.5/3.6 conditional-generation checkpoint
#: nests the text model one level deeper than ``...ForCausalLM`` does, and REAP
#: exports keep the source keys byte-for-byte (see chowder_batch/reap_prune.py,
#: which matches ``model.language_model.layers.<i>.mlp.experts.gate_up_proj``).
_WRAPPER_PREFIXES = (
    ("model.language_model.", "model."),
    ("language_model.", "model."),
)

#: Tensors a causal-LM skeleton has no module for, recorded rather than treated
#: as a load failure: a conditional-generation export carries its vision tower
#: and MTP head in the same shards.
_UNUSED_TOWER_PREFIXES = (
    "model.visual.",
    "visual.",
    "model.vision_tower.",
    "model.multi_modal_projector.",
    "mtp.",
)


def _model_key(raw: str, params: dict[str, Any], buffers: dict[str, Any]) -> str:
    """The skeleton's name for a checkpoint key, or the key unchanged.

    An exact match wins; otherwise the text model's wrapper prefix is removed
    when the skeleton names that tensor without it.
    """
    if raw in params or raw in buffers:
        return raw
    for prefix, replacement in _WRAPPER_PREFIXES:
        if raw.startswith(prefix):
            candidate = replacement + raw[len(prefix):]
            if candidate in params or candidate in buffers or EXPERT_PARAM_INFIX in candidate:
                return candidate
    return raw


def _expert_key_target(key: str) -> tuple[str, str, int | None, int]:
    """Map an expert checkpoint key to ``(layer_prefix, which, expert, offset)``.

    Accepts the fused layout transformers 5 keeps on the model
    (``...mlp.experts.gate_up_proj`` / ``...mlp.experts.down_proj``) and the
    per-expert layout its own ``save_pretrained`` writes, and older REAP exports
    carry (``...mlp.experts.<e>.gate_proj.weight`` and friends).
    """
    prefix, _, tail = key.rpartition(EXPERT_PARAM_INFIX)
    experts_module = prefix + EXPERT_PARAM_INFIX.rstrip(".")
    if tail in ("gate_up_proj", "down_proj"):
        return experts_module, ("gate_up" if tail == "gate_up_proj" else "down"), None, 0
    stem = tail[:-len(".weight")] if tail.endswith(".weight") else tail
    head, _, leaf = stem.partition(".")
    if head.isdigit() and leaf in ("gate_proj", "up_proj", "down_proj"):
        expert = int(head)
        if leaf == "down_proj":
            return experts_module, "down", expert, 0
        # gate/up are fused with gate first, exactly as ``chunk(2, dim=-1)`` reads it.
        return experts_module, "gate_up", expert, 0 if leaf == "gate_proj" else -1
    raise ValueError(
        f"unrecognised expert tensor key {key!r}: expected {EXPERT_PARAM_INFIX}"
        "gate_up_proj/down_proj or <index>.gate_proj/up_proj/down_proj.weight"
    )


def int8_expert_census(model: Any) -> dict[str, Any]:
    """What the int8 experts actually occupy: buffers, bytes, and their device."""
    int8_bytes = 0
    scale_bytes = 0
    devices: set[str] = set()
    modules = 0
    buffers = 0
    for _, module in _experts_modules(model):
        modules += 1
        for name in INT8_EXPERT_BUFFERS:
            tensor = getattr(module, name, None)
            if tensor is None:
                continue
            buffers += 1
            devices.add(tensor.device.type)
            nbytes = tensor.numel() * tensor.element_size()
            if name.endswith("_q"):
                int8_bytes += nbytes
            else:
                scale_bytes += nbytes
    return {
        "expert_modules": modules,
        "expert_int8_buffers": buffers,
        "expert_int8_bytes": int8_bytes,
        "expert_scale_bytes": scale_bytes,
        "expert_bytes": int8_bytes + scale_bytes,
        "expert_buffer_devices": sorted(devices),
    }


def load_model_int8_experts(
    base_dir: str | Path,
    *,
    device: Any = 0,
    dtype: Any = None,
    scale_dtype: Any = None,
    revision: str | None = None,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Load a local MoE checkpoint with its routed experts quantized to int8 in RAM.

    The host never holds the bf16 experts: the skeleton is built on ``meta``, the
    expert parameters are dropped from it before materialisation, and each
    shard's tensors are read one at a time -- expert tensors are quantized
    immediately and dropped before the next key is touched.

    Returns ``(model, census)``. The census carries the expert bytes resident in
    RAM (int8 + scales), the non-expert parameters' device, the number of expert
    modules the loader found, and any missing/unexpected keys.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    base = Path(base_dir)
    if not base.is_dir():
        raise ValueError(
            f"load_model_int8_experts streams local safetensors shards, so base_dir must be a "
            f"local checkpoint directory; got {str(base_dir)!r}"
        )
    target_device = resolve_device(device)
    config = AutoConfig.from_pretrained(
        str(base),
        revision=revision,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    resolved_dtype = dtype if dtype is not None else _dtype_from_config(config)
    if scale_dtype is None:
        scale_dtype = torch.float32

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=resolved_dtype)

    # Drop the expert parameters BEFORE anything is materialised: ``to_empty``
    # would otherwise allocate their bf16 bytes on the host, which is the whole
    # memory this placement exists to avoid.
    experts = _experts_modules(model)
    if not experts:
        raise RuntimeError(
            f"no experts module ({', '.join(EXPERTS_CLASSES)}) in {str(base)!r}; "
            f"expert_placement={INT8_PLACEMENT!r} needs a routed-expert MoE checkpoint"
        )
    stores: dict[str, dict[str, Any]] = {}
    for name, module in experts:
        stores[name] = _empty_int8_experts(module, scale_dtype=scale_dtype)
        for param_name in ("gate_up_proj", "down_proj"):
            if param_name in module._parameters:
                del module._parameters[param_name]

    model.to_empty(device=target_device)

    # A tied head is one tensor, as ``from_pretrained`` leaves it, and the
    # checkpoint may then carry only the embedding.
    if getattr(config, "tie_word_embeddings", False):
        model.tie_weights()

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    loaded: set[str] = set()
    unexpected: list[str] = []
    ignored: list[str] = []
    renamed = 0

    # A tensor can have several owners (a tied head is the embedding), and some
    # models initialise a child's tensor from the parent's ``_init_weights``
    # (Qwen3_5MoeSparseMoeBlock re-inits ``gate.weight``). So every owner AND its
    # ancestor chain must be marked, or a later re-initialisation would overwrite
    # the value just loaded.
    owners: dict[int, list[str]] = {}
    for module_name, module in model.named_modules():
        for tensor in list(module._parameters.values()) + list(module._buffers.values()):
            if tensor is not None:
                owners.setdefault(id(tensor), []).append(module_name)

    def mark_initialized(target: Any) -> None:
        """Tell ``initialize_weights`` these modules' tensors came from the
        checkpoint, so only the modules with nothing loaded (the rotary
        buffers) get initialised -- ``from_pretrained``'s own bookkeeping."""
        for module_name in owners.get(id(target), ()):
            parts = module_name.split(".") if module_name else []
            for depth in range(1, len(parts) + 1):
                model.get_submodule(".".join(parts[:depth]))._is_hf_initialized = True

    shards = _shard_files(base)
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for raw_key in handle.keys():
                key = _model_key(raw_key, params, buffers)
                if key != raw_key:
                    renamed += 1
                if (
                    key not in params
                    and key not in buffers
                    and key.startswith(_UNUSED_TOWER_PREFIXES)
                ):
                    ignored.append(raw_key)
                    continue
                tensor = handle.get_tensor(raw_key)
                if EXPERT_PARAM_INFIX in key:
                    layer, which, expert, offset = _expert_key_target(key)
                    store = stores.get(layer)
                    if store is None:
                        raise RuntimeError(
                            f"expert tensor {raw_key!r} belongs to {layer!r}, which is not an experts module "
                            f"on this model (experts modules: {sorted(stores)[:3]})"
                        )
                    if offset == -1:
                        offset = store[f"{which}_q"].shape[1] // 2
                    _quantize_into(store, which, tensor, expert=expert, offset=offset)
                    del tensor
                    loaded.add(key)
                    continue
                target = params.get(key)
                if target is None:
                    target = buffers.get(key)
                if target is None:
                    unexpected.append(raw_key)
                    del tensor
                    continue
                with torch.no_grad():
                    target.copy_(tensor.to(dtype=target.dtype, device=target.device))
                mark_initialized(target)
                del tensor
                loaded.add(key)

    if unexpected:
        raise RuntimeError(
            f"{len(unexpected)} checkpoint tensor(s) have no home on the model skeleton, "
            f"e.g. {unexpected[:3]}; this loader does not skip tensors silently"
        )

    for name, module in experts:
        store = stores.pop(name)
        for buffer_name in INT8_EXPERT_BUFFERS:
            module.register_buffer(buffer_name, store[buffer_name], persistent=False)
        module._is_hf_initialized = True

    # Non-persistent buffers (the rotary inverse frequencies) are in no
    # checkpoint: ``initialize_weights`` fills exactly the modules this loader
    # did not mark, which is what ``from_pretrained`` does for missing keys.
    model.initialize_weights()
    # ``from_pretrained`` records where the weights came from on the config, and
    # generation/tooling reads it; ``from_config`` blanks it.
    model.config._name_or_path = str(base)

    missing = [
        name
        for name in params
        if name not in loaded and not (getattr(config, "tie_word_embeddings", False) and name == "lm_head.weight")
    ]
    if missing:
        raise RuntimeError(
            f"{len(missing)} model parameter(s) were not in the checkpoint, e.g. {missing[:3]}; "
            "loading random weights for them would be silently wrong"
        )

    census: dict[str, Any] = {
        "loader": "int8-streaming",
        "shards": len(shards),
        "device": str(target_device),
        "dtype": str(resolved_dtype),
        "scale_dtype": str(scale_dtype),
        **int8_expert_census(model),
        "non_expert_params": len(params),
        "non_expert_params_on_device": sum(
            parameter.device.type == target_device.type for parameter in params.values()
        ),
        "non_expert_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in params.values()
        ),
        "tensors_loaded": len(loaded),
        "keys_renamed": renamed,
        "ignored_keys": sorted(ignored)[:20],
        "ignored_key_count": len(ignored),
        "unexpected_keys": [],
        # Modules that received int8 storage instead of their bf16 parameters;
        # ``prepare_int8_placement`` overwrites this with the patched-forward
        # count once the forward is installed.
        "patched_expert_modules": len(experts),
    }
    return model, census


def _dequantized_on(tensor: Any, scale: Any, device: Any, dtype: Any) -> Any:
    """Copy one int8 matrix and its scales to ``device``, dequantize there."""
    return dequantize_rows(tensor.to(device, non_blocking=True), scale.to(device, non_blocking=True), dtype)


def _int8_layer_function() -> Any:
    import torch

    class TransientLayerExpertsInt8(torch.autograd.Function):
        """A whole layer's frozen int8 experts: copied in, dequantized on the
        device, recomputed in backward, never saved."""

        @staticmethod
        def forward(
            ctx: Any,
            hidden_states: Any,
            top_k_index: Any,
            top_k_weights: Any,
            gate_up_q: Any,
            gate_up_scale: Any,
            down_q: Any,
            down_scale: Any,
            act_fn: Any,
            num_experts: int,
        ) -> Any:
            ctx.save_for_backward(hidden_states, top_k_index, top_k_weights)
            ctx.weights = (gate_up_q, gate_up_scale, down_q, down_scale)  # CPU-resident int8
            ctx.act_fn = act_fn
            ctx.num_experts = num_experts
            dev, dtype = hidden_states.device, hidden_states.dtype
            with torch.no_grad():
                gate_up = _dequantized_on(gate_up_q, gate_up_scale, dev, dtype)
                down = _dequantized_on(down_q, down_scale, dev, dtype)
                out = _layer_moe_forward(
                    hidden_states, top_k_index, top_k_weights, gate_up, down, act_fn, num_experts
                )
            del gate_up, down
            return out

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> Any:
            hidden_states, top_k_index, top_k_weights = ctx.saved_tensors
            gate_up_q, gate_up_scale, down_q, down_scale = ctx.weights
            with torch.enable_grad():
                x = hidden_states.detach().requires_grad_(True)
                w = top_k_weights.detach().requires_grad_(True)
                dev, dtype = x.device, x.dtype
                gate_up = _dequantized_on(gate_up_q, gate_up_scale, dev, dtype)
                down = _dequantized_on(down_q, down_scale, dev, dtype)
                y = _layer_moe_forward(x, top_k_index, w, gate_up, down, ctx.act_fn, ctx.num_experts)
                grad_x, grad_w = torch.autograd.grad(y, (x, w), grad_out)
            del gate_up, down
            return grad_x, None, grad_w, None, None, None, None, None, None

    return TransientLayerExpertsInt8


def install_int8_expert_forward(model: Any) -> int:
    """Patch every experts module with the int8 transient forward.

    Returns the number of patched modules. Refuses a module that is missing its
    int8 buffers, or that still carries trainable expert parameters: this path
    gives no weight gradient, so training them would be a silent no-op.
    """
    import torch

    function = _int8_layer_function()

    def forward(self: Any, hidden_states: Any, top_k_index: Any, top_k_weights: Any) -> Any:
        return function.apply(
            hidden_states,
            top_k_index,
            top_k_weights,
            self.gate_up_q,
            self.gate_up_scale,
            self.down_q,
            self.down_scale,
            self.act_fn,
            self.num_experts,
        )

    patched = 0
    for name, module in _experts_modules(model):
        missing = [buffer for buffer in INT8_EXPERT_BUFFERS if getattr(module, buffer, None) is None]
        if missing:
            raise RuntimeError(
                f"{name} has no int8 expert buffers {missing}; load it with "
                "load_model_int8_experts before installing this forward"
            )
        trainable = [param for param in module.parameters() if param.requires_grad]
        if trainable:
            raise RuntimeError(
                f"{name} has trainable expert weights; the transient-grad path keeps "
                "experts frozen and gives them no gradient, so freeze them "
                "(requires_grad=False) rather than train them to no effect"
            )
        module.forward = forward.__get__(module)
        patched += 1
    return patched


def prepare_int8_placement(model: Any, *, device_type: str = "cuda") -> dict[str, Any]:
    """Census the int8 placement, then install the int8 transient forward.

    Raises when the placement did not happen: no experts module, a bf16 expert
    parameter left behind, an int8 buffer off the CPU, or a non-expert
    parameter off the device all mean the request was not honoured.
    """
    census = int8_expert_census(model)
    leftover = [name for name, _ in model.named_parameters() if EXPERT_PARAM_INFIX in name]
    others = [(name, parameter) for name, parameter in model.named_parameters() if EXPERT_PARAM_INFIX not in name]
    census["remaining_expert_params"] = len(leftover)
    census["non_expert_params"] = len(others)
    census["non_expert_params_on_device"] = sum(
        parameter.device.type == device_type for _, parameter in others
    )
    problems: list[str] = []
    if census["expert_modules"] == 0:
        problems.append("no experts module was found")
    if leftover:
        problems.append(f"{len(leftover)} bf16 expert parameter(s) remain, e.g. {leftover[0]}")
    if census["expert_buffer_devices"] not in ([], ["cpu"]):
        problems.append(f"int8 expert buffers are not CPU-resident: {census['expert_buffer_devices']}")
    if census["non_expert_params_on_device"] != len(others):
        problems.append(
            f"{len(others) - census['non_expert_params_on_device']} non-expert parameter(s) are not on "
            f"{device_type!r}"
        )
    if problems:
        raise RuntimeError(
            f"the {INT8_PLACEMENT} placement census failed on the live model: {'; '.join(problems)}; "
            "a placement that did not happen must not be believed because it was requested"
        )
    census["patched_expert_modules"] = install_int8_expert_forward(model)
    if census["patched_expert_modules"] == 0:
        raise RuntimeError("int8 expert buffers exist but no experts module was patched")
    return census
