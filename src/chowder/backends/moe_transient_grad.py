"""Frozen CPU-resident experts that still pass gradients: the 16 GB MoE training path.

``router_healing_load`` keeps expert weights on the CPU and copies one expert's
slice to the training device per forward. That is enough to train the router,
but autograd then *saves* every copied slice for backward: across a whole
forward pass that is every expert of every layer resident on the card at once
(32 GB for the REAP-128 Qwen3.6 MoE), which is the memory the policy exists to
avoid.

``TransientExpertFFN`` saves only the expert's *input*. Backward copies the same
CPU slice again and recomputes, so gradients reach the hidden states (and so
every LoRA leaf below the layer) and the routing weights (and so the router),
while no expert weight is ever held for backward. The experts themselves stay
frozen: a trainable expert would need a weight gradient this path cannot give,
so it is refused rather than silently zeroed.

``cpu-transient-grad-bulk`` is the same contract with one copy per layer instead
of one per expert: per hit expert the old path issues two small host-to-device
copies (gate_up and down) in forward and two again in backward, three times per
step under gradient checkpointing -- ~160 pageable copies per layer for REAP-128.
The bulk path copies the layer's ``gate_up_proj``/``down_proj`` once each on the
same schedule. Peak memory is unchanged in kind (one layer's experts on the
device at a time, freed before the next), so only the transfer count moves.

Design: docs/DESIGN-MOE-16GB-TRAINING-BACKEND-2026-10-18.md (option A). This
module is the mechanism only; registering it as a runnable load policy waits on
that design's acceptance gates.
"""

from __future__ import annotations

from typing import Any

#: The experts classes this forward replaces (transformers 5.x, 3-D expert params).
EXPERTS_CLASSES = ("Qwen3MoeExperts", "Qwen3_5MoeExperts")

#: The placements that keep the experts on the CPU and copy them in per layer.
TRANSIENT_PLACEMENTS = ("cpu-transient-grad", "cpu-transient-grad-bulk")

#: The worker's expert placements. ``cpu-transient-grad`` is pre-registered in
#: docs/quals/MOE_TRANSIENT_GRAD_PLACEMENT_PREREG_2026-10-18.md: no campaign may
#: use it until that note's gates are measured on the real artifact. The bulk
#: value is the same publication status and the same gates.
EXPERT_PLACEMENTS = ("resident",) + TRANSIENT_PLACEMENTS
EXPERT_PARAM_INFIX = ".mlp.experts."


def split_device_map(base_model: str, *, revision: str | None, local_files_only: bool, device: int = 0) -> dict[str, Any]:
    """Experts on the CPU, every other parameter on ``device`` -- from a meta
    enumeration, never by materialising the model on the CPU first."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(base_model, revision=revision, local_files_only=local_files_only)
    with torch.device("meta"):
        names = [name for name, _ in AutoModelForCausalLM.from_config(config).named_parameters()]
    if not any(EXPERT_PARAM_INFIX in name for name in names):
        raise RuntimeError(
            "expert_placement='cpu-transient-grad' was requested for a base with no "
            f"{EXPERT_PARAM_INFIX!r} parameters; there is nothing to place on the CPU"
        )
    return {name: ("cpu" if EXPERT_PARAM_INFIX in name else device) for name in names}


def attach_preserving_split(base: Any, attach: Any) -> Any:
    """Run ``attach(base)`` (e.g. ``PeftModel.from_pretrained``) without PEFT
    re-planning the split: PEFT re-dispatches a model that carries an
    ``hf_device_map``, which moves the CPU experts onto the device (measured in
    #247: all 8 host expert tensors landed on the card). The map is withheld for
    the attach and restored after; the experts are then checked, not assumed."""
    sentinel = object()
    withheld = base.__dict__.pop("hf_device_map", sentinel)
    if withheld is sentinel and getattr(base, "hf_device_map", None) is not None:
        raise RuntimeError(
            "cannot withhold hf_device_map (not an instance attribute), so the attach "
            "would re-plan the CPU experts onto the device"
        )
    try:
        model = attach(base)
    finally:
        if withheld is not sentinel:
            base.hf_device_map = withheld
    moved = [n for n, p in model.named_parameters() if EXPERT_PARAM_INFIX in n and p.device.type != "cpu"]
    if moved:
        raise RuntimeError(f"the adapter attach moved {len(moved)} expert parameters off the CPU, e.g. {moved[0]}")
    return model


def prepare_split_model(model: Any, *, device_type: str = "cuda", bulk: bool = False) -> dict[str, Any]:
    """Census the live placement, freeze the experts, install the transient-grad
    forward (``bulk=True``: one copy per layer, not per expert). Returns the
    evidence; raises when the placement did not happen."""
    experts = [(n, p) for n, p in model.named_parameters() if EXPERT_PARAM_INFIX in n]
    others = [(n, p) for n, p in model.named_parameters() if EXPERT_PARAM_INFIX not in n]
    census = {
        "expert_params": len(experts),
        "expert_params_on_cpu": sum(p.device.type == "cpu" for _, p in experts),
        "other_params": len(others),
        "other_params_on_device": sum(p.device.type == device_type for _, p in others),
    }
    if not experts or census["expert_params_on_cpu"] != len(experts) or census["other_params_on_device"] != len(others):
        raise RuntimeError(
            f"the cpu-transient-grad placement census failed on the live model: {census}; "
            "a placement that did not happen must not be believed because it was requested"
        )
    for _, param in experts:
        param.requires_grad_(False)
    census["patched_expert_modules"] = install_transient_grad_expert_forward(model, bulk=bulk)
    if census["patched_expert_modules"] == 0:
        raise RuntimeError("expert parameters exist but no experts module was patched")
    return census


def _expert_ffn(x: Any, w_gate_up: Any, w_down: Any, act_fn: Any) -> Any:
    import torch.nn.functional as F

    gate, up = F.linear(x, w_gate_up).chunk(2, dim=-1)
    return F.linear(act_fn(gate) * up, w_down)


def _transient_function() -> Any:
    import torch

    class TransientExpertFFN(torch.autograd.Function):
        """One frozen expert's FFN whose weights are copied in, never saved."""

        @staticmethod
        def forward(ctx: Any, x: Any, w_gate_up: Any, w_down: Any, act_fn: Any) -> Any:
            ctx.save_for_backward(x)
            ctx.weights = (w_gate_up, w_down)  # the CPU-resident slices: views, no copy
            ctx.act_fn = act_fn
            dev = x.device
            with torch.no_grad():
                return _expert_ffn(x, w_gate_up.to(dev, non_blocking=True), w_down.to(dev, non_blocking=True), act_fn)

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> Any:
            (x,) = ctx.saved_tensors
            w_gate_up, w_down = ctx.weights
            dev = x.device
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                y = _expert_ffn(x, w_gate_up.to(dev, non_blocking=True), w_down.to(dev, non_blocking=True), ctx.act_fn)
                (grad_x,) = torch.autograd.grad(y, x, grad_out)
            return grad_x, None, None, None

    return TransientExpertFFN


def _layer_moe_forward(
    hidden_states: Any,
    top_k_index: Any,
    top_k_weights: Any,
    gate_up_proj: Any,
    down_proj: Any,
    act_fn: Any,
    num_experts: int,
) -> Any:
    """The expert loop for a whole layer, over device-resident weights: same
    expert order, same ``index_add_`` accumulation and dtype handling as the
    per-expert transient forward, so the two are interchangeable by construction."""
    import torch

    final = torch.zeros_like(hidden_states)
    # Sort the (token, slot) assignments by expert once and read every expert's
    # count in ONE device->host sync. The per-expert ``int(row)`` /
    # ``torch.where(mask[e])`` it replaces synced ~160 times per layer per pass;
    # profiling reap160 put ~42 of ~45 s/step in this loop (2026-10-20).
    top_k = top_k_index.shape[-1]
    flat = top_k_index.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=num_experts).tolist()
    token_of = order // top_k
    slot_of = order % top_k
    start = 0
    for e, count in enumerate(counts):
        if count == 0:
            continue
        token_idx = token_of[start:start + count]
        top_k_pos = slot_of[start:start + count]
        start += count
        out = _expert_ffn(hidden_states[token_idx], gate_up_proj[e], down_proj[e], act_fn)
        out = out * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, out.to(final.dtype))
    return final


def _layer_function() -> Any:
    import torch

    class TransientLayerExperts(torch.autograd.Function):
        """A whole layer's frozen experts: one host->device copy of each weight
        matrix per pass, recomputed in backward, never saved."""

        @staticmethod
        def forward(
            ctx: Any,
            hidden_states: Any,
            top_k_index: Any,
            top_k_weights: Any,
            gate_up_cpu: Any,
            down_cpu: Any,
            act_fn: Any,
            num_experts: int,
        ) -> Any:
            # Only the activations are held for backward; the device copies are
            # local and dropped here, so no expert weight outlives this call.
            ctx.save_for_backward(hidden_states, top_k_index, top_k_weights)
            ctx.weights = (gate_up_cpu, down_cpu)  # the CPU-resident matrices
            ctx.act_fn = act_fn
            ctx.num_experts = num_experts
            dev = hidden_states.device
            with torch.no_grad():
                gate_up = gate_up_cpu.to(dev, non_blocking=True)
                down = down_cpu.to(dev, non_blocking=True)
                out = _layer_moe_forward(
                    hidden_states, top_k_index, top_k_weights, gate_up, down, act_fn, num_experts
                )
            del gate_up, down
            return out

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> Any:
            hidden_states, top_k_index, top_k_weights = ctx.saved_tensors
            gate_up_cpu, down_cpu = ctx.weights
            with torch.enable_grad():
                x = hidden_states.detach().requires_grad_(True)
                w = top_k_weights.detach().requires_grad_(True)
                dev = x.device
                gate_up = gate_up_cpu.to(dev, non_blocking=True)
                down = down_cpu.to(dev, non_blocking=True)
                y = _layer_moe_forward(x, top_k_index, w, gate_up, down, ctx.act_fn, ctx.num_experts)
                grad_x, grad_w = torch.autograd.grad(y, (x, w), grad_out)
            del gate_up, down
            return grad_x, None, grad_w, None, None, None, None

    return TransientLayerExperts


#: Page-locking absent is not an error: it is the default, and the no-GPU paths
#: must return evidence rather than raise.
def pin_expert_memory(model: Any) -> dict[str, Any]:
    """Page-lock every CPU-resident expert tensor in place (no second copy in
    RAM) so the per-layer copies can be truly asynchronous.

    Returns ``{"pinned_bytes", "pinned_tensors", "failed"}``. If any tensor
    fails to register, everything pinned in this call is unregistered again and
    the failure is returned -- the caller decides whether that is fatal.
    """
    import os

    import torch

    disabled = os.environ.get("CHOWDER_NO_GPU", "").strip().lower() not in ("", "0", "false")
    if disabled or not torch.cuda.is_available():
        return {"pinned_bytes": 0, "skipped": "no cuda" if not disabled else "CHOWDER_NO_GPU"}
    experts = [(name, param) for name, param in model.named_parameters() if EXPERT_PARAM_INFIX in name]
    if not experts:
        return {"pinned_bytes": 0, "skipped": "no expert tensors"}
    cudart = torch.cuda.cudart()
    pinned: list[tuple[str, Any, int]] = []
    failed: list[dict[str, Any]] = []
    pinned_bytes = 0
    for name, param in experts:
        if param.device.type != "cpu":
            failed.append({"name": name, "error": f"not on the CPU (on {param.device.type})"})
            continue
        if not param.is_contiguous():
            failed.append({"name": name, "error": "not contiguous, cannot page-lock in place"})
            continue
        nbytes = param.numel() * param.element_size()
        try:
            code = cudart.cudaHostRegister(param.data_ptr(), nbytes, 0)
            if code != 0:
                raise RuntimeError(f"cudaHostRegister returned {code}")
        except Exception as error:  # defensive: a failed registration is evidence, not a crash
            failed.append({"name": name, "error": str(error)})
            continue
        pinned.append((name, param, nbytes))
        pinned_bytes += nbytes
    if failed:
        for _, param, _ in pinned:
            try:
                cudart.cudaHostUnregister(param.data_ptr())
            except Exception:  # ponytail: unpinning a failed pin is best-effort. Ceiling: a registration the driver will not drop leaks until process exit. Upgrade: track and retry at teardown.
                pass
        return {
            "pinned_bytes": 0,
            "pinned_tensors": 0,
            "failed": failed,
        }
    return {"pinned_bytes": pinned_bytes, "pinned_tensors": len(pinned), "failed": []}


def install_transient_grad_expert_forward(model: Any, bulk: bool = False) -> int:
    """Patch every experts module with the gradient-carrying transient forward.

    ``bulk=True`` installs the layer-level forward (one copy of each weight
    matrix per layer per pass); the default keeps the per-expert copy.

    Returns the number of patched modules. Refuses a model whose expert weights
    are trainable: this path gives them no gradient, so training them would be
    a silent no-op.
    """
    import torch
    import torch.nn.functional as F

    function = _layer_function() if bulk else _transient_function()

    def layer_forward(self: Any, hidden_states: Any, top_k_index: Any, top_k_weights: Any) -> Any:
        return function.apply(
            hidden_states,
            top_k_index,
            top_k_weights,
            self.gate_up_proj,
            self.down_proj,
            self.act_fn,
            self.num_experts,
        )

    def forward(self: Any, hidden_states: Any, top_k_index: Any, top_k_weights: Any) -> Any:
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for row in hit:
            e = int(row[0])
            if e == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[e])
            out = function.apply(hidden_states[token_idx], self.gate_up_proj[e], self.down_proj[e], self.act_fn)
            out = out * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, out.to(final.dtype))
        return final

    installed = layer_forward if bulk else forward
    patched = 0
    for name, module in model.named_modules():
        if type(module).__name__ not in EXPERTS_CLASSES:
            continue
        trainable = [p for p in (module.gate_up_proj, module.down_proj) if p.requires_grad]
        if trainable:
            raise RuntimeError(
                f"{name} has trainable expert weights; the transient-grad path keeps "
                "experts frozen and gives them no gradient, so freeze them "
                "(requires_grad=False) rather than train them to no effect"
            )
        module.forward = installed.__get__(module)
        patched += 1
    return patched
