"""The rung-3 amendment's load policy: bf16 with experts offloaded, transiently copied.

P11 rung 3 measured that the real 9B-derived MoE artifact cannot train on the
RTX 5060 Ti under the worker's historical policy — fp32 full-resident spills
into Windows' WDDM shared memory (33.36 GB measured against a 17.1 GB card) —
and that bf16 full-resident does not fit either (16.807 GB, before activations).
Strategy C placed the model correctly (9.276 GB, experts on CPU) but stock
transformers 5.16.1 cannot forward CPU-resident expert weights: every
registered experts implementation calls ``F.linear(cuda_activations, cpu_weight)``
and raises a device-mismatch ``RuntimeError``.

The amended policy (preregistered in
``docs/quals/P11_RUNG3_AMENDMENT_2026-09-14.md`` *before* this module existed)
keeps the expert weights CPU-resident and replaces only the experts forward
with a per-expert loop that transiently copies each layer's expert weight
slices to the training device for the duration of one forward. The router
healing recipe trains exactly the router gates, so the frozen experts need no
gradients, the copies run under ``no_grad``, and the transient overhead is
bounded by one layer's expert tensors. Diagnostic D measured this end to end on
the real artifact: 11.368 GB peak, 1.997 s/step, 32/32 gate gradients finite
and non-zero.

Honesty rules
-------------
* Placement is verified **from named parameters after load** — a code path
  claiming offload is not evidence; a census is.
* The base must actually have expert parameters. An offload request on a base
  without them is a contradiction and is refused, never silently degraded to
  full-resident.
* The policy is opt-in: the default stays ``fp32-resident`` and the CPU path
  never enters this module's load seam.
"""

from __future__ import annotations

import json
from typing import Any

#: The two policies the P11 rung-3 record admits. Anything else is refused at
#: spec time — an unknown policy must be preregistered before it can run.
LOAD_POLICIES: tuple[str, ...] = ("fp32-resident", "bf16-offload-transient")

#: The expert-tensor infix shared by the checkpoint naming (``model.language_model.layers.*``)
#: and the loaded-parameter naming (``model.layers.*``); matching on the stable
#: middle rather than either prefix is what made the probe's census honest.
EXPERT_PARAM_INFIX = ".mlp.experts."

#: Router gates: the only parameters the recipe ever trains.
ROUTER_GATE_SUFFIX = "mlp.gate.weight"

GIB = 1024**3


def resolve_load_placement(
    base_model_dir: str,
    parameter_names: list[str],
    *,
    device: str,
) -> dict[str, str]:
    """Return the per-parameter device map for the amended policy.

    ``parameter_names`` must come from meta-device discovery (or any cheap
    parameter-name enumeration of the same architecture) — never from
    materializing the model on CPU, which for the 9B artifact costs the exact
    RAM this policy exists to avoid.
    """
    expert_params = {n for n in parameter_names if EXPERT_PARAM_INFIX in n}
    if not expert_params:
        raise RuntimeError(
            "the bf16-offload-transient policy was requested for a base that has no "
            f"expert parameters ({len(parameter_names)} parameters carry no "
            f"{EXPERT_PARAM_INFIX!r} infix); there is nothing to offload, so the "
            "request contradicts itself. Refusing rather than silently falling back "
            "to full-resident."
        )
    return {n: ("cpu" if n in expert_params else device) for n in parameter_names}


def verify_placement_census(model: Any, *, device_type: str) -> dict[str, Any]:
    """Count expert and gate parameters per device from the live model.

    The amendment's evidence contract: every expert parameter off the training
    device, every router gate on it. Counts come from ``named_parameters()``,
    so a dispatch that silently moved a weight is caught here, not believed.
    """
    expert_total = expert_on_device = 0
    gate_total = gate_on_device = 0
    for name, param in model.named_parameters():
        if EXPERT_PARAM_INFIX in name:
            expert_total += 1
            if param.device.type == device_type:
                expert_on_device += 1
        if name.endswith(ROUTER_GATE_SUFFIX):
            gate_total += 1
            if param.device.type == device_type:
                gate_on_device += 1
    verified = expert_on_device == 0 and gate_total > 0 and gate_on_device == gate_total
    return {
        "verified": verified,
        "expert_params_total": expert_total,
        "expert_params_on_device": expert_on_device,
        "gate_params_total": gate_total,
        "gate_params_on_device": gate_on_device,
    }


def install_transient_expert_forward(model: Any) -> int:
    """Patch every experts module with the transient-copy forward.

    Returns the number of patched modules. Zero patches on a base that the
    placement census just proved has experts is a broken dispatch and is
    refused by the caller.
    """
    import torch
    from transformers.activations import ACT2FN

    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    hidden_act = getattr(text_config, "hidden_act", None) or "silu"
    try:
        act_fn = ACT2FN[hidden_act]
    except KeyError as exc:  # pragma: no cover - exotic activation names
        raise RuntimeError(
            f"the base's hidden activation {hidden_act!r} is not one transformers "
            "exposes; the transient experts forward cannot be built honestly"
        ) from exc

    import torch.nn.functional as F

    def transient_expert_forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        dev = hidden_states.device
        for hit in expert_hit:
            expert_idx = int(hit[0])
            if expert_idx == self.num_experts:
                continue
            with torch.no_grad():
                w_gu = self.gate_up_proj[expert_idx].to(dev, non_blocking=True)
                w_dn = self.down_proj[expert_idx].to(dev, non_blocking=True)
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, w_gu).chunk(2, dim=-1)
            current_hidden = act_fn(gate) * up
            current_hidden = F.linear(current_hidden, w_dn)
            current_hidden = current_hidden * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0, token_idx, current_hidden.to(final_hidden_states.dtype)
            )
        return final_hidden_states

    patched = 0
    for module in model.modules():
        if type(module).__name__ in ("Qwen3MoeExperts", "Qwen3_5MoeExperts"):
            module.forward = transient_expert_forward.__get__(module)
            patched += 1
    return patched


def load_with_policy(
    base_model_dir: str,
    *,
    load_policy: str,
    device: str,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load the base + tokenizer under the declared policy, with evidence.

    Returns ``(model, tokenizer, report)`` where the report carries the policy
    name, dtype, placement census (for the amended policy), and the patched
    expert-module count. Raises on any contradiction the policy forbids.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from .router_healing import QUALIFIED_DEVICES

    if load_policy not in LOAD_POLICIES:
        raise ValueError(f"unknown load policy {load_policy!r}; expected one of {list(LOAD_POLICIES)}")
    device_str = device.replace(":0", "")
    if device_str not in QUALIFIED_DEVICES:
        raise ValueError(f"device {device!r} is not a qualified device")

    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, local_files_only=True)
    report: dict[str, Any] = {"policy": load_policy}

    if load_policy == "fp32-resident":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_dir, dtype=torch.float32, local_files_only=True
        )
        model.to(device)
        report["dtype"] = "torch.float32"
        return model, tokenizer, report

    # bf16-offload-transient
    cfg = AutoConfig.from_pretrained(base_model_dir, local_files_only=True)
    with torch.device("meta"):
        model_meta = AutoModelForCausalLM.from_config(cfg)
    param_names = [n for n, _ in model_meta.named_parameters()]
    del model_meta

    device_map = resolve_load_placement(base_model_dir, param_names, device=device_str)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        dtype=torch.bfloat16,
        device_map=device_map,
        local_files_only=True,
    )
    census = verify_placement_census(model, device_type=device_str.split(":")[0])
    if not census["verified"]:
        raise RuntimeError(
            "the amended load policy's placement census failed on the live model: "
            f"{json.dumps(census)}. An offload that did not happen must not be "
            "believed because the code path asked for it."
        )
    patched = install_transient_expert_forward(model)
    if patched == 0:
        raise RuntimeError(
            "the placement census found expert parameters but no experts module "
            "could be patched for the transient forward; refusing a run whose "
            "forward would hit the measured device-mismatch failure"
        )
    report.update(
        {
            "dtype": "torch.bfloat16",
            "placement_census": census,
            "patched_expert_modules": patched,
            "meta_param_count": len(param_names),
        }
    )
    return model, tokenizer, report
