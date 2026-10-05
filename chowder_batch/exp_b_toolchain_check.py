"""Experiment B Phase 3: verify Granite's training toolchain on a tiny model.

This deliberately uses a random, tiny ``GraniteMoeHybridForCausalLM`` rather
than downloading the full checkpoint. It is a compatibility probe, not proof
that the full Granite H Tiny checkpoint fits or trains successfully.

The report distinguishes exact LoRA coverage of attention/Mamba projections,
PEFT ``target_parameters`` coverage of fused MoE expert tensors, per-expert
gradient and routing coverage, Mamba gradient flow, and a real bitsandbytes
4-bit CUDA linear plus trainable low-rank branch. The last item proves only
that the installed primitive can backprop on this GPU, not full-model QLoRA.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def build_tiny_config():
    from transformers import GraniteMoeHybridConfig

    return GraniteMoeHybridConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        shared_intermediate_size=128,
        # These are the checkpoint/config layer labels. Transformers maps
        # linear_attention -> .mamba and full_attention -> .self_attn modules.
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        max_position_embeddings=512,
        tie_word_embeddings=False,
        # Small Mamba dimensions avoid the library-default 17 GB scan workspace.
        mamba_n_heads=8,
        mamba_d_state=16,
        mamba_d_head=16,
        mamba_expand=2,
        mamba_n_groups=1,
        mamba_chunk_size=16,
    )


def make_model():
    from transformers import GraniteMoeHybridForCausalLM

    torch.manual_seed(0)
    model = GraniteMoeHybridForCausalLM(build_tiny_config())
    model.train()
    return model


def forward_backward_probe(model, config) -> dict:
    """Full fp32 forward/backward including loss on shifted labels."""
    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (2, 16))
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    gradient_rows = [
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    with_gradient = sum(gradient is not None for _, gradient in gradient_rows)
    nonzero_gradient = sum(
        gradient is not None and bool(torch.count_nonzero(gradient.detach()).item())
        for _, gradient in gradient_rows
    )
    missing = [name for name, gradient in gradient_rows if gradient is None]
    return {
        "loss": float(loss.detach()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "trainable_parameter_tensors": len(gradient_rows),
        "parameter_tensors_with_gradient": with_gradient,
        "parameter_tensors_with_nonzero_gradient": nonzero_gradient,
        "all_trainable_parameters_have_gradient": bool(gradient_rows) and not missing,
        "missing_gradient_parameters": missing,
    }


def _normalize_peft_path(path: str) -> str:
    for prefix in ("base_model.model.", "base_model."):
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def _lora_base_paths(model) -> set[str]:
    paths = set()
    for name, _param in model.named_parameters():
        if ".lora_A" in name:
            paths.add(_normalize_peft_path(name.split(".lora_A", 1)[0]))
    return paths


def lora_module_probe(model) -> dict:
    """Apply module-based LoRA to Linear attention/Mamba projections."""
    from peft import LoraConfig, get_peft_model

    candidates = [
        "q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj",
        "input_linear", "output_linear",
    ]
    original_modules = dict(model.named_modules())
    expected_paths = {
        candidate: sorted(
            name for name in original_modules if name.rsplit(".", 1)[-1] == candidate
        )
        for candidate in candidates
    }
    fused_expert_parameters = sorted(
        name
        for name, _parameter in model.named_parameters()
        if name.rsplit(".", 1)[-1] in {"gate_up_proj", "down_proj"}
        and ".block_sparse_moe.experts." in name
    )
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=candidates,
        ),
    )

    torch.manual_seed(1)
    ids = torch.randint(0, peft_model.config.vocab_size, (4, 16))
    loss = peft_model(input_ids=ids, labels=ids).loss
    loss.backward()

    adapted_paths = _lora_base_paths(peft_model)
    target_coverage = {}
    all_present_targets_covered = True
    for candidate, paths in expected_paths.items():
        adapted = [path for path in paths if path in adapted_paths]
        target_coverage[candidate] = {
            "modules_present": len(paths),
            "modules_adapted": len(adapted),
            "adapted_paths": adapted,
        }
        if paths and len(adapted) != len(paths):
            all_present_targets_covered = False

    adapter_b = [
        (name, param)
        for name, param in peft_model.named_parameters()
        if ".lora_B." in name and param.requires_grad
    ]
    adapter_b_by_path = {
        _normalize_peft_path(name.split(".lora_B", 1)[0]): param
        for name, param in adapter_b
    }
    for coverage in target_coverage.values():
        paths = coverage["adapted_paths"]
        coverage["modules_with_gradient"] = sum(
            path in adapter_b_by_path and adapter_b_by_path[path].grad is not None
            for path in paths
        )
        coverage["modules_with_nonzero_b_gradient"] = sum(
            path in adapter_b_by_path
            and adapter_b_by_path[path].grad is not None
            and bool(torch.count_nonzero(adapter_b_by_path[path].grad.detach()).item())
            for path in paths
        )
    adapter_b_with_grad = sum(param.grad is not None for _, param in adapter_b)
    adapter_b_nonzero = sum(
        param.grad is not None and bool(torch.count_nonzero(param.grad.detach()).item())
        for _, param in adapter_b
    )
    all_adapted_targets_have_gradient = all(
        path in adapter_b_by_path and adapter_b_by_path[path].grad is not None
        for path in adapted_paths
    )
    all_adapted_targets_have_nonzero_b_gradient = all(
        path in adapter_b_by_path
        and adapter_b_by_path[path].grad is not None
        and bool(torch.count_nonzero(adapter_b_by_path[path].grad.detach()).item())
        for path in adapted_paths
    )
    if not adapted_paths:
        all_adapted_targets_have_gradient = False
        all_adapted_targets_have_nonzero_b_gradient = False
    if not all_present_targets_covered or not all_adapted_targets_have_gradient:
        raise RuntimeError("LoRA target coverage or gradient probe was incomplete")
    fused_experts_adapted_by_module = [
        name for name in fused_expert_parameters if name in adapted_paths
    ]

    total_params = sum(param.numel() for param in peft_model.parameters())
    trainable_params = sum(
        param.numel() for param in peft_model.parameters() if param.requires_grad
    )
    return {
        "target_coverage": target_coverage,
        "all_present_targets_covered": all_present_targets_covered,
        "fused_expert_parameter_paths": fused_expert_parameters,
        "fused_expert_parameters_adapted_by_target_modules": fused_experts_adapted_by_module,
        "fused_expert_parameters_skipped_by_target_modules": sorted(
            set(fused_expert_parameters) - set(fused_experts_adapted_by_module)
        ),
        "fused_expert_modules_adapted_by_target_modules": [],
        "fused_expert_parameters_adapted_by_target_parameters": False,
        "all_adapted_targets_have_gradient": all_adapted_targets_have_gradient,
        "all_adapted_targets_have_nonzero_b_gradient": all_adapted_targets_have_nonzero_b_gradient,
        "adapter_b_tensors": len(adapter_b),
        "adapter_b_with_gradient": adapter_b_with_grad,
        "adapter_b_with_nonzero_gradient": adapter_b_nonzero,
        "loss_finite": bool(torch.isfinite(loss.detach()).item()),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_fraction": round(trainable_params / max(total_params, 1), 6),        "no_silent_skip": all_present_targets_covered,
    }


def moe_expert_lora_probe(model) -> dict:
    """Probe PEFT LoRA for the model's fused rank-3 MoE expert parameters."""
    from peft import LoraConfig, get_peft_model

    target_names = [
        name
        for name, _param in model.named_parameters()
        if ".block_sparse_moe.experts." in name
        and name.rsplit(".", 1)[-1] in {"gate_up_proj", "down_proj"}
    ]
    if not target_names:
        return {"supported": False, "no_silent_skip": False, "error": "no fused expert parameters found"}
    try:
        peft_model = get_peft_model(
            model,
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                target_modules=[],
                target_parameters=target_names,
            ),
        )
        torch.manual_seed(2)
        ids = torch.randint(0, peft_model.config.vocab_size, (8, 32))
        loss = peft_model(input_ids=ids, labels=ids).loss
        loss.backward()
        trainable = [
            (name, param)
            for name, param in peft_model.named_parameters()
            if param.requires_grad and ("lora_" in name or "lora_magnitude" in name)
        ]
        with_gradient = sum(param.grad is not None for _, param in trainable)
        nonzero = sum(
            param.grad is not None and bool(torch.count_nonzero(param.grad.detach()).item())
            for _, param in trainable
        )
        # PEFT wraps rank-3 tensors in nested ParamWrappers; kwargs.target_name
        # preserves the original parameter name exactly at the A/B factors.
        adapters = {}
        for _module_name, module in peft_model.named_modules():
            target_name = getattr(module, "kwargs", {}).get("target_name")
            if not target_name or not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                continue
            normalized = _normalize_peft_path(str(target_name))
            if normalized in target_names:
                adapters[normalized] = module
        missing = sorted(set(target_names) - set(adapters))
        per_target = {}
        for name in target_names:
            module = adapters.get(name)
            if module is None:
                per_target[name] = {"adapted": False, "a_gradient": False, "b_gradient": False, "b_nonzero_gradient": False}
                continue
            a_grad = [p.grad for p in module.lora_A["default"].parameters()]
            b_grad = [p.grad for p in module.lora_B["default"].parameters()]
            per_target[name] = {
                "adapted": True,
                "a_gradient": any(grad is not None for grad in a_grad),
                "b_gradient": any(grad is not None for grad in b_grad),
                "b_nonzero_gradient": any(
                    grad is not None and bool(torch.count_nonzero(grad.detach()).item())
                    for grad in b_grad
                ),
            }
        covered = not missing
        gradients_present = all(
            row["adapted"] and row["a_gradient"] and row["b_gradient"]
            for row in per_target.values()
        )
        return {
            "supported": covered and gradients_present and bool(trainable) and bool(torch.isfinite(loss).item()),
            "target_parameter_count": len(target_names),
            "target_parameters": target_names,
            "adapted_parameter_paths": sorted(adapters),
            "unadapted_parameter_paths": missing,
            "per_target_adapter_gradients": per_target,
            "all_targets_have_adapter_gradients": gradients_present,
            "targets_with_nonzero_b_gradient": sum(row["b_nonzero_gradient"] for row in per_target.values()),
            "trainable_adapter_tensors": len(trainable),
            "adapter_tensors_with_gradient": with_gradient,
            "adapter_tensors_with_nonzero_gradient": nonzero,
            "loss_finite": bool(torch.isfinite(loss.detach()).item()),
            "no_silent_skip": covered,
        }
    except Exception as error:
        return {
            "supported": False,
            "target_parameter_count": len(target_names),
            "target_parameters": target_names,
            "all_targets_have_adapter_gradients": False,
            "no_silent_skip": False,
            "error": f"{type(error).__name__}: {str(error)[:500]}",
        }


def moe_gradient_and_dispatch_probe(model) -> dict:
    """Measure gradients and observed router assignments per expert."""
    model.zero_grad(set_to_none=True)
    by_layer: dict[str, dict[int, dict[str, float | None]]] = {}
    dispatch_counts = {
        f"model.layers.{index}": torch.zeros(model.config.num_local_experts, dtype=torch.long)
        for index, layer in enumerate(model.model.layers)
        if hasattr(layer, "block_sparse_moe")
    }
    hooks = []
    for index, layer in enumerate(model.model.layers):
        if not hasattr(layer, "block_sparse_moe"):
            continue
        layer_name = f"model.layers.{index}"

        def capture_dispatch(_module, args, *, name=layer_name):
            top_k_index = args[1]
            dispatch_counts[name].add_(
                torch.bincount(
                    top_k_index.detach().reshape(-1).cpu(),
                    minlength=model.config.num_local_experts,
                )
            )

        hooks.append(layer.block_sparse_moe.experts.register_forward_pre_hook(capture_dispatch))
    try:
        torch.manual_seed(3)
        ids = torch.randint(0, model.config.vocab_size, (8, 32))
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
    finally:
        for hook in hooks:
            hook.remove()

    for name, param in model.named_parameters():
        if ".block_sparse_moe.experts." not in name:
            continue
        if name.rsplit(".", 1)[-1] not in {"gate_up_proj", "down_proj"}:
            continue
        layer = name.split(".block_sparse_moe.", 1)[0]
        family = name.rsplit(".", 1)[-1]
        by_layer.setdefault(layer, {})
        for expert_index in range(param.shape[0]):
            grad = param.grad
            norm = float(grad[expert_index].detach().norm()) if grad is not None else None
            by_layer[layer].setdefault(expert_index, {})[family] = norm

    layer_reports = {
        layer: {
            str(index): {
                "gate_up_proj_grad_norm": values.get("gate_up_proj"),
                "down_proj_grad_norm": values.get("down_proj"),
                "touched": any(values.get(key) is not None and values[key] > 0.0 for key in ("gate_up_proj", "down_proj")),
            }
            for index, values in sorted(experts.items())
        }
        for layer, experts in by_layer.items()
    }
    total_experts = sum(len(experts) for experts in by_layer.values())
    touched_experts = sum(details["touched"] for experts in layer_reports.values() for details in experts.values())
    dispatch_report = {
        layer: {
            "per_expert_token_assignments": counts.tolist(),
            "experts_with_assignments": int(counts.gt(0).sum().item()),
            "experts_total": len(counts),
        }
        for layer, counts in dispatch_counts.items()
    }
    router_report = {}
    shared_mlp_report = {}
    for index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}"
        router = getattr(getattr(layer, "block_sparse_moe", None), "router", None)
        if router is not None:
            router_report[prefix] = {
                name.rsplit(".", 1)[-1]: {
                    "has_gradient": parameter.grad is not None,
                    "grad_norm": float(parameter.grad.detach().norm()) if parameter.grad is not None else None,
                    "nonzero_gradient": parameter.grad is not None and bool(torch.count_nonzero(parameter.grad.detach()).item()),
                }
                for name, parameter in router.named_parameters()
            }
        shared_mlp = getattr(layer, "shared_mlp", None)
        if shared_mlp is not None:
            shared_mlp_report[prefix] = {
                name: {
                    "has_gradient": parameter.grad is not None,
                    "grad_norm": float(parameter.grad.detach().norm()) if parameter.grad is not None else None,
                    "nonzero_gradient": parameter.grad is not None and bool(torch.count_nonzero(parameter.grad.detach()).item()),
                }
                for name, parameter in shared_mlp.named_parameters()
            }

    all_router_params_have_gradient = bool(router_report) and all(
        row["has_gradient"] for layer in router_report.values() for row in layer.values()
    )
    all_shared_mlp_params_have_gradient = bool(shared_mlp_report) and all(
        row["has_gradient"] for layer in shared_mlp_report.values() for row in layer.values()
    )
    return {
        "loss_finite": bool(torch.isfinite(loss.detach()).item()),
        "router_gradients": router_report,
        "all_router_parameters_have_gradient": all_router_params_have_gradient,
        "shared_mlp_gradients": shared_mlp_report,
        "all_shared_mlp_parameters_have_gradient": all_shared_mlp_params_have_gradient,
        "expert_count_per_moe_layer": model.config.num_local_experts,
        "moe_layers": layer_reports,
        "expert_dispatch": dispatch_report,
        "experts_with_nonzero_gradient": touched_experts,
        "experts_total": total_experts,
        "all_experts_touched": total_experts > 0 and touched_experts == total_experts,
        "all_experts_received_tokens": all(
            row["experts_with_assignments"] == row["experts_total"]
            for row in dispatch_report.values()
        ),
    }


def mamba_gradient_probe(model) -> dict:
    """Confirm Granite's ``mamba`` recurrent block parameters receive gradients."""
    model.zero_grad(set_to_none=True)
    torch.manual_seed(4)
    ids = torch.randint(0, model.config.vocab_size, (2, 16))
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    gradients = [
        (name, float(param.grad.detach().norm()))
        for name, param in model.named_parameters()
        if ".mamba." in name and param.grad is not None
    ]
    nonzero = [(name, norm) for name, norm in gradients if norm > 0.0]
    return {
        "parameter_tensors_with_gradient": len(gradients),
        "parameter_tensors_with_nonzero_gradient": len(nonzero),
        "sample_nonzero": nonzero[:4],
        "supported": bool(nonzero),
    }


def qlora_probe() -> dict:
    """Test a real 4-bit CUDA linear primitive, not full-model QLoRA."""
    available = importlib.util.find_spec("bitsandbytes") is not None
    cuda = torch.cuda.is_available()
    report = {
        "bitsandbytes_importable": available,
        "cuda_available": cuda,
        "four_bit_lora_primitive_ready": False,
        "full_granite_qlora_verified": False,
        "note": "A tiny primitive test does not prove full-checkpoint Granite QLoRA compatibility.",
    }
    if not available or not cuda:
        return report
    try:
        import bitsandbytes as bnb

        free_by_device = []
        for index in range(torch.cuda.device_count()):
            candidate = torch.device("cuda", index)
            free_bytes, _total_bytes = torch.cuda.mem_get_info(candidate)
            free_by_device.append((free_bytes, candidate))
        free_by_device.sort(reverse=True, key=lambda row: row[0])
        first_error = None
        for free_bytes, device in free_by_device:
            try:
                layer = bnb.nn.Linear4bit(
                    64, 64, bias=False, compute_dtype=torch.float32,
                    compress_statistics=False, quant_type="nf4",
                ).to(device)
                for param in layer.parameters():
                    param.requires_grad_(False)
                lora_a = torch.nn.Parameter(torch.randn(4, 64, device=device) * 0.02)
                lora_b = torch.nn.Parameter(torch.zeros(64, 4, device=device))
                inputs = torch.randn(2, 8, 64, device=device, requires_grad=True)
                loss = (layer(inputs) + inputs @ lora_a.T @ lora_b.T).float().square().mean()
                loss.backward()
                break
            except Exception as error:
                first_error = first_error or error
                torch.cuda.empty_cache()
        else:
            raise first_error or RuntimeError("no CUDA device available for 4-bit probe")
        report.update({
            "device": torch.cuda.get_device_name(device),
            "free_vram_before_probe_bytes": free_bytes,
            "bitsandbytes_version": bnb.__version__,
            "four_bit_lora_primitive_ready": bool(
                torch.isfinite(loss).item() and lora_a.grad is not None
                and lora_b.grad is not None and torch.count_nonzero(lora_b.grad).item() > 0
            ),
            "loss_finite": bool(torch.isfinite(loss).item()),
            "lora_a_gradient": lora_a.grad is not None,
            "lora_b_nonzero_gradient": bool(lora_b.grad is not None and torch.count_nonzero(lora_b.grad).item() > 0),
            "base_weight_trainable": any(param.requires_grad for param in layer.parameters()),
        })
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {str(error)[:500]}"
    finally:
        if "layer" in locals():
            del layer
        if "lora_a" in locals():
            del lora_a
        if "lora_b" in locals():
            del lora_b
        if "inputs" in locals():
            del inputs
        if "loss" in locals():
            del loss
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = build_tiny_config()
    report = {
        "architecture": "random tiny GraniteMoeHybridForCausalLM (not the full checkpoint)",
        "tiny_config": {
            "hidden_size": config.hidden_size,
            "layers": config.num_hidden_layers,
            "local_experts": config.num_local_experts,
            "experts_per_token": config.num_experts_per_tok,
            "layer_types": config.layer_types,
        },
    }
    model = make_model()
    report["forward_backward"] = forward_backward_probe(model, config)
    del model
    gc.collect()
    model = make_model()
    try:
        report["lora_modules"] = lora_module_probe(model)
    finally:
        del model
        gc.collect()
    model = make_model()
    report["lora_moe_expert_parameters"] = moe_expert_lora_probe(model)
    del model
    gc.collect()
    model = make_model()
    report["moe_expert_gradients"] = moe_gradient_and_dispatch_probe(model)
    del model
    gc.collect()
    model = make_model()
    report["mamba_gradients"] = mamba_gradient_probe(model)
    del model
    gc.collect()
    report["qlora"] = qlora_probe()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
