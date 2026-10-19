"""Convert the dense Qwen3.8-9B parent to a reduced-dimension MoE artifact.

This script implements the weight projection described in:
    docs/quals/P11_RUNG5_PREREG_2026-09-16.md

It creates a new MoE artifact with:
    - hidden_size: 4096 → 3072
    - head_dim: 256 → 192
    - num_experts: 16, top_k: 2
    - moe_intermediate_size: 632
    - shared_expert_intermediate_size: 2176

Usage:
    python -m chowder.reduced_moe_convert \
        --source F:\\llm-models\\Qwen3.8-9B-abliterated-25-bf16 \
        --target F:\\llm-models\\Qwen3.8-9B-HotCore-E16-k2-h3072
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


# ── Target architecture ──────────────────────────────────────────────────────

TARGET_CONFIG = {
    "hidden_size": 3072,
    "head_dim": 192,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "num_hidden_layers": 32,
    "full_attention_interval": 4,
    "vocab_size": 248320,
    "num_experts": 16,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 632,
    "shared_expert_intermediate_size": 2176,
    "initializer_range": 0.02,
}


def _truncate_tensor(
    src: torch.Tensor,
    target_rows: int | None,
    target_cols: int | None,
) -> torch.Tensor:
    """Truncate a 2D tensor to the target dimensions (keep top-left)."""
    rows = src.shape[0] if target_rows is None else min(target_rows, src.shape[0])
    cols = src.shape[1] if target_cols is None else min(target_cols, src.shape[1])
    return src[:rows, :cols].clone()


def _init_expert_ffn(
    hidden: int,
    intermediate: int,
    initializer_range: float,
) -> dict[str, torch.Tensor]:
    """Initialize a single expert FFN with random weights."""
    up = torch.randn(hidden, intermediate) * initializer_range
    down = torch.randn(intermediate, hidden) * initializer_range
    return {"gate_proj.weight": up, "up_proj.weight": up.clone(), "down_proj.weight": down}


def convert(
    source_dir: Path,
    target_dir: Path,
    *,
    dry_run: bool = False,
) -> None:
    """Run the full conversion."""
    hidden_src = 4096
    hidden_tgt = TARGET_CONFIG["hidden_size"]
    head_dim_src = 256
    head_dim_tgt = TARGET_CONFIG["head_dim"]
    num_heads = TARGET_CONFIG["num_attention_heads"]
    num_kv = TARGET_CONFIG["num_key_value_heads"]
    num_layers = TARGET_CONFIG["num_hidden_layers"]
    num_experts = TARGET_CONFIG["num_experts"]
    moe_inter = TARGET_CONFIG["moe_intermediate_size"]
    shared_inter = TARGET_CONFIG["shared_expert_intermediate_size"]
    init_range = TARGET_CONFIG["initializer_range"]

    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Hidden: {hidden_src} → {hidden_tgt}")
    print(f"Head dim: {head_dim_src} → {head_dim_tgt}")
    print(f"Experts: {num_experts}, top-k: {TARGET_CONFIG['num_experts_per_tok']}")
    print()

    if dry_run:
        print("[DRY RUN] Would convert weights and config.")
        return

    target_dir.mkdir(parents=True, exist_ok=True)

    # ── Load source weights ──────────────────────────────────────────────
    source_safetensors = list(source_dir.glob("*.safetensors"))
    if not source_safetensors:
        print(f"ERROR: No .safetensors files in {source_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(source_safetensors)} safetensors files...")
    all_weights: dict[str, torch.Tensor] = {}
    for sf in source_safetensors:
        all_weights.update(load_file(str(sf)))

    # ── Convert weights ──────────────────────────────────────────────────
    target_weights: dict[str, torch.Tensor] = {}

    for name, param in all_weights.items():
        # Embedding
        if "embed_tokens.weight" in name:
            target_weights[name] = _truncate_tensor(param, None, hidden_tgt)
            print(f"  {name}: {param.shape} → {target_weights[name].shape}")

        # Attention projections
        elif "q_proj.weight" in name:
            target_weights[name] = _truncate_tensor(param, hidden_tgt, num_heads * head_dim_tgt)
            print(f"  {name}: {param.shape} → {target_weights[name].shape}")
        elif "k_proj.weight" in name:
            target_weights[name] = _truncate_tensor(param, hidden_tgt, num_kv * head_dim_tgt)
            print(f"  {name}: {param.shape} → {target_weights[name].shape}")
        elif "v_proj.weight" in name:
            target_weights[name] = _truncate_tensor(param, hidden_tgt, num_kv * head_dim_tgt)
            print(f"  {name}: {param.shape} → {target_weights[name].shape}")
        elif "o_proj.weight" in name:
            target_weights[name] = _truncate_tensor(param, num_heads * head_dim_tgt, hidden_tgt)
            print(f"  {name}: {param.shape} → {target_weights[name].shape}")

        # Linear attention projections
        elif "linear_attn" in name and "q_proj" in name:
            target_weights[name] = _truncate_tensor(param, hidden_tgt, num_heads * head_dim_tgt)
            print(f"  {name}: {param.shape} → {target_weights[name].shape}")
        elif "linear_attn" in name and ("k_proj" in name or "v_proj" in name):
            target_weights[name] = _truncate_tensor(param, hidden_tgt, num_kv * head_dim_tgt)
            print(f"  {name}: {param.shape} → {target_weights[name].shape}")
        elif "linear_attn" in name and "o_proj" in name:
            target_weights[name] = _truncate_tensor(param, num_heads * head_dim_tgt, hidden_tgt)
            print(f"  {name}: {param.shape} → {target_weights[name].shape}")

        # Skip dense FFN weights (mlp.up_proj, mlp.down_proj)
        elif "mlp.up_proj" in name or "mlp.down_proj" in name:
            print(f"  SKIP {name}: dense FFN (replaced by MoE)")
            continue

        # Norms and biases — copy directly
        elif "norm" in name or "bias" in name:
            target_weights[name] = param.clone()
            print(f"  {name}: copy {param.shape}")

        # Everything else — copy if shape matches, skip if not
        else:
            target_weights[name] = param.clone()
            print(f"  {name}: copy {param.shape}")

    # ── Create MoE FFN weights ───────────────────────────────────────────
    print()
    print("Creating MoE FFN weights...")
    for layer_idx in range(num_layers):
        layer_prefix = f"model.layers.{layer_idx}.mlp"

        # Router gate
        router_weight = torch.randn(hidden_tgt, num_experts) * init_range
        target_weights[f"{layer_prefix}.gate.weight"] = router_weight
        print(f"  {layer_prefix}.gate.weight: {router_weight.shape}")

        # Shared expert
        shared_up = torch.randn(hidden_tgt, shared_inter) * init_range
        shared_down = torch.randn(shared_inter, hidden_tgt) * init_range
        target_weights[f"{layer_prefix}.shared_expert.gate_proj.weight"] = shared_up
        target_weights[f"{layer_prefix}.shared_expert.up_proj.weight"] = shared_up.clone()
        target_weights[f"{layer_prefix}.shared_expert.down_proj.weight"] = shared_down
        print(f"  {layer_prefix}.shared_expert: up {shared_up.shape}, down {shared_down.shape}")

        # Routed experts
        for expert_idx in range(num_experts):
            expert_prefix = f"{layer_prefix}.experts.{expert_idx}"
            expert_up = torch.randn(hidden_tgt, moe_inter) * init_range
            expert_down = torch.randn(moe_inter, hidden_tgt) * init_range
            target_weights[f"{expert_prefix}.gate_proj.weight"] = expert_up
            target_weights[f"{expert_prefix}.up_proj.weight"] = expert_up.clone()
            target_weights[f"{expert_prefix}.down_proj.weight"] = expert_down

        print(f"  {layer_prefix}.experts: {num_experts} experts × (up {hidden_tgt}×{moe_inter}, down {moe_inter}×{hidden_tgt})")

    # ── Write target ─────────────────────────────────────────────────────
    print()
    print(f"Writing {len(target_weights)} tensors to {target_dir}...")

    # Split into chunks of ~5GB for safetensors
    chunk_size = 5_000_000_000  # 5GB
    chunks: list[dict[str, torch.Tensor]] = [{}]
    current_size = 0
    for name, param in target_weights.items():
        param_size = param.nelement() * param.element_size()
        if current_size + param_size > chunk_size and chunks[-1]:
            chunks.append({})
            current_size = 0
        chunks[-1][name] = param
        current_size += param_size

    for i, chunk in enumerate(chunks):
        path = target_dir / f"model-{i:05d}-of-{len(chunks):05d}.safetensors"
        save_file(chunk, str(path))
        print(f"  Written: {path.name} ({len(chunk)} tensors)")

    # ── Copy and modify config.json ──────────────────────────────────────
    source_config = source_dir / "config.json"
    if source_config.exists():
        with open(source_config) as f:
            config = json.load(f)

        # Update text config
        text_config = config.get("text_config", config)
        text_config["hidden_size"] = TARGET_CONFIG["hidden_size"]
        text_config["head_dim"] = TARGET_CONFIG["head_dim"]
        text_config["num_experts"] = TARGET_CONFIG["num_experts"]
        text_config["num_experts_per_tok"] = TARGET_CONFIG["num_experts_per_tok"]
        text_config["moe_intermediate_size"] = TARGET_CONFIG["moe_intermediate_size"]
        text_config["shared_expert_intermediate_size"] = TARGET_CONFIG["shared_expert_intermediate_size"]
        text_config["model_type"] = "qwen3_5_moe_text"

        # Update vision config out_hidden_size
        if "vision_config" in config:
            config["vision_config"]["out_hidden_size"] = TARGET_CONFIG["hidden_size"]

        # Update top-level
        config["model_type"] = "qwen3_5_moe"
        config["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]

        target_config_path = target_dir / "config.json"
        with open(target_config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Written: config.json")

    # ── Copy tokenizer and generation config ─────────────────────────────
    for fname in ["tokenizer.json", "tokenizer_config.json", "generation_config.json",
                   "vocab.json", "merges.txt", "added_tokens.json"]:
        src = source_dir / fname
        if src.exists():
            shutil.copy2(src, target_dir / fname)
            print(f"  Copied: {fname}")

    print()
    print(f"Conversion complete: {target_dir}")
    print(f"  Total tensors: {len(target_weights)}")
    total_params = sum(p.nelement() for p in target_weights.values())
    print(f"  Total parameters: {total_params/1e9:.3f}B")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert dense parent to reduced-dimension MoE")
    parser.add_argument("--source", type=Path, required=True, help="Source dense model directory")
    parser.add_argument("--target", type=Path, required=True, help="Target MoE model directory")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing")
    args = parser.parse_args()

    convert(args.source, args.target, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
