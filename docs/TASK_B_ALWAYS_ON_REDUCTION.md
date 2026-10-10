# Task B — Always-On Reduction Experiment

**Status**: Design phase (2026-09-16)
**Depends on**: Rung-4 prereg merged (PR #165)
**North star**: ≤10B-total / ≤3.5B-active 9B successor architecture

## Problem

The current HotCore artifact (Qwen3.8-9B-HotCore-CW-E16-k2-h2176) has:
- **Total text**: 6.120B (measured)
- **Active text**: 3.800B (measured)
- **Always-on floor**: 2.896B (embedding 1.017B + linear attn 1.409B + full attn 0.470B)
- **Vision**: ~0.304B
- **Grand total**: ~6.423B

The active parameters (3.800B) **exceed the 3.5B target** by 0.300B. The always-on floor alone is 2.896B, leaving only 0.604B budget for routed FFN (currently 0.904B active).

## Actual HotCore MoE Configuration

```json
{
  "hidden_size": 4096,
  "intermediate_size": 12288,
  "num_experts": 16,
  "num_experts_per_tok": 2,
  "moe_intermediate_size": 632,
  "shared_expert_intermediate_size": 2176,
  "head_dim": 256,
  "num_attention_heads": 16,
  "num_key_value_heads": 4,
  "vocab_size": 248320,
  "num_hidden_layers": 32,
  "full_attention_interval": 4
}
```

## Reduction Strategy

**Goal**: Reduce active parameters from 3.800B to ≤3.5B (savings: 0.300B)

### Option A: Reduce hidden dimension (4096 → 3072)

| Component | Before (hidden=4096) | After (hidden=3072) | Savings |
|-----------|---------------------|---------------------|---------|
| Embedding | 1.017B | 0.763B | 0.254B |
| Linear attn (24 layers) | 1.409B | 0.793B | 0.616B |
| Full attn (8 layers) | 0.470B | 0.264B | 0.206B |
| **Always-on** | **2.896B** | **1.820B** | **1.076B** |

With hidden=3072, keeping the same MoE config (E=16, k=2, moe_inter=632, shared_inter=2176):
- Total: 1.820B + 3.223B = **5.043B** ✓ (≤10B)
- Active: 1.820B + 0.904B = **2.724B** ✓ (≤3.5B)

**This meets both targets with significant margin.**

### Option B: Reduce head dimension (256 → 192)

If we keep hidden=4096 but reduce head_dim from 256 to 192:
- Linear attn savings: ~0.352B
- Always-on: 2.896B - 0.352B = 2.544B
- Active: 2.544B + 0.904B = 3.448B ✓

**This also meets the target, but with less margin.**

### Option C: Combined reduction

hidden=3072 + head_dim=192:
- Always-on: ~1.550B
- Active: ~2.454B ✓

**Maximum margin, but more complex conversion.**

## Recommended Architecture

**hidden=3072, head_dim=192, E=16, k=2, moe_inter=632, shared_inter=2176**

| Parameter | Value |
|-----------|-------|
| Hidden size | 3072 |
| Head dim | 192 |
| Num heads | 16 |
| Num KV heads | 4 |
| Experts | 16 |
| Top-k | 2 |
| Expert intermediate | 632 |
| Shared expert intermediate | 2176 |
| **Total** | **~5.0B** |
| **Active** | **~2.7B** |

## Weight Projection Strategy

### Step 1: Embedding projection
- Parent: `embed_tokens.weight` [248320, 4096]
- Target: `embed_tokens.weight` [248320, 3072]
- Method: Truncation (keep first 3072 dims) or PCA projection

### Step 2: Attention projection
- All Q/K/V/O projections: [4096, X] → [3072, X]
- Method: Truncation (keep first 3072 rows)

### Step 3: MoE FFN creation
- Parent dense FFN: `mlp.up_proj` [4096, 12288], `mlp.down_proj` [12288, 4096]
- Target MoE FFN:
  - 16 routed experts: each [3072, 632] up, [632, 3072] down
  - 1 shared expert: [3072, 2176] up, [2176, 3072] down
  - Router: [3072, 16]
- Method: Split dense FFN into expert-sized chunks, or initialize from random

### Step 4: Linear attention projection
- Linear attention Q/K/V/O: [4096, X] → [3072, X]
- Method: Truncation

## Open Questions

1. **Embedding projection method**: Truncation vs. PCA? Truncation is simpler but may lose information. PCA requires a calibration corpus.
2. **FFN initialization**: Split dense FFN into expert chunks, or initialize from random? Splitting preserves more information but may create correlated experts.
3. **Shared expert size**: Keep shared_inter=2176, or adjust?
4. **Head dim**: Keep 256 (simpler) or reduce to 192 (more savings)?

## Next Steps

1. Write rung-5 preregistration for the reduced architecture
2. Implement the dense→reduced MoE conversion
3. Execute the training run
4. Evaluate against rung-4 baseline
