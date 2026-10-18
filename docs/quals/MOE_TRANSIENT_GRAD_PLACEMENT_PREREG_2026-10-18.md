# Pre-registration — `expert_placement: cpu-transient-grad` (2026-10-18)

**What this registers.** A worker placement for MoE bases. Routed experts
stay frozen on the CPU and are copied to the GPU per forward and per
backward recompute (`chowder.backends.moe_transient_grad`). LoRA covers
attention, Gated DeltaNet and the shared expert
(`lora.target_preset: moe_frozen_experts`).

It exists so that the REAP-128 Qwen3.6 MoE (18.5B parameters, 16.1B of them
in routed experts) can train on the 16 GB RTX 5060 Ti.

**Config.**
```yaml
backend:
  expert_placement: cpu-transient-grad
  quantization: none
  lora:
    target_preset: moe_frozen_experts
```
The spec refuses any other preset, explicit `target_modules`, or 4-bit
quantization with this placement. The worker refuses it without CUDA. The
live placement census is written to `telemetry.expert_placement`.

**Rule.** No growth campaign may declare this placement until every gate
below has been measured on the real reap128 artifact and recorded here.

| Gate | Pass | Status |
|---|---|---|
| Exactness | On a tiny `qwen3_5_moe`, loss and every gradient match the resident model | **met in tests**: CPU, and CUDA with CPU experts; largest difference 6e-8 |
| Nothing held | The forward saves no tensor from expert storage | **met in tests** (the stock forward saves more than zero) |
| Load census | On reap128, every routed-expert parameter is on the CPU and every other parameter is on the GPU | not measured |
| Gradient | One step on reap128 gives finite, non-zero gradients on every LoRA B leaf, in layers 0, 19 and 39 | not measured |
| Memory | Peak VRAM ≤ 14.5 GB at seq 2048, with no WDDM shared-memory spill | not measured |
| Throughput | Measured s/step is recorded into `chowder.model-lifecycle.v1`; go/no-go is whether a 1,000-example recipe fits the 2.5 GPU-h per-recipe ceiling | not measured |
| Smoke | 20 steps lower the training loss, and the adapter loads in the eval worker | not measured |
| Router balance | Expert-load entropy stays within 20% of its step-0 value over the smoke run | not measured |

**Recipe identity.** `expert_placement` is excluded from `recipe_digest()`,
like `frozen_layer_streaming`. It changes where frozen weights live, never
what is computed; the exactness gate is the evidence.

Design: `docs/DESIGN-MOE-16GB-TRAINING-BACKEND-2026-10-18.md` in the hub docs,
option A.
