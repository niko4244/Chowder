# Chowder Experiment D — Low-Active-Compute Hybrid Language Model

**Status: isolated CPU research prototype; no quality, throughput, GPU-memory, HBM, or training-cost benefit has been demonstrated.** This work does not alter campaign configuration, production model paths, the existing run registry, or an active training process. The `examples/experiment_d/configs/registry.json` is a standalone design/artifact index; artifacts are written only by the offline runner to caller-selected output paths.

## Reference architectures inspected

These references provide architectural implementation examples, not evidence that Chowder's proposed combination will train or perform well. Config shapes below were read from the public Hugging Face `config.json` files in September 2026; model-card and source URLs are recorded for repeatable checks. Config values describe model geometry and do not by themselves establish a layer's active FLOPs or measured resident memory.

| Reference | Public configuration / architecture facts | License note |
|---|---|---|
| **IBM Granite 4.0 H Tiny Base** — [config](https://huggingface.co/ibm-granite/granite-4.0-h-tiny-base/raw/main/config.json), [model card](https://huggingface.co/ibm-granite/granite-4.0-h-tiny-base/raw/main/README.md), [implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/granitemoehybrid/modeling_granitemoehybrid.py), [IBM code repo](https://github.com/ibm-granite/granite-4.0-language-models) | 40 decoder layers, hidden 1536, attention 12 Q / 4 KV heads, config's exact `layer_types` gives 4 attention + 36 Mamba layers. Mamba geometry: 48 heads × 64 = 3072 inner (2× hidden), d_state 128, one group, causal-conv width 4, chunk 256. Local experts: 64, top-6, shared intermediate 1024, router auxiliary coefficient 0.0, shared input/output embeddings. SiLU, RMSNorm, configured attention position embedding `nope`, residual multiplier 0.22. Model card comparison table reports 7B total / 1B active. Model card identifies GQA, Mamba2, MoE with shared experts and SwiGLU. The official Transformers implementation uses separate attention KV cache and Mamba convolution/recurrent state; it has fused Mamba-2 scan/selective updates and an explicit top-k router/experts implementation. | Checkpoint and IBM Granite 4.0 code repository: Apache-2.0 (official model card/repository). Respect notices and pinned source revisions when reusing code. |
| **Qwen3.5-35B-A3B** — [config](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/config.json), [model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/README.md), [implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py) | Text config: hidden 2048, 40 layers, `layer_types` repeats three linear-attention layers then one full-attention layer (30/10), 16 Q / 2 KV heads at head_dim 256, linear-attention 16 QK / 32 V heads with dims 128, MoE intermediate 512, 256 experts, top-8 plus one shared expert, vocab 248,320, router auxiliary coefficient 0.001. Model card describes Gated DeltaNet + Gated Attention and 35B total / 3B activated. The Transformers implementation maintains a causal depthwise convolution state and recurrent delta-rule state on linear-attention layers; attention uses causal KV caching. MoE router selects top-k and fused experts gather selected assignments; shared expert is an additional always-on path. | This checkpoint's model card and LICENSE identify Apache-2.0. Pin the *checkpoint's* license/revision; do not infer licensing across every Qwen family release. |
| **Gemma 4 E2B / E4B** — [E2B config](https://huggingface.co/google/gemma-4-E2B/resolve/main/config.json), [E4B config](https://huggingface.co/google/gemma-4-E4B/resolve/main/config.json), [model card](https://huggingface.co/google/gemma-4-E2B/raw/main/README.md), [Transformers docs](https://huggingface.co/docs/transformers/model_doc/gemma4), [implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py) | E2B: 35 layers, hidden 1536, 8 Q / 1 KV head, sliding window 512, full attention every fifth layer per config, PLE dimension 256, vocab and per-layer input vocab both 262,144, tie embeddings, double-wide MLP. E4B: 42 layers, hidden 2560, 8 Q / 2 KV heads, full attention every sixth layer, PLE dim 256, same vocab sizes, tie embeddings, standard-width MLP. Transformers docs specify PLE as a token-identity lookup plus projected context-aware signal, summed and scaled by `1/sqrt(2)` before each decoder layer; config `vocab_size_per_layer_input` and `hidden_size_per_layer_input` define the table geometry. Model card lists 5.1B total / 2.3B effective for E2B and 8B total / 4.5B effective for E4B. E2B and E4B use a dense text backbone (no MoE in the supplied text config). | Gemma 4 model card identifies Apache-2.0; Google's current Gemma 4 license endpoint serves Apache-2.0. Older Gemma generations have separate Gemma Terms of Use: do not substitute those terms or labels for Gemma 4's license. |
| **OLMoE-1B-7B-0125** — [config](https://huggingface.co/allenai/OLMoE-1B-7B-0125/resolve/main/config.json), [model card](https://huggingface.co/allenai/OLMoE-1B-7B-0125/raw/main/README.md), [paper/repository](https://github.com/allenai/OLMoE) | Config: hidden 2048, 16 decoder layers, 16 attention heads, 64 experts, top-8, expert intermediate 1024, vocab 50,304, RMSNorm, SiLU, no shared expert field in the published base config, `norm_topk_prob=false`, auxiliary router coefficient 0.01, full causal-attention backbone (not hybrid Mamba). Model card reports 1.3B active / 6.9B total and published OLMES-style comparison scores. Repository documents open pretraining, SFT and evaluation data/code. | OLMoE model card/repository code and weights use Apache-2.0. Use the selected checkpoint's LICENSE/README rather than assume every AI2 release has identical terms. |

**Parameter/FLOP/memory comparison discipline:** official model-card headline counts (e.g. Granite H Tiny 7B/1B active; Qwen3.5-35B-A3B 35B/3B active; Gemma E2B 5.1B stored including PLE/2.3B effective; E4B 8B/4.5B effective; OLMoE-0125 6.9B/1.3B active) are source-reported conventions, not re-counted here from weight headers. Resident memory below is a *derived BF16 weight-only floor*, `2 × total parameters`, excluding all quantization metadata, non-parameter buffers, allocator/workspace, KV/state cache, runtime, and fragmentation. Decimal GB uses 10^9 bytes; GiB uses 2^30. It is not measured resident memory.

| Reference | Source-reported total / active or effective | Derived BF16 parameter-storage floor | FLOPs per token |
|---|---:|---:|---|
| Granite 4.0 H Tiny | 7B / 1B active (IBM model-card comparison table) | 14.0 GB (13.04 GiB) | Not computed from incomplete model card/config; route/layer-specific tensors and output treatment need exact header accounting. |
| Qwen3.5-35B-A3B | 35B / 3B active (Qwen model card) | 70.0 GB (65.19 GiB) | Not computed here. Attention context, Gated DeltaNet scan, dense/shared/router, vocabulary head and selected expert MACs need a matched formula/workload. |
| Gemma 4 E2B | 5.1B stored incl PLE / 2.3B effective | 10.2 GB (9.50 GiB) | Not computed here; the model's "effective" convention is not equivalent to an observed MAC count. |
| Gemma 4 E4B | 8B stored incl PLE / 4.5B effective | 16.0 GB (14.90 GiB) | Not computed here. |
| OLMoE-1B-7B-0125 | 6.9B / 1.3B active (AI2 model card) | 13.8 GB (12.85 GiB) | Not computed here. |

A consistent FLOP comparison requires exact checkpoint tensor inventory, output/tied embeddings, always-on shared paths, layer schedule and actual route/capacity semantics. Dense attention FLOPs depend on batch/sequence/context length; Mamba scan and fused expert kernels have implementation-specific work. No measured resident-memory or latency comparison is claimed.

### Source-level details and important distinctions

- **Granite H Tiny**: official Transformers source (Apache-2.0) defines a Mamba-2-style per-head state scan, grouped B/C state vectors, depthwise causal convolution and cache update paths alongside grouped-query causal attention. The exact config shows 36 of 40 layers type `mamba`, 4 attention; the model card reports 64 experts/top-6 routing, 7B total / 1B active, and a shared expert. Do not conflate its Mamba block's internal gated state computation with a tokenwise residual block.
- **Qwen3.5 MoE**: official Transformers source defines Gated DeltaNet's Q/K/V projections, causal convolution, learned gate/decay and beta, recurrent state updated with the delta rule, gated RMSNorm, output projection, plus a separate full-attention implementation. Expert tensors are fused by expert index; routing is top-k and shared expert work is additional. The A3B model's stated 3B active count includes always-on backbone/shared work and top-8 experts; simply multiplying expert capacity by `8/256` is not total active parameters.
- **Gemma 4 PLE**: documented PLE table has the explicit capacity product `vocab_size_per_layer_input × num_hidden_layers × hidden_size_per_layer_input`. E2B: `262,144 × 35 × 256 = 2,359,296,000` table parameters; E4B: `262,144 × 42 × 256 = 2,818,572,288`. These are embedding *storage*, despite cheap row lookups. The prototype implements a generic learned per-layer lookup and hidden projection, not an exact Gemma PLE port: it does not claim Gemma's projected context signal, scaling, architecture compatibility, or checkpoint tensor compatibility.
- **OLMoE** is a useful independently open dense-attention MoE reference for router auxiliary balancing and sparse top-k expert capacity, but is not a hybrid state-space reference. Its published count uses its own active-parameter convention and should not be compared without aligning tied/output/shared/router inclusion rules.

## Prototype implementation

`src/chowder/experimental_hybrid_lm.py` is opt-in and isolated from the Chowder training/model execution path. It imports PyTorch directly; PyTorch is an optional training dependency. Consumers must explicitly install the training extra. The model uses standard pre-norm residual blocks; this is the prototype's chosen baseline, not a claim of exact Granite/Qwen/Gemma residual scaling/normalization behavior.

- `HybridLMConfig` validates all core dimensions, layer types, GQA shape, Mamba head geometry, top-k, capacity and output-projection choices.
- `HybridLanguageModel` builds a narrow pre-norm decoder with residuals, configurable Mamba2-reference / causal attention schedules, dense SwiGLU or routed sparse-MoE FFNs, optional always-on shared experts, generic PLE tables, tied or standard output weights, or an explicit low-rank output head.
- The small Mamba2 reference is a readable, unfused CPU recurrence with input-dependent dt, causal depthwise convolution, diagonal per-head state, B/C readout, D residual, gate/norm and output projection. It is inspired by reference source/configs, **not a compatible Granite or Qwen implementation**.
- `CausalSelfAttention` applies RoPE and a causal mask, retains grouped KV-head states, and supports incremental decoding with mutable request cache. The cache is tagged with a config digest and batch size.
- `SparseMoEFFN` top-k routes token assignments, computes a bounded capacity during training, makes overflow policy explicit (`drop` with per-token renormalization or `error`), computes a differentiable load-balancing auxiliary loss, and only calls expert modules with at least one accepted assignment. Evaluation disables batch-size-dependent capacity dropping so outputs remain invariant to prefill/decode chunking. Shared experts are separate always-on paths.
- Save/reload uses the validated JSON config plus `torch.save` weights-only state dict. It refuses overwrite; it does not load HF weights or convert checkpoints.
- `parameter_report()` counts unique Parameters (tied embedding/head only once), bytes by dtype, resident parameter bytes, top-k active/capacity/dormant routed expert counts, PLE table parameter+byte capacity, and distinct matrix-active vs lookup-aware effective-per-token conventions.
- `flop_estimate()` is analytical—not profiled. It estimates active linear multiply-adds, Mamba convolution/state work and causal attention context work, and explicitly excludes nonlinear/norm/router/backward/kernel overhead. No memory-traffic counters are claimed.

## Configurations and ablations

The standalone config registry is `examples/experiment_d/configs/registry.json`; all configs are small synthetic CPU fixtures (vocab 256, hidden width 24/32, 4 layers). They are not a multibillion-parameter recipe.

| ID | Controlled change | Config intent |
|---|---|---|
| A | Hybrid backbone + dense FFNs | Dense reference arm. |
| B | A → top-2 of four sparse experts | Same hybrid schedule and width; top-2 expert matrix width is selected to approximately match A's dense FFN matrix parameters, apart from router overhead. |
| C | B + per-layer embeddings (dim 4) | Changes only PLE relative to B; report the embedding-table capacity and bytes separately. |
| D | C with hidden width 32 → 24 and expert FFN width 32 → 24 (dense intermediate stays 64) | Bundled narrow-width ablation; hidden/expert widths change together, so it is not a one-variable control. |
| E (optional) | C with a low-rank output matrix | Separate vocabulary-head ablation; low-rank output requires an untied input embedding and is outside the initial A–D set. |

The runner `chowder_batch/exp_d_hybrid_lm.py` offers `configs`, `smoke`, `train`, and offline `registry` subcommands. Smoke uses random weights and synthetic token IDs on CPU, compares full and incremental logits, and records exact parameter counts/analytical FLOP estimates. The bounded train command reads only explicit JSON token IDs supplied by the caller; it does not locate or prepare a dataset. Both paths write exclusive-create artifacts and label missing quality/RAM/VRAM/HBM data as unavailable rather than zero. The A–D config files now preserve the intended contrasts: B differs from A by FFN type, C differs from B only by PLE, and D bundles proportionally reduced width dimensions; optional shared experts remain available in the generator but are not mixed into these ablation arms.

The latest all-config CPU smoke artifact is `.scratch_exp_d_smoke_all_verified_20260925.json` (Python 3.11.9 / PyTorch 2.11.0+cu128; seed 123; random initialization; batch 1; sequence length 6). It reports:

| Config | Unique stored params | Active / lookup-aware effective params* | FP32 parameter bytes | Analytical FLOPs/token, context 1 / 64 | One forward (ms)** | Full-vs-incremental max logit error |
|---|---:|---:|---:|---:|---:|---:|
| A | 52,984 | 52,984 / 52,984 | 211,936 | 107,744 / 123,872 | 8.58 | 7.63e-6 |
| B | 78,072 | 53,496 / 53,496 | 312,288 | 108,768 / 124,896 | 9.59 | 5.25e-6 |
| C | 82,680 | 58,104 / 54,024 | 330,720 | 109,792 / 125,920 | 10.55 | 5.72e-6 |
| D | 50,384 | 36,560 / 32,480 | 201,536 | 66,240 / 78,336 | 9.65 | 4.29e-6 |
| E (optional) | 87,288 | 62,712 / 50,472 | 349,152 | 102,624 / 118,752 | 11.30 | 2.98e-7 |

\* “Active” here is the generator's top-k convention: unique stored parameters minus dormant routed-expert capacity; it includes full PLE table capacity. “Lookup-aware effective” excludes lookup-only table storage and counts row values used per token with tied-input/output overlap adjusted. For C/D/E, PLE capacity is explicitly `256 vocab × 4 layers × 4 dim = 4,096` table parameters (16,384 FP32 bytes); lookups do not make that stored capacity free. FLOPs are analytical estimates, not profiler counts. The FP32 parameter-byte column is tensor storage only, not process RSS/resident memory. **Each timing is a single tiny correctness-smoke forward, not a performance benchmark.** Quality, peak CPU RSS, GPU VRAM and HBM traffic were not measured. The smoke caught and led to a fix for eval-time MoE capacity dropping, which otherwise made output depend on chunk size. No ablation has been trained.

## Tests / acceptance semantics

CPU tests cover config validation; causal future-token invariance; full versus tokenwise and chunked incremental attention+Mamba output equivalence; eval-time MoE chunk invariance; cache identity/batch/state checks; expert capacity/drop/error; expert forward-call counts proving inactive experts are skipped (not evaluated then masked); router auxiliary gradients; PLE embedding lookup and exact `vocab × layers × dim` storage; tied parameter deduplication; serialization equivalence; and analytical parameter/FLOP accounting. These verify small reference semantics, not production kernels or downstream quality.

## Training/evaluation gates

No large training job, checkpoint download/load, GPU process, or active campaign is part of implementation. The current workstation resource notes are historical and must not be reused as a go decision. Before GPU work, perform a fresh read-only disk/RAM/VRAM and process/endpoint preflight and obtain explicit run authorization. The initial CPU pilot is allowed only on explicit token-ID data and must report actual runtime plus process RSS if the metric is required; this implementation does not yet sample peak RSS.

Before any scale-up:

1. Run CPU unit/property tests and compare reference block equations with pinned published implementation versions on deterministic short sequences.
2. Add a tokenizer/vocabulary plan, corpus split and fixed training token budget; compare A/B/C/D with identical seeds, schedule and tokens. Use several seeds before interpreting small quality deltas.
3. Record active FLOPs under identical training/inference shapes, route counts, per-expert accepted/dropped load, PLE table bytes, checkpoint bytes and measured memory. A theoretical active-count/FLOP advantage is not measured latency or memory bandwidth.
4. Add a conventional dense equal-parameter/equal-compute comparator and an existing pretrained A1B baseline; specify checkpoint, tokenizer and license. Distillation must be an explicit procedure with its teacher/data/compute provenance; never copy incompatible checkpoint tensors silently.
5. Evaluate perplexity, reasoning, code repair, long-context behavior, safety/tool regressions, and expert utilization/routing stability on named held-out sets. Report confidence intervals/seed variation and quality-versus-*measured* active compute and latency. No model promotion without that evidence.

## References

- IBM Granite 4.0 H Tiny model card and config: <https://huggingface.co/ibm-granite/granite-4.0-h-tiny-base>
- IBM Granite 4.0 official code overview: <https://github.com/ibm-granite/granite-4.0-language-models>
- Granite Mamba2/attention/MoE implementation (Apache-2.0 source): <https://github.com/huggingface/transformers/blob/main/src/transformers/models/granitemoehybrid/modeling_granitemoehybrid.py>
- Qwen3.5-35B-A3B model card/config/license: <https://huggingface.co/Qwen/Qwen3.5-35B-A3B>
- Qwen3.5 MoE Gated DeltaNet/MoE implementation: <https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py>
- Gemma 4 E2B/E4B model cards/configs: <https://huggingface.co/google/gemma-4-E2B> and <https://huggingface.co/google/gemma-4-E4B>
- Gemma 4 PLE documentation: <https://huggingface.co/docs/transformers/model_doc/gemma4>
- Gemma 4 Apache 2.0 license: <https://ai.google.dev/gemma/docs/gemma_4_license>
- OLMoE model card/config/paper/code: <https://huggingface.co/allenai/OLMoE-1B-7B-0125> and <https://github.com/allenai/OLMoE>
