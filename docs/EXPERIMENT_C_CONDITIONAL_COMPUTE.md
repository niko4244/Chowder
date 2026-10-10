# Experiment C — Conditional Backbone and Adaptive Compute

**Status: isolated prototype; no campaign changes and no quality or hardware savings claimed.** The current tree has research-only primitives and CPU tests; no full-model profile, gate training, GPU benchmark, unseen-task evaluation, or quality/latency Pareto result exists yet. The normal Chowder training/evaluation workers and campaign configurations do not import or enable this prototype.

## Isolation and current repository evidence

The active checkout contains unrelated edits and concurrent artifacts. The Experiment C implementation is confined to `src/chowder/conditional_compute.py`, `src/chowder/conditional_profile.py`, `chowder_batch/exp_c_profile.py`, `tests/test_conditional_compute.py`, `tests/test_conditional_profile.py`, and this report. No tracked campaign config, registry, training JSONL, dataset, model checkpoint, running endpoint, or evaluator was modified or used. Do not stage broad paths or overwrite unrelated work.

Existing model evidence is indirect only: repository notes describe a hybrid Qwen3.5 decoder with linear-attention/Mamba-style and full-attention layers, and a Spark 2.5 training campaign. They do not provide an Experiment C baseline profiler capture. Earlier workstation notes recorded a busy RTX 5060 Ti and constrained storage; those readings are stale and are not a current preflight. Do not load/download weights or start a GPU run until a fresh, read-only resource and endpoint preflight confirms headroom and the user authorizes that run.

## What is implemented

### Phase 1 — profiler primitive (not yet run on a real model)

`profile_model_call` profiles an already-loaded `torch.nn.Module` and caller-supplied request. It records warmed, synchronized latency samples separately from a hook/range-instrumented profiler pass; recursively discovers common decoder layer paths and refuses ambiguous layouts; reports per-layer `nn.Linear` projection FLOPs, profiler-supported matmul/conv FLOPs, CPU inclusive module timings, available GPU operator/kernel events, parameter-storage/residency and CUDA allocator peak; and classifies FFN versus attention, normalization and linear-attention/recurrent components. For a generation callback it labels the first top-level forward as prefill and subsequent forwards as decode. Profile artifacts are exclusive-create; repeated paths refuse overwrite.

Honest coverage limits are prominent in the JSON: PyTorch cannot provide reliable HBM traffic counters here (use Nsight Compute/CUPTI for an authorized isolated run); fused attention/SSM kernels may not expose FLOPs or leaf-module ranges; `nn.Linear` FLOPs are a shape-based theoretical count, not hardware instruction counters; nested module times overlap; weights currently on CUDA are not a whole-process VRAM census. The profiler only reports what it actually sees. Projection FLOPs are also reported by component (where module names identify attention/FFN) and per layer; module inclusive CPU times are categorized and layer parameter bytes are inventoried. Profiler capture can have nontrivial overhead, so use uninstrumented latency samples for timing and compare identical workloads.

### Variant A — conditional FFN

`ConditionalFFN` gathers only selected token rows and invokes a token-local FFN on those rows; bypassed rows receive a zero FFN delta and retain the surrounding residual unchanged. `AlwaysOnRouter` is the dense control; `FixedSkipRouter` gives deterministic skip patterns; `BudgetTokenRouter` provides stable top-k routing with per-sequence or global budgets; `LearnedTokenRouter` adds a trainable sigmoid gate and differentiable budget auxiliary loss. Padding masks, route statistics and all-on/all-off collapse reporting are included. Selected/executed fractions use valid tokens as denominator; executed work can exceed 100% when dense fallback processes padding. Gate updates can be prepared with `freeze_backbone_train_gates` and trained on caller-provided batches by `train_frozen_gate_control`; this trainer never loads data/model weights or invokes a worker.

The hard route uses a straight-through score for selected FFN rows, while the auxiliary budget objective explicitly supplies gate gradients. It is a research estimator, not an unbiased gradient estimator. A/B testing must include a frozen-backbone-only baseline, fixed patterns and actual quality measurement. Small token gathers may still be slower than dense GEMMs; the implementation is not a performance claim.

### Variant B — conditional depth (adapter contract only)

`ConditionalDepthLayer` separates a per-token keep mask from execution. `TokenwiseResidualAdapter` is a CPU reference showing correct gather/residual/scatter semantics for *independent tokenwise blocks only*. `FixedLayerRouter` expresses static whole-layer removal and `BudgetTokenRouter` supplies a token compute budget. An unsupported adapter defaults to deterministic dense execution; audits may set `fallback='error'`.

There is deliberately no generic attention, Mamba, KV-cache, or causal-mask adapter. A sparse adapter must explicitly affirm sparse execution and, when cache arguments are present, selective-cache support and an actual cache update. An all-skipped layer with a cache also falls back to dense (or errors); it cannot silently freeze recurrent/cache state. This means the CPU fixture is not evidence that production transformer depth can be skipped safely. Implementing that needs per-architecture residual/cache/state surgery, causal equivalence tests, and GPU measurements before this variant is usable on Chowder's models.

### Variant C — early prediction and deeper verification controller

`IntermediatePredictionHead` provides an optional vocabulary projection and `intermediate_head_loss` provides auxiliary next-token cross-entropy (including a finite all-ignored-label case). `AdaptiveEarlyExit` accepts calibrated threshold values and sends low-margin independent rows through a supplied deeper verifier. `calibrate_exit_threshold` chooses the least-compute development threshold that meets an explicit accuracy floor. Confidence scores are rankings, not assumed calibrated probabilities.

The controller rejects causal prefill/multi-token work and all KV/Mamba-cache execution: there is no verified model-specific cache update path in this prototype. Thus it currently tests orchestration semantics, not end-to-end autoregressive early exit. A real evaluation would need matched early/deep predictions, dev-only calibration, held-out difficult reasoning, uncommon-vocabulary and tool-use tests, and explicit invalid-call/fabrication and repair regression gates.

### Reports and artifacts

`chowder_batch/exp_c_profile.py` is an offline-only collator for explicitly supplied profile, routing and quality/compute JSON. It never opens a model or campaign registry. It writes exclusive-create JSON and optional dependency-free Pareto SVGs. It validates routing counts and flags gate collapse, but cannot establish that input measurements, task splits or scores are valid.

## Invariants and non-skippable candidates

This prototype only proposes token-local FFNs as initial bypass candidates. Attention and causal-mask computations encode context; normalization participates in layer numerics; Mamba/linear-attention blocks update recurrent state; KV-cache and generation-position updates must advance consistently even for a bypassed token. Residual streams and output heads must retain their exact shapes and semantics. Until a model-specific adapter proves those invariants, do not skip those components or report their compute as saved.

## CPU checks

The test fixtures check actual sparse FFN invocation count, residual identity on skipped rows, deterministic per-sequence/global top-k routing, fixed-depth controls, padded-token exclusion and fallback accounting, frozen-backbone gate updates, gather/scatter gradient flow, cache refusal, early/deep branch selection and rejection of multi-token rows, development threshold calibration, intermediate-head gradients, gate collapse detection, Pareto filtering and profiler hook cleanup. The profiler checks a tiny CPU fixture and compares per-layer plus attention/FFN linear-projection counts with known shapes. These tests validate primitive behavior only, not a pretrained architecture or GPU speed.

Latest focused validation: `.venv-repro/Scripts/python.exe -m pytest tests/test_conditional_compute.py tests/test_conditional_profile.py -q` — **39 passed**; Ruff check on the five Experiment C source/test files and `compileall` passed. These are CPU/prototype checks only; no model, training run, or GPU benchmark was run, so there is no real-model performance or quality evidence.

## Phase gates and stop conditions

1. **Profile first:** acquire an authorized, current baseline for prefill batches and single-token decode on the actual intended checkpoint, in an isolated worker. Save model/tokenizer revisions, dtype/quantization, input/batch/cache shapes, software/hardware, seed, profiler configuration, and resource/endpoint preflight alongside each artifact. Use representative reasoning, rare vocabulary and tool-use prompts with a development/held-out split.
2. **Train gates only on development:** frozen-backbone control first; compare unconditional and fixed compute-matched routes before learned gates. Record per-task/per-layer selection, executed compute, collapsed-gate rates and seed variation. Calibrate early exit only on development labels.
3. **Prove architecture semantics:** match dense outputs when every route is on; test causal masks, residuals, KV/Mamba state after every selected/bypassed token, gradients, fixed-seed determinism, numerical stability and restart/cache behavior. Until model-specific tests pass, keep attention, recurrent blocks and all cache-bearing paths dense.
4. **Measure hardware:** profile prefill and autoregressive decoding separately, and compare latency distributions/throughput and peak VRAM against identical dense controls. Sparse FLOPs alone do not satisfy the success condition. If routing/gather/scatter increases real inference time, reject the speed-claim candidate; report a separate memory or thermal benefit only if independently measured and useful.
5. **Evaluate quality:** multiple seeds and untouched tasks, including GSM8K, language-model perplexity, successful repairs, invalid tool calls and fabricated observations. Publish quality-versus-actual-compute and quality-versus-latency Pareto fronts with confidence intervals and routing distributions by task family.

No candidate is promoted by this prototype. Reject any candidate with unacceptable repair/tool regressions, state/cache mismatch, gate collapse, unstable behavior, or a theoretical-FLOPs-only win. GPU profiling/training, representative data authorizations, model-specific execution adapters, and full evaluation remain open.
