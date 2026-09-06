# Chowder MoE Downsizing Program

This document turns Roadmap Priority 7 (Elastic MoE research) into a concrete, falsifiable program for the Qwen3.6-35B-A3B Chowder branch.

The goal is not simply to make a smaller checkpoint. The goal is to preserve Chowder's useful behavior while reducing stored expert capacity, memory pressure, and inference cost enough to make the model practical on the target local workstation.

## Why this is now actionable

The DS4 project established several mechanisms that transfer conceptually to Qwen MoE work:

- workload-derived routed-expert activation statistics;
- per-expert importance-aware quantization;
- support for reduced expert counts rather than hard-coding the original count;
- teacher router-logit, selected-expert, routing-weight, and routed-FFN output capture;
- mixed precision by routed-expert layer rather than one global quantization choice;
- deterministic quality comparison against a stronger teacher/reference model;
- strict regression gates around every compression step.

Recent Qwen3.6 expert-pruning results make expert-count reduction a credible first compression axis. Chowder should therefore test whole-expert pruning before destructive width shrinking of every surviving expert.

## Primary hypothesis

A Qwen3.6-35B-A3B Chowder teacher can be specialized into a substantially smaller stored-expert model by retaining the experts that matter to Chowder's actual workload and distilling the removed experts' behavior into the surviving topology, while keeping the normal hard regression gate authoritative.

The first serious target is approximately 50% routed-expert retention, not an arbitrary parameter target.

## Immutable teacher

The starting trained/abliterated checkpoint becomes `Teacher-0` and is never overwritten.

Teacher-0 is used for:

- baseline capability evaluation;
- expert-usage profiling;
- router/FFN/logit distillation targets;
- promotion-gate comparison;
- recovery analysis when a compressed child fails.

Do not perform destructive pruning in place.

## Phase A — Qwen architecture audit

Before training or pruning, verify the exact checkpoint structure rather than assuming that a generic Qwen or generic MoE tool sees the same tensors Chowder believes it is modifying.

Required evidence:

- exact model type and transformer version;
- routed expert count by layer;
- experts selected per token;
- shared-expert topology;
- router module names and outputs;
- routed gate/up/down tensor names and packing;
- whether the chosen abliterated checkpoint changed all intended MoE tensors;
- trainable-module coverage under the selected PEFT/Unsloth backend.

Failure to identify a tensor/module is a hard stop, not a reason to guess a name.

### Real findings (transformers==5.16.1, read from the installed source, not guessed)

No local Qwen3.6-35B-A3B checkpoint exists on this machine as of this
writing (an exhaustive search across every attached drive found only GGUF
variants and dense, non-MoE Qwen HF checkpoints). The following was
therefore verified by reading `transformers`' own installed modeling source
for `Qwen3MoeSparseMoeBlock` / `Qwen3_5MoeSparseMoeBlock`, and confirmed
identical for `OlmoeSparseMoeBlock` — the one real local MoE checkpoint this
machine has (`OLMoE-1B-7B`). Qwen3.6 is presumed, not confirmed, to share
this same transformers-version-level shape until an actual checkpoint is
available to check.

- **Experts are not separate submodules.** Both the router and the expert
  bank are fused, batched tensors, not one `nn.Module` per expert:
  - `layer.mlp.gate` — a `*TopKRouter` holding a single `weight` matrix of
    shape `[num_experts, hidden_dim]`. Its forward returns
    `(router_logits, router_scores, router_indices)`, where
    `router_scores`/`router_indices` are `[tokens, top_k]`.
  - `layer.mlp.experts` — a `*Experts` module holding one `gate_up_proj`
    tensor `[num_experts, 2*intermediate_dim, hidden_dim]` and one
    `down_proj` tensor `[num_experts, hidden_dim, intermediate_dim]`. Its
    forward loops over the experts that were actually selected in the
    current batch, indexing `gate_up_proj[e]`/`down_proj[e]` directly.
  - Consequence for Phase C: "removing an expert" is a slice out of dim 0 of
    `gate_up_proj`/`down_proj` (plus reindexing the router's `weight` rows
    and `num_experts`), not deleting a child module.
- **Qwen3.5Moe (closest real sibling to "Qwen3.6") adds a shared expert**
  beyond the base Qwen3Moe shape: `layer.mlp.shared_expert` (a plain dense
  MLP, always active) and `layer.mlp.shared_expert_gate` (a sigmoid-gated
  scalar mixing weight). OLMoE has no shared expert. Whether Qwen3.6 keeps,
  drops, or changes this is unverified without the real checkpoint.
- **Not every layer is necessarily MoE.** Qwen3Moe supports a
  `decoder_sparse_step` config field that interleaves dense layers among
  sparse ones. OLMoE, by contrast, is uniformly sparse (all 16 layers are
  MoE). The audit must check every layer rather than assume uniformity.
- **Implementation**: `src/chowder/moe_instrumentation.py` implements this
  audit (`audit_moe_architecture`, hard-stopping via
  `MoeArchitectureAuditError` if zero layers match the verified shape) plus
  the calibration recorder for Phase B (below).
- **Local checkpoint repair note**: the local OLMoE-1B-7B directory
  (`H:/Models/olmoe-1b-7b`) had real safetensors weight shards but was
  missing `model.safetensors.index.json` (its own `.cache/huggingface/`
  download sidecars show that file was simply never fetched), so
  `from_pretrained` could not load it. The index was rebuilt losslessly by
  reading each shard's own embedded safetensors header (tensor
  name/shape/dtype) and writing the resulting `weight_map` — the same
  computation `save_pretrained` performs when splitting shards, applied
  after the fact. No weight data was read, moved, or modified.

### Real findings — Phase B calibration capture

`MoeCalibrationRecorder` (`src/chowder/moe_instrumentation.py`) hooks each
audited layer's `mlp.gate` forward, which hands it the exact flattened
hidden-state input plus `(router_scores, router_indices)` for every token.
Because experts have no separate hookable submodule, `selected_tokens` and
`router_mass` come directly from that router output, while
`gated_activation` and `output_norm` are computed by re-deriving the same
per-expert math `*Experts.forward` performs internally
(`act_fn(gate) * up`, then `down_proj`) using the live model parameters and
the real tokens routed to that expert — an exact per-expert quantity, not an
approximation, just computed outside the fused loop so it can be attributed
per expert.

`run_calibration` + `write_expert_importance_jsonl` implement items 1–3 of
the "First implementation slice" below; `chowder moe expert-importance`
(CLI) plus `moe_planning.build_uniform_pruning_plan` implement item 4. All
of this is real-hardware-validated end to end
(`tests/test_moe_instrumentation_real.py`, `CHOWDER_REAL_MOE_SMOKE=1`)
against the local OLMoE-1B-7B checkpoint: real weight loading, real CUDA
forward passes, real per-(layer, expert) statistics for all 16×64 pairs, a
real `expert_importance.jsonl`, and real dry-run 75%/50% pruning plans. This
validates the *mechanism* against a real, architecturally-equivalent local
MoE checkpoint. It is explicitly **not** a commissioning of the actual
Qwen3.6-35B-A3B target, which remains blocked on that checkpoint's local
availability.

## Phase B — Chowder Expert Importance Map

Run Teacher-0 over a representative Chowder calibration corpus and capture statistics for every `layer x expert` pair.

The corpus should include at minimum:

- code generation;
- debugging and repair;
- repository navigation and code review;
- tool calling;
- multi-step planning;
- long-context technical reasoning;
- appliance diagnostic-map reasoning;
- document extraction/synthesis;
- general reasoning/conversation ballast so specialization does not collapse basic usefulness.

Capture:

- selected-token count;
- total router mass;
- mean router mass when selected;
- gated activation magnitude;
- expert output norm/contribution;
- task/domain distribution;
- layer sensitivity;
- correlation with successful and failed benchmark cases.

A starting importance score can combine routed mass and output energy, but the individual components must remain persisted so the scoring formula can be changed without rerunning the expensive teacher trace.

## Phase C — Expert-budget search

Generate topology candidates rather than jumping directly to one pruning ratio.

Initial ladder:

- 100% experts: immutable teacher/reference;
- 75% experts: conservative student;
- 62.5% experts: intermediate student;
- 50% experts: primary compression target;
- below 50%: research-only until the 50% student is stable.

Do not assume every layer gets the same expert budget. Search a layer-aware budget after the uniform baselines establish a trustworthy comparison.

Example only, not a preset:

```text
layers 00-07: 75% experts
layers 08-31: 50% experts
layers 32-39: 62.5% experts
```

The actual allocation must come from measured sensitivity.

## Phase D — Distillation / healing

Pruning is followed by recovery training. At matched input positions, use Teacher-0 to provide targets for multiple levels of behavior.

### Router loss

Match the surviving student's routing distribution to the teacher distribution projected onto the retained expert set.

### Routed-FFN representation loss

Match the teacher routed-FFN output vector at selected layers/positions.

### Final-logit loss

Distill final output-token distributions in addition to ordinary SFT loss.

### Chowder task loss

Retain the normal Chowder training objective and replay curriculum. Distillation is not a substitute for task performance.

The exact weighting among these losses is a search variable and must be recorded as experiment provenance.

## Phase E — Mixed-precision expert allocation

Only after a stable expert-pruned student exists should Chowder search precision by component.

General principle:

- router/control paths: high precision;
- shared expert: higher precision than aggressively compressed routed experts;
- sensitive routed experts/layers: Q5/Q6-class candidate;
- normal routed experts: Q4-class candidate;
- very low-impact routed experts: Q2/IQ2-class candidate only if evaluation supports it.

The DS4 result that a small subset of routed-expert layers benefited disproportionately from higher precision is a reason to measure Qwen layer sensitivity, not a reason to copy DS4's exact layer numbers.

## Phase F — Promotion gates

Every compressed candidate must pass three independent gates.

### Gate 1 — Teacher fidelity

Compare against Teacher-0 using deterministic continuation/logprob-style measurements where possible:

- target-token NLL;
- first-token agreement;
- greedy longest-common-prefix;
- top-N recall/ranking agreement;
- hidden/router agreement for diagnostic runs.

### Gate 2 — Capability

Run Chowder's actual evaluation suites, including coding, tool use, planning, technical reasoning, and any domain-specific suites that represent production use.

### Gate 3 — Agentic recovery

For repairable tasks, measure whether the candidate can inspect a failure, revise its approach, patch, and retest. A one-shot degradation that disappears under the normal agent loop is different from a persistent capability loss, but neither is hidden: both measurements are retained.

The existing Chowder hard regression gate remains authoritative. MoE compression never creates a bypass.

## What not to do first

### Do not start with expert-width shrinking

Shrinking every surviving expert's hidden width is a separate, more destructive intervention. DS4 proved that the tensor surgery can be implemented correctly; it did not establish that the resulting intelligence loss is acceptable for Qwen3.6 Chowder.

Whole-expert pruning is the first compression axis because it preserves the internal width of experts that survived pretraining.

### Do not copy DS4 tensor names or layer-sensitive regions

Transfer the measurement strategy and scientific controls, not model-specific constants.

### Do not treat a smaller file as success

File size, VRAM fit, and tokens/s are optimization metrics. They do not override capability/regression metrics.

## Disk-constrained execution

This research must obey Chowder's local-first model-source policy.

- Reuse Teacher-0 from its existing local directory.
- Never download or stage a duplicate teacher merely to run profiling.
- Candidate outputs must be written to explicitly selected locations.
- Estimate candidate size before materializing a full pruned/distilled checkpoint.
- Prefer adapter/checkpoint experiments before exporting a new full model when the experiment can answer the hypothesis without a full export.
- Keep only promotion-worthy full-model artifacts unless the user explicitly asks to retain rejected candidates.

## Roadmap impact

This program substantially fills the design gap in Priority 7:

- per-expert load/activation statistics: concrete measurement plan;
- specialization diagnostics: concrete per-task expert map;
- router retraining/distillation: concrete teacher/student losses;
- architecture-change promotion gates: concrete three-gate protocol;
- expert-count reduction: concrete candidate ladder and layer-aware follow-up;
- mixed precision: concrete post-pruning optimization stage.

It does **not** by itself close the following roadmap items:

- Memory Fabric reliability (the core OOM-to-success claim is now real-hardware
  demonstrated per docs/MEMORY_FABRIC_ACCEPTANCE.md; a committed automated
  regression test remains blocked on this machine's WDDM driver flakiness);
- backward prefetch throughput work;
- matched multi-GPU topology/communication telemetry;
- Priority 6 learned meta-controller / expected-improvement policy;
- censoring/survivor-bias treatment for failed experiments in learned policy data;
- cross-model transfer validation;
- safe expert clone/split/expansion experiments if Chowder later needs dynamic capacity growth;
- actual Qwen3.6 implementation and real-hardware commissioning of the profiling/pruning/distillation hooks.

Priority 7 should move from RESEARCH to PROVEN only after the Qwen-specific implementation produces real teacher traces, real compressed students, and regression-gated evaluation evidence.

## First implementation slice

The smallest useful implementation should do exactly four things:

1. load an existing local Qwen3.6-A3B Teacher-0 without downloading another copy;
2. capture router selection/weights and expert contribution statistics over a fixed calibration set;
3. emit a versioned `expert_importance.jsonl` (or equivalent structured artifact) with model/config/data provenance;
4. generate a dry-run pruning plan for 75% and 50% expert retention without changing model weights.

Only after that artifact is trustworthy should Chowder gain a writer that creates a pruned student checkpoint.

**Status: items 2–4 are implemented and real-hardware-validated** —
`src/chowder/moe_instrumentation.py` (`audit_moe_architecture`,
`MoeCalibrationRecorder`, `run_calibration`, `write_expert_importance_jsonl`)
plus `chowder moe expert-importance` (CLI) and the already-existing
`moe_planning.build_uniform_pruning_plan`. Validated end to end against the
real local OLMoE-1B-7B checkpoint (`tests/test_moe_instrumentation_real.py`,
`CHOWDER_REAL_MOE_SMOKE=1`): real weight loading, real CUDA forward passes,
exact per-(layer, expert) statistics for all 16×64 pairs, a real
provenance-carrying `expert_importance.jsonl`, and real dry-run 75%/50%
pruning plans, with zero model surgery. **Item 1 remains blocked**: no local
Qwen3.6-35B-A3B checkpoint exists on this machine (confirmed by an
exhaustive search across every attached drive), so the actual named target
has not been loaded or profiled — only the mechanism has been proven, on a
real, architecturally-equivalent local stand-in.
