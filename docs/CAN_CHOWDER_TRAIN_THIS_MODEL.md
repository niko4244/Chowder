# Can Chowder train this model? Verified, with two defects found

Target: `F:\llm-models\Qwen3.8-9B-Pruned-CW-3456` — the artifact
`docs/HOT_CORE_VS_STATIC_PRUNE.md` recommends. `qwen3_5` hybrid, 24 `linear_attn`
(Mamba-style) + 8 full-attention layers, FFN pruned to 3,456 channels, 5.937B
params. Driven through `chowder.project_runner.run_project` — the same entry point
the smoke tests use, so registry, automatic baseline, protocol binding, training
worker, evaluation worker and the promotion gate are all production code.
Evidence: `evidence/hot-core-upcycling/level2-*.json`, `unsloth-adapter-diagnosis.json`.

## A green suite had never shown that Chowder trains

The tests that prove it are gated behind `CHOWDER_REAL_ML_SMOKE=1` /
`CHOWDER_REAL_UNSLOTH_SMOKE=1` and sit among the **77 skipped** on every normal
run. "1448 passed" said nothing about training. Opening the gates found two
defects, both now fixed (see `chowder/worker_env.py` and
`tests/unsloth_env_link.py`), after which:

| gated suite | before | after |
|---|---|---|
| `test_real_ml_training.py` (Transformers, tiny model) | 1/4 | **4/4** |
| `test_unsloth_peft_real.py` | always skipped, never run | **passes** |
| `test_project_runner_repair_unsloth.py` | passed once in PR #135, never re-runnable | **passes, twice** |

Note the Transformers smoke runs on **CPU in fp32 with no quantisation**, so even
at 4/4 it does not exercise the GPU/4-bit path. That is what the run below is for.

## Transformers engine on the real model: it trains

Task chosen so "it trained" is not inferred from a loss curve: invented
subject→letter facts the base model cannot know, one token each so greedy decoding
can match exactly. Train and eval use the same items deliberately — this measures
whether training takes effect, not whether it generalises.

| | value |
|---|---|
| loss | **4.8354 → 0.3767** over 50 steps |
| peak VRAM | **11.66 GB** (fits the 15.93 GiB card) |
| wall clock | 3.6 min end to end, training 132 s |
| automatic baseline | quality **0.30** (measured, not assumed) |
| candidate | quality **0.45**, `adapter_loaded: true` |
| gate | **not promoted** — the +0.15 gain did not clear the
`minimum_promotion_gain: 0.2` set *before* the run |

The gate rejecting on a real improvement is better evidence than a promote: it
applied its threshold instead of rubber-stamping. The bar was **not** lowered
afterwards to manufacture a promotion. 20 eval items is a small sample and 0.30
baseline is partly letter-guessing luck; the claim here is "training takes effect
through Chowder", not a capability number.

## Two things anyone training this architecture must know

**1. PEFT has no auto-detection mapping for `qwen3_5`.** With the default
`target_modules=None` the run fails:
`ValueError: Please specify target_modules or target_parameters`. Chowder's
curated `attention_and_mlp` preset has no entry for this model_type either, and
adding a naive one would be worse than useless: its llama-shaped list covers only
8 of 32 layers' attention here, which is the silent partial-coverage bug the
worker's own comment warns about.

**2. The list that works**, verified from the adapter PEFT actually wrote (200
modules injected, all three families the hybrid has):

```
q_proj, k_proj, v_proj, o_proj          # 8 full-attention layers
in_proj_qkv, in_proj_z, out_proj        # 24 linear_attn layers
gate_proj, up_proj, down_proj           # all 32 FFNs
```

This matches the frontier repo's `train_grpo_minimal.py` BROAD_MODULES, an
independent cross-check. For the MoE variant the same names work —
`gate_proj/up_proj/down_proj` land on `shared_expert`, and the routed bank is raw
`nn.Parameter` that PEFT cannot reach at all.

## Unsloth engine: it trains, but its adapter is INERT under Chowder's evaluator

Unsloth trained the same task more cheaply — loss **4.4014 → 0.3760**, peak VRAM
**6.24 GB** (vs 11.66 GB), 4.0 min — and then scored **0.30, exactly the baseline**,
with predictions byte-identical to the untrained model.

`adapter_loaded: true` was reported. That flag only means the load call returned.
What it actually produced, measured by loading each adapter onto the plain
transformers model and comparing logits:

| engine | saved key prefix | LoRA modules injected | max \|logit delta\| | effect |
|---|---|---:|---:|---|
| transformers | `base_model.model.model.layers.0.linear_attn…` | **200** (all families) | **14.5** | top token ` The` → ` D` |
| unsloth | `base_model.model.model.`**`language_model`**`.layers.0…` | 128 (**all 72 `linear_attn` missing**) | **0.000000** | none |

PEFT emitted `UserWarning: Found missing adapter keys` naming every key it
expected, loaded none of them, and the adapter was a no-op. The saved B matrices
are non-zero (200/200), so the training was real — the weights simply never reach
the model.

**Root cause:** the two engines load different classes. Unsloth loads the full
`Qwen3_5ForConditionalGeneration`, whose decoder layers sit under
`model.language_model.layers`; Chowder's transformers path loads the text-only
CausalLM, whose layers sit at `model.layers`. The adapters are keyed accordingly
and are mutually incompatible.

**Why this matters more than a failed test:** it fails *silently*. An
Unsloth-trained candidate evaluates as the base model, the gate compares base
against base, and the run reports "no improvement" for every Unsloth candidate on
this architecture — indistinguishable from a genuinely useless adapter. Any
previous Unsloth result on a model with this wrapper shape should be re-checked
before it is believed.

**Not fixed here**, because the fix is a design choice rather than a patch:

1. have the Unsloth worker load the same class the evaluator does;
2. evaluate Unsloth-trained adapters with Unsloth;
3. normalise adapter keys on save or load (strip/insert the `language_model`
   segment) — cheapest, but a remap that silently guesses is the kind of thing
   that produced this bug;
4. at minimum, **fail loudly** rather than report a number. **This one is now
   landed** — see below. It does not make the Unsloth path work; it makes the
   Unsloth path stop lying.

## The loud-failure guard (landed)

`chowder/adapter_guard.py`, called at **all four** adapter load sites: the text
evaluator, both parent-adapter continuation paths, and the dataset-influence
worker. Two checks, neither needing a forward pass:

1. **Key overlap** — at least one saved tensor must name a real adapter parameter
   on the live model. Zero overlap means nothing loaded.
2. **A non-zero `lora_B`** — PEFT zero-initialises `B`, so an all-zero `B` is an
   identity transform no matter how the key bookkeeping looks.

On the real artifacts: the Transformers adapter is **accepted** (400 keys matched,
200 non-zero `B`); the Unsloth adapter is **refused** — *"shares NO parameter names
with the loaded model: 400 saved tensors, 256 adapter parameters on the model, 0
matched"*, with the `language_model.` prefix explanation in the message so the
cause is actionable without re-running anything.

Also fixed: `adapter_loaded` in evaluation provenance was literally
`spec.adapter_dir is not None` — "a directory was requested", not "an adapter is in
effect". That is why the inert run reported `adapter_loaded: true`. Provenance now
records `adapter_requested`, a measured `adapter_loaded`, and the full liveness
report.

Two deliberate design points. `unsloth_worker.py` carries the check **inlined**,
because its docstring forbids importing from the `chowder` package; a test asserts
both that the inline guard is present and that the file still imports nothing from
chowder. And an unreadable `B` matrix (quantised or exotic storage) counts as live
rather than dead — a measurement gap must not be reported as a defect, so the
key-overlap check carries that decision.

## Status

* Chowder **can** train this model, through its own lifecycle, on this hardware —
  via the **Transformers** engine, with explicit `target_modules`.
* The **Unsloth** engine trains it (and more cheaply) but cannot currently be
  evaluated through Chowder on this architecture. Treat the Unsloth path as
  unverified for `qwen3_5` until the adapter-key mismatch is resolved.
