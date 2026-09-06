# Unsloth isolated executor: real-CUDA acceptance

Real-hardware commissioning of the isolated Unsloth executor (PRs #94,
#96-#98, #100-#101) on this project's own target hardware: Windows 11,
RTX 5060 Ti 16 GB (Blackwell, compute capability 12.0), 64 GB RAM.

## Status: **passed**

Every step of the planned real-CUDA acceptance sequence completed for
real, including commissioning the actual target model from the real
prior training campaign, not merely a tiny smoke model.

## Environment

```bash
chowder setup unsloth
```

produced a fully green `chowder doctor unsloth`:

```
Python              [OK] 3.13.14
Unsloth             [OK] 2026.9.2
unsloth_zoo         [OK] 2026.9.1
TRL                 [OK] 0.24.0
PEFT                [OK] 0.20.0
bitsandbytes        [OK] 0.50.2
Torch               [OK] 2.11.0+cu130
CUDA                [OK] torch CUDA='13.0'; available=True
NVIDIA GPU          [OK] NVIDIA GeForce RTX 5060 Ti
Triton              [OK] triton-windows=3.8.0.post28
xFormers             [OK] 0.0.35 (optional)
4-bit CUDA support  [OK] bitsandbytes NF4 Linear4bit forward executed on cuda:0
```

## Tiny-model commissioning

`trl-internal-testing/tiny-LlamaForCausalLM-3.2` via `UnslothPeftExecutor`,
real LoRA, real training:

- A real end-to-end run initially crashed with
  `TypeError: 'NoneType' object is not iterable` inside Unsloth's own
  `FastLanguageModel.get_peft_model` -- unlike plain PEFT's
  `LoraConfig(target_modules=None)`, Unsloth does not auto-detect target
  modules at all. Fixed by defaulting to Unsloth's own documented
  Llama-family target list (PR #98). After the fix: real training
  completed, real target modules resolved and recorded in evidence, a
  genuine standard PEFT adapter produced.
- Checkpoint/resume: a run with `save_strategy=steps`/`save_steps=2`
  produced real `checkpoint-2`/`checkpoint-4` directories; a second run
  resuming from `checkpoint-2` with a higher `max_steps` correctly
  resumed HF Trainer's own real optimizer/scheduler/global_step state.
- Cancellation: a real, mid-flight run was cancelled after 8 real
  seconds; the worker's PID was confirmed fully gone from the OS process
  table afterward (`nvidia-smi --query-compute-apps` was found to report
  a stale, unchanging process list on this Windows/WDDM machine and could
  not be trusted for this specific check), and `run()` returned promptly
  with a real `RuntimeError`, no hang.

## Real target-model commissioning

The actual model from the real prior training campaign, resolved from
this repository's own `chowder-project.json`
(`phase-a-pilot-40step-v2`) rather than guessed from shorthand:

```
Goekdeniz-Guelmez/Josiefied-Qwen3-8B-abliterated-v1
```

(Qwen3 architecture, 36 layers, hidden size 4096, ~8B parameters.) The
real prior campaign trained this model with `dataset_format: "chat"`
under the Transformers engine; the isolated Unsloth executor's current
scope is text-format datasets only (chat support is explicitly deferred
-- the isolated worker cannot import `chowder.backends.training_data`).
This commissioning run therefore used a real text-format pilot dataset
instead of the original chat-based one: it proves the real 8B target
model loads, trains, checkpoints, resumes, and produces a valid adapter
through the real Unsloth engine at real scale on this real hardware --
it does not reproduce the original chat-based campaign's exact task,
and is not claimed to.

**25-step pilot** (4-bit QLoRA, LoRA r=16/alpha=32/dropout=0.05,
max_length=256, batch_size=1):

```
SUCCESS in 91.1s
global_step: 25
peak_vram_gb: 6.51
train_loss: 1.344
resolved_target_modules: [down_proj, gate_proj, k_proj, o_proj, q_proj, up_proj, v_proj]
checkpoints: checkpoint-10, checkpoint-20, checkpoint-25
```

**150-step run, resumed from the 25-step pilot's own real
`checkpoint-20`** (same recipe, `resume_from_checkpoint` pointed at that
real checkpoint):

```
SUCCESS in 203.8s
global_step: 150
peak_vram_gb: 6.51
train_loss: 0.335  (down from 1.344 at step 25 -- genuine continued learning)
resumed_from_checkpoint: .../checkpoint-20
checkpoints: checkpoint-30 through checkpoint-150
```

Real peak VRAM (6.5 GB) leaves comfortable headroom on the 16 GB card at
this scale -- 4-bit QLoRA on a real ~8B model is well within this
hardware's real capacity.

### A real, honest observation (not a correctness issue)

The 150-step run's `run-spec.json` correctly carried the requested
`save_steps: 50`, but the real, on-disk `trainer_state.json` (and the
actual checkpoint cadence: every 10 steps, not every 50) shows
`save_steps: 10` -- the value the *original* 25-step run's checkpoint was
saved under. This looks like inherent `transformers.Trainer` resume
behavior (newer versions persist `save_steps`/`logging_steps` into
`TrainerState`, and resuming from a checkpoint appears to restore those
alongside `global_step`), not anything specific to Unsloth or to this
executor's own code -- `run-spec.json`, `UnslothPeftRunSpec`, and the
worker's `TrainingArguments` construction were all independently verified
to carry the correct requested value. The effect is strictly
conservative (checkpointing *more* often than requested, not less or not
at all), so it was not treated as a blocking defect here. Flagged as a
real, unexplained-in-full finding for a future dedicated investigation --
including checking whether `TransformersPeftExecutor`'s own resume path
shares the same characteristic, since it uses the identical underlying
`Trainer.train(resume_from_checkpoint=...)` mechanism.

## How to retry

```bash
pip install -e ".[train,dev]"
chowder setup unsloth
CHOWDER_REAL_UNSLOTH_SMOKE=1 python -m pytest -q tests/test_unsloth_peft_real.py -v
```

The real target-model commissioning above was run as a standalone script
against a real workspace with a real `chowder setup unsloth` environment,
not as a committed pytest case (an 8B-parameter real download and real
training run is too heavy for routine, repeatable CI-adjacent execution)
-- this document is the acceptance record for it, per this project's own
no-faking discipline (the same pattern `docs/DDP_ACCEPTANCE.md` and
`docs/MEMORY_FABRIC_ACCEPTANCE.md` already established).
