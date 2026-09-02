# Chowder

**Chowder is an autonomous, evidence-gated post-training laboratory designed to make local hardware behave like a disciplined AI research team.**

The project is inspired by the direction of low-VRAM post-training tools such as Soup, but Chowder's primary abstraction is not a training command. It is an **experiment graph**:

> goal → diagnose → hypothesize → run candidate experiments → evaluate → regressions gate → promote or repair → repeat

## What exists now

Research kernel (v0.1):

- **Experiment DAG** with explicit parent/lineage invariants.
- **Hypothesis objects** that record observation, suspected cause, intervention, and expected deltas.
- **Hard regression gate** separated from optimization score.
- **Budget-aware candidate tournament** with GPU-hour efficiency ranking.
- **Evolution engine** that enforces compute budget and candidate parallelism.
- **Memory Fabric planner** that maps frozen weights, activations, optimizer state, trainable state, and workspace across VRAM → RAM → NVMe, plus a real hardware calibrator that measures actual local storage/host-copy/CUDA transfer throughput instead of assuming it.

Real local executor (v0.2):

- **Hardware profiler** that inventories real CUDA/CPU/RAM/NVMe capacity (`chowder hardware-detect`), not a static config.
- **Transformers + PEFT LoRA/SFT training executor**, isolated in its own worker process, with real subprocess cancellation. Proven end-to-end (not mocked) by real-model training runs in CI: load → train → serialize an adapter → independently reload and evaluate it → verify the evidence — required to pass before anything merges to `main`. The CI job runs the full real-ML test suite, not one fixed file, so every real end-to-end test added over time is automatically part of that required gate.
- **Automatic baseline evaluation**: the untouched model is evaluated first (not user-supplied), the resolved model revision is bound into the training run so it starts from the exact snapshot the baseline measured, and `require_protocol_match=True` rejects a candidate whose evaluation protocol drifted from the baseline's instead of comparing anyway.
- **Training checkpoint/restart** on top of Trainer's own checkpoint mechanism (`save_strategy`/`save_steps`, optimizer/scheduler state), with a manifest binding dataset/config/model hashes to each checkpoint so a resume is rejected if any of those inputs changed underneath it.
- **Multi-GPU training launch** (`accelerate launch` + DDP; FSDP not yet implemented) driven by `backend.runtime.active_accelerator_count`, with GPU-hour accounting (`accelerator_seconds = wall_seconds * active_accelerator_count`) that reflects real launched processes rather than a declared estimate, and a post-run check that fails loudly if the launch did not actually engage every requested device.
- **Optimizer/schedule recipe controls**: LR scheduler type, warmup (ratio or step count), weight decay, gradient clipping, and `max_steps` (overrides epoch-based length) are all first-class training-recipe fields, not hardcoded HF defaults — `max_steps` is excluded from the checkpoint-resume bound-inputs check the same way `epochs` is (extending training length is the point of resuming), while the others are treated as optimizer-trajectory hazards a resume must match exactly.
- **Chat/message datasets with completion-only loss masking**: `backend.dataset_format: "chat"` trains on `{"messages": [...]}` rows instead of flat text, computing loss only over assistant turns. Masking is template-agnostic — it renders each assistant turn's boundary via `add_generation_prompt` and diffs token spans rather than depending on a chat template defining a `{% generation %}` block, which most real templates (including the official Llama 3.1 one) don't.
- **Auto-detected LoRA target modules and architecture presets**: omitting `backend.lora.target_modules` delegates to PEFT's own actively-maintained per-architecture mapping (`LoraConfig(target_modules=None)`) instead of guessing one fixed Llama-shaped default for every model. `backend.lora.target_preset: "attention_and_mlp"` opts into a wider, Chowder-curated module list for the well-documented Llama-family naming convention (Llama/Mistral/Qwen2/Gemma/Gemma2) when more adapter capacity than the safe default is wanted — and refuses to guess for architectures it hasn't verified, rather than silently applying a partially-wrong list.
- **Hardware-aware recipe defaults**: `quantization` and `gradient_checkpointing`, when not set explicitly, resolve from the real detected VRAM (the smallest active device under multi-GPU, not the largest) instead of one fixed value regardless of hardware — 4-bit quantization only kicks in below a conservative VRAM threshold and only when `bitsandbytes` is actually importable, and gradient checkpointing only turns off when there's real headroom to spare. An explicit config value always wins outright. Recorded in evidence (`hardware_aware_defaults`) so it's clear which values were chosen versus defaulted, and why.
- **Resilient model/tokenizer downloads**: every Hugging Face Hub download (training and evaluation alike) retries transient network/server failures with exponential backoff, while a bad model name, a bad revision, or a gated repo without access fails immediately instead of retrying into a long, pointless wait. Classification walks the real exception chain, not just the caught object's own type — `transformers` itself routinely catches a specific, typed error and re-raises a generic `OSError`, so the type worth checking usually only survives as `__cause__`.
- **Independent holdout evaluator** that reloads the base model and adapter separately from training and checks adapter/protocol evidence itself, rather than trusting the trainer's own claim of what it produced.
- **Evaluator resource contract matching the trainer's**: evaluators support `profile()` (cost estimation) and `cancel()` alongside `evaluate()`, and an evaluator crash gets the same structured-failure capture, Executor Investigator routing, and partial GPU-hour accounting a training crash already did — not silent generic-exception handling.
- **Immutable, schema-versioned SQLite persistence** for every training artifact, evaluation outcome, and result — an append-only evidence trail, not a mutable status field.
- **JSON project configuration** with fail-closed validation, plus a guided TUI (`chowder tui`, also the default with no arguments) for building and running training projects without hand-writing config.

Autonomous repair loop:

- **Structured execution-failure capture** with live routing into the **Executor Investigator**: a real failure gets fingerprinted, checked against known remediations, and — if unrecognized — routed into a bounded investigation instead of surfacing a raw traceback. Validated against a benchmark of 11 real training incidents (not synthetic examples) plus a frozen, contamination-guarded hidden set.
- **Bounded, autonomous recursive repair**: a rejected candidate can be diagnosed, repaired, and re-evaluated automatically within a GPU-hour budget, with crash-safe resume and full provenance back to the training data and parent adapter it repaired.
- **Evidence manifest hashing** for reproducibility/provenance across the whole chain.

Still ahead: FSDP for multi-GPU (DDP now supported), wiring the autonomous repair loop into the default single-command user path, and the rest of HF/model infrastructure resilience (download retries now supported; offline/local-model mode, dependency and disk-space preflight, and architecture-compatibility checks before GPU reservation remain). See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full, currently-accurate list.

## Design rules

1. **No unproven promotion.** A candidate cannot become baseline while violating a hard regression gate.
2. **Every change has a hypothesis.** "Try LR 2e-5" is not sufficient; the system records why it expects that intervention to work.
3. **Compute is a budget.** Candidate selection is constrained by GPU-hours, not only benchmark score.
4. **Decision engine is backend-agnostic.** Executors for Transformers, TRL, Unsloth, llama.cpp tooling, or compatible trainers belong behind a stable adapter protocol.
5. **Artifacts are reversible.** Adapters/checkpoints form lineage rather than silently mutating the canonical model.
6. **Evidence is first-class.** Dataset/config/model lineage and evaluation outcomes must be hashable and replayable.

## Quick start

```bash
python -m pip install -e ".[train]"   # add [qlora] too for bitsandbytes 4-bit
pytest

# Inventory real local hardware
chowder hardware-detect

# Guided setup -- also what `chowder` runs with no arguments
chowder tui

# Or drive a saved project headlessly once you have one
chowder train chowder-project.json
chowder project-validate chowder-project.json

# Deterministic VRAM/RAM/NVMe residency planning, independent of the above
chowder memory-plan \
  --vram 16 --ram 64 --nvme 1000 \
  --weights 24 --trainable 1 --activations 6 --optimizer 2 --workspace 2
```

## Target architecture

```text
                         Goal + Budget
                              |
                       Research Planner
                              |
                        Experiment DAG
                   /          |           \
             Candidate A Candidate B Candidate C
                   \          |           /
                      Evaluation Matrix
                              |
                     Hard Regression Gate
                       /              \
                  reject/repair       promote
                       |                 |
                 Failure Miner      Registry
                       |                 |
                 Curriculum Builder <---+
                              |
                        next generation

       -------------------------------------------------
                   CHOWDER MEMORY FABRIC
        GPU VRAM <-> system RAM <-> NVMe cold storage
       -------------------------------------------------
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/ROADMAP.md`](docs/ROADMAP.md).
