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
- **Transformers + PEFT LoRA/SFT training executor**, isolated in its own worker process, with real subprocess cancellation. Proven end-to-end (not mocked) by a real-model training run in CI: load → train → serialize an adapter → independently reload and evaluate it → verify the evidence — required to pass before anything merges to `main`.
- **Independent holdout evaluator** that reloads the base model and adapter separately from training and checks adapter/protocol evidence itself, rather than trusting the trainer's own claim of what it produced.
- **Immutable, schema-versioned SQLite persistence** for every training artifact, evaluation outcome, and result — an append-only evidence trail, not a mutable status field.
- **JSON project configuration** with fail-closed validation, plus a guided TUI (`chowder tui`, also the default with no arguments) for building and running training projects without hand-writing config.

Autonomous repair loop:

- **Structured execution-failure capture** with live routing into the **Executor Investigator**: a real failure gets fingerprinted, checked against known remediations, and — if unrecognized — routed into a bounded investigation instead of surfacing a raw traceback. Validated against a benchmark of 11 real training incidents (not synthetic examples) plus a frozen, contamination-guarded hidden set.
- **Bounded, autonomous recursive repair**: a rejected candidate can be diagnosed, repaired, and re-evaluated automatically within a GPU-hour budget, with crash-safe resume and full provenance back to the training data and parent adapter it repaired.
- **Evidence manifest hashing** for reproducibility/provenance across the whole chain.

Still ahead: checkpoint/resume for the training executor itself, real multi-GPU (Accelerate/DDP → FSDP), an evaluator resource contract matching the training executor's, expanded training recipes (loss masking, chat datasets, LR scheduling), and wiring the autonomous repair loop into the default single-command user path. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full, currently-accurate list.

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
