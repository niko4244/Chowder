# Chowder

**Chowder is an autonomous, evidence-gated post-training laboratory designed to make local hardware behave like a disciplined AI research team.**

The project is inspired by the direction of low-VRAM post-training tools such as Soup, but Chowder's primary abstraction is not a training command. It is an **experiment graph**:

> goal → diagnose → hypothesize → run candidate experiments → evaluate → regressions gate → promote or repair → repeat

## What exists in v0.1

- **Experiment DAG** with explicit parent/lineage invariants.
- **Hypothesis objects** that record observation, suspected cause, intervention, and expected deltas.
- **Hard regression gate** separated from optimization score.
- **Budget-aware candidate tournament** with GPU-hour efficiency ranking.
- **Evolution engine** that enforces compute budget and candidate parallelism.
- **Memory Fabric planner** that maps frozen weights, activations, optimizer state, trainable state, and workspace across VRAM → RAM → NVMe.
- **Evidence manifest hashing** for reproducibility/provenance.
- A small dependency-free CLI and regression test suite.

## Design rules

1. **No unproven promotion.** A candidate cannot become baseline while violating a hard regression gate.
2. **Every change has a hypothesis.** "Try LR 2e-5" is not sufficient; the system records why it expects that intervention to work.
3. **Compute is a budget.** Candidate selection is constrained by GPU-hours, not only benchmark score.
4. **Decision engine is backend-agnostic.** Executors for Transformers, TRL, Unsloth, llama.cpp tooling, or compatible trainers belong behind a stable adapter protocol.
5. **Artifacts are reversible.** Adapters/checkpoints form lineage rather than silently mutating the canonical model.
6. **Evidence is first-class.** Dataset/config/model lineage and evaluation outcomes must be hashable and replayable.

## Quick start

```bash
python -m pip install -e .
pytest

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
