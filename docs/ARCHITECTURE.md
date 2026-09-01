# Chowder Architecture

## 1. Control plane

The control plane is responsible for deciding what experiment should happen next. It contains:

- failure/weakness analyzer
- hypothesis generator
- experiment graph scheduler
- compute-budget allocator
- candidate tournament
- regression surgeon
- promotion policy

A control-plane decision must be serializable before execution. Execution must not silently alter the planned model, dataset, seed, or configuration.

## 2. Execution plane

Training frameworks are plugins. The eventual interface should look approximately like:

```python
class TrainingExecutor(Protocol):
    def profile(self, experiment, hardware) -> CostEstimate: ...
    def run(self, experiment, context) -> ExperimentResult: ...
    def cancel(self, run_id: str) -> None: ...
```

Initial planned executors:

1. Transformers + PEFT/TRL
2. Unsloth
3. external command executor for Soup-compatible jobs
4. llama.cpp/Ollama evaluation and deployment adapters

## 3. Memory Fabric

The Memory Fabric is a policy layer over tensor residency. v0.1 contains a deterministic capacity planner. Later versions should instrument runtime transfer/compute timings and adapt placement online.

Priority order:

1. trainable tensors/workspace stay resident on accelerator;
2. activations remain resident when capacity permits, otherwise asynchronously offload;
3. frozen layers become a hot cache and stream from RAM;
4. cold frozen weights may be backed by NVMe;
5. prefetch depth is learned from observed transfer stalls.

The planner must never claim a workload fits merely because aggregate bytes fit. Peak workspace, fragmentation, pinned-host buffers, allocator reserve, and framework overhead require explicit margins.

## 4. Evidence gate

Optimization scores and safety/regression constraints are distinct. A large average benchmark gain cannot compensate for a protected metric falling beyond its tolerance.

Every promotion should eventually emit an evidence bundle containing:

- parent artifact digest
- candidate artifact digest
- exact config
- dataset digest(s)
- seed(s)
- environment/hardware summary
- evaluation suite version
- per-metric deltas
- gate verdict

## 5. Experiment graph

The graph exists because linear `train -> eval -> deploy` loops cannot express competing hypotheses, forks, repairs, or successive halving efficiently.

Nodes are immutable experiment intents. Artifacts/results attach to nodes after execution. A promoted node may become the parent of the next generation, while rejected branches remain available for causal comparison.

## 6. Persistence and executor isolation

`RunRegistry` uses SQLite/WAL as the initial local source of truth for experiment intents, results, manifests, and lineage. This is deliberately simpler than introducing a service/database dependency before the experiment semantics stabilize.

Training backends implement `TrainingExecutor`. The control plane consumes only `CostEstimate` and `ExperimentResult`; framework-specific launch flags, device quirks, and cancellation logic stay inside the executor. This boundary is important for comparing multiple execution backends without changing the scientific policy.
