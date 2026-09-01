# Roadmap

## v0.1 — Research kernel

- [x] experiment DAG
- [x] hypothesis schema
- [x] compute budget enforcement
- [x] hard regression gate
- [x] candidate tournament
- [x] deterministic VRAM/RAM/NVMe planner
- [x] evidence manifest hashing
- [x] tests

## v0.2 — Real local executor

- [ ] hardware profiler (CUDA/ROCm/MPS/CPU/NVMe)
- [ ] Transformers + PEFT SFT executor
- [ ] dry-run memory calibration using actual allocator peaks
- [ ] subprocess isolation + run cancellation
- [ ] checkpoint/artifact registry
- [ ] SQLite run database
- [ ] YAML/JSON project configuration

## v0.3 — Scientific loop

- [ ] failure clustering
- [ ] hypothesis templates from eval deltas
- [ ] successive halving
- [ ] Bayesian/bandit experiment selection
- [ ] independent holdout/evidence evaluator
- [ ] replay/regression curriculum

## v0.4 — Regression Surgeon

- [ ] checkpoint bisect
- [ ] dataset influence approximation
- [ ] offending-sample clustering
- [ ] repair dataset generation
- [ ] repair-adapter branch
- [ ] auto-revert on failed canary

## v0.5 — Adaptive Memory Fabric

- [ ] measured PCIe/RAM/NVMe throughput calibration
- [ ] asynchronous layer prefetch
- [ ] activation offload
- [ ] dynamic hot-layer cache
- [ ] optimizer-state tiering
- [ ] online placement policy from observed stalls

## v0.6 — Meta-controller

- [ ] intervention/result dataset
- [ ] expected-improvement model
- [ ] GPU-hour-aware experiment policy
- [ ] cross-model transfer of successful training strategies

## v0.7 — Elastic MoE research

- [ ] per-expert load/gradient statistics
- [ ] expert specialization diagnostics
- [ ] safe expert clone/split experiments
- [ ] router retraining/distillation
- [ ] architecture change promotion gates
