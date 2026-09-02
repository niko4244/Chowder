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

- [x] hardware profiler (CUDA/ROCm/MPS/CPU/NVMe) — `hardware.py`
- [x] Transformers + PEFT SFT executor — `backends/transformers_peft.py` + `transformers_worker.py`, proven end-to-end by a real (not mocked) training smoke test in CI
- [ ] dry-run memory calibration using actual allocator peaks — partial: `transformers_worker.py` records real `torch.cuda.max_memory_allocated` telemetry post-run, but there is no dedicated preflight dry-run pass that estimates fit *before* committing GPU resources
- [x] subprocess isolation + run cancellation — `executors.py` (`cancel()`), isolated worker process in `transformers_worker.py`
- [x] checkpoint/artifact registry — `registry.py`, immutable SQLite-backed persistence
- [x] SQLite run database — `database.py`, versioned schema (`CURRENT_SCHEMA_VERSION`)
- [x] JSON project configuration — `project.py` + `config_validation.py` (YAML was never added; only JSON ships)

## v0.3 — Scientific loop

- [ ] failure clustering
- [ ] hypothesis templates from eval deltas
- [ ] successive halving
- [ ] Bayesian/bandit experiment selection
- [x] independent holdout/evidence evaluator — `evaluators/` (`lm_eval.py`, `transformers_text.py` + isolated workers), independently reloads base+adapter and verifies adapter SHA/protocol evidence rather than trusting the training process's own claim
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
