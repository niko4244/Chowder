# Roadmap

Reorganized around what's actually proven vs. still speculative, rather than
version milestones — a checkbox next to a bullet doesn't distinguish "real
code with real tests" from "a stub that returns a plausible-looking dict."
Each item below names the module/PR that backs the claim.

## PROVEN / MERGED

**Research kernel**
experiment DAG · hypothesis schema · compute budget enforcement · hard
regression gate · candidate tournament (`tournament.py`, `ranking.py`) ·
deterministic VRAM/RAM/NVMe planner · evidence manifest hashing

**Real local executor**
hardware profiler (CUDA/ROCm/MPS/CPU/NVMe) · Transformers+PEFT SFT executor
with a real (not mocked) training smoke test in CI · subprocess isolation +
cooperative cancellation · checkpoint/artifact registry (SQLite-backed,
immutable) · SQLite run database with versioned schema · JSON project config
+ validation · checkpoint/restart with bound-input verification · HF
download retries, offline/local-model mode, dependency + disk-space +
architecture preflight · structured run-event contract with live progress
across the worker-subprocess boundary · TUI (recipe auto-detection,
multi-GPU, checkpoint/resume, repair, cancel, history, live run-status
panel) · hardware-aware recipe defaults (quantization, gradient
checkpointing)

**Multi-GPU DDP** — real launcher (`accelerate launch --multi_gpu`), proven
on real 2×T4 Kaggle hardware, not simulated (`docs/DDP_ACCEPTANCE.md`, PR #63)

**Scientific loop**
- independent holdout/evidence evaluator (`evaluators/`) — reloads
  base+adapter independently and verifies adapter SHA/protocol evidence
  rather than trusting the training process's own claim
- failure clustering — `failures.py::cluster_failures()` buckets eval
  failures by (evaluator, suite, protocol_sha256, source_role, failure_kind)
- hypothesis templates from eval deltas — `failures.py::plan_repairs()`
  turns each failure cluster into a templated `RepairPlan`
- replay/regression curriculum — `replay_history.py::materialize_replay_history()`
  builds a deduplicated, content-hashed rehearsal corpus at a configurable
  ratio, wired into repair candidate generation

**Regression Surgeon (partial — see hardening below for what's missing)**
- repair dataset generation — `contamination.py` builds a new SFT dataset
  from independent sources and hard-refuses any prompt/answer overlap with
  holdout (`example_fingerprints()`)
- repair-adapter branch — `repair_candidates.py`/`autonomous_repair.py`
  continue training from the exact rejected adapter's hashed weights,
  forbidding LoRA topology changes
- **a full autonomous repair loop runs end-to-end**: reject → cluster
  failures → plan → fetch independent repair sources → contamination-audit
  → materialize replay curriculum → continue-train from parent adapter →
  independently re-evaluate → gate → promote-or-not
  (`test_autonomous_repair.py::test_single_hop_autonomous_repair_runs_rejected_candidate_to_promoted_repair`)

**Adaptive Memory Fabric — Priority 1 (complete)**
- real measured memory dry-run preflight (not just post-hoc telemetry) — PR #62
- Phase 7A: real per-layer/optimizer runtime telemetry, forward hooks +
  direct tensor introspection — PR #64
- Phase 7B: activation-offload — real, measured (not formula-derived)
  experiment (PR #65) **and** production wiring into the real training
  worker with checkpoint/DDP-safety handling (PR #67)
- Phase 7C: optimizer-state tiering — real bitsandbytes paged-optimizer
  experiment (PR #66) **and** production wiring, including the
  checkpoint-incompatibility discovery below (PR #68)
- Phase 7D: frozen-layer streaming — a first real Memory Fabric runtime
  (`memory_fabric.py`), experiment (PR #70) **and** production wiring
  (PR #71). The obvious approach was tried first and rejected on real
  evidence: `accelerate.hooks.AlignDevicesHook` (the same primitive
  `cpu_offload()`/`dispatch_model()` use for big-model inference)
  offloads frozen PEFT `base_layer` weights correctly for forward+backward,
  but gives **zero real peak-VRAM savings during training** — measured
  directly, an offloaded run used *more* peak VRAM than a resident one,
  because autograd's own saved-tensor references keep every layer's
  forward-time GPU weight alive until that layer's own backward node
  runs. The fix (`_FrozenLinearRestream`, the same principle gradient
  checkpointing uses for activations, applied here to weights): a
  custom `torch.autograd.Function` whose forward does not save the GPU
  weight for backward, whose backward re-streams it fresh from pinned
  CPU RAM instead. Proven bit-identical to resident training, real
  13.5% peak-VRAM reduction on a synthetic stack, working one-layer-
  ahead async prefetch via a dedicated CUDA stream. Two more real bugs
  found and fixed along the way: a meta-tensor placeholder that crashed
  inside HF's own `Trainer`/`accelerate` internals (`model.to(device)`
  is called at more than one point this module doesn't control), and a
  CPU-only CI gap (this mechanism genuinely requires CUDA, unlike
  `activation_offload`'s graceful no-op).

**Telemetry/calibration — Priority 2 (single-GPU-verifiable slice complete)**
- production-training timing breakdown — `_TrainingPhaseTimerCallback`
  (`transformers_worker.py`) wraps `Trainer.compute_loss`/`Trainer.
  accelerator.backward`/`Trainer.optimizer.step` on the instance for
  real forward/backward/optimizer-step timing, plus a background
  `torch.cuda.utilization()` sampler thread (PR #72). A real, measured
  finding shaped the design: the `torch.cuda.synchronize()` calls
  needed for accurate timing cost ~17% real wall-time overhead, so
  this is an explicit opt-in (`backend.training.detailed_timing_telemetry`,
  default off), unlike Phase 7A's dry-run telemetry which runs in an
  isolated subprocess with no production cost. Verified to coexist
  safely with activation_offload/optimizer_tiering/frozen_layer_streaming.
- persisted aggregate telemetry for future placement learning — **already
  satisfied by pre-existing infrastructure**, not new code:
  `RunRegistry.record_training_artifact()` (`registry.py`), wired up at
  `cycle.py:417`, already persists every real training run's full
  `telemetry_json`/`evidence_json` immutably to SQLite — which now
  automatically includes the new Priority 1/2 fields (activation_offload/
  optimizer_tiering/frozen_layer_streaming/production_timing evidence)
  since they're just additional keys in the same dicts `TrainingArtifact`
  already carries. `RunRegistry.list_training_artifacts()` already
  provides read access. 7E's future placement engine has a real,
  queryable execution history to build on without needing a new store.
- **deferred, not skipped**: real GPU↔GPU bandwidth/topology measurement
  and PCIe/NVLink capability measurement genuinely need 2+ GPUs to
  produce any real data, which the development machine this work was
  done on does not have — a real hardware-access constraint (the same
  class of gap Phase 5's DDP acceptance had before Kaggle 2×T4 access
  was arranged), confirmed with the user rather than silently stubbed
  or faked.

**Unlisted but real: incident-remediation benchmark harness** —
`benchmark.py`, `investigation.py`, `hypothesis_generation.py`, `probes.py`,
`closeout.py`, `remediation_runner.py`/`remediation_actions.py`,
`model_compatibility.py`, `execution_failure.py` — CUDA OOM / dependency /
hardware-failure auto-remediation, scored against real dev/hidden incident
fixtures. Distinct from model-quality regression repair above; was entirely
missing from this roadmap before this update.

## IN PRODUCTION HARDENING

Real, shipped code that needs more real-world validation before it should be
treated as fully proven:

- **Activation offload (production)** — single-GPU only; multi-GPU DDP is
  explicitly rejected at config time, not silently allowed, because the
  interaction hasn't been verified on real multi-GPU hardware. The `"auto"`
  acceptance threshold (`_MAX_ACCEPTABLE_PENALTY_RATIO = 1.2`) is a
  documented starting point, not a measured-optimal constant.
- **Optimizer-state tiering (production)** — same shape as above: real,
  merged, single-GPU only, DDP explicitly rejected pending verification. No
  PCIe-bytes-transferred instrumentation exists (bitsandbytes' CUDA-unified-
  memory paging happens inside the driver, not through a Python-hookable
  tensor copy) — `actual_optimizer_state_bytes` is reported instead.
- **Frozen-layer streaming (production)** — same shape again: real,
  merged, single-GPU only, DDP explicitly rejected (the custom autograd.
  Function's dedicated CUDA prefetch stream has only been verified on
  single-GPU hardware). Backward-direction prefetch is not yet
  implemented (backward re-streams synchronously; correctness does not
  depend on it, only backward-pass overlap does).
- **Production timing telemetry** — real, merged, but its own real ~17%
  measured overhead means it is opt-in and mostly unused by default;
  does not separately measure all-reduce time under DDP (folded into
  `backward_seconds`) or true GPU idle/stall time (approximated by
  average sampled utilization instead) — both would need real multi-GPU
  verification this instrumentation has not had.
- **Auto-revert on failed canary** — achieved structurally, not as a
  dedicated feature: `engine.py::promote()` only overwrites the baseline
  when `GateDecision.accepted` is true, so a repair that fails its
  independent holdout eval simply never replaces the working baseline. No
  test currently drives a *failing* repair through this exact path end to
  end, and there is no monitored post-promotion rollback (Chowder never
  "deploys" before evaluating, so there is nothing to roll back from yet).
- `checkpoint_discovery.py` is real and tested, but solves *resume
  compatibility validation* for the TUI, not bisection — don't confuse it
  with "checkpoint bisect" under Research below.

## NEXT

**Adaptive placement policy — 7E (Priority 1, final piece)**
A deterministic, evidence-based placement engine using hardware topology,
model/layer sizes, and the real telemetry 7B–7D now generate (all four now
have real experiment + production data flowing into `RunRegistry` per run,
per the persistence finding above). Not yet started. Real GPU↔GPU bandwidth
data (deferred above) would sharpen this once available, but is not
strictly required to begin: a first version can reason from the real
per-mechanism telemetry already being persisted (peak VRAM saved, wall-time
penalty ratios, bytes transferred) without it. Learned placement stays out
of scope until enough real execution history exists — start deterministic/
evidence-based.

**Multi-GPU telemetry (Priority 2, deferred slice)**
Real GPU↔GPU bandwidth/topology measurement and PCIe/NVLink capability
measurement — blocked on 2+ GPU hardware access (see above). Revisit if/when
that access is arranged, the same way Phase 5's DDP acceptance was.

**Memory preflight policy (Priority 3)**
- `memory_preflight = auto | always | cached | off`, where `auto` uses
  cached measurements whenever possible and only pays for a new real
  dry-run when memory pressure or config novelty justifies it
- DDP fit must stay per-rank/per-device, never compared against aggregate
  GPU VRAM

## RESEARCH

Explicitly gated on the above being stable — no design work has started on
any of these:

- **Scientific search controller** (Priority 4) — successive halving is
  **not implemented**: `cycle.py::run_generation()` is one flat
  train-all → evaluate-all → rank pass, with no budget-elimination or
  staged rounds. Bandit/Bayesian experiment selection is also **not
  implemented** — `tournament.py`/`ranking.py` are deterministic sorts
  (probe-evidence count, gate score + efficiency), not adaptive selection
  policies.
- **Regression Surgeon extensions** (Priority 5) — checkpoint bisect
  (**not implemented**, despite the similarly-named `checkpoint_discovery.py`
  solving a different problem), dataset influence approximation (**not
  implemented**), offending-*training*-sample clustering (**not
  implemented** — distinct from the already-shipped eval-failure
  clustering), independent counterexample generation, targeted repair
  adapters beyond the existing parent-adapter continuation.
- **Meta-controller** (Priority 6) — persisted intervention/result dataset,
  expected-improvement model, GPU-hour-aware experiment policy, cross-model
  transfer of successful training strategies. Only claim learned-policy
  improvement once validated against held-out experiments.
- **Elastic MoE research** (Priority 7) — per-expert load/gradient
  statistics, expert specialization diagnostics, safe expert clone/split
  experiments, router retraining/distillation, architecture-change
  promotion gates kept behind strict regression and compute-budget gates.
