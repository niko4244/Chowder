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
- Phase 7E: adaptive placement policy — a first, deterministic, evidence-
  based placement engine (`placement_policy.py`, PR #75). The real gap it
  closes: activation_offload/optimizer_tiering/frozen_layer_streaming's
  own `"auto"` modes each independently decide whether *that mechanism
  alone* is worth enabling, never reasoning about combining them. `build_
  placement_plan()` runs the three real experiments and searches all 2³
  combinations for the cheapest one (fewest mechanisms, then lowest
  worst-case penalty ratio) predicted to make a non-fitting recipe fit.
  A real finding shaped the threshold logic: measured `vram_saved_gb` is
  rarely exactly 0.0 even with no genuine benefit (allocator noise), so
  a documented `_MEANINGFUL_SAVINGS_GB = 0.01` floor keeps negligible
  "savings" from being folded into a combination. This first slice was
  deliberately informational only, not yet auto-applied — see "Adaptive
  placement policy (7E)" under In Production Hardening below for how it
  was later actually wired to drive real training (PRs #81, #82), and why
  it still isn't the final Memory Fabric acceptance milestone.

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

**Scientific search controller — Priority 4 (complete)**
- successive halving — `successive_halving.py` (PR #77). `cycle.py::
  run_generation()` was one flat train-all → evaluate-all → rank pass with
  no budget-elimination or staged rounds; `run_successive_halving()` runs
  real rounds at an increasing `max_steps` budget, keeps the top
  `survival_fraction` of gate-accepted candidates (tracked separately from
  gate-rejected ones — real provenance, never conflated), and chains each
  survivor into the next round via a REAL checkpoint resume built through
  the existing parent/child config-patch lineage (`ExperimentGraph.
  resolve_config`, not hand-reconstructed). Only the last round actually
  run is ever promoted. `cycle.py::run_generation()` is now a thin
  `run_round(experiments, promote=True)` wrapper — zero behavior change
  for every existing caller. Verified end to end on real hardware: 4 real
  candidates trained cheaply, 2 real survivors correctly separated from 2
  real cutoff-eliminations, round 2 genuinely resumed the real winner's
  checkpoint and trained additional real steps on top of it (proven by
  `global_step`, not a restart from scratch).
- bandit candidate ordering — `candidate_selection.py` (PR #78).
  `prioritize_candidates()` reorders a pool of not-yet-run candidates by
  UCB1 score over "arms" (the frozenset of dotted `config_patch` key-paths
  an experiment touches), reward = `decision.score / gpu_hours` (reusing
  `tournament.py`'s own efficiency formula) replayed against real
  `(Experiment, ExperimentResult)` history through the real hard gate.
  Never bypasses or duplicates the gate — only decides which not-yet-run
  candidate gets GPU-hours first. Cold start (no history) provably
  preserves input order (untried arms score `+inf`, `sorted` is stable).
  `RunRegistry.list_experiments()` was added as the missing join key to
  reconstruct historical `(Experiment, ExperimentResult)` pairs.
- regression-tested together: `test_cycle.py`/`test_successive_halving.py`
  exercise the promote=False deferral, gate-rejection vs cutoff-elimination
  provenance, and exact GPU-hour accounting across chained rounds.

**Regression Surgeon extensions — Priority 5 (2 of 4 slices complete)**
- checkpoint bisect — `checkpoint_bisect.py` (PR #79). The existing
  autonomous repair loop only ever asked "was the final checkpoint of a
  rejected run good enough" — `evaluate_all_checkpoints()` independently
  re-evaluates every real checkpoint a rejected run wrote and gates each
  one against the same baseline the final candidate was gated against, to
  find the earliest checkpoint that already regresses. Reuses the real
  production training/evaluation path rather than inventing new
  measurement code (a checkpoint is just a real `TrainingArtifact` with
  its own real sha256 content digest, independently re-evaluated).
  Deliberately a linear scan, not binary search, by design (checkpoint
  counts are typically single digits; cost is dominated by evaluation
  subprocess launches, not comparison count). `checkpoint_discovery.py`
  solves a different problem (resume-compatibility validation for a NEW
  run) and was deliberately not reused for enumeration.
- non-continuation repair variants — `autonomous_repair.py` (PR #80). The
  repair loop always continued training from the rejected adapter's exact
  hashed weights, which is why it hard-blocked any variant from changing
  LoRA topology. `run_single_hop_autonomous_repair(..., continue_from_
  parent=False)` skips parent-adapter verification and lifts that
  restriction for a fresh-start variant, which has no parent weights a new
  topology could conflict with — a pure integration change, plumbing into
  the `parent_adapter=None` path `repair_candidates.py::
  build_repair_candidate` already implemented and already tested (data
  model was already correct; only the single call site was missing the
  option). Replay stays orthogonal to continuation by design.
- **not yet implemented**: dataset influence approximation, offending-
  training-sample clustering, independent counterexample generation — see
  Research below.

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
- **Adaptive placement policy (7E) — now wired to real training, still
  not fully production-proven** — `build_placement_plan()` (PR #75) now
  actually drives `spec.activation_offload`/`optimizer_tiering`/
  `frozen_layer_streaming` when a mechanism's config value is `"auto"`
  and the recipe needs intervention to fit (`resolved_activation_offload`
  /etc. in `transformers_peft.py`, PR #82). Its combination search is no
  longer purely additive for 2+-mechanism combinations: `combined_
  mechanism_experiment.py` (PR #81) runs one real baseline and one real
  SIMULTANEOUS multi-mechanism training run and persists the actual
  measured combined effect (`.chowder/combined_mechanism_experiments.json`,
  work-dir-scoped cache, same convention every other mechanism experiment
  uses); `build_placement_plan()` only ever selects a 2+-mechanism
  combination when a real, empirically-validated measurement exists for
  it — an unvalidated combination is excluded entirely from
  auto-selection, never merely deprioritized. A real, measured finding
  from that module's own development: a full training run combining
  activation_offload + frozen_layer_streaming showed **zero net peak-VRAM
  reduction** despite genuinely moving real data through both mechanisms'
  hooks, while the naive additive prediction implied real savings — proof
  the safety gate is not theoretical caution, it caught a real case where
  the old additive assumption would have been wrong. Single mechanisms
  remain always-eligible (each is independently real-measured by its own
  always-run experiment) and a mechanism's own opportunistic "worthwhile
  even though not strictly required" recommendation is preserved via a
  two-tier fallback when the recipe already fits resident (the plan
  itself is scoped to "recipe does not fit", so it recommends nothing in
  that case — a real regression the pre-existing real-ML smoke suite
  caught during PR #82's own development, before merge). A follow-up real
  bug, found while attempting the acceptance run below and fixed
  separately: the calibration subprocess calls inside `build_placement_
  plan()`/`run_combined_mechanism_experiment()` used a hardcoded 300s
  timeout regardless of the recipe's own batch size, which a genuinely
  large-batch recipe's calibration run can legitimately exceed — now
  derived from the recipe's own `backend.runtime.timeout_seconds`
  instead. **Still not fully production-proven**: no real resident-OOM →
  Memory-Fabric-success acceptance run exists yet (see Next below) — do
  not treat Memory Fabric as validated end-to-end until that exists.

## NEXT

**Final Memory Fabric acceptance test (Priority 1 follow-up) — attempted for
real, not yet demonstrated cleanly**
The remaining milestone before Memory Fabric can be called production-
proven: a real workload that genuinely CUDA-OOMs under normal resident
training, then genuinely succeeds under the same model/recipe with Memory
Fabric's real placement plan applied — not faked by lowering the reported
VRAM budget. Full real-hardware attempt log, findings, and next steps:
[`docs/MEMORY_FABRIC_ACCEPTANCE.md`](MEMORY_FABRIC_ACCEPTANCE.md). Short
version: a real production calibration-timeout bug was found and fixed
along the way; a real, reproducible `activation_offload` crash was found
and flagged separately (not fixed here); this development machine's
driver-level VRAM-to-system-RAM paging fallback and shared-desktop-GPU
contention make a clean pass hard to reach on this specific hardware; and
a mechanism's isolated single-forward+backward savings not reliably
predicting a full training run's real peak VRAM was confirmed a third
time. Revisiting this needs either a dedicated/isolated GPU or the
`activation_offload` bug fixed first.

**Backward prefetch for frozen-layer streaming (Priority 1 follow-up)**
`memory_fabric.py`'s backward re-streams each frozen layer's weight
synchronously today (correctness does not depend on overlap, only
throughput does). Prefetching layer N-1's weight while layer N's backward
is still running is the next real improvement to prove and measure —
whether the overlap actually improves throughput on real hardware, not
assumed.

**Multi-GPU telemetry (Priority 2, deferred slice)**
Real GPU↔GPU bandwidth/topology measurement, PCIe/NVLink capability
measurement, P2P availability, all-reduce timing, and DDP communication
share of backward — blocked on **matched** multi-GPU hardware access (a
locally available but asymmetric 2-GPU box does not substitute for this;
inferring symmetric-pool numbers from mismatched cards would misrepresent
real DDP behavior). Revisit if/when Kaggle-2×T4-class access is arranged,
the same way Phase 5's DDP acceptance was.

## RESEARCH

Explicitly gated on the above being stable — no design work has started on
any of these:

- **Regression Surgeon extensions, remaining slices** (Priority 5) —
  dataset influence approximation (**not implemented**: which training
  examples most likely contributed to a regression; start with a
  practical approximation such as leave-cluster-out re-training measured
  via the real independent evaluator, not a claim of exact causal
  influence), offending-*training*-sample clustering (**not implemented**
  — distinct from the already-shipped eval-failure clustering, and gated
  on influence scores existing first), independent counterexample
  generation from sources independent of the protected holdout (**not
  implemented** — must never expose holdout answers to the generator).
- **Meta-controller** (Priority 6) — persisted intervention/result dataset
  (model, hardware, dataset/failure cluster, intervention, Memory Fabric
  placement, training recipe, cost, score delta, regression delta,
  throughput, peak VRAM), expected-improvement model, GPU-hour-aware
  experiment policy, cross-model transfer of successful training
  strategies. Only claim learned-policy improvement once validated
  against held-out experiments — a durable historical dataset is not
  itself a learned policy.
- **Elastic MoE research** (Priority 7) — per-expert load/gradient
  statistics, expert specialization diagnostics, safe expert clone/split
  experiments, router retraining/distillation, architecture-change
  promotion gates kept behind strict regression and compute-budget gates.
