# Roadmap

> **Model program retargeted (2026-09-06):** Chowder's primary model
> research target is now the **Qwen3.8 Native Sparse Program** — a
> directly native-Qwen3.8-derived sparse/MoE model (~3–4B *active*
> parameters/token), developed from `orcarouter/Qwen3.8-27B-Uncensored`
> with `Qwen/Qwen3.8-27B` as untouched control and
> OBLITERATUS/DavidAU variants as comparison parents. Definition, pinned
> revisions, architecture audit, and honest blockers:
> [`docs/QWEN38_SPARSE_PROGRAM.md`](QWEN38_SPARSE_PROGRAM.md). The prior
> 8B commissioning campaign and the Qwen3.6 MoE-downsizing program
> remain below as historical evidence. No distillation in the primary
> lineage — ever.

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
- **real generation-correctness bug fixed (PR #95)**: both evaluator
  workers (`base_text_worker.py`, `transformers_text_worker.py`)
  unconditionally overrode `model.generate()`'s `eos_token_id` with the
  tokenizer's scalar id, discarding the model's own (often list-valued)
  `generation_config.eos_token_id` — many instruction-tuned checkpoints
  (Qwen2/Qwen3, Llama-3) list the chat template's real turn-end token
  there alongside the base eos. On a checkpoint whose tokenizer config
  has drifted from its generation config (observed on a real abliterated
  Qwen3 checkpoint), this made the model ramble until `max_new_tokens`
  under `use_chat_template: true`, failing `exact_match`/
  `normalized_exact_match` even when the correct answer was present —
  the real cause of a training campaign scoring `0.0` on both baseline
  and candidate. `evaluators/generation.py::resolve_eos_token_ids()` now
  prefers the model's own resolved generation config; confirmed on a real
  model that this is a pure improvement (official Qwen2.5/Qwen3 keep
  these in sync already) that fixes the drifted-config failure mode.
- failure clustering — `failures.py::cluster_failures()` buckets eval
  failures by (evaluator, suite, protocol_sha256, source_role, failure_kind)
- hypothesis templates from eval deltas — `failures.py::plan_repairs()`
  turns each failure cluster into a templated `RepairPlan`
- replay/regression curriculum — `replay_history.py::materialize_replay_history()`
  builds a deduplicated, content-hashed rehearsal corpus at a configurable
  ratio, wired into repair candidate generation

**Regression Surgeon — core repair loop complete; hardening limits below**
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

**Scientific search controller — Priority 4 (library slice complete)**
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
  `global_step`, not a restart from scratch). **A real registry-persistence
  gap was later found by integration testing and fixed (PR #90)**: the
  round-1+ child experiments `run_successive_halving()` invents itself were
  proposed to the engine but never recorded in the `RunRegistry`, so
  `ExperimentCycleRunner._record_status()` hard-refused the unknown id and
  any search with a registry attached died at the start of round 1, leaving
  that round's reservations outstanding forever (measured: `spent=1.0`,
  `reserved=1.05`, `outstanding=2` against a 10.0 budget). Neither module's
  own tests caught it — `test_successive_halving.py` never attached a
  registry and `test_cycle.py` never ran a multi-round search. Every exact
  effective round (including the controller's round-0 budget patch) is now
  validated or atomically recorded before proposal can create a reservation;
  same-ID divergent or terminal evidence is refused, while an exact PLANNED
  persistence retry is idempotent (`test_search_controller_integration.py`).
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
  provenance, and exact GPU-hour accounting across chained rounds;
  `test_search_controller_integration.py` (PR #90) proves the registry,
  scheduler, selector, repair path, hard gate, checkpoints, provenance, and
  ledger hold together across their real seams.

These are proven package capabilities, not yet Chowder's default operational
controller: no production caller under `src/chowder/` invokes
`run_successive_halving()` or `prioritize_candidates()` today. Wiring them
into `project_runner.py` remains open and must not be inferred from the
library implementation or its integration tests.

**Meta-controller evidence foundation — Priority 6 (dataset slice complete, both halves)**
- `intervention_outcomes.py` (PR #89) builds a normalized, queryable
  `InterventionOutcome` view by joining the immutable experiment, result,
  and training-artifact records the registry already stores. It reuses
  `candidate_selection.dotted_paths()` for intervention-arm identity, reads
  historical gate acceptance from persisted status, and refuses ambiguous
  artifact provenance instead of guessing a producing run.
- The honesty boundary is explicit: missing evidence stays `None`. The
  scored-result view once lacked the censored half of the dataset entirely;
  `censored_outcomes.py` (this pass) now represents experiments that ended
  without a scored result -- REJECTED-before-work and FAILED -- as
  `CensoredOutcome` rows with the same arm identity, joining the structured
  `execution_incidents` classification and its capture-time measured
  GPU-hours when an incident was recorded and honestly `None` when none
  was. It invents no score for an unobserved outcome and stores no
  sub-cause the registry never kept. The context gaps this section once
  carried are now closed (see the dataset/hardware context slice below),
  and per-arm censoring rate is a first-class signal; what would still
  create survivor bias in an unrestricted learned selector -- and how the
  two views must be combined -- is documented in the module rather than
  left implicit.
- `censored_outcomes.py` (this pass) is the censored half of that dataset:
  `build_censored_outcomes()` emits a `CensoredOutcome` row for every
  result-less REJECTED/FAILED experiment (never for PLANNED/RUNNING, and
  never for a scored result -- gate-rejection is an *observed* outcome and
  stays in `intervention_outcomes`), reuses the same
  `candidate_selection.dotted_paths()` arm identity so a censored row and a
  scored row name the same intervention arm, joins
  `registry.list_execution_incidents()` for `signature_kind`/
  `fingerprint_sha256`/executor and the incident's real capture-time
  `gpu_hours_spent`, and reports `censoring_rate_by_arm()` as the
  REJECTED-vs-FAILED shape within each arm's censored rows. Documented
  policy position: per-arm censoring rate is a first-class signal, spent
  compute on crashed runs is real cost, and any reward model over
  `InterventionOutcome` alone is survivor-biased by construction -- how to
  combine the two views is explicitly left to the policy layer. The
  once-documented gap is closed: `ExperimentCycleRunner` now persists every
  non-cancelled crash's Executor-Investigator analysis into
  `execution_incidents` (after the failure is settled; a persistence
  failure becomes a diagnostic, never a mask over the crash), so FAILED
  rows from current runs carry a real classification. Absence still
  occurs -- pre-existing registries, registry-less runs, cancellations
  (no analysis is built for a deliberate stop), persistence failures --
  and is visible as `None`, never imputed.
- Dataset and hardware context (this pass): `InterventionOutcome` now
  carries the dataset identity and scale, and the hardware context beyond
  the single `active_accelerator_count` number, that the Priority-6
  context-gap item required -- read only from evidence the registry
  already stores, under the same honesty rule. Dataset identity:
  `dataset_sha256`/`replay_dataset_sha256` (both real executors verify the
  dataset on disk and record the digest they trained on, so digest match
  is what "same data" means across runs) and `filter_outcomes(
  dataset_sha256=...)` as the same-dataset selector, with the same
  "not on record" exclusion rule as every other criterion. Dataset scale
  and shape (transformers-peft `data_provenance` only): `dataset_format`,
  `primary_rows`, `replay_selected_rows`, `total_token_count`,
  `assistant_token_count`. Hardware context: `visible_accelerator_count`
  and the measured `peak_vram_gb_by_accelerator` map (both executors
  record them; one malformed entry blocks the whole map rather than
  serving a partial one), `requested_active_accelerator_count`
  (transformers-peft only), and `base_model_revision` -- so a run on one
  of two visible GPUs is now distinguishable from a single-GPU box, the
  multi-GPU telemetry context the roadmap flagged. Known, honestly-stated
  gaps: no dataset path or file name exists anywhere in the registry (a
  content digest is the only dataset identity the evidence keeps), the
  unsloth backend records no `data_provenance` block so its rows carry
  scale/shape as `None`, and evidence recorded before the real executors
  started writing these keys stays `None` -- never backfilled.
- This remains a durable evidence view, not an expected-improvement model,
  candidate selector, learned policy, or claim of cross-model transfer.

**Regression Surgeon extensions — Priority 5 (4 of 4 slices complete)**
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
- dataset influence approximation — `dataset_influence.py` (PR #85). Forward-
  only (no backward/training), per-example cross-entropy loss computed in an
  isolated subprocess per checkpoint, reusing the exact model+adapter load
  path from the real transformers-text evaluator. Ranks
  `TrainingExampleInfluence` records by `bad_checkpoint_loss -
  good_checkpoint_loss` between a run's last-good and first-regressing
  checkpoints (from `checkpoint_bisect.py`'s `CheckpointBisectOutcome`), with
  `confidence` as a z-score against the population of measured deltas -
  deliberately separate from ranking position, not a causal claim. Verified
  real end-to-end: correctly ranked genuinely "odd" training rows above
  repetitive "easy" ones after real training on a mixed dataset.
- offending-training-sample clustering — `training_sample_clusters.py`
  (PR #87). Pure, deterministic, dependency-free greedy single-link
  clustering of `TrainingExampleInfluence` records by token-set Jaccard text
  similarity — distinct from the already-shipped eval-failure clustering in
  `failures.py`, which clusters by exact-match evaluation metadata, not
  training-example text similarity. Needs no GPU or additional real-hardware
  measurement; operates entirely on already-computed influence records.
  Non-suspicious examples (`influence_score <= 0`) are excluded before
  clustering starts.
- independent counterexample generation + targeted repair —
  `dataset_regression_repair.py` (PR #88). Bridges a `TrainingSampleCluster`
  into the existing, already-hardened repair machinery
  (`repair_orchestrator.py`, `repair_candidates.py`, `contamination.py`)
  rather than rebuilding a parallel system: `build_training_regression_
  repair_request`/`_plan` adapt the cluster into the same `RepairRequest`/
  `RepairPlan` shapes the eval-gate repair path uses, with honest sentinel
  `evaluator`/`suite` values (`"dataset-influence"` / `"training-corpus"`)
  since there is no real eval suite for a training-corpus regression.
  `direct_training_allowed` is always `False` and
  `requires_independent_source` always `True`, so the hard rule (protected
  holdout must never enter generation/repair training) is enforced by the
  same contamination-audit path the eval-gate flow already uses.
  `verified_last_good_checkpoint_adapter` binds continuation to the real
  content hash of the LAST-GOOD checkpoint specifically, never the regressed
  one. `prepare_and_propose_repair_population` was refactored
  (behavior-preserving) to take an already-built `RepairRequest` instead of
  building one internally from a `FailureCluster`, so both repair paths
  share one real orchestration function instead of duplicating budget/
  replay/parent-adapter/holdout-audit logic. The repair population is gated
  by the real, unmodified promotion gate - no separate "did this fix the
  target regression" check exists; a repair that doesn't clear the gate is
  rejected exactly like any other candidate. Includes an explicit test
  proving a repair that fails to improve is correctly refused promotion even
  when every other step (real counterexamples, real contamination audit,
  real continuation from the last-good checkpoint) succeeds.

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
  documented starting point, not a measured-optimal constant. **A real
  stride-corruption crash (found during the Memory Fabric OOM-acceptance
  investigation below) is now fixed, PR #92**: `saved_tensors_hooks`'
  pack/unpack intercepts *every* tensor autograd saves for backward, not
  just a model's own activations — including transformers' expanded/
  broadcast 4D attention bias (stride 0 on the head dim). The old hooks did
  a naive `.to("cpu")`/`.to(device)` round trip, which silently
  materializes a differently-strided dense tensor that PyTorch's memory-
  efficient SDPA backward kernel rejects
  (`attn_bias.stride(2) = 66, and should be a multiple of 4`) — reproduced
  byte-for-byte on real hardware at the exact reported scale
  (`batch_size=96`), never caught by this project's existing tests because
  they all use batch sizes of 2-8. New shared
  `activation_offload_hooks.py` preserves the exact original strides of a
  non-contiguous saved tensor via `as_strided()` instead of guarding
  against the scale/config combination — see
  `docs/ACTIVATION_OFFLOAD_STRIDE_FIX.md`.
- **Optimizer-state tiering (production)** — same shape as above: real,
  merged, single-GPU only, DDP explicitly rejected pending verification. No
  PCIe-bytes-transferred instrumentation exists (bitsandbytes' CUDA-unified-
  memory paging happens inside the driver, not through a Python-hookable
  tensor copy) — `actual_optimizer_state_bytes` is reported instead.
- **Frozen-layer streaming (production)** — same shape again: real,
  merged, single-GPU only, DDP explicitly rejected (the custom autograd.
  Function's dedicated CUDA prefetch stream has only been verified on
  single-GPU hardware). Backward-direction prefetch is now implemented and
  real-hardware measured (a real 1.82x backward-wall-time speedup on a
  synthetic stack, bit-identical loss/gradients, synchronous fallback
  retained via `backward_prefetch=False`) — see the NEXT section below for
  the full real numbers and what remains unverified (a production-model,
  not synthetic-stack, throughput measurement).
- **Production timing telemetry** — real, merged, but its own real ~17%
  measured overhead means it is opt-in and mostly unused by default;
  does not separately measure all-reduce time under DDP (folded into
  `backward_seconds`) or true GPU idle/stall time (approximated by
  average sampled utilization instead) — both would need real multi-GPU
  verification this instrumentation has not had.
- **Auto-revert on failed canary** — achieved structurally, not as a
  dedicated feature: `engine.py::promote()` only overwrites the baseline
  when `GateDecision.accepted` is true, so a repair that fails its
  independent holdout eval simply never replaces the working baseline.
  `test_dataset_regression_repair.py::test_training_regression_repair_that_
  fails_to_improve_is_not_promoted` (PR #88) now drives a *failing* repair
  through this exact path end to end for the training-regression repair
  flow specifically, and `test_search_controller_integration.py::test_a_
  repair_that_fails_its_own_gate_never_replaces_the_baseline` (PR #90) does
  the same for the older eval-gate repair flow (`autonomous_repair.py`).
  There is still no monitored post-promotion rollback (Chowder never
  "deploys" before evaluating, so there is nothing to roll back from yet).
- `checkpoint_discovery.py` is real and tested, but solves *resume
  compatibility validation* for the TUI, not bisection — don't confuse it
  with checkpoint bisect (`checkpoint_bisect.py`, PR #79, now PROVEN under
  Priority 5 above).
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
- **Unsloth PEFT backend — minimal isolated executor, real-CUDA-
  commissioned once, not yet production-proven** — an explicit PEFT
  engine-selection seam (`backend_selection.py`, PR #93: `backend.type:
  peft` + `backend.engine: transformers|unsloth`, no `auto` mode) plus a
  real isolated Unsloth runtime: `chowder setup unsloth`/`chowder doctor
  unsloth` (`unsloth_env.py`, PR #94) create/verify a separate `uv`-managed
  Python 3.13 environment under `.chowder/envs/unsloth`, invoked only
  through a subprocess — Unsloth, its patched Torch, TRL, bitsandbytes,
  and Triton are never imported into Chowder's own controller process, by
  design (different, incompatible dependency envelope from Chowder's
  tested `[train]` stack). `training_data.py` (PR #96) extracted the
  backend-neutral dataset-digest/replay/chat-tokenization contract out of
  `transformers_worker.py` so a second training backend has a real
  contract to share instead of a temptation to duplicate it. `unsloth_
  peft.py`/`unsloth_worker.py` (PR #97) implement the actual isolated
  executor against the existing `TrainingExecutor` contract (profile/run/
  cancel, cancellation binding, progress polling) — text-format datasets
  only in this slice (the isolated worker cannot import
  `chowder.backends.training_data`, so chat support needs a deliberate
  cross-environment data-handoff design, not a guess), and refuses
  `activation_offload`/`optimizer_tiering`/`frozen_layer_streaming`
  outright rather than risk an unverified interaction with Unsloth's own
  patched attention/model implementation. **Real-CUDA-commissioned on
  this project's own target hardware** (RTX 5060 Ti, Blackwell, Windows):
  a real `chowder setup unsloth` produced a fully green `chowder doctor
  unsloth` (real Unsloth/Torch/CUDA/bitsandbytes NF4 forward pass), and
  the first real end-to-end training run caught one more real bug before
  it could ship broken — unlike plain PEFT's `LoraConfig(target_modules=
  None)`, Unsloth's own `FastLanguageModel.get_peft_model` does not
  auto-detect target modules at all and crashes on `None` — fixed by
  defaulting to Unsloth's own documented Llama-family target list (PR
  #98). After the fix, a real training run completed real steps and
  produced a genuine, standard, independently-loadable PEFT adapter.
  Checkpoint/resume and cancellation are also real-CUDA-commissioned
  (PR #100): `UnslothPeftRunSpec` gained `save_strategy`/`save_steps`/
  `save_total_limit`/`resume_from_checkpoint`, with a checkpoint manifest
  that additionally binds to the isolated environment's own manifest
  digest (so a rebuilt/different-version environment is refused, not
  silently trusted) and whose filename alone is the entire mechanism that
  rejects a Transformers checkpoint resumed under `engine='unsloth'` or
  vice versa. A real, mid-flight training run was cancelled after 8 real
  seconds and confirmed fully gone from the OS process table (nvidia-smi
  `--query-compute-apps` was found unreliable on this Windows/WDDM
  machine for that specific check); a real second run then resumed
  correctly from a real, earlier real checkpoint's saved step. Independent
  -evaluator integration is proven end to end (PR #101): a real project
  with `engine='unsloth'` runs through the unmodified `run_project()` ->
  `TransformersTextEvaluator` -> hard gate -> registry pipeline, catching
  two more real, previously-latent bugs in `project.py` along the way —
  a stale hardcoded rejection of `engine='unsloth'` left over from before
  the executor existed, and project validation unconditionally using the
  Transformers-only config schema/spec for every engine regardless of
  which one was actually selected. **Real-target-model commissioned**
  (`docs/UNSLOTH_REAL_CUDA_ACCEPTANCE.md`): the actual model from the
  real prior training campaign (resolved from this repo's own
  `chowder-project.json`, not guessed from shorthand) --
  `Goekdeniz-Guelmez/Josiefied-Qwen3-8B-abliterated-v1`, a real ~8B-param
  Qwen3 model -- trained for real via 4-bit QLoRA on this hardware: a
  25-step pilot (91s, 6.5 GB peak VRAM), then a real resume from that
  pilot's own checkpoint to 150 total steps (loss 1.34 -> 0.34, genuine
  continued learning), producing a real, standard, independently-loadable
  PEFT adapter throughout. Uses a text-format pilot dataset rather than
  the original campaign's chat-format one (chat support is still
  deferred, see below), so this does not reproduce that campaign's exact
  task -- it proves the real target model and real checkpoint/resume work
  correctly through the Unsloth engine at real scale, which it does.
  **Still not fully production-proven**: chat-format datasets and
  continuing from a parent adapter remain explicitly deferred (the
  isolated worker cannot import `chowder.backends.training_data`); real
  process-tree-safe cancellation hardening (e.g. a Windows job object) has
  not been added since the current single-process worker has not been
  shown to need it; and a real, honest anomaly surfaced during the
  150-step resume (the real on-disk checkpoint cadence didn't match the
  requested `save_steps`, conservatively saving *more* often than asked,
  likely an inherent `transformers.Trainer` resume characteristic rather
  than anything Unsloth-specific) has not yet had a dedicated root-cause
  investigation.

## NEXT

**Status as of this section's last real-work pass**: every concrete item
below that was actionable without new hardware has been closed for real
(Qwen-shape MoE router/expert instrumentation and frozen-layer backward
prefetch, both this pass; Unsloth integration, activation-offload stride
fix, and Memory Fabric's core OOM-to-success claim, earlier passes). The
one item still open in this section — matched multi-GPU telemetry — is
blocked on hardware this machine does not have (an asymmetric 2-GPU box
does not substitute; see that item for why substituting would misrepresent
real DDP behavior) and is not something further engineering effort on this
machine can close. The `## RESEARCH` section below (Priority 6 meta-
controller, further Elastic MoE phases) is explicitly open-ended and, per
its own header, gated on these proven foundations being stable rather than
a precondition for that stability — it is not a shippability blocker.

**Final Memory Fabric acceptance test (Priority 1 follow-up) — core claim
demonstrated for real; not yet a reliable committed test**
The milestone before Memory Fabric can be called production-proven: a real
workload that genuinely CUDA-OOMs under normal resident training, then
genuinely succeeds under the same model/recipe with Memory Fabric's real
mechanism applied — not faked by lowering the reported VRAM budget. Full
real-hardware attempt log, findings, and next steps:
[`docs/MEMORY_FABRIC_ACCEPTANCE.md`](MEMORY_FABRIC_ACCEPTANCE.md). Short
version: **this has now genuinely passed, repeatedly** — using
`torch.cuda.set_per_process_memory_fraction` (a real, in-process allocator
constraint, not a reported-hardware lie) to bypass this development
machine's driver-level VRAM-to-system-RAM paging fallback without touching
any system setting, real Qwen2.5-1.5B/fp32/LoRA r=8 training at batch=8 was
shown to genuinely, cleanly `torch.cuda.OutOfMemoryError` resident (measured
peak 18.7 GB, already exceeding the 15.93 GiB card) while
`activation_offload: "always"` genuinely succeeded under the identical
constraint (measured peak 9.3 GB) — the exact same model, recipe, and GPU.
What keeps this from being a committed, always-green regression test yet: a
newly surfaced, real Windows/WDDM driver flakiness
(`CUDA error: resource already mapped`) intermittently interrupts
`activation_offload`'s real CPU↔GPU transfers under memory pressure on this
specific machine — the same error class already flagged (but not explained)
during the stride-alignment investigation, now confirmed to recur here too,
independent of the specific VRAM ceiling. A mechanism's isolated single
-forward+backward savings not reliably predicting a full training run's real
peak VRAM was also confirmed a third time and remains an open, separate
limitation.

**Backward prefetch for frozen-layer streaming (Priority 1 follow-up) — done, real-hardware measured**
`memory_fabric.py`'s backward now prefetches layer i-1's weight one layer
ahead while layer i's backward is still computing, via
`FrozenLayerPrefetchRuntime.start_backward`/`take_backward` (the same
dedicated-CUDA-stream + `record_stream` design forward's existing prefetch
uses, walked in decreasing index order since backward visits a sequential
frozen-layer stack in that order regardless of what unrelated backward nodes
run in between). `StreamedFrozenLayers`/`stream_frozen_layers` take a
`backward_prefetch: bool = True` parameter; `False` retains this module's
original synchronous re-stream verbatim as an explicit fallback. Wired into
the real Trainer path in `transformers_worker.py` (`accelerator.backward` is
wrapped to call `start_backward()` immediately before the real backward
call) and into `frozen_layer_streaming_worker.py`'s calibration harness.

Real measurements (`tests/test_memory_fabric.py`, `CHOWDER_REAL_ML_SMOKE=1`,
RTX 5060 Ti), on a 12-layer synthetic PEFT-shaped stack sized so per-layer
H2D transfer (64MB/layer at dim=4096, fp32) and per-layer backward compute
are comparable (the production tiny smoke-test model is too small for
either cost to be visible against the other, per
`frozen_layer_streaming.py`'s own documented caveat):

- **Throughput**: median backward wall time 124.7ms with prefetch vs.
  226.4ms without — a real 1.82x speedup, not assumed.
- **Correctness**: bit-identical loss and gradients between
  `backward_prefetch=True`/`False` and a fully resident run, including
  across 5 repeated iterations (checked for the same stream-reuse race
  forward's prefetch already guards against).
- **VRAM**: `backward_prefetch=True` uses one extra layer's weight
  resident at a time versus `False` (a 64MB lookahead buffer at this size,
  ~3% of this synthetic stack's ~1.9GB peak) — an expected, bounded cost of
  the lookahead itself, not a regression relative to fully resident
  training's 0.75GB frozen-weight-only footprint (streamed keeps at most 2
  of 12 layers' weights resident either way). At this synthetic stack's
  size, activation memory (~0.75GB, one relu output per layer, unrelated to
  streaming) dominates the *total* peak enough that the original
  forward-only design's 13.5%-total-peak-reduction claim (measured on a
  different, activation-light synthetic stack) does not directly transfer
  to a throughput-oriented, activation-heavy shape like this one; the
  frozen-weight-only footprint reduction (~128MB streamed vs. 768MB
  resident) is real and unchanged either way.

Not yet done: a real Trainer-level (not synthetic-stack) throughput
measurement, since the production tiny smoke-test model remains too small
to show a meaningful signal (see above) and no larger production model has
been benchmarked this way yet.

**Multi-GPU telemetry (Priority 2, deferred slice)**
Real GPU↔GPU bandwidth/topology measurement, PCIe/NVLink capability
measurement, P2P availability, all-reduce timing, and DDP communication
share of backward — blocked on **matched** multi-GPU hardware access (a
locally available but asymmetric 2-GPU box does not substitute for this;
inferring symmetric-pool numbers from mismatched cards would misrepresent
real DDP behavior). Revisit if/when Kaggle-2×T4-class access is arranged,
the same way Phase 5's DDP acceptance was.

## RESEARCH

Remaining policy and architecture research is gated on the proven foundations
above being stable:

- **Meta-controller policy learning** (Priority 6) — the evidence-view slice
  is complete above, both halves: scored outcomes (`intervention_outcomes.py`)
  and censored outcomes (`censored_outcomes.py`, this pass, which closes the
  roadmap's own "define how censored failures/cancellations enter the
  dataset" gate by representing them explicitly and documenting the policy
  position rather than imputing scores). Still not started: the
  expected-improvement model, GPU-hour-aware experiment policy, and
  cross-model transfer of successful training strategies. The required
  hardware/dataset context gaps are closed (dataset identity/scale and
  accelerator context now live in the evidence view); remaining before
  training a selector: validate against held-out experiments versus the
  existing UCB1 baseline with zero hard-gate violations (production runs
  now persist executor-failure incidents, so the censored view's crash
  classifications are actual for current runs). A durable historical dataset is not
  itself a learned policy.
- **Elastic MoE research** (Priority 7) — per-expert load/gradient
  statistics, expert specialization diagnostics, safe expert clone/split
  experiments, router retraining/distillation, architecture-change
  promotion gates kept behind strict regression and compute-budget gates.
  docs/MOE_DOWNSIZING.md's "First implementation slice" (Phase A/B: audit,
  calibration capture, `expert_importance.jsonl`, dry-run pruning plan) is
  now real code (`src/chowder/moe_instrumentation.py`,
  `chowder moe expert-importance`), verified against the actual installed
  transformers==5.16.1 Qwen3Moe/Qwen3_5Moe/Olmoe source (a fused
  batched-expert design, not per-expert submodules — see that doc's Phase A
  section) and real-hardware-validated end to end against the local
  OLMoE-1B-7B checkpoint (16 layers × 64 experts, real router hooks, real
  per-expert gated-activation/output-norm math, real dry-run 75%/50%
  pruning plans). What remains genuinely open: no local Qwen3.6-35B-A3B
  checkpoint exists on this machine (exhaustively searched), so the actual
  named target is not yet commissioned — only the mechanism is proven, on a
  real architecturally-equivalent stand-in. Phases C–F (budget search,
  distillation, mixed precision, promotion gates) have not started.
- **Qwen3.8 Native Sparse Program** (Priority 0 — the model program;
  see [`docs/QWEN38_SPARSE_PROGRAM.md`](QWEN38_SPARSE_PROGRAM.md) for the
  pinned parent manifest, the architecture audit, and the milestone-1
  checklist that gates "underway" claims). All four parent revisions are
  pinned; parent B's auto-gate was accepted with the account's token and
  its full architecture audit is recorded (dense `qwen3_5`, 64L/5120h,
  MTP 15 tensors, vision 333 tensors, `Qwen2Tokenizer`, 18 shards /
  51.7 GiB / zero GGUF). The protected nine-dimension evaluation harness
  is implemented in `src/chowder/parent_eval.py` with real tests:
  complete-coverage spec validation, capability/behavior separation by
  construction, a protocol fingerprint excluding candidate identity, a
  fail-closed tokenizer-identity gate, hash-only protected indexes, and
  FK-anchored persistence into `evaluation_runs`. Still open before the
  Phase-4 tournament: parent weights on disk, protected suite content,
  and the evaluation runs themselves. The Phase-6 conversion plan is
  generated and merged
  ([`docs/PHASE6_CONVERSION_PLAN.md`](PHASE6_CONVERSION_PLAN.md), PR
  #120): partition-conversion of the dense FFN's intermediate dimension
  into experts. The converter and exactness harness are implemented
  (`src/chowder/dense_to_moe.py`, `src/chowder/conversion_exactness.py`;
  validation-ladder stages 1–2 done — tiny-fixture forward measured
  within the documented association gate, f32 max_abs 1.341e-07 / bf16
  1.953e-03, bitwise dense recovery, exactly uniform routers — and the
  real fusion code path exercised on parent A's actual layer-0 bytes).
  Two plan claims were corrected as implementation errata (top-k = E is
  required for init exactness; only down_proj carries the ×E scaling).
  The full parent-A conversion has not run (~52 GiB output; a disk
  decision).
  Phase 11 accounting is implemented
  (`src/chowder/parameter_accounting.py`, exposed as
  `chowder moe account-parameters`): real safetensors-header
  census (stdlib-only, index cross-checked, fail-closed on unknown
  dtypes and missing top-k), with `a_label()` refusing to exist without
  measured routing geometry. Measured on the cached parent A: 27.78B
  total parameters, dense floor 9.78B active/token — correcting the
  plan's ~10.55B estimate.
- **Teacher Fabric / Remote Intelligence Distillation** (Priority 8) —
  architecture documented in
  [`docs/TEACHER_FABRIC.md`](TEACHER_FABRIC.md) (provider-neutral design,
  data contracts, integration map, threat model, benchmark protocol, open
  research questions). **Slice A is implemented**:
  `src/chowder/teacher_fabric.py` holds the signal taxonomy, capability
  declaration/negotiation over a registry, request/signal/artifact schemas
  with fail-closed validation and canonical digests, the
  `TeacherProvider` protocol (mirroring `TrainingExecutor`'s
  profile/query/cancel shape), and a deterministic offline
  `FakeTeacherProvider` test double — with the tokenizer-compatibility
  gate failing closed on every token-aligned signal kind (rejection or an
  explicit caller-invoked downgrade, never approximation) and zero
  network code. **Slice B is implemented**:
  `src/chowder/teacher_signal_store.py` is the content-addressed,
  budgeted signal store — atomic writes with interrupted-write recovery,
  verified-or-absent reads (payloads re-hashed on every read; corruption
  is refused, never served), exact dedup over
  `(request_digest, payload_file_sha256)`, a required no-default
  `local_cache_max_bytes` with a measured footprint, explicit caller
  eviction only, and registry migration 4 adding the append-only
  `teacher_signals` ledger (evidence survives cache eviction; identical
  re-acquisition replays idempotently). Slices C–J (black-box repair
  integration, cost accounting/query controller, selected-token scorer,
  real remote commissioning, remote jobs, microjobs, multi-teacher,
  selection research) **have not started**; no real provider is
  commissioned and no student-improvement claim exists — none has been
  measured. The hard
  regression gate remains the sole promotion authority and is untouched.
