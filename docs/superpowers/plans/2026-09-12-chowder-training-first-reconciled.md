# Chowder Training-First, Evidence-Complete, A3.5B Successor Implementation Plan

> **For agentic workers:** Implement task-by-task using the installed Plan Architect Execute workflow or the available executing-plans/subagent-driven-development skill. Read the applicable skill before execution. Checkboxes below describe future work, not completed changes. Re-read live source before applying any proposed edit.

**Goal:** Chowder completes one bounded router-training experiment through its normal saved-project interface, with measured preflight, demonstrated trainability, resumable state, independent evaluation, and durable accounting; then use that path to pursue a useful successor with **<=10B total / <=3.5B active parameters**.

**Architecture:** Extend Chowder's existing executor, project runner, experiment cycle, registry, checkpoint, evaluation, and resource contracts. The router workload gets a narrow built-in backend and a verified delta loader, not another standalone training driver or experiment database. Frontier supplies research targets and protocols; Ornith remains a separately governed proposer, not a replacement for Chowder's training system.

**Tech stack:** Existing Python, PyTorch, Transformers, PEFT/Unsloth where independently qualified, safetensors, SQLite RunRegistry, and current CI. No new orchestration framework, distributed service, or dependency is planned.

**Status:** PLAN ONLY, 2026-09-12. This document creates no training authorization, compute reservation, promotion, merge, environment change, or scheduled task. The older September 11 plan and all research artifacts remain preserved.

## 1. Decision summary

The next milestone is still:

> **Chowder completes one bounded router-training experiment through its normal interface.**

Do not declare that milestone complete from a helper test, a dense LoRA run, a model load, or a smaller parameter-count calculation. First qualify the reusable path on tiny real models, then exercise the actual 9B-derived router workload. Tiny-model success is an intermediate engineering result; the actual workload can still fail memory or cost preflight.

Three separate verdicts must remain visible:

| Verdict | What establishes it | What it does not establish |
|---|---|---|
| Training-path qualified | All five user acceptance conditions, including a real resume and independent application of the saved delta | Useful reasoning or a successful sparse architecture |
| Architecture/quality qualified | Frozen parameter convention, <=10B total, <=3.5B active, actual efficient execution, and preregistered retained quality | Champion promotion or release permission |
| Champion promotion | Capability improves AND Fable behavior does not regress AND efficiency does not regress AND regressions pass, plus the existing statistical gate chain and human approval | Automatic permission to distribute weights |

An experiment may train correctly and be scientifically rejected. A 6/50 GSM8K score may satisfy its historical preregistration without being a usable model. Neither distinction should be erased by a single green status.

### What changed from the September 11 roadmap

- A real 500-step dense-pruned LoRA run now completed in Chowder. Reuse that foundation; do not rebuild basic training or repeat 500 steps to prove the same point.
- The completed run saved an adapter but **not resumable training state**. Full-lifecycle memory evidence and exact intended-component gradient evidence remain incomplete.
- The current router orchestration helper exists, but its injected callbacks are not a production router backend. Its delta read does not enforce delta application; its optional JSON resume dictionary is not a restore implementation.
- The direct preset, adapter-evidence, scorer-identity, and CI defects come before new research.
- Reuse an existing 9B-derived MoE artifact for engineering if it passes identity/load checks. Do not require another full conversion merely to repeat the negative hot-core hypothesis.
- The fraction sweep is exploratory generation evidence, not a sparse-compute implementation or a new passing checkpoint.
- Architecture planning can use cheap header/meta calculations in parallel. Expensive architecture recovery and campaign expansion wait for the router milestone.

## 2. Evidence baseline and preservation boundary

Primary assessment: [September 12 audit](C:/Users/nikma/Chowder-Protected/runs/2026-09-11-gsm8k-continuation/AUDIT_2026-09-12.md).

Prior roadmap retained: [September 11 training-first plan](2026-09-11-chowder-training-first-9b-a35b.md). This document supersedes its execution order and stale repository/run status, not the historical evidence it records.

Source reviewed for this plan: `C:/Users/nikma/Chowder-router-healing`, clean at `661fabe76e13360f4fc49568708d0e5b8827f86b`. The handoff checkout holding this document is a separate documentation worktree, not the source revision to execute. Live source wins over either snapshot.

| Evidence | Preserve this interpretation |
|---|---|
| Original `F:/llm-models/_a4b/realtrain-gsm8k` | Failed at optimizer step 323 after a progress-file write failure; no candidate evaluation. Keep original preregistration `9f66960` and failure evidence. |
| Rerun `F:/llm-models/_a4b/realtrain-gsm8k-2` | Completed 500 steps; strict GSM8K 0/50 -> 6/50; actual total 3.463111 GPU-hours. Keep original preregistration plus pre-launch addendum `5de632d` and the recorded RECOVERS/local gate verdict. |
| Rerun limitations | Empty trainer-state directory; no resume demonstration; evaluation memory/headroom absent; token-cap/repetition behavior prevents a usability claim. Do not invent a retroactive unconditional PASS or FAIL. |
| Fraction sweep | Eight-prompt mask screen: 75% kept survives its rule; 56.25% does not. Masks retain full dense matrix work; metric definitions changed from the earlier control. This is not throughput evidence or independent-sample replication. |
| A4b hot-core comparison | Retain the measured negative PPL result and close that tested hypothesis. Do not generalize PPL to generative reasoning or repeatedly re-open the arm without new evidence. |
| PR158 | Audit snapshot: OPEN/BLOCKED; four jobs fail a floating-point equality assertion. Refresh checks before integration; this plan authorizes no merge. |

Protect the five unrelated frontier modifications: `evals/mmlu_pro_eval.py`, `kaggle/cycle4_math_kernel/run_cycle4_math_train.py`, `tests/test_mmlu_pro.py`, `training/adapter_delta.py`, `training/train_dpo_trl.py`. Compare their hashes to the audit inventory before and after implementation. Never broad-stage, reset, stash, or sweep them into a Chowder change.

Do not rerun the ledger migration on real research rows as a test. Do not edit old SQLite result rows to make lifecycle status look clean. Use read-only access for historical inspection and a consistent SQLite backup for preservation; test recovery/migrations on copies. Link corrections as new audit evidence.

The earlier prepared paired-control driver remains unrun and source-pinned to an older revision. Do not bypass its identity refusal or restart it automatically. The later completed control/sweep evidence must be reconciled first. The existing heartbeat remains paused.

Claude Desktop's native window is not directly accessible through the available computer interface. Repository changes, handoffs, logs, and artifacts are evidence; unseen chat content is not claimed as reviewed. Shared gbrain/second-brain tools were unavailable during planning.

## 3. Smallest implementation boundary

All source/test paths below are relative to the execution worktree rooted from the verified Chowder source. New files are explicitly marked **new**; all other paths already exist.

| Concern | Reuse / modify | Do not replace it with |
|---|---|---|
| User entry point | `cli.py` commands `project-validate` and `train`; `project.py`, `project_runner.py`, `backend_selection.py` | A separate router launch script or CLI family |
| Execution ownership | `executors.py`, `cycle.py`, `engine.py`, `registry.py`, `resources.py`, `cancellation.py` | Manual status writes and a second budget ledger |
| Process identity | `worker_env.py`, existing worker specs/results | Merely recording a Git label while workers import another checkout |
| Resume compatibility | `backends/transformers_peft.py` `_bound_inputs`, `_write_checkpoint_manifest`, `_verify_resume_checkpoint`; Unsloth counterpart; `checkpoint_discovery.py` | An untyped JSON dictionary with `default=str` |
| Model identity | `local_model_manifest.py`, existing file/artifact hashes | A mutable directory path or a fast size-only manifest presented as a weight digest |
| Router recipe and freeze | `router_healing.py`, `router_healing_run.py` | Suffix counts treated as empirical gradient evidence |
| Router execution | **New** `backends/router_healing.py`, **new** `backends/router_healing_worker.py`, using existing executor contracts | Expanding the current callable scaffold into a competing lifecycle |
| Evaluation | `evaluators/transformers_text.py`, `transformers_text_worker.py`, `base_text.py`, `base_text_worker.py`, `scoring.py`, `protocol.py` | A bespoke scorer or in-memory trainer-model evaluation |
| Measurements | `memory_preflight.py`, `backends/memory_preflight_worker.py`, `telemetry.py`, `evaluators/vram.py`, `parameter_accounting.py`, `sparse_accounting.py` | Calling all allocations “VRAM” or masked tensors “less active compute” |

The current `router_healing_orchestrator.py` is an evidence scaffold. Source inspection shows that it receives an already-loaded model before its timing boundary, does not use the normal resource reservation lifecycle, optionally serializes resume state using `default=str`, and discards the loaded delta before invoking an evaluator that receives no delta argument. Retain compatible helper behavior where useful, but remove its claim of demonstrated five-condition completion. The normal project/cycle path becomes the authoritative production route.

**Scope reduction:** parameter totals, router diagnostics, and memory are initially structured evidence, not fake prompt suites or a new general metric-plugin framework. The pilot's ordinary text metric can still receive Chowder's existing gate decision. A separate persisted engineering checklist qualifies the path; champion promotion is disabled. Architecture-specific hard constraints are added when the successor research path requires them, not by rewriting global PEFT gate semantics now.

## 4. Delivery order and parallel work

```text
P0 preserve and reconcile
  -> P1 preset + P2 adapter evidence + P3 portable CI       [disjoint edits may run in parallel]
  -> P4 immutable identity + P5 exact trainability         [coordinate shared worker edits]
  -> P6 full-lifecycle measurement and cost qualification
  -> P7 real save / interrupt / resume on existing PEFT
  -> P8 router worker + P9 verified delta / independent evaluation
  -> P10 normal-project lifecycle and fault closeout
  -> P11 tiny real model -> small CUDA -> actual 9B-derived router pilot
  -> R1 geometry decision -> R2 recovery -> R3 expert adaptation -> R4 qualification / promotion
```

Header-only architecture accounting and P12 historical-result triage can proceed alongside engineering after P0. They do not justify more GPU research. GPU jobs are strictly serialized; agents may parallelize code/review, not compete for the same devices. Use one owner for shared `project_runner.py`/worker edits.

Each task follows: add a failing regression, verify that it fails for the intended reason, make the smallest change, run focused tests, inspect the exact diff, then checkpoint the authorized slice. No broad refactor or dependency upgrade is bundled with these tasks. Each task ends with an evidence-backed handoff before the next dependent task starts.

### P0 — Preserve and reconcile the execution source

**Dependencies:** none. **Files:** no source edits; a new reconciliation inventory beside the next protected engineering run after execution is authorized.

- [ ] Refresh `git worktree list --porcelain`, HEAD, dirty/untracked inventory, remote PR head and checks. Identify any active editor/job before selecting an execution worktree.
- [ ] For each pending/local artifact classify it as already upstream, retained local work, deferred unrelated work, or superseded with preserved evidence. Record content hashes, paths, owner if known, and the integration decision.
- [ ] Preserve the old and rerun logs, preregistrations, deltas, conversion manifests, raw outputs, CSV backup, and registry databases. Verify a consistent backup rather than copying a live SQLite main file without its transaction state.
- [ ] Use an isolated implementation worktree from the agreed current revision. Do not mutate a checkout an active worker may import. Pin parent and child source/runtime identities in subsequent runs.
- [ ] Keep existing files in the five protected frontier paths unchanged. Do not integrate the entire older documentation branch just to obtain this plan.

**Verification:** inventory can account for every in-scope pending change; source/dirty hashes are stable; historical evidence remains readable; no process, dataset, or registry row was altered.

### P1 — Make the preset support the actual loaded text decoder

**Dependencies:** P0. **Modify:** `src/chowder/backends/transformers_worker.py`. **Test:** `tests/test_transformers_backend.py`.

- [ ] Parameterize the existing Qwen curated-list test over `qwen3_5` and the observed `qwen3_5_text` type. Call `_resolve_target_modules` on the actual small/meta decoder in the real-Transformers test, not only on its leaf inventory.
- [ ] Add the narrow alias to `_ATTENTION_AND_MLP_MODULES_BY_MODEL_TYPE`:

```python
"qwen3_5_text": _QWEN3_5_TARGET_MODULES,
```

- [ ] Retain explicit-target precedence and unsupported-architecture refusal, including the MoE text/wrapper types. This dense preset change does not establish MoE PEFT support.
- [ ] Verify all ten expected leaf families and their exact resolved paths for the representative 32-layer model; do not hardcode the number 200 as a universal model invariant.

**Run:** `python -m pytest tests/test_transformers_backend.py -q` in the pinned test runtime. **Exit:** both wrapper and loaded-text resolver paths pass; unrelated model presets are unchanged. If the local TorchAO/Triton import problem recurs, keep it an explicit environment blocker; use the existing real-CPU CI environment or a separate supported test environment, never mutate an active training environment to force a pass.

### P2 — Stop treating unreadable adapters as live

**Dependencies:** P0. **Modify:** `src/chowder/adapter_guard.py`. **Test:** `tests/test_adapter_guard.py`.

- [ ] Add readable-zero, readable-nonzero, unreadable, nonfinite, zero-key-overlap, and partial-key-overlap cases. The key invariant is that an exception must never increment verified-nonzero counts.
- [ ] Report verified-zero, verified-nonzero, unreadable/invalid counts and offending names separately. A wholly unreadable adapter must raise an evidence-incomplete error, not claim either liveness or inertness.
- [ ] Preserve the current matched-key failure. In the strict training qualification added in P5, require complete intended saved/live key coverage; one live B tensor is insufficient evidence that all intended components were loaded.
- [ ] Treat legacy results as potentially affected, not automatically invalid. Historical disposition is P12.

**Regression shape, using the existing test fixtures:**

```python
def test_unreadable_b_cannot_qualify_as_live(tmp_path):
    class Unreadable:
        def detach(self):
            raise RuntimeError("unreadable test tensor")

    class Model:
        def named_parameters(self):
            return [("layer.lora_B.default.weight", Unreadable())]

    directory = _write_adapter(tmp_path / "adapter", ["layer.lora_B.weight"])
    with pytest.raises(AdapterNotLiveError, match="unreadable|incomplete"):
        assert_adapter_is_live(Model(), directory)
```

**Run:** `python -m pytest tests/test_adapter_guard.py -q`. **Exit:** unknown never satisfies verified liveness; errors preserve the distinction between missing evidence and a measured zero adapter.

### P3 — Repair the portable CI assertion without weakening the science

**Dependencies:** P0. **Modify/test:** `tests/test_schedule_audit.py`; correct the exactness wording in `src/chowder/schedule_audit.py` if retained.

- [ ] Replace exact zero/infinite-separation expectations with tight numeric checks while retaining cosine classification and offset `-1`. For the existing normalized residual use `abs_tol=1e-12`, far below the classification tolerance; require separation at least the existing `MIN_SEPARATION`, including infinity as a valid value.
- [ ] Keep synthetic wrong-schedule, noisy, partial, and inconclusive controls. Add a minimally perturbed floating-point trajectory so the portability regression is exercised even on Windows.
- [ ] Correct “nine orders” to approximately five for `1.97e-9` versus `2e-4`. Do not change GSM8K recovery thresholds, recorded losses, or the captured LR fixture.

**Proposed assertion body:**

```python
assert verdict.matches("cosine")
assert math.isclose(verdict.residual_fraction_of_peak, 0.0, rel_tol=0.0, abs_tol=1e-12)
assert verdict.separation >= MIN_SEPARATION
assert verdict.offset == -1
```

**Run:** `python -m pytest tests/test_schedule_audit.py tests/test_lr_scheduler_honoured.py -q`, followed by the existing Windows/Linux matrix and real Transformers/PEFT CPU job. **Exit:** all required checks green; no scientific preregistration changes. A local Windows pass alone is insufficient.

### P4 — Bind source, model, data, and scorer identities

**Dependencies:** P1/P2/P3. **Modify:** `src/chowder/local_model_manifest.py` only if its existing full mode needs extension; `protocol.py`, `worker_env.py`, evaluator baseline/candidate spec/result plumbing, and checkpoint bound inputs in the existing backends. **Tests:** `tests/test_local_model_manifest.py`, `test_protocol.py`, `test_worker_env.py`, `test_transformers_evaluator.py`, `test_base_text_evaluator.py`, backend resume tests.

- [ ] Reuse `build_local_model_manifest(..., mode="full")` for the pilot's local base. Hash actual weight contents plus config/tokenizer/template/generation semantics; a fast manifest remains explicitly weaker evidence. Preserve path relocation separately from content identity.
- [ ] Publish an explicit scorer implementation version and content digest in both evaluator paths. Bind dataset bytes, split/item order, render/chat/thinking policy, generation settings, tokenizer identity, relevant dependency versions, seed, and performance measurement conditions.
- [ ] Keep candidate delta identity in the evaluation artifact envelope, not in the shared comparison-protocol fingerprint. For this baseline-plus-delta pilot, bind both runs to the same immutable base. Later comparisons across architectures require separately pinned model identities and a shared evaluation-procedure contract, not disabling protocol checks.
- [ ] Record parent/child resolved `chowder` module location and content/revision identity; fail before training when they disagree. Use a frozen execution checkout throughout the run, including later evaluator imports.
- [ ] Extend existing checkpoint compatibility with immutable base/trainable-set identity. Keep operational paths separate from mathematical recipe identity; do not gratuitously break valid relocation or the existing supported continuation modes.
- [ ] New evidence uses a new version. Old fingerprints and result rows remain unchanged and are linked to corrective audits.

**Verification:** changing scorer implementation, chat template, dataset order, or base bytes changes/rejects the appropriate identity; moving identical artifacts does not change their content identity; swapping a delta does not silently change the common evaluation protocol. Wrong base/scorer/code is refused before any training or score is accepted.

### P5 — Demonstrate exact intended-component trainability

**Dependencies:** P1/P2, coordinate with P4. **Modify:** `src/chowder/target_coverage.py`, existing worker evidence plumbing, `router_healing.py`. **Tests:** `tests/test_target_coverage.py`, `test_transformers_backend.py`, `test_router_healing.py`.

- [ ] Resolve the expected full module/parameter paths from the loaded architecture and declared recipe before adapter injection or freezing. Compare exact expected versus actual sets after injection; missing, extra, or unreadable required paths block qualification.
- [ ] Keep counts useful as a summary, not the proof. Include a regression where seven of eight expected `q_proj` paths exist: the old family-presence guard passes; strict qualification must fail.
- [ ] Observe gradients after backward and before clearing them, then record actual optimizer deltas over a declared multi-batch probe. For every intended component require finite, nonzero gradient at least once and a measurable update; distinguish `grad is None`, zero, nonfinite, unreadable, and update absent.
- [ ] Do not demand every parameter have nonzero gradient on the very first step. For example, initialization can delay a LoRA A update until B changes. The frozen probe window and intended-set policy must account for this without dropping failed targets after seeing results.
- [ ] For router-only mode, train exactly `mlp.gate.weight` per intended layer. Shared gates and experts are explicitly frozen for this milestone. Structural zero-path checks remain negative controls, not substitutes for real autograd.
- [ ] Use a tiny actual Qwen MoE with E=4, k=2 and non-identical expert outputs. Test normalized top-1 degeneracy and a zero frozen shared branch. Expert-row utilization is a separate diagnostic from per-tensor reachability.
- [ ] Verify frozen tensors do not change. Use complete hashes on tiny fixtures and a recorded, cost-accounted verification strategy on the full workload; do not silently replace full verification with sampled checks while claiming equality.

**Verification:** real forward/backward/update, not only `requires_grad`; missing expected path and deliberately dead required component each cause refusal. Run the three focused test modules, with the actual-model tests mandatory in the qualified ML environment rather than hidden among optional skips.

### P6 — Measure and budget the whole lifecycle

**Dependencies:** P4/P5. **Modify:** existing `memory_preflight.py`, preflight worker, `telemetry.py`, `resources.py`, executor/evaluator telemetry seams, `evaluators/vram.py`. **Tests:** `tests/test_memory_preflight.py`, `test_evaluator_vram_reporting.py`, `test_resource_accounting.py`, `test_transformers_resource_accounting.py`.

- [ ] Keep `profile()` non-computing: prior hash-bound measurements or an explicitly approved conservative reservation. Put model loading, warmup, and measured steps inside an accountable execution boundary or a separately registered calibration attempt.
- [ ] Inventory actual loaded tensor classes, logical shape/dtype, physical storage and quantization metadata, devices, trainable set, and optimizer representation. Raw expert parameters that remain BF16 must be reported as BF16 even when the requested loader says 4-bit.
- [ ] Measure load time, first forward/backward/update, steady-state step samples, checkpoint publication, reload, baseline generation, and candidate generation. Use accelerator synchronization where timing requires it; record its overhead.
- [ ] Capture PyTorch peak allocated/reserved, process CUDA memory when available, sampled minimum device free memory, other-process occupancy, and host RSS/commit separately. Record cadence and unavailable fields; sampled headroom is not proof of an unobserved instantaneous minimum.
- [ ] Refuse on measured/projected budget overflow or required-memory evidence missing before expensive continuation. GPU OOM, host-pressure refusal, NaN, timeout, cancellation, and telemetry failure each retain a structured outcome and incurred cost.
- [ ] Keep nonessential progress writes best-effort. Required safety measurements becoming unavailable cause a safe stop/qualification failure; neither case should silently report a successful zero-cost run.
- [ ] Retain both measured actual usage and the engine's conservative failure charge. Budget is not just optimizer time; baseline/evaluation dominated the completed GSM8K rerun.

**Verification:** injected missing telemetry is unknown, not zero; failed calibration is charged; forecast includes independent evaluation; cancellation releases reservations once; estimator-versus-actual differences are durable. Do not claim hardware fit from the CPU tests.

### P7 — Qualify existing PEFT save/interruption/resume before porting it

**Dependencies:** P4/P5/P6. **Modify only as needed:** existing PEFT backends/workers and checkpoint discovery. **Tests:** existing backend resume suites, `tests/test_checkpoint_discovery.py`; **new** `tests/test_training_resume_end_to_end.py` for a real small-model continuation witness.

- [ ] Use the existing checkpoint manifest and `resume_from_checkpoint` route. Keep inference delta and resumable checkpoint distinct. Inventory optimizer, scheduler, scaler if used, global step, CPU/CUDA/Python/NumPy RNG, and dataset/sampler position state.
- [ ] Checkpoint at optimizer boundaries for the first qualification. Publish a complete checkpoint marker/manifest only after required files are durable. Partial writes or missing optimizer/RNG state must be rejected, not treated as an implicit fresh start.
- [ ] Run the deterministic CPU comparison: uninterrupted steps 1..8 versus a run configured for 8 total steps, interrupted after checkpoint 4, then restored in a fresh process to step 8. Keep the same total scheduler horizon; training “4 steps then changing the horizon to 8” is a different schedule and is not a valid exact-resume control. Pre-register the CPU comparison's tolerance policy before writing the test: bitwise for reduction-order-deterministic quantities (final tensors, step counters, consumed sample IDs), and an explicit abs/rel tolerance (never a bare `==`) for any float recomputed across processes — the same class of defect that broke the schedule-audit CI on Linux while passing Windows (fixed 2026-09-12 with `abs_tol=1e-12` vs `MAX_RESIDUAL_FRACTION=1e-2`).
- [ ] Compare final tensors, optimizer/scheduler state, consumed sample IDs/order, step counters, and LR sequence. Repeat resume discovery and ensure there is no duplicate charging or duplicated data application.
- [ ] Test graceful cancellation separately from abrupt worker death. The former may save a final complete boundary; the latter may only retain the last checkpoint. Both keep the original attempt's terminal reason and costs.
- [ ] Preserve supported deliberate longer-horizon continuation as a separate operation with recorded revised schedule/budget. Do not globally remove existing compatibility behavior just to simplify the fixed-horizon equivalence test.
- [ ] After CPU proof, perform a separately authorized, short real dense-pruned GPU qualification, not another 500-step GSM8K campaign. Pre-register tolerance, checkpoint cadence, data, and maximum cost; retain independent reload and output-change evidence.

**Framework contract:** PyTorch distinguishes a model state from a general checkpoint containing optimizer and continuation information; Transformers accepts a checkpoint path to restore model/optimizer/scheduler state. These APIs are the foundation, not evidence that every Chowder workload already resumes correctly. See [PyTorch checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training) and [Transformers Trainer.train](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.train).

**Exit:** fresh-process continuation actually matches the fixed-horizon control under the frozen tolerance. An adapter directory, `trainer_state.json` alone, or a stringified optimizer object is insufficient.

### P8 — Add a narrow real router backend

**Dependencies:** P7; reuse P4-P6 contracts. **New:** `src/chowder/backends/router_healing.py`, `router_healing_worker.py`, `tests/test_router_healing_backend.py`. **Modify:** `router_healing.py`, `router_healing_run.py` only where the real worker needs stricter recipe/state handling.

- [ ] Implement the existing `TrainingExecutor.profile/run/cancel` contract. Parent owns registry/resource lifecycle; worker owns loading, freeze policy, optimizer steps, checkpoints, and structured results. Return `TrainingArtifact` plus `ResourceUsage`; do not manually create a parallel registry lifecycle.
- [ ] Use the installed Transformers/PyTorch path first. Share only existing suitable helpers; avoid inheriting PEFT-specific adapter assumptions or refactoring both existing engines into a new framework. Unsloth custom-MoE support is not assumed from its successful dense LoRA run.
- [ ] Version the recipe: immutable base/tokenizer/corpus identities, exact trainables, fixed k, seed, loss, router normalization/temperature, optimizer/scheduler, precision/placement, checkpoint and hard step/token/time limits. Reject unsupported modes and nonfinite configuration values before loading.
- [ ] Start with router-only language-model cross-entropy, k=2, experts/shared path frozen, and no new auxiliary-loss term. Validate that the actual forward has a usable router gradient. If it fails, stop and revise the recipe explicitly; do not silently add a balancing loss or change k during the run. Pre-declare the revision ladder so a stop does not become an invitation to ad-hoc exploration — permitted next recipes, in order, are: (a) verify gradient transport through the router path first (normalization/temperature placement, dtype/precision under bf16) with zero recipe change; (b) switch routing to fp32 compute while keeping the same loss; (c) add a load-balancing auxiliary loss as a versioned recipe change; (d) revisit k. Anything outside this ladder requires a new written recipe before any run.
- [ ] Calibrate within the budgeted attempt. Any calibration optimizer updates count toward its step/token limit and state; reset/read-only probes also consume time and cost. Capture per-layer router updates and utilization.
- [ ] Use the P7 checkpoint semantics with the router tensor payload; no optional `resumable_state={}` success path and no `default=str` serialization of tensors.

**Verification:** a tiny real MoE completes updates, saves, resumes, reports exact trainables, preserves frozen weights, and returns an ordinary artifact to the parent. Test dead router signal, malformed recipe, wrong source identity, and worker cancellation. GPU loading is forbidden in `profile()` tests.

### P9 — Make delta application and independent evaluation real

**Dependencies:** P8 and P4. **Modify:** `src/chowder/router_healing_run.py`, evaluator spec/worker seams. **New:** `tests/test_router_healing_checkpoint.py`. **Extend:** `tests/test_router_healing_run.py`, `test_transformers_evaluator.py`.

- [ ] Use safetensors for new router tensor payloads plus a versioned manifest. Declare whether stored tensors are replacement trained values or additive differences; the first implementation uses replacement values and applies them exactly once. Preserve the historical JSON format as legacy evidence, not a resume checkpoint.
- [ ] Hash every payload and bind base, recipe, trainable names/shapes/dtypes, and effective routing configuration. Validate all fields, finite values, missing/extra tensors, and content hashes before mutating the loaded model.
- [ ] Publish immutable, non-overwriting completed artifacts. Interrupted publication must not produce an eligible candidate. Safely reject stale/partial/tampered artifacts.
- [ ] Extend the existing text evaluator with an explicit `artifact_kind='router-delta'`. In a new process, load the verified base, apply the verified router values, assert live tensor equality against the payload, and evaluate with the P4 protocol.
- [ ] Require distinct worker identity/process evidence and no surviving training model. Pass the actual artifact reference to evaluation; reading and discarding its JSON does not qualify.
- [ ] Identity-delta control: trained values equal to base values reproduce base logits/output within the frozen tolerance. Nonidentity control: a deliberately effective tiny-model router change alters logits and is observed after fresh reload. Removing/tampering with that delta must fail or restore the baseline control, not leave a falsely claimed applied adapter.
- [ ] Retain full raw generations, token IDs/counts, stop reason, score inputs, repetition metrics and their precise version. PPL is supplementary, not a replacement for generative reasoning or usability.

**Verification:** fresh subprocess scores the loaded delta; all mismatch controls refuse before scoring; the common protocol stays equal for the baseline/candidate pair while artifact identities differ.

### P10 — Connect the normal interface and make terminal accounting reconstructible

**Dependencies:** P8/P9. **Modify:** `src/chowder/backend_selection.py`, `project.py`, `project_runner.py`, `cycle.py`, `run_events.py`; reuse `engine.py`, `registry.py`, `cancellation.py` and make only demonstrated missing-invariant repairs. **Tests:** existing project/backend/cycle/registry/cancellation/resource suites; **new** `tests/test_router_healing_end_to_end.py`.

- [ ] Add explicit `backend.type='router-healing'` selection without changing PEFT defaults. `chowder project-validate` validates the complete saved spec before training; `chowder train` runs it through the existing experiment cycle. No new launcher command.
- [ ] Disable automatic PEFT-shaped repair and champion replacement for engineering fixtures. Record the ordinary candidate gate decision plus a separate engineering-qualification evidence object with all five conditions and missing reasons.
- [ ] Store training artifact, complete evaluation outcome, immutable gate/adjudication artifact, source hashes, measured/charged cost, and terminal cause using existing registry evidence/artifact facilities. Add schema only if a specific required value cannot be represented.
- [ ] Finish/reconcile automatic baseline lifecycle status as well as candidate status in new runs. Capture final reservation/cost counters in closeout; a `planned` baseline with completed evaluation must not recur unnoticed.
- [ ] Distinguish cancellation, interrupted resumable attempt, infrastructure failure, evaluator failure, quality rejection, and engineering qualification. Reuse existing status enums plus structured reasons where sufficient; do not conflate a wrong answer with an execution incident.
- [ ] Respect existing immutable experiment versus run/attempt identity semantics. A retry or continuation must link the checkpoint parent and use an appropriate new run/attempt identity; never overwrite a terminal experiment under a reused deterministic ID. Preserve exact-idempotent ingestion and refuse divergent duplicates.
- [ ] Refactor the callable router scaffold into helpers/delegation to the authoritative lifecycle where practical, or clearly keep it non-production. Do not leave two supported budget/accounting implementations.

**Fault matrix:** cancel before load; cancel after checkpoint; kill worker after checkpoint; OOM during load/backward; failed optional progress write; failed required checkpoint write; evaluator death; scorer/base/delta mismatch; repeated closeout; duplicate resume; budget exhaustion. Every case preserves usable artifacts, charges spent work, avoids inappropriate promotion, and leaves no stranded reservation.

**Exit:** close and reopen a temporary registry and reconstruct the whole attempt, cost, checkpoint lineage, gate, and all five condition states solely from persisted records/artifacts. No “pass” exists only in the returned Python object or chat.

### P11 — Run the bounded acceptance ladder, then stop and assess

**Dependencies:** P10 and green required CI. **Artifacts:** new protected run directory with frozen project JSON, preregistration, manifests, worker outputs, checkpoints, independent evaluation, and registry closeout. No new standalone training script.

1. **Tiny actual-model CPU integration.** Use the real model class and production worker, E=4/k=2, short corpus, actual optimizer updates, interrupt/resume, separate evaluator process, and reopened registry. Ordinary mocks remain unit tests, not the end-to-end proof.
2. **Small-shape CUDA integration.** Use the intended loader representation and the same worker path. Verify gradient transport through frozen/quantized operations, save/resume, memory scope, and a cancellation control. Freeze CUDA tolerances before the run.
3. **Actual 9B-derived router workload.** First inspect and full-manifest the existing `F:/llm-models/Qwen3.8-9B-HotCore-CW-E16-k2-h2176` artifact and its conversion provenance. Its observed config is a 32-layer `qwen3_5_moe_text` decoder with E=16/k=2; that is candidate engineering input, not accepted weight integrity, fit, quality, or A3.5B compliance. Reuse it if compatible. Do not reconvert or reopen hot-core quality optimization merely because this fixture exists.

Proposed engineering limits for the actual-workload run: at most **16 optimizer steps**, batch 1, accumulation 1, sequence length 128, at most **2,048 processed training tokens**, checkpoint every 4 optimizer steps, and a separate **4-item / 256-new-token** baseline/candidate evaluation. These are a new engineering protocol, not the historical 50-item GSM8K protocol, and are not capability evidence. If gradient coverage cannot be demonstrated within this fixed window, report failure/inconclusive and revise a later protocol explicitly.

Memory expectation, stated before launch so a preflight refusal is expected and legible, not a surprise: with 16 experts and k=2 at hidden 4096 and intermediate 2176, the frozen expert stack alone is roughly 32 × 16 × (3 × 4096 × 2176) ≈ **13.7B BF16 parameters (~27 GB)** before activations, KV, optimizer router state, or the language head. Router-only training therefore likely cannot fit the full-precision expert stack on a single local GPU even though it trains only ~1.4M router parameters. If measured preflight confirms this, the pre-approved fallback is the same recipe with frozen experts in 4-bit/NF4 storage and router math in fp32 — a *representation* change to frozen tensors only, recorded as such — before any narrower geometry (R1) is considered. Treat an NF4 arm as a separate measured run, not a silent substitution.

- [ ] Freeze exact inputs, trainable paths, decode/stop/scoring settings, tolerances, checkpoint source, hardware, and stop policy before launch.
- [ ] Derive and approve wall/GPU-hour/memory limits from P6 measurements. These limits are deliberately not fabricated in this plan. No run starts without enough reserved allowance for baseline, training, checkpoint/restart, independent candidate evaluation, and closeout.
- [ ] Inspect GPU ownership immediately before each stage; do not stop another user's training, Ollama, or supervised service to obtain capacity. If occupied, report/defer; do not start a competing heavy job.
- [ ] If the full workload cannot fit or is too slow, preserve measured preflight refusal and bring back the hardware/representation options. Do not silently substitute the 27B model, a mock, or the tiny model and call the actual milestone complete.
- [ ] Reopen the registry, verify all five conditions against artifacts, and publish one terminal assessment. A poor or flat scientific score is permitted; missing engineering evidence is not.

**Definition of done:** one actual 9B-derived router attempt passes all five engineering conditions through saved-project validation/train; its cancellation/failure and continuation behavior is covered by the acceptance ladder; the actual-run delta and resume work in new processes; all costs/decisions survive registry reopen. Then stop expansion until the result is reviewed.

### P12 — Reconcile old Unsloth/VLM-wrapper results without deleting evidence

**Dependencies:** P0/P2/P4; read-only inventory can run alongside P5-P10. **Files:** existing evidence docs and a scoped audit manifest, not frontier's unrelated modified files.

- [ ] Inventory prior Unsloth results using VLM-wrapper models, prioritizing any used to choose/promote a candidate. Record base/adapter identities, loader class, saved/live key prefixes, worker revision, and whether a trained-adapter liveness or output-delta check actually ran.
- [ ] Mark each verified-applied, confirmed-inert, or unresolved. Do not treat a score equal to baseline as proof of inertness, or a changed score as complete key coverage.
- [ ] Perform header/key checks first. Any necessary generation reevaluation gets a new protocol/run ID, approved budget, and link to the old measurement; no overwrite or wholesale automatic rerun.
- [ ] Correct the sweep's changed-metric “jitter” wording, mask-versus-built limitation, overlapping prompts, LR orders, and speculative Windows handle attribution in scoped docs. Retain raw artifacts and original preregistered decisions.
- [ ] For PPL ratios, make the denominator explicit: a MoE value about 26% higher than prune corresponds to prune about 21% lower than MoE, not two interchangeable 26% claims.

**Exit:** influential historical results have a visible trust status; unresolved historical evidence cannot silently justify the next architecture or champion decision. This triage is not allowed to consume the entire router-integration milestone.

## 5. Five-condition acceptance matrix

| User condition | Tasks | Durable witness required |
|---|---|---|
| Actual loaded formats, memory, measured step cost | P4, P6, P8, P11 | Full base identity; runtime tensor inventory; cold/warm timing; correctly scoped memory; approved limit; charged calibration |
| Intended tensors train and each component can receive gradients | P1, P2, P5, P8 | Exact intended versus actual paths; per-component finite nonzero gradient/update over fixed probe window; frozen-weight integrity |
| Delta and resumable state | P7-P9, P11 | Hash-bound inference payload plus complete checkpoint; fresh-process fixed-horizon continuation with optimizer/scheduler/RNG/data state |
| Independent base-plus-delta evaluation, fixed versioned protocol | P4, P9, P11 | New process applies verified payload; identity/nonidentity controls; pinned protocol/scorer; full raw generation/stop evidence |
| Actual cost, cancellation/failure, gate in existing registry | P6, P10, P11 | Reopened registry reconstructs lineage, measured versus charged cost, terminal cause, gate and engineering decision; no stranded reservations |

## 6. Route to a genuine <=3.5B-active successor

### Non-negotiable feasibility calculation

Canonical target: [BENCHMARK-TARGETS.md](C:/Users/nikma/frontier-lowram-autoresearch/BENCHMARK-TARGETS.md). The actual parent tensor-header census is:

```text
text model total             8,953,803,264
dense FFN matrices           4,831,838,208
fixed non-FFN text floor      4,121,965,056
required active ceiling      3,500,000,000
```

Under the declared convention including full non-routed text parameter blocks, **the unchanged backbone is already 621,965,056 parameters over budget before any routed FFN executes**. Even a 4B interpretation would not fit. Quantization reduces bytes, not this parameter count. Removing inactive vision does not reduce this text-only floor. Keeping 75% of FFN channels would still leave a 7.746B text model.

The architecture ledger must enforce, per layer where necessary:

```text
P_total  = P_fixed + P_routers + P_shared + sum(all routed expert parameters)
P_active = P_fixed + P_routers + P_shared + sum(selected expert parameters)
P_total <= 10,000,000,000; P_active <= 3,500,000,000
```

Count unique tied weights once and declare input-embedding lookup versus full-table architectural accounting separately from FLOPs. Include adapters/shared branches in the appropriate ledger. Runtime top-k and actual dispatched work must agree with the recorded configuration. Masks and smaller k labels alone are not measured speedups.

### R1 — Choose one budget-valid architecture, not a search grid

**Dependencies:** inexpensive census/shape work may start after P0; full weight transformation/training waits for P11. **Reuse:** `parameter_accounting.py`, `sparse_accounting.py`, `dense_to_moe.py`, conversion manifests and existing accounting tests.

- [ ] Revalidate the earlier 3,072-wide shape witness in a supported isolated runtime. It was a meta-model calculation, not a trained checkpoint: 32 layers, E=16, expert width 768, shared width 512, k=2, unchanged vocabulary and untied embeddings; prior calculated total **6,476,770,432**, active **3,305,876,608**. At k=3 it was **3,532,369,024**, above the strict ceiling. These are historical witness values until the new emitted tensors are checked.
- [ ] Treat that narrower Qwen-derived text backbone as the first candidate to evaluate, not a guaranteed architecture or a directive to generate full weights now. Its roughly 194M active headroom must absorb any later added path.
- [ ] Compare alternatives only on a small decision sheet: lineage, fixed active floor, total/active budget, transform complexity, supported kernels/loader, memory/step forecast, and expected recovery requirement. Embedding tying, vocabulary changes, layer removal, or a different base require explicit new lineage/quality decisions; they are not free arithmetic tricks.
- [ ] Retain text-only scope for the first candidate. Preserved vision weights do not prove a width-reduced model retains multimodal behavior; MTP is counted only if corresponding tensors and execution exist.
- [ ] Accept exactly one geometry only after the actual shape/tensor ledger passes. A legal architecture is necessary, not proof of the ambitious benchmark targets or local compute sufficiency.

**Exit:** a reviewed architecture decision with emitted/meta shape checks, <=10B/<=3.5B ledger, supported execution path and bounded recovery proposal, or a documented rejection. No full-size campaign starts from parameter arithmetic alone.

### R2 — Create a valid smaller backbone and restore generation

**Dependencies:** R1/P11. **Implementation:** reuse the converter for expert mapping; add a narrowly scoped whole-backbone transform only if existing conversion cannot express it without conflating responsibilities. A separate reviewed implementation slice precedes full-size writes.

- [ ] Prove an identity transform first, then tiny reduced-shape transformations. Map residual axes, embeddings/output head, attention heads, GatedDeltaNet packed projections/convolution/state axes, norms, and FFNs consistently. Config-only width changes are prohibited.
- [ ] Preserve source checkpoint and source-to-destination tensor/channel maps. Write derived weights to a new directory, full-manifest them, and verify the ledger against actual emitted tensor shapes.
- [ ] Run matched dense-parent and transformed-model controls through Chowder with identical rendering, stop policy and scoring. Verify finite logits, valid tokenization, non-pathological generation, termination and repetition; capture full raw tokens. PPL is only a secondary diagnostic.
- [ ] Pre-register one bounded recovery arm using training-side data/teacher material with recorded provenance, no hidden evaluation leakage. Qualify its exact trainable set and resume path before investing in recovery.
- [ ] If controlled generation or teacher-relative retention fails beyond the frozen tolerance/budget, reject or revise the geometry. Do not spend larger training budgets merely because its active count is attractive.

**Exit:** a loadable budget-valid smaller architecture whose held-out generation/retention passes its preregistered recovery screen. This is still not final benchmark attainment.

### R3 — Learn useful routing and, if necessary, experts

**Dependencies:** R2. **Reuse:** P8-P10 worker/artifact/evaluation contracts; no separate research loop.

- [ ] Compare the selected untrained conversion with router-only training first. If frozen experts cannot recover sufficient quality under the bounded test, explicitly extend the trainable allowlist for expert adaptation in a separate tested slice.
- [ ] Verify real gradients/updates for fused expert parameters and any low-rank or quantized adaptation. Ordinary `nn.Linear` PEFT support does not establish support for raw fused expert tensors.
- [ ] Use one control and one challenger per decision, with matched measured compute/data budgets and a frozen hypothesis. Avoid multiplying k, expert count, core share, initialization, LR, seeds, and datasets into a Cartesian campaign.
- [ ] If shared experts or routing losses change, version the architecture/recipe and prove the new gradient path. Do not resurrect an all-zero multiplicative branch as trainable by changing its label.
- [ ] Track held-out capability, raw-generation usability, router utilization, dead experts, actual latency/throughput and cost. Expand to replication only for a finalist that passes the development screen.

**Exit:** one candidate meets the agreed development quality/efficiency screens under the active/total ceilings, or the tested arm is closed with preserved negative evidence.

### R4 — Independently qualify, promote, then integrate Ornith proposals

**Dependencies:** R3 and P12 disposition for any historical result used in selection. **Sources:** [benchmark targets](C:/Users/nikma/frontier-lowram-autoresearch/BENCHMARK-TARGETS.md), [successor preregistration section 6](C:/Users/nikma/frontier-lowram-autoresearch/SUCCESSOR-EVOLUTION-PREREG-V1.md), [Ornith integration plan](C:/Users/nikma/frontier-lowram-autoresearch/ORNITH-FABLE-SUCCESSOR-PLAN.md).

- [ ] Reuse and qualify existing benchmark adapters with positive/negative controls; do not rebuild the frontier benchmark system inside Chowder. Freeze datasets, versions, aggregation, statistical tests, seeds, judges and hardware protocol before candidate scoring.
- [ ] Apply the canonical <=1 percentage-point aggregate conversion-loss and >=1.5x fixed dense-9B throughput targets alongside parameter limits. Pin the aggregation and denominator. Also measure throughput relative to the current champion; a looser earlier screening threshold does not relax final nonregression.
- [ ] Measure fixed-work decode performance and end-to-end correct-work throughput so short/truncated/repetitive answers cannot manufacture an efficiency win. Reduced parameter count is not performance evidence on the local hardware.
- [ ] Apply every required capability/behavior/regression family and existing paired/stratified statistical gate. Preserve missing/untested status. The small engineering evaluation cannot stand in for these benchmarks.
- [ ] Enforce the exact four-way AND rule and the full Fable judge/ensemble requirements in the canonical target document, plus L0 approval. Record independent decisions for engineering, architecture qualification, champion promotion, and release target attainment.
- [ ] Only after manual Chowder runs reproduce, integrate Ornith proposal -> independent verification/admission -> versioned curriculum -> saved Chowder project -> independent evaluation -> human-governed promotion.
- [ ] Keep hidden prompts, answers, item-level failures and judge material inaccessible to Ornith and training. Limit repeated hidden-set access and expose only permitted aggregate feedback. Joint RL, new autonomous services, and external paid training remain separate projects/authorizations.

**Exit:** independently reproduced evidence satisfying the canonical targets and governance, or a clear rejection/partial result. No plan can guarantee the requested benchmark quality from the available parent and compute budget; each gate exists to find out before spending more.

## 7. Budget, stop rules, and verification commands

For every GPU run, freeze a reservation based on measured phases:

```text
wall budget = load + baseline eval + calibration + steps * conservative measured step time
            + checkpoint publication + restart/load + candidate eval + closeout allowance
GPU-hours   = sum(per-device attributable elapsed seconds) / 3600
stop        = first exhausted step, token, wall, GPU-hour, or required-memory limit
```

Record the conservative percentile/margin used. Teacher generation, judge/API calls, CPU-only preparation, and disk hashing are visible separate costs/times where applicable. Unknown measurements stay unknown. Preserve conservative failure charging and actual usage separately; never count the same leg twice.

Commands below describe implementation verification, not work executed while writing this plan. Run from the isolated source worktree with a pinned, verified Python runtime. PowerShell CPU-only checks use `$env:CUDA_VISIBLE_DEVICES = '-1'` in that test process; do not change a live worker environment. The ordinary suite may skip optional ML tests, so report both counts and which real-model gates actually ran.

```powershell
python -m pytest tests/test_transformers_backend.py tests/test_adapter_guard.py tests/test_schedule_audit.py tests/test_lr_scheduler_honoured.py -q
python -m pytest tests/test_local_model_manifest.py tests/test_protocol.py tests/test_worker_env.py tests/test_target_coverage.py -q
python -m pytest tests/test_memory_preflight.py tests/test_evaluator_vram_reporting.py tests/test_resource_accounting.py tests/test_transformers_resource_accounting.py -q
python -m pytest tests/test_router_healing.py tests/test_router_healing_run.py tests/test_router_healing_orchestrator.py -q
```

After the explicitly new tests have been implemented:

```powershell
python -m pytest tests/test_training_resume_end_to_end.py tests/test_router_healing_backend.py tests/test_router_healing_checkpoint.py tests/test_router_healing_end_to_end.py -q
python -m pytest -q
python -m ruff check src tests
```

The new end-to-end tests must exercise the real saved-project entry points using temporary local fixtures. The user-facing commands already exist:

```text
chowder project-validate <frozen-project.json>
chowder train <frozen-project.json>
```

`<frozen-project.json>` denotes the actual artifact produced after P11 input/budget freeze, not a runnable configuration supplied by this plan. No hashes or budget values are fabricated to make a launch command look complete.

Every delivery handoff records: revision and exact files changed; tests passed/failed/skipped; which hardware/model path was exercised; new artifact/checkpoint/registry references; measured cost; remaining acceptance gaps; and the next allowed action. Required CI includes the Linux matrix and real Transformers/PEFT CPU job, not only Windows and lint.

## 8. Immediate execution handoff and deferred work

**First implementation batch:** P0 preservation/reconciliation, then P1 actual text-decoder preset, P2 truthful adapter evidence, and P3 portable schedule test. Review and verify those small slices before changing the training lifecycle.

**Execution protocol (added 2026-09-12, after the first batch):** implementation slices may run in parallel across agents within the plan's coordination rules — one owner per shared file, GPU strictly serialized, every worker re-reading live source before editing (a bare `import chowder` on this host resolves to a *different* worktree, `C:/Users/nikma/Chowder`; all direct-python verification must pin `PYTHONPATH=src`, as pytest already does). Every slice is logged to GitHub: one commit per slice with its failing-test-first evidence in the message, pushed to the PR branch, plus a PR comment per slice recording tests passed/failed/skipped, hardware/model path exercised, and remaining acceptance gaps. PR merge remains forbidden without the user's explicit approval.

**First-batch completion record (2026-09-12):** P1–P3 implemented on `feat/hot-core-upcycling`; local verification 1,582 passed / 77 skipped / 0 failed (`CUDA_VISIBLE_DEVICES=-1`), ruff clean. Remaining gaps: cross-platform CI confirmation (Linux matrix + real-CPU smoke), P0 reconciliation inventory, and everything from P4 onward.

**Second batch:** P4 identity, P5 per-component trainability, P6 accountable measurements, and P7 fixed-horizon save/interruption/resume. This makes the existing dense training path meet the missing evidence contract.

**Third batch:** P8-P10 router backend/delta/evaluator/registry integration, followed by P11's bounded acceptance ladder. This is the central development milestone, not a side research task.

**After the milestone:** choose one budget-valid geometry and advance R1-R4 one gate at a time. Preserve all rejected hypotheses and measured artifacts. Do not promise a completion date for architecture quality before the first bounded recovery result and measured compute forecast exist.

Deferred now: another full-size top-k ladder, a wide core-share/expert/hyperparameter campaign, re-conversion without an identified compatibility need, blanket reevaluation of every historical result, a new scheduler/registry/dashboard, automatic champion promotion, joint Ornith RL, and release/export. None is needed to make Chowder train and verify its intended router workload correctly.

**Planning verification:** current source interfaces and the prior roadmap were reconciled; the implementation sequence covers all five user conditions. No source code, model, dataset, registry, PR, runtime, job or automation was changed while creating this document.
