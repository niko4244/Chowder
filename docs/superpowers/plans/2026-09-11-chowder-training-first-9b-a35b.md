# Chowder Training-First and 9B/A3.5B Successor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. If those skill names are unavailable, use the installed Plan Architect Execute workflow; do not claim to have invoked an unavailable skill.

**Goal:** Make Chowder complete one bounded, reproducible router-training experiment through its normal interface, then use that proven path to investigate a Qwen3.8-9B-derived successor satisfying at most 10B total and 3.5B active parameters.

**Architecture:** Chowder owns execution, resource accounting, artifacts, independent evaluation, and durable adjudication. The frontier research repository supplies the successor targets, preregistrations, datasets, and benchmark adapters. Ornith proposes independently verified curriculum; it does not replace Chowder's trainer or see the hidden promotion set.

**Tech Stack:** Existing Python Chowder project/CLI, RunRegistry, ExperimentCycleRunner, resource/budget machinery, installed PyTorch/Transformers, existing safetensors and PEFT where actually supported. No new scheduler, experiment database, orchestration framework, or general plugin system.

**Status:** Assessed and planned on 2026-09-11. The numerical shape checks below were performed; the implementation tasks remain unexecuted. This document does not authorize a training launch, paid compute, model promotion, destructive cleanup, or merging another editor's work.

## 1. Decision and authority

The target is the **Qwen3.8-9B-abliterated successor**, not the 27B conversion. The canonical specification is [BENCHMARK-TARGETS.md](C:/Users/nikma/frontier-lowram-autoresearch/BENCHMARK-TARGETS.md:1). It requires <=10B total, <=3.5B active, >=1.5x dense-9B throughput, <=1 percentage-point aggregate conversion loss, and the full capability/behavior targets. The narrower A3.5B bound takes precedence over informal "3-4B" shorthand.

The following are separate outcomes, not interchangeable milestones:

| Outcome | Necessary evidence | What it does not imply |
|---|---|---|
| Training-path qualification | Actual gradient-bearing updates, delta, resumable state, independent evaluation, durable costs and verdict | A useful or promotable model |
| Architecture qualification | <=10B total, <=3.5B active under a frozen counting convention, <=1pp aggregate loss, >=1.5x measured throughput versus the fixed dense-9B control | Better unseen capability than the incumbent |
| Champion promotion | Capability improves AND Fable behavior does not regress AND throughput does not regress AND regression families pass, plus statistical gates and L0 approval | Final target attainment unless all absolute target checks also pass |
| Release target attainment | Architecture bounds plus the complete target table and applicable validated ensemble requirements | Automatic permission to distribute weights |

These outcome names are this plan's implementation distinction. They do not assert that Chowder already exposes these policies.

Promotion composes with [SUCCESSOR-EVOLUTION-PREREG-V1.md section 6](C:/Users/nikma/frontier-lowram-autoresearch/SUCCESSOR-EVOLUTION-PREREG-V1.md:158). Do not substitute a weighted composite for the four-condition conjunction. The older >=0.80 throughput-retention screen does not relax final throughput nonregression. Measure ratios against both the fixed dense-9B control and the current champion, with the denominator recorded.

The later FableBench section requires >=65% ensemble wins for its promotion grade, not only the headline 55% single-judge floor: >=2 pinned independent judges, >=75% agreement, >=50% decisive pairs, decisive 95% interval above chance, and no individual judge above 40% position bias. Preserve controls, order swaps, and disagreement-as-tie handling. See [the exact source](C:/Users/nikma/frontier-lowram-autoresearch/BENCHMARK-TARGETS.md:160).

## 2. What is established now

### 2.1 Live repository and local work

At the read-only GitHub check on 2026-09-11, main was `799f8a2543f899b357ec7e154cd3180d5929c00c`; PRs 154 and 155 were merged. PR156 remained open at `00bc24d22514bb1c8491597a42e0132350229817`, with all six reported checks successful. This is a snapshot, not merge authorization or a claim about subsequent state.

The local handoff worktree was clean at `71bbe1dc8acf3d9170a042dbea8146d8343d87c3` before this plan was added. The local router-healing worktree was at `cf9e0a0862e9b74c4c3f0f758d0a9d0966332826`; therefore local helper inspection and the live PR head are different revision identities. The local main worktree remained at `beffd23ed363baaff94011829a38ee5e06d6d933`, not the live main revision.

Current integration gaps in the inspected implementation:

- [backend_selection.py](C:/Users/nikma/Chowder-router-healing/src/chowder/backend_selection.py:90) dispatches Transformers/Unsloth PEFT, not a router trainer.
- [router_healing_run.py](C:/Users/nikma/Chowder-router-healing/src/chowder/router_healing_run.py:53) supplies recipe/artifact/goal helpers, not the complete executor/registry/evaluator workflow.
- [project.py](C:/Users/nikma/Chowder-router-healing/src/chowder/project.py:119) requires `evaluation.type='transformers-text'` and suite names equal all goal metric names. Measured parameter counts and throughput are not prompt suites.
- [project_runner.py](C:/Users/nikma/Chowder-router-healing/src/chowder/project_runner.py:270) selects the existing trainer and text evaluator; candidate evaluation assumes PEFT artifacts.
- [transformers_text_worker.py](C:/Users/nikma/Chowder-router-healing/src/chowder/evaluators/transformers_text_worker.py:132) loads a PEFT adapter rather than a router delta.
- [gate.py](C:/Users/nikma/Chowder-router-healing/src/chowder/gate.py:67) distinguishes incremental acceptance from `goal_met`. A parameter ceiling added as a target alone does not necessarily block acceptance.
- GPU calibration cannot run invisibly inside `profile()`: current preflight/profile failure handling precedes the accountable training failure boundary. Preserve the established reservation lifecycle.
- Failure budget charging is conservative: [engine.py](C:/Users/nikma/Chowder-router-healing/src/chowder/engine.py:146) charges at least the reservation. Record measured consumption separately; do not silently weaken that policy.

### 2.2 Preserve the 27B ladder, narrow its conclusions

Read in full before planning:

- [topk-ladder-finding.md](C:/Users/nikma/Chowder-Protected/runs/v3-20260909/topk-ladder-finding.md)
- [QWEN38_SPARSE_PROGRAM.md](C:/Users/nikma/Chowder-handoff-update/docs/QWEN38_SPARSE_PROGRAM.md)
- [HANDOFF.md](C:/Users/nikma/Chowder-handoff-update/docs/HANDOFF.md)
- Raw [topk-ladder-screen.json](C:/Users/nikma/Chowder-Protected/runs/v3-20260909/topk-ladder-screen.json)

| Active experts | PPL | Relative to k=16 |
|---|---:|---:|
| 16 | 5.4436 | 1.00x |
| 8 | 67.8181 | 12.46x |
| 4 | 51,030.3841 | 9,374.36x |
| 3 | 243,980.5894 | 44,819.63x |
| 2 | 1,208,031.2529 | 221,917.30x |

This is a strong negative result for the tested **untrained zero-router partition** on 10,902 scored tokens from 24 passages. It is not a trained-router experiment, an oracle-subset bound, or a comparison of all expert initializations. k=8 was slower than k=16 in that run, so a reduced expert count is not measured speedup.

PPL 5.44 alone does not prove equivalence to the dense parent. That needs a matched parent run and, for an exactness claim, appropriate logit/output comparison. Likewise, this experiment does not prove that no learned routing with frozen experts could ever improve. The evidence is sufficient to stop repeating this untrained full-size ladder as the main development activity.

The unchanged 27B text backbone cannot meet the A3.5B target by FFN routing alone. Retiring that route for that architecture must not retire the separate 9B successor target.

### 2.3 Exact 9B tensor census

Source: real safetensors headers in [the BF16 target directory](F:/llm-models/Qwen3.8-9B-abliterated-25-bf16), read with Chowder's existing parameter-accounting module. This was a header census, not a full weight-content hash or a model load.

| Category | Parameters |
|---|---:|
| Attention and GatedDeltaNet | 2,087,461,376 |
| Input embedding plus untied output head | 2,034,237,440 |
| Norms | 266,240 |
| Dense FFNs | 4,831,838,208 |
| Vision | 456,010,480 |
| Whole checkpoint | 9,409,813,744 |
| Text-only total | 8,953,803,264 |
| Text non-FFN subtotal | **4,121,965,056** |

There were 760 tensors. Config mentions an MTP layer, but no MTP tensor category was present in the observed inventory. Config labels are not evidence that tensors were loaded or executed.

Counting convention for this plan: include the full non-routed language-model parameter blocks, including input/output embedding tables; count unique tied weights once; add the routed experts actually selected. This conventional architectural A-count is **not** a claim that every embedding row is arithmetically multiplied on every token. Report operation/FLOP counts and embedding-lookup semantics separately. Do not change conventions to manufacture a passing label. Confirm and freeze this convention before an architecture verdict because the target document does not specify every counting detail.

Under that convention, the current 9B model is over the 3.5B budget before any FFN expert executes. Dividing its FFN into 16 equal experts gives at least 4.726B at top-2 or 5.028B at top-3, before newly added shared experts/router parameters. Quantization changes storage, not this count. Vision is stored but not automatically active on text-only decoding; adding all vision parameters to every text-token A-count is also incorrect.

### 2.4 A budget-compatible shape exists, but quality is unproven

A CPU `meta` construction against the installed `Qwen3_5MoeForCausalLM` succeeded with all parameters on `meta`; no weights were transferred and no forward, backward, GPU work, or benchmark was run.

Proposed **text research candidate**, not a chosen production architecture:

| Setting | Value |
|---|---:|
| Layers / residual width | 32 / 3,072 |
| Full-attention Q/KV heads / head width | 12 / 3 / 256 |
| GatedDeltaNet key/value heads / head dimensions | 12 / 24 / 128,128 |
| Vocabulary / tied embeddings | 248,320 / false |
| Routed experts / width per expert | 16 / 768 |
| Active routed experts | 2 |
| Shared-expert width | 512 |

| Measured meta-model category | Parameters |
|---|---:|
| Embeddings and output head | 1,525,678,080 |
| Attention/GatedDeltaNet/norm backbone | 1,174,547,584 |
| Routers | 1,572,864 |
| All routed experts | 3,623,878,656 |
| Shared experts plus their gates | 151,093,248 |
| Text total | **6,476,770,432** |
| Text active, top-2 | **3,305,876,608** |
| Text active, top-3 | **3,532,369,024 — exceeds the strict target** |

This is a feasibility witness for tensor geometry, not an achieved successor. It has about 194M active-parameter headroom at top-2; every later shared path, adapter, dense layer, or routed-width increase must fit inside the remaining budget. The 10B limit is a ceiling, not a reason to inflate this design immediately.

The witness is text-only. Original vision weights remain preserved. Retaining multimodal support requires an explicit plan for the changed 4,096-to-3,072 vision projection and its evaluation; neither a compatible multimodal checkpoint nor its exact count was validated here. Do not advertise retained vision or MTP capability based on this text-model construction.

## 3. Sequenced delivery

The critical path is **T0 -> T1/T2 -> T3 -> T4 -> T5 -> T6 -> T7 -> R1 -> R2 -> R3 -> R4**. T1 accounting and T2 recipe/trainability can proceed in parallel after source reconciliation. Do not run concurrent GPU-heavy tasks on the local shared cards. Each implementation slice gets its focused regression tests before review; stage only exact files belonging to that slice.

### T0 — Preserve and reconcile sources before implementation

**Dependencies:** None. **Deliverable:** a revision- and hash-backed reconciliation inventory, with no deletion of research evidence.

**Sources:** [Chowder worktrees](C:/Users/nikma/Chowder), [protected runs](C:/Users/nikma/Chowder-Protected/runs/v3-20260909), [frontier repository](C:/Users/nikma/frontier-lowram-autoresearch).

- [ ] Refresh live main/PR state and every worktree's HEAD and dirty inventory immediately before coding. Coordinate with the active Claude editor; do not race its PR156 merge.
- [ ] Preserve tracked diffs, untracked research drivers, configs, logs, and result files in a timestamped non-overwriting directory under `C:/Users/nikma/Chowder-Protected/runs`. Use exact source paths, sizes, mtimes, SHA256, and originating revision. Verify destination copies against the inventory. Do not recursively copy model weights, caches, or environments.
- [ ] For an active SQLite registry, use its supported consistent backup mechanism, not a raw copy of the main database file while WAL writes continue. Validate opening the backup before changing schema or provenance mappings.
- [ ] Keep the full 27B parent/conversion, raw ladder, evaluation freezes, and original 9B BF16 checkpoint immutable. Put corrections in an addendum and new versioned artifacts; never edit raw results to match the new narrative.
- [ ] Reconcile the root worktree's activation/evaluator edits, the activation-offload worktree, protocol-v4 dirty files, and duplicate untracked router helpers separately. Compare each with merged implementation; use focused tests before retaining or superseding. No broad stash, reset, staging, or removal.
- [ ] Treat the frontier ledger migration as an external historical source. The current `frontier.csv` and `frontier.csv.pre-a4b.bak` exist, and the CSV contains three records. Do not rerun the migration against research data for a test. Add migration/idempotence tests using temporary CSV copies only.
- [ ] Preserve historical `running` fields as historical assertions until logs and job identity support a terminal update. They are not proof that those jobs remain alive today. Import legacy runs with original identifiers/source hashes and unknown fields intact; never invent zero cost or successful outcomes.
- [ ] Make new Chowder registry runs authoritative for execution. Keep the CSV as a compatibility/reporting view or linked legacy source, not a competing scheduler or independent promotion authority.

**Verification:** compare hashes before/after preservation; reopen registry backup; distinguish local commit, live PR head, and dirty-file fingerprint. Reconciliation is complete only when every inventoried change is classified as retained, already upstream, separately deferred, or explicitly superseded with recoverable evidence. This plan itself does not claim that backup/reconciliation work has happened.

### T1 — Correct accounting and define measured evidence

**Dependencies:** T0. **Files:** modify [parameter_accounting.py](C:/Users/nikma/Chowder-router-healing/src/chowder/parameter_accounting.py), [sparse_accounting.py](C:/Users/nikma/Chowder-router-healing/src/chowder/sparse_accounting.py), [test_parameter_accounting.py](C:/Users/nikma/Chowder-router-healing/tests/test_parameter_accounting.py), [test_sparse_accounting.py](C:/Users/nikma/Chowder-router-healing/tests/test_sparse_accounting.py).

- [ ] Add failing cases for text-only versus multimodal execution, absent MTP, shared experts, tied weights, heterogeneous layer widths/top-k, and unknown modules. Unknown active-path semantics block qualification rather than being silently excluded.
- [ ] Keep stored tensor count/bytes distinct from logical parameter count for packed quantized tensors. Read representation metadata rather than treating packed byte dimensions as original model shape.
- [ ] Emit a versioned accounting definition, per-category subtotals, per-layer expert geometry, selected route width, and execution mode. Validate runtime top-k against recorded configuration. For variable routing, report the frozen target statistic and distribution; the first candidate uses fixed k=2.
- [ ] Report physical GPU allocated/reserved peaks, host RSS/commit, and actual tensor storage separately from the architectural A-count. Neither 4-bit naming nor a high CUDA virtual-allocation peak proves device fit.
- [ ] Add the real 9B census and the meta-model witness below as regression expectations. Do not allocate the full checkpoint in the ordinary CPU test suite.

Small deterministic arithmetic check:

```python
def test_9b_budget_arithmetic():
    fixed = 2_087_461_376 + 2_034_237_440 + 266_240
    assert fixed == 4_121_965_056
    assert fixed > 3_500_000_000
    routed = 3_623_878_656
    total = 6_476_770_432
    assert total - routed + routed * 2 // 16 == 3_305_876_608
    assert total - routed + routed * 3 // 16 > 3_500_000_000
```

**Verification:** run the two existing accounting test modules plus the new regression. Fail on deliberate vision/MTP misclassification and on a false <=3.5B claim.

### T2 — Freeze a valid router-training recipe and gradient contract

**Dependencies:** T0. **Files:** modify [router_healing.py](C:/Users/nikma/Chowder-router-healing/src/chowder/router_healing.py), [router_healing_run.py](C:/Users/nikma/Chowder-router-healing/src/chowder/router_healing_run.py), and their [gradient](C:/Users/nikma/Chowder-router-healing/tests/test_router_healing.py) and [run-spec tests](C:/Users/nikma/Chowder-router-healing/tests/test_router_healing_run.py).

- [ ] Extend the existing recipe with content-bound tokenizer/base/corpus/protocol identities, corpus order/tokenization, trainable names/shapes/dtypes, loss, top-k schedule, optimizer, scheduler, runtime/quantization/placement, and hard step/token/time limits. Use explicit versioning; reject unknown keys, non-hex hashes, NaN/Inf numeric fields, and unsupported modes before loading.
- [ ] Separate recipe identity from output/resume paths. Keep experiment recipe identity, attempt identity, and checkpoint parent identity distinct so resuming cannot rewrite immutable terminal records.
- [ ] First milestone trains **router tensors only**. Explicitly exclude the current all-zero frozen shared branch. A shared gate multiplying a frozen zero expert output cannot receive a useful gradient. Unfreezing every zero multiplicative FFN projection is not automatically a remedy.
- [ ] Test exact intended trainable-name equality; extra trainables are a failure. Over a declared multi-batch probe, every required trainable tensor must receive finite nonzero gradients and change after an optimizer step. Frozen weights must retain their original hashes. Log expert-row coverage separately from tensor-level reachability.
- [ ] Test zero/tied routing logits and the actual weight normalization. A normalized top-1 path can erase router-weight gradients; do not assume a differentiable training signal merely because `requires_grad=True`. Start the fixture with E=4, k=2, non-identical expert outputs, and verify rather than assume.
- [ ] Use the current zero-shared conversion as a negative control. A later nonzero shared-expert construction is a new architecture lineage with its own equivalence and recovery tests, not an unrecorded initialization fix.

**Verification:** actual small Qwen model autograd, not a mock alone; finite loss, nonzero router gradient, optimizer update, frozen checksum preservation, and a deliberately dead required component causing refusal.

### T3 — Add the smallest real executor, with accountable preflight

**Dependencies:** T1, T2. **Files:** create `C:/Users/nikma/Chowder-router-healing/src/chowder/backends/router_healing.py` and `C:/Users/nikma/Chowder-router-healing/src/chowder/backends/router_healing_worker.py`; create `C:/Users/nikma/Chowder-router-healing/tests/test_router_healing_backend.py`. Reuse contracts in [executors.py](C:/Users/nikma/Chowder-router-healing/src/chowder/executors.py:73), [resources.py](C:/Users/nikma/Chowder-router-healing/src/chowder/resources.py), and the existing [PEFT subprocess lifecycle](C:/Users/nikma/Chowder-router-healing/src/chowder/backends/transformers_peft.py:565).

- [ ] Implement `TrainingExecutor.profile/run/cancel`, bound cancellation and progress callbacks, `TrainingArtifact`, and `ExecutionFailure` using existing conventions. The worker returns artifacts/evidence; the parent owns registry writes.
- [ ] `profile()` reads a hash-bound prior measured profile or a conservative user-approved reservation; it does not load a model onto the GPU. Put measured load/step calibration inside the accountable `run()` stage, or an explicitly registered calibration experiment.
- [ ] Before optimization, inventory actual loaded tensor classes, shapes, logical dtypes, storage/quantization metadata, devices, and trainable set. Report cold load, first forward/backward/optimizer, steady-state step distribution, checkpoint time, GPU and host peaks.
- [ ] Detect raw fused 3D expert parameters left in BF16 by a loader nominally requested as 4-bit. An advertised quantization mode is not acceptance. Any alternative loader must prove expert computation, input-gradient propagation through frozen operations, and independent reload before use.
- [ ] Reserve warmup/calibration, checkpoint, and evaluation allowance. Stop if measured memory or projected remaining cost exceeds the approved limit. Emit structured evidence for OOM, commit-headroom refusal, NaN, timeout, and worker exit; retain a valid checkpoint when one exists.
- [ ] Keep the existing conservative failure charge intact and record both `measured consumption` and `budget charged`. Never report a paid failed calibration as a free rejection.

**Verification:** a real CPU optimizer loop, mocked process-failure boundary tests, and ledger reconciliation after cancellation. The target is one narrow backend, not generalized backend refactoring.

### T4 — Save both inference delta and exact-enough resumable state

**Dependencies:** T3. **Files:** extend [router_healing_run.py](C:/Users/nikma/Chowder-router-healing/src/chowder/router_healing_run.py), the T3 executor/worker, and `C:/Users/nikma/Chowder-router-healing/tests/test_router_healing_checkpoint.py` (new).

- [ ] Store tensor payloads with existing safetensors support and versioned JSON metadata. Allow only declared tensor names/shapes/dtypes; validate finite values and all identities before mutating a loaded model.
- [ ] Publish non-overwriting checkpoints atomically using temporary siblings and final publication only after all payloads are complete. Hash the resulting immutable files. Reject partial, corrupt, wrong-base, extra-tensor, and incompatible-protocol artifacts.
- [ ] Inference artifact: base manifest + router delta + effective routing configuration + loader/runtime/recipe identities. Resumable state additionally includes optimizer, scheduler, scaler if applicable, global step, Python/NumPy/Torch CPU/CUDA RNG state, data cursor/order, and accumulation state or an explicit optimizer-boundary-only checkpoint rule.
- [ ] Use optimizer-boundary checkpoints for the first implementation. Cancellation records any unfinished accumulation work without claiming it was applied. Treat local optimizer-state deserialization as a trusted, hash-verified artifact boundary.
- [ ] Test uninterrupted N steps versus M steps followed by a fresh-process resume to N, with deterministic CPU settings and declared tolerances. CUDA bitwise equivalence is not assumed; its tolerance and nondeterminism policy belong in the protocol.

**Verification:** identical CPU continuation under the pinned deterministic fixture; bad hashes rejected before application; frozen base unchanged; interrupted publication never mistaken for a complete checkpoint. Model-only weights are insufficient for optimizer continuation; see [PyTorch checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

### T5 — Independent evaluation through the normal saved-project interface

**Dependencies:** T4. **Files:** modify [backend_selection.py](C:/Users/nikma/Chowder-router-healing/src/chowder/backend_selection.py), [project.py](C:/Users/nikma/Chowder-router-healing/src/chowder/project.py), [project_runner.py](C:/Users/nikma/Chowder-router-healing/src/chowder/project_runner.py), [transformers_text.py](C:/Users/nikma/Chowder-router-healing/src/chowder/evaluators/transformers_text.py), [transformers_text_worker.py](C:/Users/nikma/Chowder-router-healing/src/chowder/evaluators/transformers_text_worker.py), and their existing tests.

- [ ] Add explicit `backend.type='router-healing'` dispatch without changing existing PEFT defaults. Validate the new backend's fields separately and disable PEFT-shaped automatic repair for this first recipe.
- [ ] Add explicit `artifact_kind='router-delta'` at the existing text-evaluator boundary; retain legacy PEFT behavior. In a fresh process, load the verified base and apply the verified delta, then assert actual effective routing configuration.
- [ ] Evaluate the baseline through the same rendering, tokenizer, quantization, placement, truncation, generation, scoring, and dataset-ID protocol. Propagate canonical-rendering fields instead of allowing the candidate spec to drop them. An identity delta must produce the same evaluation path and outputs as the no-delta control.
- [ ] Represent auxiliary measured metrics explicitly in project validation. Require exact declared coverage of prompt-suite metrics plus auxiliary metrics; reject duplicate names, unsupported sources, missing measures, and NaN. Do not create fictional prompt suites called `num_experts_per_tok` or `throughput`.
- [ ] Freeze protocol version, dataset content and split IDs, tokenizer/template, decode limits, scoring code, seeds, and hardware performance configuration before results. Keep the final hidden set separate from router calibration and any diagnostic data.
- [ ] Save per-item predictions/scores, evidence hashes, reload provenance, and metric denominators. Measure throughput with fixed inputs, fixed token-work controls, and actual end-to-end generation; do not obtain speedup by silently shortening answers or dropping work.

**Verification:** the existing `chowder project-validate <saved-project>` and `chowder train <saved-project>` path selects the new backend, reloads a real delta in a different process, and produces independently scored evidence. No new user-facing command is required. The saved pilot config is created only after input freezes and budget values are available, rather than publishing a runnable-looking config with fake hashes.

### T6 — Persist the exact verdict and all terminal outcomes

**Dependencies:** T5. **Files:** modify [cycle.py](C:/Users/nikma/Chowder-router-healing/src/chowder/cycle.py), [router_healing_run.py](C:/Users/nikma/Chowder-router-healing/src/chowder/router_healing_run.py), [run_events.py](C:/Users/nikma/Chowder-router-healing/src/chowder/run_events.py); reuse [registry.py](C:/Users/nikma/Chowder-router-healing/src/chowder/registry.py) and its evidence/artifact mechanisms. Extend [test_cycle.py](C:/Users/nikma/Chowder-router-healing/tests/test_cycle.py) and [test_registry.py](C:/Users/nikma/Chowder-router-healing/tests/test_registry.py).

- [ ] Persist a complete immutable adjudication artifact at the existing cycle boundary: decision kind, accepted/rejected/not-evaluated state, reasons, regressions, missing evidence, thresholds, baseline/candidate/protocol hashes, measured cost, budget charge, and approval state. Reuse registry evidence references; add schema only if the existing mechanism cannot represent this.
- [ ] Keep default weighted-gate semantics unchanged for existing projects. For this successor opt-in, hard architecture ceilings and the explicit four-condition champion policy are additional constraints. Fewer experts alone cannot promote the successor champion.
- [ ] Engineering qualification runs have champion promotion disabled. A scientifically rejected candidate may qualify the complete training path; a preflight refusal without an optimizer step does not.
- [ ] Preserve cancellation, failure, interrupted/resumable state, and evaluation failure separately from a completed quality rejection. Recover a killed worker using a fresh attempt identity with a checkpoint-parent link. Repeat ingestion must be idempotent and divergent immutable records must still fail loudly.
- [ ] Test wrong protocol, missing metric, size violation despite positive weighted gain, flat capability plus faster throughput, behavior regression plus higher capability, and missing human approval. None may replace the champion.

**Verification:** close and reopen the registry, reconstruct the attempt and its verdict solely from persisted artifacts/rows, and verify no stranded reservation. Do not infer a gate decision only from an in-memory ranked list or a final prose summary.

### T7 — Complete the bounded training milestone

**Dependencies:** T6. **Files:** add `C:/Users/nikma/Chowder-router-healing/tests/test_router_healing_end_to_end.py`; add the frozen pilot project under the protected run directory after authorization; update [HANDOFF.md](C:/Users/nikma/Chowder-handoff-update/docs/HANDOFF.md) with actual evidence status.

- [ ] CPU integration: actual tiny Qwen MoE, E=4/k=2, 2-4 layers, bounded corpus, optimizer updates, checkpoint/resume, fresh-process evaluation, and registry reopen. Keep full-size weight loading out of ordinary CI.
- [ ] Fault integration: cancellation during training, worker death after a checkpoint, evaluator failure, disk-write failure, tampered corpus/base/delta/protocol, and repeated resume. Every case retains provenance and reconciled budget without promotion.
- [ ] Small-shape CUDA: same production worker/model class and proposed loader representation, three optimizer steps minimum, save/resume/reload, telemetry, and cancellation test. First inspect GPU ownership; never terminate another editor's job to obtain capacity.
- [ ] Prepare the actual-workload engineering fixture with the existing converter using the original 9B width/layer geometry, a validated 16-way FFN partition, and the explicitly inactive zero shared branch. Test original-width 9B config support and full-route reconstruction on the tiny fixture first; extend the converter narrowly if it contains 27B-specific assumptions. Save the derived model and conversion manifest to a new protected directory, never over the original 9B checkpoint. This fixture is permitted to exceed A3.5B because it tests the training path with champion promotion disabled. It does not depend on R2's backbone surgery.
- [ ] Actual 9B-derived pilot: after that conversion and resource approval, run one saved project with a proposed limit of 16 optimizer steps, batch size 1, sequence length 128, at most 2,048 training tokens, and an explicit approved wall/accelerator budget. Calibration optimizer updates and their tokens consume these same limits; all extra read-only calibration work still consumes the wall/accelerator reservation. These are initial engineering limits, not a healing-quality prescription or compute authorization.
- [ ] Include a separately frozen small evaluation and its baseline in the reservation. Measure calibration steps inside the accountable run. If there is no budget for independent evaluation, do not start a training-only run and call the milestone complete.
- [ ] If the full 9B-derived representation fails memory/step-cost preflight, retain the measured failure and stop expansion. The small-shape success remains useful, but the full-workload milestone is still open. Do not substitute the 27B parent or a mock and declare it done.

**Exit criteria:** all five user acceptance conditions in section 5 hold for one actual 9B-derived run. Existing PEFT, cancellation, resume, project-schema, evaluator, resource-accounting, and registry tests remain passing. Then, and only then, begin the broader architecture campaign.

## 4. Research roadmap after the training path works

### R1 — Measure the right compression question

**Dependencies:** T7. **Sources:** [the current diagnostic](C:/Users/nikma/frontier-lowram-autoresearch/training/a4b_sparsity_diag.py), [dense_to_moe.py](C:/Users/nikma/Chowder-router-healing/src/chowder/dense_to_moe.py), [conversion_exactness.py](C:/Users/nikma/Chowder-router-healing/src/chowder/conversion_exactness.py). Keep any new workload under Chowder's execution/accounting path.

- [ ] Relabel the current sign-intersection diagnostic as a **dReLU counterfactual**, with no universal partition/upcycling kill rule. Its `>0.5` active-fraction conclusion does not prove that only duplicated experts can work. Sign counts are neither current SiLU output contribution mass nor the best attainable top-k reconstruction error.
- [ ] On a frozen training-side calibration set, capture actual SiLU FFN channel contributions and grouped output reconstruction error. Compare existing partitions, activation-informed groups, selected scaling rules, and a bounded best-subset diagnostic where feasible. Distinguish per-layer approximation from end-to-end quality.
- [ ] Measure matched dense-parent versus full-route conversion under the same loader and protocol before using the word equivalent. Compare zero-router and trained-router-only controls with declared token budgets. Trainable-expert recovery comparisons wait until R3's validated trainability extension; R1 does not require that later capability.
- [ ] Log routing entropy/utilization, dead experts, retained output mass, loss, and runtime. A good oracle result motivates routing work; a poor tested oracle is a bound on that grouping/search protocol, not a proof about all MoEs.
- [ ] Keep all research arms budget-capped. Run one bounded control plus one challenger first; do not multiply expert counts, k values, learning rates, seeds, and datasets into a full Cartesian search.

**Exit criteria:** a written choice of one geometry/init/recovery arm supported by measured reconstruction, trainability, runtime, and held-out pilot evidence, or an explicit rejection with the failed hypothesis retained.

### R2 — Reduce the 9B non-routed floor without changing lineage silently

**Dependencies:** R1. **Implementation boundary:** extend [dense_to_moe.py](C:/Users/nikma/Chowder-router-healing/src/chowder/dense_to_moe.py) only for MoE conversion; use a separate narrowly scoped Qwen shape-transform module if whole-backbone pruning would overload that converter. Require a dedicated implementation review before generating full-size weights.

- [ ] Start with the 3,072-width meta witness, retaining 32 layers, vocabulary, and Qwen lineage. Treat it as one candidate, not the final architecture. Compare alternatives only if measured recovery or throughput rejects it.
- [ ] Produce explicit channel/head selection maps from training-side calibration. Transform embeddings/output, residual-stream axes, Q/K/V/gated output projections, norms, FFNs, GatedDeltaNet packed projections/convolutions/state-related axes consistently. Validate legal head ratios and parameter counts from emitted tensors.
- [ ] Test an identity transform first: full shape and all channels must reproduce original tensors and outputs within the frozen numeric tolerance. Test each reduced-shape mapping on a tiny actual hybrid-attention model before applying it to the 9B checkpoint.
- [ ] Preserve original checkpoint/tokenizer and a complete source-to-destination mapping manifest. No blind config-only width change, blind embedding tying, vocabulary truncation, or assertion of retained multimodal capability.
- [ ] Write the architectural-recovery preregistration before any continued training. Distinguish recovery of a surgically altered model from the previously closed raw OpenStax QA-transfer hypothesis. A prior killed arm is not reopened without an explicit reason and new evidence.
- [ ] Before running architectural recovery, select the already-working Chowder PEFT executor if its declared trainable layers support the transformed dense model. Otherwise extend the proven worker in a separate narrow slice for the exact backbone-recovery trainable allowlist. In either case, first pass actual gradient/update, frozen-hash, checkpoint/resume, independent-reload, and accounting tests on the reduced tiny architecture. Do not wait until R3 to implement trainability needed by an R2 experiment.
- [ ] Use a small recovery pilot first with protected length/throughput and retention checks. If the required quality cannot be recovered within the approved budget, reject or revise the architecture; do not quietly relax the <=3.5B or <=1pp targets.

**Research basis and limits:** [Sheared LLaMA](https://arxiv.org/abs/2310.06694) supports structured pruning followed by continued training as a real route to smaller native descendants. Its results are not proof that this Qwen hybrid architecture, 25% width reduction, or the available local budget preserves the requested quality.

### R3 — Learn useful experts, not only new router weights

**Dependencies:** R2. **Implementation:** reuse the validated R2 recovery path where applicable and extend the proven worker's declared recipe specifically for expert adaptation in one reviewed slice; preserve router-only mode and its regression tests.

- [ ] Compare the selected fine-grained initialization with one capacity-budget-compatible control. Whole-FFN duplication across many experts generally changes the total-parameter budget; reject any arm exceeding 10B before training.
- [ ] Choose a shared-expert initialization with nonzero gradient paths and explicit channel allocation or reconstruction residual. Verify that a claimed exact full-route reconstruction still holds after including it. Do not add a shared path on top of an already complete dense sum and call that exact.
- [ ] Permit expert adaptation and, if required, constrained backbone recovery only through an exact trainable allowlist. Low-rank or quantized expert adaptation is acceptable only after the actual fused-parameter implementation proves support; standard linear-layer QLoRA assumptions do not suffice.
- [ ] Use staged recovery (full-route reconstruction control, reduced-k training, fixed k=2 candidate) only where each stage has a declared hypothesis and budget. Freeze router temperature, weight normalization, auxiliary losses, and schedules in the recipe.
- [ ] Evaluate untrained, router-only, and expert-adapted candidates under equalized budget and protocol. Keep one finalist before spending on seed replication and the full suite. A small loss reduction is a screen, not the release quality gate.

**Research basis and limits:** [Upcycling Large Language Models into Mixture of Experts](https://arxiv.org/abs/2410.07524) studies fine-grained initialization, scaling, routing order, and continued training. It motivates controlled tests rather than an assumption that random partitioning or a few router steps will succeed. [TurboSparse](https://arxiv.org/abs/2406.05955) is a separate activation-function/training route; a published training budget is not a universal mathematical minimum, and adopting dReLU would require a new architecture/recovery protocol. Neither route is expanded before Chowder can execute and adjudicate it.

### R4 — Qualify, promote, and only then connect autonomous proposals

**Dependencies:** R3. **Sources:** [benchmark targets](C:/Users/nikma/frontier-lowram-autoresearch/BENCHMARK-TARGETS.md), [preregistration](C:/Users/nikma/frontier-lowram-autoresearch/SUCCESSOR-EVOLUTION-PREREG-V1.md), [Ornith plan](C:/Users/nikma/frontier-lowram-autoresearch/ORNITH-FABLE-SUCCESSOR-PLAN.md).

- [ ] Validate harness positive/negative controls before recording candidate numbers. Reuse existing benchmark adapters; do not rewrite all 18 harnesses in Chowder. Preserve item-level outputs, pins, and scoring lineage through a narrow adapter.
- [ ] Run cheap held-out loss, n=384 development with blocking throughput, n=480 salt-disjoint promotion, and paired GSM8K/ARC-C/BFCL/reasoning-calibration retention at alpha=.05. Preserve the source's paired McNemar and subject-stratified bootstrap requirements.
- [ ] Apply architecture qualification and complete target attainment separately. The <=1pp aggregate conversion check must use a frozen aggregation/weighting rule, not PPL percentage change or weights selected after seeing results.
- [ ] Record both correct-work throughput and fixed-token kernel/decode throughput so shorter or truncated answers cannot create an apparent efficiency win. Final comparison includes identical hardware, batch/context conditions, precision, warmup, and output policy.
- [ ] Apply the exact four-condition champion rule, statistical gates, hidden-set discipline, and L0 approval. Preserve missing/untested outcomes explicitly; a machine-learning gate pass is not a deployment authorization.
- [ ] Only after manual runs reproduce, connect Ornith proposal -> independent verification/admission -> versioned curriculum -> saved Chowder project -> independent evaluation -> human-governed promotion. This is integration of existing roles, not a new autonomous training service.
- [ ] Keep hidden prompts, answers, item-level failures, and judge material inaccessible to the proposer and training corpus. Expose bounded aggregate feedback; limit repeated hidden-set queries and maintain an access/audit boundary. Future fresh sealed holdouts require a versioned policy, not silent replacement.
- [ ] Ornith is also a real local proposer/scaffold-teacher model; possessing its GGUF does not implement joint GRPO. That later RL project has separate infrastructure/compute approval. Preserve the source plan's Fable-as-evaluation-reference boundary; do not ingest Fable transcripts as training data without the required explicit decision.

## 5. User acceptance matrix

| Requested condition | Implementation | Required durable evidence |
|---|---|---|
| Preflight actual loaded formats, memory, measured step cost | T1, T3, T7 | Tensor census; loader identity; measured host/device peaks; cold/warm step phases; approved limits; billed calibration |
| Train intended tensors; each can receive gradients | T2, T3, T7 | Exact trainable set; multi-batch finite nonzero gradient coverage; optimizer deltas; frozen hashes; dead shared-path negative control |
| Save delta and resumable training state | T4 | Atomic delta and checkpoint manifests; optimizer/scheduler/RNG/data cursor; fresh-process continuation comparison |
| Independent base+delta reload under fixed versioned evaluation | T5 | New process; checked base/delta hashes; protocol identity; matched baseline; item-level predictions and scores |
| Actual cost, cancellation/failure, gate in existing registry | T6, T7 | Reopened registry lineage; measured versus charged budget; complete adjudication artifact; fault outcomes; no stranded reservation |

Not done if only helper tests pass, only a tiny fixture trains, only the model loads, the evaluator reuses the training object, a checkpoint lacks optimizer state, a rejected candidate is promoted for lower k, or the final verdict exists only in chat.

## 6. Budget and scheduling policy

Engineering order is firm; training duration and capability attainment are not promised. The first seven tasks are bounded implementation slices, followed by a real-workload decision point. Research stages are sequential gates, not a calendar promise or automatic long run.

For every proposed run, derive the reservation from measured hardware-specific data:

```text
planned wall seconds = cold load + calibration + optimizer steps * conservative measured step time
                     + checkpoint/resume overhead + baseline evaluation + candidate evaluation
accelerator hours    = sum(per-accelerator active elapsed seconds) / 3600
stop condition       = first exhausted limit: optimizer steps, training tokens,
                       wall seconds, or accelerator hours (each in its own unit)
```

Use a conservative measured percentile or explicit safety margin, record its choice, and distinguish wall time from multi-device accelerator time. Include teacher generation and judge/API costs separately when present. They are not free because they occur outside the optimizer loop.

Local GPU work is serialized with other owners. A parent checkpoint residing mostly in shared/host memory is not evidence of practical training capacity. If the full 9B-derived pilot cannot fit or has unusable measured step cost, bring the measured options back for a hardware/budget decision; do not launch external paid compute or silently reduce the workload and report full-model success.

The larger 27B tournament expansion, full-grid MoE search, dReLU recovery campaign, and joint Ornith RL remain deferred until the relevant preceding gates pass. Successful ledger migration is useful provenance work; it is not evidence that Chowder trains this workload yet.

## 7. Reproduction and provenance appendix

These are read-only checks already performed or repeatable without loading the target weights. They do not replace a full model manifest before training.

Observed source SHA256 values:

| Source | SHA256 |
|---|---|
| Protected ladder finding | `971acfc93f7f1f8595c2dec770891c04478dd181c03b43f08837d8bada716c2c` |
| Protected raw ladder JSON | `3643278cd3d70f1a3bfb3e4a1d00b148273f46d07913de40e4b6bd47b6fadbda` |
| Benchmark target specification | `79b1a4256fe49e1d134bdbc0e7a98a4d3b2d4eb8285ec732b82b75ad34dfc61b` |
| 9B config.json | `583a72ebde6d3fab6b7e47a64e46e5a3cc2913c3eac824b20a618c36d5ff588f` |

Header census using the inspected helper revision:

```powershell
& 'C:\Users\nikma\Chowder\.venv-repro\Scripts\python.exe' -c "import sys,json; sys.path.insert(0,r'C:\Users\nikma\Chowder-router-healing\src'); from chowder.parameter_accounting import account_parameters; print(json.dumps(account_parameters(r'F:\llm-models\Qwen3.8-9B-abliterated-25-bf16').to_dict(),indent=2))"
```

Repeatable shape-only witness in a Python session with the installed training runtime (no checkpoint reads or GPU allocation):

```python
import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

config = Qwen3_5MoeTextConfig(
    vocab_size=248320, hidden_size=3072, num_hidden_layers=32,
    num_attention_heads=12, num_key_value_heads=3, head_dim=256,
    linear_num_key_heads=12, linear_num_value_heads=24,
    linear_key_head_dim=128, linear_value_head_dim=128,
    num_experts=16, num_experts_per_tok=2,
    moe_intermediate_size=768, shared_expert_intermediate_size=512,
    tie_word_embeddings=False,
)
with torch.device('meta'):
    model = Qwen3_5MoeForCausalLM(config)
assert all(p.device.type == 'meta' for p in model.parameters())
total = sum(p.numel() for p in model.parameters())
routed = sum(p.numel() for n, p in model.named_parameters() if '.experts.' in n)
assert total == 6_476_770_432
assert routed == 3_623_878_656
assert total - routed + routed * 2 // 16 == 3_305_876_608
```

Validation boundaries: no full 9B tensor-value hash, weight transfer, GPU training, quality evaluation, preservation backup, migration rerun, or promotion was performed while writing this plan. A previous wrapper-config meta attempt failed before model construction; the successful witness above uses the required text configuration and is the only successful architecture-shape result claimed.

## 8. First implementation handoff

Start with T0, then parallelize T1 accounting and T2 recipe/gradient tests. Review those before adding the executor. Use the existing router helpers if PR156 lands, but first re-resolve the actual merged revision; do not treat this plan's snapshot as more authoritative than live code.

The next user-visible milestone remains: **Chowder completes one bounded router-training experiment through its normal interface.** A passing A3.5B parameter shape is useful evidence for the subsequent research path, not permission to skip that milestone.
