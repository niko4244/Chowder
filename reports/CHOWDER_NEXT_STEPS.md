# Chowder prioritized engineering backlog

> Audit snapshot: 2026-10-22 UTC. Main 0bc566c5db052b90e3aa1223fef94445d1045dc5; PR #268 24f2f8865b215b477796364b1ec30e944d5ec087; recovery PR #269 e3f0fbd77bc679ea9a856461294cd26b40316562. GitHub code and published evidence were inspected; local Windows workspace inaccessible (Chat On Steroids Core root /strata invalid). Source claims below are **historical reports**, not rerun hardware measurements. No model weights or protected local evidence were accessed in this audit.

| Rank | Owner-sized task | Acceptance / prerequisite | Status |
|---|---|---|---|
| P0 | Qualify PR #269 recovery hardening | CI green; simulate hard kill before and after each journal/record boundary; verify repeated entry, total spend and cross-process ownership | IN PROGRESS; no auto-resume |
| P0 | Require signed/verified safe-resume plan before attempting a killed campaign | Exact artifact hashes, immutable prior attempt set, recomputed charges, single-writer lease, no re-spend | BLOCKED by incomplete recovery design |
| P0 | Qualify #268 current int8 retention-default build | Re-run 3-step gate with isolated real RTX 5060 Ti, <=14.5 GiB, no WDDM spill, complete log/revision | BLOCKED by hardware access and approval |
| P0 | Settle recipe feasibility | 1000 * measured step time <= 2.5h or explicitly pre-register a distinct budget/recipe and preserve previous failed target | UNMET on current host |
| P1 | Complete int8 placement witnesses | 20+ real optimizer steps with >=1% loss drop, finite nonzero LoRA B grads, router entropy +/-20%, adapter reload | NOT CERTIFIED |
| P1 | Finish REAP128/REAP160/dense/random prereg gate matrix | Exact paired quality, Q4 speed, residency, termination/control; frozen fallbacks only | UNVERIFIED |
| P1 | Produce actual Gen-0 freeze and model package | Full safetensors hashes, tokenizer/config, routing, quant, legal status, replayable loader, immutable judge record | BLOCKED |
| P1 | Resolve data licence N2 and contaminated-source screening | Audited per-row manifest/hashes, Fable-free control, explicit permission decision | OPEN |
| P2 | Review old PRs #226/#210 and clean model-mix docs | Rebase/resolve tested deltas, do not merge blindly; doc correction on audit branch | PARTIAL |
| P2 | Re-run CLI/TUI, cancellation, recovery and package CI matrix | Full passed test log with exact OS/python/source revision | NOT RUN IN THIS AUDIT |

**Compute policy:** Synthetic and non-destructive CPU checks can run without a GPU reservation. Expensive CUDA experiments and Gen-1 training need explicit authorization and a prospective budget that meets the frozen rules. Promotion always needs explicit human approval.


## Local verification addendum — 2026-10-22 (supersedes earlier GitHub-only findings)

The approved Windows workspace became accessible. The original main clone is a dirty rescue worktree and was not modified or reset. Isolated detached worktrees were created for testing. The physical REAP160 checkpoint at G:\llm-cold\bf16\Qwen3.6-REAP160 exists: 26 shards, 1,026 index-matched BF16 tensor headers, 46,039,444,192 payload bytes and 46,039,573,000 shard-file bytes; no header offset/index defects. Locally hashed config SHA256 matches 15413a791b9eff423755743bb05b997c93f6ae6f4ce02a1966c9bdfdc2cdc3b9. Full weight bodies remain unhashed.

Existing frozen experiment artifacts were reconciled. REAP128 fails GSM8K Gate 1 (paired bound -0.037807 versus -0.03), triggering the registered REAP160 fallback. REAP160 historical Gate 0, GSM8K, HumanEval and negative-control records PASS. A separate speed-resolution record reports 3.8523x Q4 speed versus dense with zero experts offloaded; all 12 raw comparison input files matched their recorded SHA256 hashes. This supports the fallback but does not constitute a new immutable Gen-0 freeze or a fresh CUDA run.

The local 24-step int8 witness SHA256 20e466d5362b8492fb9b55c312a57486283e641d46844e41c3ae4a85799617d7 records gradient PASS (19/19 LoRA B leaves), smoke PASS (mean loss 1.421 -> 0.551), router-balance PASS (worst entropy change 1.8%), WDDM spill NOT MEASURED. PR #268 revision 108f9322 fixes both the Markdown |g| table parsing defect and a witness --check that wrote despite claiming read-only; the real witness was imported through the validated generated-table mechanism. 7/8 placement gates now register measured values, with only WDDM unmeasured, and campaign admission still correctly refuses. Current-retention VRAM remeasurement and the 2.5 GPU-hour budget remain open.

Fresh Windows Python 3.11.9 focused CPU tests: recovery 90 PASS; int8/preflight 54 PASS; witness/gate/preflight 75 PASS; targeted Ruff clean. An initial 30-failure probe with artificially hidden CUDA was traced to a TorchAO import-time device query and succeeded when rerun with CHOWDER_NO_GPU=1 but CUDA device visibility restored, without GPU training.

The Gen-1 mix file checks passed against its current template-validated, 2048-token manifest: Fable 767 rows SHA256 384b6f10ae579af10ce36cbbe5278716b681d3223beab7343742b51c1857024d, Fable-free control 647 rows SHA256 681b2aa957d042028db8fa8710be6538b23f6563e088b9f752410bf5da174c9e. License decision N2 remains open. The card had approximately 12 GiB occupied and 4 GiB free, so no expensive CUDA tests or promotions were authorized or run.
