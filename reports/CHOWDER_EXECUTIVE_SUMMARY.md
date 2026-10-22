# Chowder engineering and custom MoE — executive verdict

> Audit snapshot: 2026-10-22 UTC. Main 0bc566c5db052b90e3aa1223fef94445d1045dc5; PR #268 24f2f8865b215b477796364b1ec30e944d5ec087; recovery PR #269 e3f0fbd77bc679ea9a856461294cd26b40316562. GitHub code and published evidence were inspected; local Windows workspace inaccessible (Chat On Steroids Core root /strata invalid). Source claims below are **historical reports**, not rerun hardware measurements. No model weights or protected local evidence were accessed in this audit.

## Answer first

**Chowder is an implemented research/control-plane system, but the custom REAP-MoE model is not qualified as a release-ready, reproducibly packaged Gen-0 or proven autonomous improvement.** The previous audit was stale: PR #268 now records real 20-step CUDA training and a sanctioned int8 VRAM-gate pass for an older setting, yet the frozen overall qualification still lacks gradient/smoke/router/WDDM evidence and recipe throughput is above budget.

## Work completed in this pass

1. Reconciled main HEAD, open PRs and current #268 test/benchmark documentation; did not equate PR content with merged software.
2. Found and patched journal stale-terminal/torn-verdict fail-open behavior on PR #269; fsynced tmp verdict and added regression tests. CI started for the new head; final result not yet available at audit snapshot.
3. Added pre-load witness input validation and CPU-only negative tests to int8 PR #268 (commit 24f2f886, CI started).
4. Created an audit branch with all twelve user-specified deliverables, including machine-readable status/finding/model manifests.
5. Corrected stale MoE Gen-1 dataset documentation to match the merged 2,048-token drop-not-truncate builder and optional Self-OSS source.

## Principal blockers

- Local Windows workspace root changed; no first-hand checkpoint/storage verification or GPU execution possible.
- PR #268 remains unmerged and not completely gated for the exact shipped retention-default code/config.
- Historical best 27.48 s/step projects to 7.63 GPU-h for 1,000 steps against 2.5h ceiling.
- REAP quality selection and output package lack complete accessible cryptographic and benchmark evidence.
- Fable/Anthropic usage decision N2 remains open; human promotion N5 remains compulsory.

## Workstream verdicts

| Scope | Verdict |
|---|---|
| Repository/static audit | COMPLETED for GitHub; local reconciliation BLOCKED |
| Recovery P1 mitigation | COMPLETED as committed PR change; validation UNVERIFIED until CI |
| Full safe resume | BLOCKED |
| Int8 engineering | PARTIAL implementation, historical CUDA measurements |
| Current build's frozen placement acceptance | BLOCKED |
| REAP128/REAP160 Gen-0 model selection | UNVERIFIED |
| Gen-1 sustained campaign and improvement | BLOCKED (budget/gates/authorization) |
| Production-ready model package | BLOCKED |

No model promotion or costly training run was undertaken.


## Local verification addendum — 2026-10-22 (supersedes earlier GitHub-only findings)

The approved Windows workspace became accessible. The original main clone is a dirty rescue worktree and was not modified or reset. Isolated detached worktrees were created for testing. The physical REAP160 checkpoint at G:\llm-cold\bf16\Qwen3.6-REAP160 exists: 26 shards, 1,026 index-matched BF16 tensor headers, 46,039,444,192 payload bytes and 46,039,573,000 shard-file bytes; no header offset/index defects. Locally hashed config SHA256 matches 15413a791b9eff423755743bb05b997c93f6ae6f4ce02a1966c9bdfdc2cdc3b9. Full weight bodies remain unhashed.

Existing frozen experiment artifacts were reconciled. REAP128 fails GSM8K Gate 1 (paired bound -0.037807 versus -0.03), triggering the registered REAP160 fallback. REAP160 historical Gate 0, GSM8K, HumanEval and negative-control records PASS. A separate speed-resolution record reports 3.8523x Q4 speed versus dense with zero experts offloaded; all 12 raw comparison input files matched their recorded SHA256 hashes. This supports the fallback but does not constitute a new immutable Gen-0 freeze or a fresh CUDA run.

The local 24-step int8 witness SHA256 20e466d5362b8492fb9b55c312a57486283e641d46844e41c3ae4a85799617d7 records gradient PASS (19/19 LoRA B leaves), smoke PASS (mean loss 1.421 -> 0.551), router-balance PASS (worst entropy change 1.8%), WDDM spill NOT MEASURED. PR #268 revision 108f9322 fixes both the Markdown |g| table parsing defect and a witness --check that wrote despite claiming read-only; the real witness was imported through the validated generated-table mechanism. 7/8 placement gates now register measured values, with only WDDM unmeasured, and campaign admission still correctly refuses. Current-retention VRAM remeasurement and the 2.5 GPU-hour budget remain open.

Fresh Windows Python 3.11.9 focused CPU tests: recovery 90 PASS; int8/preflight 54 PASS; witness/gate/preflight 75 PASS; targeted Ruff clean. An initial 30-failure probe with artificially hidden CUDA was traced to a TorchAO import-time device query and succeeded when rerun with CHOWDER_NO_GPU=1 but CUDA device visibility restored, without GPU training.

The Gen-1 mix file checks passed against its current template-validated, 2048-token manifest: Fable 767 rows SHA256 384b6f10ae579af10ce36cbbe5278716b681d3223beab7343742b51c1857024d, Fable-free control 647 rows SHA256 681b2aa957d042028db8fa8710be6538b23f6563e088b9f752410bf5da174c9e. License decision N2 remains open. The card had approximately 12 GiB occupied and 4 GiB free, so no expensive CUDA tests or promotions were authorized or run.
