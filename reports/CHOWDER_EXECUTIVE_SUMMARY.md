# Chowder engineering and custom MoE — executive verdict

> Final audit 2026-10-22. Main 0bc566c5; MoE PR268 184291f; recovery PR269 cf54480 with local 82-test verification; REAP PR271 73f1593. GitHub and read-only Windows workspaces inspected, 39 model files independently SHA-256 verified twice. No GPU training was started in this audit. Frozen gates and required approvals remain binding.

## Answer first

**Chowder is an implemented research/control-plane system, but the custom REAP-MoE model is not qualified as a release-ready, reproducibly packaged Gen-0 or proven autonomous improvement.** The previous audit was stale: PR #268 now records real 20-step CUDA training and a sanctioned int8 VRAM-gate pass for an older setting, the local 24-step witness now supplies accepted gradient, smoke and router results. WDDM no-spill remains unmeasured and recipe throughput is above budget.

## Work completed in this pass

1. Reconciled main HEAD, open PRs and current #268 test/benchmark documentation; did not equate PR content with merged software.
2. Found and patched journal stale-terminal/torn-verdict fail-open behavior on PR #269; fsynced tmp verdict and added regression tests. CI started for the new head; final result not yet available at audit snapshot.
3. Added pre-load witness input validation and CPU-only negative tests to int8 PR #268 (commit 24f2f886, CI started).
4. Created an audit branch with all twelve user-specified deliverables, including machine-readable status/finding/model manifests.
5. Corrected stale MoE Gen-1 dataset documentation to match the merged 2,048-token drop-not-truncate builder and optional Self-OSS source.

## Principal blockers

- The Windows workspace and model files were verified after re-targeting an approved root; the card is in use so new expensive GPU execution is not authorized.
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

## Final local reconciliation (2026-10-22)

Full cryptographic inventory: 39 physical REAP160 files, including 26 BF16 shards, hashed twice independently with matching SHA-256 for each file. Package inventory digest: 06aaa74bab80d6a025e25428eacfe8006d0d43fad2406370e563ff1a0c2f1c36. Per-file hashes are in artifacts/custom_moe_model_manifest.json; a sharded checkpoint has no claimed monolithic SHA-256.

Frozen REAP observer D0 locally PASS (40 layers, perplexity 10.646096, threshold less than 20). REAP128 v1 Gate1 FAIL; REAP160 v1 quality gates PASS and historical separate Q4 fit/speed proof. Amendment 4 requires REAP128_v2 scoring before Gen0 can be selected. PR271 adds fail-closed D0/cardinality/duplicate and ordered-arm disclosure; 15 tests PASS. No v2 model/score was located.

Local int8 witness imported 24-step gradient, loss-smoke and router PASS; seven of eight gate rows measured; WDDM operator no-spill remains unmeasured. Historical best 27.48s/step implies 7.63h per 1000 steps versus frozen 2.5h ceiling. Current retention-default VRAM remeasurement remains outstanding.

Additional P1 findings: automatic safe resume remains unavailable, promotion ledger commit can precede terminal verdict, failed evaluation charges need stronger reconciliation, and model placement gate identity currently uses config/size rather than weight contents. Recovery work includes budget fallback and cross-process writer locking with focused tests but awaits complete CI/merge. No expensive GPU run, Gen1 training or promotion occurred.

## Verified Gen-1 mix shortfall (2026-10-22)

The actual file hashes and ordered-subsequence relationship were independently verified: train-control has 647 rows, train-fable 767 rows, and all 647 control rows appear in the same order in the Fable arm, with exactly 120 extra Fable rows. Manifest reports 2,048-token screening and template validation. However the Nemotron agentic source retained **0 of 120** planned rows, OpenThoughts-3 retained **9 of 150**, and Ornith retained **38 of 39**. The current Fable-free control contains no Nemotron tool-use training rows. This is an explicit registered shortfall, not silently repaired by adding unapproved examples. A Fable-vs-no-Fable comparison would also differ in training exposure; do not market it as a dose-matched comparison. Ornith license metadata is undeclared and Fable N2 remains unresolved, so release/data-rights qualification is BLOCKED even though contamination screening dropped 259 rows and all output hashes match.

## Verified deployable GGUF artifact (2026-10-22)

Existing REAP160 Q4_K_M GGUF: C:\llm-work\reap\gguf\reap160-q4_k_m.gguf, 13,821,482,880 bytes, independently calculated SHA-256 8c840461752e1343e81bc12ab9b95d422a4b45c0b0048f00938fe882f299f8f5, identical to the previously saved serve.json digest. Its GGUF header is version 3 with 733 tensors and 41 metadata entries. Historical llama-server load-only evidence reports --fit on -np 4 -c 16384 --jinja, all 41/41 layers GPU-resident, zero routed experts offloaded, and 14,275 MiB VRAM. The archived speed resolution reports 3.8523x the dense Q4 comparison. The file and hash are verified now; loading/generation were not rerun in this audit because the RTX 5060 Ti is occupied. This identifies a real inference artifact, but final custom Gen0 selection remains blocked by amendment-4 REAP128_v2 evaluation and training/release gates.
