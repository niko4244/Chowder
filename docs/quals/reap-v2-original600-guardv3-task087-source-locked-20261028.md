# REAP128_v2 original AutomationBench600 guard.v3 — task 087 continuation (2026-10-28)

**Disposition: authentic, partial, research-only. Gen-0 remains `BLOCKED`, `chosen: null`, `release_approved: false`.** This note records one new original AutomationBench v1.0.6 evaluation, not a completed 600-item benchmark or a frozen four-slot model-selection gate. No original certificate, benchmark source, GGUF, model weight, or frozen selector was edited.

## Source-locked original execution

The live worktree was `feature/reap128v2-selection-20261025` at `a137d896f992a5dc4a5fc912e75ff02a63666ce8`; local HEAD, remote feature HEAD, and draft PR #282 head agreed before execution. Thirteen preexisting untracked contributor paths were left untouched. The original guard.v3 checkpoint, **not** the unrelated guard.v4 pilot, was read and resumed through the committed `chowder_batch.reap_original600_guardv3_git_pinned_resume_20261027` wrapper. The wrapper loads the original runner, guard, and scorer from immutable Git commit `84ff9703d6e2aba68eda8585ee461e3d84abec0c`, checks their source hashes and original owner, and does not permit conservative resource-floor bypass.

| Original evidence | Authenticated identity / observation |
| --- | --- |
| Checkpoint root | `F:\reap-v2-q4-original-automationbench600-guardv3-20261027` |
| Original owner SHA-256 | `b6aa1f37e81943344217ede4429066c468bce2176a74eb8b15fcc0fe4a4e0529` |
| Original GGUF SHA-256 | `2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7` |
| Original benchmark source SHA-256 | `1c8feeec1f680b05b392a5d00fc634edb143c3e7bdd152c476af50cebcec57b5` |
| Original official benchmark | AutomationBench v1.0.6, source Git `4a8e1061254004d9dac807054eed33fad7d1ff14` |
| Original guard policy | `guard.v3`; source SHA-256 `b9eb375519a0eb7018f7cbc007e697478b77df7d734facfc20d239005bd2ee3a` |
| Original model serving | One-slot research `-np 1 -c 73728`; **not** the frozen `-np 4 -c 16384` profile |
| Before this continuation | **87/600** independently scored, task 087 unscored/unknown |
| Conservative preflight | **READY**: 32.29 GiB available physical RAM, 43.52 GiB commit, 15,221 MiB free VRAM, 28.79 GiB free F: disk, ports 8892/8894 free; no competing native model server reported |
| Execution | `python -m chowder_batch.reap_original600_guardv3_git_pinned_resume_20261027 --execute --max-new 1` with bytecode writes disabled; no bypass |
| Previous unfinished task 087 | Preserved as `task-087.unfinished-881ca4bf7d04bd0f`; **not** retroactively scored |
| New original task 087 | One genuine original evaluator execution; strict pass **0**, completed **1** |
| Task 087 original raw evaluator SHA-256 | `786737ac1713895587dde6d4c5ab3d4947aac78035098d1ac647caf19f8a343b` |
| Task 087 immutable scored receipt SHA-256 | `bb63251615308b0b2ac2112910d0206c276da7a15ce608b1a06bb488e989de3e` |
| Task 087 receipt write time | `2026-10-29T00:08:09.2833984Z` |
| After this continuation | **88/600** independently scored, **3 strict passes**, **512 still unscored**; no active unscored index |

The original source-locked runner returned `ORIGINAL_PUBLIC600_PARTIAL_REAL_EXECUTION_RESUMABLE`, `new_completed_this_session: 1`, `actual_passes: 3`, `original_frozen_gen0_serving_comparable: false`, and `release_approved: false`. It terminated its own native serving session after the one task. The separate Git-pinned reader independently recomputed the checkpoint as `PARTIAL_AUTHENTIC_OFFICIAL_TOOL_EVALUATOR_EXECUTION`, 88 scored, 3 strict passes (3.40909% of **scored** items, **not** 3/600 final official accuracy), 602 positive assertions with 141 passed, and 385/385 excluded assertions passed. Native sessions are distinct and **must not be pooled** into a frozen four-slot comparison.

## Adversarial verification and current resource boundary

Read-only recomputation used `python -m chowder_batch.reap_original600_guardv3_git_pinned_reader_20261027` in a fresh interpreter, with original runner/guard/scorer SHA pins from Git `84ff970`. The two **tracked** source-locked reader/resume test modules passed **28 tests**, with one skipped, and Ruff passed. A broader run that also included the **preexisting untracked legacy** `test_reap_original600_guardv3_locked_resume_20261027.py` yielded **3 failures, 39 passes, 1 skip**: the old wrapper assumes the working-tree runner/auditor still equal historical commit `84ff970` and assumes a worktree `guard.v4` policy. Those assumptions are no longer true; the legacy independent historical auditor also correctly refused current runner-source drift. Do **not** relax source pins or rewrite contributor files to make those obsolete tests pass. Use the newer tracked Git-pinned reader/resume for the authentic historical checkpoint.

After completion, ports 8892/8894 were free and no `llama-server.exe` process remained. However, at approximately 19:12 CDT the GPU showed **9,952 MiB used and 99% utilization** with no attributable Windows native llama-server process. The other GPU workload's exact owner and start time were not independently established; it must be treated as protected. No additional model load or benchmark continuation was attempted after that observation. Every future attempt must recheck actual process ownership, Podman/WSL, physical RAM/commit/VRAM/disk and port occupancy, and source/owner identity immediately before admission.

## Scientific interpretation

This single new scored item is genuine original research progress but **does not** change the completed, independently certified research-only Q4/Q8 HumanEval164 results (151/164 and 152/164), the frozen Q4 GSM8K 1224/1319, the separate Q8 GSM8K 1220/1319, the original historical REAP128-v1 Gate1 **FAIL**, or the unknown REAP160 native speed. No Q8-versus-dense frozen Gate2, dense same-host first200 throughput, full1319 Gate1, BF16 chained numerical, random128 control, HLE or 11-frontier official complete score was established. The unresolved optional Amendment 4 paired McNemar interval remains a prospective independent-statistical-authority issue. No merge, deployment, release, Gen1 training, or frozen-source/weight/certificate mutation was performed.

**Gen-0:** `{"status":"BLOCKED","chosen":null,"release_approved":false}`.
