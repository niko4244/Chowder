# Chowder engineering and custom MoE — executive verdict

> Audit snapshot: 2026-10-22 UTC. Main 0bc566c5db052b90e3aa1223fef94445d1045dc5; PR #268 fe94342c1073eb39e69be01ba224c63b38057b95; recovery PR #269 e3f0fbd77bc679ea9a856461294cd26b40316562. GitHub code and published evidence were inspected; local Windows workspace inaccessible (Chat On Steroids Core root /strata invalid). Source claims below are **historical reports**, not rerun hardware measurements. No model weights or protected local evidence were accessed in this audit.

## Answer first

**Chowder is an implemented research/control-plane system, but the custom REAP-MoE model is not qualified as a release-ready, reproducibly packaged Gen-0 or proven autonomous improvement.** The previous audit was stale: PR #268 now records real 20-step CUDA training and a sanctioned int8 VRAM-gate pass for an older setting, yet the frozen overall qualification still lacks gradient/smoke/router/WDDM evidence and recipe throughput is above budget.

## Work completed in this pass

1. Reconciled main HEAD, open PRs and current #268 test/benchmark documentation; did not equate PR content with merged software.
2. Found and patched journal stale-terminal/torn-verdict fail-open behavior on PR #269; fsynced tmp verdict and added regression tests. CI started for the new head; final result not yet available at audit snapshot.
3. Created an audit branch with all twelve user-specified deliverables, including machine-readable status/finding/model manifests.
4. Corrected stale MoE Gen-1 dataset documentation to match the merged 2,048-token drop-not-truncate builder and optional Self-OSS source.

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
