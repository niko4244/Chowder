# Historical guard.v3 AutomationBench cross-ledger forensic accounting

**2026-10-29; research-only; no model inference.** This certificate is an
arithmetic reconciliation of the historical original Q4_K_M AutomationBench
v1.0.6 guard.v3 run, **not** a new 600-task score or a frozen Gen-0 gate.

The CPU-only verifier
`chowder_batch.reap_original600_guardv3_crossledger_forensics_20261029`
loads the historical evaluator, runner, and guard from Git commit
`84ff9703d6e2aba68eda8585ee461e3d84abec0c` using the already pinned
source hashes. It checks original owner SHA-256
`b6aa1f37e81943344217ede4429066c468bce2176a74eb8b15fcc0fe4a4e0529`,
the source-locked 88-task scorer, and all six SHA-pinned original guard/native
session pairs. The original F: evidence is inspected through one Windows
handle-locked, bounded, private snapshot, with a post-read integrity check.

| Independently checked quantity | Count |
| --- | ---: |
| Receipt-backed scored original tasks | **88/600** |
| Original strict passes | **3/88** |
| Model calls reported by the 88 scored official task outputs | **2,299** |
| Requests in six original guard.v3 journals | **2,396** |
| Guard request count minus scored task model-call count | **97** |
| Receiptless archived task-023 output model calls | **44** |
| Receiptless archived task-024 output model calls | **35** |
| Combined receiptless archived model-call reports | **79** |
| Remaining *unreconciled count difference* | **18** |
| Additional log-only unfinished task archives (041, 087) | **2** |
| Native launches beyond guard journal requests | **2** |

The archived task-023 and task-024 JSON files have pinned SHA-256 hashes
`130a511126b1d67f679ef55ac3f7822e803e90cab8c11b8c93a96db8ce0a0c5e`
and `b721c45992041043f04c7d7aa7b1417934935c3f1711026621cfc3f6876e3508`,
respectively. They are preserved under `.unfinished-` directories, without
completion receipts. Their reported 44 and 35 model calls **do not qualify
them as scored benchmark tasks**, and must not be pooled with their later
receipt-backed retries.

All four archived evaluator logs are separately SHA-pinned (task-023:
`1617dc5491dbddaf9e5b425381dd3ad82f285e81e43dae06275ef2f026997c5e`,
task-024:
`619f8f6610282b110be34188995f8ba6fd331030cb03868c2c8ab7d229a85d11`,
task-041:
`51bee54d2db9490e1412208039f6fa70fdadd31772b5d6aaee654011f8413e7a`,
task-087:
`f64f93655c2e02433d5d026151dedcbee4ca8eebc902942fbcf9cdd68b9b3672`).
Each `.unfinished-` suffix is independently reconstructed from the exact
canonical file-hash inventory algorithm in the historical runner. Empty,
missing, extra, or modified archive files are refused.

Deterministic cross-ledger certificate SHA-256:

`203167537757f3d2288cf82ccea3d78b1805ca45a2cc6911bf5176d9fc5302b2`

## Interpretation and limits

The identity `2,396 - 2,299 - (44 + 35) = 18` is an **arithmetic** fact
about two different instrumentation layers, not proof that every official
model-call counter corresponds one-to-one with a guard request. The residual
18 is not assigned to task-041, task-087, retries, initialization, or any
other cause without request-bound evidence. The two unmatched native launches
are also not presumed to be setup probes. Neither source contains
independently authenticated HTTP prompt/response bodies or native task IDs
bound to canonical task identities.

This fixed historical snapshot verifier refuses unexpected archive inventory,
receiptless archive score promotion, malformed/duplicate-key JSON, wrong
canonical task names, model-call count changes, modified pinned session
bytes, invalid session inventory, archive fingerprint/log changes, negative
residuals, or source/owner drift.
Its pinned archived-output hashes establish reproducibility of the observed
files, **not** retrospective cryptographic attestation of their execution.

The original completed research-only HumanEval Q4 151/164 and Q8 152/164
certificates are unchanged. The historical REAP128-v1 Gate 1 FAIL, REAP160
native speed UNKNOWN, and original dense/frozen Gate 1/Gate 2, BF16 chained
numerical, REAP160 quality, HLE and frontier-suite requirements remain open.
No Amendment 4 statistical method has been prospectively approved.

**Gen-0:** `{"status":"BLOCKED","chosen":null,"release_approved":false}`.
No source-bound F: files, weights, approvals, or protected GPU processes were
modified; no merge, deployment, or Gen-1 training is authorized.
