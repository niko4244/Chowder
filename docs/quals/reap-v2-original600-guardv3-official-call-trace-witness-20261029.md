# Original guard.v3 AutomationBench official call-trace consistency

**2026-10-29; CPU-only, research-only.** The source-locked historical
AutomationBench v1.0.6 guard.v3 audit reports **88/600 receipt-backed scored
tasks, 3 strict passes**, and two additional receiptless archived outputs.
This witness independently counts *fields inside those original evaluator
outputs* rather than accepting their `num_model_calls` counters alone.

Reproduce with:

`python -m chowder_batch.reap_original600_guardv3_official_call_trace_witness_20261029`

The verifier first replays the existing historical Git-pinned scorer and
cross-ledger forensic checks against one Windows handle-locked, private copy
of the original F: evidence. Historical guard source commit:
`84ff9703d6e2aba68eda8585ee461e3d84abec0c`; owner SHA-256:
`b6aa1f37e81943344217ede4429066c468bce2176a74eb8b15fcc0fe4a4e0529`.
The underlying cross-ledger certificate remains unchanged at
`203167537757f3d2288cf82ccea3d78b1805ca45a2cc6911bf5176d9fc5302b2`.

| Original official evaluator evidence | Observed count |
| --- | ---: |
| Scored original outputs | **88** |
| Receiptless archived outputs, **not scored** | **2** |
| `num_model_calls` in scored outputs | **2,299** |
| `num_model_calls` in receiptless outputs | **79** |
| Assistant-role messages across all 90 outputs | **2,378** |
| Recorded steps across all 90 outputs | **2,378** |
| Recorded stop reasons across all 90 outputs | **2,378** |
| Recorded finish reasons across all 90 outputs | **2,289** |
| Task-level disagreements among model calls, assistant messages, steps, stop reasons | **0/90** |
| Guard journal requests (separate instrumentation layer) | **2,396** |
| Unreconciled aggregate guard-request difference | **18** |

The 89 fewer finish-reason records are **not** treated as missing model calls:
the original outputs preserve a separate `stop_reasons` list with exactly one
entry per reported call. The verifier refuses altered model-call counts,
missing/extra assistant messages, incorrect step counts, stop-reason count
changes, malformed message roles, duplicate JSON keys, nonfinite numbers,
changed original output SHA pins, and changes to the historical source-locked
population. It records only a hash of message **roles**, not private content.

New deterministic official call-trace certificate SHA-256:

`22b9f2160011d70ceb7bccbc6a3ac321ea60ff6b06c52a4b8ae366190142c6ef`

## Limits and decision

These fields are **internally consistent but not independently generated**:
all originate from the same archived evaluator JSON. The result is stronger
source-bound arithmetic corroboration, **not** independent native execution
attestation, per-request HTTP prompt/response authentication, or evidence
that every guard journal request corresponds to one official model call.
The 18-request aggregate difference remains unassigned. The receiptless
archives remain unscored; completed research-only HumanEval Q4/Q8 results are
not rerun or promoted to frozen Gate 2.

Original dense comparisons, frozen Gate 1/Gate 2, BF16 chained numerics,
REAP160 speed/quality, HLE and frontier suites, and prospective Amendment 4
statistical authority remain outstanding. Historical REAP128-v1 Gate 1 FAIL
and REAP160 native speed UNKNOWN remain unchanged.

**Gen-0:** `{"status":"BLOCKED","chosen":null,"release_approved":false}`.
No original source/evidence/certificate or weights were changed; no model
inference, merge, deployment, or Gen-1 training was performed.
