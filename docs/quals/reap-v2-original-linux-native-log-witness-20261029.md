# Original Linux Q4/Q8 native-server log accounting witness (research only)

This CPU-only read-only audit is independent of the mutable production
scorer. It uses the archived original Q4 and Q8 Linux server logs, native
receipts and 1,319 GSM8K + 164 HumanEval row archives, all with exact
pre-existing SHA-256 pins. No model is loaded and no completed benchmark or
HumanEval Python execution is rerun.

Run `python -m chowder_batch.reap_original_linux_native_log_witness_20261029`.
The native server logs contain **1,483 distinct request lifecycles per arm**.
Each request has one new-prompt event, one prompt-evaluation timing event and
one generation timing event on a consistent one-of-four slot, in order. The
recorded completion-token **multiset** exactly matches the union of archived
GSM8K and HumanEval completion-token counts for each arm, not just their sums.

| Source-bound research witness | Q4_K_M | Q8_0 |
| --- | ---: | ---: |
| Complete native request lifecycles | 1,483 | 1,483 |
| Archived GSM8K responses | 1,319 | 1,319 |
| Archived HumanEval responses | 164 | 164 |
| Native generation tokens | 497,474 | 507,841 |
| Native evaluated prompt tokens | 151,704 | 151,704 |
| Native declared prompt tokens | 167,973 | 167,973 |
| Native-to-archive generation-token multiset | Exact | Exact |

Both original arms also have identical **prompt-token multisets**, consistent
with their shared canonical task population. Concurrent slot scheduling
changes the ordering of requests; no positional correspondence is assumed.
Canonical GSM8K correctness is independently regraded by the previously
committed canonical grader. Archived HumanEval rows are counted and checked
for identity and session only; this witness does **not** re-execute their
generated Python or replace the independently sealed HumanEval certificates.

**Limits:** A llama-server timing log contains task IDs and token counts, not
response text or an independent cryptographic signature of the request and
response. Even an exact per-request token multiset cannot establish which
generated text belongs to which task, authenticate a model's weights at
execution time, or independently prove a native HTTP request/response
transcript. The two Linux sessions are different, are not pooled for a
same-session comparison, and are **not** the frozen Windows Gate 1 or
Q8-versus-dense Gate 2. The historical REAP128-v1 Gate 1 FAIL and REAP160
native-speed UNKNOWN are unchanged.

The module therefore always reports `status=BLOCKED`, `chosen=null`,
`release_approved=false`, and both frozen gates unqualified. It is a
strictly additive historical research provenance check, not release authority.
