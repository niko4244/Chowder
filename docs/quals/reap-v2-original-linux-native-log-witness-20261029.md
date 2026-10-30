# Original Linux Q4/Q8 native-server log accounting witness (research only)

This CPU-only read-only audit is independent of the mutable production
scorer. It uses the archived original Q4 and Q8 Linux server logs, native
receipts and 1,319 GSM8K + 164 HumanEval row archives, all with exact
pre-existing SHA-256 pins. No model is loaded and no completed benchmark or
HumanEval Python execution is rerun.

Run `python -m chowder_batch.reap_original_linux_native_log_witness_20261029`.
The native server logs contain **1,483 distinct request lifecycles per arm**.
Each request has one new-prompt event, one prompt-evaluation timing event,
one generation timing event, and one explicit nontruncated slot-release event
on a consistent one-of-four slot, in order. Every release's final token count
equals the request's declared prompt tokens plus generated tokens minus one.
The recorded completion-token **multiset** exactly matches the union of archived
GSM8K and HumanEval completion-token counts for each arm, not just their sums.
The strengthened parser also enforces **global per-slot exclusivity**: a
second request cannot start on an occupied slot, timing must belong to its
currently active task, and all slots must be idle at log completion. This
rejects an interleaving that passed the earlier per-task-only order check.

| Source-bound research witness | Q4_K_M | Q8_0 |
| --- | ---: | ---: |
| Complete native request lifecycles | 1,483 | 1,483 |
| Archived GSM8K responses | 1,319 | 1,319 |
| Archived HumanEval responses | 164 | 164 |
| Native generation tokens | 497,474 | 507,841 |
| Native evaluated prompt tokens | 151,704 | 151,704 |
| Native declared prompt tokens | 167,973 | 167,973 |
| Native-to-archive generation-token multiset | Exact | Exact |
| Concurrent active serving slots observed | 4 | 4 |
| Explicit nontruncated releases, with per-task token accounting | 1,483 | 1,483 |
| Cold first-slot prompt requests | 4 | 4 |
| Later requests reusing 11 prompt tokens | 1,479 | 1,479 |
| Native prompt tokens saved by reuse | 16,269 | 16,269 |
| Individually unique archived-row/native-task timing fingerprints | **1,483/1,483** | **1,483/1,483** |
| Native prompt-token lengths equal for the **same canonical item** across arms | **1,483/1,483** | **1,483/1,483** |

Each archived row's `completion_tokens` and full-precision
`predicted_per_second` determine its native generation duration as
`(completion_tokens - 1) * 1000 / predicted_per_second` milliseconds.
The independently pinned native server log prints that duration rounded to
0.01 ms. Requiring the same token count and a maximum 0.0051 ms rounding
discrepancy produces **exactly one native task ID per archived row** in both
original cohorts, with no task reused and no unmatched native tasks. This
upgrades the previous aggregate token-multiset correlation to a source-bound
**per-item numeric timing fingerprint**.

The deterministic JSON-sorted row-ID-to-native-task mapping SHA-256 values are:
`f6c4612fa2a61a23a2b83ebf5e78c2fe3daa256d844f22f496b53213a13b6040`
(Q4) and
`86afb604b4a5d3f6a95564e3042fb0700f9a644d1518dbec809340e79415c861`
(Q8). These are reproducible derived mapping digests, **not** independently
signed native execution attestations or replacements for original evidence.

Both original arms also have identical **prompt-token multisets**, consistent
with their shared canonical task population. Concurrent slot scheduling
changes the ordering of requests; no positional correspondence is assumed.
The verifier now follows each archived row's unique timing match back to its
native task's declared prompt-token count, checks all 1,319 canonical `gsm8k-N`
and 164 `HumanEval/N` identities, and requires **per-item** prompt lengths to
match between Q4 and Q8. The sorted per-item length-map SHA-256 is
`c11f1973905352303f6890e4ec5781e3776a89aacbd46281a881f2f92811e8b0`.
An adversarial swap of two item lengths preserves the aggregate multiset but
is now rejected. Identical lengths do **not** establish identical prompt text,
tokenizer content, or authenticated HTTP request bytes.
The per-request prompt-cache deficit is exactly zero on the first request in
each of four slots and exactly 11 tokens on every subsequent request. The
auditor rejects any different cache-deficit distribution, even if the aggregate
prompt token totals remain unchanged. This is a historical server-accounting
observation, **not** proof of individual response-text provenance.
Canonical GSM8K correctness is independently regraded by the previously
committed canonical grader. Archived HumanEval rows are counted and checked
for identity and session only; this witness does **not** re-execute their
generated Python or replace the independently sealed HumanEval certificates.

**Limits:** A llama-server timing log contains task IDs and token counts, not
response text or an independent cryptographic signature of the request and
response. Even the now-unique per-row numeric timing fingerprint cannot prove
that the archived response **text** came from the native task, authenticate
the model's weights at execution time, or independently prove a native HTTP request/response
transcript. The two Linux sessions are different, are not pooled for a
same-session comparison, and are **not** the frozen Windows Gate 1 or
Q8-versus-dense Gate 2. The historical REAP128-v1 Gate 1 FAIL and REAP160
native-speed UNKNOWN are unchanged.

The module therefore always reports `status=BLOCKED`, `chosen=null`,
`release_approved=false`, and both frozen gates unqualified. It is a
strictly additive historical research provenance check, not release authority.
