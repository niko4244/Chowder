# Original guard.v3 AutomationBench native/guard session reconciliation

**Date:** 2026-10-29. **Scope:** historical research-only original Q4_K_M
AutomationBench v1.0.6, guard.v3. This is **not** an official complete 600
score, not the frozen four-slot Gate 1/2 experiment, and not release authority.

The additive CPU-only verifier
`chowder_batch.reap_original600_guardv3_native_journal_witness_20261029`
reads the original F: tree through the existing Windows handle-locked snapshot.
It loads the historical guard/runner/scorer directly from Git commit
`84ff9703d6e2aba68eda8585ee461e3d84abec0c`, validates original owner
SHA-256
`b6aa1f37e81943344217ede4429066c468bce2176a74eb8b15fcc0fe4a4e0529`,
and independently reconciles six original guard request journals with their
six corresponding native llama.cpp logs. A seventh native log is pinned and
confirmed to contain zero model requests. All seven native logs and six
guard journals are source-pinned by exact SHA-256 in the verifier; no digest
is learned from a self-reported receipt.

## Source-bound observed result

| Original evidence field | Independently observed |
| --- | ---: |
| Historical original scored tasks | 88/600 |
| Historical strict passes (existing score, reconfirmed) | 3/88 |
| Native task launches | 2,398 |
| Native slot releases | 2,395 |
| Releases with both final prompt/decode timings | 2,393 |
| Guard journal request indices, contiguous within sessions | 2,396 |
| Native launches exceeding guard journal entries | **2** |
| Launched native tasks without slot release | **3** |
| Native releases without final timing records | **2** |
| Guard `TOOL_ARGUMENTS_UNREPAIRABLE` events | 5 |

The two excess native launches occur in the first two historical sessions.
They **must not** be silently attributed to a setup probe, a scored task, or
an original client request: the historical records do not authenticate their
purpose. The three unreleased tasks and two released-but-untimed tasks remain
**UNKNOWN**, not completed decodes. The five unrepairable tool-argument events
are counted as journal requests, not dropped as if no model call occurred.
Counts in these categories are not a proof that they refer to disjoint
canonical benchmark tasks.

Deterministic session-accounting SHA-256:

`5105f7f928d5c10eb447d2cb89f8b3e34792aa58235c202e2366f6e97525847e`

The verifier rejects source-byte drift, missing or unexpected sessions,
duplicate/gapped guard indices, guard-policy mixing, malformed JSON, native
slot overlap, duplicate or reordered native task identifiers, truncated
releases, malformed final timing, and more than one unfinished terminal
native task per session. An incomplete or untimed final task is reported
explicitly rather than accepted as complete. Its unit tests use synthetic
adversarial records only; the original F: evidence is never modified.

## Scientific interpretation and limitations

The audit establishes a source-locked *count-level* correspondence between
original guard.v3 proxy request events and native serving task lifecycles.
It does **not** establish a per-request mapping of prompt bytes, HTTP
response bodies, native task IDs, or generated text, and it does not prove
the original GGUF was rehashed inside the historical serving process.
No retrospective hash or numeric token/timing match can create those
missing historical attestations.

The independently replayed 88 official WorldState task outcomes remain
3 strict passes, with 512 tasks unscored. Original dense same-session
comparisons, frozen Gate 1/Gate 2, BF16 chained numerical verification,
REAP160 speed and quality, and the remaining frontier suites are still
unqualified. Amendment 4's nonzero-margin paired McNemar CI ambiguity
remains prospectively unresolved.

**Gen-0:** `{"status":"BLOCKED","chosen":null,"release_approved":false}`.
No model inference, original evidence rewrite, deployment, merge, Gen-1
training, or approval-map change is performed by this verifier.
