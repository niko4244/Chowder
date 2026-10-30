# REAP128_v2 prospective native capture — temporal and response-model consistency

2026-10-30. **Research engineering only. Gen-0 remains `BLOCKED`,
`chosen=null`, `release_approved=false`.** No original F: artifacts, completed
HumanEval164 journals, weights, frozen evaluator, or certificates were modified.

## Independently reproduced false acceptance

The existing, contributor-owned, **untracked**
`chowder_batch/reap_v2_native_http_capture_witness_20261030.py` verifies that
the event's self-reported `gguf_sha256` equals the serving receipt's model SHA.
It did not check an explicit `model` identifier inside the raw native HTTP
response. An independent CPU-only review altered a synthetic valid response to
say `"model": "forged-foreign-model.gguf"`, recomputed the response hash, and
observed `_verify_events(...) == 2` rather than refusal. This is a real
**consistency** gap, not evidence of a real-model substitution or a release.

The new, standalone
`chowder_batch.reap_v2_native_capture_temporal_contract_20261030` adds a
prospective check that refuses absent, conflicting, duplicate-key, malformed,
oversized or incorrectly hashed response-model claims. A valid response-model
alias must be independently established from the **actual frozen llama-server
invocation and version**, not learned from an event or an unapproved `serve.json`.
No such independently approved alias or collector source is available today:
the production approval maps are intentionally empty, and the production
entrypoint refuses. The production entrypoint independently pins the five
registered original arm populations and GGUF SHA-256 values in source; it
does **not** accept a caller-provided model hash or population. Its pure helper
accepts an expected alias only to enable
CPU-only adversarial tests; it never authenticates the native model.
The production entrypoint also requires a separately approved evaluator
`--parallel` value and passes that bound into the temporal checker. A reviewer
found and reproduced a gap in the first draft: the response-model helper did
not forward the optional concurrency bound, allowing five overlapping HTTP
requests to pass despite a declared four-worker evaluator. This was corrected
before committing, with an adversarial regression test.

## Prospective capture sequence contract

The original evaluator defaults to four HTTP workers, but its `--parallel`
option is independent of the server's frozen `-np 4` decoding slots. **Never
infer an HTTP concurrency ceiling from `-np 4`.** The prospective checker
enforces a supplied HTTP concurrency ceiling only when the actual evaluator
`--parallel` value has been independently authenticated; otherwise it reports
the observed peak without treating it as a frozen scientific requirement.
For future collector development, assign `capture_seq` at request **start**,
and append records in that order, buffering out-of-order completions. A single
collector instance must use one monotonic clock. The independent CPU-only
validator requires exactly contiguous sequence numbers in file order,
nondecreasing start timestamps, strictly increasing per-request start/end
intervals, and no more simultaneous open HTTP requests than the independently
declared evaluator worker count (when supplied). It refuses mixed
process/session/model/collector identities, repeated request IDs, duplicate
JSON keys, malformed framing, and incomplete or extra records. Overlapping
requests within the authenticated HTTP worker limit are permitted; a server
with four decoding slots can still have more than four queued HTTP requests.
The prospective checker additionally requires exactly the canonical task ID
set `gsm8k-0` through `gsm8k-(n-1)`; padded or out-of-range aliases are not
separate legitimate benchmark items.

These requirements are **prospective** and must not be imposed retroactively
on already completed historical journals. Historical guard.v3 records lack
the necessary native request identifiers and raw HTTP bytes; the unresolved
18-request discrepancy cannot be assigned to tasks by this checker.

## Limitations and integration requirements

Both validators check consistency of **self-reported** fields. A malicious
collector can fabricate timestamps, response bytes, request IDs, or a matching
model alias. Neither validator supplies external source ownership, authentic
native transport, independent signatures, process-instance binding, or an
official frozen Gate1/Gate2 result. The new module is deliberately **not wired
into** the contributor-owned untracked witness or frozen evaluator; integration
requires an independently reviewed collector and source-owned immutable
approval evidence, not monkeypatched test approvals.

The completed original Q4/Q8 HumanEval research results remain 151/164 and
152/164, respectively, and the paired SHA-256 certificate remains
`5a2ff5e6ab9f371171de96d368672d52854dd47705c6abb6047de432520893c7`.
The original Windows Q4 GSM8K 1224/1319 and separate Linux Q8 1220/1319 are
unchanged. Missing dense same-session comparisons, BF16 chained numerics,
REAP160, random128, HLE and eleven frontier suites remain unqualified.

CPU-only verification command:

`python -B -m pytest -q -p no:cacheprovider tests/test_reap_v2_native_capture_temporal_contract_20261030.py`

No new inference, native HTTP capture, F: write, process termination, PR merge,
deployment, or Gen-1 training is authorized or claimed.
