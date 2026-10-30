# REAP128_v2 Phase52 — original GSM8K correctness admission and cross-host research pairing

2026-10-30. Gen-0 remains **BLOCKED**, `chosen=null`, `release_approved=false`.
This work did not execute model inference, generated Python, or any benchmark
again. The two completed HumanEval164 research certificates were untouched.

## Frozen admission improvement

The existing original v2 frozen-evidence bridge verified original v2 GSM8K
score booleans, canonical item IDs, serving context and independently pinned
row bytes, but did **not** independently recompute each original v2 GSM8K
`correct` value. A source-approved hash attests to bytes, not to the
correctness of the evaluator's grading. This is distinct from the historical
prerequisite path, which already performed source-bound canonical regrading.

`_verify_original_sessions` now invokes the existing SHA-pinned original
GSM8K scorer for **all five** registered v2 GSM8K cohorts: three full 1,319
arms and both 200-item Q4 first-prefix arms. Every item is checked against
the complete frozen 1,319-answer canonical manifest and original
`final_number_match` semantics, including the `terminated` requirement.
Tests demonstrate that even synthetically re-sealed summary/reviewer hashes
cannot promote a wrong `correct` label, and that all five arms reach the
independent regrader. Production approval SHA maps remain empty. The bridge
does **not** establish native HTTP request/response authenticity; independent
original native provenance and remaining frozen gates are still required.

## New source-pinned, original cross-host research comparison

The read-only tool
`chowder_batch.reap_original_gsm8k_windows_q4_linux_q8_pair_20261030`
authenticates the two preexisting F: row files against source-owned SHA-256
pins, independently regrades all 1,319 canonical items with **two**
implementations, and joins outcomes by exact canonical item ID. No F: file
was changed. Original row source hashes:

| Original evidence | SHA-256 |
| --- | --- |
| Windows Q4 original GSM8K1319 | `66d0f39545dd8769e837ed5772a265871cfe148bb9a529ea2188454d72c1243f` |
| Separate Linux Q8 original GSM8K1319 | `9ff2b418fecabdc419d94b2a06f6d9b115bbbd4bfba59ccae2d0d9f81e8ef749` |
| Frozen canonical GSM8K population | `913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89` |
| Reconstructed paired per-item witness | `af5b1f2c7aa6c6759ed3bfb54f450d1fc27ddb7ba0a06227c3409599713ead77` |

The deterministic compact JSON certificate (`sort_keys=True`, compact
separators, UTF-8, no nonfinite values) has SHA-256
`a20e3aa83e551d25402c6ddcf683c9c3f265472377ec87c0da7c2227f9681855`.
This is a research-integrity fingerprint, **not** a native capture or
release-authority signature.

| Verified paired result | Count |
| --- | ---: |
| Both correct | 1,199 |
| Windows Q4 correct, Linux Q8 incorrect | 25 |
| Linux Q8 correct, Windows Q4 incorrect | 21 |
| Neither correct | 74 |
| Windows Q4 correct | **1,224 / 1,319** |
| Linux Q8 correct | **1,220 / 1,319** |
| Windows Q4 natural terminations | 1,281 |
| Linux Q8 natural terminations | 1,280 |

The paired research quality difference is **+4/1,319 in favor of Windows
Q4**, with exact two-sided conditional McNemar `p =
11588642805541/17592186044416` (approximately **0.658738077**).
This is an exact paired correctness calculation, not a noninferiority result.
It does not compare Q8 against dense, does not establish same-host native
serving, and cannot satisfy original frozen Gate1 or Gate2. This pair also
differs from the previously published **Linux Q4 vs Linux Q8** research
pairing (Q4 1,229/1,319), which must not be pooled with Windows Q4.

The paired tool explicitly returns `native_http_lineage_authenticated=false`,
`same_host_frozen_comparison=false`, `research_only=true`,
`status=BLOCKED`, `chosen=null`, `release_approved=false`. SHA-pinned
row bytes do not independently prove native request/response lineage,
model weights, or physical-host/session ownership.

## Reproduction and scientific constraints

From the authorized worktree, CPU only:

`python -B -m chowder_batch.reap_original_gsm8k_windows_q4_linux_q8_pair_20261030`

Tests use synthetic in-memory cases and temporary files, never a rerun of
the original model or F: mutation. The existing original Windows Q4 and
Linux Q8 scores are reproduced, not changed. The original Q4 HumanEval
151/164 and Q8 HumanEval 152/164 research-only certificates remain final.

Still outstanding: independently authenticated native request/response
lineage, same-host dense Q4 first200, full original Q8-vs-dense 1,319/164
frozen Gate1/Gate2, full BF16 chained numerics, REAP160 speed/quality,
random128 control, official HLE completion and the remaining registered
frontier suites, and prospective Amendment4 statistical adjudication.
