# Phase 32 — Completed authentic random128 negative control; Q8 HLE resource retry

**2026-10-26, private experiments only.** Frozen REAP128_v2
BF16/Q8_0/Q4_K_M selector unchanged: `BLOCKED`, `chosen: null`,
`release_approved: false`. Draft PR #282 must stay unmerged. No private
model weights, decoded completions, HumanEval-generated programs or HLE
question IDs are committed.

## Independent random128 original-Q8 negative control: COMPLETE

Recovered completed Kaggle job:
`nikmarco/reap128v2-random128-full-postload`. Original source model SHA-256
`601871e00403658708eb4eebf3b05f9dc54a93b326e31f2f87b48cb59f92241b`;
exact native session ID `a1219c22cab446eca47fefed12f05681`;
original serving flags `--fit on -np 4 -c 16384 --jinja`.

The previously pinned `chowder_batch.reap_external_full_audit` source
independently validated the **complete** original 1,319 GSM8K and 164
HumanEval canonical populations, each row ID/session/correctness bit,
original scorer/source SHA, 41/41 native GPU offload, 4×4096 context,
320 MiB KV, server log source, and actual scored-time resource telemetry.

| Original source-scored suite | Original random128 Q8 | REAP128_v2 Q8 | REAP128_v2 Q4 |
|:--|--:|--:|--:|
| GSM8K canonical 1,319 | **964/1319 = 73.0857%** | 1220/1319 = 92.4943% | 1229/1319 = 93.1766% |
| HumanEval original 164 | **45/164 = 27.4390%** | 152/164 = 92.6829% | 151/164 = 92.0732% |

The random128 HumanEval raw scorer count **has not** been independently
replayed under a proven low-privilege mount/network/PID sandbox. Never
claim its 45 correctness decisions as independently executed.

Strict private audit receipt:

`F:\reap-v2-random128-completed-original-private-audit-20261026.json`

SHA-256:
`ce8137fd9f9fa089ba511b6ad07a3c75e444a8154c13b9d272b6ef262403220b`.

Original machine receipt SHA-256:
`31d502678ae9b62df173d400b2358e3156ba5bc89739f14136fd0b3be29c8f6c`.
GSM8K JSONL SHA-256:
`6788a79f3e4b0ce2d7917301512a5d77e0ce0b498f389cd84a07bb8db391e8a6`.
HumanEval JSONL SHA-256:
`841d5b8db225f303991075963b9eee36425aca62f3eb32785473b675ade3be8c`.
Native log SHA-256:
`a2b7d8852e2d574c1911730632b6582b1911089bba7574565e1fab04013c380c`.

The authentic 1,079 measured resource observations include **958
scoring-time**, minimum **10.508 GiB genuinely unused cgroup**, minimum
**21.090 GiB physical available RAM**, minimum GPU-free **4,967/5,363 MiB**,
and **17.889 GiB** observed recovery following *read-only clean file-cache*
reclamation. These are sampled observations, not instantaneous peaks or
Windows commit and do not prove frozen Windows GPU/CPU parity.

## Actual item-level matched GSM8K control comparisons

The **existing** `reap_external_paired_descriptive` gate independently
required original source-attested complete 1,319-row sessions, then checked
all canonical IDs, row digests, audit receipts, 41/41 native offload and
resource witnesses. Full descriptive partitions:

| Random128 comparison | Both right | Both wrong | Random only right | REAP only right | Random minus REAP |
|:--|--:|--:|--:|--:|--:|
| Versus Q4_K_M | **938** | **64** | **26** | **291** | **−20.09098 pp** |
| Versus Q8_0 | **924** | **59** | **40** | **296** | **−19.40864 pp** |

Private exact-ID pair receipts and SHA-256:

- `F:\reap-v2-random128-vs-q4-paired-descriptive-20261026-new.json`
  `7a456bb80900501cf560f7181b39e1c128f2149bb71fd58a61fb928a0254aaa1`.
- `F:\reap-v2-random128-vs-q8-paired-descriptive-20261026-new.json`
  `1cd65da9408427f17a1bcc77d32bc000742d8c128bd5fbb697b5f975999599ce`.

This is descriptive original Linux **negative-control** evidence, not an
Amendment4 original Windows fit certificate, an approved CI method,
HumanEval independent replay, dense/REAP160 speed ratio, or Gen-0 approval.

## Q8 matched-128 HLE exact-input Kaggle attempt: no scored Q8 rows yet

The original actual-executed Q4 HLE research dataset was exactly 128
completed text-only question IDs, Q4 source row SHA-256
`3cdc96a324ef3fb4d2fa6f0a7a30ed83c6ec7084cca8b654a66469e2b9f1ceeb`.
Its preliminary non-official exact/normalized match was **3/128**.
The attempted Q8 notebook uses the exact same ID set (digest
`65a8f15e747c2a00577aa0a5c5c06ec7724e4c31ec27a99c6a416c3f587acbea`),
the **same exact protocol file hash**
`cbacf0bb2215fce041ca62e991002d2772bee04891bc5e1773d3b0e2c44c49b2`
and protocol manifest SHA-256
`ebdb19e818257c21335889f9bea51c9939293444d315ba0211e21d2c599e70d8`;
temperature 0, seed 0, 2×8192 native sequence context, unchanged
non-official deterministic scorer and full-source Q8 GGUF SHA
`89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6`.

Private notebook Kaggle URL:
https://www.kaggle.com/code/nikmarco/reap128v2-q8-matched128-hle-textonly-20261026

- **Version 1:** source-protocol pin mismatch, safely refused before Q8 load.
- **Version 2:** correct source/pin, but genuinely measured Kaggle initial
  unused cgroup **28.799 GiB** versus original complete Q8 reference startup
  **29.443 GiB**. It refused during model loading at the unmodified
  conservative **10 GiB** cgroup stop, before serving any Q8 HLE responses.
  Private receipt SHA-256:
  `009f9ffd4ba6acbd6fcf3fab33accee6ebc9660a2c89a3b23b0a26b0ceff865e`.
- **Version 3:** separately revised private notebook from the exact
  executed Q4 source. It reclaims unused *image-bearing non-text-only HLE
  rows*, Python/Arrow memory, and clean HLE Parquet file cache **after
  preserving the exact 128 source ID set and verified population count**,
  before the Q8 model load. Native Q8, scorer, prompt, all original source
  hashes, decoding, context and the 10-GiB cgroup stop are unchanged.
  The previously incorrect completed-run status prefix was also aligned
  to the independent Q8 paired auditor's exact expected status.
  Private source SHA-256:
  `be03a8270639b4e1842f1cbdf7c322b149efc9f6b18e050e4643d6e628bad58f`.
  **Kaggle status on last check: RUNNING; no Q8 score available.**

The separate SHA/ID/source-bound independent paired HLE auditor refuses
any Q8 output that is incomplete, truncated, ungraded, from a different
protocol or model SHA, duplicate ID, or missing complete summary.
Never substitute version-2 aborted outputs as Q8 HLE accuracy.

## Remaining full scientific gaps

Full official HLE LLM judge, complete nontruncated 600+ HLE population,
official full 600 AutomationBench for Q8/Q4, the other nine frontier suites,
unverified HumanEval generated-code execution isolation, strict original
Windows model-fit and BF16/forward parity, dense/REAP160 matched comparisons,
and final original release gates remain outstanding. Do not merge draft
PR #282 or select Gen-0 on preliminary Linux results.
