# Freebuff / Chowder codec experiments — independent archive audit (2026-10-27)

**Finding:** The local Freebuff-labelled October 14, 2026 experiments contain authentic,
parseable model-score receipts for a dense Qwen2.5 checkpoint and an OLMoE checkpoint.
They are **compression-methodology research, not original REAP128_v2 inference**. They
do not satisfy any of the eleven frozen REAP128_v2 frontier evaluations or establish
frozen Gen-0 release eligibility. The original model weights were not changed and
no new inference was launched during this evidence audit.

## Audit provenance and reproduction

- Sources (external local artifacts, **not copied into Git**):
  `G:\freebuff-stage0\run\`, `G:\freebuff-stage1\run\`, and selected
  `G:\freebuff-stage1\` status logs. Original artifacts are dated 2026-10-14.
- Independent verifier: `chowder_batch/freebuff_stage01_source_audit_20261027.py`.
- Audited machine-readable digest: `evidence/freebuff-stage01-audit-20261027.json`.
- Adversarial test cases: `tests/test_freebuff_stage01_source_audit_20261027.py`.
- Re-run read-only on the machine retaining the archived sources:

```powershell
python chowder_batch/freebuff_stage01_source_audit_20261027.py `
  --stage0-root G:\freebuff-stage0\run `
  --stage1-root G:\freebuff-stage1\run
python -m pytest -q tests/test_freebuff_stage01_source_audit_20261027.py
python -m ruff check chowder_batch/freebuff_stage01_source_audit_20261027.py tests/test_freebuff_stage01_source_audit_20261027.py
```

The verifier pins the exact SHA256 of the original row ledgers, reported curves,
block inventories, manifests, Stage 0 sampling metadata and Stage 1 failure logs.
It refuses altered inputs, duplicate or missing scored item indices, score-domain
violations, inconsistent curve values, and mismatched Stage 1 derived rows.
Successful auditing means **existing source files and their recorded scoring agree**;
it is not a fresh independent model-inference replication, independent ground-truth
regrading from generated text, or proof that large serialized arm weight payloads
match self-reported manifest hashes. The manifests' `verified:true` flags are
recorded but not independently treated as payload validation.

## Stage 0: dense Qwen2.5-1.5B-Instruct contrast

Checkpoint: `Qwen/Qwen2.5-1.5B-Instruct`, total 1,543,714,304 model parameters,
CUDA placement; arms BF16, weight-only INT8, and NF4. **Not REAP128_v2.**
The source report is a full run (not `probe-run`), with 48 locally sampled holdout
items per domain in GSM8K numeric, MATH500 numeric, and MMLU lettered.

All 864 JSONL rows parse, with unique block/domain/codec/item keys. Two
*representative dense block labels* (layer 0 and layer 27) contain identical
predictions, **not distinct per-expert measurements**. There are therefore only
432 unique logical domain/item/codec scoring observations (3 × 48 × 3 codecs),
repeated twice in the source rows. The source's two-label 6.0-bit allocation
is a demonstrator, not a physical mixed-codec per-expert deployment.

Correct answers per 48-item holdout, computed from raw rows for **one** representative
label (identical for both):

| Domain | BF16 | INT8 | NF4 |
|---|---:|---:|---:|
| GSM8K numeric | 19/48 | 20/48 | 18/48 |
| MATH500 numeric | 7/48 | 8/48 | 8/48 |
| MMLU lettered | 1/48 | 1/48 | 1/48 |

The signed holdout delta is codec minus BF16; the reported domain "damage" is
`max(0, -delta)`. INT8 has zero nonnegative damage across these three tiny
holdouts; NF4 GSM8K damage is **1/48 = 0.0208333**, with no measured damage
in the other two domains. These scores are neither full official benchmark
populations nor validated downstream REAP model-quality scores.

**Runtime failure:** measured reported total was **7,815.2 seconds** against a
**7,200-second** ceiling (615.2 seconds / ~8.54% over). The embedded
`under_ceiling` flag is `false`. Its derived average allocation was 6.0 bits,
but timing acceptance was not achieved.

## Stage 1: OLMoE-1B-7B-0924 per-expert contrast

Checkpoint: `allenai/OLMoE-1B-7B-0924`; **not REAP128_v2**. The frozen inventory
contains 16 layer-8 expert blocks, each with 6,291,456 parameters. A separate
96-item calibration corpus (32 × 3 domains) was used for criticality preparation;
it is **not** the 48-item-per-domain holdout population used in these scores.

The original `ledger.jsonl` has ten events. It records a 2,304-row baseline
(16 blocks × 3 domains × 48 items) and only **three completed** 144-row altered
arms: `pruned-l8e54`, `crit2bit-l8e54`, and `pruned-l8e49`. A fourth arm,
`crit2bit-l8e49`, was **started** but has no `arm_measured` event or scored rows.
The entire raw file has 2,736 rows (2,304 + 3 × 144), covering 57 complete
48-item block/domain/codec groups. All keys are unique.

The `rows_partial_derived.jsonl` 432-row file is an **exact, order-preserving
subset** of `rows.jsonl` for expert 54: BF16, pruned, and crit2bit, each with
144 rows. It contains **no new inference results**. Only expert 54 has the
full paired two-codec contrast; the other 15 original blocks do not.

For expert 54, the source-backed results (correct per 48) are:

| Domain | BF16 | Pruned (0 bit) | Crit2bit (2 bit) |
|---|---:|---:|---:|
| GSM8K numeric | 1/48 | 1/48 | 1/48 |
| MATH500 numeric | 4/48 | 5/48 | 2/48 |
| MMLU lettered | 0/48 | 0/48 | 0/48 |

The reported damage for expert 54 pruning is zero across the holdouts despite
one MATH500 item improvement; crit2bit loses **2/48 = 0.0416667** MATH500
points versus baseline. These very small, low-performing holdouts cannot
establish a safe global pruning or quantization decision. For expert 49, the
measured pruning arm scored GSM8K 2/48 vs BF16 1/48, MATH500 3/48 vs
BF16 4/48, and MMLU 0/48 in both arms; there is no confirmed crit2bit score.

**Global optimization not completed / refused:** `finisher.log` records
`ValueError: no feasible global bit allocation fits target_average_bits=1.0`
with `minimum_survivors_per_layer=1`, no domain damage ceiling. The log reports
both codecs missing blocks and an evidence landing exit **1**; the expected
`run/allocation_plan.json` is **absent**. A `crit2bit-l8e49` model manifest
being marked verified does **not** imply a scored inference arm.

**Performance qualification failed:** `profiler.log` reports a decode profiling
exit **1** and real-width 1024 MoE dispatch block probe **segfault exit 139**.
Windows/Cygwin `fork` resource errors appear in the stage logs. Earlier small
profile JSON files do not supersede these failed final probes, and neither
sustained native decode nor the frozen serving contract is proven by Stage 1.

## Key source SHA256 digests

| Original local source | SHA256 |
|---|---|
| Stage 0 `run/stage_0_report.json` | `aa8fe621eb8ee5f5fc4158e3b4b4ddd62f1dd468dec43536c8017dd59cf4c0ec` |
| Stage 0 `run/rows.jsonl` | `48676aedb333fb5a8b8c4bee94182eb796e83ff3c688bf9537b145adc4f4e362` |
| Stage 0 `run/block_rate_curves.json` | `6397b93cfda076906038113ee19a7aa3ab4d2023b5a05e2bc399438652c9abf8` |
| Stage 1 `run/ledger.jsonl` | `78cef1e07014c9d1a8e5f82dca69d4d571af0c7adfd4603919b9a22847ce7f51` |
| Stage 1 `run/rows.jsonl` | `55ecaf2b3ed396dfc68d81748bb1c3404a3f04198a689a08aa22b75cd5c43028` |
| Stage 1 `run/rows_partial_derived.jsonl` | `8b383bb6e8d54b67a5fd68dffe0d564a791106a615828d46ff185c7e02eeda91` |
| Stage 1 `run/block_rate_curves.json` | `be0c46be888fae8ae2f7e8d5ab3a650cee83bc7eb9441cb13ab9d2c54fd99280` |
| Stage 1 `finisher.log` | `08ad0256c19d4bb50c924683cbc42544a4e72d2909b73c4989533cfe89b3c0fb` |
| Stage 1 `profiler.log` | `babacdd1d9179441cf159469cabf6c32e7e703584a88102c53ef1f204f56d4d2` |

All other pinned source SHA256 values, including Stage 0 arm manifests and
Stage 1 block inventories, are in the read-only auditor and generated digest.
No Freebuff configuration, session credential, model weights or secret-bearing
material was copied to the repository.

## Relation to frozen REAP Gen-0 and next evidence

1. Preserve the original REAP128_v2 BF16/Q8_0/Q4_K_M evidence and run the
   **eleven registered frontier suites on the actual frozen model**, not on
   these two unrelated exploratory checkpoints.
2. Complete and independently score the original REAP scientific gates,
   including paired numerical parity, source binding, real sustained resource
   measurements under `--fit on -np 4 -c 16384 --jinja`, and frozen Amendment 4.
3. For any future Freebuff codec campaign intended as release evidence,
   preregister the correct REAP model identity, a complete block/codec population,
   actual per-model output receipts, performance and time ceilings, and a feasible
   source-locked global allocation. Do not substitute this incomplete Stage 1.

**Frozen selection remains:**

```json
{"status":"BLOCKED","chosen":null,"release_approved":false}
```

PR #282 must remain open/draft; do not merge, deploy, or begin Gen-1 training
from this exploratory report.
