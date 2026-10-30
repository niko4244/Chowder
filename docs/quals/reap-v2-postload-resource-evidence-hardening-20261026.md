# REAP128_v2 original postload resource attestation hardening — 2026-10-26

**Scientific decision remains `BLOCKED`, `chosen:null`,
`release_approved:false`.** This is a CPU-only source/evidence validation
improvement, NOT a successful Q8, random128, dense reference or Windows gate.

## What was missing

The previously published `reap_external_full_audit.py` independently
verified complete original Q4/Q8/random128 source identities, hashes,
sessions, pinned CUDA output topology and original suite scores, but for
the **two new postload experiments** it did not independently require the
native receipt's *actual monitored admission and cache recovery*. A receipt
with a complete source-bound row set could therefore omit the new 4-GiB
measured clean-cache recovery, the 14-GiB post-advisory headroom or proof
that every monitored runtime sample remained above the extra-conservative
10-GiB unused-cgroup stop. The descriptive paired consumer likewise did
not require a verified postload resource witness for those two model arms.

## New independent fail-closed checks

For precisely `reap128_v2_q8` and independently constructed
`random128_v2_q8`, the complete-cohort auditor now requires:

- Real original pre-load headroom: **at least 24 GiB** measured physical
  RAM available and **24 GiB** unused finite cgroup memory, with exactly two
  explicitly measured GPU-free VRAM readings each **at least 12,288 MiB**.
- Actual, independently supplied `pre_reclaim_host`,
  `post_reclaim_host`, `before_inference_host`, and
  `after_inference_host` resource records; no nonfinite, negative,
  missing, malformed or boolean-as-number values.
- **At least 4 GiB actual measured cgroup recovery** after the original
  native model fully loads, an exact (rounding-tolerant) match to the
  recorded recovered amount, **at least 14 GiB** unused cgroup after
  recovery, and source-bound cgroup `file_gib` attribution decreasing by
  **at least 4 GiB**. Hypothetically reclaimable file cache is never added
  to real unused headroom.
- **At least 7 GiB** available physical RAM and **10 GiB** genuinely
  unused cgroup memory across original baseline, postload, before/after
  scoring and every recorded runtime-supervisor sample, including a
  valid exactly-two-GPU reading in each sample.
- An independently emitted
  `original_measured_postload_resource_witness` with the actual minimum
  sampled available RAM/cgroup, recovered/file-cache GiB, sample count
  and original GPU-free startup readings. Both the original auditor and
  paired descriptive consumer reject absent/corrupted resource proof.

These are **validations of the existing experiment's stricter declared
safety safeguards**, not modifications to frozen Amendment 4 gates or
serving configuration. Q4's previously completed full Linux result is
untouched: Q4 never claimed a postload-Q8 cache-recovery experiment, and
its historical auditor path remains valid.

The validated physical and cgroup figures are sampled headroom minima,
**not** independently established Windows available commit, peak total
RAM allocation, total GPU allocation, Windows CUDA parity, Windows fit or
complete safety against an unobserved between-sample excursion.
The native receipt remains subject to separate original source, complete
result-row SHA, scorer and 41/41 CUDA-offload/4-way context audits.

## Source-backed regression

New adversarial checks cover both Q8 and random128 original model arms;
missing complete resource structure, 24-GiB startup admission, forged
cache recovery, invalid file-cache attribution, missing/invalid GPU
inventories, 7-GiB physical/10-GiB cgroup runtime floor violations,
nonfinite resources and absent downstream paired-safety witness. They
were observed failing before the code change and pass afterward.
Targeted external full-audit plus paired-descriptive matrix:
**118 passed, 2 warnings**, CPU-only. All earlier source-bound complete
Q4 results are preserved without recalculating costly private samples.
The subsequent full adjacent CPU-only qualification matrix passed
**564 tests**, with **1 intentionally deselected** and **5 pre-existing
warnings** (including the unknown pytest `asyncio_mode` config and
deprecation notices).

## External work still outstanding

The exact original private full Q8 kernel
`nikmarco/reap128v2-q8-full-postload-reclaim` and original independent
control `nikmarco/reap128v2-random128-full-postload` were still
`KernelWorkerStatus.RUNNING` at the start of this continuation. Neither
result has been recovered or scored by these added tests. Completion
requires immutable private native receipt, complete 1,319 GSM8K and
164 HumanEval rows, complete original server log, fresh per-arm independent
HumanEval executable replay and descriptive paired comparison.

Mandatory fresh dense/reference full qualification, matched 200-item
dense speed, REAP160 speed, genuine Windows/Linux and BF16 forward parity,
and prospective optional CI-method resolution remain unqualified.
No model selection, release, PR merge, deployment or Gen-1 promotion.
