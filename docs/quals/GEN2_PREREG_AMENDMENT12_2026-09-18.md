# Gen-2 preregistration amendment 12 — making the arms measurable, and measuring the parent under the declared instrument

**Date:** 2026-09-18
**Status:** pre-compute infrastructure amendment. **No Gen-2 candidate training has
been run** (the arm measurements below are evaluation compute on existing models).
No threshold, benchmark set, protocol, budget, stopping rule, contamination
requirement, provenance rule, trusted-ancestor rule or verdict semantic is
altered.

## A. Why the base-only measurement could not run

Amendment 11 added `chowder growth campaign measure-ancestor`, but its first real
runs could not complete on this host:

1. **Attempt 1** — `torch.OutOfMemoryError`: a 64 MiB allocation failed on the
   first forward with ~10.9 GiB nominally free.
2. **Attempt 2** — a native `ACCESS_VIOLATION` (exit `0xC0000007`) during
   `from_pretrained` weight loading, after `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   was set to try to work around (1).

The OOM is the shape accelerate warns about for exactly this model class:
*"Current model requires 256 bytes of buffer for offloaded layers, which seems
does not fit any GPU's remaining memory … consider using `offload_buffers=True`."*
`dispatch_offloaded` pinned every decoder layer's **weights** to the host but left
its **buffers** on the card, so the residual fragment could not be satisfied
during generation even with room nominally reported free.

## B. The fix

`chowder.evaluators.placement.dispatch_offloaded` now passes
`offload_buffers=True`, so an offloaded layer's buffers ride with its weights.
Verified two ways:

* an isolated probe loaded the declared dense base and generated tokens under
  `placement="offload"` **successfully** (`GEN OK torch.Size([1, 9])`) where the
  un-fixed path OOM'd;
* the real `measure-ancestor` run then progressed past the first forward (GPU
  utilisation sustained, no OOM) instead of dying on it.

The candidate evaluator uses the same placement, so this is a fix to the one
offload path, not a special case for the arm.

## C. The parent arm under the declared instrument

The arm measurement is now one production path, `campaign_prepare.measure_arm`,
with two callers:

* `chowder growth campaign measure-ancestor` — the dense base, **no adapter
  loaded**, rows `MEASURED_PARENT` under `trusted_ancestor_version`;
* `chowder growth campaign measure-parent` — the **parent adapter over the
  declared base**, under *this campaign's declared instrument*, rows
  `MEASURED_PARENT` under `parent_version`.

This closes the blocker Amendment 11 named. The Gen-1 durable evidence could not
serve as the parent's target row because it measured
`generation-diagnostics@gen1-eval-protocol-v1`, not the Gen-2 instrument; with
`measure-parent` the parent is measured under the declared instrument, so the
promotion rule's target comparison has both arms. The rows are never
candidate-measured: a parent measurement cannot become the candidate's evidence.

## D. What did NOT change

Every threshold, benchmark set, mini-slice protocol (16 items, seed 1234, greedy,
512 tokens, chat template), budget, stopping rule, provenance rule and verdict
semantic. `competence` and honesty rules from Amendments 2/5/11 stand: the
ancestor arm is still a fresh measurement of the untouched dense base, and the
parent arm is the parent adapter, not the candidate.

## E. Honest note on runtime

Both arms run the base under `placement="offload"` with the frozen 512-token
ceiling, and the Gen-0 base does not emit EOS (its measured EOS rate is 0.000),
so **every** generation runs the full cap. On a 16 GB card sharing the accelerator
with another process this is an hours-scale job per arm. The commands refuse
rather than write a partial arm if the worker fails; a started-but-unfinished
measurement simply has no arm yet, which readiness reports as
`READINESS_ANCESTOR_ARM`, not as a pass.
