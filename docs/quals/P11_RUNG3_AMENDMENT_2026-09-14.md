# P11 rung 3 amendment — bf16 + CPU-resident experts + transient frozen-expert copies

Dated **2026-09-14**. This amends the rung-3 preregistration
([`P11_RUNG3_PREREG_2026-09-14.md`](P11_RUNG3_PREREG_2026-09-14.md), pushed as
PR #161 commit `23756d4`) after its preregistered verdict was executed and
recorded ([`P11_RUNG3_RESULT_2026-09-14.md`](P11_RUNG3_RESULT_2026-09-14.md),
pushed as `79b963b`, plus the PR evidence comment). The preregistered text is
not rewritten; the rung-3 refusal stands as the verdict of record.

## Why an amendment is warranted

The prereg's own verdict logic allowed exactly this branch: when the only
arithmetic fit is refused for a *measured* reason, the recorded options
include a dated amendment that preregisters the candidate the diagnostic
measured. That reason exists here and is not a configuration problem:

- Strategy C loaded and placed perfectly (9.276 GB peak, 64/64 expert
  parameters on CPU, 32/32 router gates on GPU — T5 passed), and was refused
  because **stock transformers 5.16.1 cannot run a forward pass with
  CPU-resident expert weights**: every registered experts implementation
  (`eager`, `batched_mm`, `grouped_mm`) computes
  `F.linear(cuda_activations, cpu_weight)` and raises
  `RuntimeError: Expected all tensors to be on the same device`. The
  grouped-mm kernel auto-selected on this torch/CUDA combination additionally
  requires expert weights co-located with activations.
- The post-prereg diagnostic D (clearly labeled, not part of the verdict)
  measured the amendment candidate end to end on the real 18.82 GB artifact:
  **peak 11.368 GB** (limit 14.0), **mean 1.997 s/step** (batch 2, seq 64;
  limit 3.0), loss 3.036 finite, **32/32 router gate gradients finite and
  non-zero** (sampled max_abs 0.087–0.348), load ~5 s.

## What is being preregistered now

A second, explicitly opt-in load policy for the router-healing backend, named
**`bf16-offload-transient`**, alongside the unchanged default
**`fp32-resident`**:

1. **Load**: the base is loaded as `torch.bfloat16` (never fp32) with each
   expert parameter pinned on CPU and every other parameter on the training
   device, via a per-parameter `device_map` built from meta-device name
   discovery (the same placement strategy C measured). No full-resident fp32
   load exists under this policy; the WDDM shared-memory spill failure mode
   measured in strategy A cannot occur.
2. **Forward**: expert weights stay CPU-resident. The experts forward is a
   per-expert loop that transiently copies each layer's expert weight slices
   to the training device for the duration of the forward and drops them
   afterwards. Copies run under `no_grad`: the router-healing recipe trains
   exactly the router gates, so the frozen experts need no gradients, and the
   transient overhead is bounded by one layer's expert tensors.
3. **What the worker must prove** (evidence fields in its result, all
   measured, never asserted):
   - a placement census **from named parameters**: every expert parameter on
     CPU, every router gate on the training device (the T5 contract);
   - the load policy name and the loaded dtype;
   - the existing frozen-tensor before/after digests, exact gate-path scope,
     and per-tensor trainability evidence unchanged from rung 2's contract;
   - the existing CUDA preflight (measured free memory, step-cost probe,
     projections, refuse-before-step-1) unchanged.
4. **Recipe honesty**: the load policy is part of `recipe_digest()`. Two runs
   differing only in load policy are *different recipes* — a payload's
   recipe must say how its base was resident, because that changes the
   measured cost and the memory contract.
5. **Scope limits**: `fp32-resident` remains the default for every device and
   every existing path; nothing changes for `device="cpu"` runs or for any
   existing test or project file. `bf16-offload-transient` is selected only by
   explicit declaration in the research spec or backend knobs, and is refused
   when the base has no expert parameters (there is nothing to offload; the
   request is then a contradiction, and a silent fallback to full-resident
   would be exactly the silent-scope-creep class this program refuses).
   The evaluation worker accepts the same two policy names with the same
   validation and the same default, so a candidate arm cannot be scored under
   a different load contract than its base arm without that difference being
   declared and digested.

## Measured basis for the pilot budgets (rung 3b, if it runs)

Derived from D's measurements ×1.5, as the prereg required: wall budget
≈ 45 s for a 12-step pilot (measured ~30 s), peak-memory refusal line 14.0 GB
(measured 11.368 GB), step-cost ceiling 3.0 s/step (measured 1.997 s). These
are the budgets a rung-3b preregistration must cite; they are recorded here so
the amendment and the measurement travel together.

## Acceptance criteria (before any pilot)

- Failing tests first: policy validation at spec time (unknown policy refused;
  defaults unchanged), recipe-digest sensitivity to the policy, the CUDA
  offload run proving the census and trainability contract on a tiny real MoE,
  and the no-expert contradiction refusal.
- Mutation checks: neutralizing the placement-census verification, the policy
  validation, or the recipe-payload extension must each fail exactly the test
  that pins it.
- The CPU contract is untouched: the full suite must pass with no behavior
  change for `device="cpu"` / default-policy runs.
