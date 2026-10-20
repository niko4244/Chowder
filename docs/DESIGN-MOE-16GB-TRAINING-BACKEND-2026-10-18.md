# Design: training the REAP-128 Qwen MoE on a 16 GB card (Chowder backend)

The canonical design document lives in the hub docs at
`docs/DESIGN-MOE-16GB-TRAINING-BACKEND-2026-10-18.md` (option A: frozen
experts on the CPU, a transient autograd function per layer). This repo copy
records the layer-bulk amendment so the change is reviewable with the code it
describes.

## Layer-bulk expert transfer (`cpu-transient-grad-bulk`)

Option A copies each hit expert's two weight matrices to the device in forward
and again in backward. On REAP-128 that is ~128 hit experts × 2 copies × 3
passes (checkpointed forward, recomputed forward, backward) ≈ 770 small
pageable copies per layer per step; the measured 39.6 s per 2048-token step is
essentially all transfer.

`TransientLayerExperts` copies the whole layer once per pass instead: two bulk
`.to(device, non_blocking=True)` calls for `gate_up_proj` and `down_proj`
(roughly 0.8 GB per layer for REAP-128) rather than ~160 small ones, on the
same forward/backward schedule. The output, the expert order and the
`index_add_` accumulation are identical to the per-expert path by
construction: both run the same loop, one over per-expert CPU slices and one
over device copies of the same matrices. Backward recomputes with grad enabled
w.r.t. the hidden states and the routing weights, so gradients reach the router
and every LoRA leaf below the layer exactly as before, and no expert weight is
ever saved for backward.

**Peak memory bound.** One layer's experts are on the device at a time, and
only for the duration of the call: the copies are local to the autograd
function's forward or backward, and are dropped before the next layer runs.
That is the same bound as the per-expert path (which also holds one expert at a
time), so the bulk copy does not raise the peak; it changes only the number and
size of host-to-device transfers. `expert_pinning: registered` page-locks the
CPU expert tensors in place (no second copy in RAM) so those transfers can be
genuinely asynchronous; it is off by default.

**Speed is not measured yet.** The 39.6 s/step figure above is the per-expert
path. The bulk path's s/step, peak VRAM and any WDDM shared-memory spill must
be re-measured with the gates in
`docs/quals/MOE_TRANSIENT_GRAD_PLACEMENT_PREREG_2026-10-18.md` before any
growth campaign declares this placement. The CPU tests prove gradient and
output equivalence only; they say nothing about GPU speed.
