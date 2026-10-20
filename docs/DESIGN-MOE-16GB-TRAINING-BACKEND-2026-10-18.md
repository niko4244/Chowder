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

## int8 frozen experts (`cpu-transient-grad-int8`, option B)

The bulk path still keeps the routed experts in host RAM as bf16. For REAP-160
(Qwen3.6-35B-A3B pruned to 160 experts/layer, 42.9 GB bf16) the experts alone
are ~40 GB, and this 64 GB host has ~24-33 GB available, so a training step
pages to disk (measured: step 1 = 1,188 s). Option B keeps the same frozen
experts in RAM as **int8 with one scale per output row** and dequantizes on the
training device after the copy.

**Memory math.** Expert weights ~20 G elements (~40 GB bf16) become one byte
each: 20 GB. The scales add one fp32 per output row -- 4 bytes per row of `w`
weights, so `4/w` per weight: ~1.6% at `w=256` (the routed `down_proj` rows,
`hidden`) and ~0.8% at `w=512` (`gate_up_proj` rows), i.e. ~20.1-20.3 GB in
total, roughly **8.1-8.5 bits per weight instead of 16**. The loader streams the
shards so the bf16 experts are never resident: the model is built on `meta`, the
expert parameters are dropped from the skeleton before it is materialised, and
each shard's tensors are read one at a time (an expert tensor is quantised and
released before the next key is touched). The census it returns -- expert bytes,
non-expert parameters on the device, patched modules -- is the measurement to
run against reap160; the figures above are arithmetic from the checkpoint size,
not a measurement of it.

**The artifact's key layout.** `chowder_batch/reap_prune.py` slices
`model.language_model.layers.<i>.mlp.experts.gate_up_proj` [E, 2I, H] and
`...experts.down_proj` [E, H, I] in place and copies every other tensor
byte-for-byte, so reap160's shards carry **fused** expert keys under a
`model.language_model.` prefix (the conditional-generation nesting) and a
multimodal `config.json` (`model_type: qwen3_5_moe` with a `text_config`).
`from_pretrained` rewrites that prefix; the streaming loader does it itself
(`_model_key`), and reports how many keys it renamed. Tensors a causal-LM
skeleton has no module for -- a conditional-generation export's vision tower, or
the MTP head -- are skipped and listed in the census (`ignored_keys`), never
dropped silently.

**Quantisation.** Symmetric per-output-row int8: `scale = absmax(row)/127`
(fp32 in RAM), `q = round(w/scale)` clamped to `[-127, 127]`. Dequantisation is
`q * scale` in fp32, cast to the compute dtype, performed on whichever device
the int8 was copied to. Only the routed experts are quantised; routers,
attention, shared experts, embeddings and norms stay in the load dtype.

**What changes numerically.** Each frozen expert weight carries at most half a
step of its row's grid, i.e. `absmax(row)/254` (<= `absmax(row)/127`), about
0.4% of the row's magnitude. That perturbs the expert outputs, and so the
gradients the router (`top_k_weights`) and every LoRA leaf below the layer
receive; the routing decision itself is still made by the bf16 router. The CPU
tests on a tiny synthetic MoE put the relative error at ~1.1e-2 on the layer
output, ~1.6e-2 on the hidden-state gradient and ~0.8e-2 on the routing
gradient against the bf16 bulk path (tolerance 2e-2). That is a real numerical
change, not a bit-identical optimisation, so a run under this placement is not
the same run as under `cpu-transient-grad-bulk`.

**Peak device memory** is still one layer's worth of experts at a time, but the
transient is slightly larger than the bulk path's: while a layer is
dequantized, its int8 copy and the bf16 result are both on the card (~1.5x the
layer's bf16 bytes), where the bulk path peaks at one bf16 layer, and both are
dropped before the next layer runs. The fp32 product is accumulated in 16 MB
row blocks (`DEQUANT_ELEMENTS_PER_BLOCK`) so a full fp32 layer never exists.
Transfer volume halves; the transfer count doubles (int8 and its scales
instead of one bf16 matrix), so whether the smaller copies win is a
measurement, not an argument.

`expert_pinning: registered` is refused with this placement: the int8 buffers
are not page-locked, so the copies stay synchronous.

**Gates.** This placement inherits the pre-registration in
`docs/quals/MOE_TRANSIENT_GRAD_PLACEMENT_PREREG_2026-10-18.md`, and those gates
-- s/step, peak VRAM, WDDM spill -- must be **re-measured** on the real
artifact before any campaign declares `cpu-transient-grad-int8`. The CPU tests
here prove the mechanism (round trip bound, no bf16 expert resident, streaming,
forward/gradient agreement, checkpointing, admission) and say nothing about GPU
speed or memory.
