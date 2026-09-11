# Hot-core upcycling

`src/chowder/channel_importance.py` + `src/chowder/hot_core_upcycle.py`.
A second dense→MoE init, beside `dense_to_moe`'s exactness-preserving partition
(which is untouched and still the right thing when exactness is the requirement).

## Why a second converter

The partition converter's ladder collapsed: perplexity 5.44 at `top_k == E` and
243,981 at `top_k = 3`. That was read as sparsity being unavailable. Measurement
on the real dense 9B says otherwise — it was two separable, independently large,
independently fixable choices:

| cause | fix | held-out effect at f = 0.28 |
|---|---|---|
| channels partitioned by **index** | rank by activation magnitude | 614.7× → 1.69× |
| `down_proj` pre-scaled **×E** | pre-scale ×`top_k` instead | removes the (E/k) blowup |

The ×E scale is what a partition needs to be exact at `top_k == E`; because
`Qwen3_5MoeTopKRouter` renormalises unconditionally, it also multiplies every
surviving channel by E/k at any smaller `top_k` — a blowup of a residual stream
the norms never saw. Exactness therefore *requires* dense compute, and buys no
saving. This converter gives that up deliberately.

## The design

1. **Rank**, never index. `channel_importance.measure_channel_importance` sums
   `|h|` per channel (`h = silu(gate(x)) * up(x)`, the vector `down_proj`
   consumes) over a calibration split, via a forward hook on the unmodified
   parent. The ranking is an artifact with its own digest, bound to one
   checkpoint by `source_manifest_sha256`.
2. **Hot core → `shared_expert`**, which `dense_to_moe` fills with zeros.
   Replicating a core inside every routed expert is never better than a plain
   static prune at equal active compute, because `top_k = k` recomputes it k
   times: active cost `k·(h+c)`, coverage only `h + k·c`. In the shared expert
   it is computed once, so **active cost equals coverage**.
3. **Cold channels → routed experts**, dealt round-robin by rank. Contiguous
   rank blocks would make expert 0 the warm one and give a router every reason
   to collapse onto it.
4. **Scaling.** Cold `down_proj` ×`top_k` (renormalised routed weights are
   `1/top_k` each). Shared `down_proj` ×2, because the shared branch is gated —
   the block computes `expert_output + sigmoid(shared_expert_gate(x)) *
   shared_expert(x)` and a zero-init gate gives **exactly 0.5**, so a hot core
   placed there would otherwise arrive at half scale. ×2 leaves the gate at 0,
   where its derivative is largest. `gate`/`up` are copied verbatim: silu is not
   positively homogeneous, so scaling the gate factor changes the function.
5. **`top_k` must be a power of two**, not `num_experts`. The bf16-exactness
   requirement is on the uniform routed weight `1/top_k` and the ×`top_k`
   pre-scale. E is free, which matters because storage is not the binding
   constraint and finer granularity reaches more channel combinations.

Corollary: a non-zero shared expert also removes the zero-gradient defect that
made `shared_expert_gate` unlearnable (all-zero frozen shared projections give
∂/∂gate exactly 0). One fix, two defects.

## This init is not exact, and says so

Provenance records `exactness_contract: "NONE — ..."`. Only
`hot_core_size + top_k · moe_intermediate_size` of `intermediate_size` channels
are present for a given token. What it offers instead is graceful degradation.

## Validation on the real 9B

`F:\llm-models\Qwen3.8-9B-abliterated-25-bf16` → E=16, `top_k`=2, core 2176,
cold 632/expert. Active 3,440 of 12,288 channels (f = 0.2799), 63.3% core.

The mask sweep predicted **1.81×–2.09×** for f=0.28 at a 50–75% core — a
prediction made before this converter existed, from a different code path (a
forward hook on the dense model, no weight surgery).

| | |
|---|---|
| dense parent, nf4, 32 held-out prompts | ppl **5.3137** |
| converted, same load and prompts | ppl **9.8444** |
| ratio | **1.853×** → agrees with the prediction |
| conversion time | 917 s (stdlib byte surgery, no torch) |
| output | 18.82 GB, loads in **stock transformers**, no custom code |
| stored FFN | 4.832B = **1.000× dense** (nothing is replicated) |
| active FFN | **1.353B** (from 4.832B) |
| total params | 9.410B, identical to dense, inside a ≤10B cap |

Byte-level check on the real artifact: all **664** non-consumed source tensors
are byte-identical, including all **108** vision-tower `.mlp.` tensors; the 96
dense decoder MLP tensors are consumed and absent.

The ranking split (even-index) is disjoint from the eval split (odd-index).

## Two things this does NOT buy

* **No memory saving under 4-bit.** Measured peak allocation: dense 5.72 GiB,
  converted **15.07 GiB**. bitsandbytes replaces only `nn.Linear`, and
  `Qwen3_5MoeExperts.gate_up_proj`/`down_proj` are raw `nn.Parameter`, so the
  whole expert bank stays BF16 regardless of `load_in_4bit`. The active-params
  reduction is a **FLOPs** result; memory gets worse, and at 15.07 GiB it only
  just fits a 16 GiB card.
* **No demonstrated capability.** 1.853× is perplexity on 32 prompts from one
  pool. Nothing here shows a trained router recovering any of the
  static→oracle gap (measured at up to 4.30×), which is the entire thesis.
  That is the next experiment, not a result.

## Router init caveat, recorded rather than glossed

The router is written as zeros, so logits are tied and which `top_k` experts win
is an implementation detail of `torch.topk`, not a guarantee. At step 0 experts
outside the tie-break winners may receive no token. This is reproducible but
cold-starts the bank; a training arm should break the tie deliberately rather
than rely on tie order.
