# Pre-registration — `expert_placement: cpu-transient-grad-int8` (2026-10-20)

**What this registers.** A worker placement for MoE bases whose routed experts
stay frozen and quantized to int8 in host RAM and are dequantized on the device
after each per-layer copy (`chowder.backends.moe_int8_experts`). It is the int8
variant of `cpu-transient-grad-bulk`: the experts are read from the shards
directly at load (never through `from_pretrained`), so the host never holds a
bf16 copy of them.

It exists because the bf16 placement does not fit this box: the REAP-160 Qwen3.6
MoE keeps 160 experts per layer, and one bf16 expert layer at a time plus the
resident non-expert parameters already exceeds the 16 GB card on some layers,
while the bf16 expert set needs ~40 GB of host RAM. As int8 + per-row scales the
expert set is ~20.2 GB, which a 64 GB host can hold.

**Config.**
```yaml
backend:
  expert_placement: cpu-transient-grad-int8
  quantization: none
  lora:
    target_preset: moe_frozen_experts
```
`TransformersPeftRunSpec` refuses any other preset, explicit `target_modules`,
4-bit quantization, or `expert_pinning: registered` with this placement, and the
worker refuses it without CUDA. The live census is written to
`telemetry.expert_placement`.

**Rule.** No campaign may declare this placement until every gate below is
measured on the artifact the campaign trains, with no gate in FAIL. The rule is
not prose-only: `chowder.growth.campaign_prepare.placement_gate_refusal` reads
the table below and refuses the declaration while a gate is unmeasured, failed,
or was measured on a different config digest.

**The table is generated.** `chowder_batch/moe_int8_gates.py` measures the gates
it can and replaces everything between the markers below; the prose above and
below them is hand-authored. Do not edit inside the block — a hand-edited number
is indistinguishable from a measured one, which is exactly what a gate table
must never allow. Run it, then commit the document:

```
python chowder_batch/moe_int8_gates.py --base-model <artifact> \
    --steps 3 --seq-len 2048 --out gates-int8.json
```

A run whose numbers are worse than the recorded ones exits non-zero and still
writes the table, so the regressed numbers are the ones on record. A run against
a different artifact refuses to overwrite the table at all (`--allow-new-baseline`
starts a new baseline). `--check` compares without writing. `--steps 0` measures
the census only and carries the recorded throughput forward rather than erasing
it. `--project-only` needs neither the card nor the weights: it reads the shard
headers and prints what a placement step must occupy beside the recorded
`load_census` and `host_rss`, writing no table -- a projection is not a
measurement, and a number in that block would be indistinguishable from one. `--witness` records a training run's four gates instead of measuring, and is
the only way those rows ever stop saying "not measured".

**What each gate is, and what passes.**

| Gate | Measured as | Pass |
|---|---|---|
| `load_census` | The loader's own live census: expert bytes resident as int8 + scales, modules patched, live bf16 expert parameters left on the model | Every routed-expert parameter is an int8 CPU buffer with **0** live bf16 expert parameters. The census is an assertion, not a report: a placement that did not happen raises in the loader rather than recording a row |
| `memory_vram` | Peak allocated VRAM per **placement step** (forward + backward on `inputs_embeds` through every expert layer, no LoRA, no optimizer) | Peak ≤ **14.5 GB** on the 16 GB card, with no WDDM shared-memory spill |
| `host_rss` | Process peak working set across load and steps | The number is recorded and compared run-to-run (a >10% growth is a regression). The design budget is 24 GB: anything that keeps the bf16 experts resident cannot be under ~40 GB, so a peak far above 24 GB means the int8 grid was copied |
| `throughput` | The same placement step, timed | Recorded, and a >10% regression fails the run. Go/no-go is the worker's own measured s/step against the 2.5 GPU-h per-recipe ceiling — this runner measures the placement, never a training step |
| `gradient` | The LoRA B gradients of the **first** step whose backward populated `.grad`, read once: for the first, middle and last decoder layer that holds a LoRA leaf, every leaf's gradient | All leaves present, all finite, all with a non-zero peak. A layer with no LoRA leaf fails the gate |
| `smoke` | The run's own logged loss points and its optimizer step count | At least **20 steps**, and the mean of the last third of the logged losses at least **1%** below the mean of the first third |
| `router_balance` | Expert-load histograms per MoE layer, from the router's own output: the first step's window and the most recent one | Every layer's entropy within **20%** of its step-0 entropy, and no layer with zero entropy at step 0 |
| `wddm_spill` | Peak allocated and reserved device memory against the card's size | **Operator gate.** No tool can observe the driver paging, so the verdict is recorded with the basis it rests on, and a verdict without a basis is refused |

**How the four witness gates get in.** `chowder.placement_witnesses` derives them
from the evidence a real run collects -- `PlacementWitnessCallback` (LoRA-leaf
gradients, router histograms) plus the run's own loss log and device memory -- and
the worker writes `placement-witnesses.json` beside its adapter and carries the
same record in `telemetry.placement_witnesses`. The runner records it without
taking a measurement:

```
python chowder_batch/moe_int8_gates.py --witness placement-witnesses.json \
    [--spill-verdict pass|FAIL --spill-basis "what the verdict rests on"]
```

That writer replaces only the four witness rows, refuses a witness measured on a
different artifact, refuses to clear a verdict another run earned, and refuses an
operator verdict with no basis. A gate the run did not observe lands as "not
measured" with the reason, so a short or interrupted run narrows the table
instead of filling it.

**Recipe identity.** `expert_placement` is excluded from `recipe_digest()`, like
`frozen_layer_streaming`: it changes where frozen weights live and at what
precision they rest, never what is computed. The int8 path is **not**
bit-equivalent to the bf16 placement — the exactness gate above is what licenses
it, and it is a worker-run gate, not this runner's.

Design: `docs/DESIGN-MOE-16GB-TRAINING-BACKEND-2026-10-18.md` in the hub docs,
option B.

**Census history, 2026-10-20.** Both runs are `chowder_batch/moe_int8_gates.py
--arms int8 --steps 0 --cpu` against `G:\llm-cold\bf16\Qwen3.6-REAP160`
(26 shards, 26/26 read, 40 layers quantised, 27 min each, CPU only so the card a
training run holds is never asked for memory). The table below is the runner's
own record -- the second run's. In GiB (the table's "GB" labels are GiB):

| | `f02b880` (first) | `1a4b535` (on record) |
|---|---|---|
| non-expert parameters | 9.09 (fp32) | **4.55 (bf16)** |
| peak working set (`host_rss`) | 30.1 | **25.56** |
| peak private bytes (watchdog) | 39.13 | **33.50** |
| private bytes on return (loader's own line) | 36.34 | **30.71** |
| int8 stores + scales | 18.75 + 0.07 | 18.75 + 0.07 |
| lowest host-available RAM | 1.36 GB | 5.03 GB |

The 25.56 GB peak is live data, not a copied grid: 18.75 int8 + 0.07 scales +
4.55 non-expert + ~2.3 interpreter/torch/transformers. The gap to the private
figures is int8 storage that `torch.empty` commits before the loader writes it:
committed and resident are different questions on Windows, and only the
working-set row is what the process actually held. The branch head before the
row-block fix was reported holding **37.8 GB** (the unit as reported, not
re-measured here) against ~20 GB of int8 expected.

The first census found that the **non-expert parameters were materialised in
fp32**: every tensor in the shards is BF16, but this multimodal config names its
dtype only in `text_config`, so a loader reading `config.dtype` fell back to fp32
and upcast 4.55 GiB to 9.09 GiB -- ~4.5 GiB of host RAM and double those
parameters' VRAM in a step, on the card this placement exists to fit.
`1a4b535` resolves the dtype the way `from_pretrained` does (config, then
sub-config, then the checkpoint's own header) and the census now records
`dtype: torch.bfloat16` from `text_config`; the second run is that fix, with
int8 stores and scales unchanged to the byte, so the 4.54 GiB difference is the
dtype and nothing else.

**Projection: what that dtype change should do to a placement step. Not measured.**
The card was never used for either census, so everything below is arithmetic from
the artifact's shapes plus the one figure the census did measure. `--steps 3` on
the card records `peak_vram_gb`, `reserved_gb` and `s_per_step`, and that is what
settles it. Shapes: 40 layers, 160 experts, top-8 of them per token, hidden 2048,
moe-inter 512, vocab 248320, 22.573 B parameters of which experts are 20.133 B,
embedding and head 0.509 B each, and the attention/router matmuls 1.423 B (30
GatedDeltaNet layers, 10 full-attention).

Exact, because both sides are the census' own bytes:

| component | fp32 (before) | bf16 (after) | delta |
|---|---|---|---|
| non-expert parameters, resident on the card all step | 9.092 GiB | **4.546 GiB** | **-4.546** |
| one layer's dequantized experts, transient per pass | 1.875 GiB | **0.938 GiB** | **-0.938** |
| int8 staged to the card, per row block | unchanged | unchanged | 0 |

The rest follows from the dtype but not from one number: **every activation tensor
halves**, and before this fix they were fp32 because the hidden states come from
the embedding, whose dtype followed the non-expert parameters. At seq 2048 the
GatedDeltaNet qkv projection output alone (2048 x 8192) drops 0.063 GiB to 0.031
GiB, and each of 40 layers keeps its own set for the backward pass, so the
activation term is GiB-scale -- but how much autograd retains per layer is an
implementation detail, and this table does not pretend to size it.

One term moves the wrong way in this runner: the placement step scores
`logits.float().square().mean()`. With fp32 logits `.float()` was a no-op; with
bf16 logits it materialises an fp32 copy, so seq 2048 x 248320 = 508.6 M logits
cost 0.947 GiB (bf16) + 1.895 GiB (the `float()` copy) + 1.895 GiB (`.square()`)
instead of 1.895 + 1.895. **+0.947 GiB in the gate's own expression**, which a
real cross-entropy step would not pay. Net expectation for `peak_vram_gb`: at
least the 5.5 GiB of table above, less that 0.95, plus whatever the activation
halving is worth.

**Step time.** Per token the forward matmuls are 2.85 GFLOP on the non-expert
path (1.423 B matmul parameters x 2) against 0.050 GFLOP through the 8-of-160
active experts (3.146 M MACs per expert: `[1024, 2048]` gate_up plus `[2048, 512]`
down), so the non-expert path is ~57x the expert FLOPs. It also ran in fp32: this
torch build has `allow_tf32=False` and `float32_matmul_precision = highest`, so
those matmuls had no tensor cores, and the dequantized expert weights were fp32
too (`_dequantized_on` follows the hidden states' dtype). bf16 puts both on
tensor cores and halves the weight bytes every expert matmul reads (0.938 vs
1.875 GiB per layer, twice per step with the backward recompute). But the step is
also 160 small matmuls per layer, which is launch- and copy-bound rather than
FLOP-bound, so expect a modest improvement, not 57x. No number is claimed here.

*Correction, same day:* an earlier revision of the paragraph above said top-2,
0.50 GFLOP and 5.7x. The artifact's `text_config.num_experts_per_tok` is **8**,
and the per-expert count is 3.15 M MACs, so the expert path was understated ~10x
in FLOPs and ~4x in routing width. Nothing else in this document depended on the
ratio -- the weights, the residency and the int8 grid are untouched -- but the
step-time expectation was wrong in the direction that flatters bf16, so it is
corrected rather than quietly restated.

**Reproduced without the card.** `--project-only` recomputes the table above from
the 26 shard headers -- no weights, no model, no device -- and prints it beside
whatever is recorded:

```
python chowder_batch/moe_int8_gates.py --project-only \
    --base-model G:\llm-cold\bf16\Qwen3.6-REAP160 --gates gates-int8.json
```

It returns 18.82 GiB of int8 + scales against the census' 18.82 (agreement to
the byte: one int8 byte per expert parameter and one fp32 scale per row is what
`_empty_int8_experts` allocates), 4.55 GiB of non-expert parameters against the
census' 4.55, and therefore a 23.37 GiB host-resident total against the recorded
25.56 peak -- the 2.19 GiB gap being interpreter + torch, which is the same
figure the census note above attributes. The device terms are the ones the card
would have to hold: 0.94 GiB for one layer's dequantized experts, 0.06 GiB of
row-block scratch, a step's weights floor of 5.54 GiB, and 4.74 GiB more for the
gate's own loss expression (logits + `.float()` copy + `.square()`), i.e. 10.27
GiB before a single activation. The activation row is reported as `unsized`, not
invented, and a recorded number that cannot be reconciled with this arithmetic
makes the mode exit 5 rather than print both figures as if they described one
placement.

**Numerics.** The weights are the same values either way -- the shards are BF16
and fp32 storage was an exact upcast -- but the compute dtype changes: bf16
matmuls with fp32 accumulation instead of fp32 throughout, and the dequant now
rounds once to bf16 instead of staying fp32. That makes this placement closer to
the bf16 bulk path, not bit-identical to the fp32-loading version the first
census measured. The `gradient` gate and the worker's exactness check are what
license it, and both remain unmeasured.

<!-- BEGIN moe-int8-gates (generated by chowder_batch/moe_int8_gates.py; do not edit by hand) -->

| gate | value | verdict |
|---|---|---|
| `load_census` | 18.82 GB int8+scales, 40 modules patched, 0 live bf16 expert params | measured |
| `memory_vram` | not measured (CPU arm) | not measured |
| `host_rss` | 25.56 GB peak (25.56 GB at load) | measured |
| `throughput` | not measured (--steps 0) | not measured |
| `gradient` | not measured by this runner (needs a LoRA run) | not measured |
| `smoke` | not measured by this runner (needs a dataset) | not measured |
| `router_balance` | not measured by this runner (needs a smoke run) | not measured |
| `wddm_spill` | not observable from here; compare peak_vram_gb to the card | not measured |

- artifact: `G:\llm-cold\bf16\Qwen3.6-REAP160` config sha256 `15413a791b9eff423755743bb05b997c93f6ae6f4ce02a1966c9bdfdc2cdc3b9`, 26 shards, 46039573000 bytes
- measured at 2026-10-20T15:31:24Z on cpu (0 steps, seq 2048), code `1a4b535`
- int8 placement step: not measured; bulk comparison arm: not measured

<!-- END moe-int8-gates -->
