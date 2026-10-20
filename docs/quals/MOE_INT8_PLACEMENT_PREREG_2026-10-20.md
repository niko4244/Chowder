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
| `memory_vram` | Peak allocated VRAM per **placement step** (forward + backward on `inputs_embeds` through every expert layer, with per-layer recompute as a worker run does it, no LoRA, no optimizer, scoring chunked cross-entropy) | Peak ≤ **14.5 GB** on the 16 GB card, with no WDDM shared-memory spill. Three answers, not two: a peak over the gate only because of the step's own scoring is `harness-bound`, defined under "What a FAIL means", and is not a pass |
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
activation term is GiB-scale -- and `chowder_batch/moe_activation_probe.py`
measures it below rather than guessing it.

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
row-block scratch, a step's weights floor of 5.54 GiB, and 1.66 GiB more for the
step's own scoring (the logits the model returns plus one 256-position chunk of
cross-entropy), i.e. 7.20 GiB before a single activation. Without a measured activation term that row reads
`unsized` rather than silently standing in a zero; with `--activations-gib` from
the probe below it becomes a total. A recorded number that cannot be reconciled
with this arithmetic makes the mode exit 5 rather than print both figures as if
they described one placement.

**The activation term, measured. And it is the whole story.**
`chowder_batch/moe_activation_probe.py` builds one decoder layer of each kind from
this artifact's text config -- shapes only, the weights are never read, because
every tensor autograd saves is sized by shape and dtype -- installs the
placement's own int8 forward so the experts behave as they do at run time, and
counts what autograd retains from the forward pass (deduplicated by storage, the
Function's backward recompute excluded, since that churn is not residency). Two
sequence lengths per kind give a line, `bytes = fixed + slope x seq`:

| layer kind | seq 128 | seq 512 | slope | fitted at seq 2048 |
|---|---|---|---|---|
| `linear_attention`, 30 layers | 0.142 GiB | 0.370 GiB | 638 KB/token | **1.283 GiB** |
| `full_attention`, 10 layers | 0.075 GiB | 0.127 GiB | 147 KB/token | **0.337 GiB** |

That is 5.01 GiB for the whole 40-layer model at seq 128, 12.38 GiB at seq 512
and **41.86 GiB at seq 2048**. The gate's own placement step therefore projects
to 5.54 + 41.86 + 1.66 = **49.06 GiB**, not the 7.20 that weights alone suggest:
the activation term is eight times the weight floor and 3.6x the 14.5 GB gate on
its own. (Upper bounds: overlapping views of one storage are counted separately.
The seq-2048 column is the fitted line, not a measurement, and the line was
checked at a third point: seq 576 predicts 0.4083 GiB and measures **0.4082 --
0.04% apart over a 4.5x range** of sequence length. The slope is the per-chunk
tensors of the linear-attention block, each a fixed slice of tokens, which is why
it is a line at all.)

**What that means for `memory_vram` before anyone reads it as a verdict on int8.**
`_placement_steps` *ran* forward + backward with autograd in eager mode and never
called `gradient_checkpointing_enable()`, so every layer kept its activations
until backward reached it. A worker run checkpoints by default
(`TransformersPeftRunSpec.gradient_checkpointing`, defaulted True), so the step a
campaign pays for holds one layer's activations, not forty. The gate *as first
written* therefore sized a step no campaign performs, and its FAIL would have
meant "activations of an un-checkpointed step", not "the int8 placement does not
fit". That is settled below, on the strength of the same probe run with
checkpointing on: the runner now enables it by default and stamps the row with
which step was measured.

**Gate definition, settled: the step a worker pays for.** `memory_vram` and
`throughput` measure the placement step **with gradient checkpointing on**, which
is what `TransformersPeftRunSpec` does by default and therefore what a campaign
actually pays. The runner defaults to it, records `gradient_checkpointing` in the
record, and stamps the row `checkpointed` -- or `eager` for a deliberate
`--no-gradient-checkpointing` run -- so the two can never be read as one another.
`chowder_batch/moe_activation_probe.py --gradient-checkpointing` measures the
other side of the same layer, at seq 128 and 512 and fitted to 2048:

| retention, GiB | eager | checkpointed |
|---|---|---|
| one `linear_attention` layer | 1.283 | 0.039 |
| one `full_attention` layer | 0.337 | 0.039 |
| all 40 layers | 41.86 | **1.56** |
| one layer being recomputed in backward | -- | 1.28 |

So the gate's step projects to 5.54 (weights) + 1.56 (inputs retained at layer
boundaries) + 1.28 (the layer being recomputed) + 1.66 (the step's own scoring) =
**10.04 GiB at seq 2048**, inside the 14.5 GB gate by 4.46 GiB, where the eager
step was 49.06 GiB, 3.4x over. At seq 512 it is 7.25 GiB. The scoring term still
belongs in the total -- a step has to score something -- but it no longer decides
the answer, and it is now a fixed size rather than a function of the sequence:
the whole-sequence `logits.float().square()` it replaced held 4.74 GiB, a third
of the gate spent on an expression no campaign evaluates.

**Reachability, computed rather than summed in a paragraph.** Those four terms
are no longer hand-added: `--project-only` composes the activation term from the
probe's two records and answers the gate itself --

```
python chowder_batch/moe_int8_gates.py --project-only \
    --base-model G:\llm-cold\bf16\Qwen3.6-REAP160 --seq-len 2048 \
    --activation-probe probe-checkpointed.json --activation-probe-eager probe-eager.json
```

which prints `activations: layer inputs + one layer re-run 2.85`, `step total:
floor + checkpointed activations + scoring 10.04` and `gate 14.5 GB: 10.04 GiB
projected, 4.46 GiB under -- reachable` (exit 0; a projected step over the gate
exits 6, so "reachable" cannot print beside the code that means otherwise). The
records are the same probe run with and without `--gradient-checkpointing`, and
the runner refuses a swapped pair, a record that does not say which step it
measured, and a record missing a layer kind -- each of those would print a
plausible number for a step no campaign runs. What it adds is the 1.56 GiB
retained and the 1.28 GiB of the layer in re-run, on top of the 5.54 GiB floor
and 1.66 GiB of scoring the header projection already derived; on the same
artifact the total is 10.04 GiB against the 14.5 GB gate.

Also settled by the attempt, and it is the same point: `--steps 3 --seq-len 2048`
was run on 2026-10-20 and **refused**, exit 2 in 8 s -- "cuda device has 2.39 GiB
free of 15.93 GiB, below the 6.0 GiB this runner will not step under" (the
training run still holds the card: 13.9 of 16.3 GB used, 8% utilisation). No peak
was taken, and `memory_vram` stays `not measured` until one is: this is the step
it will record, and 4.46 GiB of margin is the first headroom on this gate that
belongs to the placement rather than to the harness.

**What a FAIL means, decided before the run.** `memory_vram` has three answers,
not two, because the step scores a loss of its own: a peak over the gate can be
the placement's fault or the harness's, and after the fact both readings are
plausible. The rule is fixed here, while no card measurement exists:

| outcome | condition | what it obliges |
|---|---|---|
| `pass` | peak <= 14.5 GB | the gate is met |
| `FAIL` | peak > 14.5 GB and (peak - harness) > 14.5 GB | do not declare the placement |
| `harness-bound` | peak > 14.5 GB, but not once the harness is taken off | one prescribed re-run, whose verdict is final |

`harness` is `harness_marginal_gb` in the record: the step's peak allocated
memory after the backward minus the peak after the forward, both taken per step,
so it is what this run's scoring actually added rather than an estimate. A run
that cannot report it cannot attribute, and an over-gate peak is then a `FAIL`.
The `harness-bound` re-run is `--score-tokens 1`: the same forward and backward
through every layer, scoring the last position only, so the head cannot be the
peak. If that run is over the gate, the placement is over.

Two things the rule deliberately cannot do. It cannot become a pass: a
`harness-bound` peak leaves `memory_vram_pass` false and the declaration rule
refuses, because the placement has not been shown to fit. And it cannot adjust a
number: the peak recorded and printed is the peak measured, never a corrected
one, with the harness term beside it rather than subtracted into it -- a table
whose measurement has been edited is a table with no measurements in it.

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
