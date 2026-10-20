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
it. `--witness` records a training run's four gates instead of measuring, and is
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

**First census, 2026-10-20 on `f02b880`.** `--arms int8 --steps 0 --cpu` against
`G:\llm-cold\bf16\Qwen3.6-REAP160` (26 shards, 26/26 read, 40 layers quantised,
27 min). The table below is the runner's own record; two things the gate row
cannot say on its own, in GiB (the table's "GB" labels are GiB):

- The **30.1 GB peak working set is live data, not a copied grid**: 18.75 of
  int8 stores, 0.07 of scales, 9.09 of non-expert parameters, and ~2.3 of
  interpreter/torch/transformers. A watchdog sampling beside the runner saw peak
  private bytes of **39.13 GiB** and the loader's own last line reported
  **36.34 GiB** private as it returned. The gap to the working-set row is int8
  storage that `torch.empty` commits but the loader has not yet written: committed
  and resident are different questions on Windows, and only the working-set figure
  is the one the process actually held. For comparison, the branch head before the
  row-block fix was reported holding **37.8 GB** (the unit as reported, not
  re-measured here) against ~20 GB of int8 expected.
- **The non-expert parameters were materialised in fp32.** Every tensor in the
  shards is BF16, but this multimodal config carries no top-level `dtype`, so
  `_dtype_from_config` falls back to fp32 and the loader upcasts 4.55 GiB of bf16
  non-expert weights to 9.09 GiB. `from_pretrained` would keep the checkpoint's
  dtype. That is ~4.5 GiB of host RAM here, and it doubles those parameters'
  VRAM in a training step, so it should be settled before `memory_vram` is
  measured on the card. Not changed in `f02b880`: it is a separate defect from
  the two this commit fixed, and this census is the one on record.

<!-- BEGIN moe-int8-gates (generated by chowder_batch/moe_int8_gates.py; do not edit by hand) -->

| gate | value | verdict |
|---|---|---|
| `load_census` | 18.82 GB int8+scales, 40 modules patched, 0 live bf16 expert params | measured |
| `memory_vram` | not measured (CPU arm) | not measured |
| `host_rss` | 30.1 GB peak (30.1 GB at load) | measured |
| `throughput` | not measured (--steps 0) | not measured |
| `gradient` | not measured by this runner (needs a LoRA run) | not measured |
| `smoke` | not measured by this runner (needs a dataset) | not measured |
| `router_balance` | not measured by this runner (needs a smoke run) | not measured |
| `wddm_spill` | not observable from here; compare peak_vram_gb to the card | not measured |

- artifact: `G:\llm-cold\bf16\Qwen3.6-REAP160` config sha256 `15413a791b9eff423755743bb05b997c93f6ae6f4ce02a1966c9bdfdc2cdc3b9`, 26 shards, 46039573000 bytes
- measured at 2026-10-20T14:51:49Z on cpu (0 steps, seq 2048), code `f02b880`
- int8 placement step: None s; bulk comparison arm: None s

<!-- END moe-int8-gates -->
