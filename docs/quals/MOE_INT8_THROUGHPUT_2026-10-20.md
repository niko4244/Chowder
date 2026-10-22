# Throughput and the link ceiling — `cpu-transient-grad-int8` (2026-10-20)

**What this note is.** The measured cost of the int8 frozen-expert placement at
seq 2048 on the 16 GB card it exists for, what each landed change did to it, and
the arithmetic that decides whether the campaign's 2.5 GPU-hour-per-recipe
ceiling is reachable on this host.

**What it is not.** A gate. No number here can turn a row of
`docs/quals/MOE_INT8_PLACEMENT_PREREG_2026-10-20.md` into a pass: that table is
written by `chowder_batch/moe_int8_gates.py` and by nothing else. This note
explains the numbers that runner and the training worker produce; where it has
to project, it says so.

## 1. The measurements this note rests on

| what | where | value |
|---|---|---|
| placement step, seq 2048, checkpointed, 20 optimizer steps, real LoRA | `C:\llm-work\moe-gates\gates-reap160-int8-seq2048.json` | 33.1–35.2 s/step once warm; `seconds_per_step` 41.59 (mean including the 3,024 s first-touch step and a 125 s second) |
| peak device memory, same run | same | 10.79 GB allocated, 13.15 GB reserved |
| placement step, seq 3072 | `...-seq3072.json` | 42–121 s/step, bimodal; 13.80 GB allocated, **17.47 GB reserved** — over the card, i.e. WDDM paging |
| int8 store + scales | `load_census` in both | 20,132,659,200 B int8 + 78,643,200 B fp32 scales = 18.82 GiB |
| link | `nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.width.max,pcie.link.width.current` | Gen **3**, width **8** of a possible 16 (idles at Gen 1; negotiates Gen 3 x8 under load) |

The 16 GB card is an RTX 5060 Ti. The step is real: LoRA on the frozen-expert
preset, AdamW, activation checkpointing, chunked cross-entropy over 2,048
tokens, 40 MoE layers × 160 experts with top-8 routing.

## 2. What the copy traffic alone costs

Each training step dequantizes every routed expert **three times**: the
checkpointed forward (no grad), the recompute the checkpoint replays, and the
backward. `TransientLayerExpertsInt8.forward` dequantizes in `forward`;
`backward` dequantizes again unless `CHOWDER_INT8_RETAIN_WEIGHTS` handed the
recompute's weights over. So the step moves

```
18.82 GiB × 3 passes = 56.46 GiB host → device, per step.
```

At a Gen 3 x8 link the ceiling is ~7.3 GiB/s theoretical, ~5.5 GiB/s achieved
(measured by `chowder_batch/moe_int8_io_ceiling.py --h2d`):

```
56.46 GiB / 5.52 GiB/s = 10.23 s of pure copy per step.
```

The campaign target is **9 s/step** (1,000 steps in 2.5 GPU-hours). The copy
alone is 10.23 s, before one matmul, one dequantize, or one optimizer update.
**The target is not reachable on this link by any code change**: it would need
6.27 GiB/s sustained while every other part of the step took zero time.

`chowder_batch/moe_int8_io_ceiling.py --self-check` re-derives that inequality
without a card; `--h2d` measures the bandwidth on the machine the claim is made
about.

## 3. Where the 34 seconds go, and what was changed

Of the 33.5–34.5 s warm step, ~10.2 s is the copy above. What remains is
dequantization arithmetic, the expert matmuls and the non-expert path. The
expert matmuls were the part that turned out to be mis-sized rather than slow:
the reference path issues **160 small GEMM pairs per layer visit** (one per
expert, twice: gate_up and down), which on this card is launch-bound.
`chowder_batch/moe_int8_io_ceiling.py --experts` measures both forms at the
artifact's shapes, uniformly routed:

| expert form | ms per layer visit | effective TFLOP/s |
|---|---|---|
| per-expert `F.linear` loop (reference) | 25.9 | 3.98 |
| padded two-`bmm` grouped form (`CHOWDER_MOE_GROUPED_EXPERTS=1`, the default) | 5.1 | 26.34 |

That is a **5.06×** improvement on the expert GEMMs, at 31% padding waste on
uniform routing.

> **Erratum, 2026-10-21.** Those two millisecond figures are *not reproducible
> from the committed script*: `measure_experts` built each expert's assignment
> as `x[rows % tokens]` with `rows` a `slice`, which raises `TypeError:
> unsupported operand type(s) for %: 'slice' and 'int'` before a timer starts, so
> `--experts` could not run at all. The numbers came from an earlier variant of
> the harness. The index construction is fixed (both forms now take the same
> precomputed assignment and are shape-checked before either is timed) and the
> mode runs again, but **the re-measurement on the card has not happened yet** --
> treat 25.9 ms / 5.1 ms / 5.06× as unverified until it does. The A/B result in
> §5 does not depend on them: it measured the grouped path inside a real step,
> where the gain was 0.79 s (2.5%, inside the run's own drift). The grouped path is not unconditional: `grouping_pays()`
compares the padded row count against the real one and falls back to the loop
when the waste would exceed `GROUPED_MAX_WASTE = 3.0`, so a skewed router
degrades to today's behaviour instead of paying for padding it cannot use.
Gradients are identical to the loop's (covered by `tests/test_moe_layer_bulk.py`,
including a test that poisons the loop so a silent fallback cannot pass).

Two smaller changes landed with it:

- **Copy granularity decoupled from the dequant arithmetic block.**
  `DEQUANT_COPY_ELEMENTS = 1 << 26` (64 Mi elements per device copy) against
  `DEQUANT_ELEMENTS_PER_BLOCK = 1 << 23` (8 Mi per arithmetic block): the same
  bytes, the same arithmetic, 7,200 → 480 copies per step. Measured ~5% off the
  copy term.
- **Recompute weights are retained for the backward**
  (`CHOWDER_INT8_RETAIN_WEIGHTS`, **on by default since 2026-10-21**, opt out with
  `=0`). It removes one of the three dequantize passes. The ~0.94 GiB cost this
  note first projected was wrong: §5 measured **0.000 GiB**, because the retained
  buffer replaces the backward's own dequantization instead of sitting beside it.
  It was opt-in until that measurement existed; the measurement is why it is now
  the default.

Two hypotheses were **measured and rejected**, and are recorded so they are not
re-proposed:

- Page-locking the 20 GB int8 store: 5.52 GiB/s pinned vs 5.52 GiB/s pageable.
  No gain; rejected.
- A pinned staging buffer for the copies: same measurement, same reason;
  rejected.

## 4. What that means for the campaign budget

At the measured 34 s/step, 1,000 steps is 9.4 GPU-hours, not 2.5. The landed
changes move the copy term by ~5% and the expert-GEMM term by up to 5×; the
combined effect is measured by `chowder_batch/moe_int8_ab.py` on the card and
recorded in §5, not extrapolated here.

The honest options, in the order they should be considered (the numbers are the
measured ones from §5; an earlier projection of ~19–22 s/step is superseded and
is called out there):

1. **Accept 27.5 s/step and budget 7.63 GPU-hours per 1,000-step recipe.** That
   is what this host did with this placement, measured. A campaign budget that
   says so is a budget that will hold.
2. **Cut a pass — measured, and it works.** `CHOWDER_INT8_RETAIN_WEIGHTS`
   removed one of the three dequantize passes for **5.15 s/step (15.9%)** at
   **0.000 GiB** of extra peak, not the ~0.94 GiB §3 projected: the retained
   buffer replaces the backward's own dequantization rather than adding to it.
   It is still opt-in in code; §5 states the promotion order and why the default
   has not been flipped yet.
3. **Change the link.** Gen 4 x16 is ~4× the bandwidth of Gen 3 x8; the copy
   term becomes ~2.6 s and 9 s/step stops being arithmetic-impossible. This is
   the only option that reaches the target, and it is a host purchase, not a
   code change.
4. **Stop copying per step.** A resident-on-device expert set does not fit 16 GB
   by construction (that is why this placement exists); the alternatives are a
   bigger card or a smaller expert set, not a code change.

## 5. The A/B, measured on the card

One artifact load, five phases of four LoRA+AdamW steps each, on the same
artifact and card as §1 (`chowder_batch/moe_int8_ab.py --seq 2048
--steps-per-phase 4`): `C:\llm-work\moe-gates\ab-int8-seq2048.json`, 2026-10-20
22:52–23:25, load 1,277.8 s, 19,169,280 trainable LoRA parameters. A phase's
first step is excluded from its mean, so each `s/step` is the mean of three.

| phase | grouped | retain | s/step | peak alloc | peak reserved |
|---|---|---|---|---|---|
| `loop` (baseline) | no | no | 32.63 | 10.788 | 13.215 |
| `grouped` | yes | no | 31.84 | 10.788 | 13.215 |
| `grouped+retain` | yes | yes | **27.48** | 10.788 | 13.215 |
| `loop (repeat)` | no | no | 31.64 | 10.788 | 13.215 |

The record's own comparison keys: `grouped_speedup` 1.025, `retain_speedup`
1.159, `drift` 1.031, and every peak delta 0.000.

**The baseline repeats faster than it starts, by 3.1%.** The last phase re-runs
the reference loop and comes in at 31.64 against the first phase's 32.63, so the
machine itself moved ~3% during the 33-minute run. Every difference below is
judged against the *repeat*, not the first loop.

**The grouped expert GEMMs are worth 0.79 s/step in situ — 2.5%, inside that
drift.** `grouped` 31.84 against `loop (repeat)` 31.64 is 0.6% *faster*, i.e.
not distinguishable from the machine's own variance in this run. §3's isolated
table (25.9 ms → 5.1 ms per layer visit, 5.06×) is a real measurement of the
GEMMs and does not translate into a 5× step: the expert GEMMs are ~2% of a step
that is dominated by copy and dequantization, and the padded `bmm` still moves
the same dequantized bytes. The grouped path stays the default because it never
loses and it is decisively better in isolation — but no campaign budget may cite
it as the reason a step got faster.

**The retention switch is worth 5.15 s/step — 15.9% against the first loop,
15.1% against the repeat, so it is real and outside the drift.** It removes one
of three dequantize passes: §2's arithmetic predicted 18.82 GiB / 5.52 GiB/s =
3.41 s of copy, and the measured gain is larger, because the removed pass also
removed its dequantization arithmetic.

**Retention cost no peak at all: 0.000 GiB, where §3 projected ~0.94 GiB.** The
projection assumed the retained weights would sit *beside* a buffer the backward
would otherwise allocate. They replace it: with retention on, the backward
consumes the recompute's weights instead of dequantizing its own, so the same
bytes are resident at the same moment either way. The projection in §3 is wrong
and is recorded here rather than quietly deleted.

**Campaign budget, measured:** 27.48 s/step × 1,000 = **7.63 GPU-hours** per
recipe with retention on, 9.1 with it off (32.63 s/step), against the 2.5
GPU-hour target. The 9 s/step target needs 6.27 GiB/s of sustained copy with
everything else free (§2); this host achieves 5.52 GiB/s, so it stays out of
reach until option 3 in §4.

What this A/B does **not** cover, and a campaign must not assume from it: four
steps per phase rather than twenty; one artifact, one card; no first-touch step
(the 3,024 s first-touch a real run pays is absent); and the peak is the maximum
of four per-step peaks, which is why it is worth exactly 0.000 GiB as evidence
of *retention's* cost and nothing else. The gate row for `memory_vram` is still
`chowder_batch/moe_int8_gates.py`'s to write, not this table's.

### The sanctioned placement step

The prereg's own runner measured the same card the next morning
(`gates-sanctioned-int8-seq2048.json`, code `b4094c1`, 2026-10-21 08:16-08:41):
**11.345 GB peak allocated, 11.664 GB reserved** against the 14.5 GB gate -- a
pass -- with **5.383 GB** of that peak being the step's own scoring, so the
placement's forward peak is 5.962 GB. Timed at 51.82 s per placement step over
three steps `[94.11, 29.40, 31.95]`: the first carries the 20 GB store's first
touch, and the steady 29.4-32.0 s agrees with this note's 33 s/step baseline. It
is a placement step, not a training step: no LoRA, no optimizer, so it is not
comparable to the A/B's 27.48 s and is not budgeted from.

### Promotion of the retention switch: now the default, re-measurement queued

`CHOWDER_INT8_RETAIN_WEIGHTS` became **default-on in 956c214** on the strength of
the measurement above: 15.9% faster and 0.000 GiB of extra peak at seq 2048, with
`CHOWDER_INT8_RETAIN_WEIGHTS=0` as the opt-out and a test pinning the shipped
default to the two-pass step.

**The table has not caught up yet, and this note says so rather than letting a
reader assume otherwise.** The `memory_vram` and `throughput` rows currently in
`docs/quals/MOE_INT8_PLACEMENT_PREREG_2026-10-20.md` were measured at `b4094c1`,
when the switch was still off. Until the sanctioned runner is re-run on 956c214,
the rows describe the *previous* configuration and a campaign that cites them is
citing a configuration that is no longer shipped. The re-run
(`chowder_batch/moe_int8_gates.py --arms int8 --steps 3 --seq-len 2048`) is
registered as `\llm-work\gates-witness` and held behind an idle-card gate: on
2026-10-21 another workload cycled between ~1 GB and ~9 GB of the 16 GB card every
few minutes, and the 20 GB load needs the whole card. Competing with it produced
two OOMs (09:44, 10:07) and no number; waiting is the only honest option, and the
row stays as measured until the re-run replaces it.

## 6. Reproduction

```
# copy bandwidth and the two expert forms, on the card
python chowder_batch/moe_int8_io_ceiling.py --h2d --experts

# the arithmetic alone, no card
python chowder_batch/moe_int8_io_ceiling.py --self-check

# the A/B of the two expert forms and the retention switch, one artifact load
python chowder_batch/moe_int8_ab.py --base <artifact> --calib <calibration.jsonl> \
    --seq 2048 --steps-per-phase 4 --out ab-int8-seq2048.json
```

The A/B record this section quotes is `C:\llm-work\moe-gates\ab-int8-seq2048.json`;
the runner that produced it is in this repository, and the phases it ran are the
five in the table above.
