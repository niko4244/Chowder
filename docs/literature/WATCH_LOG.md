# Literature watch log

The curated record produced by the watch described in `README.md`. One dated
section per pass. A pass with nothing relevant says so. Every entry cites its
`arXiv:<id>`, names the mechanism, and maps to a concrete Chowder surface -- an
existing intervention family, a named backend, or `candidate -- unregistered`.
Numbers quoted here are the *papers'* numbers, on their models and protocols;
none of them is a Chowder measurement.

---

## 2026-10-08 -- the looped-sequence wave, plus compression and judge-integrity findings

**Method.** Read `arXiv:2610.07940` directly; queried the arXiv API for six
Chowder surfaces (training methods, compression, PEFT/adapters, distillation,
efficient inference, evaluation integrity) over the recent window via
`watch.py`; read abstracts and, where a claim was load-bearing, the body.
Registry cross-checked against `src/chowder/growth/interventions.py` (13
families) so every "maps to" names a real surface.

### Deep read -- `arXiv:2610.07940`, *Hybrid Latent Attention for Looped Language Models* (6 Oct 2026)

**What it establishes.** Looped language models apply the same layer stack T
times per token, deepening the model with no new parameters, but multiplying the
KV cache by T. HLA keeps *exact* keys/values inside a sliding window of W recent
tokens and stores each older token as a **compact latent the query reads
directly, without reconstructing K/V**. It is **uptrained**: the pretrained
looped weights stay frozen and only the added projection parameters are trained
to reproduce the original attention. Reported on Ouro looped models (T=4, 1.4B
and 2.6B): cache shrinks 10.7x/token, 4.0-8.8x more concurrent sequences per
GPU, decode throughput 2.5x at 1K and up to 7.4x at 16K, retaining >97% of the
original accuracy on math/knowledge/reasoning and 96-100% on long-context
retrieval; after SFT it is on par with the fine-tuned original on
competition-level math.

| Fact | Evidence | Consequence for Chowder |
| --- | --- | --- |
| The premise is *looped* attention: the cache is inflated by T loops | Paper's framing | Chowder's parents are dense (Qwen-class). The multiplication HLA compresses **does not exist** in a dense parent, so **no number here transfers**. Class: *watch*. |
| The intervention shape is "freeze the backbone, uptrain added params to reproduce the original function" | Paper's uptraining setup | This is exactly the shape of a Chowder compression family: an added-parameter mechanism gated on a **capability-retention floor**. The *interface* transfers even where the *premise* does not. |
| The evaluation protocol is cache-shrink x concurrency x throughput at fixed capability retention | Paper's tables | A reusable measurement schema for any attention-cache compression family: report shrink, concurrency, throughput at 1K/16K, and a retained-accuracy floor -- never throughput alone. |
| Latent-for-old-tokens keeps recent tokens exact | Paper's method | The "exact window + lossy history" split is the transferable idea: bounded exactness with a measured loss only on the tail. |

**Verdict.** Recorded as *watch*. HLA is the headline of a coherent
looped-architecture research wave (below). It informs a **candidate family
shape** -- `compression.attention-latent`, uptrained cache compression under a
retention floor -- but it is not runnable against any current Chowder parent,
and pretending otherwise would be the borrowed-number failure the rules forbid.

### The looped-sequence cluster (all premise on a looped parent -- *watch*)

| Paper | Claim | Surface |
| --- | --- | --- |
| `arXiv:2610.10381` ResidualQuant (7 Oct) | KV states across loops are highly similar, so represent all-but-the-final loop as low-precision **residuals** to the final loop; with least-squares scaling, rotations and loop-wise mixed precision, reaches INT2, ~80.7% KV-storage reduction, up to +13.0% accuracy over rotation-only baselines, 2.73x decode / 4.15x peak throughput on an RTX 5090 | same *watch* class; second data point in the attention-cache compression family |
| `arXiv:2610.09827` Dual-QK (7 Oct) | paired non-orthogonal query/key transforms for INT2 KV, 40% query-channel pruning; 6.8x KV compression at 128K context, ~8.3x less KV read volume, up to 3.75x decode throughput (SGLang) | same *watch* class |
| `arXiv:2610.00673` Closing the Loop (Oct) | *practical training recipes* for looped LMs | *watch*: relevant only if a looped architecture becomes a parent |

Chowder has no looped parent today, so this whole cluster stays *watch*. The one
transferable asset is the **protocol** (shrink x concurrency x throughput at a
retained-accuracy floor), which belongs in the eval dimensions of any future
cache-compression family.

### Compression -- direct candidates on existing families

| Paper | Mechanism | Maps to | Transfer class |
| --- | --- | --- | --- |
| `arXiv:2610.10385` OrBIT (7 Oct) | discover the *coding geometry* of an embedding codec (learned local charts from orbit dynamics) instead of fixing low-rank/codebook geometry; 37.9x on GPT-2 and >23x on 7B tables vs 16-bit, competitive rate-distortion vs quantization and low-rank baselines | **`compression.low-rank-vocab`** -- same object (embedding tables); the family's basis is the measured *flat-spectrum -> unacceptable degradation* result, and OrBIT is a different geometry on that exact object | requires codec training; candidate **variant/reopen** of an existing family |
| `arXiv:2610.09969` TR-PTQ (7 Oct) | attributes PTQ loss to **specific structural sources** (learned LayerNorm scale params, compounded GELU approximations) and shows SoftMax is robust; integer-only log-domain primitives give <1.5% absolute degradation | **`compression.ptq`** -- this is the diagnostic the family lacks: its basis measured a margin shift of +0.0076 *with* accuracy collapsing 0.3 -> 0.0 | measurement (the error-source diagnosis transfers immediately) |
| `arXiv:2610.09877` Layerwise Error Attribution (7 Oct) | separable per-layer error score separating propagated from local error; bit allocation with no external solver, **robust to corrupted calibration**, 28x-2570x faster allocation | **`compression.ptq`** -- a better allocator than the family's current margin-based one | measurement/recipe |
| `arXiv:2610.00717` Tucker attention compression (Oct) | sequential functional structured Tucker decomposition of the attention maps | `candidate -- unregistered` (`compression.attention-tucker` shape) | requires a conversion run |
| `arXiv:2610.00694` Divergence -> decision flips (Oct) | how compression turns divergence into **decision flips** | **`compression.*` eval dimensions** -- a metric for compression-induced decision instability | measurement |

### Training methods

| Paper | Mechanism | Maps to | Transfer class |
| --- | --- | --- | --- |
| `arXiv:2610.10536` ExpDis (7 Oct) | **decouple exploration from optimization** in RLVR: train explorer policies with a novelty bonus, filter trajectories for correctness/quality, **distill into a separate student trained without the bonus**, alternate for several rounds | **`training.teacher-distillation`** and the generation-to-generation survivor transition -- the "explore in a throwaway policy, keep gains in the survivor" shape is directly reusable | requires retraining |
| `arXiv:2610.10426` CoTrace (7 Oct) | harness-model **co-evolution** with **component-wise promotion**; a harness-aware data recipe that routes trajectories by **provenance matching** and conditions policy training on **verified rollouts matched to the adopted runtime**; harness-matched corpus beats larger pooled corpora at lower compute | **`runtime.harness-repair`** / **`runtime.harness-evolution`** -- independent external validation of Chowder's declared-runtime-matching rule, plus a concrete recipe for trajectory routing | measurement + requires retraining |
| `arXiv:2610.10411` EDR (7 Oct) | train parallel/semi-AR **speculative drafters** by directly minimizing *expected decoding rounds* (a Markov-reward formulation) and -- key for us -- an **exact offline evaluator for round counts** enabling **paired drafter comparison on shared target rollouts without running speculative decoding** | **`inference.speculative`** -- the offline paired evaluator is a measurement transfer into the family's eval dimensions | measurement + requires retraining |
| `arXiv:2610.10332` TPD (7 Oct) | **task-progress distillation**: pair each demonstrated action with a short stage label; the student selects by jointly scoring admissible stage-action pairs | **`training.teacher-distillation`** | requires retraining |
| `arXiv:2610.10349` AutoAdapt (7 Oct) | automatically discover latent domains, train **per-domain LoRA adapters independently** with **parameter-free routing**; parity with one all-domain adapter, no full-model retraining | **`training.adapter-continuation`** (PRODUCTION) -- a modular-specialization recipe that avoids domain interference by construction | requires retraining |
| `arXiv:2610.10347` Dataset Pruning from First Principles (7 Oct) | label-free, training-free subset selection via variance minimization on a polytope; the same framework increases **within-batch diversity** to cut SGD variance at fixed batch size | **`training.sft-curriculum`** / data selection | measurement + recipe |
| `arXiv:2610.00650` Self-Evolving Coding Rules (Oct) | agents evolve their own coding rules | **`runtime.harness-evolution`** | watch |

### Evaluation integrity -- the judge and settlement path

| Paper | Claim | Surface |
| --- | --- | --- |
| `arXiv:2610.00054` *The First Token Is Not the Verdict* (Oct) | **hidden costs of reading LLM judges without generating** -- judge verdicts read off non-generating passes differ from generated ones | **the judge/settlement path** (settlement audit, #211): a way a judge score is an artifact of *how* it was read, which the promotion gate must refuse. Strongest integrity catch of this pass. |
| `arXiv:2610.00164` Metric-construction coupling (Oct) | **the choice of metric construction inflates** measured recovery | **measurement provenance** -- Chowder's provenance tests target exactly this failure class |
| `arXiv:2610.00202` Causal auditing of synthetic RLVR corpora (Oct) | distinguishing a data artifact from a shortcut by causal auditing | **`training.teacher-distillation`** data provenance |
| `arXiv:2610.00779` Group-level signals for synthetic data curation (Oct) | per-example curation signals are insufficient; **group-level** signals are required | **`training.sft-curriculum`** data curation |
| `arXiv:2610.00320` Refusal localizes, the damage relocates (Oct) | few-sample fine-tuning moves safety damage rather than removing it | **retention gate / risks** of any adapter-continuation family |

### Candidate families this pass surfaces (unregistered -- proposals to consider)

These are *shapes*, not registered mechanisms. Each would need a mechanism, a
smoke row that actually runs, and the runnability gate before the loop could
propose it (README rule 7).

1. `compression.attention-cache` -- uptrained, frozen-backbone attention-cache
   compression under a retained-accuracy floor. Interface from `2610.07940`,
   protocol from `2610.10381` / `2610.09827`. Blocked on a non-loop headroom
   premise; the *family shape* is ready.
2. `compression.ptq` -- **not** a new family but a concrete upgrade path: adopt
   the TR-PTQ error-source diagnostic (`2610.09969`) and the layerwise
   allocator (`2610.09877`) to attack the recorded 0.3 -> 0.0 collapse.
3. `compression.low-rank-vocab` reopen -- a learned-codec geometry (`2610.10385`)
   as a named new hypothesis against the family's flat-spectrum rejection.
4. `runtime.harness-evolution` -- borrow CoTrace's provenance-matched trajectory
   routing (`2610.10426`) and EDR's offline paired evaluator (`2610.10411`).

### What this pass does **not** do

- It registers no family, changes no maturity label, and promotes nothing.
- It claims no improvement for Chowder; every number above is the paper's.
- It records the looped-sequence cluster as *watch* rather than chasing it: the
  premise (a looped parent) is not true here, and the rules forbid borrowing a
  gain measured under a premise Chowder does not have.
- It does not touch the frozen evaluation protocol or the evidence store.

**Next actions.** (a) Read `2610.00054` in full against the judge/settlement
code -- it is small and directly load-bearing; (b) if a `compression.ptq`
upgrade is wanted, read `2610.09969` + `2610.09877` and propose it against the
family's recorded negative basis; (c) keep the looped cluster on *watch* and
re-check only if a looped architecture enters a campaign.
