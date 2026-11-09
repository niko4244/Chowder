# Literature watch log

The record produced by the watch described in `README.md`. Two kinds of dated
section live here, and they are not the same thing:

- **Curated sections** -- written by whoever read the papers. Every entry cites
  its `arXiv:<id>`, names the mechanism, and maps to a concrete Chowder surface
  (an existing intervention family, a named backend, or
  `candidate -- unregistered`). Numbers quoted are the *papers'* numbers, on
  their models and protocols; none of them is a Chowder measurement.
- **Automated pool drops** (`## Automated pool drop -- <date>`) -- appended by
  the scheduled run in `.github/workflows/literature-watch.yml`. A drop is a
  *triage list*: unvetted, unread, and carrying only ids, dates, and the surface
  term that matched -- deliberately no abstract text and no number, so it cannot
  carry a claim into this log. A drop is not a finding and registers nothing
  (rule 7). When one of its hits is worth keeping, promote it into a curated
  section; delete the drop if it adds nothing. Drops are transient by design:
  the scheduled run prunes those older than 30 days, and that horizon is longer
  than its fetch window, so a prune can never bring a paper back around.

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

---

## 2026-10-08, second pass -- the full read of `2610.00054`, and the correction it forced

**Method.** Read the paper in full (abstract page and the HTML full text), then
read the two artifacts it would bear on: `docs/gen2/judge_gen2.py` (the frozen
Gen-2 judge) and `src/chowder/evaluators/scoring.py` (production's scoring
owner). Both readouts below were then *run* on both implementations, verbatim,
rather than compared by eye.

### The correction

The first pass above recorded `2610.00054` as "directly load-bearing for the
judge/settlement path". The full read shows that is **too strong, and the entry
stands corrected here rather than quietly dropped** (rule 6):

- the paper's mechanism is a verdict read from the **first generated token's
  logits**. The Gen-2 judge does not do that -- it generates
  (`PROTECTED_DECODING`: greedy, 512 tokens) and reads the post-reasoning
  surface. The logits in this repo are router gate logits used as a tamper
  fingerprint (`backends/router_healing_eval_worker.py`), and
  `router_healing_run.judge_router_healing` is a pass-through to
  `gate.evaluate_candidate`;
- "settlement" in this repo means the run's decision and accounting artifacts
  (T13/T21/T23), not a model readout. The paper touches nothing there.

So the paper's numbers transfer nowhere, and its position-bias metric has no
target in Chowder. **Class: not-applicable** for the judge and settlement paths
as readouts.

### What does transfer, and where it landed

The transferable part is the *class* -- **the readout is not the conclusion** --
and the disclosure discipline the paper asks for: say which readout produced the
figure, and measure how often the readout is the conclusion instead of assuming
it. That lands on one row, T5 in the frozen judge, which computes
`expected.lower() in _answer_surface(completion).lower()`: **presence**, not
correctness. The prereg is honest about the quantity ("expected string
*present* after reasoning", "within-reasoning answer presence"); the judge's row
label says "answer correctness" and its detail names no readout.

| Finding (first-party, this repo) | Evidence |
| --- | --- |
| The judge's presence readout accepts a **mention** as the answer | `Canberra is not the capital of Australia; Sydney is.` scores correct against `Canberra`, while production's `final_answer` + `normalize` scores it 0 |
| It accepts text production declares to be **no answer at all** | on a completion that opens reasoning and never closes it, `_answer_surface` returns the reasoning and presence scores `Canberra` correct, while `final_answer` returns `''` ("there is no answer yet"); T10 tolerates that shape at up to 25% |
| The disagreement is systematic, not incidental | 13 constructed cases, 13 disagreements, every one in the same direction (presence correct, production wrong) |
| The collision set of the digit expectations is large | 271 of the 999 integers in [1, 999] contain `2` (and `4`, `5`, `7` each), so a wrong answer drawn from that set scores correct for the four single-digit prompts |

**What these numbers are.** They are ours, measured here by running both
implementations -- but the completions were *constructed* to be mentions rather
than conclusions, so they measure the **reachability** of the collision set, not
its frequency. There are no recorded instrument completions to measure the
frequency on (`evidence/` holds zero `(expected, completion)` pairs), and the
paper's rates (0-5.5% at the token level; 89.7% against 47.5%) are its own,
measured on its judges, prompts and formats. The frequency is exactly what the
proposal below exists to make measurable on the next run.

**Outcome.** `docs/gen2/JUDGE_AMENDMENT_PROPOSAL_T24.md` -- a proposed amendment
17 to the frozen judge (gate T24): disclose the readout, measure where presence
and production's declared scoring disagree, gate only the unsafe direction, and
move no existing threshold. Proposed, **not applied**: amending a frozen
instrument is the repo owner's call, and the no-visible-candidate-results rule
was verified to hold before proposing it (zero candidate artifacts under
`evidence/`). Next action (a) is closed.

### What this pass does **not** do

- It registers no family, changes no maturity label, promotes nothing, and
  changes no line of `judge_gen2.py`.
- It claims no rate for the collision set: the constructed cases demonstrate
  reachability, and the paper's rates stay the paper's.
- It does not claim position bias, or first-token readouts, exist anywhere in
  Chowder -- the measured gap is between two readouts of the same recorded
  completions, which is a different and narrower thing.


## Automated pool drop -- 2026-11-09

> **Unvetted, machine-appended -- nothing here is registered.** These are
> arXiv search hits that named a surface mechanism in a watched primary
> category, listed for triage. No paper here has been read, no result is
> quoted, and no intervention family, maturity label, or gate is touched
> (`README.md`, rule 7). Curate anything real into a dated curated section
> under the rules, or delete this drop if it adds nothing.

- window: last 14 day(s) | surfaced: 82 | new: 74 | already in the log: 8 | off-scope: 12
- source: `.github/workflows/literature-watch.yml` (`watch.py --append-log`: the primary-category gate plus the surface-term screen)

### training-methods

- **Verifier Noise Is a Temperature: Testing Tilt Predictions in Finite-Budget LLM-Guided Search** -- `arXiv:2611.09699v1` -- 2026-11-06 -- cs.LG -- matched: optimizer
  - https://arxiv.org/abs/2611.09699
- **Steerable Multi-Objective Group-Relative Preference Optimization with Jointly Trained Low-Rank Adapters** -- `arXiv:2611.09634v1` -- 2026-11-06 -- cs.CL -- matched: post-training, reinforcement learning, preference optimization
  - https://arxiv.org/abs/2611.09634
- **Solver-Free Decision-Focused Learning via Neural Optimization Layers for Edge Selection** -- `arXiv:2611.09560v1` -- 2026-11-06 -- cs.LG -- matched: pretraining
  - https://arxiv.org/abs/2611.09560
- **e-means: outperforming k-means (in one-tenth the epochs)** -- `arXiv:2611.09501v1` -- 2026-11-06 -- cs.LG -- matched: data curation, scaling law
  - https://arxiv.org/abs/2611.09501
- **Not All Entropy Is Equal: Reward-Conditioned Entropy Allocation in Reinforcement Learning with Verifiable Rewards** -- `arXiv:2611.09448v1` -- 2026-11-06 -- cs.LG -- matched: rlvr, reinforcement learning
  - https://arxiv.org/abs/2611.09448
- **Safety-Focused Reinforcement Learning Using Risk-Sensitive Coupled Control** -- `arXiv:2611.09290v1` -- 2026-11-06 -- cs.LG -- matched: reinforcement learning
  - https://arxiv.org/abs/2611.09290
- **Rethinking MoE Load Balancing as Feedback Control** -- `arXiv:2611.09252v1` -- 2026-11-06 -- cs.LG -- matched: curriculum, pretraining, data mixture
  - https://arxiv.org/abs/2611.09252

### compression

- **FlowMoP: Stochastic Multi-Person Motion Prediction** -- `arXiv:2611.09882v1` -- 2026-11-06 -- cs.CV -- matched: compress
  - https://arxiv.org/abs/2611.09882
- **Breaking the Privacy Promise: Gradient Inversion Attacks on Federated Low-Rank Adapted Vision Transformers** -- `arXiv:2611.09652v1` -- 2026-11-06 -- cs.CV -- matched: low-rank
  - https://arxiv.org/abs/2611.09652
- **Transformer-FSQ: Discrete Latents for Extreme KV-Cache Compression** -- `arXiv:2611.09598v1` -- 2026-11-06 -- cs.LG -- matched: quantis, kv-cache, compress
  - https://arxiv.org/abs/2611.09598
- **Query-Guided RoPE Modulation for Lost-in-the-Middle Mitigation in Long-Context LLMs** -- `arXiv:2611.09554v1` -- 2026-11-06 -- cs.CL -- matched: compress
  - https://arxiv.org/abs/2611.09554
- **From LoRA to Language: Recovering Skill Documents from Adapters** -- `arXiv:2611.09454v1` -- 2026-11-06 -- cs.LG -- matched: low-rank
  - https://arxiv.org/abs/2611.09454
- **MINE: Entropy-coded Memory-efficient Inference for Large Language Models** -- `arXiv:2611.09424v1` -- 2026-11-06 -- cs.LG -- matched: quantiz, compress
  - https://arxiv.org/abs/2611.09424
- **Robust Deep Prototype Classifiers via Lipschitz Training** -- `arXiv:2611.09327v1` -- 2026-11-06 -- cs.LG -- matched: quantiz
  - https://arxiv.org/abs/2611.09327
- **Fused-Softmax for efficient quantized attention** -- `arXiv:2611.09225v1` -- 2026-11-06 -- cs.AI -- matched: quantiz
  - https://arxiv.org/abs/2611.09225
- **RoleWalk: training-free structural embeddings for directed graphs that transfer across graphs** -- `arXiv:2611.09189v1` -- 2026-11-06 -- cs.LG -- matched: prun
  - https://arxiv.org/abs/2611.09189
- **ResQAdam: Residual-Preserving Adaptive Optimization for Memory-Efficient Quantization Aware Training** -- `arXiv:2611.08924v1` -- 2026-11-06 -- cs.LG -- matched: quantiz, low-rank
  - https://arxiv.org/abs/2611.08924
- **Spectral and Subspace Dynamics of LoRA in Supervised Fine-Tuning and Reinforcement Learning** -- `arXiv:2611.08414v1` -- 2026-11-05 -- cs.LG -- matched: low-rank
  - https://arxiv.org/abs/2611.08414
- **Muir: Linear Attention with Sensitivity-Allocated Gating and Matched-Filter Readout** -- `arXiv:2611.08376v1` -- 2026-11-05 -- cs.LG -- matched: low-rank, compress
  - https://arxiv.org/abs/2611.08376
- **StrataFlux: Memory-Hierarchy-Native Low-Bit LLM Inference** -- `arXiv:2611.08363v1` -- 2026-11-05 -- cs.LG -- matched: quantiz, kv-cache
  - https://arxiv.org/abs/2611.08363
- **AutoCompress: Automated Hardware-Guided Model Compression** -- `arXiv:2611.08290v1` -- 2026-11-05 -- cs.LG -- matched: prun, compress
  - https://arxiv.org/abs/2611.08290

### peft-adapters

- **RuACT: Recovering Inactive Rubrics as Adaptive Critics and Teachers** -- `arXiv:2611.09597v1` -- 2026-11-06 -- cs.CL -- matched: lora
  - https://arxiv.org/abs/2611.09597
- **Turkish Multimodal Video Fact-Checking: A New Dataset and a Comparison of Zero-Shot Agentic and Fine-Tuned Systems** -- `arXiv:2611.08651v1` -- 2026-11-05 -- cs.CV -- matched: lora
  - https://arxiv.org/abs/2611.08651
- **Beyond K/V Caches: Cross-Model State Handoff in Hybrid Linear-Attention LLMs** -- `arXiv:2611.08133v1` -- 2026-11-05 -- cs.CL -- matched: adapter
  - https://arxiv.org/abs/2611.08133
- **Axis-LoRA: Probing and Adapting Embedding Models for Personal-Memory Retrieval** -- `arXiv:2611.07820v1` -- 2026-11-05 -- cs.CL -- matched: lora, adapter
  - https://arxiv.org/abs/2611.07820
- **Automated Video-to-QA Pipeline: End-to-End Question Generation and Verification Using Large Language Models** -- `arXiv:2611.07755v1` -- 2026-11-05 -- cs.CL -- matched: lora
  - https://arxiv.org/abs/2611.07755

### distillation

- **Mosaic:montage-agnostic self-distilled graph learning for subject-independent ieeg seizure detection** -- `arXiv:2611.09496v1` -- 2026-11-06 -- cs.LG -- matched: distillation
  - https://arxiv.org/abs/2611.09496
- **COMBIO: Automated Discovery of a Compact Composite Biomarker from Routine Laboratory Panels** -- `arXiv:2611.09102v1` -- 2026-11-06 -- cs.LG -- matched: distil
  - https://arxiv.org/abs/2611.09102
- **WearableBridge: Task-Guided Latent Transport for Free-Living Activity Recognition** -- `arXiv:2611.09067v1` -- 2026-11-06 -- cs.LG -- matched: distillation
  - https://arxiv.org/abs/2611.09067
- **Failure Profiling and Reachable Trajectory Selection for Reasoning Distillation** -- `arXiv:2611.09042v1` -- 2026-11-06 -- cs.CL -- matched: distillation, teacher model
  - https://arxiv.org/abs/2611.09042
- **Knowledge Distillation from Neural Networks to Differentiable Logic Networks** -- `arXiv:2611.09024v1` -- 2026-11-06 -- cs.LG -- matched: distillation
  - https://arxiv.org/abs/2611.09024
- **MotionBench-XAI: Benchmarking What Shapes Shapley Explanations in Spatiotemporal Machine Learning** -- `arXiv:2611.08797v1` -- 2026-11-05 -- cs.LG -- matched: distil
  - https://arxiv.org/abs/2611.08797
- **MEDALLION: Manifold Embedding Distillation and Attributions for Low-dimensional Latent InterpretatiON** -- `arXiv:2611.08767v1` -- 2026-11-05 -- cs.LG -- matched: distillation
  - https://arxiv.org/abs/2611.08767
- **Unpacking the Compute Efficiency of Strong-to-Weak OPD against GRPO** -- `arXiv:2611.08453v1` -- 2026-11-05 -- cs.LG -- matched: distillation
  - https://arxiv.org/abs/2611.08453
- **Discovering Update Rules for Opponent Shaping in General-Sum Games** -- `arXiv:2611.08335v2` -- 2026-11-05 -- cs.LG -- matched: self-play
  - https://arxiv.org/abs/2611.08335
- **Learning Dynamics of On-Policy Distillation** -- `arXiv:2611.08016v1` -- 2026-11-05 -- cs.LG -- matched: distillation, teacher-student
  - https://arxiv.org/abs/2611.08016
- **CoRD: Coupled Rollouts Fill the Blind Spot of On-Policy Distillation** -- `arXiv:2611.07959v1` -- 2026-11-05 -- cs.LG -- matched: distillation
  - https://arxiv.org/abs/2611.07959
- **Optimal Learning in Games with Arbitrary Delays: Best-of-Both-Worlds for Self-Play and Adversarial Opponents** -- `arXiv:2611.07773v1` -- 2026-11-05 -- cs.LG -- matched: self-play
  - https://arxiv.org/abs/2611.07773
- **Reinforcing What the Student Has Learned: Teacher-Guided On-Policy Distillation** -- `arXiv:2611.07722v2` -- 2026-11-05 -- cs.LG -- matched: distillation
  - https://arxiv.org/abs/2611.07722
- **IMUDINO: Self-distilled Foundation Model for Inertial Sensing** -- `arXiv:2611.07437v1` -- 2026-11-05 -- cs.LG -- matched: distil
  - https://arxiv.org/abs/2611.07437
- **Test-Time Self-Play: Scaling without a Verifier** -- `arXiv:2611.07367v1` -- 2026-11-05 -- cs.LG -- matched: self-play
  - https://arxiv.org/abs/2611.07367
- **Using Transformers for Teaching and Learning via Symbolic Distillation of Neural A-life Agents** -- `arXiv:2611.07254v1` -- 2026-11-05 -- cs.LG -- matched: distillation
  - https://arxiv.org/abs/2611.07254
- **$D^2$-OPD: Separating Direction and Distance in Extrapolative On-Policy Distillation** -- `arXiv:2611.07241v1` -- 2026-11-04 -- cs.LG -- matched: distillation
  - https://arxiv.org/abs/2611.07241
- **Recursive Sparse Attention: Million-Token Length Extrapolation at Logarithmic Cost** -- `arXiv:2611.07232v1` -- 2026-11-04 -- cs.CL -- matched: distil
  - https://arxiv.org/abs/2611.07232
- **Fingerprinting the Dispatchers behind Distributed Model Extraction** -- `arXiv:2611.06661v1` -- 2026-11-04 -- cs.LG -- matched: distillation
  - https://arxiv.org/abs/2611.06661

### efficient-inference

- **Intent-Aware Caching for Efficient LLM Serving** -- `arXiv:2611.09715v1` -- 2026-11-06 -- cs.AI -- matched: latency, serving
  - https://arxiv.org/abs/2611.09715
- **TimeOmni-Live: Continuous & Live Time Series Understanding with Long-Term Memory** -- `arXiv:2611.09502v1` -- 2026-11-06 -- cs.LG -- matched: latency
  - https://arxiv.org/abs/2611.09502
- **RPEG: Training Reasoning Language Models Scalably and Cost-Effectively** -- `arXiv:2611.09207v1` -- 2026-11-06 -- cs.DC -- matched: latency
  - https://arxiv.org/abs/2611.09207
- **Large Language Model Deployment on Resource-Constrained Edge Devices: A Practitioners Survey** -- `arXiv:2611.09075v1` -- 2026-11-06 -- cs.LG -- matched: latency
  - https://arxiv.org/abs/2611.09075
- **QRAE: Query-Relevant Attention Entropy for Context Compression in DeepSearch Agents** -- `arXiv:2611.07976v1` -- 2026-11-05 -- cs.LG -- matched: latency
  - https://arxiv.org/abs/2611.07976
- **Equivariant Deep Learning in Direct-Product Space for Electronic-Structure Hamiltonian Prediction** -- `arXiv:2611.07695v1` -- 2026-11-05 -- cs.LG -- matched: inference efficiency
  - https://arxiv.org/abs/2611.07695
- **ACSeL: Adaptive Configuration Search for Hardware-Constrained LLM Inference** -- `arXiv:2611.07570v1` -- 2026-11-05 -- cs.LG -- matched: throughput, latency, serving
  - https://arxiv.org/abs/2611.07570
- **Latency-Aware Adaptive Best-of-$N$** -- `arXiv:2611.07453v1` -- 2026-11-05 -- cs.CL -- matched: latency, serving
  - https://arxiv.org/abs/2611.07453
- **FastAFD: When Does Attention-FFN Disaggregation Pay on Rack-Scale NVLink?** -- `arXiv:2611.07149v1` -- 2026-11-04 -- cs.DC -- matched: speculative decoding, throughput, serving
  - https://arxiv.org/abs/2611.07149
- **LowRAM: Low-Rank Keys and Amortised Values for Composable KV Cache Compression at Test-Time** -- `arXiv:2611.07025v1` -- 2026-11-04 -- cs.LG -- matched: throughput, memory footprint
  - https://arxiv.org/abs/2611.07025

### evaluation-integrity

- **Khmer PFGen-Bench: Building a Judge-Free Benchmark for Open-Ended Generation in a Low-Resource Language** -- `arXiv:2611.09222v1` -- 2026-11-06 -- cs.CL -- matched: judge
  - https://arxiv.org/abs/2611.09222
- **Copies Are Not Confirmations: How News Duplication Fools LLM Judges and Continual Learners** -- `arXiv:2611.09203v1` -- 2026-11-06 -- cs.LG -- matched: llm judge
  - https://arxiv.org/abs/2611.09203
- **AlienBench: A Benchmark of Structural Novelty in Language Model Outputs** -- `arXiv:2611.09094v1` -- 2026-11-06 -- cs.CL -- matched: llm judge
  - https://arxiv.org/abs/2611.09094
- **Terminal-style Mathematical Modeling Bench: The First Agentic and Verifiable Benchmark for Mathematical Modeling** -- `arXiv:2611.08237v1` -- 2026-11-05 -- cs.CL -- matched: llm judge
  - https://arxiv.org/abs/2611.08237
- **Decodability Is Not Validity: Evaluating Nuisance Sensitivity in Language Model Concept Probes** -- `arXiv:2611.07805v1` -- 2026-11-05 -- cs.CL -- matched: llm judge
  - https://arxiv.org/abs/2611.07805
- **Compositional Unalignment: Auditing Safety in Large Vision-Language Models** -- `arXiv:2611.07216v1` -- 2026-11-04 -- cs.LG -- matched: llm judge
  - https://arxiv.org/abs/2611.07216
- **Adapting LLM Judges to Observed Human Annotation Policy through Soft Prompt Tuning** -- `arXiv:2611.07126v1` -- 2026-11-04 -- cs.CL -- matched: llm judge, annotat
  - https://arxiv.org/abs/2611.07126
- **Measuring Risk, Not Harm: Detecting Psychological Harm Risk Factors in LLM Conversations** -- `arXiv:2611.06806v1` -- 2026-11-04 -- cs.CL -- matched: judge, annotat
  - https://arxiv.org/abs/2611.06806
- **Mass Producing Training Data with Surprising Side Effects** -- `arXiv:2611.06324v1` -- 2026-11-04 -- cs.CL -- matched: llm judge
  - https://arxiv.org/abs/2611.06324
- **The Harness Is the Model: Agent Scaffolding Changes a Fixed LLM's Compliance with Banking Conduct Rules, Unevenly Across Models** -- `arXiv:2611.06116v1` -- 2026-11-04 -- cs.CL -- matched: llm judge
  - https://arxiv.org/abs/2611.06116
- **PAYWORTHY: Can LLMs Predict Which Freelance Jobs an AI Agent Will Complete?** -- `arXiv:2611.05759v1` -- 2026-11-03 -- cs.AI -- matched: llm judge
  - https://arxiv.org/abs/2611.05759
- **AGORA: Reliability-Aware Aggregation of LLM Judges via Item-Response Theory** -- `arXiv:2611.05556v1` -- 2026-11-03 -- cs.AI -- matched: llm judge
  - https://arxiv.org/abs/2611.05556
- **ToolSpace: Scaling Tool-Centric Environment Synthesis for Long-Horizon Agents** -- `arXiv:2611.05036v1` -- 2026-11-03 -- cs.CL -- matched: judge
  - https://arxiv.org/abs/2611.05036
- **Same Rule, Different Verdicts: Rubric Language and Clause Structure in LLM Judges for Korean Support Dialogues** -- `arXiv:2611.04091v1` -- 2026-11-03 -- cs.CL -- matched: llm judge
  - https://arxiv.org/abs/2611.04091
- **Swapping the Judge Changes the Ranking: A Preregistered Study of Cross-Provider Replication in LLM-Judged System Comparisons** -- `arXiv:2611.04090v1` -- 2026-11-03 -- cs.CL -- matched: llm judge
  - https://arxiv.org/abs/2611.04090
- **Same Outputs, Different Scores: Auditing the Unchecked Graders of Multimodal LLM Benchmarks** -- `arXiv:2611.03377v1` -- 2026-11-02 -- cs.AI -- matched: judge, leaderboard
  - https://arxiv.org/abs/2611.03377
- **Solver-Like Behavior in Large Language Models as Formalizers** -- `arXiv:2611.02395v1` -- 2026-11-02 -- cs.CL -- matched: llm judge
  - https://arxiv.org/abs/2611.02395
- **MemShift: Separating Detection from Repair in LLM-Derived Personal Memory** -- `arXiv:2611.02338v1` -- 2026-11-02 -- cs.CL -- matched: llm judge
  - https://arxiv.org/abs/2611.02338
- **ClickSight: An LLM Framework for Interpreting Student Clickstreams Through Learning Strategies** -- `arXiv:2611.02021v1` -- 2026-11-01 -- cs.AI -- matched: llm judge
  - https://arxiv.org/abs/2611.02021
