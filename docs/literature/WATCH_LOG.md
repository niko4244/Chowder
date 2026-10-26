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


## Automated pool drop -- 2026-10-26

> **Unvetted, machine-appended -- nothing here is registered.** These are
> arXiv search hits that named a surface mechanism in a watched primary
> category, listed for triage. No paper here has been read, no result is
> quoted, and no intervention family, maturity label, or gate is touched
> (`README.md`, rule 7). Curate anything real into a dated curated section
> under the rules, or delete this drop if it adds nothing.

- window: last 14 day(s) | surfaced: 70 | new: 65 | already in the log: 5 | off-scope: 17
- source: `.github/workflows/literature-watch.yml` (`watch.py --append-log`: the primary-category gate plus the surface-term screen)

### training-methods

- **Reinforcement Learning with Multi-Step Transition Predictions: A Bellman Theory** -- `arXiv:2610.32887v1` -- 2026-10-23 -- cs.LG -- matched: reinforcement learning
  - https://arxiv.org/abs/2610.32887
- **Multiplicative vs. Additive State Coupling in Learned Optimizer Schedules** -- `arXiv:2610.32801v1` -- 2026-10-23 -- cs.LG -- matched: optimizer
  - https://arxiv.org/abs/2610.32801
- **Accordion: Elastic-Depth Inference via Fixed-Prefix Layer Looping** -- `arXiv:2610.32492v1` -- 2026-10-23 -- cs.LG -- matched: pretraining, reinforcement learning, fine-tuning
  - https://arxiv.org/abs/2610.32492

### compression

- **LoCA: Enabling Multi-Modality in Geospatial Location Encoders** -- `arXiv:2610.32894v1` -- 2026-10-23 -- cs.CV -- matched: low rank
  - https://arxiv.org/abs/2610.32894
- **Sustainability of TSFM Deployment and Pretraining: From Carbon Footprint Analysis to Mitigation via Pruning** -- `arXiv:2610.32859v1` -- 2026-10-23 -- cs.LG -- matched: prun
  - https://arxiv.org/abs/2610.32859
- **Neural LoFi with Backward Coupling: A Spectral Theory of Cross-Layer Feature Learning** -- `arXiv:2610.32849v1` -- 2026-10-23 -- cs.LG -- matched: low-rank
  - https://arxiv.org/abs/2610.32849
- **Earth embeddings are decomposable records of the land surface** -- `arXiv:2610.32617v1` -- 2026-10-23 -- cs.LG -- matched: compress
  - https://arxiv.org/abs/2610.32617
- **Sharpness-aware Orthogonalized Momentum Algorithms** -- `arXiv:2610.32465v1` -- 2026-10-23 -- cs.LG -- matched: quantiz
  - https://arxiv.org/abs/2610.32465
- **BiLoCo: Binary Low-Rank Corrections for Post-Training Quantization** -- `arXiv:2610.32243v1` -- 2026-10-23 -- cs.LG -- matched: quantiz, low-rank
  - https://arxiv.org/abs/2610.32243
- **Rethinking LoRA Initialization for Robust Asymmetric Learning Rates** -- `arXiv:2610.32142v1` -- 2026-10-23 -- cs.LG -- matched: low-rank
  - https://arxiv.org/abs/2610.32142
- **MPQ: A Message-Passing View of Post-Training Quantization** -- `arXiv:2610.32127v1` -- 2026-10-23 -- cs.AI -- matched: quantiz, compress
  - https://arxiv.org/abs/2610.32127
- **DBCM: Dual-Branch Calibration Module for Test-Time Adaptation in Time Series Forecasting** -- `arXiv:2610.32071v1` -- 2026-10-23 -- cs.LG -- matched: low-rank
  - https://arxiv.org/abs/2610.32071
- **MAGIC: Margin-gated Contrastive Debiasing with Evidence Restoration for VLM Test-time Robustness Scaling** -- `arXiv:2610.32013v1` -- 2026-10-23 -- cs.CV -- matched: prun
  - https://arxiv.org/abs/2610.32013
- **GS-MoE: Training-Free MoE Compression via Groupwise Subspace Sharing and Pruning** -- `arXiv:2610.32009v1` -- 2026-10-23 -- cs.LG -- matched: prun, low-rank, compress
  - https://arxiv.org/abs/2610.32009
- **Every Weight Is a Path of Shared Discrete Decisions** -- `arXiv:2610.31931v1` -- 2026-10-22 -- cs.LG -- matched: quantiz, prun, low-rank, compress
  - https://arxiv.org/abs/2610.31931
- **Decode Late, Multiply Once: Rethinking the Dot Product for Quantized Weights** -- `arXiv:2610.31913v1` -- 2026-10-22 -- cs.LG -- matched: quantiz
  - https://arxiv.org/abs/2610.31913
- **Learning Structured Sparsity for Large Models: Bridging Model Compressibility and Efficient Inference** -- `arXiv:2610.31897v1` -- 2026-10-22 -- cs.LG -- matched: low-rank, compress, sparsit
  - https://arxiv.org/abs/2610.31897

### peft-adapters

- **Interpretable sparse fine-tuning with TopKLoRA enables faithful circuit discovery** -- `arXiv:2610.32866v1` -- 2026-10-23 -- cs.LG -- matched: lora, adapter, parameter-efficient
  - https://arxiv.org/abs/2610.32866
- **DIAL-S2ST: Learning Latency Control for Simultaneous Speech-to-Speech Translation** -- `arXiv:2610.32214v1` -- 2026-10-23 -- cs.CL -- matched: lora, adapter
  - https://arxiv.org/abs/2610.32214
- **Parameter-Efficient Adaptation of Large Language Models to a Low-Resource Agglutinative Language: A Corpus, Datasets, and a Reproducible Benchmark for the Bashkir Language** -- `arXiv:2610.31397v1` -- 2026-10-22 -- cs.CL -- matched: qlora, parameter-efficient
  - https://arxiv.org/abs/2610.31397
- **CoWriter: Guiding Methodological and Synergetic Human-LLM Co-Authoring** -- `arXiv:2610.31277v1` -- 2026-10-22 -- cs.CL -- matched: lora
  - https://arxiv.org/abs/2610.31277

### distillation

- **On-Policy Self-Training: Data-Free Self-Improvement in Large Language Models** -- `arXiv:2610.32770v1` -- 2026-10-23 -- cs.CL -- matched: distillation
  - https://arxiv.org/abs/2610.32770
- **TileShare: Training-Free Acceleration of Overlapping Tiled ViT Decoding** -- `arXiv:2610.32233v1` -- 2026-10-23 -- cs.CV -- matched: distillation
  - https://arxiv.org/abs/2610.32233
- **REACT: Representational Early Analysis and Control of Chain-of-Thought Trajectories** -- `arXiv:2610.32113v1` -- 2026-10-23 -- cs.LG -- matched: distil
  - https://arxiv.org/abs/2610.32113
- **ReMedI: Inference-Time Refinement and Self-Distillation for Controllable Molecular Optimization** -- `arXiv:2610.32103v1` -- 2026-10-23 -- cs.LG -- matched: distillation
  - https://arxiv.org/abs/2610.32103
- **Can Test-Time Memory Evolution Outperform Test-Time Training for Scientific Discovery?** -- `arXiv:2610.32046v1` -- 2026-10-23 -- cs.AI -- matched: distil
  - https://arxiv.org/abs/2610.32046
- **REDAR: Reinforcement-Driven Adaptive Regularisation for Class Incremental Learning** -- `arXiv:2610.31991v1` -- 2026-10-22 -- cs.LG -- matched: distillation
  - https://arxiv.org/abs/2610.31991
- **Don't Forget What You Just Learned: Self-Directed Repair Turns Agent Mistakes Into Lasting Lessons** -- `arXiv:2610.31846v1` -- 2026-10-22 -- cs.AI -- matched: distil
  - https://arxiv.org/abs/2610.31846
- **A Fix for Softmax and Cross-Entropy** -- `arXiv:2610.31642v1` -- 2026-10-22 -- cs.LG -- matched: distillation
  - https://arxiv.org/abs/2610.31642
- **OP-Print: Tracing Membership Imprints in On-Policy Self-Distillation** -- `arXiv:2610.31616v1` -- 2026-10-22 -- cs.LG -- matched: distillation, teacher-student
  - https://arxiv.org/abs/2610.31616
- **When to Follow the Teacher: Advantage-Based On-Policy Distillation** -- `arXiv:2610.31563v1` -- 2026-10-22 -- cs.LG -- matched: distillation
  - https://arxiv.org/abs/2610.31563
- **Learning to Be Monitored: Causal Bounds and Intervention-Aware Post-Training Recipes for Chain-of-Thought Monitorability** -- `arXiv:2610.31733v1` -- 2026-10-22 -- cs.LG -- matched: distillation
  - https://arxiv.org/abs/2610.31733
- **DIAL: Steerable Language Models with Distilled Instruction-Adherence Levels** -- `arXiv:2610.31438v1` -- 2026-10-22 -- cs.AI -- matched: distillation
  - https://arxiv.org/abs/2610.31438
- **HERD: Harness-Evolving Recursive Distillation for Continual Learning** -- `arXiv:2610.31401v1` -- 2026-10-22 -- cs.CL -- matched: distillation
  - https://arxiv.org/abs/2610.31401
- **Same molecule, Different Knowledge: Cross-Identifier Binding Failures in LLMs For chemistry** -- `arXiv:2610.30903v1` -- 2026-10-22 -- cs.LG -- matched: distillation
  - https://arxiv.org/abs/2610.30903
- **Building Presentation Intelligence for High-Quality Slide Generation** -- `arXiv:2610.30888v1` -- 2026-10-22 -- cs.LG -- matched: teacher model
  - https://arxiv.org/abs/2610.30888
- **Tracing Subliminal Learning Through Pretraining** -- `arXiv:2610.30884v1` -- 2026-10-22 -- cs.LG -- matched: distillation
  - https://arxiv.org/abs/2610.30884
- **One Teacher, Many Views: Reducing Reference Sensitivity in On-Policy Self-Distillation** -- `arXiv:2610.30880v1` -- 2026-10-22 -- cs.LG -- matched: distillation
  - https://arxiv.org/abs/2610.30880

### efficient-inference

- **KnitTrans: Coordinated Trajectory Transfer for Effective and Efficient LLM Reinforcement Learning** -- `arXiv:2610.32389v1` -- 2026-10-23 -- cs.DC -- matched: latency, serving
  - https://arxiv.org/abs/2610.32389
- **Correct but Unselectable: The Hidden Interface Tax in Multi-Candidate Reasoning** -- `arXiv:2610.32235v1` -- 2026-10-23 -- cs.LG -- matched: latency
  - https://arxiv.org/abs/2610.32235
- **Speculative Diffusion is a Tale of Two Tradeoffs** -- `arXiv:2610.31819v1` -- 2026-10-22 -- cs.LG -- matched: speculative decoding, draft model
  - https://arxiv.org/abs/2610.31819
- **Efficient Omni-modal Large Language Model Inference via Per-Query Modality Selection** -- `arXiv:2610.31768v1` -- 2026-10-22 -- cs.LG -- matched: latency
  - https://arxiv.org/abs/2610.31768
- **Evaluating the Applicability of LLM Inference Optimizations on Apple Silicon** -- `arXiv:2610.31479v1` -- 2026-10-22 -- cs.LG -- matched: speculative decoding, throughput, serving
  - https://arxiv.org/abs/2610.31479
- **Joint or Per-Option? Compact Encoders for Closed-Set Decisions from 17 Million to 1 Billion Parameters** -- `arXiv:2610.30970v1` -- 2026-10-22 -- cs.CL -- matched: latency
  - https://arxiv.org/abs/2610.30970
- **Energy and Latency First: How Deployment Constraints Reverse Time-Series Model Selection** -- `arXiv:2610.30965v1` -- 2026-10-22 -- cs.LG -- matched: latency
  - https://arxiv.org/abs/2610.30965
- **Memory-Augmented Online Test-Time Adaptation for EEG Motor Imagery via Perturbation Consistency and Confidence Calibration** -- `arXiv:2610.30422v1` -- 2026-10-21 -- cs.LG -- matched: latency
  - https://arxiv.org/abs/2610.30422
- **SWE-VERIFIER: Selective Verification for Cost-Efficient Coding Agents** -- `arXiv:2610.30284v1` -- 2026-10-21 -- cs.LG -- matched: inference efficiency
  - https://arxiv.org/abs/2610.30284
- **Looped Uniform Diffusion Language Models** -- `arXiv:2610.30066v1` -- 2026-10-21 -- cs.CL -- matched: inference efficiency
  - https://arxiv.org/abs/2610.30066

### evaluation-integrity

- **CounselPlex: A Stressor-Aware Full-Duplex Counseling Model** -- `arXiv:2610.32702v1` -- 2026-10-23 -- cs.CL -- matched: judge
  - https://arxiv.org/abs/2610.32702
- **How Ground Truth Shapes Retrieval Scores: Evidence-Aware Evaluation of a Retrieval-Augmented Generation System for Breast Lesions** -- `arXiv:2610.31606v2` -- 2026-10-22 -- cs.CL -- matched: llm judge, annotat
  - https://arxiv.org/abs/2610.31606
- **An Empirical Study of Recent Large Language Models for Knowledge Graph to Text Generation on Standard Benchmarks** -- `arXiv:2610.31529v1` -- 2026-10-22 -- cs.CL -- matched: annotat
  - https://arxiv.org/abs/2610.31529
- **From Clinical Judgment to Checkable Rules: Clinician-Led Guardrails for Mental-Health Chatbots** -- `arXiv:2610.30690v1` -- 2026-10-22 -- cs.CL -- matched: llm judge
  - https://arxiv.org/abs/2610.30690
- **Behind Aggregate Scores: What Systematically Varying Arena-Hard's LLM Judge Reveals** -- `arXiv:2610.30512v1` -- 2026-10-21 -- cs.CL -- matched: llm judge
  - https://arxiv.org/abs/2610.30512
- **What the Author Wrote, What the Agent Reads: Silent Cue Loss in Enterprise Document Ingestion** -- `arXiv:2610.30026v1` -- 2026-10-21 -- cs.CL -- matched: llm judge
  - https://arxiv.org/abs/2610.30026
- **Real PMIDs, Wrong Papers: Harness Design for Biomedical Claim-Verification Agents** -- `arXiv:2610.29812v1` -- 2026-10-21 -- cs.AI -- matched: llm judge
  - https://arxiv.org/abs/2610.29812
- **Signals and Queries: Token-Efficient Agentic Context Discovery** -- `arXiv:2610.29326v1` -- 2026-10-21 -- cs.CL -- matched: judge
  - https://arxiv.org/abs/2610.29326
- **Hierarchical Masked Diffusion Language Models** -- `arXiv:2610.29255v1` -- 2026-10-21 -- cs.CL -- matched: llm judge, annotat
  - https://arxiv.org/abs/2610.29255
- **Our Own Jev: Typed Decisions in One Forward Pass, from an In-House Judge to Open LoRA Models** -- `arXiv:2610.29052v1` -- 2026-10-21 -- cs.CL -- matched: llm judge, leaderboard
  - https://arxiv.org/abs/2610.29052
- **SANE: Automated Scorer-Specific Metamorphic Testing for LLM Judges** -- `arXiv:2610.28886v2` -- 2026-10-20 -- cs.CL -- matched: llm judge, evaluator
  - https://arxiv.org/abs/2610.28886
- **Capable Teacher, Myopic Scorer: Reward-Free Teacher Repair for Agentic On-Policy Distillation** -- `arXiv:2610.28586v1` -- 2026-10-20 -- cs.CL -- matched: llm judge
  - https://arxiv.org/abs/2610.28586
- **Building Telecom Language Models with RSOD: Domain Data, Specialist Post-Training, Memory-Conscious Distillation, and Networking Evaluation** -- `arXiv:2610.27993v1` -- 2026-10-20 -- cs.AI -- matched: judge
  - https://arxiv.org/abs/2610.27993
- **Why Do LLMs Accept Generics? Explaining Acceptability using Psycholinguistic and Model-Derived Factors** -- `arXiv:2610.27928v1` -- 2026-10-20 -- cs.CL -- matched: judge
  - https://arxiv.org/abs/2610.27928
- **Memory as Code: Declarative Retrieval over Codified Long-Term Memory** -- `arXiv:2610.27700v1` -- 2026-10-20 -- cs.CL -- matched: judge
  - https://arxiv.org/abs/2610.27700
- **Corpus Quality Improves, Retrieval Utility Does Not: An Ablation of Preprocessing and Refinement in a Speech-to-QA Pipeline** -- `arXiv:2610.27652v1` -- 2026-10-20 -- cs.CL -- matched: llm judge
  - https://arxiv.org/abs/2610.27652
- **Recoverability and Verifier Strength: Two Measured Preconditions for Unattended Agent Self-Improvement** -- `arXiv:2610.27465v1` -- 2026-10-20 -- cs.AI -- matched: llm judge
  - https://arxiv.org/abs/2610.27465
