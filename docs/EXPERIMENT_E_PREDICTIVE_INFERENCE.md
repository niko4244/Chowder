# Experiment E: Predictive Inference with Sparse Knowledge Access

**Question:** can a small model, speculative decoding, and selective retrieval
reduce the average cost of operating the 9B teacher — without hiding a quality
regression or making the teacher look unnecessary when it still verifies?

**Reference model (not an oracle):** `Qwen3.8-9B-abliterated-25` (Q4_K_M GGUF
via llama.cpp, 57 tok/s). **Small model:** `Spark-X2.5-4B` (Q8_0 GGUF via
llama.cpp, ~55 tok/s; 131072-token vocab, reasoning-style outputs).

Artifacts: `C:/Users/nikma/Chowder/.chowder-spark-calib/exp-e/`
(`tasks.json`, `spec_baseline.json`, `spec_ngram.json`, `spark_baseline.json`,
`full_run.json`, `repairs_rerun.json`).

## 0. Model serving reality (context for every number below)

* An EAGLE-style learned draft head is **not buildable here**: the two models
  use incompatible tokenizers (131072 vs 248320) and no trained draft head
  exists for either. This is a hard architectural rejection, documented rather
  than hand-waved.
* The HF transformers path for the 9B teacher (19 GB bf16) segfaults
  intermittently under this machine's memory pressure; the teacher-cache run
  of Experiment A succeeded earlier, but this experiment moved entirely to
  llama.cpp GGUF serving for stability and speed (57 tok/s vs ~7 tok/s).
* The Spark GGUF loads in llama.cpp directly (Ollama's wrapper 500s on it —
  an upstream conversion issue, not the weights).

## 1. Phase 1 — three baselines (identical eval tasks, greedy, n=10 GSM8K / n=8 factual)

| variant | GSM8K acc | factual acc | latency/task | tokens/task |
|---|---:|---:|---:|---:|
| A. teacher alone | **0.90** | 0.375 | 12.2 s | 638 |
| B. small alone | 0.70 | 0.375 | 9.98 s | 550 |
| C. small + bm25 | 0.70 | 0.75 | 10.16 s | 552 |
| C. small + dense | 0.70 | **0.875** | 10.72 s | 576 |
| C. small + sparse-learned | 0.60 | 0.625 | 10.54 s | 572 |

Readings:

* Retrieval is the single biggest factual win: dense RAG takes the small model
  from 0.375 → 0.875 factual accuracy (+0.50) and cuts latency 9.0 s → 3.8 s
  (shorter grounded answers). **Citation rate 1.00** — answers grounded in
  retrieved evidence cite their sources, because evidence is wrapped in
  `[source: doc_id]` markers the model echoes.
* Retrieval does not help math: GSM8K is unchanged at 0.70 across all retrieval
  variants — arithmetic failures are generation failures, not knowledge gaps.
* The teacher wins GSM8K (0.90 vs 0.70) but is **no better on factual recall
  than the bare small model** (0.375 both). The teacher is a better reasoner,
  not a better knowledge base — precisely why it must not be treated as an
  oracle: its factual answers are as ungrounded as the small model's.
* The learned sparse memory (Phase 3's experimental variant) is the weakest
  retriever and it also dragged GSM8K below the no-retrieval baseline (0.60 vs
  0.70) — bad retrievals actively inject noise into the prompt. With only 7
  corpus documents its bottleneck code overfits document self-similarity
  (train loss 0.009) and misses query semantics.

## 2. Phase 2 — speculative decoding

* Prototype: llama.cpp `--spec-type ngram-simple` (context n-gram drafting +
  exact greedy verification by the model itself).
* **Output equivalence: perfect on all 4 prompts** — speculative outputs are
  token-for-token identical to the plain-decode reference (temperature 0).
* Speedups, measured as tokens/second (server timings, median of 3):

| prompt type | plain | speculative | speedup |
|---|---:|---:|---:|
| copy_function | 57.1 | 117.1 | **2.05x** |
| copy_tools | 57.2 | 161.2 | **2.82x** |
| copy_repeat | 57.5 | 108.4 | 1.89x |
| generative | 57.1 | 54.7 | 0.96x |

* The speedup is real tok/s on identical outputs, not an acceptance-rate
  proxy. On generative text the draft misses and costs ~4% (0.96x) —
  the honest overhead measurement.
* Scope: n-gram drafting pays off exactly where outputs copy the context
  (repair flows, structured repeats). It does not help open generation.
  A learned draft head could broaden this, but is blocked by the tokenizer
  mismatch above.

## 3. Phase 3 — retrieval subsystems (storage, latency, accuracy)

| method | lookup latency | storage | retrieval accuracy* |
|---|---:|---:|---:|
| none | 0 ms | 0 | n/a |
| bm25 | ~0.3 ms | ~0 (inverted index) | 2/2 spot-checks |
| dense (nomic 768-d) | 52–76 ms | 21,504 B | 2/2 |
| sparse-learned (64-d code) | 48–56 ms | 198,400 B | 1/2 |

*Factual-task end-to-end accuracy (the retrieval-to-answer pipeline) is the
Phase-1 table's real measure; the spot-checks here isolate the retriever.

* BM25 is both the fastest and cheapest and matches dense on spot-checks;
  dense wins end-to-end factual accuracy (0.875 vs 0.75) because soft
  matching helps the model *use* the evidence, not just find it.
* The learned sparse memory's storage exceeds the dense vectors it compresses
  at this corpus size (198 KB vs 21 KB) — its savings profile only makes sense
  with large corpora. Verdict at this scale: not competitive, honestly
  reported as such.
* Evidence vs assumptions: every retrieved passage is wrapped in
  `[source: doc_id]` markers, and grading separates citation correctness from
  answer correctness. The 1.00 citation rate under retrieval shows the model
  distinguishes retrieved evidence from generated assumptions when the
  boundary is made explicit.

## 4. Phase 4 — adaptive compute routing

Calibration on dev tasks (Spark self-review as the router signal, allowed to
see only dev data):

| calibration stat | value |
|---|---:|
| dev small-model accuracy | 0.786 |
| router precision (confident ⇒ correct) | 0.786 |
| **confident-and-wrong** (dangerous cell) | **3 of 14** |
| unconfident-but-correct | 0 of 14 |

Routing on eval tasks (self-review gate; escalation to teacher when unsure):

| route | tasks | accuracy |
|---|---:|---:|
| routed small (confident) | 11 | 0.55 |
| routed large (unconfident) | 7 | 0.57 |
| always-large control | 18 | 0.72 |
| always-small (phase1 none) | 18 | 0.58 |

The router **did not beat the always-large control on accuracy** (0.55–0.57 vs
0.72), and its calibration shows why: self-review confidence is not
correctness — Spark rated 3 wrong answers as correct (and would have shipped
them). The router's small-route accuracy (0.55) is close to always-small
(0.58): the confidence signal barely filters anything. Routing saved large-
model invocations on 11 of 18 tasks (61% invocation reduction) but the
quality cost is real and the memo's warning stands: **confidence is not
correctness.**

## 5. Phase 5 — comparative evaluation

Combined metrics across the full pipeline (full_run.json + repairs_rerun.json):

* **Repair completion:** 0/4 for small, 0/4 after teacher escalation — all
  verified through the runtime harness (green test observations required; a
  claimed fix is never accepted on model say-so). Invalid reads occurred on
  2 of 4 tasks; premature completion on all 4. These match the batch-009
  findings: these repair tasks exceed both models' tool-use capability.
* **Unsupported claims:** none shipped in graded metrics — the router's
  confident-and-wrong answers are recorded and called out rather than averaged
  away, and repairs are harness-verified.
* **Large-model invocation frequency:** 7/18 routed tasks (39%) vs 100% in the
  control — the cost lever exists, but this router spends it badly.
* **Cost attribution** (which savings come from where):
  * **Speculative decoding:** up to 2.8x tok/s with zero quality change, but
    only on copy-shaped outputs; ~4% overhead elsewhere. Saves latency, not
    accuracy risk — equivalent outputs make it the safest lever.
  * **Retrieval:** the only lever that *improves* quality (factual +0.50) while
    cutting latency ~2.4x on factual tasks (shorter grounded answers). Costs
    0–76 ms lookup + small vector storage.
  * **Routing:** the only lever that trades quality for cost, and with
    self-review confidence as the signal it currently loses quality (−0.15
    vs always-large). Its value depends entirely on a better signal
    (harness verification for repairs is such a signal; LLM self-review is
    not).

## 6. Conclusions

1. **Where the savings are:** retrieval (quality-positive, latency-positive on
   knowledge tasks) > speculative decoding (quality-neutral, up to 2.8x on
   copy-shaped work) > routing (quality-negative with current signal).
2. **The teacher still earns its keep:** it is the best reasoner (GSM8K 0.90)
   and the verifier of last resort for repairs. Nothing in this experiment
   makes it unnecessary — the best cheap configuration (small + dense RAG)
   still loses 0.15 GSM8K to it, and self-review routing cannot close that gap
   without shipping confident errors.
3. **A viable configuration** for future work: small+dense-RAG for factual
   recall, ngram-speculative teacher decoding for repair-shaped generation,
   harness-verified escalation for repairs, and no self-review routing until a
   trustworthy confidence signal exists.
4. **Do not** route on LLM self-assessed confidence; the dangerous cell
   (confident-and-wrong) is empirically populated (3/14 dev tasks).

## Reproducibility

* Servers: `llama-server.exe -m <teacher GGUF> --port 18081 -c 2048 --temp 0
  --spec-type none|ngram-simple` and `-m <spark GGUF> --port 18082 -c 8192
  --temp 0` (Spark needs ≥8192 ctx for multi-turn harness loops).
* Task suite: `python chowder_batch/exp_e_tasks.py <out>`; corpus/retrievers:
  `exp_e_corpus.py` (requires the local Ollama server for embeddings).
* Spec measurements: `exp_e_spec_llamacpp.py --port <p> --out <out>` (run
  against both spec-type none and ngram servers; outputs must match).
* Full pipeline: `exp_e_run.py --tasks <tasks.json> --out <out>`; repairs
  rerun standalone with the same module (`run_repair_task`).
* All runs greedy (temperature 0); router calibrated only on dev tasks; all
  accuracy claims graded against validated ground truth, all repairs verified
  by actual test observations.

## 7. 2026-09-25 — Phase 4: logprob-margin router (implemented; one part now measured)

Everything in sections 1–6 above is **measured history** under the old
self-review router. This section records the Phase-4 replacement, which is
implemented and unit-tested, plus the one piece of it that has now been
executed live — the Experiment F margin-shift measurement below.

**Still unrun: the Phase-4 routing comparison itself.** No calibration exists
against a real teacher and no small-vs-always-large comparison has been
measured, because the teacher/Spark endpoints are not running: ports
18081–18083 now *refuse* connections rather than timing out, and Ollama
serves only GGUF quants on 11434. Do not quote a *routing* margin from this
document. The only measured margins are the Experiment F ones, and they are
scoped to the model and PTQ config named there — they are not the router's
own numbers.

Correction to the earlier text in this section: it blamed "llama-server PID
3764 holds both GPUs". That was wrong. PID 3764 is `hermes-agentsd.py` on port
7779 and holds no GPU. GPU 0 (RTX 5060 Ti) was available and was used for the
Experiment F run below.

### Why the signal changed

Conclusion 4 above stands: LLM self-assessed confidence routed the dangerous
confident-and-wrong cell (3/14 dev tasks) to the small model. Phase 4 replaces
self-review with a signal the model cannot inflate by phrasing: the
**selected-token logprob margin** of its own greedy output.

### Implementation (`chowder_batch/exp_e_confidence.py`, `exp_e_run.py`)

* **Signal.** For each selected output token, margin = logprob(selected token)
  − best alternative logprob among the returned `top_logprobs` (5 requested).
  The completion score is the **mean** of per-token margins. A margin exists
  only if every position has a *distinct* alternative token with a finite
  logprob; any missing/malformed entry makes the whole response unscorable.
* **Fail-closed plumbing.** Strict parsing (no bools-as-numbers, finite
  checks); `max_tokens > 0` is validated when logprobs are requested; HTTP
  failures and logprob-less responses are recorded as `logprob_error` with
  `logprobs_supported: false` and `logprob_margin: null` — never guessed.
* **Dev-only calibration** (`calibrate_margin_threshold`). Candidate cutoffs
  are every observed dev margin; eligibility requires precision ≥ 0.80 with at
  least `min_samples` selected responses; the winner is the highest-coverage
  eligible cutoff (all candidates are retained in the record for audit).
  Statuses: `calibrated`, `no_threshold_meets_precision`,
  `insufficient_dev_logprobs` — the latter two fail closed.
* **Routing** (`small_route_allowed`). The small model answers only when
  calibration status is `calibrated` **and** the live margin is a finite
  number ≥ the dev threshold. Every other condition — missing logprobs,
  uncalibrated dev split, malformed margin — routes to the teacher.
* **Stricter grading** (`grade_citation`). A citation now counts only if the
  response contains an actual `[source: doc_id]` string **and** the cited doc
  was retrieved. Historical 1.00 citation rates were computed under a laxer
  rule and are **not comparable** to future Phase-4 numbers.
* **Corpus provenance.** `exp_e_run.py --corpus` embeds a `corpus_manifest`
  block re-validated through `ingest_verified_documents` (exact per-document
  sha256, duplicate-text rejection) so every retrieval claim traces to hashed
  source text.
* **Per-precision thresholds (2026-09-25; shift measured, routing unrun).** Quantization can flatten
  the margin distribution, so a BF16-calibrated threshold is never applied to
  a quantized arm: `calibrate_margin_threshold_per_precision` calibrates each
  precision independently (records tagged `precision_arm`; a failing arm keeps
  its fail-closed status instead of borrowing a threshold), and
  `margin_shift_fails_closed` blocks the small route when the measured
  quantized-vs-BF16 mean-margin shift is unmeasured or exceeds the declared
  tolerance (`--max-quantized-margin-shift`, default 0.2). Guard logic lives in
  `exp_e_confidence.py`; the Kaggle-side mirror is contract-tested against it.
* **Held-out transfer gate (2026-09-25; implemented and run; pass path still unexercised — see the n=20 section below).** A dev-split threshold is
  fitted on the tasks it was chosen on, so it proves nothing about unseen
  tasks. `heldout_transfer_gate(calibration, heldout_rows)` re-scores the
  already-chosen threshold on a held-out split of the *same precision arm* and
  fails closed unless the separation survives (>= `min_tasks` held-out rows
  with margins, >= `min_samples` clearing the threshold, held-out precision >=
  `min_precision`). It refuses any split it cannot trust: a calibration with no
  recorded `fit_task_ids`, a held-out row with no task id, or any overlap
  between the fit set and the held-out set (`heldout_contaminated`).
  `calibrate_margin_threshold` now records `fit_task_ids` for exactly this
  purpose. `quant_route_allowed` **requires** the gate
  (`heldout_gate=`), and it is only honoured when the gate was measured for the
  same `precision_arm` and the same `threshold` as the calibration being
  served — BF16 transfer evidence cannot authorize an INT8 arm. Wired in
  `exp_e_run.py` via `--quantized-heldout-rows` / `--quantized-precision-arm`;
  the BF16 lane is untouched (`status: not_applicable_bf16_lane`).
* **Green-retention guard (2026-09-26; implemented, and measured on the n=20
  report).** A margin-shift bound is evidence about the margin *scale*, never
  about whether the arm still solves anything — at n=20 the shift passed while
  the quantized arm lost every green. `validate_quantized_green_retention`
  pairs per-task `{"task", "correct"}` outcomes for the reference and
  quantized arms (the same paired design as the margin table) and measures the
  fraction of the reference arm's greens the quantized arm loses. Retention is
  counted per task, so equal green *totals* on different tasks still count as
  losses and gains never offset them. `green_loss_fails_closed` blocks when
  that fraction is unmeasured, non-finite, outside `[0, 1]`, or above
  `--max-quantized-green-loss-fraction` (default
  `DEFAULT_MAX_GREEN_LOSS_FRACTION = 0.0`: no reference green may be lost).
  A reference arm with **zero** greens also blocks: with no demonstrated
  capability there is nothing whose retention could be evidenced.
  `quant_route_allowed` **requires** the measured fraction and tolerance, so a
  passing margin-shift bound can never license routing by itself;
  `router_rows_from_report` emits the measured record as `green_retention`.
  Wired in `exp_e_run.py` via `--quantized-green-loss-fraction`; the quantized
  lane activates when *either* measurement is supplied, so providing one and
  omitting the other fails closed, and the BF16 lane is untouched.
* **One admission artifact per quantized arm (2026-09-26).**
  `quantized_arm_admission` aggregates the four guards — per-precision
  calibration (tag + finite threshold), held-out transfer gate, margin shift,
  green retention — into one record carrying each guard's measured input, its
  verdict, and a `refusals` list naming every failing guard. Admission is
  granted only when all four pass, and a missing measurement or tolerance is a
  refusal here: unlike the per-guard helpers, whose `None` tolerance means
  "no quantized arm is served", the artifact only ever describes a quantized
  arm, so "unmeasured" can never be read as "safe". `exp_e_run.py` records
  `phase4.arm_admission` (`not_applicable_bf16_lane` on the BF16 lane) and
  requires `admitted` for every small-route decision through
  `quantized_route_decision`, alongside the per-query checks.
  `verify_arm_admission` re-derives every verdict from the recorded
  measurements and checks a sha256 digest over them, so an artifact that
  misreports its own verdict — or whose inputs were edited after writing —
  fails verification. This is tamper-evidence, not a signature: there is no
  key on this lane, and the truthfulness of a measurement still rests on how
  it was produced, not on the digest.
* **Measured evidence instead of hand-typed scalars (2026-09-26).**
  `exp_e_run.py --quantized-evidence <report.json>` takes the measured
  Experiment F report itself. `load_quantized_evidence` derives the serving
  arm's name, the margin shift, the green-retention fraction, the report-side
  calibration/gate records, and the quantized arm's held-out rows from its
  per-task data (`router_rows_from_report`), verifies the derived admission
  artifact, and refuses malformed or non-report files. Supplying
  `--quantized-evidence` together with any of the four scalar inputs it
  replaces is a CLI error, so derivation and declaration can never be mixed.

### Experiment F — the measured margin shift (2026-09-25, RUN)

```
python chowder_batch/exp_f_ptq_margin.py --model Qwen/Qwen2.5-1.5B-Instruct \
  --ptq-config int8_smoothquant --tasks 12 --max-new-tokens 160 --max-turns 4
```

→ `evidence/exp_f_ptq_margin_qwen25_1p5b_int8sq_20260925.json`. Paired arms on
identical tasks, greedy, `state_aware` harness, RTX 5060 Ti; modelopt 0.47.0 /
torch 2.11.0+cu128 / transformers 5.16.1; 706 quantizers inserted, 196 modules
smoothed.

| quantity | measured |
| --- | --- |
| BF16 mean margin | 4.5040 |
| INT8-SmoothQuant mean margin | 4.5419 |
| **mean margin shift (quant − bf16)** | **+0.0379** |
| shift vs `DEFAULT_MARGIN_SHIFT_TOLERANCE` (0.2) | within → guard does **not** fail closed |
| tasks with a usable margin | 12/12 in both arms (0 nulls) |
| green rate | **0/12 in both arms** |

Three findings worth keeping:

* **The tolerance consumes a paired statistic, and that is load-bearing.**
  Per-task paired deltas run from −4.16 (`exp_f_repair_3`) to +3.31
  (`exp_f_repair_5`) — a spread roughly 20× the tolerance. An earlier 2-task
  smoke run of the same script measured a shift of **−2.0551**, which looks
  like a 10× violation; at n=12 the paired mean is **+0.0379**. The 2-task
  number was sampling noise, not a quantization effect. It is recorded here
  only as the reason the router must consume an n-task paired shift instead of
  a spot check, and why any future shift claim should state its `n`.
* **The shift bound passed and the router was still blocked — by the gate.**
  With 0/12 greens in both arms, no cutoff reaches 0.80 precision: both
  per-precision calibrations return `no_threshold_meets_precision` with
  `threshold: null`, and every retained candidate scores precision 0.0. The
  held-out gate therefore records `heldout_rejected` with reason "calibration
  is not calibrated", and `quant_route_allowed` stays closed. That is the
  intended ordering — the tolerance cannot license an arm that has no
  validated threshold, and here it is the second gate, not the tolerance,
  that does the blocking.
* **This task set cannot exercise the gate's precision arm.** 0/12 greens
  means the margin→correctness relation has no positive class, so the gate's
  *discrimination* is still untested: it has never seen a case where the
  threshold could pass. Exercising that needs tasks the 1.5B model sometimes
  solves — a task-set change, not a threshold tweak. Do not read the
  rejection above as evidence that the gate is calibrated; read it as evidence
  that it fails closed.

### Attempting to exercise the gate's pass path (2026-09-26, BF16 pilots)

The measurement above leaves the gate's *pass* path unexercised, so the next
question was what it takes to make the small model actually solve tasks. Two
answers came back, and the first one invalidates the earlier 0/12 reading:

* **Nothing was ever executed.** `_render_prompt` passes JSON-schema `TOOLS` to
  `apply_chat_template`, so the model answers in Qwen's native JSON dialect,
  while `runtime_eval.parse_tool_call` parses only
  `<tool_call>name<arg_key>…`. Every call was silently discarded —
  `tool_calls: 0`, `reward: -21.0` (no-green −10, no-writes −8, premature −3).
  The 0/12 green rate therefore described the *prompt format*, not the model
  and not quantization. `--tool-call-format json` adds the decode-side twin of
  `_render_prompt`; malformed JSON is left untouched so no action is invented.
* **A batched turn must not lose its write.** The model emits `read_file` then
  `write_file` in one turn; the harness acts on one call. Since
  `_evaluate_task` grades the workspace and only a write mutates it, keeping
  the read discards the fix and manufactures a greenless run. The translator
  keeps the first *advancing* call and reports `dropped_calls` /
  `reordered_turns`.
* **Verification is a separate act.** Green comes only from a passed
  `run_tests`; the model writes the fix and then narrates. `--difficulty`
  covers this: `hard` (bare goal, byte-identical to prior runs), `guided` (the
  goal also names the verification step), `mixed` (blocks of two, so the
  interleaved calibration/held-out split gets the same composition in both
  halves).

Measured, 12 tasks, BF16, `Qwen/Qwen2.5-1.5B-Instruct`:

| difficulty | green rate | correct-task margins | incorrect-task margins |
| --- | --- | --- | --- |
| `hard` | 0/12 | — | — |
| `guided` | **5/12** | 4.53, 4.91, 5.06, 5.31, 5.42 | 3.74, 3.96, 3.99, 4.03, 4.10, 4.15, 4.27 |

The margin separates the two classes *completely* — a cutoff near 4.4 scores
precision and recall 1.0 on this set. The signal the router calibrates on is
therefore real and correctly ordered; the earlier flat result was the tool
contract, not the signal. Raising the budget to 6 turns / 256 tokens reproduced
this exactly — same 5 green tasks, same margins, same call counts — so the ~40%
success rate is a property of the model and the extra budget buys nothing.

**The gate still rejects, and that is the informative part.** With 5/12 greens
the calibration half holds only 2 correct of 6, so no cutoff reaches 0.80
precision at ≥ `min_samples` selected: `no_threshold_meets_precision` →
`heldout_rejected` / "calibration is not calibrated". The binding constraint is
the *success rate*, not the margin quality — at ~40% the 0.8-and-≥4 pair is
structurally unreachable inside a 6-task half. Reaching the pass path needs
≈ n=20 (a 10-task held-out half) or a stronger small model. Loosening
`min_precision`/`min_samples` to make it pass would only make the gate agree
with itself, so the defaults stand.

Scope: these numbers are for **Qwen2.5-1.5B-Instruct under INT8-SmoothQuant
only** (the report carries
`transfer_scope.transfers_to_other_architectures: false`). They say nothing
about the Spark-X2.5-4B teacher lane. Fake-kernel quantization on this lane
licenses no throughput or speedup claim, and none is made.

### Experiment F at n=20, `guided` + JSON tool calls (2026-09-26, RUN)

```
python chowder_batch/exp_f_ptq_margin.py --model Qwen/Qwen2.5-1.5B-Instruct \
  --ptq-config int8_smoothquant --tasks 20 --difficulty guided \
  --tool-call-format json --max-new-tokens 160 --max-turns 4
```

→ `evidence/exp_f_ptq_margin_qwen25_1p5b_int8sq_guided20_20260926.json`.
Same paired design, ~1 h 40 m wall clock (INT8 arm 3–10 min/task).

| quantity | BF16 | INT8-SmoothQuant |
| --- | --- | --- |
| green rate | **6/20** | **0/20** |
| mean margin | 4.4662 | 4.4737 |
| tasks with a usable margin | 20/20 | 20/20 (0 nulls) |
| mean runtime reward | −1.8 | −12.0 |
| executed tool calls | 29 | 20 |

Mean margin shift **+0.0076** — far inside the 0.2 tolerance, so
`margin_shift_fails_closed` is False again; paired per-task deltas span
−2.123 … +2.457. This run was commissioned to give the held-out gate a 10-task
half. It did **not** reach the pass path, and why is now measured:

* **The quantized arm loses every task the BF16 arm solves.** All six BF16
greens (`exp_f_repair_1, _3, _6, _9, _10, _14`, margins 4.53–5.79) came back
non-green under INT8-SmoothQuant — none kept, none gained — and the quantized
arm emitted fewer *executable* calls (20 = one per task, vs 29 in BF16), the
shape of degraded structured output rather than one bad edit. On this lane,
under this recipe, the small arm stops closing repair tasks at all.
* **The two guards disagree, and the disagreement is the finding.** The shift
bound passed (+0.0076) while the arm scored nothing at all, so a router
licensed by the tolerance alone would have served an arm that never once
succeeded. The per-precision calibration and the held-out gate blocked it:
both arms return `no_threshold_meets_precision` (the INT8 calibration half has
zero positives, so every candidate scores precision 0.0; BF16 tops out at
precision **0.75** with 4 selected, below the 0.80 bar), so the gate records
`heldout_rejected` / "calibration is not calibrated" and
`quant_route_allowed` stays closed. **The gate is load-bearing, not
hardening**: of the two checks, only it saw this. The green-retention guard
above is the codification of the finding: fed this same report it measures
6/6 reference greens lost (fraction 1.0) and refuses the arm with the shift
verdict still False.
* **"Reaching the pass path needs ≈ n=20" is falsified, and is retired.**
Precision is a ratio, not a count: the BF16 miss that caps the ceiling
(`exp_f_repair_16`, margin 4.966) *outranks* a green (`exp_f_repair_6`,
4.909), so adding tasks grows numerator and denominator together and leaves
the ceiling at 0.75. Reaching `heldout_validated` needs better margin
*ordering* or a quantized arm that still succeeds — a stronger small model, a
gentler PTQ recipe, or tasks that are easier without being trivial — not a
longer task list. Loosening `min_precision`/`min_samples` remains off the
table.
* **The n=12 "perfect separation" is retired as a small-sample artifact.**
At n=20 two *incorrect* tasks sit inside the green band (4.966, 5.121). The
signal is real and correctly ordered, but it is not clean, and no earlier
separation claim should be reused.

### Experiment F at n=20 with `int8_weight_only` (2026-09-26, RUN)

```
python chowder_batch/exp_f_ptq_margin.py --model Qwen/Qwen2.5-1.5B-Instruct \
  --ptq-config int8_weight_only --tasks 20 --difficulty guided \
  --tool-call-format json --max-new-tokens 160 --max-turns 4
```

→ `evidence/exp_f_ptq_margin_qwen25_1p5b_int8wo_guided20_20260926.json`.
Same paired design, ~41 min wall clock (vs ~1 h 40 m for INT8-SmoothQuant).
Weight-only INT8 leaves activations in BF16, and the arm still solves tasks:

| quantity | BF16 | INT8 weight-only | INT8-SmoothQuant, for contrast |
| --- | --- | --- | --- |
| green rate | **6/20** (0.30) | **5/20** (0.25) | 0/20 (0.00) |
| mean margin | 4.4662 | 4.5543 | 4.4737 |
| mean margin shift | — | **+0.0881** (within 0.2) | +0.0076 |
| mean runtime reward | −1.8 | −1.95 | −12.0 |
| executed tool calls | 29 | 39 | 20 |
| reference greens | — | **3 of 6 retained** (lost `repair_6`, `_9`, `_14`; gained `repair_0`, `_4`) | 0 of 6 |

* **The SmoothQuant collapse is the recipe, not 8-bit weights.** Weight-only
  quantization keeps the arm solving tasks (green 5/20, reward −1.95 vs −12.0)
  and emitting *more* executable calls than BF16 (39 vs 29). The earlier
  "INT8 loses everything" was specific to SmoothQuant's activation
  quantization.
* **First arm where the green-retention guard has real work to do.** Reference
  6 greens → 3 retained, 3 lost, 2 gained: `green_loss_fraction` **0.5**,
  above the default 0.0 tolerance, so the guard refuses. Both sides matter -
  the arm gains two tasks BF16 cannot solve *and* loses half of BF16's
  greens - and the measured pair makes both visible. A declared
  `--max-quantized-green-loss-fraction 0.5` would pass this guard (it honours
  the declared bound); nothing else does.
* **The shift bound passed again while half the reference greens were lost.**
  Third independent instance of the pattern (n=12: shift passed with no
  positives at all; n=20 SmoothQuant: shift passed with a dead arm; here:
  shift passed, retained 3 of 6). Per-task deltas span −2.129 … +1.707.
* **The gate still blocks, and the requirement is now quantified: a 10-task
  calibration half needs ≥4 correct tasks before any cutoff can be eligible.**
  With precision ≥0.80 and `min_samples=4`, an eligible cutoff must select ≥4
  rows of which ≥3.2 are correct, so coverage alone requires ≥4 successes in
  the half. Both arms here have exactly 3 greens in their fit half (BF16:
  `_6, _10, _14`; weight-only: `_0, _4, _10`), so both calibrations return
  `no_threshold_meets_precision`: BF16's best ≥4-selected candidate is
  precision 0.75, the weight-only arm's is 0.6. Under the fixed interleaved
  split, ≥4 greens per half is *guaranteed* only from ≥14 correct tasks of 20;
  at ~25–30% success density more tasks of this difficulty only reshuffle
  which half falls short. Ordering still matters above that floor: the
  weight-only arm's misses include a high-margin row (`repair_6`, 5.678), so
  its ceiling sits *below* BF16's despite a similar success rate.
* **A passing tolerance, a live arm, and a better recipe still license
  nothing.** `router_rows_from_report` on this report refuses admission on
  calibration, held-out gate, **and** green retention; only the margin-shift
  guard passes.

### Required before any Phase-4 claim

1. Fresh backend preflight proving the correct teacher/Spark models on known
   ports (current endpoints fail this), plus a task set with enough success
   density: SmoothQuant solved 0/20 at n=20, weight-only 5/20, but each left
   only 3 successes per 10-task half — below the ≥4 a precision-0.80 cutoff
   needs — so the held-out gate's pass path stays unproven. The pass path
   needs ≥4 successes per half *and* margins ordered above the misses.
2. Dev-split logprobs (`dev_logprobs_available`) → calibration record.
3. Only then the Phase-4 routing comparison: small+router vs always-large,
   alongside the historical self-review router for continuity.

Unit coverage lives in `tests/test_exp_e_confidence.py` (strict parsing,
min-samples, fail-closed routing, the held-out transfer gate and its
contamination/transferability refusals, the green-retention guard — per-task
pairing, vacuous-reference refusal, the passing-shift-cannot-license case —
and the admission artifact's aggregation/verification),
`tests/test_exp_e_run.py` (driver wiring: either measurement alone leaves the
quantized lane closed, routing follows the aggregate admission, the evidence
file derives its measurements and refuses conflicts, and the measured n=20
report refuses on retention with the shift passing), `tests/test_exp_e_pipeline.py` (corpus
provenance, retrieval train/holdout integrity, batch-010 export guards), and
`tests/test_batch010_contract.py` (the batch-010 producer/consumer contract —
real `build_batch010_dataset` output fed through the Kaggle lane's
`load_teacher_rows`, in both drift directions).
