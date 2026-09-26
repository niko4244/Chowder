# RFT Self-Improvement Flywheel — Design

Status: design, not implemented. Grounded in the machinery that exists after
the GSM8K campaign (docs/SPARK_GSM8K_CAMPAIGN.md) and the six generations of
evidence it produced.

## The problem this solves

Gens 3-6 all failed the same way: every continuation from the gen-2
checkpoint (0.73) trained on **gold-answer SFT data** and regressed. Three
recipes, three regressions (0.70, 0.66, 0.68), plus a fourth at maximum data
(0.70). The common factor is not data quantity, rank, or learning rate — it
is that gold GSM8K answers are **terse, off-policy text**: the model's own
reasoning distribution never produces those exact chains, so gradient
descent on them pulls the adapter off its own manifold. The peak was always
the starting checkpoint.

Rejection-sampling fine-tuning (RFT / STaR) attacks exactly this: the
training targets are **the model's own chains that reach correct answers**.
On-policy by construction, so continued training reinforces what the model
can already almost do instead of fighting it.

## The flywheel, in Chowder's vocabulary

One loop, four stages, each stage an existing Chowder concept:

1. **SAMPLE** — a suite declared with `n_samples: k` (shipped: self-
   consistency, commit `568b6a2`) runs k temperature-sampled chains per
   training prompt. Prompts = the training slices WITHOUT their gold
   answers, `expected` kept for grading only.
2. **SELECT** — per prompt, keep chains whose extracted final number equals
   gold (the `reasoning_final_number_match` extractor decides, no human in
   the loop). Dedup: one winning chain per distinct final-expression form;
   cap ~4 winners/prompt. Prompts with zero winners become the *hard set*
   for the next iteration's sampling (more samples, higher temperature).
3. **TRAIN** — a normal generation. Dataset = winning chains rendered as
   text (the proven recipe), mixed with replay of the previous generation's
   selected chains (anti-forgetting, pattern proven in gen 5) at lr 3e-5.
4. **GATE** — `run_project()` measures the candidate on the holdout and the
   promotion gate decides. **The gate's selection is the flywheel's
   selection**: a promoted candidate's sampling run becomes the next
   iteration's SAMPLE source; a refused candidate's data is discarded with
   the run, append-only.

The protocol-identity rules stay load-bearing: sampling temperature, k,
dataset digest, and scorer content hash are all inside the evaluation
protocol fingerprint, so every generation's comparison is only ever made
against a baseline measured under the same protocol. A k or temperature
change mid-objective is refused (gen 6 proved the gate fires —
`evaluation protocol changed within the objective`) and requires the
intentional migration path. Dataset-selection policy changes are dataset
changes: new digest, new objective version, explicit.

## What the promotion gate selects that a static loop cannot

- **Data provenance becomes measured, not asserted.** Every chain in the
  next generation's dataset descends from a candidate the gate measured and
  promoted. The registry's append-only evidence trail is the dataset's
  lineage.
- **Refusals are information, not just verdicts.** A refused candidate with
  a high training-data pass-rate but a low holdout score says the selection
  is overfitting its own sampling distribution — visible in evidence
  (per-suite rows vs training pass-rate) before the next iteration spends.
- **The bar can ratchet on evidence.** After a promotion, the objective's
  minimum moves to (new score + one noise-floor step). With the 500-prompt
  holdout (CI ±0.039) the ratchet step is real: 0.02 gains are resolvable.

## Hard problems this design must not hand-wave

1. **Self-consistency scoring vs selection scoring.** Voting (k>1) is an
   *evaluation* protocol; selection needs per-chain grades, not the vote.
   The worker already records per-chain observations for sampled rows; the
   selection stage consumes `chain_observations` + chain texts, not the
   voted score. Do not blur the two protocols.
2. **Reward hacking surface.** Selection trusts the extractor. Known
   extractor pathologies (last-number-wins, comma splits) are regression-
   tested; the filter must log every rejected chain with its reason so an
   extractor bug shows up as a data-quality artifact, not silent training
   corruption. A periodic audit sample of accepted chains against gold
   reasoning (not just the final number) bounds the drift.
3. **Entropy collapse.** Iterating on your own outputs shrinks diversity;
   generation 3+ of the flywheel needs its temperature schedule and the
   hard-set mechanism to keep sampling diverse. Watch per-iteration
   pass-rate: healthy RFT has pass-rate rising while hard-set size falls.
4. **Cost bound.** Each iteration = k×|train| sampled chains (~1500
   prompts × k=8 ≈ 12k generations ≈ 5-6 h on the 4B) + training + holdout
   eval. Bounded by the gpu_hour_budget the objective already declares;
   the flywheel stops when the budget stops, and exhaustion is never
   success (lifecycle contract).

## Sizing the first iteration

- Base for sampling: gen-2 promoted adapter (peak, 0.73 measured; 500-prompt
  re-measure pending — this design's iteration 0 uses whatever that run
  reports as the honest floor).
- Prompts: gen1+gen2 training questions (400) + 200 unseen training-range
  questions; gold answers used only for grading.
- k=8, temperature 0.8 (above the eval's 0.7 — selection wants diversity,
  voting wants consensus).
- Expected yield at ~73% single-shot pass: ~450 prompts with ≥1 winner →
  ~1.2-1.5k selected chains. One generation trains on those + replay.
- Success criterion for iteration 1: holdout score (500-prompt protocol)
  ≥ gen-2 + 0.02 with promotion gate clean. If iteration 1 also regresses,
  the RFT hypothesis is refuted for this model at this scale and the
  distillation path (stronger teacher) takes priority — that conclusion
  would itself be evidence, recorded the same way.
