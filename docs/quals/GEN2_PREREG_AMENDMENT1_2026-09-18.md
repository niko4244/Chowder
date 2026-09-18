# GEN2 preregistration — Amendment 1 (2026-09-18)

Written **before any gen2 compute**. No gen2 training, evaluation or model
load has run: the only gen2 artifacts in existence are the declaration
(`docs/gen2/gen2_campaign.json`), the frozen judge (`docs/gen2/judge_gen2.py`)
and the preregistration itself. Therefore **no candidate result was visible
when this amendment was written**, and no threshold below changes.

This amendment does three things:

1. makes the executable path say the same thing as the frozen text about the
   device ceiling (section A);
2. fixes identity so a base digest and an adapter digest are never the same
   field (section B);
3. makes the frozen judge's gates implement the rules the frozen text already
   states, instead of weaker approximations of them (section C).

No promotion threshold, benchmark set, budget ceiling, statistical rule or
stopping rule changes. Section D states that explicitly, item by item.

---

## A. The device ceiling: admission-only until device time is measured

**What was wrong.** Prereg section 5 says: *"Settlement: each attempt's and
the campaign's actual measured cost must fit these ceilings after execution,
or the cycle refuses."* That is true of the **wall** ceilings. It was not true
of the **device** ceiling, and could not be: nothing in the executable path —
trainer, registry, or evaluator — reports device GPU-hours, so the only device
figure available at settlement is the `0.0` placeholder a wall-only ledger
writes. #177 made exactly that figure fail closed
(`ACTUAL_DEVICE_GPU_HOURS_UNMEASURED`), which means handing a declared device
ceiling to settlement would refuse **every** real attempt after burning its
compute. A gate that must be switched off within a day is not a gate.

**Why it was caught before compute.** `chowder growth campaign settle` and the
new campaign runner both distinguish "declared device time is measured" from
"wall only"; the frozen judge did not, so the same accounting artifact could be
uncertified by production and PASS in the judge. That divergence is the defect
this amendment removes, and it was found by reading the two paths against each
other during the certification-boundary audit, not from a result.

**Corrected rule (now the only rule).**

- `budget.device_time_measured` is declared explicitly in the campaign
  manifest. Gen2 declares **`false`**: the trainer charges wall time.
- With `false`, the device ceilings — per recipe and campaign — are
  **admission constraints on the projected plan**. They refuse a plan that
  does not fit *before* any compute, at one owner (the executor's `admit`
  seam for recipes, the campaign projection control for the campaign total).
- With `false`, the **wall** ceilings are the post-run settlement gates, and
  the campaign's settlement is the production `settle_campaign()` answer.
- If a future trainer measures device time, the campaign declares
  `device_time_measured: true` and the device ceilings become hard post-run
  settlement gates in the same code path; an unmeasured device figure then
  fails closed with `ACTUAL_DEVICE_GPU_HOURS_UNMEASURED`. This is tested in
  `tests/test_growth_gen2_judge.py` and `tests/test_growth_campaign.py`.

Section 5's sentence is therefore read as: *the wall ceilings settle post-run;
the device ceilings are admission-only while `device_time_measured` is false.*

## B. Identity: base and parent adapter are separate, verified objects

**What was wrong.** One field carried two meanings. Section 2 of the manifest
described `F:\llm-models\Qwen3.8-9B-abliterated-25-bf16` as the parent while
`parent_model_digest` held `ca8769c5…`, which is the **gen1 adapter** digest,
not the base tree's. A reader could not tell which object the digest covered,
and the runner verified the base tree against the adapter's digest.

**Corrected declaration.**

```json
{
  "base_model_path": "F:/llm-models/Qwen3.8-9B-abliterated-25-bf16",
  "base_model_digest": "59e767aab1dabceef098ceb367d872ef735114412383ae8e622a95c64297555f",
  "parent_adapter_path": "…/2026-09-17-gen1-protocol-compliance/attempts/attempt-10/work/.chowder/runs/gen1-protocol-compliance-a10-46861b2929bd/adapter",
  "parent_adapter_digest": "ca8769c5e7e06575c95501391e328811b55f2208d9cbb10b86e95eb40d8786cd"
}
```

- `base_model_digest` is the frozen Generation-0 content digest from the
  Generation-0 freeze; `parent_adapter_digest` is the gen1 adapter artifact
  digest recorded in
  `…/2026-09-17-gen1-protocol-compliance/chosen_candidate.json` and
  re-verified byte-for-byte with the production `directory_digest` when this
  amendment was written.
- Both digests have validator-enforced sha256 syntax, and the execution path
  recomputes each against **its own** tree before any compute. A base digest
  is never accepted as an adapter digest.
- `parent_adapter_*` is omitted when the parent generation *is* the base.
- The old `parent_model_digest` field is gone rather than overloaded.

## C. The judge now implements the frozen rules, not approximations of them

Each item below is a gate that the frozen text already required and the judge
did not enforce. None of them loosens anything; every one is stricter.

| # | Frozen requirement | What the judge did | What it does now |
|---|---|---|---|
| C1 | Actual cost ≤ declared ceilings, settled | compared the campaign wall ceiling inline in the judge | calls the production `settle_campaign()` on a `ComputeCost` rebuilt from the artifact's own `device_measured`; the judge and `chowder growth campaign settle` cannot disagree |
| C2 | Both protected mini-slices measured | iterated whatever slice rows existed | requires exactly `math500@2024-04` and `mgsm@2022-11`, once each, candidate-measured, with no undeclared row substituting |
| C3 | Slices pinned by index/seed/decoding | checked nothing | requires `n_samples=16`, indices 0–15, seed 1234, no shuffle, `temperature=0.0 / do_sample=false / max_new_tokens=512`, `prompt_policy=chat_template`, and a named raw artifact |
| C4 | Parent arm is parent evidence | read `parent_score` **out of the candidate's own file** | reads `parent_evaluation.json` (gen1, `MEASURED_PARENT`) as its own provenance-bound `EvalReport`; the candidate file is never authoritative for the parent |
| C5 | Contamination CLEAN on every evaluated benchmark | checked only entries that happened to exist (an empty manifest passed) | coverage is exact over the frozen evaluated set, via the production `MetricBinder` interpretation; a missing benchmark is UNKNOWN, `POSSIBLE`/`KNOWN_CONTAMINATION` is a hard FAIL (TAINTED), and the declared training-source section must exist and be CLEAN |
| C6 | Candidate identity | checked the digest was 64 characters | resolves the artifact, recomputes the canonical digest (`directory_digest` for a tree, `sha256_file` for a file), and fails on mismatch, mutation, or a missing artifact |
| C7 | Paired per-prompt rule or absolute+strict path | compared final rates only | applies the frozen rule exactly, through `statistics.compare` with `min_effect=0.25` and the ≥12/16 strict-improvement requirement |
| C8 | Branch protection | promoted against the unresolved gen1 parent alone | adds the trusted-ancestor arm (gen0) and requires no regression against it, so a gen2 that merely matches an already-regressed gen1 fails |

**Artifact arms.** Three `EvalReport` JSON files, one per measured generation,
parsed by the production `EvalReport`:

- `candidate_evaluation.json` — gen2, rows `MEASURED_THIS_GENERATION`;
- `parent_evaluation.json` — gen1, rows `MEASURED_PARENT`;
- `baseline_evaluation.json` — gen0, rows `MEASURED_PARENT`.

Each carries one `generation-diagnostics@gen2-response-surface-v1` run whose
`metadata.per_prompt` holds the raw completions (the judge scores duplication,
echo, format and correctness itself, so a claimed flag cannot substitute for
evidence), plus one run per protected slice with the protocol metadata of C3.
Prompt identity is a stable `prompt_id`, with the frozen prompt text as the
fallback; a duplicated or unaligned identity refuses to pair.

**Branch rules (unchanged in language, now mechanised).**

- `PROMOTED` — every gate PASS, including T16 (trusted-ancestor protection).
- `INCONCLUSIVE` — any gate UNKNOWN and none FAIL, i.e. missing/insufficient
  evidence, including an unavailable parent or ancestor arm.
- `REJECTED` — any hard failure: target rule failed, protected regression
  against the parent **or** the trusted ancestor, budget overrun, identity
  mismatch, protocol mismatch.
- `TAINTED` — a contamination FAIL.
- Scoped repair stays `INCONCLUSIVE + target_repair_validated`; no new verdict
  class is introduced.

**An unavailable gen1 arm.** If the parent carries no usable protected
measurement, the immediate-parent gate passes **only** when the trusted
ancestor resolved the risk (no regression against gen0); otherwise it is
UNKNOWN. An unresolved gen1 therefore never becomes a trusted protection
baseline by default. This is the "case C" behaviour and is tested.

**Freezing of the instrument.** The judge freezes its own copy of the 16
instrument prompts and their expected answers, and a test asserts that copy
equals the literals in `docs/gen1/run_gen1_cycle.py`, so the instrument cannot
drift silently.

## D. What did NOT change

Unchanged, item by item:

- target thresholds: duplication ≤ 0.125, echo ≤ 0.062, format 8/8;
- protected thresholds: correctness ≥ 15/16, EOS ≥ 0.900, cap < 0.100,
  unclosed ≤ 0.250, loops = 0, trigram ≥ 0.900, slice regression ≤ 0.0625;
- the statistical rule and its constants (`min_effect = 0.25`, ≥12/16 strict);
- the benchmark set, including both protected mini-slice identities;
- every budget ceiling: device 0.30/recipe · 0.60/campaign, wall
  0.75/recipe · 1.50/campaign (admission and settlement units unchanged —
  only *which control owns the device ceiling* is now stated honestly);
- the stopping rules;
- the promotion policy version.

Because the device ceilings keep their admission teeth and the wall ceilings
keep their settlement teeth, the effective evaluation envelope is unchanged:
this amendment costs no additional compute and enlarges no budget.

## E. Consequence for the cycle

The cycle may proceed with its certification boundary hardened: the frozen
judge, the campaign declaration, the runner, the production settlement and
this amendment now say the same thing. A candidate that fails any of A–C
refuses rather than certifying, and the judge cannot certify anything the
production path would refuse.

A full gen2 promotion additionally requires the gen0 trusted-ancestor arm
(one extra 16-item × 2-slice measurement, the same protocol and cost as the
parent arm) so that gen2 is not adjudicated against an unresolved parent
alone. If that measurement cannot be afforded inside the declared envelope,
promotion is INCONCLUSIVE — not assumed safe.
