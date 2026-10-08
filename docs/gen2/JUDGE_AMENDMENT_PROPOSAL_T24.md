# Proposal: amendment 17 to the frozen Gen-2 judge (gate T24) — **proposed, not applied**

**Status: PROPOSED.** Not implemented. Amending the frozen judge is the repo
owner's call (the T23 precedent), so this records the defect, the measurement
and the minimal change, and stops there. Every measurement below is reproducible
from the two files it names.

## Where this comes from

`arXiv:2610.00054`, *The First Token Is Not the Verdict* (Villuri, Shaik,
Doboli; 4 Sep 2026), read in full on 2026-10-08 as the literature watch's own
next action (`docs/literature/WATCH_LOG.md`). Its finding is about LLM judges
whose verdict is read from the first generated token's logits: the readout
overstates position bias, because on pairs where the judge would not have led
with a verdict the forced read locks to the first slot (89.7% flip on swap
against 47.5% after generation). Its recommendation is readout disclosure --
report the rate at which the judge leads with a verdict, say which readout
produced the figure, and treat the answer format as an experimental variable.

**That mechanism is not present in this repo.** The Gen-2 judge generates
(`PROTECTED_DECODING`, 512 tokens, greedy) and reads the post-reasoning surface;
the logits here belong to router gate decisions
(`backends/router_healing_eval_worker.py`), which are a tamper fingerprint, and
`router_healing_run.judge_router_healing` is a pass-through to
`gate.evaluate_candidate`. So the paper's numbers transfer nowhere, and its
position-bias metric has no target in Chowder.

What transfers is the class it names -- *the readout is not the conclusion* --
and the disclosure discipline it asks for. That is what this proposal acts on,
at the one row in the judge that has the property.

## The defect, as measured

T5 is `judge_gen2.py:729`: `expected.lower() in
_answer_surface(completion).lower()`. The quantity it computes is **presence**,
and the prereg is honest about that -- it calls the row "answer-correct
(expected string **present** after reasoning)" and "Within-reasoning answer
presence (`answer_correct`)" (`GEN2_PREREG_2026-09-17.md:66,85`). The judge's own
row label is not: it reads **"answer correctness"**, and its detail line names
no readout at all.

Presence is not the conclusion, and the two authorities in this repo already
disagree about the readout for the same completion. Running both
implementations verbatim -- the judge's `_answer_surface` plus containment, and
production's `final_answer` plus `normalize`:

| completion | `judge_gen2` T5 (presence) | production (`evaluators/scoring.py`) |
| --- | --- | --- |
| `Shipping is the process of transporting goods.` vs `ping` | correct | wrong |
| `The task was abandoned.` vs `done` | correct | wrong |
| `Rainbows form when light refracts through droplets.` vs `rain` | correct | wrong |
| `The brain processes sound.` vs `rain` | correct | wrong |
| `There are 17 continents.` vs `7` | correct | wrong |
| `1, 2, 3, 4, 5, 6, 7` vs `5` | correct | wrong |
| `The first three primes are 1, 2, 3.` vs `2` | correct | wrong |
| `Water boils at 1000 degrees Fahrenheit.` vs `100` | correct | wrong |
| `100 divided by 4 is 125.` vs `25` | correct | wrong |
| `Canberra is not the capital of Australia; Sydney is.` vs `Canberra` | correct | wrong |

and the sharpest case, on a completion T10 tolerates at up to 25%: a completion
that opens its reasoning and exhausts the budget before any close marker,
ending `...It is Canberra, I believe.` `_answer_surface` returns the
**reasoning**, presence scores it
correct, and production's `final_answer` returns `''` -- its own docstring: an
unclosed think "means the budget was exhausted mid-reasoning, so there is no
answer yet and the extraction is empty" -- so it scores 0.

Thirteen constructed cases disagree thirteen times, every one the same way --
presence says correct, production says wrong -- and the unclosed-reasoning case
above disagrees in the same direction, from text that holds no answer at all.
The digit expectations collide
mechanically as well: 271 of the 999 integers in `[1, 999]` contain `2`, and
`4`, `5`, `7` each likewise, so any wrong answer drawn from that set scores
correct for the prompts whose expected answer is a single digit.

**What this measurement is, and is not.** These completions were constructed to
be mentions rather than conclusions, so thirteen-of-thirteen measures the
*reachability* of the collision set, not its frequency. The frequency is unknown
here and cannot be borrowed: `evidence/` holds no recorded instrument
completions at all (zero `(expected, completion)` pairs), and the paper's rates
are measured on its judges, prompts and formats. The amendment exists so this
repo measures its own rate on the next instrument run instead of arguing about
it -- which is the disclosure the paper actually asks for.

## Why this is not a taste argument about readouts

- The readout is already a **declared choice** in production:
  `evaluators/scoring.py:score(prediction, expected, scoring)` takes the mode
  (`exact_match`, `normalized_exact_match`, `final_number_match`), and that
  module records that getting it wrong has already cost this repo once -- the
  comma-splitting extractor that split `$70,000` and scored a correct answer
  wrong, after which a self-improvement loop "trained on a failure" that was not
  one.
- The judge's own docstring says it "deliberately owns no second implementation
  of anything the production engine already decides" and lists its delegations
  (report parsing, statistics, resource settlement, contamination). Correctness
  scoring is owned by production, and T5 is a second, looser readout of it that
  no delegation covers.
- T5 carries a `HOLD: must not regress` row at 15/16, so this is not cosmetic: a
  presence count that survives a real capability loss is exactly how a
  no-regression row stops noticing regressions.

## The amendment

### 1. Two named reasons, beside the existing `ACCOUNTING_*` block

```python
#: Why the instrument's presence readout could not be reconciled with
#: production's declared scoring (T24). Presence is what the prereg declares for
#: T5; this row asks only that a count labelled as an answer be reachable as an
#: answer when production reads the same completion.
READOUT_DISAGREES_WITH_PRODUCTION = "READOUT_DISAGREES_WITH_PRODUCTION"
READOUT_UNMEASURED = "READOUT_UNMEASURED"
```

### 2. One gate function, beside `_instrument_gates`

`_readout_disclosure_gate(verdict, candidate)` reads exactly the rows T5 already
reads (`metadata["per_prompt"]`), runs production's `final_answer` and **both**
declared readouts over each completion, and emits one row. It gates only the
irreducible disagreement -- an item presence accepts that *every* declared
readout refuses -- and reports the other direction without gating on it, since
that direction can only refuse more:

| state | T24 |
| --- | --- |
| instrument run, `per_prompt` rows, or the scoring import unavailable | UNKNOWN (`READOUT_UNMEASURED`) |
| an item passes presence while `normalized_exact_match` and `final_number_match` both refuse it | FAIL (`READOUT_DISAGREES_WITH_PRODUCTION`; the detail names the item ids and both readouts) |
| production accepts items that presence refuses | PASS, count in the detail -- a presence false negative can only refuse, so it is disclosed, not gated |
| presence and production agree on every item | PASS |

### 3. T5 stops overstating its own readout

The row label becomes the prereg's own words -- `answer presence >=
{PROTECTED_ANSWER_CORRECT_MIN}/16` in the f-string's text -- and the detail
line names both the readout and its disclosure row. **No threshold moves and
T5's pass/fail is unchanged.** This is the part that satisfies the paper's second recommendation
directly: the figure that gates now says what it measured.

### 4. One wiring line

In `_instrument_gates`, after the T5 row: `_readout_disclosure_gate(verdict, candidate)`,
plus the `("T24", "answer readout disclosure")` entry in the UNKNOWN roster
that a missing candidate arm fills.

### 5. Three documentation lines

- the module docstring's amendment list gains amendment 17 and T24;
- the frozen-policy `INFO` row gains `+ GEN2_PREREG_AMENDMENT17_2026-10-08.md`;
- the instrument paragraph records that T5 measures *presence*, that production
  owns the canonical readout, and that T24 compares the two.

### 6. The bound, declared where thresholds live

`docs/quals/GEN2_PREREG_AMENDMENT17_2026-10-08.md` declares T24's bound, before
the change and in the same series as amendments 1-16. The value proposed here is
**0 disagreements**: T24 is a no-regression sentinel over evidence the run
already recorded, and a bound of zero can only make the judge refuse more, never
certify more. Any larger bound is the owner's declaration to make, and its
consequence is visible in this table rather than hidden in a default.

## Governance

1. **A named prereg amendment**, written before the change, as amendment 16 was:
   `docs/quals/GEN2_PREREG_AMENDMENT17_2026-10-08.md`.
2. **The no-visible-candidate-results rule.** Verified for this proposal:
   `evidence/` holds zero `candidate_evaluation.json`, zero
   `chosen_candidate.json` and zero `campaign-run.json` -- the Gen-2 run refused
   at the `candidate_evaluation` phase, exactly as the T23 record states. No
   candidate results are visible, T5's threshold does not move, and T24 adds a
   declared bound over evidence the run already wrote.
3. **A test that fails without it**, plus one that shows the existing fixture
   cannot see this defect. `tests/test_growth_gen2_judge.py:72` synthesises
   completions through `_completion(...)`, whose wrong answer is the sentinel
   `"definitely-wrong"` -- a string sharing no substring with any expected
   answer, so the pinned suite is blind to the containment channel by
   construction, not by luck. The amendment's test adds a mention-shaped wrong
   answer and asserts three things: T24 FAILs with
   `READOUT_DISAGREES_WITH_PRODUCTION` (unwire the gate and that assertion is
   the ignore-source proof), T24 is UNKNOWN when `per_prompt` is absent, and a
   genuine answer to the same prompt keeps T24 PASSing, so the row is not a
   blanket refusal.

## What the amendment deliberately does NOT do

- **It does not change T5's readout.** Replacing presence with production's
  scoring would re-score the arms that already ran and would contradict the
  quantity the prereg declared. The frozen policy's job is to stay frozen; this
  amendment's job is to make the discrepancy visible and measured.
- **It does not claim presence is the wrong sentinel for this instrument.** T5
  is a capability sentinel inside a surface-compliance instrument, and presence
  is its cheap form. The row asks only that a number labelled as an answer be
  reachable as an answer under production's readout, and that the gap be
  measured on the same evidence.
- **It does not import the paper's rates, metric or mechanism.** The paper is
  cited as class evidence that a readout can mislead an audit while leaving use
  intact; its numbers are its own, and T24's bound is a local declaration to be
  measured here, on this instrument, by this repo.
- **It does not weaken any gate.** T24 can only refuse: a disagreement makes the
  judge refuse more, never certify more. The exit code, the verdict algebra and
  every existing threshold are untouched.
- **It is not applied.** No line of `judge_gen2.py` changes until the owner
  accepts the amendment and writes the prereg entry; this document is the
  proposal, not the change.





