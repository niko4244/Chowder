# GEN2 prereg amendment 17 — 2026-10-08: the answer readout disclosure (judge gate T24)

Additive to the 2026-09-17 prereg and its amendments 1–16. Admissible under the
freeze on the same grounds amendments 15 and 16 stated: **no Gen-2 candidate
evaluation exists** (`evidence/` holds zero `candidate_evaluation.json`, zero
`chosen_candidate.json` and zero `campaign-run.json`, and the declared state root
`runs/2026-09-17-gen2-response-surface/` is not present in this checkout at all),
**and no threshold moves** — T24 compares two readouts over rows the run already
wrote, and its bound (0 irreducible disagreements) can only make the judge refuse
more, never certify more.

## A. The defect, measured

T5 computes *presence*: `expected.lower() in _answer_surface(completion).lower()`.
The prereg declares exactly that quantity for the row — "Answer-correct (expected
string **present** after reasoning)" (§1) and "Within-reasoning answer presence
(`answer_correct`)" (§2, ≥ 15/16). The judge's own label said **"answer
correctness"** and its detail named no readout, so an operator reading the table
saw a correctness figure that no correctness readout produced.

Correctness is not a private judgement call here; production owns it and declares
the mode. `chowder.evaluators.scoring.score(prediction, expected, scoring)`
exposes `exact_match`, `normalized_exact_match` and `final_number_match`, and its
`final_answer` reads the answer after the last `` response`` and returns `""` for
an unclosed ` thinking` — its own words: "the budget was exhausted mid-reasoning,
so there is no answer yet and the extraction is empty".

Running both readouts verbatim over constructed completions, thirteen cases
disagree thirteen times and always in the same direction — presence accepts what
every declared readout refuses:

| completion | expected | T5 (presence) | production |
| --- | --- | --- | --- |
| `Shipping is the process of transporting goods.` | `ping` | correct | wrong |
| `The task was abandoned.` | `done` | correct | wrong |
| `Rainbows form when light refracts through droplets.` | `rain` | correct | wrong |
| `The brain processes sound.` | `rain` | correct | wrong |
| `There are 17 continents.` | `7` | correct | wrong |
| `1, 2, 3, 4, 5, 6, 7` | `5` | correct | wrong |
| `The first three primes are 1, 2, 3.` | `2` | correct | wrong |
| `Water boils at 1000 degrees Fahrenheit.` | `100` | correct | wrong |
| `100 divided by 4 is 125.` | `25` | correct | wrong |
| `Canberra is not the capital of Australia; Sydney is.` | `Canberra` | correct | wrong |
| ...`\n\nThere are 17 continents.` after a closed think | `7` | correct | wrong |
| unclosed reasoning: "the French for good morning is 'bonjour', I am fairly sure." | `bonjour` | correct | wrong |
| unclosed reasoning: "it is Canberra, I believe." | `Canberra` | correct | wrong |

The digit expectations collide mechanically as well: of the 999 integers in
`[1, 999]`, 271 contain `2`, and 271 contain `4`, `5` and `7` likewise; `25` is
contained in 20 of them, and `100` and `391` in one each (themselves).

**What this measurement is, and is not.** These completions were constructed to
be mentions rather than conclusions, so thirteen-of-thirteen measures the
*reachability* of the collision set, not its frequency. The frequency is unknown
here and is not borrowed from anywhere: no Gen-2 instrument completions exist in
this checkout. T24 is written so the next instrument run measures this repo's own
rate instead of arguing about it.

The sharpest shape is the unclosed reasoning budget: production refuses it
outright (there is no answer surface), and T10 tolerates up to 25% of it
(`PROTECTED_UNCLOSED_THINK_MAX = 0.250`), so a run can legitimately carry
completions whose "answer surface" is the reasoning itself.

## B. The gate: T24

One function, `_readout_disclosure_gate(verdict, candidate)`, over the rows T5
already reads (`metadata["per_prompt"]`) and nothing else. It re-scores nothing:
it runs both readouts production declares for text scoring
(`PRODUCTION_READOUT_MODES = ("normalized_exact_match", "final_number_match")`;
`exact_match` is absent because it cannot accept anything those two refuse)
through production's own `score`, imported through one function so that an
unavailable readout is UNKNOWN rather than a judge that cannot run.

| state | T24 |
| --- | --- |
| no candidate arm, no per-prompt rows, or production's readout unavailable | `UNKNOWN` — `READOUT_UNMEASURED` |
| an item passes presence while **every** declared readout refuses it | `FAIL` — `READOUT_DISAGREES_WITH_PRODUCTION`; the detail names the item ids and both readouts |
| presence refuses an item a declared readout accepts | `PASS`, counted in the detail — this direction can only refuse more, so it is disclosed, not gated |
| presence and production agree on every item | `PASS` |

The bound lives where thresholds live, and is declared here before the change:

```python
PROTECTED_READOUT_DISAGREEMENT_MAX = 0  # of the candidate's per-prompt rows
```

T24 is emitted by its own function in every branch — including the branches where
no candidate arm or no per-prompt rows exist — rather than listed in the
`_instrument_gates` UNKNOWN rosters, so every T24 row carries its reason code.
That is the one deviation from the proposal's sketch
(`docs/gen2/JUDGE_AMENDMENT_PROPOSAL_T24.md` §4, which put the row in the roster
that a missing candidate arm fills); the intent is unchanged and the roster form
cannot name the reason it is UNKNOWN.

## C. What did not move

T5's readout, its threshold (`>= 15/16`), its pass/fail, and every other
threshold are unchanged. `branch_verdict`, the exit code and T1–T23 are
untouched. The T5 row label now reads the prereg's own words
(`answer presence >= 15/16`) and its detail names the readout and the disclosure
row. The count itself is the same expression as before, extracted into one helper
(`_answer_present`) that T24 also calls, so the row that counts and the row that
audits the count cannot drift into two readings of the same evidence.

## D. Proof

- Eight new tests in `tests/test_growth_gen2_judge.py` (84 in the file, all
  passing), over the instrument's own rows through the file's existing
  `_prompt_entries`/`_run_root` fixtures rather than a bespoke fixture:
  - `test_a_mention_that_passes_presence_cannot_certify` — T24 FAIL, item and
    both readouts named in the detail, with T5 still PASS on the same rows;
  - `test_the_readout_gate_is_what_refuses_a_mention` — **the unreachable-source
    proof, kept in the suite**: with the gate replaced by a no-op the same root
    returns 0 and certifies, so T24 is the only thing refusing it;
  - `test_an_unclosed_reasoning_budget_is_not_an_answer_surface` — the sharpest
    shape, with T4 and T5 unmoved;
  - `test_a_presence_miss_production_accepts_is_disclosed_not_gated` — `1,00` for
    `100`: the direction that is reported and not gated, on a run that still
    certifies;
  - `test_a_run_without_per_prompt_evidence_is_unknown_not_a_pass` and
    `test_an_unavailable_production_readout_is_unknown` — both fail-closed;
  - `test_the_pinned_fixtures_wrong_answer_cannot_reach_the_channel` — why no
    assertion written before this amendment could see the defect: the pinned
    fixtures' wrong answer is `"definitely-wrong"`, which shares no substring
    with any expected answer, so T5 fails on that root while T24 passes;
  - `test_a_genuine_answer_keeps_the_disclosure_row_passing` — the control.
- Re-measured immediately before this amendment: 13 of 13 constructed mentions
  disagree, 271 of 999 integers contain each single-digit expectation.

## E. What this amendment does not do

- It does not change T5's readout. Replacing presence with production's scoring
  would re-score arms that already ran and would contradict the quantity the
  prereg declared. T5 stays a capability sentinel; T24 makes the gap visible and
  measured.
- It does not claim presence is the wrong sentinel for this instrument, and it
  does not import the paper's rates (`arXiv:2610.00054`, cited in the proposal)
  — the paper is class evidence that a readout can mislead an audit while leaving
  use intact; T24's bound is a local declaration, measured here, on this
  instrument.
- It does not weaken any gate. T24 can only refuse: it is a disclosure row, it
  certifies nothing, and a candidate whose readouts agree is unchanged.
- It does not make production authoritative over T5. Where the disagreement is
  permissive, T24 refuses; where production is the more permissive readout, T24
  discloses rather than gating.
