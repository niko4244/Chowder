# GEN2 prereg amendment 19 — 2026-10-08: T1 renders one row per run (F7, no decision moved)

Additive to the 2026-09-17 prereg and its amendments 1–18. Admissible under the
freeze on the grounds amendments 15–18 stated: **no Gen-2 candidate evaluation
exists** (`evidence/` holds zero `candidate_evaluation.json`, zero
`chosen_candidate.json` and zero `campaign-run.json`, and the declared state root
`runs/2026-09-17-gen2-response-surface/` is not present in this checkout at all),
**and no threshold, comparison or gate moves**. Amendment 18 renamed T1 and
recorded F7 — the duplicate row — as open for the owner rather than smuggling a
row-set change into a rename. This amendment takes that decision and proves it
moves no decision.

## A. The defect, measured

`judge()`'s `ArmError` branch emitted a T1 row carrying the arm's refusal, and
`_instrument_gates`' roster then emitted its own T1 row for the same missing
artifact. On an empty run root the table read, twice:

```text
T1  candidate instrument provenance + row identity  UNKNOWN  candidate artifact candidate_evaluation.json is missing
T1  candidate instrument provenance + row identity  UNKNOWN  candidate evaluation artifact unavailable
```

Amendment 18 unified the name; the row was still there twice. The same shape
exists where the arm *opens*: an arm carrying one benchmark row twice and no
pinned instrument run rendered a FAIL row for the duplicate identity and a
separate UNKNOWN row for the missing run, and an operator had to combine the two
by hand to answer "what does T1 say about this run".

This is **not** the shape the judge's other multi-row gates have. T11 reports
three distinct checks under one id (a per-slice measurement, a regression bound,
and the undeclared-row substitution rule) and T19 reports five (one per arm, plus
provenance), each a different question with its own name and detail. T1's two rows
were one question, split by which branch produced it.

## B. The correction

T1 is now written once per run, from collected findings, with the worst status
winning (`FAIL` > `UNKNOWN` > `PASS`) and every reason in one detail. Measured
after the amendment:

```text
# empty run root
T1  candidate instrument provenance + row identity  UNKNOWN  candidate artifact candidate_evaluation.json is missing

# open arm, one benchmark row twice, no pinned instrument run
T1  candidate instrument provenance + row identity  FAIL     candidate arm duplicates rows for ['math500@2024-04']; no single generation-diagnostics@gen2-response-surface-v1 run carrying MEASURED_THIS_GENERATION

# clean fixture
T1  candidate instrument provenance + row identity  PASS     measurement_origin=MEASURED_THIS_GENERATION
```

`refusal` is a new keyword-only argument of `_instrument_gates`, defaulted to
`None`; every existing caller (including
`tests/test_growth_gen2_instrument_wiring.py`, which calls it directly) keeps
working unchanged, and a call with no refusal still reads `candidate evaluation
artifact unavailable`.

## C. What did not move

The **statuses** a reader can see are unchanged: UNKNOWN where the arm cannot be
read, FAIL where a row identity is duplicated, UNKNOWN where no pinned run exists
(FAIL wins if both hold, which is the same refusal a FAIL row already produced),
PASS on the clean fixture. `branch_verdict` still sees a status set containing the
same values, so PROMOTED / REJECTED / INCONCLUSIVE / TAINTED are computed from the
same information — the row *set* changed, not the decision. No threshold, bound,
normalization, comparison or gate expression was touched.

The row-set change is the point of the amendment and is stated plainly: a run
whose candidate arm cannot be read now renders **one** T1 row where it used to
render two, and a run with two T1 findings renders one row where it used to
render two. Anything that counted T1 rows for such a run counts one fewer; the
judge's own exit code, verdict and per-gate statuses do not change, and no
production surface reads this table's row count.

## D. Proof

Three test cases in `tests/test_growth_gen2_judge.py` (94 in the file, all
passing):

| test | shape | pinned facts |
| --- | --- | --- |
| `test_an_unreadable_candidate_arm_renders_t1_once_with_its_own_refusal` | empty run root | one row, the amendment-18 name, UNKNOWN, and the arm's own refusal naming `candidate_evaluation.json` |
| `test_an_open_arm_with_two_t1_findings_still_renders_one_row` | open arm, duplicated benchmark row, no instrument run | one row, FAIL, both reasons in the detail |
| `test_a_clean_arm_renders_t1_exactly_once` | clean fixture | one row, PASS, `measurement_origin=` — the control against a blanket UNKNOWN |

Both new pins were revert-proven against the exact code they replaced, run from
the working tree and restored byte-for-byte afterwards:

| revert | test | expected | measured |
| --- | --- | --- | --- |
| the unreadable arm's second T1 row restored | `...renders_t1_once_with_its_own_refusal` | fail | fail |
| findings written as one row each again | `...two_t1_findings_still_renders_one_row` | fail | fail |

The judge's other suites were re-run with the change: `test_growth_gen2_judge.py`
(94), `test_growth_gen2_judge_agreement.py`, `test_growth_gen2_instrument_wiring.py`,
`test_growth_settlement_adversarial.py` and `test_growth_certification_coupling.py`
— 143 passed.

## E. What this amendment does not do

- **It does not touch another gate's row count.** T11's and T19's several rows are
  several different checks; the test that pins T1's single row does not assert a
  repo-wide rule.
- **It does not change production's certification.** `certify_protection` in
  `src/chowder/growth/certification.py` still carries its own
  `requirement="candidate measured evidence"` row for its own reader; no test
  couples the two, as amendment 18 recorded.
- **It does not restate F2–F6** (amendment 18's renames) and changes no name. T1's
  name stays `candidate instrument provenance + row identity`.
- **It is not evidence about a candidate.** No Gen-2 candidate evaluation exists;
  nothing here measures a model, and nothing here can certify or refuse more than
  the judge already did.
