# GEN2 prereg amendment 18 — 2026-10-08: the row-label audit applied (F2–F7, no decision moved)

Additive to the 2026-09-17 prereg and its amendments 1–17. Admissible under the
freeze on the same grounds amendments 15–17 stated: **no Gen-2 candidate
evaluation exists** (`evidence/` holds zero `candidate_evaluation.json`, zero
`chosen_candidate.json` and zero `campaign-run.json`, and the declared state root
`runs/2026-09-17-gen2-response-surface/` is not present in this checkout at all),
**and no threshold moves**. This amendment is stronger still: it changes **no
comparison, no bound, no gate and no branch rule** — only the string an operator
reads. Where amendment 17 could at least refuse more, this one cannot change a
single pass/fail, and §D proves that by measurement rather than by assertion.

It applies the corrections `docs/gen2/GATE_LABEL_AUDIT_2026-10-08.md` raised
against the frozen judge's own row names and one detail string: F2 (inverted
polarity word), F3 (T21 named a profile it does not read), F4 (T13's "actual
cost"), F5 (T9's missing "mean"), F6 (T1 named provenance for a row that also
refuses duplicate row identity), and F7 (an unreadable candidate arm rendered T1
twice, under two different names).

## A. The defect, measured

### F2 — the refusal row named the opposite direction

T2 and T3 are lower-is-better rates. `_target_gate` decides with
`comparison.verdict == "regressed"` meaning *improved*, but its detail printed
production's verdict verbatim, and `compare()` answers in a higher-is-better
vocabulary. Measured on the judge's own fixtures, before and after:

```text
before  T2  answer-duplication target  FAIL  paired=improved (delta +0.6875, min_effect 0.25); candidate rate 0.688 vs parent 0.000; ...
after   T2  answer-duplication target  FAIL  paired not better (production's compare() answers 'improved' in a higher-is-better vocabulary; delta +0.6875, min_effect 0.25); candidate rate 0.688 vs parent 0.000; ...
```

That candidate made duplication **worse** — 0.688 against a 0.000 parent — and
the row told the reader `paired=improved`. The same string on the passing path
read `paired improvement — paired=regressed`, where `regressed` is the word for
*the improvement the rule wants*. The corrected detail names the row's own
direction first (`paired better` / `paired not better`) and keeps production's
word beside it with its vocabulary named, so the trace to the function that
produced it survives and the polarity can no longer be read wrong:

```text
after   T2  answer-duplication target  PASS  paired improvement — paired better (production's compare() answers 'regressed' in a higher-is-better vocabulary; delta -0.6875, ...)
```

### F7 — an unreadable candidate arm rendered T1 twice, under two names

On the empty run root, `judge()`'s `ArmError` branch emitted T1 under the name
`candidate measured evidence` while `_instrument_gates`' roster emitted its own
T1 row for the same missing artifact under `candidate instrument provenance`:
two rows, one threshold, two names for the same fact. Measured after this
amendment, the names are one — and the duplicate row is still there:

```text
T1  candidate instrument provenance + row identity  UNKNOWN  candidate artifact candidate_evaluation.json is missing
T1  candidate instrument provenance + row identity  UNKNOWN  candidate evaluation artifact unavailable
```

The duplicate row is **not** removed here. T1, like T11, T12 and T19, is a
threshold that reports more than one fact on purpose, and changing the number of
rows a root emits is a behaviour change, not a rename: `branch_verdict` reads the
row set, and collapsing the pair would silently move what an operator counts.
F7 is therefore recorded as an open finding for the owner, and the new
`test_the_unreadable_candidate_arm_uses_one_t1_name` pins **both** facts — one
name, still two rows — so neither the stale second name nor a quiet merge can
drift by.

## B. The corrections

| row | before | after | sites |
| --- | --- | --- | --- |
| T1 | `candidate instrument provenance` / `candidate measured evidence` | `candidate instrument provenance + row identity` | 5 (roster, duplicates-FAIL, no-run UNKNOWN, PASS, unreadable-arm UNKNOWN) |
| T9 | `distinct-trigram >= {min}` | `distinct-trigram ratio mean >= {min}` | 2 (decision row + roster) |
| T13 | `actual cost settled within the declared ceilings` | `cost settles within the declared ceilings` | 5 (decision row + 4 UNKNOWN branches) |
| T21 | `the run's promotion decision agrees with the declared profile` | `run decision on the declared retention profile` | 1 (requirement string) |
| T2/T3 detail | `paired={verdict}` | `paired better` / `paired not better` + production's word with its vocabulary named | 1 expression |

Two of the audit's proposed wordings are longer than the renderer's name column
and are shortened here to fit it. `Verdict.render()` in
`docs/quals/quals_harness.py` formats each row as
`f"{threshold:<10} {name:<46} {status:<8} {detail}"`, so a 47-character name
shifts the status and detail columns and makes the table hard to read
mechanically. The audit proposed `candidate instrument provenance and row
identity` (48) and `the run's decision agrees on the declared retention profile`
(59); the applied forms are 46 and 43 characters. The 46-character limit is why
the longest row name in the judge is exactly 46 — checked in the rendered table
(§D), not by counting alone.

Nothing else changed. The audit's F3–F6 findings are applied with the shortened
wordings; their substance is unchanged, and the audit record is updated to say so.

One cross-reference this rename breaks on purpose: production's certification
carries the row `requirement="candidate measured evidence"` with the detail
`candidate arm unavailable` (`certify_protection` in
`src/chowder/growth/certification.py`) — the exact wording of the judge's old
unreadable-arm T1 name. The judge's T1 is not that row: it also refuses duplicate
prompt identity and reports `measurement_origin`, which is why it gets one name
that says so while the specifics stay in each row's own detail. The certification
artifact keeps its own wording for its own reader, and no test couples the two;
F7 in the audit records the divergence.

## C. What did not move

Every threshold, every comparison, every gate, `branch_verdict`, the exit code
and the row set are byte-identical in behaviour. The only expression touched in a
decision path is the detail string of `_target_gate`; `paired_improved` already
existed and still decides exactly as it did. On the clean fixture all six rows
still PASS and the judge returns 0; on the gen1-shaped defect fixture T2 and T3
still FAIL and it returns 1; on the empty root T13 is still UNKNOWN under the new
name. The agreement test that reads a refused run's table
(`tests/test_growth_gen2_judge_agreement.py`) keeps its T13 assertion under the
new name, which is what makes the rename visible in a second table rather than
only in the file that defines it.

## D. Proof

Six new test cases plus one control in `tests/test_growth_gen2_judge.py` (92 in
the file, all passing), and each rename was reverted in place to prove the pin
catches it — the revert measurement, run against the working tree and restored
byte-for-byte afterwards:

| revert | test | expected | measured |
| --- | --- | --- | --- |
| T1 name, all 5 sites | `test_the_row_names_say_what_the_rows_measure[T1]` | fail | fail |
| T9 name | `...[T9]` | fail | fail |
| T13 name, all 5 sites | `...[T13]` | fail | fail |
| T21 name | `...[T21]` | fail | fail |
| T13 name (UNKNOWN branch) | `test_the_settlement_row_names_itself_the_same_way_when_it_cannot_decide` | fail | fail |
| one T1 site left as `measured evidence` | `test_the_unreadable_candidate_arm_uses_one_t1_name` | fail | fail |
| F2 polarity sentence | `test_the_target_detail_names_the_direction_in_the_rows_own_terms` | fail | fail |
| every rename and F2 reverted | `test_the_renames_moved_no_decision` | **pass** | pass |

The last row is the load-bearing one: with all five names and the F2 detail
reverted to their pre-amendment text, the clean root still certifies and the
defect root still refuses on T2 and T3. No decision moved — measured, not
asserted.

The renames are also pinned by name in the rendered table, not only in the source:
`_row_name(output, threshold)` re-reads the row out of the same rendered string an
operator sees, so a name that fits the source but breaks the 46-character column
fails the test. With T1's applied name the rendered table is:

```text
T1         candidate instrument provenance + row identity PASS     measurement_origin=MEASURED_THIS_GENERATION
T9         distinct-trigram ratio mean >= 0.9             PASS     measured 0.97; parent 0.973
T13        cost settles within the declared ceilings      PASS     device 0.4000 (unmeasured) / wall 1.1000 against the campaign envelope
T21        run decision on the declared retention profile PASS     run verdict PROMOTED; declared-gate breaches none
```

## E. What this amendment does not do

- **It does not merge T1's duplicate row** (F7). That changes the row set, and the
  owner decides it; the pin keeps it from happening silently.
- **It does not change `compare()`.** Production's higher-is-better vocabulary is
  its own contract, used by every caller; the judge labels the polarity instead of
  asking production to speak a second dialect.
- **It does not touch T5 or T24** (amendment 17's rows), and it restates no
  threshold. The renames are the audit's findings F2–F6 only.
- **It is not evidence about a candidate.** No Gen-2 candidate evaluation exists;
  nothing here measures a model, and nothing here can certify or refuse more than
  the judge already did.
