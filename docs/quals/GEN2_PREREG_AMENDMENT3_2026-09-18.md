# GEN2 preregistration — Amendment 3 (2026-09-18)

Written before any gen2 compute. It states **what the frozen judge verifies each
row against**, so the evaluator that produces the gen2 arms emits evidence the
judge can actually check. It changes no threshold, no benchmark set, no budget,
no stopping rule, no verdict class and no gate; it removes two ways a gate could
be satisfied without the evidence behind it.

## A. What was missing

The judge verified *declarations* in two places where it had to verify *evidence*:

1. **Contamination (`T12`).** The judge read
   `<run_root>/gen2_contamination_manifest.json` — whatever file happened to have
   that name — and never the artifact the campaign declares as
   `contamination_manifest_path`, which is the manifest the runner's firewall
   actually bound. A campaign pinning `KNOWN_CONTAMINATION` could be certified by
   a different `CLEAN` document sitting in the run root under the expected name.
2. **Protected mini-slices (`T11`).** A slice row was protocol-checked
   (`n_samples=16`, indices 0–15, seed 1234, no shuffle, greedy,
   `max_new_tokens=512`, chat-template prompt, a non-empty `raw_artifact_ref`) but
   the reference was never resolved and no digest was required. A row could name
   a file that does not exist, carry no `artifact_sha256`, declare `n_samples=16`
   with empty `per_sample_scores`, and still satisfy the gate — the same
   declaration-versus-bytes gap that made the candidate-adapter digest
   (`T15`) verifiable in PR #179.

Both are the class of defect this cycle exists to remove: a gate that can be
satisfied by a document rather than by the measurement it claims.

## B. Why it was caught before compute

No gen2 model load, no gen2 training, no gen2 evaluation has happened: there is
no `2026-09-17-gen2-response-surface` run directory, and no gen2 result exists to
condition anything below. The findings came from an adversarial read of the
frozen judge against the runner's own evidence writer, not from any result.

## C. Corrected rule (now the only rule)

**C1. The judged contamination evidence is the campaign's pinned artifact.**
`contamination_manifest_path` is authoritative:

- no pin declared → `T12`/`T18` `UNKNOWN` (`CONTAMINATION_PIN_ABSENT`) — the run
  root's own file is *not* used as a fallback;
- pin declared but absent/unreadable/not a JSON object → `UNKNOWN`
  (`CONTAMINATION_PIN_MISSING`);
- the run root carries no `gen2_contamination_manifest.json` → `UNKNOWN`
  (`CONTAMINATION_EVIDENCE_NOT_IN_RUN_ROOT`);
- the run root's copy differs from the pin by even one byte → `T18` `FAIL`
  (`CONTAMINATION_EVIDENCE_NOT_PINNED`), verdict `TAINTED`;
- the run root's copy is byte-identical to the pin → `T18` `PASS`, and coverage
  is judged from that document over the frozen set (instrument + both
  mini-slices, or the campaign's declared target + protected sets), with the
  training-source section required non-empty and `CLEAN` as before.

The runner already copies the pinned manifest in verbatim, so a real run
satisfies this by construction. In production the frozen manifest pins
`<state_root>/gen2_contamination_manifest.json`, i.e. the run root's own copy.

**C2. Every protected measurement is bound to bytes that exist.** For each
required slice row, in addition to the protocol fields:

- `raw_artifact_ref` must resolve — an absolute path, or a path relative to the
  run root that does not climb out of it — and it must exist
  (`MEASUREMENT_ARTIFACT_MISSING` / `MEASUREMENT_ARTIFACT_ESCAPES_RUN_ROOT`);
- `metadata.artifact_sha256` must be a 64-character lowercase sha256
  (`MEASUREMENT_DIGEST_ABSENT`) and must equal the digest the judge recomputes
  over that artifact's bytes, using the repo's canonical single-file/directory
  digest (`MEASUREMENT_DIGEST_MISMATCH`);
- `per_sample_scores` must carry exactly `n_samples` values whose mean *is* the
  row's `score` (`MEASUREMENT_SAMPLES_INCONSISTENT`) — 16 samples of a 16-item
  exact-match slice, so the aggregate is the samples and not a number beside
  them.

Any failure is `T11` `FAIL` with the named reason above: an unpinned, unhashed,
absent or self-contradictory measurement cannot certify.

**C3. The run carries the measurements it judges.** The campaign runner
materialises the artifacts the declared arms' rows name (relative references,
resolved next to the report that declares them) into the run root at the same
relative path, before any judged artifact is written. A reference that resolves
to nothing refuses the run; two arms that name the same relative path with
different content refuse too. The arm JSON is still copied verbatim — provenance
remains the evaluator's declaration.

**C4. What the evaluator must therefore emit.** For each protected row: the
frozen protocol metadata, the sixteen per-sample values, a `raw_artifact_ref` to
the produced slice artifact, and that artifact's sha256. The gen0 ancestor arm of
amendment 2 carries the same obligations.

## D. What did NOT change

- Thresholds, benchmark sets, the mini-slice protocol, budgets, stopping rules,
  verdict classes, the scoped-repair convention, and amendment 1's device policy
  (`device_time_measured: false`: device ceilings admission, wall ceilings
  settlement).
- Provenance rules: candidate evidence stays `MEASURED_THIS_GENERATION`, parent
  and ancestor arms stay `MEASURED_PARENT`; no gate was loosened to admit an
  unpinned or unhashed artifact, and none is now satisfied by a declaration
  alone.
- `T16`'s requirement that the trusted-ancestor arm exist: an absent ancestor
  report is still `UNKNOWN`, never inherited from the parent.

## E. Consequence for the cycle

A gen2 `PROMOTED` now requires, on top of everything already frozen:
contamination coverage computed from the *pinned* artifact, with the run root
carrying that exact artifact, and every protected slice (candidate, gen1 parent
and gen0 ancestor) bound to an existing artifact whose recomputed digest matches
the row and whose samples add up to its score. `T18` joins the threshold table as
an integrity gate. A run that assembles its arms by hand around a clean file the
campaign never pinned is refused rather than certified — which is the point.

## F. Operator checklist before the gen2 run

1. Keep `contamination_manifest_path` pointing at the run's own
   `gen2_contamination_manifest.json` (the manifest the firewall wrote).
2. Emit the protected and instrument rows with `raw_artifact_ref` +
   `metadata.artifact_sha256` + the sixteen per-sample values (C4).
3. Confirm the declared arm reports live beside the artifacts their rows name, so
   the runner can carry them into the run root (C3).
4. `chowder growth campaign validate docs/gen2/gen2_campaign.json`, then run;
   `docs/gen2/judge_gen2.py <state_root>` reads only the pinned evidence.
