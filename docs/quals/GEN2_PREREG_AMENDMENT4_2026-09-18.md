# GEN2 preregistration — Amendment 4 (2026-09-18)

Written before any gen2 compute. It declares **when** the certification gates run
and **what binds** an evaluation to the model it claims, and it puts the policy
the run certifies with into the campaign declaration so the run and its audit
cannot hold different rules. It changes no threshold, no benchmark set, no budget,
no stopping rule and no verdict class.

## A. What was wrong

The certification gates (T11 protected slices, T16 trusted-ancestor protection,
T17 parent regression) were enforced **only by the frozen judge**, at audit time,
over a run root. The campaign runner's own promotion path — the generic
parent-vs-candidate rule — could record a `PROMOTED` generation in the
`GenerationLedger` *before* any of those gates ran. Concretely:

```
gen0 protected 0.50   gen1 protected 0.00 (unresolved regression)   gen2 protected 0.00
generic campaign rule:  gen2 vs gen1 = no regression  -> PROMOTED, ledger row written
frozen judge:           gen2 vs gen0 = -0.50          -> T16 FAIL, REJECTED
```

Two truths about one run, with the durable one (the ledger) written first. The
gate T16 exists to prevent exactly this, and an after-the-fact audit cannot
un-record a generation.

Two related gaps came with it:

- **Nothing bound an arm to the model it measured.** `chosen_candidate.json`
  verifies the selected artifact and its digest; `candidate_evaluation.json`
  declares `MEASURED_THIS_GENERATION`, a generation label, scores and protocol
  metadata. Nothing required those two documents to describe the same adapter, so
  "adapter A was evaluated" was two independent claims rather than one;
- **generation identity was not checked.** `run_for()` matched a row by
  benchmark id and provenance, not by generation, so a protocol-correct *gen1*
  row carrying `MEASURED_PARENT` could sit in `baseline_evaluation.json` and be
  read as the gen0 arm. Amendment 2 says the ancestor rows must be
  `generation_version=gen0`; nothing enforced it.

## B. Why it was caught before compute

No gen2 model load, no gen2 training, no gen2 evaluation has happened: there is
no `2026-09-17-gen2-response-surface` run directory, and no gen2 result exists to
condition anything below. The finding came from an adversarial read of the runner's
promotion order against the judge's gates.

## C. Corrected rule (now the only rule)

**C1. Certification runs before the lineage record.** The campaign runner applies
the certification mechanism — protected-slice coverage and protocol, branch
protection against the trusted ancestor, and the digest binding below — to the
evidence set the run wrote, and hands the *vetoed* decision to `finalize`. A run
whose certification is `FAIL` is `REJECTED`; a certification that cannot be
decided is `INCONCLUSIVE`; and because `finalize` records a generation only for a
`PROMOTED` decision, no ledger row can claim a generation the certification
refused. The veto is recorded as its own phase (`certification_veto`) beside the
resource veto that already worked this way.

**C2. The mechanism is production's, and the policy is declared.** The
verification and comparison live in `chowder.growth.certification`; the frozen
judge calls the same functions and keeps only its frozen policy values. The
campaign manifest declares that policy under `protection`:

```json
"protection": {
  "trusted_ancestor_version": "gen0",
  "slice_regression_max": 0.0625,
  "n_samples": 16,
  "seed": 1234,
  "shuffle": false,
  "decoding": {"temperature": 0.0, "do_sample": false, "max_new_tokens": 512},
  "prompt_policy": "chat_template"
}
```

A campaign that declares no `protection` policy cannot certify, and therefore
cannot promote. An undeclared key inside `protection` refuses at load; a partial
one refuses at load. The shipped `docs/gen2/gen2_campaign.json` declares the
values above, and the judge's gate **T20** fails if a declaration and the frozen
constants ever disagree — so the run's policy and the audit's policy are the same
numbers or the audit refuses.

**C3. Every arm names the bytes it measured.** An evaluation report carries
`model_identity`, and certification requires:

| arm | must name | checked against |
| --- | --- | --- |
| candidate | `adapter_digest` | the digest of the artifact the run selected (`chosen_candidate.json`) |
| parent | `adapter_digest` | the declared `parent_adapter_digest` |
| ancestor | `base_model_digest` | the declared `base_model_digest` |

A report that names nothing is `UNKNOWN` (undecided evidence); a report that names
a different model is a hard `FAIL` (`ARM_ADAPTER_DIGEST_MISMATCH` /
`ARM_BASE_DIGEST_MISMATCH`). This is judge gate **T19**, and the runner applies the
same rule through the same code, so a candidate evaluation that measured something
other than the selected adapter cannot be certified by either path.

**C4. An arm's generation is part of its identity.** A row must carry the
provenance *and* the generation its role declares (`gen2` for the candidate,
`gen1` for the parent, `gen0` for the trusted ancestor). A protocol-correct row of
the wrong generation is refused (`ARM_GENERATION_MISMATCH`).

**C5. Recipe accounting is exact.** Judge gate **T14** compares the recipes in
`cycle_compute_accounting.json` against the campaign's declared recipe set: a
declared-but-unaccounted recipe or an accounted-but-undeclared one is a failure,
not merely "at least two recipes were counted".

## D. What did NOT change

- Every threshold, benchmark set, the mini-slice protocol, the budgets, the
  stopping rules, the verdict classes and the scoped-repair convention.
- Amendment 1's device policy, amendment 2's gen0 ancestor declaration, and
  amendment 3's evidence verification (pinned contamination manifest, artifact-
  bound measurements with recomputed digests).
- The requirement that the candidate arm is *candidate-measured*: the digest
  binding is added to provenance, it does not replace it.

## E. Consequence for the cycle

A gen2 `PROMOTED` now requires, in this order: the declared protection policy and
the declared evidence inputs; a real run; settlement inside the envelope; a
certification that the required protected slices are candidate-measured,
protocol-exact, bound to real bytes and to the selected adapter's digest, that
the candidate holds against both the gen1 parent and the gen0 trusted ancestor,
and that the accounting covers exactly the declared recipes; and only then the
mechanical promotion decision and its append-only lineage record. The remaining
gap for a *real* gen2 run is measurement, not policy: a candidate evaluation
produced by the run for the artifact it selected, and the input documents the
manifest still has to declare. Both are declared here as requirements rather than
assumed by a default.

## F. Operator checklist before the gen2 run

1. Keep `protection` in the campaign declaration equal to the frozen values
   (the judge's T20 refuses a divergence).
2. Evaluate the selected candidate and record the artifact digest it measured in
   the report's `model_identity` (`adapter_digest`); declare the parent and gen0
   arms with their own digests.
3. Confirm the arms' rows carry the generation their role names.
4. `chowder growth campaign validate docs/gen2/gen2_campaign.json`, then run
   `chowder growth campaign run`; a certification `FAIL`/`UNKNOWN` means no
   promoted generation is recorded, whatever the promotion rule said.
