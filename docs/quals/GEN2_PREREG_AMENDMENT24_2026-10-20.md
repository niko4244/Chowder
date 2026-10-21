# GEN2 prereg amendment 24 — 2026-10-20: the ancestor arm's raw artifacts live beside its report

Adds to the 2026-09-17 prereg and amendments 1–23 and supersedes none of
them. **This is an evidence-placement fix, not a quality gate.** No slice,
threshold, retention bound, comparison, decoding parameter, recipe, step count,
ceiling or prepared input moves. The declaration still names `prepared-v16` and
the same `baseline_eval_report_path`, unchanged.

## A. What run 10 did and why it was refused (defect #10)

Run 10 launched 2026-10-19 16:20 from main `59579a5` (prepared-v16; readiness
READY 19/19). Attempts 30 and 31 trained (2.16 GPU-h measured), then the run was
refused at `certification_evidence`:

> the ancestor arm's math500@2024-04 row names raw artifact
> 'predictions-math500.jsonl', which does not exist next to
> …\2026-09-18-gen2-gen0-arm\gen0-baseline-evaluation.json

- The ancestor (Gen-0) arm was measured on 2026-09-19 into the campaign's
  `ancestor-arm/` (`predictions-math500.jsonl` 01:20, `predictions-mgsm.jsonl`
  01:43). The wrapped report `gen0-baseline-evaluation.json` was written four
  seconds later into a separate directory, `2026-09-18-gen2-gen0-arm/`, with
  relative `raw_artifact_ref`s that resolve beside it, where the files never were.
- Certification correctly refuses evidence it cannot hash. No earlier Gen-2 run
  reached certification with the raw-artifact check (#237/#238) in place, so the
  layout had never been tested.

## B. The fix

1. **Evidence placement (no code).** The two files were copied from
   `ancestor-arm/` beside the report. Their sha256s equal the report's declared
   `artifact_sha256` exactly:
   - `predictions-math500.jsonl` `42750d176a68bf604401b4de6547f76580820e7754ae9e64e02c20760ea69e19`
   - `predictions-mgsm.jsonl` `5d4755a08d5a67576a1c0eb7dd1c8ab53d780358b91d684b051e97ecf1936cbe`

   Nothing in the report changed; the certification check is unchanged and as
   strict as before.
2. **Earlier detection (PR #266).** Readiness now resolves the parent and
   ancestor arms' raw artifacts with the same resolver certification uses, so a
   missing artifact refuses before any compute. Verified on this campaign: with
   the files absent, readiness reported REFUSED with the defect #10 message.

## C. Rerun

Run 10's state is archived as `superseded-2026-10-20-run10/` (`campaign-run.json`,
`cycle_compute_accounting.json`, `evaluation/`, README). Run 11 launches from
main on the same declaration once readiness is READY. The P1 stop rule is
unchanged: after three more infrastructure refusals, Gen-2 stops for a design
review of the campaign runner.
