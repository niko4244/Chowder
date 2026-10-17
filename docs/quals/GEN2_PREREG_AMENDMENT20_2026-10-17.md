# GEN2 prereg amendment 20 — 2026-10-17: the candidate's in-run evaluation runs at the declared throughput

> **Status: APPLIED 2026-10-17.** PR #229 merged as `7cf99ea`, and steps D2–D6
> ran. The evidence is in §F. The Gen-2 run itself has not started (§E).

This adds to the 2026-09-17 prereg and its amendments 1–19 and supersedes none
of them.

## A. The defect, measured on the 2026-09-20 run

`campaign-run.json` in the state root (written 2026-09-20 13:20 from
`prepared-v8`) records `verdict: REFUSED`, `refused_by: candidate_evaluation`,
`CANDIDATE_EVALUATION_NOT_PRODUCED`, `attempts: []`. These are the twelve
attempt directories under `attempts/`:

| attempts | outcome |
|---|---|
| a01–a04 | training evidence written, `candidate_succeeded` unset, no failure reason recorded |
| a05–a08 | preflight refused: profiled lifecycle cost (0.8 / 1.05 GPU-h) did not fit the remaining budget |
| **a09, a10, a12** | **training succeeded** (adapter written), then `TimeoutError: evaluation … exceeded timeout` |
| a11 | `transformers worker failed with exit code 3221225477` (`0xC0000005`, access violation) — a separate defect, not addressed here |

How the timeout happened (a12):
- The in-run eval spec had `batch_size: 1` on every suite, `placement: offload`,
  bf16. The 9B in bf16 (~18 GB) exceeds the 16 GB card, so layers were
  offloaded to CPU.
- `generation-diagnostics`: 16 rows took **79 minutes** (11:54 → 13:13).
- `math500` had written **0 rows** when the 5,400 s timeout fired.
- The parent arm (`parent-arm/ancestor-eval-spec.json`) ran the same three
  suites × 16 rows at **`batch_size: 16`** and finished in 3,706 s.

Cause: `_write_project_template` in `src/chowder/growth/campaign_prepare.py`
built each in-run suite without `batch_size`, so the transformers-text worker
defaulted to 1. The template's own comment states the in-run evaluation
"measures exactly what an arm measures … at the declared batch size". It did
not.

This was an **instrument violation** as well as a speed problem. The module
docstring of `tests/test_growth_evaluation_execution.py` records that batch-16
and single-row decoding agree on only 14 of 16 Math500 rows on the frozen
Gen-0 base. A batch-1 candidate arm is therefore not comparable to batch-16
parent and ancestor arms, whatever its timing.

**It is still present in the READY declaration:**
`prepared-v10/project-template.json` has no `batch_size` in any suite.
Readiness does not detect this, so a run on `prepared-v10` would reproduce the
refusal.

## B. The correction (PR #229, commit `e97e1c1`)

- Every template suite now carries
  `"batch_size": int(manifest.evaluation_execution.batch_size)`. That is the
  single declared value the arms and `SubprocessEvaluationFn` already read.
- `test_the_in_run_evaluation_uses_the_same_declared_throughput[1|16]` failed
  with `KeyError: 'batch_size'` before the change and passes after.
- The prepare, readiness, runner, evaluation-binding and training-binding
  suites pass: 129 tests.

## C. Admissibility

- **No Gen-2 candidate evaluation exists.** The state root holds 0
  `candidate_evaluation.json` and 0 `chosen_candidate.json` (checked
  2026-10-17).
- **Disclosure:** amendment 19 stated "zero `campaign-run.json`". That was true
  of its checkout, which lacked the state root. The state root itself holds the
  2026-09-20 refusal record described in §A. That record contains no candidate
  measurement (`attempts: []`, `candidate_metrics: {}` in every attempt), so it
  makes no candidate result visible.
- **No threshold, gate, slice, decoding parameter or comparison moves.**
  `evaluation_execution.batch_size: 16` is already declared in
  `docs/gen2/gen2_campaign.json`. This amendment makes the candidate arm obey
  it, as the arms already did.
- **The in-run timeout is unchanged** at 5,400 s. The parent's 3,706 s at
  batch 16 fits with ×1.46 margin, but only on an uncontended GPU; see §E.

## D. Execution steps (not yet run)

**D1. Wait for PR #229 to merge.** Then run the full suite on main.

**D2. Archive the 2026-09-20 run record** so its accounting cannot mix with
the next run's (needs owner approval):

```
<state_root>/campaign-run.json             -> <state_root>/superseded-2026-09-20/
<state_root>/cycle_compute_accounting.json -> <state_root>/superseded-2026-09-20/
```

`attempts/a01–a12` stay where they are as evidence. The next run's attempt
numbering must not collide with them; check what `campaign run` does with an
existing `attempts/` directory before running.

**D3. Prepare into a new directory** (`prepared-v11`). Never overwrite
`prepared-v10`, which is the evidence for amendment 14:

```
PYTHONPATH=src python -m chowder.cli growth campaign prepare \
  docs/gen2/gen2_campaign.json \
  --out-dir "C:/Users/nikma/Chowder-Protected/runs/2026-09-17-gen2-response-surface/prepared-v11" \
  --parent-evidence "C:/Users/nikma/Chowder-Protected/runs/2026-09-17-gen1-protocol-compliance" \
  --parent-measurement "C:/Users/nikma/Chowder-Protected/runs/2026-09-17-gen2-response-surface/prepared-v10/parent-eval-report.json" \
  --write-declaration "C:/Users/nikma/Chowder-Protected/runs/2026-09-17-gen2-response-surface/prepared-v11/gen2_campaign.prepared.json"
```

The parent arm is **carried forward, not re-measured**: the three
`MEASURED_PARENT` gen1 rows of 2026-09-19 (0.5625 / 0.0 / 0.0). Their
`artifact_sha256` and `slice_sha256` must equal `prepared-v10`'s.

**D4. Verify before declaring:**
- `growth campaign readiness …/prepared-v11/gen2_campaign.prepared.json` →
  `READY`, all seventeen checks `ok`, no reason codes.
- Every suite in `prepared-v11/project-template.json` →
  `config.evaluation.suites[*].batch_size == 16`.
  Readiness does not check this, so it must be checked directly.
- `prepared-v11` vs `prepared-v10`: the only intended difference is the suites'
  `batch_size`. Diff every other file and explain any other change (paths that
  embed the directory name are expected).

**D5. Declare.** Point the eight `prepared-v10` paths in
`docs/gen2/gen2_campaign.json` at `prepared-v11`, and append this amendment's
tag to `notes`. Commit with the D4 evidence.

**D6 (owner decision, optional): base model on NVMe.**
- `base_model_path` names `F:/llm-models/…-bf16`. F: is a USB HDD that measured
  ~2.2–6.9 MB/s on 2026-10-16.
- A copy at `C:/llm-models/Qwen3.8-9B-abliterated-25-bf16` verifies to the same
  `chowder.model-content.v1` digest, `59e767aab1da…`. All four shard sha256s
  also match their Hugging Face etags.
- Identity is content-based, so changing the path moves no identity. If
  accepted, change `base_model_path` in the same declaration commit as D5, and
  re-run D4 against it.

## E. What this amendment does not do

- **It does not start the run.** `chowder growth campaign run` remains a
  separate decision. It needs an exclusive GPU window: no `llama-server` and no
  Ollama on the card, since the 5,400 s in-run budget assumes the parent arm's
  uncontended conditions.
- It does not re-measure the Gen-1 parent arm or the Gen-0 ancestor arm.
- It does not address a11's access violation. If it recurs, it is a failed
  attempt under the campaign's own rules and should be investigated on its own.
- It changes no judge code, and it is not evidence about a candidate.

## F. Applied — evidence (2026-10-17)

- **D1.** PR #229 merged (`7cf99ea`) with all six CI jobs green: Linux py3.10,
  3.11 and 3.12, Windows py3.12, the real transformers PEFT CPU smoke, and
  lint/build.
- **D2.** `campaign-run.json` (sha256 `9160efd72ded…`) and
  `cycle_compute_accounting.json` (`b5479f03d7f9…`) were moved, hashes
  unchanged, to `<state_root>/superseded-2026-09-20/`. `_reserve_attempt`
  takes the smallest unused `attempt-NN`, so the next run starts at
  `attempt-13` and never reuses a01–a12.
- **D6 (accepted).** `base_model_path` is now
  `C:/llm-models/Qwen3.8-9B-abliterated-25-bf16`.
  `model_content_digest` = `59e767aab1dabceef098ceb367d872ef735114412383ae8e622a95c64297555f`,
  and all four shard sha256s equal their Hugging Face etags.
- **D3.** `prepare` wrote `prepared-v11` in 24 s. The corpus is unchanged
  (10,340 examples, 79,904 tokens, CLEAN, verifier pass 1.0).
- **D4.**
  - Every `project-template.json` suite has `batch_size: 16`.
  - The `slices/` hashes are identical to `prepared-v10`.
  - These files are identical once directory names are normalised:
    `contamination`, `corpus-quality`, `data-registry`,
    `evaluation-material`, `parent-profile`, `training-material`.
  - Every remaining difference is explained:
    - `project-template`: only the three `batch_size` fields.
    - `parent-eval-report`: only the `date` stamp. Rows, scores and digests
      are carried forward.
    - `hardware-budget`: a fresh device probe, so timings differ.
    - The prepared declaration: `base_model_path` (D6), `notes`, and
      `retention_profile` values `-0.0625`. The last is amendment 15's sign
      fix; `prepared-v10`'s prepared declaration predates it, while the repo
      declaration already carried it.
  - Readiness on both `prepared-v11/gen2_campaign.prepared.json` and
    `docs/gen2/gen2_campaign.json`: **READY**, nineteen of nineteen checks
    `ok`, no reason codes. The base digest took 43 s from NVMe; it took
    321 s from F: in amendment 14.
- **D5.** `docs/gen2/gen2_campaign.json` names `prepared-v11` for all eight
  run-phase inputs, and its `notes` carry this amendment's tag.
