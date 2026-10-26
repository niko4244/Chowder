# Source-bound Q8/random128 HumanEval pre-execution admission — October 26, 2026

**Original Gen-0 decision remains `BLOCKED`, `chosen:null`,
`release_approved:false`.** This module executes no model-generated code,
does not contact Kaggle, and cannot replace genuine isolated HumanEval replay
or a Windows-equivalent benchmark. Existing frozen evaluator/scorer and all
sixteen immutable historical Kaggle snapshots remain unchanged.

## New, separate source-only preflight

Production module: `chowder_batch/reap_humaneval_private_preflight.py`.
The source-only CLI accepts only `reap128_v2_q8` and the separately
constructed `random128_v2_q8`. It requires the immutable final original
remote native receipt, the previously completed **independent full-cohort
external audit** and original full HumanEval row stream from the **same
32-hex serving session**, plus the original canonical HumanEval dataset,
source-pinned evaluator and original scorer. It:

- Revalidates the native original GGUF/model/source/suite identities and
  actual measured postload resource admission from the source receipt,
  including original four-way CUDA geometry and independent completed
  full-population audit provenance.
- Requires the external full auditor's **previously attested row hash**
  to match the exact private original 164-line JSONL bytes. It never
  establishes trust merely by hashing an unverified row file against itself.
- Verifies that the native machine receipt SHA is the source SHA from the
  earlier independent audit, with exact model, serving session, original
  GSM8K 1,319 + HumanEval 164 counts and row hashes, and untouched
  `human_eval_independent_execution=PENDING...` pending state.
- Independently requires the exact original `humaneval.json` raw-file SHA
  `5144d31cf337b25ce262b780ed103b128b1e9d289d3c0d15f4fe34933bb705d5`,
  164 unique canonical IDs and valid `header`, `test`, `entry_point`
  strings, with original frozen evaluator and scorer SHA.
- Rejects missing/duplicate/foreign IDs, mixed serving sessions, impossible
  correctness + unterminated rows, wrong exact row SHA, manipulated full
  audit model/resource/CUDA/selection fields, incomplete populations and
  inconsistent correct/terminated aggregates.
- Produces **only SHA hashes, model/source/session identifiers, summary
  counts and a canonical-ID ordered correctness/termination boolean digest**
  in a new private exclusive `x`-opened JSON file. No generated program
  text, canonical test source, credentials or original weights are copied
  into the certificate, standard output or Git.

Invoke only after remote results have *actually finished* and
`reap_external_full_audit` has independently validated them:

```powershell
python -B -m chowder_batch.reap_humaneval_private_preflight `
  --arm reap128_v2_q8 `
  --machine '<EXCLUSIVE_PRIVATE_COMPLETED_NATIVE_JSON>' `
  --trusted-full-audit '<EXCLUSIVE_PRIVATE_INDEPENDENT_FULL_AUDIT_JSON>' `
  --humaneval '<EXCLUSIVE_PRIVATE_ORIGINAL_164_JSONL>' `
  --canonical-humaneval '<ORIGINAL_CANONICAL_HUMANEVAL_JSON>' `
  --original-evaluator '<PINNED_ORIGINAL_REAP_EVAL_PY>' `
  --original-scorer '<PINNED_ORIGINAL_SCORING_PY>' `
  --out '<NEW_EXCLUSIVE_PRIVATE_PREFLIGHT_JSON>'
```

For `random128_v2_q8`, use its *distinct* native receipt, independent audit,
original 164 rows and separately exclusive output. Do not reuse the historical
Q4 source/session/hash or assume the candidate's 151/164 Q4 result is the Q8
or random control score.

## Security boundary / remaining execution blocker

The preflight's successful status is deliberately
`PREFLIGHT_PRIVATE_164_SOURCE_BOUND_SANDBOX_REQUIRED`, **not** a scoring
PASS. It explicitly records `candidate_code_executed:false` and
`sandbox_verified:false`. No subprocess, Python `exec`, model, GPU,
service interruption, kernel submission or generated-code evaluation is
performed.

Actual later independent HumanEval execution requires a **separate** verified
private CPU-only sandbox with positive network, mount and process-tree
isolation (including descendant cleanup), nonroot credential drop, bounded
CPU/RAM/PIDs/files, and executable source/runtime provenance. The earlier
Q4 private low-privilege notebook used Python `-I`, conditional UID drop
and `setrlimit` with 12s timeout, but this does **not** prove filesystem
or network namespace isolation or descendant cleanup. Kaggle notebook
metadata `enable_internet:false` and `enable_gpu:false` are useful but
cannot by themselves prove those kernel security boundaries. If a disposable
harmless sandbox probe cannot establish the constraints, explicitly refuse
code execution and leave the independent replay pending.

The preflight's input audit is source-bound but not digitally signed;
cryptographic hashes do not establish who produced a forged upstream JSON.
Verify the exact private Kaggle kernel version, inputs, native output
session/log and independent audit before accepting it. Neither this source
admission nor eventual Linux replay establishes original Windows quality
or CUDA numerical parity.

## Verification

New CPU-only source-level adversarial tests exercise both original arms,
full 164 original task identities, mismatch and tamper cases, resource/native
attestation, no-code-execution output, and fail-closed nonexistent/linked
source paths. Latest scoped run: **23 passed, 1 Windows symlink-creation
test skipped, 1 existing pytest configuration warning**; Ruff clean.
The broader CPU-only adjacent scientific matrix afterward passed **587
tests**, with the same **1 Windows symlink test skipped**, **1 intentional
preregistration-related deselection**, and **5 warnings**.

**Still unresolved:** Both original private full Q8 and random128 runs were
RUNNING at the most recent live check; no new raw result exists to admit.
Fresh dense reference qualification, full speed ratio, REAP160 speed,
BF16 CUDA parity, Windows/Linux equivalence and optional prospective paired
CI governance are not certified. No release, PR merge, model changes,
deployment or Gen-1.
