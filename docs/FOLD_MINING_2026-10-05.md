# Fold mining: PR #203's rescued experiments into registered intervention families (2026-10-05)

**What this is.** PR #203 (`fold/main-clone-rescue-2026-10-03`) carries 87 files
rescued from the stale main clone. This change mines that work: the experiment
mechanisms and their tests enter `main` by **file-level extraction**
(`git checkout <fold-branch> -- <path>`), and each experiment registers as an
`InterventionFamily` in `src/chowder/growth/interventions.py` carrying the
artifacts it ships and the measurements behind its maturity label. The fold
branch is not merged, not rebased, and nothing depends on it.

## Why the mining is file-level, not a merge

The fold's base is `2369d37`; `main` is **26 commits ahead** of it. The fold's
12 *modified* files are diffs against that older main, and the fold's own
description calls the worker/evaluator edits "superseded-by-evolution risk". A
merge would be a lane for those hunks to land silently against files that have
since changed for other reasons. New-to-main files carry no such risk, so the
mining takes those, one experiment at a time, and leaves the stale diffs behind
with the reasons recorded below.

Landed files are **byte-identical** to the fold branch:

```
$ for f in <landed paths>; do git diff --quiet origin/fold/main-clone-rescue-2026-10-03 -- "$f" || echo "$f"; done
(no output)
```

## What the registry looked like before, and why nothing failed

Four families already cited rescued modules as their basis --
`architecture.conditional-ffn` (`conditional_compute.py` +
`conditional_profile.py`), `architecture.hybrid-lm`
(`experimental_hybrid_lm.py`), `compression.low-rank-vocab` (`low_rank_*`), and
`compression.ptq` ("kaggle QAT lane tests") -- and **none of those files existed
on `main`**. Two more families, `inference.retrieval` and
`inference.speculative`, declared themselves "no in-repo implementation yet;
proposal only" while the rescued work contained working, tested
implementations of both.

The labels were claims no test could check. This change makes them checkable
and then satisfies them.

## Mined

| family | label | mechanism landed | basis that was actually measured |
| --- | --- | --- | --- |
| `architecture.conditional-ffn` | research | `conditional_compute.py`, `conditional_profile.py`, `exp_c_profile.py`, 2 test files | fold rescue; EXPERIMENT_C |
| `architecture.hybrid-lm` | research | `experimental_hybrid_lm.py`, `exp_d_hybrid_lm.py`, `examples/experiment_d/configs/*`, 2 test files | fold rescue; EXPERIMENT_D |
| `compression.low-rank-vocab` | research | `low_rank_vocab.py`, `low_rank_checkpoint.py`, `low_rank_convert.py`, `low_rank_real_eval.py`, test | fold rescue; LOW_RANK_VOCAB_EXPERIMENT (flat spectrum -> unacceptable degradation) |
| `compression.ptq` | research | `exp_f_ptq_margin.py`, `kaggle/run_qat_distill_lane.py`, 3 shipped evidence records, 2 test files | exp_f: margin shift +0.0076 while accuracy 0.3 -> 0.0 |
| `inference.retrieval` | research | `exp_e_corpus.py` (BM25 / dense / learned-sparse), 2 test files | EXPERIMENT_E phase 3: factual +0.50, citation rate 1.00; learned-sparse not competitive at this corpus size (198 KB vs 21 KB) |
| `inference.speculative` | research | `exp_e_speculative.py`, `exp_e_spec_llamacpp.py` | EXPERIMENT_E phase 2: up to 2.8x tok/s, identical outputs, ~4% overhead elsewhere |
| `inference.confidence-routing` | **rejected** | `exp_e_confidence.py`, test | EXPERIMENT_E phase 4: routed 0.55/0.57 vs always-large control 0.72; confident-and-wrong 3 of 14 |
| `runtime.harness-repair` | research | `runtime_eval.py`, `run_runtime_harness_compare.py`, `batch009_harness_experiment.py`, the trace->reward tooling and the batch-004..009 trace records, 2 test files | batch-009 traces; EXPERIMENT_E phase 5 |
| `runtime.harness-evolution` | research | `harness_evolution.py`, test | fold rescue; RRSI-style regularized selection |
| `training.teacher-distillation` | research | `exp_b_teacher_data.py`, `exp_b_restricted_python.py`, `exp_b_granite_baseline.py`, `exp_b_toolchain_check.py`, test | fold rescue; EXPERIMENT_B |

The one REJECTED family is the point of the exercise as much as the research
ones: `inference.confidence-routing` enters with the measurement that rejected
it as its basis, is absent from every permitted set, and returns only through an
explicit reopen naming a NEW confidence signal (`reopen: {family_id:
hypothesis_id}`). A rescued experiment whose own data says "do not do this" is
registered as such, not as another research candidate.

## Landed without a family, and why

The mining maps artifacts to families wherever a family owns them, but a few
landed files are not any family's mechanism:

- `docs/quals/P11_RUNG5_PREREG_2026-09-16.md` and its
  `docs/quals/judge_rung5_2026-09-16.py` are a governance pair from a different
  lane. The judge compiles and imports the existing `docs/quals/quals_harness.py`;
  nothing collects or runs it. Landed as a record, not as a live surface.
- `campaign/phase_a/train_pilot.jsonl` and `eval_pilot_exactness.jsonl` are
  orphan pilot rows: 24 SFT messages and 10 prompt/expected pairs that no code
  in the tree reads. They came with the rescue and are kept visible rather than
  silently dropped; they are inputs for a manual pilot, not a family's evidence.
- The five `examples/experiment_d/configs/*.json` (plus their registry) are
  cited by `architecture.hybrid-lm` as the config directory that holds them.
- The Experiment B/C/D/E and low-rank write-ups under `docs/`, and the
  `evidence/` records, are cited by the families that measured with them.

## What enforces the labels now

- `InterventionFamily.implementation` names the in-repo artifacts behind a
  family (modules, experiment drivers, tests, evidence records).
- `__post_init__` refuses an implementation path that is absolute or escapes
  the repo with `..`: a claim cannot point outside the tree it audits.
- `tests/test_growth_interventions_evidence_hypotheses.py` refuses a registered
  family that declares no mechanism, refuses any declared artifact that does
  not exist, and refuses prose that disagrees with provenance (a family that
  ships artifacts may not still call itself a proposal).
- The measured claim is load-bearing, not decorative:
  `test_the_measured_ptq_record_underwrites_the_quantization_label` reads the
  shipped exp_f record and asserts the flat-margin/collapsed-accuracy pattern
  the family's note is about.

Revert proof, measured: move `src/chowder/low_rank_vocab.py` and
`evidence/exp_f_ptq_margin_qwen25_1p5b_int8sq_guided20_20260926.json` out of the
tree and both guards fail; restore them and the file passes (26 passed).

## Deferred, deliberately

The remaining 5 new files and all 12 modified files stay with PR #203. Each
deferral has one reason: the slice needs a rebase against `main`'s evolved
worker, and nothing in this mining pretends those hunks are current.

| not mined | why |
| --- | --- |
| `backends/transformers_worker.py` (+352/-22), `evaluators/*`, `backends/transformers_peft.py`, `tests/test_reward_training_cpu.py` | the reward-training slice; `repair_only` appears nowhere in `main`'s `src`, so the lane has no runnable path here |
| `build_repair_reward_data.py`, `batch007_repair_only_reward_train.jsonl` | the repair-only reward lane: the builder reads `batch006_repair_replay_train_text.jsonl`, which is not in the repo, and calls a machine-local tokenizer at module import; its output has no consumer until the deferred trainer exists |
| `adapter_bundle.py`, `tests/test_repair_only_peft.py` | the test passes on `main` (measured: 1 passed in 23.9s) but exercises raw PEFT layout, not the bundle manifest; the bundle's only consumer is the deferred reward slice |
| `gate.py`, `models.py`, `project.py` (`runtime_reward_min`, `runtime_nonexistent_read_rate_max`) | only tested by the deferred reward slice; landing them would put an untested runtime veto into the promotion gate |
| `config_validation.py` keys, `adapter_guard.py` `.repair.` normalization | same reward/repair-only slice |
| `pyproject.toml` | mined partially: the `ptq` extra and the `numpy` / `rank_bm25` dev dependencies are required by landed tests; no other hunk |

## Verification

- Mined suites: **163 passed** in 44.21s
  (`test_conditional_compute`, `test_conditional_profile`,
  `test_experimental_hybrid_lm`, `test_low_rank_vocab`,
  `test_runtime_harness_mechanisms`, `test_exp_e_confidence`,
  `test_exp_e_pipeline`, `test_exp_e_run`, `test_exp_f_ptq_margin`,
  `test_kaggle_qat_lane`, `test_batch010_contract`, `test_exp_b_teacher_data`,
  `test_exp_d_hybrid_lm`).
- Registry: 13 families, 73 declared artifacts, every path present.
- `ruff check src tests` (the CI gate, `select = [E9, F63, F7, F82]`): clean.
- Light-CI import surface, measured by re-running the mined suites with
  `torch`/`transformers`/`peft`/`datasets`/`modelopt`/`safetensors` refused at
  import time (what the three `.[dev]`-only CI jobs have): **56 passed, 10
  skipped, 0 collection errors**. The optional-dependency guards are the fold's;
  the two dev dependencies the suites need directly (`numpy`, `rank_bm25`) are
  declared in the mined `pyproject.toml` hunk.
- Extended lint (`--select F401,F811,F841`), which is **not** part of the CI
  gate: 14 findings (unused imports and two unused locals) inside the extracted
  files, left as-is so that every landed file stays byte-identical to the fold
  branch -- the provenance property is worth more than cosmetic lint, and the
  findings disappear with the rebase the deferred slices need anyway.
- Full suite on this branch: **2973 passed, 77 skipped, 0 failed** in
  700.83s (`FULL_EXIT=0`). The same suite measured 2803 passed on `main`
  earlier in this worktree, so the delta is exactly the 163 mined tests
  plus the 7 new registry tests -- the mining changed no existing result.
