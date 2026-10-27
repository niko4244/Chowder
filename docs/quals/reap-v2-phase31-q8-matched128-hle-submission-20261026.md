# Phase 31 — Q8 matched HLE research: source-bound Kaggle submission

**Status at 2026-10-26 (US Central evening):** original REAP128_v2
Q8_0 HLE matched-128 Kaggle notebook **RUNNING**, not scored yet.
The original scientific Gen-0 selector remains `BLOCKED`, `chosen:null`,
`release_approved:false`. This is an independent research protocol, not
official HLE judging or a source for original frozen Windows model selection.

## Exact same 128 HLE questions as completed Q4, not a new random sample

The Q4 HLE Kaggle run `nikmarco/reap128v2-q4-hle-textonly-20261026`
completed 128 original text-only question IDs from a 2,158-item population
(planned 600 before time-budget truncation), **3/128** correct with a
non-official deterministic normalized-match heuristic. Its original source
file is bound privately by SHA-256
`717ac6a3cd190e7c66bb53f48b50e2b2a9be4485239b5fe629c182b18145ca4c`,
the Q4 128-row JSONL SHA-256 is
`3cdc96a324ef3fb4d2fa6f0a7a30ed83c6ec7084cca8b654a66469e2b9f1ceeb`,
and the sorted unique ID-set digest is
`65a8f15e747c2a00577aa0a5c5c06ec7724e4c31ec27a99c6a416c3f587acbea`.
Question texts, answer keys and user-private dataset files are *not*
committed or reproduced here.

The actually **executed** Q4 Kaggle notebook was pulled from Kaggle rather
than reusing an older local draft. This mattered: its real HLE source
protocol is **13,041 bytes** at SHA-256
`cbacf0bb2215fce041ca62e991002d2772bee04891bc5e1773d3b0e2c44c49b2`,
and its content manifest is SHA-256
`ebdb19e818257c21335889f9bea51c9939293444d315ba0211e21d2c599e70d8`.
The mounted private Kaggle protocol dataset was independently downloaded,
checked against that byte count and hash, and found identical.

This pinned protocol uses the text-only official `cais/hle` test data at
SHA-256 `6d0ee0602e8aea6b159509577e884f48ecac7b8e3f6822a35f51335a446c726a`;
exact same question strings, temperature 0, seed 0, max 8192 output tokens,
two concurrent model requests, and native `--fit on -np 2 -c 16384 --jinja`.
No official HLE LLM-as-judge is invoked. The identical protocol is a stricter
paired **research** comparison than drawing new unrelated items; it is not a
certified official HLE leaderboard evaluation.

The Q8 model must equal **19,748,435,968 bytes** with SHA-256
`89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6`.
The same source-pinned original Linux llama.cpp binary SHA, two-T4 native
admission and explicit observed 41/41 CUDA layer offload are required before
cohort execution. Conservative runtime physical RAM and cgroup floors,
GPU headroom checks, immutable read-only post-load clean file-cache advisory
and loopback-owned server teardown prevent unsafe eager model loads.

## Private Kaggle job and honest operational history

Private URL: https://www.kaggle.com/code/nikmarco/reap128v2-q8-matched128-hle-textonly-20261026

Version 1 **refused before loading Q8** due to a stale protocol source
digest inherited from a draft notebook. It was **not** a failed inference
or an accuracy score. Version 2 used the independently recovered exact
executed Q4 source/protocol and was accepted by Kaggle; most recent
status **RUNNING**. No Q8 correctness claim or completed output file was
available at that check. Do not report a paired score before downloading,
checking and independently auditing all 128 Q8 rows.

Private source and downloaded receipts are under:

`F:\reap-v2-q8-matched128-hle-private-20261026-v3`

Private notebook source SHA-256:
`46816ed05d95e3171f4eb8153fda49ddfd832359e1ad245f6b8a936b69572828`.

The experimental generator is
`chowder_batch/reap_q8_matched_hle_kaggle_20261026.py`.
It reads the locally authenticated original Kaggle/Q4 files but commits
**none** of the 128 actual question IDs or result contents. Its source
verification fails closed on altered originals or an already-used output
directory.

## Independent paired replay review

`chowder_batch/reap_q8_q4_matched_hle_audit_20261026.py` is a read-only
128-row comparator requiring identical source IDs, row integrity, unique
native session per arm, both complete/graded/untruncated cohorts, official
HLE judge explicitly unavailable, model hashes, identical protocol source
and manifest hashes, no Q8 item errors, and exact summary row hash.
It reports only `both_correct`, `q8_only_correct`, `q4_only_correct`,
`both_incorrect`, observed accuracy and paired difference, not model
completions or private questions.

Failure to find an actual Q8 output must return
`BLOCKED_UNVERIFIED_HLE_PAIR`. No source/scorer tampering or invented
successful GPU sessions may turn that into a score. Current original
scientific selection and protected PR #282 remain unchanged.

Only HLE is in progress for Q8 among the requested 11 frontier categories.
No Q8 AutomationBench, Terminal-Bench, DeepSWE, ProgramBench, NL2Repo,
Agents' Last Exam, ApexBench, Chartography, BabyVision, or ZeroBench
frontier score has yet been verified. Historical full original Q8 GSM8K
and HumanEval are separate baselines, not substitutes for these tasks.
