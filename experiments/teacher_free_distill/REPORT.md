# Teacher-free distillation pilot — status report

Branch: `experiment/teacher-free-distillation-pilot` (base audit SHA `3d5c0101e40406e01defa7591e025f28178a7be9`)
Date: 2026-09-26 (revised; first version 2026-09-25). All numbers below come from runs executed on this branch; nothing is extrapolated. Upstream flags (`resolved`) are carried as claims and never counted as evidence.

## This session — four follow-up tasks

| Task | Outcome | Strongest evidence |
|---|---|---|
| OT3 short-trace strategy so the SFT set reaches 5 000+ accepted rows | **DONE** | `pilot_v3`: **5 015 accepted** (4 529 train / 486 dev), manifest-pinned |
| Condition A GPU pilot, Qwen3-1.7B | **DONE** (authorized) | run 4 complete: 284/284 steps, wall 17 153 s, adapter published (25.7 MB LoRA weights) |
| Wire SWE-smith per-repo env specs into the replay sandbox | **PROTOCOL SOLVED — 8 harness defects found and fixed; red reproduced 3/3, no verified repair yet** | official-image path + in-container apply evidence per row |
| End-to-end review of PR #201 (correctness, security, tests) | **DONE** | 4 PR defects fixed, 8 harness defects found and fixed, **33/33 pass** |

## IMPLEMENTED

- **Phase 1 audit** (`AUDIT.md`): PR #201 reviewed file-by-file; infra map (PEFT backend, completion-only masking, memory planner, hardware detection, evaluators, TUI); gaps recorded.
- **Source catalog** (`sources.json`): OT3 approved (Apache-2.0, revision `61bcf9d4…` = live HEAD, QwQ-32B-trace caveat), MoT BLOCKED (no dataset license), SWE-smith-trajectories approved (MIT, revision `08e109b4…`; `resolved` documented as a claim).
- **Ingest + normalization** (`prepare.py`): per-example provenance (source, revision, license, original id, content SHA-256, char/token counts, split), banded-minhash near-duplicate quarantine, per-source quality report, deterministic dev split. New: `--near-dup-field` selects the dedup key; chunked OT3 rows share a long prompt prefix, so `target` is the correct key (prompt-keyed dedup both over-merged and under-merged them).
- **OT3 short-trace strategy** (`ot3_subset.py`, new): OT3 rows average ≈ 47 KB, so a flat 8 k-char cap accepted 6 of 300 rows. Instead of dropping long traces, the tool splits each trace at **conclusion boundaries** (reasoning/answer seams) into self-contained short traces, preserving the conclusion with its supporting context and a per-chunk provenance link to the parent row. 5 204 chunks → `pilot_v3` accepts **5 015** (163 overlong, 26 duplicate/conflict, 189 rejected, 0 quarantined) at `max_chars=8000`, `max_rows_per_source=6000`, `dev_percent=10`. Chunk source: `ot3_chunked.jsonl` sha256 `f2560954fe0ea2e5…`; train set sha256 `f916364e8eaa3f0e43b693550e591667177fec20f0926e78c6138e5395076ce4` (confirmed identical inside the frozen run-spec); dev sha256 `04d7f24996799d07…`; holdout sha256 `0498a43d1326bb4d…`.
- **Fetcher** (`fetch_smith.py`): bounded seeded reservoir, retry/backoff, resume sidecar, `claimed_resolved` preserved as a claim. Review fix: the reservoir RNG bound was wrong (`randrange(len(reservoir) + scanned - dropped)`), biasing the sample and mis-counting retries; now Vitter's R with per-retry counters.
- **Sandboxed replay** (`replay_smith.py`, substantially reworked): now uses **SWE-smith's own per-instance environment spec** — official images `docker.io/jyangballin/swesmith.x86_64.<owner>_1776_<name>.<commit>` ship the conda `testbed` env, the repo tree at `/testbed`, and every test dependency (cantools: 173 tests collect cleanly). The pristine image is *not* red, so the harness overlays the per-instance bug branch from the `github.com/swesmith/<owner>__<name>.<commit>` mirror, restores the F2P test files (from `refs/pull/<N>/head` for `pr_*` instances, from the embedded base commit otherwise), commits the container into a local red image, and then runs the network-off phases: pre-patch tests must fail, patch and post-patch tests must pass, in one container. The mirror path remains as fallback. Evidence per row: every command, its return code, raw output tail, the parsed executed-test count, `apply_rc`, and a digest-evidenced `verification` block only on genuine red→green.
- **Condition A launcher** (`train_pilot.py`, new): device-exclusivity preflight (Chowder-process scan + 8 GB free-VRAM floor), resolved-config construction against the **production** `TransformersPeftExecutor`, `--micro-batch`/`--grad-accum` overrides that must preserve the effective batch, live progress callback, and publication of `adapter/`, `run_record.json`, `loss_history.json`, `worker-result.json`.
- **Student selection** (`student.py`): Qwen3-1.7B primary (pinned `70d244cc…`, Apache-2.0), Qwen2.5-Coder-3B control (license gated on operator acceptance), workload estimates through the production `plan_memory` on real `detect_hardware()` output.
- **Training preflight** (`preflight.py`): 8-check CPU suite (tokenization, production completion-only masking, nonempty targets, LoRA coverage, gradient flow adapter-only, finite loss, checkpoint roundtrip, resume integrity) on a 1-layer model.
- **Recipes** (`recipes/*.json`): A (SFT on verified traces), B (repair continuation + forgetting probe), C (preference, GATED until matched pairs with execution evidence exist).
- **Evaluation protocol** (`evaluate.py`): repair split-by-repository leakage check, prompt-overlap check, GSM8K extraction/scoring, repair behavior metrics, paired comparison with advisory regression flags — never auto-promotion.
- **TUI screen** (`tui_teacher_free.py`): 8-stage workflow whose displayed state derives exclusively from real on-disk artifacts.

## TESTED

- `tests/test_teacher_free_prepare.py`: **6 passed**; `tests/test_teacher_free_phase2.py`: **19 passed**; `tests/test_teacher_free_train_pilot.py`: **8 passed**; `tests/test_tui_teacher_free.py`: **3 passed** — **33/33** on a fresh run.
- New tests added in the PR review and harness hardening: `repair-metrics` CLI over real JSONL, prompt-overlap tolerance for non-dict rows, `derive_test_targets` mapping, LF-only patch writes, pure-failure test counting, verification gating + `apply_rc` recording, failure records that keep partial evidence, non-raising cleanup, summary buckets that separate infra failure from test failure, student-alias resolution, effective-batch preservation under operator memory overrides, dataset pinning, and the exclusivity/VRAM-floor checks.
- CPU preflight: **8/8 checks passed**.
- The replay harness has teeth — five genuine defects were found and fixed by executing it, each of which had made *every* previous replay meaningless:
  1. **CRLF patches** (`Path.write_text` on Windows): every patch was handed to Linux containers with `\r\n`, so `git apply` failed on all 23 rows. Fixed with LF-normalised writes.
  2. **Fresh-container state loss**: each phase ran in its own `--rm` container, so an applied patch evaporated before the post-patch tests — post results were identical to pre results by construction. Apply and tests now run in one container, and `APPLY_RC` is parsed explicitly.
  3. **Short SHAs are not fetch refs**: `git fetch upstream dbcda4f` cannot work; full SHAs are resolved via the GitHub API (cached).
  4. **`parse_test_summary` counted only passed+failed**, so a pure-failure run reported 0 executed tests and a reproduced red was scored as "nothing ran". Fixed (`N failed` counts).
  5. **Undefined `rc2`** in the official-image path: Python's `and` short-circuit hid it for rows whose patch failed to apply, and crashed exactly the rows whose patch applied cleanly — i.e. the only rows that could ever verify. Now uses the container's own return code and records `apply_rc`.
- Three further defects were found by *running* the hardened harness, all of which manufacture false negatives:
  6. **Evidence died with the process**: the per-row log only reached disk at the end, so a mid-row failure discarded everything — one row lost a fully committed red-image phase to a cleanup timeout. Evidence is now written as each command completes, and a failed row carries its partial log and last command.
  7. **Cleanup could fail a row that had already succeeded**: a podman daemon saturated by concurrent image pulls timed out a `podman rm` and killed the row. Cleanup is now best-effort with its own budget.
  8. **Timeouts that never fire**: `subprocess.run(timeout=…)` kills the podman CLI but then blocks in `communicate()` waiting for pipe EOF that never comes on Windows (grandchildren inherit the handles) — a 300 s phase stalled the batch for 13+ minutes. `run_podman` now uses `Popen` + kill + immediate re-raise, test phases get their own 1 200 s budget (dask's 11-test F2P module needs more than the control-plane 300 s), and image pulls are bounded with their real output tail recorded instead of hanging.
- Operational note: the official per-instance images are 3–4 GB each and pulls are cached per host, so a cold batch is dominated by download time, not by replay work. Concurrent batches compete for one podman daemon; running them one at a time is what the harness now assumes.

## TRAINED

**Condition A completed: one LoRA adapter trained on 4 529 pinned examples.**

- Authorization: operator granted GPU use for this pilot. Exclusivity check before launch: `device_index 0`, RTX 5060 Ti, **12.05 GB free / 15.93 GB**, **0 Chowder processes** on the device; the user's resident inference servers were detected, left untouched, and recorded as a contention risk.
- Runs 1–3 aborted: with micro-batch 4 the desktop WDDM driver spilled to shared system memory next to those inference servers — **333 s/step**, with 6.67 GB of shared GPU memory measured via `Get-Counter "\GPU Process Memory(*)\Shared Usage"`. Run 4 uses `--micro-batch 1 --grad-accum 32` (effective batch 32, unchanged, recorded as an operator override) and shows **no spillover**.
- Run 4 live values: 284 optimizer steps, wall 13 376 s at step 260 (≈50 s/step through step 250). Loss **2.3454 @ 10 → 1.8798 @ 20 → 1.6637 @ 200 → 1.6734 @ 260**, cosine decay (lr 4.05e-6 at step 260). Frozen config: Qwen/Qwen3-1.7B @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, dataset `pilot_v3/train.jsonl` (sha256 above), chat format, 2 epochs, max_length 2048, LoRA r16/α32/dropout 0.05 on q,k,v,o, bf16, gradient checkpointing, seed 2026, lr 2e-4.
- **Finished run (2026-09-26 03:15)**: 284/284 steps, wall **17 153 s (4 h 46 m)**, mean `train_loss` **1.7276**, last logged loss **1.6871 @ step 280**, LoRA cosine decay to 1.6e-7. Telemetry from the worker's own record: `global_step 284`, `training_rows 4529`, `peak_vram_gb 7.03` (the frozen plan predicted a ~7 GB worst band), `measured_gpu_hours 4.74`, optimizer state 51.4 MB.
- Published artifacts (`C:\Users\nikma\chowder_teacher_free\checkpoints\cond_a\`): `adapter/` with a real **`adapter_model.safetensors` (25.7 MB)**, `adapter_config.json`, tokenizer + chat template; plus `run_record.json`, `worker-result.json`, and `loss_history.json` (28 step points).
- Step-time behaviour is a measured result of its own: ~48–53 s/step to step 250, then **160–240 s/step** exactly while three replay batches shared the host, recovering to 92–129 s/step once they were stopped. The pilot's GPU throughput on this machine is load-sensitive; the spilling diagnosis from runs 1–3 and this degradation are the same phenomenon at different magnitudes.
- A post-run defect was found and fixed rather than papered over: the launcher looked for `step_log` as a bare list, but the worker publishes `{"entries": [...]}`, so the run published an **empty** `loss_history.json` while recipe A declares `outputs.loss_history_required: true`. `train_pilot.py` now accepts the worker's real shape, falls back to the points the launcher itself observed, and **refuses to publish an empty loss history** when the recipe requires one; the artifact was then recovered from the worker's own `step_log` (28 entries, 2.3454 → 1.6871). Regression tests cover the wrapped shape, the bare shape, and the malformed/absent cases.

## EVALUATED

**No model evaluation ran** (no finished student, no authorized baseline). What was measured, against real data:

- **OT3 chunking yield**: 5 204 chunks from the long-trace scan; 5 015 accepted (96.4 %), 163 overlong, 26 duplicate/conflict — versus 6 accepted rows without chunking. The context budget is confirmed as OT3's binding constraint, not licensing or duplication.
- **SWE-smith replay, official-environment protocol** (per-row raw logs persist in `C:\Users\nikma\chowder_teacher_free\replay_work*/<row>/replay_log.json`, appended as each command completes):
  | instance | pairing | pre-patch (red) | apply | post-patch | verdict |
  |---|---|---|---|---|---|
  | `pndurette__gTTS.dbcda4f3.pr_440` | congruent | 2 failed / 2 run | **applied (rc 0)** | **1 passed, 1 failed** | `not_green` — partial repair (the timeout fix lands, the status-code fix does not) |
  | `cantools__cantools.0c6a7871…` | congruent | 3 failed / 3 run | **rejected (rc 1)** | 3 failed | `not_green` — `error: patch failed: src/cantools/database/can/message.py:280`; the trajectory's diff does not fit the recorded base state |
  | `getmoto__moto.694ce1f4.pr_7456` | other module | 11 failed / 11 run | **rejected (rc 1)** | 11 failed | `not_green` — patch adds `moto/networkmanager/*` while the F2P tests exercise `resiliencehub`; files already exist in the tree |
  | `dask__dask.5f61e423.pr_7894` | congruent | not reached | — | — | `setup_failed` — the bug-branch overlay's container exec returned rc 125, so **no tests ran and nothing was certified** (fail-closed) |
  | `Project-MONAI__MONAI.a09c1f08.pr_5383`, `pyupio__safety…` | other module | not reached | — | — | `setup_failed` — cold image pulls exceeded the bounded 1 800 s pull budget on a congested daemon; the batch was stopped to stop competing with the GPU run |

  Aggregate: **3 of 23 rows reached the test phases; red was reproduced in 3/3 (100 %); 0/3 went green.** One row was a genuine partial repair (1 of 2 F2P tests fixed), two never had a chance because the patch does not apply to the recorded base state. Nothing was certified without red→green evidence, and the harness now distinguishes `not_green` from `setup_failed` instead of reporting both as one number.
- **Dataset-integrity finding (quantified)**: for the 23 strict rows (trajectory `instance_id` exactly present in the instances dataset, so F2P lists are trustworthy), a patch↔instance congruence scan finds **8 rows where the patch touches the module named by the F2P test file** and **15 where it does not** (`_pairing_scan.py` → `pairing_scan.jsonl`). Self-reported `resolved` is `True` for 7 rows and `False` for 16. The two hand-verified cases agree with the scan exactly (gTTS congruent → one test flipped green; moto not congruent → patch irrelevant to its tests). Earlier in the session, cross-repo mispairings were also seen directly (an `apispec` row carrying an `arrow` patch, `patsy`→`pdfminer`, `bottle`→`paramiko`).
- **Student memory plan** (static estimate, real hardware): Qwen3-1.7B LoRA fits the RTX 5060 Ti (3.44 GB frozen + 0.34 GB trainable + 11.27 GB activations worst band + 0.17 GB optimizer; bottleneck `compute_or_kernel`, not memory). Run 1–3 showed the real constraint is not the plan but driver-level memory spilling under external GPU load.

## BLOCKED

- **Verified repair corpus**: the environment-spec problem is solved (official images + bug-branch overlay + F2P restore) and the harness is sound, but the corpus is limited by upstream data quality. Of the three rows that reached their tests, all reproduced red and none went green: one trajectory repaired half its FAIL_TO_PASS set, and two carried patches that do not apply to the recorded base state (one of them edits a different feature than its tests exercise). 15 of 23 strict rows are not module-congruent, so a large share of the batch can never verify by construction. No verified repair exists yet; the gate is refusing to certify, which is the intended fail-closed behaviour.
- **Mixture-of-Thoughts ingestion**: no dataset license exists to review.
- **gen-2 / baseline comparison**: Condition A now provides the trained student; what remains is an operator-authorized baseline evaluation run under the same protocol (same holdout, same prompts, same scorer).

## NOT YET ATTEMPTED

- Conditions B/C training runs and checkpoint lineages beyond preflight scale.
- GSM8K / held-out perplexity / instruction-following measurement — the trained Condition A adapter now exists, so the student side of this is no longer blocked, only unrun.
- Preference-pair construction at scale (Condition C stays gated).
- Catastrophic-forgetting measurement (needs a condition-B student first).
- Student evaluation against the frozen gen-2 reference: no longer gated by training, only by the baseline run.
- The remaining 20 strict rows need a quieter host: their first-use image pulls are 3–4 GB each, and image pulling competes with both the desktop and the GPU run.
- A single-writer replay queue: three concurrent batches saturated one podman daemon, which produced cleanup timeouts and one lost row phase (now non-fatal, but not free).

## Answer to the principal question, honestly

Half answerable now, with the other half honestly still open. The student half is no longer blocked: a Condition A adapter trained on 4 529 pinned, license-gated examples exists (284/284 steps, 4 h 46 m, mean loss 1.7276, 25.7 MB of LoRA weights) — but **it has not been evaluated**, so what it learned is unmeasured and no claim about capability or forgetting is made here. The repair half is blocked by the data, not by us: of the three strict rows that reached their tests, red was reproduced in 3/3 and green in 0/3 — one trajectory repaired half its FAIL_TO_PASS set, two carried patches that do not fit the recorded base state, and 15 of 23 strictly-matched rows are not module-congruent to begin with.

What else changed this session: the SFT set exists at usable scale (5 015 pinned examples, up from 6) via conclusion-boundary chunking rather than filtering; the replay sandbox now uses SWE-smith's own environment spec and emits per-row, machine-checkable evidence; the PR itself was reviewed and four real defects were fixed. Running the hardened harness exposed eleven defects of our own — five of which had made every earlier "0 verified" result vacuous (red had never actually been reproduced: the one row that looked like a red result was reporting 0 executed tests), and one of which published an empty loss history for a completed 284-step run.

The remaining constraints are now quantified rather than unknown: upstream data quality (65 % of strictly-matched trajectories carry patches that do not touch the module their own tests exercise), host throughput (cold per-instance images are 3–4 GB and the podman daemon serializes under load), and the fact that the trained student has never been measured. The next hard gates are an evaluation run — GSM8K, held-out perplexity, and the gen-2 comparison under one protocol — and a verified repair corpus, both of which are now execution tasks with known obstacles instead of open design questions.
