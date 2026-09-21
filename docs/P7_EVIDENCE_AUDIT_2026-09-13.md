# Training-first continuation audit — 2026-09-13

## Outcome and execution source

PR [#158](https://github.com/niko4244/Chowder/pull/158) merged at
2026-09-13 03:24:46 UTC as `2ffb5f2e11510bf27ddb762113ec83f6d3b2124d`.
All six required checks passed on that PR. The repository was fetched again
during this audit; no open PRs were returned. This session did not merge it.

The follow-up lives on `codex/p7-evidence-guards` in
`C:/Users/nikma/Brainz/workspaces/Chowder-p7-evidence-fixes-20260912`, based on
that merged main. It is local, not pushed. The other agent's checkout remains
clean at `a6ae0a8`; its tree matched merged main before our changes.

**The tiny PEFT CPU checkpoint restart is now independently exercised. The
normal-interface router-training milestone is not complete.** Do not interpret
PR158's title, a helper test, or a consistent resume report as full workload
qualification.

## Reconciliation and preservation

Three commits on `docs/engineering-run-results` were absent from main after the
code merge. They were replayed unchanged with `cherry-pick -x`, preserving authorship:

| Original | Local replay | Artifact |
|---|---|---|
| `c254c0c` | `5f46ef0` | [Reconciled training-first plan](superpowers/plans/2026-09-12-chowder-training-first-reconciled.md) |
| `a51a933` | `1a02fa9` | [P0 preservation/reconciliation inventory](P0_RECONCILIATION_2026-09-12.md) |
| `2971fae` | `a972c8c` | [P12 Unsloth result triage](P12_UNSLOTH_RESULT_TRIAGE_2026-09-12.md) |

A path-scoped diff against `2971fae` confirmed these three files are identical.
Their September 12 status observations are historical, not current PR/GPU state;
this audit supersedes those observations without rewriting the evidence.

Untouched boundaries:

- Primary `Chowder` checkout: `e8b8696`, six modified and seven untracked entries,
  including activation-offload and evaluator work. No reset, stash, staging, or edits there.
- Handoff checkout: `2971fae`, clean. Router-healing checkout: `a6ae0a8`, clean.
- Frozen verification checkout: detached `dd5ca706d56309324a3e35c67a7c522410e6f53f`, clean.
- Frontier's five modified files remain modified; fresh SHA-256 prefixes match P0:
  `mmlu_pro_eval.py` `013252a6046457ea7647`, `run_cycle4_math_train.py`
  `ee61853cc89c0b480797`, `test_mmlu_pro.py` `1ba57e5fe7fc13cbabcb`,
  `adapter_delta.py` `bc7ae6f6e95a1fd51aa9`, `train_dpo_trl.py` `4dfd5c4da865193709ea`.
- Research datasets, model weights, historical registries, preregistrations,
  prior run directories, and automation state were not changed. New CPU audit
  artifacts were written separately; no GPU training or evaluation was launched.

## P7 findings and corrections

### Empty or unreadable state was counted as present

On frozen `dd5ca70`, zero-byte optimizer, scheduler, and RNG files were accepted
as a complete checkpoint; `assert_resumable(require_rng=True)` accepted them.
The inventory used file metadata without a read probe.

`9a31347` adds a one-byte read probe and rejects empty/unreadable pieces. Six
negative controls reproduced failures before the fix and passed after it.
This remains a cheap metadata/readability check: it does **not** deserialize
untrusted pickle payloads or prove nonempty tensor files are valid.

### A bare worker assertion was accepted as a witness

On the same source, `{"resume": {"matched": true}}` was enough for the parent
to report `state="witnessed"`, without checkpoint path or counters.

`9a31347` validates the declared path, exact integer counters, parent source
inventory, required fields, strict boolean flags, and top-level final-step
telemetry. A consistent record now explicitly says
`verification="source-metadata-and-worker-report"`. Missing legacy telemetry
remains unknown. Report consistency is not independent proof that optimizer/RNG
tensors were restored.

The upstream `a6ae0a8` no-op correction is preserved. At-horizon and below-horizon
zero-step continuations keep distinct progress labels; neither demonstrates
additional optimization. `44faf2f` adds compatibility tests for both cases.

### The original continuation fixture was weaker than its description

It called `train()` twice in the same Python process. Its eight unique strings
all became the same input after 16-token truncation: `the cat sat on t`.
Consequently, loss agreement could not meaningfully distinguish sample order.

`44faf2f` changes the existing test, not the production training architecture:

- Put the distinguishing word first; assert all eight tokenized rows remain distinct.
  This assertion failed on the old fixture (`1 != 8`) before the correction.
- Launch control and continuation in separate worker interpreters, using the
  existing source-identity and worker-environment helpers, CPU-only/offline,
  with a 180-second timeout per worker.
- Keep the total scheduler horizon at eight steps and resume checkpoint four.
- Compare newly computed loss/LR suffixes with named CPU absolute tolerances
  (`1e-6` loss, `1e-12` LR, zero relative tolerance). Keep copied history and
  integer counters exact; retain the existing `1e-6` final-adapter tolerance.
  These are engineering regression tolerances, not changed research thresholds.
- Rename/document the partial-save test as a synthetic inventory control, not
  an actual forced-interruption experiment.

## Retained independent experiment

Evidence root:
`C:/Users/nikma/Chowder-Protected/runs/2026-09-11-gsm8k-continuation/`.

`verify_p7_independently.py` and `p7-independent-cpu-20260912/` are frozen
diagnostic artifacts against `dd5ca70`, not scripts silently retargeted to new
source. The directory retains protocol, source identity, specs, stdout/stderr,
results, base, adapters, and real checkpoint state.

- Tiny local Qwen2, CPU, eight distinct tokenized inputs, eight-step horizon.
- Control PID 19816, 25.3236091 seconds; resumed PID 82088, 24.8533182 seconds.
  Both exited successfully; no concurrent GPU job was started.
- Newly executed steps 5–8 had exactly matching loss and LR sequences.
- Final adapter maximum absolute difference: **0.0**, against predeclared `1e-6`.
- Checkpoint four contained nonempty optimizer, scheduler, trainer state,
  training arguments, and RNG state. The corrected parent accepted its retained
  real worker result with the narrower verification label.

This supports deterministic tiny-CPU continuation equivalence. It does not
directly compare consumed sample IDs or every optimizer/RNG tensor, exercise
stochastic/dropout restoration, kill a worker mid-run, evaluate an independently
reloaded router delta, or close a normal-project registry lifecycle.

Artifact SHA-256:

| Artifact under evidence root | SHA-256 |
|---|---|
| `verify_p7_independently.py` | `00227e9d52264a997e059b8fd46df7e164985ccfe5c2583838d121181aa1d298` |
| `p7-independent-cpu-20260912/report.json` | `351b0cddafbdd6a4f2e7963bb3ff8373cf9b29a383d98437a7defb800d11295c` |
| `p7-fixes-full-suite-20260913.xml` | `6076d6d4da2429721be4000fe10c9a04e45b2356d1e847503b5d29b4fd7ee377` |
| `p7-fixes-real-resume-smoke-20260913.xml` | `b3d92d76801e785fd7f0b7d708f96c2d783a0d52ab55ca10b3198de17b9a5049` |

## Verification on merged main plus the fixes

Runtime: `C:/Users/nikma/Chowder/.venv-repro/Scripts/python.exe`, Python 3.11.9;
torch 2.11.0+cu128, Transformers 5.16.1, PEFT 0.20.0, datasets 4.3.0,
safetensors 0.8.0, pytest 8.4.2, ruff 0.15.11. Source imports were pinned to the
isolated worktree with `PYTHONPATH=src`; CUDA was hidden from test processes.
No dependency installation or global environment change was made.

| Check | Result |
|---|---|
| Combined targeted guard/checkpoint/backend/real-continuation suite | 260 passed, 21 skipped, 109.26 seconds |
| Full suite after final fixture/tolerance edits | **1,814 passed, 77 skipped**, 165.17 seconds; XML retained above |
| `python -m ruff check src tests` | Passed |
| `git diff --check` | Passed; only Windows line-ending advisories |
| Explicit opt-in real Llama continuation and no-op/activation-offload smoke | **2 passed**, 112.46 seconds; XML retained above |
| Retained real worker result passed through the current parent guard | Passed: four new steps, `source-metadata-and-worker-report` |

The targeted result preceded the final tolerance/docstring changes; the full
result includes them. PR158's six green remote checks do not cover this unpushed
follow-up. The default suite's 77 skipped tests are not counted as passes.
Two of those gated tests were explicitly enabled and passed separately; the
other skipped paths remain unqualified by this audit.

Reproduction from this isolated checkout (no packages are installed by these commands):

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONPATH = 'src'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:CHOWDER_REAL_ML_SMOKE = '0'
& 'C:\Users\nikma\Chowder\.venv-repro\Scripts\python.exe' -m pytest -q -p no:cacheprovider
& 'C:\Users\nikma\Chowder\.venv-repro\Scripts\python.exe' -m ruff check src tests

$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:CHOWDER_REAL_ML_SMOKE = '1'
& 'C:\Users\nikma\Chowder\.venv-repro\Scripts\python.exe' -m pytest tests/test_transformers_backend.py::test_real_tiny_llama_resumes_training_from_a_real_checkpoint tests/test_transformers_backend.py::test_real_tiny_llama_resumes_across_a_different_activation_offload_setting -q -p no:cacheprovider
```

The two opt-in tests used the already cached tiny Llama model in offline mode;
they will need that cache or an explicitly prepared equivalent environment on
another machine. The strengthened Qwen2 continuation fixture creates its tiny
model/tokenizer locally and does not require a model download.

## Next implementation contract

P8 files are still absent: `backends/router_healing.py`,
`backends/router_healing_worker.py`, `tests/test_router_healing_backend.py`.
The callback-based `router_healing_orchestrator.py` is not the production path:
it receives a preloaded model outside its timer, permits stringified resume
state, and discards the loaded delta before calling evaluation.

Use the existing executor/project/cycle/registry machinery in this order:

1. **Close remaining P7 controls where needed.** Add genuine graceful-stop and
   abrupt-death tests, direct state/sample-position evidence, and terminal-cost
   checks; do not relabel the synthetic inventory test as those controls.
2. **P8: narrow router executor and worker.** Implement existing `profile/run/cancel`
   contracts. Begin CPU-only with real tiny MoE E=4/k=2, router-only cross-entropy,
   exact gate-path allowlist, frozen experts/shared branch, fixed horizon and
   hard step/token/time limits. Invoke the existing gradient/update probe in the
   worker; merely reporting path coverage is insufficient. Count calibration
   steps and cost. Reuse source identity, manifests, checkpoint and lifecycle helpers.
3. **Repair the existing digest before CUDA.** `trainability._tensor_digest`
   converts the whole flattened tensor to FP32 on its current device, then
   calls `.numpy()` without `.cpu()`. It is not CUDA-ready and may allocate the
   full frozen fused tensor before sampling. Use a bounded, device-safe read
   strategy and test the actual device path; preserve full-versus-sampled labels.
4. **P9: verified router payload and independent evaluation.** Save replacement
   router values separately from resumable optimizer/scheduler/RNG state, with
   base/key/shape/dtype/content binding. Add identity and nonidentity application
   controls in a fresh evaluator process. The current evaluator calls
   `PeftModel.from_pretrained`; router artifacts must not be sent there unchanged.
5. **P10: thin normal-project wiring and closeout.** Register the backend only
   with matching artifact/evaluator dispatch. Reuse `chowder train <project>`;
   no new CLI or registry. Reopen the registry and reconstruct gate, lineage,
   actual versus charged cost, cancellation/failure, and settled reservations.
   Keep engineering qualification separate from champion promotion.
6. **P11: bounded ladder.** Tiny real CPU normal-interface cycle, then separately
   preregistered small CUDA qualification, then the actual 9B-derived pilot if
   measured representation/memory/cost fits. Do not infer fit from trainable
   parameter count or start another full-size research sweep first.

The strict target remains the **9B successor, <=10B total / <=3.5B active**,
not the 27B program. The reconciled plan's measured non-FFN floor is
4,121,965,056 parameters: expert sparsity alone cannot meet the target on the
unchanged backbone. A budget-valid narrower/different always-on backbone is a
later, explicitly gated architecture task, not a reason to bypass the training
path. Parameter arithmetic and perplexity do not establish useful generation.

## Agent routing

The requested Nestor skill was used to inspect the compiled roster and select
the loadable software-engineering capability. Its native read-only Claude
invocation failed before inference with HTTP 404 for the configured
`claude-fable-5` model (reported token usage and cost zero). No model substitution,
gateway change, queue worker, or exposure change was made. Nestor's single-shot
queue has no file/tools loop and was not used for this repository task.

A warm existing Codex reviewer provided bounded read-only cross-checks and the
next P8 contract; earlier it owned the inventory-readability patch only. The
root reconciled the results and verified them. Nestor remains the preferred
roster when its runtime is available; this session did not establish successful
Nestor execution. Shared gbrain/second-brain tools were unavailable; no shared
memory write was claimed or attempted.
