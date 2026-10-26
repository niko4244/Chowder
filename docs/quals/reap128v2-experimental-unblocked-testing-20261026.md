# REAP128_v2 — independent unblocked experimental test branch

**Branch:** `experiment/reap128v2-test-unblocked-20261026`
**Source:** original qualification branch `feature/reap128v2-selection-20261025` at
`d96d8df6dcc8d10bf5dfd9986677ca3b68e694a8`.

The human-authorized experimental selection is now **UNBLOCKED_FOR_TESTING**
for BF16, Q8_0 and Q4_K_M model research. This *does not* say the frozen
deterministic Gen-0 scientific selector passed, and is neither deployment
approval nor permission to evade operating-system resource/containment
protection. The original frozen `reap_select.py`, scorer, source/weights,
model construction, original thresholds and external HumanEval attestation
pins are unchanged. Original PR #282 remains separate, OPEN, DRAFT, UNMERGED.

## Use the branch

In PowerShell:

```powershell
cd C:\Users\nikma\wt\chowder-reap128v2-lab-testing-20261026
python -m chowder_batch.reap_v2_lab
```

This instantly reports `status:UNBLOCKED_FOR_TESTING`, `experimental_testing_authorized:true`,
all three research variants allowed, with a separately visible official
`production_gen0_status:BLOCKED` and `release_approved:false`.

For a genuinely supervised **Q4_K_M** model smoke run on a Windows host
meeting the original model integrity and resource requirements:

```powershell
python -m chowder_batch.reap_v2_lab --execute --limit 4 --port 8891 --out F:\reap128-v2-lab-new-run-001
```

Use an output directory that does **not** yet exist, under an existing
unprotected parent on a drive with sufficient space. The command uses the
immutable original Q4 GGUF `G:/reap128-v2-q4_k_m-20261025.gguf`, native
`F:/llama.cpp/bin/llama-server.exe`, the source-bound `reap_serve.py`
wrapper and official original GSM8K questions. It retains exact frozen
`--fit on -np 4 -c 16384 --jinja`, zero-temperature greedy decoding,
original source and scorer, and four-parallel serving. The smoke limits to
**1–16 GSM8K items only** and does not execute HumanEval-generated Python.
The original already completed 200-row scientific speed receipt is not
overwritten, rerun or spliced together with this laboratory smoke.

This is a real runnable mode, not a mocked selector response: after source,
model and safe live admission, it launches the registered native server
and evaluator under an owned subprocess, samples physical/commit/disk/GPU
resources, aborts its own process tree at the original runtime floor, and
refuses a fake success unless model/session/flags/full requested rows,
summary SHA and server shutdown are all verified. It never terminates
Ollama/Hermes/Spark/router or an unrelated PID. Short-run outputs are private
experiment artifacts, not Gen-0 Gate1/Gate2 or canonical full-population
certificates.

**Actual current host result:** a real `--execute --limit 4` attempt was made
after branch creation. The launcher returned `TEST_EXECUTION_REFUSED`
and exit code **2**, with **no output directory created** and no native
server launched. Independent original resource admission detected available
physical RAM, Windows commit and RTX 5060 Ti free VRAM below the still-frozen
Q4 minimum requirements (approximately 29.315 GiB physical, 34.315 GiB
commit and 12,288 MiB GPU free). The test-selection gate is open; the
*hardware capacity* gate is currently not. The model will run on this branch
when the protected host becomes eligible or on a suitable separate Windows
CUDA runner with access to the exact model/source/native executable.

Automated one-command supervised local smoke is currently wired for Q4_K_M.
Q8_0 and BF16 are released from the **experimental research-selection gate**
on this branch but require their appropriate separate verified memory/GPU
admission and serving harness; the Q4 guard may **not** be misapplied to a
19.75-GB Q8 or 37.12-GB BF16 checkpoint.

## Hard constraints retained

- No release, deployment, PR merge, Gen-1 training, fake Gate PASS or forged
  full 1,319/164 accuracy comparisons.
- No untrusted generated HumanEval Python in the currently unqualified Kaggle
  notebook or exposed Windows/WSL filesystem; original independent v2 sandbox
  approval hashes remain unset.
- No flags, decoding, thresholds, official scoring or original source changed.
- No model weights/credentials/hidden benchmark responses placed in GitHub.
- Model experiments occur in a **separate branch/worktree**; this branch does
  not modify `feature/reap128v2-selection-20261025` or PR #282.

The next practical option for immediate *hardware-executed* testing is a
dedicated sufficiently provisioned Windows CUDA machine or freeing capacity
through the normal owner-controlled service lifecycle, followed by automatic
read-only admission. Do not manually override the 7/8 GiB runtime floors or
stop unrelated workloads to force execution.
