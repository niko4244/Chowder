# REAP128_v2 Q4 / AutomationBench v1.0.6 public-task execution gate

**Experimental only.** The frozen Gen-0 selector is `BLOCKED`, and the original
draft selection PR remains unmerged. The associated gate is
`chowder_batch/reap_q4_automationbench_official_gate_20261026.py`. It is a
separate worker branch from experimental commit
`1aec3a4b1a3ec810d90d81e6779409b0dc64f6eb`. No original model or result
is included here.

## Exact source and scorer

- Zapier official upstream: <https://github.com/zapier/AutomationBench> at Git
  commit **`4a8e1061254004d9dac807054eed33fad7d1ff14`**.
- At that exact commit, `pyproject.toml` declares package `automation-bench`,
  version `1.0.6`, `requires-python >=3.13`, and
  `auto-bench = automationbench.scripts.eval:main`.
- `automationbench/runner.py` binds **official**
  `API_TOOLS = [api_search, api_fetch, base64_encode]` from
  `automationbench/tools/api/__init__.py` and constructs `WorldState` from the
  benchmark task. The `api_fetch` dispatcher at
  `automationbench/tools/api/fetch.py:292-360` routes supported URLs into
  in-memory simulated state; direct external requests or shell invocations
  were not present in the inspected API routes and implementations.
- The **official** `automationbench/rubric/__init__.py:157-180` computes
  `task_completed_correctly = float(partial_credit == 1.0)` from verified
  assertions; `partial_credit` is diagnostic and must never be substituted for
  the strict headline score.
- Public score population: 100 tasks each across `sales`, `marketing`,
  `operations`, `support`, `finance`, `hr`, totaling **600**. The public
  `simple` domain (200 tasks) is excluded. The official leaderboard's harder
  **private** set is not accessible; no direct leaderboard-equivalent score is
  claimed. <https://zapier.com/benchmarks>
- Selected pinned tracked Git blob identities and locked upstream commit are
  embedded in `EXPECTED_BLOBS`. An upstream checkout with tracked edits,
  untracked files, local `.env` or Python startup hooks is refused.

## Proposed execution sequence on an independently admitted Kaggle T4×2 host

The commands below are preparation and do not authorize a GPU job on an
occupied host. Provision a private volume and a Python 3.13 interpreter, clone
the upstream repository into that volume and checkout exactly the pinned SHA.
Run `uv sync --frozen` there using the official `uv.lock` file. **Neither this
worker nor the gate attempts to install software or launch the Q4 server.**

From this isolated Chowder worktree, source verification is read-only:

```bash
python -m chowder_batch.reap_q4_automationbench_official_gate_20261026 \
  --official-source /private/AutomationBench
```

Before any GPU load, only if the two T4s are free and host quota admits,
capture the resource witness into a **new private** path:

```bash
python -m chowder_batch.reap_q4_automationbench_official_gate_20261026 \
  --official-source /private/AutomationBench \
  --q4 /private/reap128-v2-q4_k_m-20261025.gguf \
  --capture-prelaunch /private/run-001/prelaunch.json
```

Prelaunch requires no active CUDA compute PIDs, **≥12,288 MiB free on each T4**,
≥7 GiB physical RAM available, ≥8 GiB uncommitted capacity, ≥8 GiB unused
cgroup v2, and exact original GGUF bytes plus SHA-256. A private witness is
valid for at most two hours and is checked for the same host boot, model and
device inventory. Keep the witness outside all Git and model directories.

After independently starting the authentic native server with the original
model, on `127.0.0.1:8790`, `--fit on -np 4 -c 16384 --jinja`, its port must
belong to the stated PID, the PID must appear in the NVIDIA compute census,
no other GPU compute process may be present, and ≥2 GiB unused VRAM must remain
per T4. The gate also checks the current physical/commit/cgroup reserve.
Only then does the following become admissible:

```bash
python -m chowder_batch.reap_q4_automationbench_official_gate_20261026 \
  --official-source /private/AutomationBench \
  --q4 /private/reap128-v2-q4_k_m-20261025.gguf \
  --server-pid <OWNED_LLAMA_SERVER_PID> \
  --prelaunch-json /private/run-001/prelaunch.json \
  --private-json /private/run-001/one-sales.json \
  --execute
```

This delegates actual tool calls and scoring to the unchanged official
`automationbench.scripts.eval` Python module with `--api chat_completions`,
`--toolset api`, `--max-concurrent 1`, `--max-steps 50`,
`--no-ensure-complete`, `--num-examples 1`, plus a fixed local-only model
endpoint and no hosted API credentials. After a real independently verified
one-task scored receipt, the same command with `--full-public-600`, a **fresh
exclusive private output filename** and sufficient host/quota admission would
request all six public domains (`--num-examples -1`). The upper-bound task step
count is 600×50 model responses; actual compute/token/time budget remains
unmeasured.

**Result status after CLI exit:** output is explicitly
`OFFICIAL_PUBLIC_RUN_COMPLETED_UNVERIFIED` until an independent result audit
proves task IDs, denominator, session identity, binary `task_completed_correctly`
metrics and full coverage. Aborted, timed-out or incomplete tasks cannot be
converted to a verified full score. All raw task responses, grader outputs,
stdout/stderr, model weights and admission receipts remain privately stored.

## Known release gates and current limitations

The worker ran **zero GPU/model inference jobs**; the T4x2 resource admission,
original model load, Jinja tool-call compatibility and full public dataset
population under the pinned Python 3.13 dependencies have not yet been proven
in a real Kaggle environment. The script deliberately does not launch or
terminate protected jobs, modify model weights, change original scoring,
select a Gen-0 model or certify host resource isolation. A model's tool
arguments are dispatched by the trusted official Python code; separate
container/network sandbox qualification is still required before treating
any source outside the pinned, inspected `api` routes as executable.

**Source reviewer note for prime's independent Windows runner:** using only
`git diff --quiet HEAD --` misses untracked `sitecustomize.py` and shadow
imports. Treat upstream version/source validity as a full clean tracked-and-
untracked checkout. A 600-row digest of only `(example_id,prompt)` is a prompt
population witness, not a digest of `initial_state`, assertions and tool
contracts.
