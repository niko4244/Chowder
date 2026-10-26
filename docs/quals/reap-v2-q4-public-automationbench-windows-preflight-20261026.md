# REAP128_v2 Q4 public AutomationBench research lane — 2026-10-26

## Authority and result

The experiment branch may test the actual immutable Q4_K_M artifact. The original
Amendment 4 deterministic Gen-0 scientific selector remains
`{"status":"BLOCKED","chosen":null,"release_approved":false}`.
The original draft PR #282 must remain open, unmerged, and draft. This lane
never writes original model weights, release gates, protected model responses,
or the scientific worktree.

**No new Windows Q4 inference was launched during this intervention.** A CPU-only
real-time admission reading at 2026-10-26 19:07:45 UTC returned `REFUSED`:

| Available hardware | Observed | Original source-derived startup minimum |
| --- | ---: | ---: |
| Physical RAM (GiB) | 15.4307 | 29.3146 |
| Windows commit (GiB) | 14.7598 | 34.3146 |
| VRAM (MiB) | 1,015 | 12,288 |
| Private output disk (GiB) | 35.1515 | 8 |

This is **not** a discretionary Q4 block. These thresholds were derived from
the source-bound previously aborted Windows Q4 evaluation. Do not lower them
or treat process suspension as a substitute for measuring freed physical
memory and available Windows commit.

### Reversible process intervention review

The dedicated Spark llama server on 127.0.0.1:8767 was independently seen
idle, but unloading it plus the narrow currently loaded Ollama embedding
model would still leave available commit and physical RAM well under the
original startup floors. Ollama also had active established client
connections. Reclaiming protected Windows Terminal, Codex, WSL, Hermes,
Brainz, developer shells or other work to cover the remaining deficit would
risk user sessions. Consequently, no protected service was interrupted or
model process terminated in this phase. A CPU-intensive **inspection command
created by this investigation** was stopped by exact verified PID after it
proved unexpectedly expensive; it was not a model or user workload.

The dedicated Spark listener and Ollama server were still present during
this review. Future automated reloads can change usage abruptly. If a new
admission attempt is authorized, use the fresh current-state checks; never
reuse this snapshot to claim startup capacity.

### Official public benchmark source verified

Zapier AutomationBench repository commit:
`4a8e1061254004d9dac807054eed33fad7d1ff14`, package `1.0.6`.
Exact isolated interpreter: Python 3.13.13. The full official public
600-task loader was executed on this host during this phase and verified
six domains of 100 scored tasks (`sales`, `marketing`, `operations`,
`support`, `finance`, `hr`). The source-ID/prompt population SHA-256 is
`1c8feeec1f680b05b392a5d00fc634edb143c3e7bdd152c476af50cebcec57b5`.
That digest is a prompt-population witness, not a digest of complete world
state, assertions or hidden/private leaderboard tasks.

The official evaluator is `python -m automationbench.scripts.eval`, with
`--api chat_completions --toolset api --max-concurrent 1 --max-steps 50`.
The model's original inference flags stay
`--fit on -np 4 -c 16384 --jinja`.
The strict official headline score counts only fully completed tasks;
partial assertions do not count as wins. The public 600-task suite is **not**
equivalent to the harder official private benchmark leaderboard.

### Source and qualification implementation

- `chowder_batch/reap_q4_automationbench.py` is the Windows owned-child runner.
  It refuses unsafe live hardware before hashing the 11.56 GB GGUF, runs an
  exact Q4 SHA test when admitted, binds to an exclusive F: private result
  folder, checks native loopback PID/model identity, recomputes 41/41 CUDA
  layer and zero-CPU-expert witnesses, and monitors immutable original
  runtime floors. It never returns a verified score merely because the CLI
  process exits zero.
- `tests/test_reap_q4_automationbench_20261026.py` is CPU-only refusal,
  source-shadow, CLI, owned-model and CUDA-identity coverage.
- The independent Kaggle/Linux source-bound gate is in
  `chowder_batch/reap_q4_automationbench_official_gate_20261026.py`, with
  its separate tests and detailed accompanying document. It does not launch
  another GPU session itself.

Focused CPU matrix at this point: **29 passed**, Ruff clean. No model-driven
test was represented as passing.

At inspection, two independent Kaggle GPU notebooks were running:
`nikmarco/reap128v2-random128-full-postload` and
`nikmarco/reap128v2-q4-hle-textonly-20261026`. They must not be cancelled
to start a new AutomationBench job. Account quota remained around 15 GPU
hours, but quota is distinct from available concurrent execution slots.

### Read-only invocation

```powershell
cd C:\Users\nikma\wt\chowder-reap128v2-lab-testing-20261026
python -m chowder_batch.reap_q4_automationbench --out F:\reap-v2-automationbench-private-20261026\next-public-task --count 1
```

`--execute` is intentionally omitted here. Executing is permitted only after
an independently passing original Q4 admission and official source/model
checks, on an available isolated host with no protected jobs displaced.
Independent row auditing is necessary for any scored full-600 result.
Any chart cell without direct benchmark evidence must remain `N/T`.
