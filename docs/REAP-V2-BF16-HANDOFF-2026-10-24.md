# Handoff — frozen REAP128_v2 original BF16 704×40 (completed 2026-10-24)

The experiment is **finished and measured**. Read
`REAP-V2-BF16-D0-RESULT-2026-10-24.md` for the result itself. This file is the
operational handoff: what is running, what was changed, what is preserved, and
what must not be repeated.

## 1. Result in one line

704 frozen sequences × 40 layers, 1,192,632 tokens, original BF16 weights at
pinned revision `995ad96e…` → **calibration perplexity 15.451270210051794**,
frozen gate `< 20` → **D0 PASS**, published as a sealed `observation.json`
(`799cc170767f6b17b40c3b01eb3f95ec71310a1154a1ce0570fb07468dde76ed`).

## 2. Machine state — restored and verified

| Service | Port | PID | Verification |
|---|---:|---:|---|
| Local GGUF router (`F:\llama.cpp\models-local.ini`) | 8766 | 36372 | `/health` → HTTP 200 |
| Sharp-Spark sidecar (`spark-x2.5-4b`, Q8_0, 4.11 B params) | 8767 | 49852 | `/health` → HTTP 200, `/v1/models` lists `spark-x2.5-4b` |
| Ollama | 11434 | 15608 | `/api/tags` → HTTP 200, **11 models**, store `F:\Ollama\models` |

Also verified:

- **The temporary Hermes quiesce override is gone.** `C:\Users\nikma\.hermes\services.yaml`
  no longer carries a `restart_policy` block on `ollama`; the original entry had
  none, so the supervisor default applies again. A dated comment remains in place
  of the block so the edit is auditable. The file parses as YAML and the
  supervisor was restarted (PID 61240, task `HermesSupervisor` = Running) so it
  reloaded the config.
- **The keep-awake helper is stopped** (`hold_keep_awake.py` and its wrapper
  chain were terminated by exact PID).
- **Chat On Steroids** (PID 51828, port 8765) was never touched — it is the
  communication bridge and owns 8765 outright. All Windows infrastructure and
  unrelated processes were left alone.
- No pagefile, Windows setting, model file or historical artifact was modified.
- `OllamaWatchdog` scheduled task remains **Disabled** (it was already disabled
  before this work; leaving it disabled is deliberate — it raced the supervisor).

## 3. Two stale launchers found — read before the next reboot

Both are pre-existing drift, not caused by this run, and both will bite the next
time these services are restarted by script.

**`F:\llama.cpp\start-spark.cmd` (reached via `SparkX25-LlamaSidecar.vbs`)**

- It points at `I:\Ollama\models\blobs\sha256-5c2c3c19…` — **that path does not
  exist**. The identical 4,073,119,744-byte blob lives at
  `F:\Ollama\models\blobs\sha256-5c2c3c19…`.
- It asks for `--port 8765`, which is **owned by Chat On Steroids**, so it cannot
  bind.
- It therefore cannot restore the sidecar as-is. The restore actually performed
  was the same command with the `F:` blob path and `--port 8767` (the port the
  sidecar held when it was quiesced), preserving its other flags
  (`--alias spark-x2.5-4b -ngl 999 --ctx-size 131072 --jinja -fa on --top-p 0.95
  --temp 0.7 --min-p 0.0`), logging to `F:\llama.cpp\spark-8767.log`.
  **Fix the script before relying on it again** (model path and port).

**`start-ollama.vbs`**

- It sets `OLLAMA_MODELS=H:\Ollama\models`; **H: does not exist**. The real store
  is `F:\Ollama\models`, which is set as a user environment variable, so the
  supervisor-launched Ollama is correct and the vbs is simply a trap.
  Do not "restore" Ollama with that script.

## 4. Doc correction worth propagating

The earlier operator notes listed the Sharp-Spark sidecar as listening on
**8767**, while every on-disk launcher says **8765** (a port Chat On Steroids now
owns). The protected-listener check used 8766/8767/11434. Both were quiesced for
this run, so nothing was missed, but a future admission run should treat the
sidecar's real port as configuration, not a constant — a sidecar that quietly
rebinds to 8765 would be invisible to a hard-coded 8767 check.

## 5. Preserved artifacts — do not delete

| Path | Size | Contents |
|---|---:|---|
| `C:\llm-work\reap\observe-v2-bf16-atomic-20261024-222950-223f34d3` | 0.47 MiB, 41 files | Sealed `observation.json`, `layer-00..39.pt` |
| `F:\reap-v2-bf16-atomic-20261024-222950-223f34d3` | 9.1 GiB, 86 files | Two 4,885,020,672-byte BF16 slabs + 41 generation manifests |
| `C:\llm-work\reap\supervisor-v2-bf16-atomic-20261024-222950-223f34d3-*` | — | Every admission attempt, per-session `supervisor-state.json`, `resources.jsonl`, observer stdout/stderr |
| `C:\llm-work\reap\RUN-STATUS-20261024.md` | — | Live run record incl. admission-override reasoning |
| `C:\llm-work\reap\SERVICE-QUIESCE-RESTORE-20261024.md` | — | Pre-quiesce PID/creation-time identities |

The `F:` scratch is the only copy of the committed hidden state. Keep it until
the attestation/parity release decision; it is what a parity re-check would read.

**Never** resume or overwrite the historical r2 output
(`observe-v2-full-registered-20261023-r2`, `F:\reap-v2-hidden-full-20261023-r2`).
Its single-buffer format has no `CURRENT`; it is evidence of the failure, not a
checkpoint.

## 6. Re-verify the result at any time (read-only, ~1 s)

```powershell
python -B -c "from chowder_batch.reap_gates import _d0; print(_d0(r'C:\llm-work\reap\observe-v2-bf16-atomic-20261024-222950-223f34d3\observation.json'))"
```

Expected: `{'status': 'PASS', 'statistic': 15.451270210051794, 'threshold': 20.0, 'n': 40, 'reason': None}`.
The full independent cross-check (identity digests, 40 distinct layers, exact
router census) is printed in the result document.

## 7. If this experiment must be reproduced from scratch

Use **new** output, scratch and log directories; never point at the completed
run's paths.

```powershell
python -u -B -m chowder_batch.reap_observe_supervised --launch `
  --model-dir 'G:\llm-cold\bf16\Qwen3.6-35B-A3B' `
  --weights-dir 'G:\llm-cold\bf16\Qwen3.6-35B-A3B' `
  --calibration 'C:\llm-work\reap\calib-v2-ornith-bare-20261023\calibration.jsonl' `
  --out '<NEW_C_OUTPUT_DIR>' --scratch '<NEW_F_CHECKPOINT_DIR>' --log-dir '<NEW_LOG_DIR>'
```

Practical notes learned from this run:

- Budget **~2.7 h of pure layer compute** plus **three full 71.9 GB source-digest
  passes** (admission, optional resume, pre-publication) and the closing NLL
  pass. The observed wall clock was **6 h 23 m** with one abort/resume cycle.
- The 8-hour per-invocation safety limit was never approached; a resume starts a
  fresh budget.
- Admission is the real bottleneck on a busy desktop, not the GPU. Each agent
  shell call adds 3–6 process births inside the 60 s window against a limit of
  11, so **monitoring the run frequently is itself enough to fail admission**.
  Prefer `run_until_admitted.py` for retries and check rarely.
- Quiesce the three model services for the window; they hold GPU and commit
  headroom the observer's floors need. Restoring them afterwards is cheap
  (§2), but allocate an extra ~10 minutes for the sidecar model load.
- Expect the RAM sawtooth: free physical RAM drops to ~9 GiB while each 4.55 GiB
  slab is written and recovers to ~19–23 GiB afterwards. That is page-cache
  churn, not a leak. The genuine danger signal is the compound
  fanout-plus-decline rule, which correctly fired once at layer 39.

## 8. Remaining work (release gates, not blockers to the result)

1. **Independent execution attestation** — a hash chain is not proof of
   execution. Trusted execution, signed producer attestation, or an equivalent
   independent oracle is still required for a fully release-grade claim.
2. **CUDA numerical parity** — a dedicated small-model equivalence check against
   a reference implementation. CPU storage/crash tests do not certify arithmetic.
3. **Sudden-power-loss durability** of the atomic-rename + `fsync` protocol on
   the external USB volumes.
4. **Launcher hygiene** — fix `start-spark.cmd` (model path + port) and retire or
   correct `start-ollama.vbs`, so a reboot restores the same services that were
   quiesced here.
5. **PR #280** — this result and the two runtime fixes (`e4c380c`, `2442a8f`)
   plus the resume pre-screen fix (`e239096`) are on
   `fix/reap-bf16-atomic-resume-20261024`; merge when reviewed.

Items 1–3 are statements about how strongly the result can be trusted, not
suggestions that the measurement is wrong. The measurement was produced by a real
run on real frozen inputs and is reported with those limits attached.
