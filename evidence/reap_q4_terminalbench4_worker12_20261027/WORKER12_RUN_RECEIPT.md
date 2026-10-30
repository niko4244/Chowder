# Worker-12 original REAP128_v2 Q4 Terminal-Bench 4.0 trial

## Authenticated inputs

- Official Harbor CLI: `harbor==0.25.0`, installed only in `.venv` under this directory.
- Official dataset: `terminal-bench/terminal-bench@4.0.0`, downloaded with `harbor datasets download` into `source/terminal-bench`; 66 published tasks present.
- Selected **unmodified** original task: `interleaved-vigenere`.
- Original `task.toml` SHA-256: `d2b9b4f98304d5458ccde463b9670a9ddd89cd4c01bbd03e93a4b92860bcab8a`.
- Original `instruction.md` SHA-256: `d9a28686a9f18d5f76ef876f7b035d90bdede6d0625056e581fc65fba0e2f3cd`.
- Official task environment image: `harborframework/terminal-bench:interleaved-vigenere-environment-9d9fa0a5aba4b859@sha256:683e32f166cb4698c641bf45cb7efbeb631e879d8d5143bc6cdf799f16d4b592`.
- Official separate verifier image: `harborframework/terminal-bench:interleaved-vigenere-verifier-16783c64c1923947@sha256:bfe3a07dc06a39182f64dbb1d92af539340587f778409fe62bd82bfacb36f259`.
- Original Q4 GGUF: `G:/reap128-v2-q4_k_m-20261025.gguf`, **independently hashed** to SHA-256 `2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7` (11,563,915,264 bytes).

## Isolation and runtime

- The prime's Q4 AutomationBench run uses GPU server port 8892. This trial uses a **separate CPU-only** `llama-server` PID **24184**, port **8896**, `-ngl 0 -c 8192 -t 2`, therefore makes no requests to the prime's server.
- The original Harbor task was started using `--env podman --agent terminus-2` and the official grader, with `--model openai/reap128_v2_q4 --base-url http://127.0.0.1:8896/v1 --api-format openai_chat_completions`.
- Native rootless Podman Linux VM is active; the task's original Podman container `interleaved-vigenere__g2dhgb9__env_main_1` has `Privileged=false` and **zero host mounts**. Its default bridge network is enabled; no claim of full network denial is made for this task.
- Separate independent security smoke with `python:3.13-slim` and `--network none --user 65534:65534 --read-only --cap-drop=all --security-opt=no-new-privileges` confirmed denied access to common Windows host-mount paths and `connect_ex` returned ENETUNREACH (101).
- On this Windows host, the global `podman-compose` launcher crashes with `_thread.daemon_threads_allowed`. The task trial uses isolated `.venv/Scripts/podman-compose.exe` version 1.6.0 through a per-command `PODMAN_COMPOSE_PROVIDER` environment variable. No global package or source was changed.

## Real execution evidence

- First official Harbor attempt (`jobs/q4-tb4-interleaved-vigenere-worker12-20261027`) failed during environment initialization due to the broken global Compose launcher. This attempt has no model score.
- Second attempt (`jobs/q4-tb4-interleaved-vigenere-worker12-v2-20261027`) **started the original task container** and generated `trajectory.json`, proving actual model inference and model-issued sandbox actions, including first model reply with 1,245 prompt tokens and 474 completion tokens.
- Harbor records live trial progress in its `result.json`, task `trial.log`, and `trajectory.json` files. **No score may be claimed until the separate original verifier completes and its result is independently checked.**
- Run was started in connected terminal session `52386`; its exit code and verifier receipts determine the outcome.
- Live original task reached at least **three completed model actions**, with exact prompt/completion token pairs `(1245,474)`, `(4221,804)`, `(6193,521)` in native Harbor `trajectory.json`. A later request of **13,753 tokens exceeded the separately configured CPU server's 8,192-token context**; the official Terminus-2 agent automatically attempted fallback summarization. This is an actual protocol/runtime limitation and is **not** a task failure score by itself.
- Terminal-Bench is a full agentic task requiring many shell actions. At the context failure, Q4 had inspected `/app` and sample input paths and **had not yet created either required task artifact** (`/app/cracker.py`, `/app/requirements.txt`). Independent `podman exec ... test -f` established both absent. A partial-task '0' is not a defensible original benchmark score.
- The official agent's fallback full summary subsequently failed: `Model openai/reap128_v2_q4 hit max_tokens limit. Response was truncated`; the agent attempted a separate short summary. The 8,192-token CPU context does not have the same geometry as the prime's 73,728-token GPU lane, and any completed score from this lane would require a **NONCOMPARABLE** context disclosure unless the benchmark protocol formally accepts it.

## Original launch command

Run in the feature worktree after sourcing the isolated environment and only if the CPU Q4 server is available:

```powershell
$env:OPENAI_API_KEY = 'local-q4-isolated-worker12'
$env:PODMAN_COMPOSE_PROVIDER = (Resolve-Path 'evidence/reap_q4_terminalbench4_worker12_20261027/.venv/Scripts/podman-compose.exe').Path
& 'evidence/reap_q4_terminalbench4_worker12_20261027/.venv/Scripts/harbor.exe' run --path 'evidence/reap_q4_terminalbench4_worker12_20261027/source/terminal-bench/interleaved-vigenere' --agent terminus-2 --model 'openai/reap128_v2_q4' --base-url 'http://127.0.0.1:8896/v1' --api-format openai_chat_completions --env podman --n-concurrent 1 --max-retries 0 --job-name 'q4-tb4-interleaved-vigenere-worker12-v2-20261027' --jobs-dir 'evidence/reap_q4_terminalbench4_worker12_20261027/jobs' --yes
```

## Final worker-12 disposition

**`BLOCKED_NO_OFFICIAL_SCORE`**, after executing real original Q4 inference and original task sandbox actions. Model generated 3 complete agent steps; independent trajectory SHA-256 at cutoff: `a2cb476b59041fdc1bee33725eada48c1967d71c3f34d42a57d809060afbc026`. The next 13,753-token request could not fit the CPU server's 8,192-token context. The official Terminus-2 `full summary` max-token-truncated; the subsequent `short summary` generated more than 2,300 tokens without producing the task files. At cutoff the original job still reported `n_running_trials=1, n_completed_trials=0, n_errored_trials=0, evals={}`. Therefore there is **no official grader output, pass/fail or score**. Specifically, a missing score must not be treated as zero.

The worker deliberately terminated **only its own** benchmark Python process PID `28180`, CPU-only `llama-server` PID `24184`, and stopped Podman task container `interleaved-vigenere__g2dhgb9__env_main_1` after documenting the measured blocker. This released CPU and host RAM for the prime's concurrent 600-task GPU evaluation. The saved Harbor job result may remain `n_running_trials=1` because the runner was externally terminated; the manual-stop receipt supersedes that stale status. Original task and model inputs remain immutable. No official grade was run.

**Needed to continue and obtain a defensible score:** a dedicated, source-pinned Q4 inference lease able to serve the original agent's growing 13,753-token and later prompts (32,768 tokens minimum for the observed next request, likely 65,536+ for a longer trajectory), while keeping untrusted shell commands in the existing Podman task environment. The prime's GPU endpoint `8892` was already running another evaluation and was never used. The known minimum blocker is the model context/resource lease; the official Harbor 4.0 source and original container start were proven working. Resume as a **new** named job with the recorded CLI and larger dedicated Q4 endpoint; the old run cannot be construed as an official scored trial.
