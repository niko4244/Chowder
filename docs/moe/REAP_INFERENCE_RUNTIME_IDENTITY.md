# REAP160 local runtime identity preflight (read-only)

`chowder_batch.reap_runtime_preflight` checks the existing llama.cpp server's
model census **without** issuing a generation request or triggering preset
autoload. It first reads the real REAP160 GGUF metadata/topology against the
local BF16 config/pruning record, then issues one GET to
`http://127.0.0.1:<port>/v1/models`. The request has a 3-second timeout and no
redirect/proxy behavior; only IPv4 loopback is contacted. It refuses unless
exactly one entry declares both `status.value=loaded` and a single absolute,
existing `--model` path resolving to the GGUF under test. A matching alias,
model ID, unloaded preset, or unrelated resident model is **not** sufficient.

```powershell
python -B -m chowder_batch.reap_runtime_preflight `
  --port 8766 `
  --gguf 'C:\llm-work\reap\gguf\reap160-q4_k_m.gguf' `
  --bf16-model 'G:\llm-cold\bf16\Qwen3.6-REAP160' `
  --out 'C:\Users\nikma\_audit\reap160-existing-endpoint-8766-20261023.json'
```

Receipt output is exclusive-create (`x`); use a new path on a subsequent
observation. Exit code 0 means **endpoint metadata readiness only**, not that
training, frozen speed/fit, or a model inference request passed. Exit code 2
means refusal; its JSON receipt states why. A server without an inspectable
loaded state or model path is not falsely classified as a matching checkpoint.

On 2026-10-23, the real existing port 8766 served non-REAP presets, and port
8767 served the unrelated Sharp-Spark 4B model. Both returned `REFUSED` with
zero qualifying REAP160 candidates. The two distinct evidence receipts are:

- Port 8766: SHA-256 `a0bddeb55575a25611f84124cfc2c1320718dfa562bfb7f4cce520b3bbd87166`.
- Port 8767: SHA-256 `00605ba02e33baae17be621ab9b16312247ddc457ce6682a960161bebb054e66`.

No live generator was called, no resident model/server was changed, and no
GPU-hours were consumed by this preflight. A server-declared loaded path is
not itself authenticated weight content; benchmark evidence would additionally
require a source-bounded GGUF SHA, captured process identity and resident
memory, prompt/decode timings, EOS behavior, structured tool-call trials and
independently reviewed outputs. **Do not start or switch to a REAP160 model
automatically as part of preflight.** Frozen preregistered selection and owner
promotion gates are unaffected.
