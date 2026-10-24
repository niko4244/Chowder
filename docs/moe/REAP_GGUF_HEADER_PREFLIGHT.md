# REAP GGUF metadata-only inference admission

`chowder_batch/reap_gguf_header_preflight.py` checks that a GGUF header has the
same routed MoE structure as the local BF16 model config and original pruning
record. It reads metadata with `gguf.GGUFReader` (memory-mapped, no tensor
deserialization or model inference) and hashes only the small config and
pruning JSON files. It does not hash the 13.8 GB GGUF payload or the BF16
checkpoint shards.

```powershell
python -B chowder_batch/reap_gguf_header_preflight.py `
  --gguf 'C:\llm-work\reap\gguf\reap160-q4_k_m.gguf' `
  --bf16-model 'G:\llm-cold\bf16\Qwen3.6-REAP160' `
  --out 'C:\Users\nikma\_audit\reap160-gguf-header-preflight-20261023.json'
```

The output path must not exist. On the existing REAP160 GGUF, the validator
checks all 40 layers, 160 expert slots per layer, top-8 routing, 162 explicitly
shaped essential tensors (router plus three expert projections per layer and
vocabulary input/output), and all 40 pruning-survivor sets. The recorded
file type is 15 (GGUF Q4_K_M), quantization version 2, with 733 total tensor
headers. Changing a field, tensor axis, selected survivor count, or layer
inventory causes a refusal.

This is **not** a frozen Gen-0 speed/fit gate, content-authenticated weight
lineage, evidence of a successful llama.cpp model load, VRAM measurement,
prompt or decode benchmark, tool-call pass, or permission to promote weights.
The previously archived GGUF SHA-256 is not reverified by this utility.
Runtime evidence still requires an authorized isolated inference session, exact
artifact hashes, per-process VRAM and host-use observation, prompt/decode
latencies, EOS/stop completion, and independent output checks.
