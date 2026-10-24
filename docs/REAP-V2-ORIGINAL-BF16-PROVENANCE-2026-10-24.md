# REAP128_v2 original BF16 revision pin and token census

**Status: source identity pinned; 704×40 D0 still NOT_MEASURED.**

Original public Qwen source: `Qwen/Qwen3.6-35B-A3B`, immutable Git
revision `995ad96eacd98c81ed38be0c5b274b04031597b0`.

An exact-revision Hugging Face Hub tree API response was retrieved over TLS:

```text
https://huggingface.co/api/models/Qwen/Qwen3.6-35B-A3B/tree/995ad96eacd98c81ed38be0c5b274b04031597b0?recursive=false&expand=true
```

The reviewed `chowder_batch/reference/reap_v2_original_qwen36_bf16_995ad96e.json`
contains **26 upstream LFS SHA-256 OIDs and per-shard sizes**, totalling
71,903,776,776 bytes, plus **11 original model/tokenizer metadata** entries.
Small metadata content SHA-256 digests were derived from local file bytes
**only after** their original Git blob OIDs or upstream LFS hashes matched
the exact-revision API response. Its SHA-256 is
`54e7e0d121cab5fd7e331be1ee38c5b4deddd7eee7f0cc7b112a4322f8418dca`,
cross-checked by the versioned Python validator.

The repository pinned manifest, rather than writable
`.cache/huggingface/download/*.metadata` or the observer's own receipt, is
the expected source of truth. The frozen observer preflight checks the
original 11 small payloads and all shard names/sizes before any GPU model
load; the usual 26-shard streaming SHA-256 in `fingerprint_inputs()`
must also equal the independently pinned 26 upstream payload digests.
On resume and before publishing the final D0 observation, it verifies them
again. The v2 selector, supervised result validator and pruning consumer
reject receipt artifact hashes differing from the fixed pin.

## Independent frozen calibration retokenization

Frozen registered calibration:
`f6a09b694ae43a6d8b44b5180a861856501295eabf3d5297eb9749a008cdd929`.
The original 704 packed calibration rows declare **1,192,636** tokens.
CPU-only retokenization through the verified original tokenizer using
`add_special_tokens=False` yielded exactly **1,192,632** tokens, minimum
row length 263, maximum 2048. Four SWE Smith rows, zero-based
40, 86, 128 and 178, retokenize from a declared 2048 to 2047 tokens.
These are expected frozen evidence and should not be "corrected."

The exact ordered BF16 observer `token_ids_sha256` is
`78b8f482f61cfd4b20bba6982a20c8eeb5a757b20069f2520273d4a315335f92`.
All **704** ordered retokenized row lengths are in the versioned manifest.
They are checked against the runtime checkpoint input and later against
the sealed final observation. A forged identity with merely the same token
sum or changed row order is refused.

## Security and scientific qualification limits

The manifest authenticates expected **publisher file identities** to the
extent the Hub exact revision and its TLS API response are trusted.
The 26 local 71.9-billion-byte model payloads were **not rehashed in this
engineering/CPU test session**; that verification happens only at admitted
observer startup/resume/finalization. A local forged Hugging Face download
metadata file is not a replacement for the publisher pin.

A chain of ordinary file hashes is **not a cryptographic proof of execution**.
An actor able to synthesize fully plausible 256-expert floating-point
statistics, corresponding self-hashes and fake perplexity can still produce
self-consistent untrusted artifacts using public expected-source hashes.
Strong independent producer attestation / trusted execution and a genuine
CUDA numerical parity run remain **unresolved release gates**. The CPU tests
prove source/format rejection and storage crash behavior, not genuine
40-layer forward-computation accuracy or pass-level D0 evidence.

No existing model files, restored services, historical r2 artifacts or
official weight shards were modified; no full CUDA/model computation ran.
