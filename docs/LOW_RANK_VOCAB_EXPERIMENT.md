# Experiment A: Low-Rank Vocabulary Embeddings and Output-Projection Compression

**Status: matrix/pilot stage complete. Recommendation: do not scale without a
hybrid (low-rank + low-rank-residual) design; pure SVD factorization is
rejected by the evidence.**

Checkpoint: `F:/llm-models/Qwen3.8-9B-abliterated-25-bf16`
(`Qwen3_5ForConditionalGeneration`, `qwen3_5`, bf16, unquantized, untied,
24-layer linear-attention hybrid, vocab 248320, hidden 4096).
Artifacts: `C:/Users/nikma/Chowder/.chowder-spark-calib/low-rank-embed/`
(`inventory.json`, `factorization_results.json`, `probes.json`,
`teacher_cache.safetensors`, `real_eval.json`, `recovery_pilot.json`).

## 1. Architecture and tensor inventory (Phase 1)

The memo's conditional resolved favorably: the repo's HF cache holds only a
Q4 GGUF of this model, but `F:/llm-models/` holds the **full-precision bf16
safetensors** it was quantized from (the `-GGUF` twin's README states the
relationship). No GGUF was treated as a trainable tensor; no download was
needed.

| item | value |
|---|---|
| architecture / model_type | `Qwen3_5ForConditionalGeneration` / `qwen3_5` (text `qwen3_5_text`) |
| hidden x vocab | 4096 x 248320 |
| tying | **untied** (`lm_head.weight` and `model.language_model.embed_tokens.weight` are distinct bf16 tensors, 2.03 GB each) |
| total checkpoint | 18.8 GB, 760 tensors, bf16 |
| unique params | ~8.8 B (embedding 1.017 B + lm_head 1.017 B + decoder ~6.7 B) |
| used per token | ~7.7 B (one 4096-param embedding row + full 248320x4096 head every decode step) |
| quantization | none (dense bf16, trainable as-is) |
| teacher VRAM/RAM | ~20.2 GB VRAM-equivalent; loads on this machine via accelerate CPU offload |
| one-matrix workspace | 1.89 GB bf16 / 3.79 GB fp32; Gram workspace 0.06 GB |

## 2. Factorization (Phase 2)

Separate modules (`src/chowder/low_rank_vocab.py`) replace each matrix with
`U [vocab, r] @ V [r, hidden]`: `LowRankEmbedding` and `LowRankLMHead`
(`head_a = V`, `head_b = U` maps the checkpoint layout). Both preserve token
IDs, vocabulary layout, and output shape exactly. `factorize_svd` is
memory-aware (exact SVD when the matrix fits, streaming randomized otherwise);
`src/chowder/low_rank_checkpoint.py` reads/writes factors without duplicating
the checkpoint and records a `chowder-low-rank-v1` manifest.

Ranks 2048/1536/1024/512 were cut from **one** covariance eigendecomposition
per matrix (fp64 `eigh` on the 4096x4096 Gram), on GPU, in ~2 s per matrix.
All factors persisted (~4.9 GB total):

| rank | params/matrix | compression | head latency (ms, b=1) |
|---:|---:|---:|---:|
| 2048 | 517 M | 1.97x | 10.1 -> 6.7 |
| 1536 | 388 M | 2.62x | 10.0 -> 4.0 |
| 1024 | 258 M | 3.94x | 10.2 -> 3.1 |
| 512 | 129 M | 7.89x | 10.2 -> 1.5 |

## 3. Quality probes (Phase 3)

Tokenizer census: **184,741 of 248,320 tokens (74%) are rare-unicode
(multilingual tails); only 36,555 ASCII-word tokens.** CJK: 0.

Matrix-level probes (embedding rows as hidden-state surrogates):

| rank | row err | top-1 vs orig | KL |
|---:|---:|---:|---:|
| 2048 | 0.560 | 0.109 | ~3e-5 |
| 1024 | 0.740 | 0.021 | ~6e-5 |
| 512 | 0.831 | 0.006 | ~7e-5 |

The tiny KL was a **surrogate artifact** (near-flat input distributions), not
safety. With real teacher hidden states:

| rank | val KL (nats) | top-1 | top-5 | CE@teacher-argmax |
|---:|---:|---:|---:|---:|
| 2048 | 3.58 | 0.233 | 0.257 | 6.13 |
| 1536 | 3.91 | 0.183 | 0.203 | 6.52 |
| 1024 | 4.32 | 0.133 | 0.160 | 7.05 |
| 512 | 4.91 | 0.050 | 0.100 | 7.72 |

Damage is **uniform across token classes** (digits 0.56, ascii 0.57, punct 0.65,
rare unicode 0.56 at r2048): the spectrum is too flat for SVD to protect rare
tokens specifically — there is no "safe direction" to discard. The low KL in
the matrix probe and the high KL on real states together show why matrix-level
error alone cannot clear a candidate.

## 4. Recovery pilot (Phase 4)

One bounded teacher pass (32 prompts x 23 positions, 359 positions) cached
hidden states + logits; the pilot then trains **only the head factors** with
KL(teacher || student) from the cache (299 train / 60 val positions, 150 steps,
AdamW 1e-4, batch 64, ~8-25 s per rank):

| rank | recon KL | distilled KL | recon top-1 | distilled top-1 |
|---:|---:|---:|---:|---:|
| 2048 | 3.58 | **2.16** | 0.233 | 0.317 |
| 1024 | 4.32 | 2.87 | 0.133 | 0.250 |
| 512 | 4.91 | 3.58 | 0.050 | 0.200 |

Distillation recovers substantially (KL -40%, top-1 up 2-4x) but plateaus far
above deployment quality. Note the pilot isolates the head only; a full
candidate would also factorize the embedding, shifting layer-0 inputs and
compounding the damage.

## 5. Why the spectrum is the obstacle

The embedding/head spectra of this model are close to flat: at rank 2048 (half
the hidden width) the factors capture only 63-68% of matrix energy. Every
additional singular direction carries nearly equal weight, so there is no
low-rank backbone to keep and no safe tail to drop. This matches the Phase-3
observation that rare-token rows degrade exactly as much as frequent ones.
Energy captured: 63/51/38/24% (embedding) and 68/57/44/30% (head) at
2048/1536/1024/512.

## 6. Recommendation

1. **Do not promote any pure-SVD candidate.** Even at half-rank the real-KL is
   ~3.6 nats with 23% top-1 agreement; nothing here is deployable. This is an
   evidence-based rejection, not a serialization accident.
2. **If this direction is pursued further**, the evidence points to a
   **hybrid design**: keep a low-rank backbone (e.g. r=1024) plus a learned
   sparse/quantized residual correction, trained with distillation from the
   cached teacher. Pure low-rank is provably insufficient on flat spectra.
3. **The machinery is reusable**: factor modules, memory-aware SVD, no-duplicate
   conversion, teacher caching, and the paired probes all work on the real
   checkpoint and are CPU-tested (`tests/test_low_rank_vocab.py`, 8 tests).
   They transfer to any checkpoint with a genuinely low-rank vocabulary
   geometry.
4. **Cost of re-running**: teacher cache ~13 min once; rank evaluation ~1 min;
   pilot ~1 min. The experiment is fully reproducible from the artifacts.

## Stop conditions honored

- Numerical stability was fine everywhere (fp64 covariance eigh; no NaNs).
- Quality degradation was the binding constraint: every candidate fails the
  "unacceptable degradation" stop condition on real-KL grounds.
- No large training run was launched; the pilot gated it out.
- No candidate was promoted for serializing or generating text.
