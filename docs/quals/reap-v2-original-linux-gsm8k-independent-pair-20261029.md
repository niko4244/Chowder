# REAP128_v2 — independent historical Linux GSM8K1319 paired verification

**2026-10-29. RESEARCH ONLY. Gen-0 BLOCKED, chosen null, release_approved false.**

This read-only CPU verification independently regraded both previously completed
original Linux GSM8K1319 archives using the immutable canonical 1,319-item
source digest `913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89`.
It does not import the mutable production scoring implementation. It checks
each recorded response against the canonical expected answer, natural termination,
full task-ID coverage, session identity, and independently recomputed trigram
ratio. Archived raw rows, model/session audit receipts and paired publication
must match their independently recorded SHA-256 digests before any result is
accepted. It never starts inference, executes generated programs, or edits F:.

| Distinct original Linux research arm | Correct | Natural termination | Archived JSONL SHA-256 |
| --- | ---: | ---: | --- |
| Q4_K_M, session `a2a6b83e345145839cc333561b2f3baf` | 1229/1319 | 1281/1319 | `3b09d337f567729192e9c302ec698cf1e34d7451538847939e3bf9a36a9557db` |
| Q8_0, session `95876063255e45159f57e29306e0e885` | 1220/1319 | 1280/1319 | `9ff2b418fecabdc419d94b2a06f6d9b115bbbd4bfba59ccae2d0d9f81e8ef749` |

Paired same-canonical-item descriptive counts: **1198 both correct, 68 both
incorrect, 22 Q8-only correct, 31 Q4-only correct**. Q8 minus Q4 is
**−9/1319 = −0.0068233510**. The independent results reproduce the previously
published research-only paired receipt SHA-256
`125df9301e222f3f9220742514f279b1b6ff4b31444b709afa0ab4bfe8e13b93`.

Source code: `chowder_batch/reap_original_linux_gsm8k_independent_pair_20261029.py`.
Adversarial tests:
`tests/test_reap_original_linux_gsm8k_independent_pair_20261029.py`.
The frozen source and original F: evidence were **not modified**.

## Explicit nonclaims

The audit receipts contain model SHA-256 **claims**, not an independently
authenticated original native HTTP request-to-response chain. Historical
correctness is regraded from archived response text, not from a new model
execution. The two Linux arms have distinct serving sessions and cannot be
pooled with the Windows Q4 1224/1319, original dense reference, or frozen
Gate1/Gate2. The paired descriptive delta is **not** a prospectively approved
Amendment 4 noninferiority interval. This work establishes neither the original
dense same-session throughput nor BF16 chained numerical parity.

The original research-only HumanEval164 Q4/Q8 certificates remain complete and
untouched (151/164 and 152/164). Historical REAP128-v1 Gate1 remains FAIL;
REAP160 native speed remains UNKNOWN. No official frozen gate was promoted,
no model weights were changed, and no release or Gen-1 training is authorized.
