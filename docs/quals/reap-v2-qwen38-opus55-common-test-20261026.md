# Experimental REAP128_v2 vs Qwen3.8-27B vs Opus 5.5 — common-source evaluation

**2026-10-26 | experiment/reap128v2-test-unblocked-20261026 | no scientific Gen-0 promotion.**

## Which models, actual outcomes, and provenance

- REAP128_v2 Q8_0 GGUF original source SHA `89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6`, original Linux session `95876063255e45159f57e29306e0e885`.
- REAP128_v2 Q4_K_M original source SHA `2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7`, original Linux session `a2a6b83e345145839cc333561b2f3baf`.
- Qwen **3.8-27B**, not Qwen3.5-27B or the local abliterated Qwen3.8-9B. Official hosted ID `qwen/qwen3.8-27b`, locally configured alternative `groq-qwen3.8-27b` / `routeway-qwen3.8-27b`.
- Claude **Opus 5.5**, not Opus 5 or Sonnet 5.5. Hosted ID `anthropic/claude-opus-5.5`, local LiteLLM alias `claude-opus-5-5`.

The locally cached original official `openai/gsm8k`, `main/test`, 1,319-item rendering was independently rehashed **PASS** to immutable `913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89`. No new dataset was invented, downloaded or replaced. An outcome-independent fixed random seed `20261026` preselected the following 16 distinct official task IDs **before any hosted response was seen**:

`gsm8k-{17,183,232,272,393,466,555,676,699,726,820,828,995,1109,1189,1247}`.

Original Q8 and Q4 full-population JSONL rows passed their full-file original SHA pins `9ff2b418fecabdc419d94b2a06f6d9b115bbbd4bfba59ccae2d0d9f81e8ef749` and `3b09d337f567729192e9c302ec698cf1e34d7451538847939e3bf9a36a9557db`, and full 1,319 unique item/session IDs. These are previously recorded official Linux outputs, not freshly rerun REAP model inference.

| Original-source, fixed 16-item research comparison | Correct / 16 | Provenance |
|---|---:|---|
| REAP128_v2 Q8_0 | **15 / 16 (93.75%)** | Authenticated original Linux full-session row subset |
| REAP128_v2 Q4_K_M | **15 / 16 (93.75%)** | Authenticated separate original Linux full-session row subset |
| Qwen3.8-27B | **UNKNOWN / NOT TESTED** | No authenticated response returned by accessible funded endpoint |
| Claude Opus 5.5 | **UNKNOWN / NOT TESTED** | No authenticated response returned by accessible funded endpoint |

**No model ranking is valid.** A 16-question fixed subset is a smoke-level paired comparison, not a statistically definitive capability ranking, and public API latency cannot be compared directly with local native CUDA model tokens/sec. REAP historical full Linux quality remains Q8 **1,220/1,319** GSM8K and Q4 **1,229/1,319** GSM8K, independent of this slice. HumanEval remains excluded from this API campaign because original independent generated-code sandbox qualification is unavailable; no new model-generated code has been executed.

## Actual provider availability audit and attempt

The operator's present environment exposed `OPENROUTER_API_KEY` and `LITELLM_API_KEY` names; **no token values were printed, persisted or committed**. Both API model catalogs listed the named exact models, and OpenRouter's authentication introspection returned success.

Actual bounded external attempts (no model-generated benchmark answer was obtained):

| Provider route | Result | Meaning |
|---|---|---|
| OpenRouter `qwen/qwen3.8-27b` | **HTTP 402 Payment Required** | Registered first common-source API request refused; no scored row |
| Local LiteLLM `groq-qwen3.8-27b` | **HTTP 403 Forbidden** | Harmless one-question route probe refused |
| Local LiteLLM `routeway-qwen3.8-27b` | **HTTP 402 Payment Required** | Harmless route probe refused |
| Local LiteLLM `claude-opus-5-5` | **HTTP 400 Bad Request** | Two harmless parameter-compatibility probes refused; second omitted temperature; access/balance-related diagnostic observed, but exact cause not independently established |

Only OpenRouter's 402 attempt used the identical frozen original sample prompt. The subsequent two local Qwen and Opus probes used the harmless public arithmetic prompt `6 × 7` to establish basic route availability; **they must not be counted as the registered benchmark**. The absence of responses leaves Qwen and Opus accuracies UNKNOWN, not 0%. No provider switch, fallback model or rewritten model ID was silently counted as equivalent.

The failed OpenRouter run's exclusive private metadata was preserved at:

`F:\reap-v2-qwen38-opus55-math-20261026-first16\`

`manifest.json` records original source SHA, exact preselected IDs, model routing and small cost cap. `research_summary.json` records `INCOMPLETE_NOT_RANKABLE` with **both hosted arms n=0**. No API key or hosted full response is stored in that directory; no new original model weight/output is added to Git.

## Reproducible executable test lane

The experimental branch contains `chowder_batch/reap_v2_threeway_math_20261026.py`. Default is a free, read-only `PLAN_ONLY` verification of model identities and selection:

```powershell
cd C:\Users\nikma\wt\chowder-reap128v2-lab-testing-20261026
python -m chowder_batch.reap_v2_threeway_math_20261026
```

Once the user's own OpenRouter billing quota supports the two exact model IDs, execute only in a **new exclusive private output folder**:

```powershell
python -m chowder_batch.reap_v2_threeway_math_20261026 --execute --provider openrouter --count 16 --max-cost-usd 0.75 --out F:\reap-v2-qwen38-opus55-funded-run-new
```

For a functioning locally configured LiteLLM gateway with both intended underlying routes and sufficient quota, use `--provider litellm` and another fresh output folder. Runtime uses exact rendered official prompts, identical system message, temperature zero, output token maximum 1,024, explicitly requested medium reasoning, original production `final_number_match` scorer and natural-finish requirement. **Reasoning is not identical to REAP's native thinking-disabled decoding**, making this a source-matched research comparison rather than an Amendment 4 serving-parity proof. Result text and raw per-call usage are private; no keys are ever written. Model substitution, HTTP error, incomplete rows, non-finite usage and sample mismatch all fail closed. At currently cataloged provider prices, the conservative planned upper-cost reservation is **$0.448 for 16 paired prompts**, bounded by the `$0.75` command limit; this is a planning guard rather than a guarantee against later provider price changes.

Do not retry the previously refused account configuration without a **material change to available authorized credits/routes**. Never replace an exact named model with Qwen3.8-9B, Opus 5 or another provider's automatically chosen fallback to fabricate a fair test.

**Scientific status unchanged:** `{"status":"BLOCKED","chosen":null,"release_approved":false}`; original PR #282 remains OPEN/DRAFT/UNMERGED. The separate experimental branch remains `UNBLOCKED_FOR_TESTING`, which does not grant access to unqualified generated-code execution or unsafe CUDA resource admission.
