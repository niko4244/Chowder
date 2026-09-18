# Gen-2 preregistration — Amendment 9 (2026-09-18)

Pre-compute. No Gen-2 candidate compute has run, and this amendment changes no
frozen quantity: no target or protected threshold, benchmark identity, slice
index, seed, decoding setting, cost ceiling, selection rule, trusted ancestor or
stopping rule is touched. It records (1) the operator-facing readiness surface
that amendment 7's in-run `readiness` phase made possible, and (2) a base-identity
defect the new surface immediately surfaced and that blocks Gen-2 from starting.

## 1. `chowder growth campaign readiness` — one authoritative, zero-compute gate

Amendment 7 put every pre-compute prerequisite *inside* `run_campaign` (identity,
declared inputs, contamination, plan, projection, arms, evaluator coverage), but
the only way to observe them was to start a run and read the phase that refused.
Amendment 9 adds the inspection those checks were always describable as:

```
chowder growth campaign readiness docs/gen2/gen2_campaign.json
```

It starts no process and loads no model. It reports a structured, machine-readable
document — `status` (`READY`/`REFUSED`), a `checks[]` list with a stable
`reason_code` per failure, and a flat `reason_codes[]` — and exits non-zero unless
every check passes, so CI can gate on it. The checks are exactly the ones
`run_campaign` applies before it trains, read from the same helpers
(`undeclared_inputs`, `_verify_digest`, `_load_binder`, `plan_campaign`,
`_select_recipes`, `settle_campaign_projection`, `build_evaluator` and the
evaluator's own `admit`), so a `READY` report is a run that will not refuse before
it spends, and a `REFUSED` one is a run that stops before training. A check whose
prerequisite failed is reported `skipped`, never passed.

The identifiers are `READINESS_*` constants in `chowder.growth.campaign_runner`
(`READINESS_SCHEMA`, `_DECLARED_INPUT`, `_BASE_IDENTITY`,
`_PARENT_ADAPTER_IDENTITY`, `_CONTAMINATION`, `_PROJECT_TEMPLATE`,
`_TRAINING_MATERIAL`, `_DATA_REGISTRY`, `_HARDWARE_BUDGET`, `_PARENT_PROFILE`,
`_PARENT_ARM`, `_ANCESTOR_ARM`, `_PROTECTION_POLICY`, `_PLAN`, `_RECIPE_SET`,
`_CAMPAIGN_PROJECTION`, `_EVALUATOR`, `_EVALUATOR_COVERAGE`).

## 2. Blocker discovered: the declared base digest is on a basis production does not verify

Running readiness against the committed declaration refuses with, among others,
`READINESS_BASE_IDENTITY`. The cause is not a changed model:

* The frozen dense base at `F:/llm-models/Qwen3.8-9B-abliterated-25-bf16` still
  matches the Gen-0 freeze file-for-file. Every one of the ten semantic files in
  `/c/Users/nikma/Chowder-Protected/runs/2026-09-16-gen0-eval-freeze/identity_manifest.json`
  hashes to the recorded sha256 (four shards, index, tokenizer, tokenizer config,
  chat template, config, generation config).
* But `identity_manifest.json` records `model_content_digest_basis` as *"sha256
  over sorted 'name size sha256' lines of files[]"* — a digest over that semantic
  file set — and `base_model_digest = 59e767aa…7555f` is exactly that figure.
* Production verifies `base_model_digest` with `training_binding.directory_digest`,
  which recursively hashes **every** file in the tree. That includes a
  HuggingFace download cache (`.cache/huggingface/**`, 16 files), `README.md`,
  `.gitattributes` and `processor_config.json`. That digest is `8eb92aa6…653b`.

So the declaration pins a semantic content digest the verifier never computes, and
`_verify_digest` can never accept it — for the base only. The parent adapter
digest in the same file (`ca8769c5…`) *does* verify, because it was pinned with
`directory_digest`. The two identity fields were populated on two different bases.

This is a pre-existing defect, not a regression: `run_campaign` already refused
the committed declaration at `_verify_parent_identity`. The readiness command's
only role is that it reports *all* blockers at once instead of one per attempt.

The fix is a policy decision and is deliberately **not** taken in this amendment:
either (a) repin `base_model_digest` to `directory_digest(base)` — consistent with
the adapter field and the verifier, but it folds a volatile HF download cache into
frozen identity, so any cache touch breaks it again; or (b) make the verifier use
the freeze's semantic `model_content_digest` basis for the base — consistent with
the already-frozen Gen-0 identity and stable against cache churn, but it changes
the identity rule and therefore needs its own preregistered, qualified change.
(b) is the scientifically correct direction; neither is applied here, and Gen-2
remains `REFUSED` until one is.

## Status

`chowder growth campaign readiness docs/gen2/gen2_campaign.json` →
`REFUSED`, `reason_codes = ["READINESS_DECLARED_INPUT", "READINESS_BASE_IDENTITY"]`.
The declaration still provides none of the seven inputs a run reads from disk
(project template, training material, data registry, hardware budget, parent
profile, parent evaluation, evaluation material), and the base identity does not
verify. Gen-2 execution remains blocked, by construction, until both are fixed.
