# Gen-2 preregistration — Amendment 10 (2026-09-18)

Pre-compute. No Gen-2 candidate compute has run, and this amendment changes no
frozen quantity: no target or protected threshold, benchmark identity, slice
index, seed, decoding setting, cost ceiling, selection rule, trusted ancestor or
stopping rule is touched. It resolves the base-identity blocker amendment 9
recorded, on the basis amendment 9 said was correct (b), and nothing else.

## 1. The base is pinned to a model-content digest, not a whole-tree digest

Amendment 9 found that `base_model_digest` in the shipped declarations is the
Gen-0 freeze's semantic `model_content_digest` (`59e767aa…7555f`), while
production verified it with `training_binding.directory_digest`, which hashes
**every** file under the base directory — including a HuggingFace download cache
the base acquires after it is fetched (`.cache/huggingface/**`, 16 files),
`README.md`, `.gitattributes` and `processor_config.json`. That whole-tree digest
is `8eb92aa6…653b`, so the declared figure could never verify.

A base identity must be stable against cache churn and blind to nothing that
matters. The fix is therefore (b), as amendment 9 called it: a base is verified
against a **model-content** digest over the payload files only.

`chowder.local_model_manifest.model_content_digest(model_dir)` is that basis. It
hashing nothing but the model's payload — the weight shards and the semantic
metadata files — and hashes `"<name> <size> <sha256>"` lines joined by newlines,
so the volatile cache and the provenance-only files cannot move it. The payload
file set is the one the freeze recorded: the `.safetensors` shards plus
`model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json`,
`chat_template.jinja`, `generation_config.json`, `config.json`. `README.md`,
`.gitattributes` and `processor_config.json` are not model files and are not
included; neither is anything under a hidden path.

Basis identifier: `chowder.model-content.v1`. The adapter keeps
`directory_digest`: it is a small byte-stable directory a run writes itself, with
no cache in it, and its declared digest (`ca8769c5…`) already verifies.

## 2. The freeze's own basis string does not describe its digest — recorded, not rewritten

The Gen-0 `identity_manifest.json` records `model_content_digest_basis` as
*"sha256 over sorted 'name size sha256' lines of files[]"*. That string is
inaccurate: the freeze's producer (`identity_manifest.py`, preserved in the
freeze directory) joins the lines in a fixed **list** order — weights first, then
the metadata files — and never sorts them. A digest over the sorted lines of the
same ten files is a different value (`532fcdd5…`), so "sorted" cannot be the
construction. The stored `59e767aa…` is reproducible only from the producer's
order.

Production therefore encodes that order explicitly
(`local_model_manifest._MODEL_CONTENT_FILE_ORDER`: weights by name, then
`model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json`,
`chat_template.jinja`, `generation_config.json`, `config.json`) and reproduces
the frozen figure exactly on the frozen tree — ten payload files, digest
`59e767aa…7555f`, verified. The freeze's own `identity_manifest.json` is
historical evidence and is **not** modified; the discrepancy is recorded here and
in the module.

## 3. What changed in the run, and what did not

* `campaign_runner._verify_base_identity` is the single owner of base identity:
  it recomputes the model-content digest of `base_model_path` and refuses with
  the measured digest, the basis and the payload file count when it does not
  match. `_verify_parent_identity` and the readiness `base_identity` check both
  route through it.
* The adapter check is unchanged: still `directory_digest`, still its own field.
* No threshold, benchmark set, protocol, budget, provenance rule, trusted
  ancestor, contamination requirement or verdict semantic is altered. This is an
  identity *basis* correction, not a policy relaxation: the base is still
  required to hash exactly to its declared figure before any compute, and a
  substituted `config.json` still refuses.

## Status

`chowder growth campaign readiness docs/gen2/gen2_campaign.json` now reports
`base_identity: ok` — *"base model-content digest matches 59e767aab1da over 10
payload files (basis chowder.model-content.v1)"* — and
`parent_adapter_identity: ok`. `READINESS_BASE_IDENTITY` is no longer a reason
code. The declaration still refuses on `READINESS_DECLARED_INPUT`: the seven
inputs a run reads from disk are not yet produced. Gen-2 remains blocked until
they exist and the Gen-0 trusted-ancestor arm is measured.
