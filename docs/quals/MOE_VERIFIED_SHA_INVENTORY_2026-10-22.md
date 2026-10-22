# MoE checkpoint SHA inventory: enrollment and admission

The `cpu-transient-grad-int8` placement must be bound to the **actual checkpoint
payload**. The original prereg's config-only table cannot certify a training
campaign. The gate runner, witness producer and admission guard now use a
one-time, full-SHA inventory. A legacy gate table must be requalified under
the new content-bound evidence format; copying forward its previously measured
rows does not establish this identity.

## Enroll once, before a placement run

Choose an evidence path outside the checkpoint directory (a new file):

```powershell
$env:CHOWDER_MOE_SHA_INVENTORY = 'C:\llm-work\moe-gates\reap160-sha256-inventory.json'
python chowder_batch/moe_sha_enroll.py --model 'G:\llm-cold\bf16\Qwen3.6-REAP160' --out $env:CHOWDER_MOE_SHA_INVENTORY
```

Enrollment reads and hashes every safetensors shard and each semantic payload
(config, tokenizer, index, etc.) **once**, rejecting a file that changes during
the read. It creates the manifest exclusively; an existing manifest is never
overwritten. It records each SHA256, size, filesystem timestamps and file ID,
the content digest compatible with `chowder.model-content.v1`, and an inventory
digest. The path must be kept available to both gate and witness processes via
`CHOWDER_MOE_SHA_INVENTORY`. Alternatively, without that environment variable,
the default is `.chowder-moe-sha256-inventory.json` in the model directory.

The gate runner fails before loading a large checkpoint when enrollment is
missing or stale. When a gate record is produced, its generated table binds the
checkpoint's content digest **and** the inventory digest. The witness writer
records the same binding, and import checks both. Campaign admission validates
the declared checkpoint against that inventory and refuses old config-only
tables, missing inventories, changed payloads or mismatched content pins.

## How repeat verification stays inexpensive

After enrollment, repeated checks read the small manifest, compare the complete
payload filename set and each weight shard's length, timestamps, volume and file
identity. Only small semantic files are rehashed. They do **not** hash the 46 GB
of REAP160 weights on every iteration.

The lightweight checks catch ordinary in-place writes, replacements, truncation,
missing/extra shards and metadata changes. They are **not cryptographic proof
against a malicious process that rewrites bytes in place and restores every
filesystem identity and timestamp**. A full byte-for-byte recheck against a
separately retained SHA inventory is required at a security boundary where that
adversary is in scope. New or intentionally updated model contents require a
new inventory and new gate/witness measurements.

No production model, Q4 GGUF or GPU run was changed as part of this migration.
