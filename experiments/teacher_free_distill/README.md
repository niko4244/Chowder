# Teacher-free distillation pilot (isolated experiment)

**Scope:** dataset export, provenance gate, normalized Chowder chat material, and CPU tests. This does not run a teacher or student, claim knowledge transfer, or change the primary model lineage. Chowder's primary Qwen3.8 sparse-model program explicitly excludes distillation. Keep this experiment separate.

## Data-source status

| Source | Format | Status |
|---|---|---|
| [OpenThoughts3](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) | conversations with from/value | Card declares Apache-2.0; inspect underlying source terms and approve explicitly. |
| [Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts) | messages with role/content | BLOCKED: an explicit dataset license was not confirmed. |
| [SWE-smith](https://github.com/SWE-bench/SWE-smith) | locally replayed repair traces | BLOCKED until source-repository licenses and trajectory provenance are reviewed. Repository tooling is MIT, but that does not settle all downstream content. |

All entries in sources.json default to approved=false. Review upstream rights, record a usable revision and review reference, and **only then** set approved=true for an eligible source. Do not commit external datasets to GitHub.

## Bounded teacher-free data export

After source review, install the existing Chowder training extras, which include Hugging Face datasets. Export a small pilot without loading a large model:

~~~bash
python experiments/teacher_free_distill/stream_hf.py \
  --catalog experiments/teacher_free_distill/sources.json \
  --source open_thoughts3 --sample-size 1000 --scan-limit 20000 \
  --out /data/chowder_teacher/ot3.jsonl
~~~

The export is deterministic and records its revision and seed. Reservoir sampling is uniform **only over the scanned prefix**, not the full large corpus; inspect shards and source/domain distributions before extrapolating.

## Prepare native Chowder chat rows

Supply a private, immutable final-benchmark exclusion list in JSONL. It is read for contamination checks, never copied into the output:

~~~json
{"prompt":"A held-out evaluation prompt"}
{"repository":"private/heldout-repo","task_id":"bug-17"}
~~~

Then prepare the allowed source:

~~~bash
python experiments/teacher_free_distill/prepare.py \
  --catalog experiments/teacher_free_distill/sources.json \
  --input open_thoughts3=/data/chowder_teacher/ot3.jsonl \
  --holdout /data/chowder_teacher/private_holdout.jsonl \
  --max-rows 1000 --out /data/chowder_teacher/pilot_v1
~~~

The output contains train.jsonl, dev.jsonl and manifest.json. Each training row has a messages array compatible with Chowder's existing chat-data contract. The manifest includes source revisions, SHA-256 input digests, sample counts and exact normalized-prompt exclusion counts.

The development split comes from *training-eligible* material. It is **not** a sealed evaluation benchmark. Exact normalized-prompt checking does not detect all semantic or near-duplicate contamination; add independent audits before performance claims.

## Software-repair trajectories

Downloaded trajectories are **not** verified just because they claim that a patch was successful. SWE-smith's published environment creation targets Ubuntu and Docker. Execute downloaded projects only in disposable sandboxes with resource restrictions and independent tests, never directly on a work PC.

The intended replay adapter produces records in this shape:

~~~json
{
  "task_id": "bug-17",
  "repository": "example/repo",
  "task": "Fix parsing edge case",
  "events": [
    {"kind":"tool","action":{"tool":"read_file","path":"parser.py"},"observation":"...","verdict":"verified_good"},
    {"kind":"test","action":{"tool":"run_tests"},"observation":"1 passed","returncode":0,"tests_executed":1,"verdict":"verified_good"}
  ],
  "verification": {
    "method":"sandbox_replay",
    "returncode":0,
    "tests_executed":1,
    "trace_sha256":"SHA256_OF_CANONICAL_EVENTS"
  }
}
~~~

Use prepare.digest(events) for the trace digest. A matching digest catches accidental mismatches but cannot authenticate a forged replay claim; preserve independent execution logs. The preparer quarantines records without matching replay metadata and an observed successful test event. It emits only individually marked verified_good actions as supervised targets.

**Not yet implemented here:** the secure SWE-smith Docker replay runner, quality review of entire teacher corpora, genuine matched preference construction, model training, and live gen-2 comparison.

## Training and promotion

Use the existing Transformers/PEFT backend and completion-only chat masking for a separate 1.5B–7B student experiment. Bind train.jsonl to a new Chowder project; do not alter the current Qwen3.8 primary lineage or running GPU jobs. Run an independent baseline evaluation first. A preference experiment must use genuinely verified chosen/rejected pairs, not synthetic negative labels on otherwise unverified actions.

Compare baseline and student under the same evaluation protocol: GSM8K, completely unseen repair tasks, nonexistent reads, recovery after failed first fixes, EOS termination, throughput and measured peak VRAM. Keep the final benchmark untouched while selecting hyperparameters, publish per-task traces and do not auto-promote from development scores.

## CPU tests

~~~bash
python -m pytest tests/test_teacher_free_prepare.py -q
~~~

The current tests cover license fail-closed behavior, OpenThoughts-style message conversion, duplicate suppression, reserved-holdout exclusion, repair trace evidence and unsupported claims of green tests.
