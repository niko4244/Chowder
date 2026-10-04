#!/usr/bin/env bash
# Phase-1 determinism verification (fixed byte-writing pipeline).
# 1. Re-chunk the OT3 source twice (default chunker): byte-digests must match
#    each other and the recorded rebuild.
# 2. Re-chunk with --ids --preceding-context: digest must match
#    ot3_chunked_v2.jsonl (same default target).
# 3. Re-prepare pilot_v4 from the existing v2 chunk file into a fresh dir:
#    train/dev digests must equal pilot_v4/manifest.json output_sha256.
set -u
WT=/c/Users/nikma/Chowder/.claude/worktrees/tfd-revised
DATA=/c/Users/nikma/chowder_teacher_free
cd "$WT" || exit 1

python experiments/teacher_free_distill/ot3_subset.py \
  --input "$DATA/ot3_sft.jsonl" --out "$DATA/ot3_chunked_default_verify_a.jsonl" \
  > .phase1-verify-chunk-a.json 2>&1
python experiments/teacher_free_distill/ot3_subset.py \
  --input "$DATA/ot3_sft.jsonl" --out "$DATA/ot3_chunked_default_verify_b.jsonl" \
  > .phase1-verify-chunk-b.json 2>&1
python experiments/teacher_free_distill/ot3_subset.py \
  --input "$DATA/ot3_sft.jsonl" --out "$DATA/ot3_chunked_v2_verify.jsonl" \
  --ids --preceding-context > .phase1-verify-chunk-v2.json 2>&1

python experiments/teacher_free_distill/prepare.py \
  --catalog experiments/teacher_free_distill/sources.json \
  --input open_thoughts3="$DATA/ot3_chunked_v2.jsonl" \
  --out "$DATA/pilot_v4_determinism" \
  --max-rows 6000 --max-chars 8000 --near-dup-field target \
  > .phase1-verify-prepare.json 2>&1

python - <<'PY' > .phase1-verify-result.json 2>&1
import hashlib, json
from pathlib import Path

data = Path("C:/Users/nikma/chowder_teacher_free")

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

result = {}
a = sha(data / "ot3_chunked_default_verify_a.jsonl")
b = sha(data / "ot3_chunked_default_verify_b.jsonl")
result["chunker_default_deterministic"] = a == b
result["chunker_default_digest"] = a
result["chunker_v2_digest"] = sha(data / "ot3_chunked_v2_verify.jsonl")
result["chunker_v2_recorded_digest"] = sha(data / "ot3_chunked_v2.jsonl")
result["chunker_v2_matches_recorded"] = (
    result["chunker_v2_digest"] == result["chunker_v2_recorded_digest"])

manifest = json.loads((data / "pilot_v4" / "manifest.json").read_text(encoding="utf-8"))
det = data / "pilot_v4_determinism"
result["prepare_train_matches_pilot_v4"] = (
    sha(det / "train.jsonl") == manifest["output_sha256"]["train"])
result["prepare_dev_matches_pilot_v4"] = (
    sha(det / "dev.jsonl") == manifest["output_sha256"]["dev"])
det_manifest = json.loads((det / "manifest.json").read_text(encoding="utf-8"))
result["prepare_manifest_integrity_ok"] = det_manifest["integrity"]["ok"]
result["prepare_train_rows"] = det_manifest["train_rows"]
result["prepare_dev_rows"] = det_manifest["dev_rows"]
print(json.dumps(result, indent=2))
PY
echo DONE
