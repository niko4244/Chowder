#!/usr/bin/env bash
# A4 GSM8K (batch 1, like every other GSM8K arm), local bf16, pinned worktree d98edb5.
set -u
WT=/c/Users/nikma/Chowder/.claude/worktrees/eval-d98edb5
G=/c/Users/nikma/chowder_teacher_free/eval_gsm8k2048
DIR=$G/results/condition_a4__final
cd "$WT" || exit 1
export PYTHONPATH="$WT/src" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
echo "=== $(/usr/bin/date -u +%FT%TZ) A4 GSM8K start"
python -m chowder.evaluators.transformers_text_worker --spec "$DIR/eval_spec.json" --result "$DIR/result.json" \
  --chowder-identity "$G/chowder_identity_d98edb5.json" > "$DIR/worker_stdout.log" 2>&1
echo "=== $(/usr/bin/date -u +%FT%TZ) A4 GSM8K rc=$?"
