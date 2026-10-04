#!/usr/bin/env bash
# GSM8K eval of the Kaggle fp16 A3 adapter, after the local A3 eval releases the GPU.
set -u
WT=/c/Users/nikma/Chowder/.claude/worktrees/tfd-revised
EV=/c/Users/nikma/chowder_teacher_free/eval_gsm8k2048
DIR=$EV/results/condition_a3_kaggle__final
until grep -q "eval rc=" /c/Users/nikma/chowder_teacher_free/eval_a3_recovered.log 2>/dev/null; do sleep 60; done
cd "$WT" || exit 1
export PYTHONPATH="$WT/src" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
echo "=== $(/usr/bin/date -u +%FT%TZ) kaggle-A3 eval start"
python -m chowder.evaluators.transformers_text_worker --spec "$DIR/eval_spec.json" \
  --result "$DIR/result.json" --chowder-identity "$EV/chowder_identity.json" > "$DIR/worker_stdout.log" 2>&1
echo "=== $(/usr/bin/date -u +%FT%TZ) kaggle-A3 eval rc=$?"
