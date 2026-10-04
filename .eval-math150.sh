#!/usr/bin/env bash
# MATH-150 (math_verify_match, 4096 tokens, batch 8) for base then Kaggle-A3; A4 is added after it trains.
# Runs from the eval worktree pinned at d98edb5 so the source identity cannot drift mid-queue.
set -u
WT=/c/Users/nikma/Chowder/.claude/worktrees/eval-d98edb5
EV=/c/Users/nikma/chowder_teacher_free/eval_math150
cd "$WT" || exit 1
export PYTHONPATH="$WT/src" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
for ARM in base condition_a3_kaggle; do
  DIR=$EV/results/${ARM}__final
  echo "=== $(/usr/bin/date -u +%FT%TZ) $ARM start"
  python -m chowder.evaluators.transformers_text_worker --spec "$DIR/eval_spec.json" \
    --result "$DIR/result.json" --chowder-identity "$EV/chowder_identity.json" > "$DIR/worker_stdout.log" 2>&1
  echo "=== $(/usr/bin/date -u +%FT%TZ) $ARM rc=$?"
done
echo "=== MATH150_QUEUE_DONE"
