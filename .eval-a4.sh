#!/usr/bin/env bash
# A4 evals, after the MATH-150 base/A3 queue releases the GPU: GSM8K (batch 1, like its other arms), then MATH-150 (batch 8).
set -u
WT=/c/Users/nikma/Chowder/.claude/worktrees/eval-7445759
cd "$WT" || exit 1
export PYTHONPATH="$WT/src" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
until grep -q MATH150_QUEUE_DONE /c/Users/nikma/chowder_teacher_free/eval_math150.log 2>/dev/null; do sleep 60; done
G=/c/Users/nikma/chowder_teacher_free/eval_gsm8k2048; M=/c/Users/nikma/chowder_teacher_free/eval_math150
for PAIR in "$G/results/condition_a4__final|$G/chowder_identity_7445759.json" "$M/results/condition_a4__final|$M/chowder_identity.json"; do
  DIR=${PAIR%%|*}; ID=${PAIR##*|}
  echo "=== $(/usr/bin/date -u +%FT%TZ) $DIR start"
  python -m chowder.evaluators.transformers_text_worker --spec "$DIR/eval_spec.json" --result "$DIR/result.json" \
    --chowder-identity "$ID" > "$DIR/worker_stdout.log" 2>&1
  echo "=== $(/usr/bin/date -u +%FT%TZ) $DIR rc=$?"
done
echo "=== A4_EVALS_DONE"
