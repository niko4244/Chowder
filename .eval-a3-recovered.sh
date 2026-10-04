#!/usr/bin/env bash
# GSM8K eval of the recovered local A3 adapter (same spec/protocol as the base arm).
set -u
WT=/c/Users/nikma/Chowder/.claude/worktrees/tfd-revised
EV=/c/Users/nikma/chowder_teacher_free/eval_gsm8k2048
DIR=$EV/results/condition_a3__final
cd "$WT" || exit 1
export PYTHONPATH="$WT/src" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
echo "=== $(/usr/bin/date -u +%FT%TZ) eval start"
python -m chowder.evaluators.transformers_text_worker --spec "$DIR/eval_spec.json" \
  --result "$DIR/result.json" --chowder-identity "$EV/chowder_identity.json" > "$DIR/worker_stdout.log" 2>&1
echo "=== $(/usr/bin/date -u +%FT%TZ) eval rc=$?"
