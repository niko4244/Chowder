#!/usr/bin/env bash
# Operator-authorized paired evaluation: base and Condition A, final split.
# Sequential single-task processes on GPU 0; resident inference servers untouched.
set -u
WT=/c/Users/nikma/Chowder/.claude/worktrees/tfd-revised
RUN=/c/Users/nikma/chowder_teacher_free/eval_run
cd "$WT" || exit 1
export PYTHONPATH="$WT/src"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

run_arm () {
  local dir="$1"
  echo "=== $(date -u +%H:%M:%S) starting $dir ==="
  python -m chowder.evaluators.transformers_text_worker \
    --spec "$RUN/results/$dir/eval_spec.json" \
    --result "$RUN/results/$dir/result.json" \
    --chowder-identity "$RUN/chowder_identity.json" \
    > "$RUN/results/$dir/worker_stdout.log" 2>&1
  echo "=== $(date -u +%H:%M:%S) finished $dir rc=$? ==="
}

run_arm base__final
run_arm condition_a__final
echo DONE
