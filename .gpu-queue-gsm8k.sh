#!/usr/bin/env bash
# Replacement queue (2026-09-26): the OT3 2048/3072 sweeps were stopped because
# the OT3 final split cannot support a claim (see REPORT.md correction). Runs the
# external GSM8K paired eval at a 2048 budget under the fixed scorer (EOS-gated,
# reopened-<think> is a miss), then the previously approved A2 training step.
set -u
WT=/c/Users/nikma/Chowder/.claude/worktrees/tfd-revised
DATA=/c/Users/nikma/chowder_teacher_free
EV=$DATA/eval_gsm8k2048
cd "$WT" || exit 1
export PYTHONPATH="$WT/src"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

for ARM in base condition_a; do
  DIR="$EV/results/$ARM""__final"
  echo "=== $(date -u +%T) starting $DIR ==="
  python -m chowder.evaluators.transformers_text_worker --spec "$DIR/eval_spec.json" \
    --result "$DIR/result.json" --chowder-identity "$EV/chowder_identity.json" \
    > "$DIR/worker_stdout.log" 2>&1
  echo "=== $(date -u +%T) finished $DIR rc=$? ==="
done
PYTHONIOENCODING=utf-8 python experiments/teacher_free_distill/eval_protocol.py compare --results "$EV/results"
echo EVALS_DONE

echo "=== $(date -u +%T) now waiting for A2's 8 GiB floor ==="
python .wait_vram.py
PYTHONIOENCODING=utf-8 python experiments/teacher_free_distill/train_pilot.py --recipe experiments/teacher_free_distill/recipes/a2_sft_supervised.json \
  --data-dir "$DATA/pilot_v4" --output "$DATA/checkpoints/cond_a2" --yes
echo "=== $(date -u +%T) A2 training rc=$? ==="
echo QUEUE_DONE
