#!/usr/bin/env bash
# Operator-approved sequential GPU queue (v4): the six eval specs run first
# (measured peak 3.8 GB, fits beside the resident servers), then A2 training
# waits for its own 8 GiB floor via .wait_vram.py. Nothing is ever killed or
# reconfigured; if the servers never shrink, A2 simply never launches.
set -u
WT=/c/Users/nikma/Chowder/.claude/worktrees/tfd-revised
DATA=/c/Users/nikma/chowder_teacher_free
cd "$WT" || exit 1
export PYTHONPATH="$WT/src"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

run_spec () {
  local spec="$1"; local out="$2"
  echo "=== $(date -u +%T) starting $out ==="
  python -m chowder.evaluators.transformers_text_worker     --spec "$spec" --result "$out/result.json"     --chowder-identity "$DATA/eval_run/chowder_identity.json"     > "$out/worker_stdout.log" 2>&1
  echo "=== $(date -u +%T) finished $out rc=$? ==="
}

for B in 2048 3072; do
  for ARM in base condition_a; do
    DIR="$DATA/eval_sweep$B/results/$ARM""__final"
    run_spec "$DIR/eval_spec.json" "$DIR"
  done
done
for ARM in base condition_a; do
  DIR="$DATA/eval_gsm8k/results/$ARM""__final"
  run_spec "$DIR/eval_spec.json" "$DIR"
done
echo EVALS_DONE

echo "=== $(date -u +%T) now waiting for A2's 8 GiB floor ==="
python .wait_vram.py
PYTHONIOENCODING=utf-8 python experiments/teacher_free_distill/train_pilot.py   --recipe experiments/teacher_free_distill/recipes/a2_sft_supervised.json   --data-dir "$DATA/pilot_v4"   --output "$DATA/checkpoints/cond_a2" --yes
echo "=== $(date -u +%T) A2 training rc=$? ==="
echo QUEUE_DONE
