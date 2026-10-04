#!/usr/bin/env bash
# A3 queue (operator-approved 2026-09-27): train A3 on complete OpenR1 traces, evaluate it
# on the pinned GSM8K set (same spec as the base arm, which is reused), then ALWAYS resume
# the paused GPU services -- even if training or eval fails.
set -u
WT=/c/Users/nikma/Chowder/.claude/worktrees/tfd-revised
DATA=/c/Users/nikma/chowder_teacher_free
EV=$DATA/eval_gsm8k2048
ts () { /usr/bin/date -u +%FT%TZ; }
cd "$WT" || exit 1
export PYTHONPATH="$WT/src" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8

echo "=== $(ts) A3 training start"
python experiments/teacher_free_distill/train_pilot.py \
  --recipe experiments/teacher_free_distill/recipes/a3_sft_openr1_complete.json \
  --data-dir "$DATA/openr1_pilot" --output "$DATA/checkpoints/cond_a3" --yes > "$DATA/a3_train.log" 2>&1
rc=$?; echo "=== $(ts) A3 training rc=$rc"

if [ $rc -eq 0 ]; then
  DIR="$EV/results/condition_a3__final"
  echo "=== $(ts) A3 GSM8K eval start"
  python -m chowder.evaluators.transformers_text_worker --spec "$DIR/eval_spec.json" \
    --result "$DIR/result.json" --chowder-identity "$EV/chowder_identity.json" > "$DIR/worker_stdout.log" 2>&1
  echo "=== $(ts) A3 GSM8K eval rc=$?"
fi

echo "=== $(ts) resuming services"
rm -f "$USERPROFILE/.hermes/state/service-pause/8081.pause"   # brainz-daemon restarts Gemma-E2B next cycle
wsl.exe -d Ubuntu -- sh -c 'cd /mnt/i/llm-models/sharp-spark-x2.5-4b && setsid nohup ./build/bin/llama-server -m /mnt/i/llm-models/sharp-spark-x2.5-4b/Sharp-Spark-X2.5-4B-Q6_K_XL.gguf --alias sharpspark-x2.5-4b --host 127.0.0.1 --port 8765 -ngl 99 -c 131072 -t 8 --split-mode none --jinja --temp 0.6 --top-p 0.95 --top-k 20 --flash-attn on -np 2 > /tmp/spark8765.log 2>&1 &'
echo "=== $(ts) QUEUE_DONE"
