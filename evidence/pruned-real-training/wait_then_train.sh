#!/bin/bash
# Wait for the GPU to free up (Ollama loads/unloads on its own idle timeout) and
# only then start the run. Never kill the user's service to make room.
export PYTHONPATH="C:/Users/nikma/Chowder-router-healing/src"
SP="/c/Users/nikma/AppData/Local/Temp/claude/C--Users-nikma/d2ee79ac-3407-48b9-8397-d6ba96c24813/scratchpad"
NEED_FREE=9000   # MiB; Unsloth train ~6 GB, eval ~5.5 GB, plus headroom
for i in $(seq 1 240); do   # up to ~40 min of waiting
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  free=$((total - used))
  if [ "$free" -ge "$NEED_FREE" ]; then
    echo "[$(date +%H:%M:%S)] GPU free=${free} MiB >= ${NEED_FREE}; starting run"
    exec python "$SP/real_train_gsm8k.py" --engine unsloth --work "F:/llm-models/_a4b/realtrain-gsm8k"
  fi
  [ $((i % 6)) -eq 1 ] && echo "[$(date +%H:%M:%S)] waiting: free=${free} MiB (need ${NEED_FREE})"
  sleep 10
done
echo "GPU never freed up within the wait window; not started"
exit 3
