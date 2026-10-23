# Passive WDDM memory counter witness (Windows only)

`chowder_batch/wddm_process_memory.py` collects read-only Windows GPU Process
Memory `Shared Usage` and `Dedicated Usage` counters for an already-running
process PID. It does **not** start model processes, load model weights or CUDA,
or consume training GPU hours. The counters are acquired by `Get-Counter` and
paired by GPU LUID/adapter instance and timestamp. PID start-time identity is
rechecked at the end to avoid accidentally measuring a reused PID.

From an isolated Chowder worktree, while a *separately authorized* model run is
already in progress:

```powershell
python -B chowder_batch/wddm_process_memory.py --pid 12345 --samples 180 \
  --interval 1 --out C:\llm-work\moe-gates\wddm-run-12345.json
```

Use an output path that does not already exist. The exact PID, process start
time, and full observed time window must be bound to the model checkpoint,
backend, optimizer, dataset and compute accounting before the result can be
interpreted as evidence for the frozen placement gate.

**This command never returns a WDDM gate `PASS`.** A zero shared-memory
counter is not sufficient: counters may underrepresent CUDA/WDDM allocations
and brief spill episodes may fall outside the sample interval. Corroborate
concurrent per-process telemetry, torch allocated/reserved peaks, driver VRAM,
host paging, full-gradient step duration, and an explicit operator verdict.
The existing frozen max-VRAM, no-WDDM-spill and per-arm GPU-hour thresholds are
unchanged. A two-sample service baseline is *not* a training qualification.
