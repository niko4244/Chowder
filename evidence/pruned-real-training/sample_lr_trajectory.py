"""Sample the live learning rate so the schedule can be verified, not assumed.

`unsloth_worker`'s progress callback publishes {step, loss, learning_rate} every
optimiser step (logging_steps=1) but OVERWRITES progress.json each time, so the
trajectory only exists if something watches it. This records one row per distinct
step into a JSONL.

Why it is worth sampling at all: until 2026-09-11 the Unsloth worker silently
dropped `lr_scheduler_type`, so a recipe asking for cosine trained on linear and
nothing in the config, the telemetry, or the result said otherwise. The config now
carries "cosine"; that proves the key reached the project, NOT that it reached the
optimiser. Only the realised LR curve proves that.

Stops when the adapter's step count reaches max_steps, when the run's report
appears, or after a long stall -- so it cannot outlive the training leg.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

WORK = Path(r"F:\llm-models\_a4b\realtrain-gsm8k-2")
OUT = Path(r"F:\llm-models\_a4b\lr-trajectory.jsonl")
REPORT = WORK / "realtrain-report.json"
POLL_SECONDS = 1.0
STALL_LIMIT_SECONDS = 900.0  # 15 min with no new step ends it


def find_progress() -> Path | None:
    for candidate in WORK.rglob("progress.json"):
        return candidate
    return None


def main() -> int:
    print("waiting for training to start (progress.json)...", flush=True)
    progress = None
    while progress is None:
        if REPORT.is_file():
            print("report appeared before any training step; nothing to sample", flush=True)
            return 0
        progress = find_progress()
        if progress is None:
            time.sleep(POLL_SECONDS * 3)
    print(f"sampling {progress}", flush=True)

    seen: set[int] = set()
    rows: list[dict] = []
    last_new = time.time()
    with OUT.open("w", encoding="utf-8") as handle:
        while True:
            try:
                payload = json.loads(progress.read_text(encoding="utf-8"))
            except Exception:
                # mid-rename or partial write: just try again
                time.sleep(POLL_SECONDS)
                continue
            step = payload.get("step")
            if isinstance(step, int) and step not in seen:
                seen.add(step)
                row = {
                    "step": step,
                    "learning_rate": payload.get("learning_rate"),
                    "loss": payload.get("loss"),
                    "max_steps": payload.get("max_steps"),
                    "wall_seconds": payload.get("wall_seconds"),
                }
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                last_new = time.time()
                if len(rows) % 50 == 0:
                    print(f"  {len(rows)} steps sampled (latest step {step})", flush=True)
                max_steps = payload.get("max_steps")
                if isinstance(max_steps, int) and step >= max_steps:
                    print(f"reached max_steps {max_steps}; {len(rows)} rows -> {OUT}", flush=True)
                    return 0
            if REPORT.is_file():
                print(f"report appeared; {len(rows)} rows -> {OUT}", flush=True)
                return 0
            if time.time() - last_new > STALL_LIMIT_SECONDS:
                print(f"stalled {STALL_LIMIT_SECONDS:.0f}s with no new step; "
                      f"{len(rows)} rows -> {OUT}", flush=True)
                return 0
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
