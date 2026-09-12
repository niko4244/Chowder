"""Publishing training progress must never be able to kill the training run.

Both training workers published progress by writing a temp file and renaming it
over the live one, unguarded. On Windows that rename can fail with a sharing
violation even though the write succeeded, and an exception inside a
`TrainerCallback.on_log` propagates straight out of `Trainer.train()`.

It happened, and it cost a real run: a 500-step GSM8K training job died at **step
323 of 500** after 16.4 minutes with

    PermissionError: [WinError 5] Access is denied:
      '...\\adapter\\progress.tmp' -> '...\\adapter\\progress.json'

The forensic evidence is unambiguous about where the fault was. Both files
survived: `progress.json` held step 322 (loss 1.1002) and `progress.tmp` held step
323 (loss 1.0737). The payload was written correctly; only the rename failed. Loss
was falling steadily from ~4.8, the adapter directory was left empty, and 323 steps
of real training were discarded because a *telemetry* write raised.

`transformers_worker` carried the same pattern with a comment asserting the rename
is "atomic on POSIX/NTFS". That is true of the *semantics* when it succeeds, and
says nothing about whether it can fail — which is the assumption that broke.

So: retry briefly (a transient antivirus or indexer handle clears in milliseconds),
then give up and keep training. Failures are counted and reported rather than
swallowed, because a progress path that is *always* unwritable is a real problem
worth seeing — just not worth destroying a run over.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

#: Brief, because this runs inside a training callback on the hot path.
_ATTEMPTS = 3
_BACKOFF_SECONDS = 0.05


def write_progress_best_effort(
    payload: Mapping[str, Any], final_path: str | Path, *, attempts: int = _ATTEMPTS
) -> bool:
    """Publish `payload` as JSON at `final_path`. Never raises on an OS error.

    Returns True when the file was published, False when it was not. A False means
    the caller should count it; it must never mean the caller stops training.
    """
    final = Path(final_path)
    tmp = final.with_suffix(".tmp")
    blob = json.dumps(dict(payload))

    for attempt in range(attempts):
        try:
            tmp.write_text(blob, encoding="utf-8")
            os.replace(tmp, final)
            return True
        except OSError:
            # Sharing violation, a vanished directory, a full disk -- none of these
            # are worth losing the run over. Back off and try again.
            if attempt + 1 < attempts:
                time.sleep(_BACKOFF_SECONDS)
        except Exception:  # pragma: no cover - defensive: telemetry is never fatal
            return False
    return False
