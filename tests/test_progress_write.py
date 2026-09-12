"""Publishing progress must never be able to kill a training run.

This is a regression for a real loss, not a hypothetical: a 500-step GSM8K run died
at **step 323** with

    PermissionError: [WinError 5] Access is denied:
      '...\\adapter\\progress.tmp' -> '...\\adapter\\progress.json'

Both files survived the crash and prove where the fault was -- `progress.json` held
step 322 (loss 1.1002), `progress.tmp` held step 323 (loss 1.0737). The payload was
written correctly; only the rename failed, and 323 steps of real training were
discarded because a telemetry write raised inside `TrainerCallback.on_log`, which
propagates out of `Trainer.train()`. A re-run under the same config reached 1.0750 at
step 323, within 0.1% of the recovered value.
"""

import json
import os
from pathlib import Path

import pytest

from chowder.progress_write import write_progress_best_effort

PAYLOAD = {"step": 323, "max_steps": 500, "loss": 1.073659062385559}


def test_publishes_the_payload_and_leaves_no_temp_file(tmp_path):
    final = tmp_path / "progress.json"
    assert write_progress_best_effort(PAYLOAD, final) is True
    assert json.loads(final.read_text(encoding="utf-8")) == PAYLOAD
    assert not final.with_suffix(".tmp").exists()


def test_overwrites_a_previous_publish(tmp_path):
    final = tmp_path / "progress.json"
    write_progress_best_effort({"step": 322, "loss": 1.1002}, final)
    write_progress_best_effort(PAYLOAD, final)
    assert json.loads(final.read_text(encoding="utf-8"))["step"] == 323


def test_the_winerror_5_that_killed_the_run_does_not_raise(tmp_path, monkeypatch):
    """The exact failure: os.replace refuses with a sharing violation."""
    final = tmp_path / "progress.json"

    def deny(src, dst):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(os, "replace", deny)
    # must return False rather than propagate -- propagating is what lost 323 steps
    assert write_progress_best_effort(PAYLOAD, final) is False
    assert not final.exists()


def test_a_transient_failure_is_retried_and_then_succeeds(tmp_path, monkeypatch):
    """An antivirus or indexer handle clears in milliseconds, so a retry is the
    difference between losing a run and a momentary hiccup."""
    final = tmp_path / "progress.json"
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(5, "Access is denied")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    assert write_progress_best_effort(PAYLOAD, final) is True
    assert calls["n"] == 2
    assert json.loads(final.read_text(encoding="utf-8"))["step"] == 323


def test_an_unwritable_directory_does_not_raise(tmp_path, monkeypatch):
    """A vanished or read-only run directory is also not worth a run."""
    final = tmp_path / "gone" / "progress.json"
    assert write_progress_best_effort(PAYLOAD, final) is False


def test_a_non_os_error_is_also_contained(tmp_path, monkeypatch):
    final = tmp_path / "progress.json"

    def explode(src, dst):
        raise RuntimeError("something exotic")

    monkeypatch.setattr(os, "replace", explode)
    assert write_progress_best_effort(PAYLOAD, final) is False


def test_neither_worker_publishes_progress_unguarded():
    """The guard has to be in BOTH workers. unsloth_worker inlines it because its
    docstring forbids importing from the chowder package, so it is checked by
    source; transformers_worker imports the helper."""
    import chowder

    src = Path(chowder.__file__).resolve().parent / "backends"

    transformers = (src / "transformers_worker.py").read_text(encoding="utf-8")
    assert "write_progress_best_effort" in transformers
    assert "tmp_path.replace(self._progress_path)" not in transformers, (
        "transformers_worker still renames progress unguarded"
    )

    unsloth = (src / "unsloth_worker.py").read_text(encoding="utf-8")
    assert "tmp_path.replace(progress_path)" not in unsloth, (
        "unsloth_worker still renames progress unguarded"
    )
    # the inlined guard: retries, contains OSError, and counts persistent failure
    assert "except OSError" in unsloth
    assert "progress_failures" in unsloth
    assert "progress_write_failures" in unsloth, "failures must reach telemetry"
    # and it still imports nothing from chowder
    assert "from chowder" not in unsloth and "from ..progress_write" not in unsloth
