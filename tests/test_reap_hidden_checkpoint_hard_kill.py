"""CPU-only real process-termination tests for REAP BF16 checkpoint recovery.

Unlike monkeypatched exceptions, these tests kill an actual checkpoint writer
while its OS file lock and memory mappings remain open. They do not load model
weights, initialize CUDA, or simulate sudden power loss.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from chowder_batch import reap_hidden_checkpoint as cp


IDENTITY = {
    "source_sha256": "a" * 64,
    "token_ids_sha256": "b" * 64,
    "calibration_sha256": "c" * 64,
    "lengths": [1, 1, 1],
    "num_experts": 1,
    "top_k": 1,
}


CHILD = r"""
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from chowder_batch import reap_hidden_checkpoint as cp

cp.shutil.disk_usage = lambda _: SimpleNamespace(free=100 * cp.GIB)
identity = {
    "source_sha256": "a" * 64,
    "token_ids_sha256": "b" * 64,
    "calibration_sha256": "c" * 64,
    "lengths": [1, 1, 1],
    "num_experts": 1,
    "top_k": 1,
}
scratch, stage, ready = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
ck = cp.BF16CheckpointHidden([1, 1, 1], 4, scratch, identity)
for i in range(3):
    ck[i] = torch.full((1, 1, 4), i + 1, dtype=torch.bfloat16)
ck.finish_embeddings()
ck.begin_layer(0)
for i in range(3):
    ck[i] = ck[i] + 1
    if i == 0 and stage == "mid_layer":
        with ready.open("xb") as f:
            f.write(b"MID_LAYER")
            f.flush()
            os.fsync(f.fileno())
        break
else:
    if stage != "after_commit":
        raise RuntimeError("unexpected termination stage")
    stats = {
        "count": torch.tensor([3.], dtype=torch.float64),
        "gate_sum": torch.tensor([1.], dtype=torch.float64),
        "norm_sum": torch.tensor([2.], dtype=torch.float64),
        "reap_sum": torch.tensor([0.8], dtype=torch.float64),
    }
    ck.finish_layer(0, stats, {"layer": 0, "routed_tokens": 3})
    with ready.open("xb") as f:
        f.write(b"AFTER_COMMIT")
        f.flush()
        os.fsync(f.fileno())
# The writer intentionally leaves its file mapping and OS lock open.
while True:
    time.sleep(0.05)
"""


def _complete_layer(ck: cp.BF16CheckpointHidden) -> None:
    ck.begin_layer(0)
    for i in range(3):
        ck[i] = ck[i] + 1
    stats = {
        "count": torch.tensor([3.], dtype=torch.float64),
        "gate_sum": torch.tensor([1.], dtype=torch.float64),
        "norm_sum": torch.tensor([2.], dtype=torch.float64),
        "reap_sum": torch.tensor([0.8], dtype=torch.float64),
    }
    ck.finish_layer(0, stats, {"layer": 0, "routed_tokens": 3})


@pytest.mark.parametrize("stage,expected_generation", [
    ("mid_layer", 0),
    ("after_commit", 1),
])
def test_os_killed_writer_only_recovers_committed_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    stage: str, expected_generation: int,
) -> None:
    """Windows TerminateProcess / POSIX SIGKILL, never GPU or model loading."""
    monkeypatch.setattr(
        cp.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * cp.GIB)
    )
    scratch = tmp_path / f"checkpoint-{stage}"
    ready = tmp_path / f"ready-{stage}"
    project_root = Path(__file__).resolve().parents[1]
    worker = subprocess.Popen(
        [sys.executable, "-u", "-B", "-c", CHILD, str(scratch), stage, str(ready)],
        cwd=project_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 35
        while not ready.exists() and worker.poll() is None:
            if time.monotonic() >= deadline:
                pytest.fail("child never reached the durable interruption marker")
            time.sleep(0.05)
        if not ready.exists():
            stderr = worker.communicate(timeout=5)[1]
            pytest.fail(f"writer exited before marker: {stderr[-1800:]!r}")
        assert worker.poll() is None, "writer did not survive to hard-kill point"
        worker.kill()
        worker.wait(timeout=15)
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=15)
        if worker.stderr is not None:
            worker.stderr.close()

    recovered = cp.BF16CheckpointHidden(
        [1, 1, 1], 4, scratch, IDENTITY, resume=True
    )
    try:
        assert recovered.generation == expected_generation
        assert len(recovered.timings) == expected_generation
        if expected_generation == 0:
            _complete_layer(recovered)
            assert recovered.generation == 1
        recovered.begin_final_read()
        for i in range(3):
            assert torch.equal(
                recovered[i],
                torch.full((1, 1, 4), i + 2, dtype=torch.bfloat16),
            )
    finally:
        recovered.close()
