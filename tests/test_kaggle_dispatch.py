"""`chowder.kaggle_dispatch` push/poll/pull plumbing, with a fake `kaggle` CLI."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from chowder.kaggle_dispatch import (
    ACCELERATOR_T4X2,
    KaggleDispatchError,
    SMOKE_SCRIPT,
    KaggleJobSpec,
    build_parser,
    main,
    parse_status,
    run_job,
    stage_kernel,
)


def _spec(tmp_path: Path, **kw) -> KaggleJobSpec:
    script = tmp_path / "job.py"
    script.write_text("print('hi')\n")
    return KaggleJobSpec(owner="nik", slug="chowder-smoke-2xt4", title="Smoke", script=script,
                         timeout_seconds=600, **kw)


class FakeKaggle:
    def __init__(self, statuses, output_files=("smoke_result.json",), push_rc=0):
        self.statuses = list(statuses)
        self.output_files = output_files
        self.push_rc = push_rc
        self.calls: list[list[str]] = []

    def __call__(self, args):
        args = list(args)
        self.calls.append(args)
        verb = args[2]
        if verb == "push":
            return subprocess.CompletedProcess(args, self.push_rc, "Kernel version 1 successfully pushed.", "boom")
        if verb == "status":
            return subprocess.CompletedProcess(args, 0, f'nik/chowder-smoke-2xt4 has status "{self.statuses.pop(0)}"', "")
        if verb == "output":
            dest = Path(args[args.index("-p") + 1])
            for name in self.output_files:
                (dest / name).write_text("{}")
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(args)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_metadata_requests_private_t4x2_gpu(tmp_path):
    meta = json.loads((stage_kernel(_spec(tmp_path, model_sources=("a/b/c/d",)), tmp_path / "k")
                       / "kernel-metadata.json").read_text())
    assert meta["id"] == "nik/chowder-smoke-2xt4"
    assert meta["machine_shape"] == ACCELERATOR_T4X2
    assert meta["enable_gpu"] is True and meta["is_private"] is True
    assert meta["code_file"] == "job.py" and meta["model_sources"] == ["a/b/c/d"]
    assert (tmp_path / "k" / "job.py").is_file()


@pytest.mark.parametrize("bad", [{"timeout_seconds": 0}, {"timeout_seconds": 13 * 3600}])
def test_spec_refuses_unbounded_or_over_cap_timeouts(tmp_path, bad):
    script = tmp_path / "job.py"
    script.write_text("")
    with pytest.raises(ValueError):
        KaggleJobSpec(owner="nik", slug="chowder-smoke", title="t", script=script, **bad)


def test_spec_refuses_bad_slug(tmp_path):
    with pytest.raises(ValueError):
        _spec(tmp_path).__class__(owner="n", slug="Bad Slug!", title="t", script=tmp_path / "job.py",
                                  timeout_seconds=60)


@pytest.mark.parametrize("text,expected", [
    ('u/k has status "complete"', "complete"),
    ('u/k has status "KernelWorkerStatus.RUNNING"', "running"),
    ('u/k has status "cancelAcknowledged"', "cancelAcknowledged"),
    ("u/k has status running", "running"),
])
def test_parse_status_accepts_cli_formats(text, expected):
    assert parse_status(text) == expected


def test_parse_status_rejects_garbage():
    with pytest.raises(KaggleDispatchError):
        parse_status("403 Forbidden")


def test_run_job_success_pushes_with_timeout_and_accelerator_then_pulls(tmp_path):
    fake, clock = FakeKaggle(["queued", "running", "complete"]), Clock()
    record = run_job(_spec(tmp_path), tmp_path / "w", runner=fake, poll_seconds=30, clock=clock, sleep=clock.sleep)
    push = fake.calls[0]
    assert push[:3] == ["kaggle", "kernels", "push"]
    assert push[push.index("-t") + 1] == "600" and push[push.index("--accelerator") + 1] == ACCELERATOR_T4X2
    assert record.status == "complete" and record.output_files == ["smoke_result.json"]
    assert record.wall_seconds == 60
    assert json.loads((tmp_path / "w" / "job_record.json").read_text())["status"] == "complete"


def test_failed_kernel_still_pulls_log_then_raises(tmp_path):
    fake, clock = FakeKaggle(["running", "error"], output_files=("chowder-smoke-2xt4.log",)), Clock()
    with pytest.raises(KaggleDispatchError, match="error"):
        run_job(_spec(tmp_path), tmp_path / "w", runner=fake, clock=clock, sleep=clock.sleep)
    rec = json.loads((tmp_path / "w" / "job_record.json").read_text())
    assert rec["status"] == "error" and rec["output_files"] == ["chowder-smoke-2xt4.log"]


def test_stuck_kernel_stops_polling_after_timeout_plus_grace(tmp_path):
    fake, clock = FakeKaggle(["running"] * 100), Clock()
    with pytest.raises(KaggleDispatchError, match="past timeout"):
        run_job(_spec(tmp_path), tmp_path / "w", runner=fake, poll_seconds=300, clock=clock, sleep=clock.sleep)
    assert len([c for c in fake.calls if c[2] == "status"]) < 100


def test_push_failure_raises_before_polling(tmp_path):
    fake = FakeKaggle([], push_rc=1)
    with pytest.raises(KaggleDispatchError, match="exited 1"):
        run_job(_spec(tmp_path), tmp_path / "w", runner=fake)
    assert len(fake.calls) == 1


def test_cli_parses_smoke_and_points_at_the_staged_script():
    args = build_parser().parse_args(["smoke", "--owner", "nik", "--timeout-minutes", "30"])
    assert args.timeout_minutes == 30 and args.slug == "chowder-smoke-2xt4"
    assert Path(args.script) == SMOKE_SCRIPT and SMOKE_SCRIPT.is_file()


def test_cli_smoke_requires_owner(monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    assert main(["smoke"], runner=FakeKaggle([])) == 2


def test_cli_smoke_end_to_end_with_fake_kaggle(tmp_path, capsys):
    rc = main(["smoke", "--owner", "nik", "--work-dir", str(tmp_path), "--poll-seconds", "0"],
              runner=FakeKaggle(["complete"]))
    assert rc == 0 and '"status": "complete"' in capsys.readouterr().out


def test_smoke_script_help_runs_without_gpu():
    result = subprocess.run([sys.executable, str(SMOKE_SCRIPT), "--help"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
