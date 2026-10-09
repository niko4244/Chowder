"""Remote dispatch of a Chowder job to a Kaggle GPU kernel.

`kaggle_launcher` and the `kaggle/*.py` scripts run *inside* a Kaggle
notebook; nothing in the repository could start one. This module is that
missing half: it stages a script into a kernel folder, pushes it through
the official `kaggle` CLI, polls it to a terminal state and pulls its
output back, returning a `KaggleJobRecord` a ledger can persist.

Boundaries
----------
- It shells out to the `kaggle` CLI (the same dependency
  `kaggle/upload_protected_suite.py` already declares) and never reads,
  logs or copies `~/.kaggle/kaggle.json` / `KAGGLE_API_TOKEN` itself.
- Every CLI call goes through an injectable `runner`, so the push /
  poll / pull plumbing is fully testable with no network and no token.
- A job always carries a hard `timeout_seconds` (passed as `kaggle
  kernels push -t`), because every second a kernel runs is charged to
  the weekly accelerator quota whether or not it produces anything.
- A kernel that ends in any state other than `complete` is reported as
  a failure with its log attached, never as a success with missing
  output.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

#: Staged kernel script for the 2xT4 worker smoke test (repo checkout / editable install).
SMOKE_SCRIPT = Path(__file__).resolve().parents[2] / "kaggle" / "smoke_qwen3_30b_a3b.py"

#: `machine_shape` / `--accelerator` value for the "GPU T4 x2" notebook
#: option (the only GPU shape since P100 retirement on 2026-09-15).
ACCELERATOR_T4X2 = "NvidiaTeslaT4"

TERMINAL_OK = "complete"
TERMINAL_FAILED = frozenset({"error", "cancelAcknowledged", "cancelRequested", "cancelled"})
_RUNNING = frozenset({"queued", "running", "new"})
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{3,48}[a-z0-9]$")
_STATUS_RE = re.compile(r'has status "?([A-Za-z_.]+)"?')

Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]


class KaggleDispatchError(RuntimeError):
    """A push, poll or pull failed, or the kernel did not complete."""


@dataclass(frozen=True)
class KaggleJobSpec:
    owner: str
    slug: str
    title: str
    script: Path
    timeout_seconds: int
    accelerator: str = ACCELERATOR_T4X2
    enable_internet: bool = True
    is_private: bool = True
    dataset_sources: tuple[str, ...] = ()
    model_sources: tuple[str, ...] = ()
    kernel_sources: tuple[str, ...] = ()
    extra_files: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if not _SLUG_RE.match(self.slug):
            raise ValueError(f"slug must be 5-50 chars of [a-z0-9-]: {self.slug!r}")
        if self.timeout_seconds <= 0 or self.timeout_seconds > 12 * 3600:
            raise ValueError("timeout_seconds must be in (0, 43200] -- Kaggle's 12h session cap")
        if self.script.suffix != ".py":
            raise ValueError(f"script must be a .py file: {self.script}")

    @property
    def kernel_ref(self) -> str:
        return f"{self.owner}/{self.slug}"

    def metadata(self) -> dict:
        return {
            "id": self.kernel_ref,
            "title": self.title,
            "code_file": self.script.name,
            "language": "python",
            "kernel_type": "script",
            "is_private": self.is_private,
            "enable_gpu": True,
            "enable_internet": self.enable_internet,
            "machine_shape": self.accelerator,
            "dataset_sources": list(self.dataset_sources),
            "model_sources": list(self.model_sources),
            "kernel_sources": list(self.kernel_sources),
            "competition_sources": [],
        }


@dataclass
class KaggleJobRecord:
    kernel_ref: str
    accelerator: str
    timeout_seconds: int
    pushed_at: float
    finished_at: float | None = None
    status: str = "pushed"
    output_dir: str | None = None
    output_files: list[str] = field(default_factory=list)
    push_stdout: str = ""
    error: str | None = None

    @property
    def wall_seconds(self) -> float | None:
        return None if self.finished_at is None else self.finished_at - self.pushed_at

    def to_dict(self) -> dict:
        data = asdict(self)
        data["wall_seconds"] = self.wall_seconds
        return data


def _default_runner(args: Sequence[str]) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(list(args), capture_output=True, text=True, timeout=600)


def _kaggle(runner: Runner, *args: str) -> str:
    result = runner(["kaggle", *args])
    if result.returncode != 0:
        raise KaggleDispatchError(
            f"`kaggle {' '.join(args)}` exited {result.returncode}: "
            f"{(result.stderr or result.stdout).strip()[-2000:]}"
        )
    return result.stdout


def stage_kernel(spec: KaggleJobSpec, staging_dir: Path) -> Path:
    """Write the script, extra files and kernel-metadata.json to a clean folder."""
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    shutil.copy2(spec.script, staging_dir / spec.script.name)
    for extra in spec.extra_files:
        shutil.copy2(extra, staging_dir / extra.name)
    (staging_dir / "kernel-metadata.json").write_text(
        json.dumps(spec.metadata(), indent=2), encoding="utf-8"
    )
    return staging_dir


def parse_status(stdout: str) -> str:
    match = _STATUS_RE.search(stdout)
    if not match:
        raise KaggleDispatchError(f"unrecognised `kaggle kernels status` output: {stdout.strip()!r}")
    raw = match.group(1)
    # CLI prints either `complete` or the enum form `KernelWorkerStatus.COMPLETE`.
    tail = raw.split(".")[-1]
    return tail.lower() if tail.isupper() else tail


def push(spec: KaggleJobSpec, staging_dir: Path, *, runner: Runner = _default_runner,
         clock: Callable[[], float] = time.time) -> KaggleJobRecord:
    stage_kernel(spec, staging_dir)
    stdout = _kaggle(
        runner, "kernels", "push", "-p", str(staging_dir),
        "-t", str(spec.timeout_seconds), "--accelerator", spec.accelerator,
    )
    if "error" in stdout.lower() and "successfully" not in stdout.lower():
        raise KaggleDispatchError(f"kernel push reported an error: {stdout.strip()}")
    return KaggleJobRecord(
        kernel_ref=spec.kernel_ref, accelerator=spec.accelerator,
        timeout_seconds=spec.timeout_seconds, pushed_at=clock(), push_stdout=stdout.strip(),
    )


def wait(record: KaggleJobRecord, *, runner: Runner = _default_runner, poll_seconds: float = 60.0,
         clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
         grace_seconds: float = 900.0) -> KaggleJobRecord:
    """Poll until terminal. Gives up `grace_seconds` past the job's own timeout."""
    deadline = record.pushed_at + record.timeout_seconds + grace_seconds
    while True:
        status = parse_status(_kaggle(runner, "kernels", "status", record.kernel_ref))
        record.status = status
        if status == TERMINAL_OK or status in TERMINAL_FAILED:
            record.finished_at = clock()
            return record
        if status not in _RUNNING:
            raise KaggleDispatchError(f"unexpected kernel status {status!r}")
        if clock() > deadline:
            record.error = f"still {status!r} past timeout+grace; not waiting further"
            record.finished_at = clock()
            return record
        sleep(poll_seconds)


def pull_output(record: KaggleJobRecord, output_dir: Path, *, runner: Runner = _default_runner) -> KaggleJobRecord:
    output_dir.mkdir(parents=True, exist_ok=True)
    _kaggle(runner, "kernels", "output", record.kernel_ref, "-p", str(output_dir), "-o", "-q")
    record.output_dir = str(output_dir)
    record.output_files = sorted(
        str(p.relative_to(output_dir)) for p in output_dir.rglob("*") if p.is_file()
    )
    return record


def run_job(spec: KaggleJobSpec, work_dir: Path, *, runner: Runner = _default_runner,
            poll_seconds: float = 60.0, clock: Callable[[], float] = time.time,
            sleep: Callable[[float], None] = time.sleep) -> KaggleJobRecord:
    """push -> wait -> pull, writing `job_record.json` beside the output.

    Output (including the kernel log) is pulled on failure too, so the
    record always carries the evidence; the error is raised afterwards.
    """
    record = push(spec, work_dir / "kernel", runner=runner, clock=clock)
    wait(record, runner=runner, poll_seconds=poll_seconds, clock=clock, sleep=sleep)
    try:
        pull_output(record, work_dir / "output", runner=runner)
    except KaggleDispatchError as exc:
        record.error = (record.error + "; " if record.error else "") + f"output pull failed: {exc}"
    (work_dir / "job_record.json").write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    if record.status != TERMINAL_OK or record.error:
        raise KaggleDispatchError(
            f"{record.kernel_ref} ended {record.status!r}"
            + (f" ({record.error})" if record.error else "")
            + f"; see {work_dir / 'job_record.json'}"
        )
    return record


def quota(*, runner: Runner = _default_runner) -> str:
    """Raw `kaggle quota` text (weekly GPU hours used / remaining)."""
    return _kaggle(runner, "quota").strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m chowder.kaggle_dispatch",
                                     description="Dispatch jobs to a Kaggle 2xT4 GPU kernel")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("quota", help="Show weekly Kaggle GPU quota")
    smoke = sub.add_parser("smoke", help="Push the 2xT4 Qwen3-30B-A3B smoke kernel, wait, pull evidence")
    smoke.add_argument("--owner", default=None, help="Kaggle username (default: $KAGGLE_USERNAME)")
    smoke.add_argument("--slug", default="chowder-smoke-2xt4")
    smoke.add_argument("--script", default=str(SMOKE_SCRIPT))
    smoke.add_argument("--timeout-minutes", type=int, default=45, help="Hard cap charged to quota")
    smoke.add_argument("--poll-seconds", type=float, default=60.0)
    smoke.add_argument("--model-source", action="append", default=[], help="Kaggle Models mount, repeatable")
    smoke.add_argument("--work-dir", default="runs/kaggle")
    return parser


def main(argv: Sequence[str] | None = None, *, runner: Runner = _default_runner) -> int:
    import sys

    args = build_parser().parse_args(argv)
    try:
        if args.command == "quota":
            print(quota(runner=runner))
            return 0
        owner = args.owner or os.environ.get("KAGGLE_USERNAME")
        if not owner:
            print("--owner (or KAGGLE_USERNAME) is required", file=sys.stderr)
            return 2
        script = Path(args.script)
        if not script.is_file():
            print(f"smoke script not found: {script}", file=sys.stderr)
            return 2
        spec = KaggleJobSpec(
            owner=owner, slug=args.slug, title=args.slug.replace("-", " ").title(),
            script=script, timeout_seconds=args.timeout_minutes * 60,
            model_sources=tuple(args.model_source),
        )
        work_dir = Path(args.work_dir) / time.strftime("%Y%m%d-%H%M%S")
        print(f"pushing {spec.kernel_ref} ({spec.accelerator}, cap {args.timeout_minutes} min) -> {work_dir}")
        record = run_job(spec, work_dir, runner=runner, poll_seconds=args.poll_seconds)
    except (KaggleDispatchError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(record.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
