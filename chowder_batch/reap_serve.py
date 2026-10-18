"""Serve one REAP arm with the pre-registered llama-server flags, then run reap_eval.

Usage:
  python chowder_batch/reap_serve.py --gguf <path> --arm <name> --port 8790 --out <eval dir> \
      -- --port 8790 --arm <name> --suites gsm8k --out <eval dir>

Flags are frozen by the prereg (section 3): ``--fit on -np 4 -c 16384 --jinja``.
Writes <out>/<arm>/server.log and <out>/<arm>/serve.json; the server is always
terminated on exit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_SERVER = r"C:\llama.cpp\bin\llama-server.exe"
FLAGS = ["--fit", "on", "-np", "4", "-c", "16384", "--jinja"]
HEALTH_TIMEOUT = 900

_KEEP = re.compile(r"llama_params_fit|common_fit|n_cpu_moe|overflowing|offloaded \d+/\d+ layers|exps.*(?:overridden|CPU)", re.I)
_CPU_MOE = re.compile(r"n_cpu_moe\s*[=:]?\s*(\d+)", re.I)
_OVERFLOW = re.compile(r"\((\d+) overflowing\)")
_OVERRIDE = re.compile(r"blk\.(\d+)\..*exps.*CPU", re.I)


def parse_offload(text: str) -> tuple[list[str], int | None]:
    lines = [l.strip() for l in text.splitlines() if _KEEP.search(l)]
    moe = [int(m.group(1)) for l in lines if (m := _CPU_MOE.search(l))]
    if moe:
        return lines, moe[-1]
    over = [int(m.group(1)) for l in lines if (m := _OVERFLOW.search(l))]
    if over:
        return lines, sum(over)
    layers = {m.group(1) for l in lines if (m := _OVERRIDE.search(l))}
    return lines, len(layers) if layers else None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _wait_health(proc, port: int, timeout: int) -> None:
    t0 = time.time()
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {proc.poll()}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        if time.time() - t0 > timeout:
            raise TimeoutError(f"llama-server not healthy after {timeout}s")
        time.sleep(2)


def _vram_used_mib() -> int | None:
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=30)
        return int(done.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def _stop(proc) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ours, eval_args = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--port", type=int, default=8790)
    p.add_argument("--out", required=True)
    p.add_argument("--server", default=DEFAULT_SERVER)
    p.add_argument("--timeout", type=int, default=HEALTH_TIMEOUT)
    args = p.parse_args(ours)

    arm_dir = Path(args.out) / args.arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    log_path = arm_dir / "server.log"
    gguf = Path(args.gguf)
    cmd = [args.server, *FLAGS, "--port", str(args.port), "-m", str(gguf)]
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    t0 = time.time()
    with log_path.open("wb") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            _wait_health(proc, args.port, args.timeout)
            ready = round(time.time() - t0, 1)
            lines, n = parse_offload(log_path.read_text(encoding="utf-8", errors="replace"))
            (arm_dir / "serve.json").write_text(json.dumps({
                "gguf": str(gguf), "gguf_sha256": _sha256(gguf), "flags": FLAGS,
                "experts_offloaded": n, "offload_log_lines": lines,
                "vram_used_mib": _vram_used_mib(), "started": started, "ready_seconds": ready,
            }, indent=2), encoding="utf-8")
            eval_py = str(Path(__file__).with_name("reap_eval.py"))
            return subprocess.run([sys.executable, eval_py, *eval_args]).returncode
        finally:
            _stop(proc)


if __name__ == "__main__":
    raise SystemExit(main())
