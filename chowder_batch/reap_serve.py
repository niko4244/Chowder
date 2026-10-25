"""Serve one REAP arm with the pre-registered llama-server flags, then run reap_eval.

Usage:
  python chowder_batch/reap_serve.py --gguf <path> --arm <name> --port 8790 --out <eval dir> \
      -- --port 8790 --arm <name> --suites gsm8k --out <eval dir>

Flags are frozen by the prereg (section 3): ``--fit on -np 4 -c 16384 --jinja``.
Creates <out>/<arm>/server.log and <out>/<arm>/serve.json exclusively. Resumed
sessions retain these original receipts and use separate per-session records.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

DEFAULT_SERVER = r"C:\llama.cpp\bin\llama-server.exe"
FLAGS = ["--fit", "on", "-np", "4", "-c", "16384", "--jinja"]
HEALTH_TIMEOUT = 900

_KEEP = re.compile(r"llama_params_fit|common_fit|n_cpu_moe|overflowing|offloaded \d+/\d+ layers|exps.*(?:overridden|CPU)", re.I)
_CPU_MOE = re.compile(r"n_cpu_moe\s*[=:]?\s*(\d+)", re.I)
_OVERFLOW = re.compile(r"\((\d+) overflowing\)")
_OVERRIDE = re.compile(r"blk\.(\d+)\..*exps.*CPU", re.I)
_ALL_LAYERS = re.compile(r"offloaded (\d+)/(\d+) layers to GPU", re.I)
# llama-server logs neither the fit decision nor layer placement at its default
# verbosity (3); 4 does, at ~30 lines per request. Set through the environment so
# the pre-registered FLAGS (and inference) stay unchanged.
LOG_VERBOSITY = "4"


def parse_offload(text: str) -> tuple[list[str], int | None]:
    lines = [l.strip() for l in text.splitlines() if _KEEP.search(l)]
    moe = [int(m.group(1)) for l in lines if (m := _CPU_MOE.search(l))]
    if moe:
        return lines, moe[-1]
    over = [int(m.group(1)) for l in lines if (m := _OVERFLOW.search(l))]
    if over:
        return lines, sum(over)
    layers = {m.group(1) for l in lines if (m := _OVERRIDE.search(l))}
    if layers:
        return lines, len(layers)
    # A clean fit with every layer on the GPU and no expert tensor overridden to CPU
    # is a measured zero, not an unknown (reap160_q4 read UNKNOWN for this reason).
    full = [m for l in lines if (m := _ALL_LAYERS.search(l))]
    fitted = any("successfully fit" in l for l in lines)
    if fitted and full and full[-1].group(1) == full[-1].group(2):
        return lines, 0
    return lines, None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_stat(path: Path) -> tuple[int, int, int, int, int]:
    info = path.stat()
    return (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_dev, info.st_ino)


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


def _verify_eval_target(eval_args: list[str], *, port: int, arm: str, out: Path) -> None:
    """Bind evaluation to the just-launched server and its output identity.

    The evaluation CLI accepts its own independent ``--port``/``--arm``/``--out``.
    Without this check, serving one GGUF can silently score a different resident
    endpoint or append results under an unrelated arm.  Empty args are left to
    reap_eval's required-argument validation for backward compatibility.
    """
    if not eval_args:
        return
    for name in ("--port", "--arm", "--out"):
        count = sum(value == name or value.startswith(name + "=") for value in eval_args)
        if count != 1:
            raise ValueError(f"eval {name} must be supplied exactly once; found {count}")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suites")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--parallel", type=int)
    try:
        parsed, unknown = parser.parse_known_args(eval_args)
    except (argparse.ArgumentError, SystemExit) as exc:
        raise ValueError("eval argument syntax is invalid") from exc
    if unknown:
        raise ValueError(f"eval has unrecognized arguments: {unknown!r}")
    if (parsed.port != port or parsed.arm != arm or
            parsed.out.resolve() != out.resolve()):
        raise ValueError("eval --port/--arm/--out must match the launched REAP server and output")


def _require_free_loopback_port(port: int) -> None:
    """Reject an existing listener before startup (not a post-start PID proof)."""
    if not 1 <= port <= 65535:
        raise ValueError("server port must be in 1..65535")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"server port {port} is in use or unavailable; refusing foreign health endpoint") from exc


def _require_owned_listener(proc, port: int) -> None:
    """On the Windows execution host, bind a healthy listener to our child PID."""
    if os.name != "nt":
        return
    script = ("$ErrorActionPreference='Stop'; "
              f"Get-NetTCPConnection -State Listen -LocalPort {port} | "
              "Where-Object { $_.LocalAddress -eq '127.0.0.1' -or "
              "$_.LocalAddress -eq '0.0.0.0' } | "
              "Select-Object -ExpandProperty OwningProcess")
    try:
        check = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive",
                                "-Command", script], capture_output=True, text=True, timeout=20)
        owners = {int(line.strip()) for line in check.stdout.splitlines() if line.strip()}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise RuntimeError("could not verify Windows REAP loopback listener owner") from exc
    if check.returncode != 0 or owners != {proc.pid}:
        raise RuntimeError("healthy REAP loopback listener is not owned by launched server PID")


def _admit_output(arm_dir: Path, *, gguf: Path, digest: str) -> bool:
    """Allow a fresh directory or an original, model-bound resumable session."""
    if arm_dir.is_symlink() or not arm_dir.is_dir():
        raise ValueError("REAP arm output must be a real directory")
    existing = [path for path in arm_dir.iterdir() if path.name != ".serve.lock"]
    if not existing:
        return False
    serve_path = arm_dir / "serve.json"
    log_path = arm_dir / "server.log"
    if (not serve_path.is_file() or not log_path.is_file()
            or serve_path.is_symlink() or log_path.is_symlink()):
        raise ValueError("REAP existing output has incomplete or unclaimed serving identity")
    try:
        serving = json.loads(serve_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise ValueError("REAP existing serving identity cannot be read") from exc
    if not isinstance(serving, dict) or not isinstance(serving.get("gguf"), str):
        raise ValueError("REAP existing serving model identity is malformed")
    expected = {"gguf": serving["gguf"], "gguf_sha256": digest, "flags": FLAGS}
    if (Path(serving["gguf"]).resolve() != gguf
            or any(serving.get(k) != v for k, v in expected.items())):
        raise ValueError("REAP existing serving model identity/flags disagree; use a new output arm")

    from chowder_batch.reap_eval import IDENTITY_SCHEMA

    # Never give legacy historical results a retrospective model witness.
    for results in arm_dir.glob("*.jsonl"):
        suite = results.stem
        witness = arm_dir / f"{suite}.inputs.json"
        if results.is_symlink() or not witness.is_file() or witness.is_symlink():
            raise ValueError(f"REAP existing {suite} results lack a trustworthy input witness")
    for witness in arm_dir.glob("*.inputs.json"):
        suite = witness.name.removesuffix(".inputs.json")
        if witness.is_symlink():
            raise ValueError("REAP existing input witness cannot be a symlink")
        try:
            record = json.loads(witness.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            raise ValueError("REAP existing input witness is unreadable") from exc
        if (not isinstance(record, dict) or record.get("schema") != IDENTITY_SCHEMA
                or record.get("suite") != suite or record.get("model") != expected):
            raise ValueError("REAP existing input witness model identity disagrees")
    summary_path = arm_dir / "summary.json"
    if summary_path.exists():
        if not summary_path.is_file() or summary_path.is_symlink():
            raise ValueError("REAP existing evaluation summary is not a regular file")
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            raise ValueError("REAP existing evaluation summary is unreadable") from exc
        if not isinstance(summary, dict):
            raise ValueError("REAP existing evaluation summary is malformed")
        for suite, info in summary.items():
            results = arm_dir / f"{suite}.jsonl"
            if (not isinstance(suite, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", suite)
                    or not isinstance(info, dict) or not results.is_file()
                    or not (arm_dir / f"{suite}.inputs.json").is_file()
                    or info.get("file_sha256") != _sha256(results)):
                raise ValueError("REAP existing evaluation summary is unclaimed or modified")
    return True


def _stop(proc) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def _run_eval(server, command: list[str]) -> int:
    """Stop scoring if the model server exits before the evaluator finishes."""
    evaluator = subprocess.Popen(command)
    try:
        while True:
            try:
                code = evaluator.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if server.poll() is not None:
                    raise RuntimeError("REAP model server exited during evaluation") from None
                continue
            if server.poll() is not None:
                raise RuntimeError("REAP model server exited before evaluation completed")
            return code
    finally:
        _stop(evaluator)


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

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.arm):
        raise ValueError("REAP arm must be a simple name without path separators")
    gguf = Path(args.gguf).resolve(strict=True)
    if not gguf.is_file() or Path(args.gguf).is_symlink():
        raise ValueError("REAP GGUF must be an existing regular model file")
    _verify_eval_target(eval_args, port=args.port, arm=args.arm, out=Path(args.out))

    arm_dir = Path(args.out) / args.arm
    if arm_dir.is_symlink():
        raise ValueError("REAP arm output cannot be a symlink")
    arm_dir.mkdir(parents=True, exist_ok=True)
    lock_path = arm_dir / ".serve.lock"
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError("REAP arm already has an active or abandoned session lock") from exc
    try:
        lock.write(f"pid={os.getpid()}\n")
        lock.flush()
        model_stat = _model_stat(gguf)
        digest = _sha256(gguf)
        if _model_stat(gguf) != model_stat:
            raise RuntimeError("REAP GGUF changed while hashing; refusing serving")
        resumed = _admit_output(arm_dir, gguf=gguf, digest=digest)
        _require_free_loopback_port(args.port)
        session = uuid.uuid4().hex if resumed else None
        log_path = arm_dir / (f"server.resume-{session}.log" if resumed else "server.log")
        receipt_path = arm_dir / (f"serve.resume-{session}.json" if resumed else "serve.json")
        cmd = [args.server, *FLAGS, "--port", str(args.port), "-m", str(gguf)]
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        t0 = time.time()
        with log_path.open("xb") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "LLAMA_ARG_LOG_VERBOSITY": LOG_VERBOSITY})
            try:
                _wait_health(proc, args.port, args.timeout)
                if proc.poll() is not None:
                    raise RuntimeError("launched llama-server exited before evaluation")
                _require_owned_listener(proc, args.port)
                if _model_stat(gguf) != model_stat:
                    raise RuntimeError("REAP GGUF changed during server startup")
                ready = round(time.time() - t0, 1)
                lines, n = parse_offload(log_path.read_text(encoding="utf-8", errors="replace"))
                receipt = {
                    "gguf": str(gguf), "gguf_sha256": digest, "flags": FLAGS,
                    "log_verbosity": LOG_VERBOSITY,
                    "experts_offloaded": n, "offload_log_lines": lines,
                    "vram_used_mib": _vram_used_mib(), "started": started, "ready_seconds": ready,
                }
                if resumed:
                    receipt["session_log"] = log_path.name
                with receipt_path.open("x", encoding="utf-8") as record:
                    json.dump(receipt, record, indent=2)
                eval_py = str(Path(__file__).with_name("reap_eval.py"))
                code = _run_eval(proc, [sys.executable, eval_py, *eval_args])
                if _model_stat(gguf) != model_stat:
                    raise RuntimeError("REAP GGUF changed during evaluation")
                return code
            finally:
                _stop(proc)
    finally:
        lock.close()
        lock_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
