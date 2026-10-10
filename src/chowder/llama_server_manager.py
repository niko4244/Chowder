"""llama-server lifecycle management for teacher serving and local evaluation.

Chowder drives llama.cpp's ``llama-server`` as an out-of-process service:
teacher chain generation, GGUF-quantized evaluation, and any other HTTP
inference that must not live inside the training process. This module owns
exactly the servers it starts:

- **Managed instances are recorded.** Every ``start`` writes a state file
  (pid, port, model, timestamps) under the state directory. Only servers
  recorded there are managed; foreign llama-servers on the machine are never
  touched.
- **Start refuses collisions instead of thrashing.** If the port already
  answers, or a managed instance with the same name is alive, or VRAM cannot
  plausibly hold the model, ``start`` raises :class:`LlamaServerError` with
  the evidence. It never kills an existing process to make room.
- **Stop is pid-scoped.** ``stop`` terminates the recorded pid, waits, and
  escalates to kill. A stale state file (pid already dead) is cleaned up
  silently.

VRAM arbitration is deliberately conservative: it reads free VRAM from
``nvidia-smi`` and refuses to start when the estimated model footprint does
not fit. Estimation uses the caller-supplied ``expected_vram_mib`` when
given, otherwise the GGUF file size plus 20% for KV cache and runtime
overhead (only for full offload, ``n_gpu_layers >= 99``).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_SERVER_BIN = Path(
    "C:/Users/nikma/AppData/Local/Microsoft/WinGet/Packages/"
    "ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe/llama-server.exe"
)
DEFAULT_STATE_DIR = Path.home() / ".chowder" / "llama-servers"
HEALTH_POLL_SECONDS = 0.5
VRAM_OVERHEAD_FACTOR = 1.2


class LlamaServerError(RuntimeError):
    """Raised for lifecycle failures: collisions, unhealthy starts, bad specs."""


@dataclass
class ServerSpec:
    """Everything needed to launch one llama-server instance."""

    name: str
    model_path: Path
    port: int
    n_gpu_layers: int = 99
    ctx_size: int = 4096
    threads: int = 8
    host: str = "127.0.0.1"
    gpu_index: int | None = None
    expected_vram_mib: int | None = None
    extra_args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)

    def command(self, server_bin: Path) -> list[str]:
        args = [
            str(server_bin),
            "-m",
            str(self.model_path),
            "--port",
            str(self.port),
            "--host",
            self.host,
            "-ngl",
            str(self.n_gpu_layers),
            "-c",
            str(self.ctx_size),
            "--threads",
            str(self.threads),
            *self.extra_args,
        ]
        return args

    def estimate_vram_mib(self) -> int | None:
        """Best-effort VRAM footprint estimate in MiB, or None if unknown."""
        if self.expected_vram_mib is not None:
            return self.expected_vram_mib
        if not self.model_path.exists():
            return None
        size_mib = self.model_path.stat().st_size / (1024 * 1024)
        if self.n_gpu_layers >= 99:
            return int(size_mib * VRAM_OVERHEAD_FACTOR)
        return None  # partial offload: unknown split, arbitration skipped


@dataclass
class ServerHandle:
    """A running (or recorded) managed server instance."""

    spec: ServerSpec
    pid: int
    started_at: float
    log_path: Path

    def to_dict(self) -> dict:
        data = asdict(self)
        data["spec"] = asdict(self.spec)
        data["spec"]["model_path"] = str(self.spec.model_path)
        data["log_path"] = str(self.log_path)
        return data


def _http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def query_gpu_vram_mib() -> dict[int, tuple[int, int]]:
    """Return {gpu_index: (used_mib, total_mib)} via nvidia-smi, or {} if unavailable."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    gpus: dict[int, tuple[int, int]] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            gpus[int(parts[0])] = (int(parts[1]), int(parts[2]))
        except ValueError:
            continue
    return gpus


class LlamaServerManager:
    """Start, health-check, stop, and list llama-server instances it owns."""

    def __init__(
        self,
        server_bin: Path | None = None,
        state_dir: Path | None = None,
        health_base_url: str | None = None,
    ) -> None:
        self.server_bin = Path(server_bin) if server_bin else DEFAULT_SERVER_BIN
        if not self.server_bin.exists():
            raise LlamaServerError(
                f"llama-server binary not found at {self.server_bin}; "
                "install llama.cpp (winget install ggml.llamacpp) or pass server_bin"
            )
        self.state_dir = Path(state_dir) if state_dir else DEFAULT_STATE_DIR
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # Test seam: health checks go through this callable.
        self._health_ok = (
            (lambda url: _http_ok(url)) if health_base_url is None else (lambda url: _http_ok(health_base_url))
        )

    # ---------------------------------------------------------------- state

    def _state_path(self, name: str) -> Path:
        return self.state_dir / f"{name}.json"

    def _load_handle(self, name: str) -> ServerHandle | None:
        path = self._state_path(name)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            spec = ServerSpec(
                name=data["spec"]["name"],
                model_path=Path(data["spec"]["model_path"]),
                port=data["spec"]["port"],
                n_gpu_layers=data["spec"].get("n_gpu_layers", 99),
                ctx_size=data["spec"].get("ctx_size", 4096),
                threads=data["spec"].get("threads", 8),
                host=data["spec"].get("host", "127.0.0.1"),
                gpu_index=data["spec"].get("gpu_index"),
                expected_vram_mib=data["spec"].get("expected_vram_mib"),
                extra_args=data["spec"].get("extra_args", []),
            )
            return ServerHandle(
                spec=spec,
                pid=data["pid"],
                started_at=data["started_at"],
                log_path=Path(data["log_path"]),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def _save_handle(self, handle: ServerHandle) -> None:
        self._state_path(handle.spec.name).write_text(
            json.dumps(handle.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    def _pid_alive(self, pid: int) -> bool:
        # On Windows os.kill(pid, 0) is NOT a liveness probe: it calls
        # TerminateProcess and kills the target. Use OpenProcess instead.
        if os.name == "nt":
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return False
                return exit_code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    # --------------------------------------------------------------- start

    def start(self, spec: ServerSpec, timeout_s: float = 300.0) -> ServerHandle:
        """Launch a managed server and wait until /health answers.

        Raises LlamaServerError (leaving nothing behind) when the port is
        occupied, a same-name managed instance is alive, VRAM cannot fit the
        estimate, or the server fails its health check within timeout_s.
        """
        if not spec.model_path.exists():
            raise LlamaServerError(f"model file not found: {spec.model_path}")

        if self._health_ok(f"http://{spec.host}:{spec.port}/health"):
            raise LlamaServerError(
                f"port {spec.port} already answers /health; refusing to start "
                f"'{spec.name}' over an unknown live server"
            )

        existing = self._load_handle(spec.name)
        if existing and self._pid_alive(existing.pid):
            raise LlamaServerError(
                f"managed server '{spec.name}' already running (pid {existing.pid}, "
                f"port {existing.spec.port}); stop it first or pick another name"
            )

        estimate = spec.estimate_vram_mib()
        if estimate is not None and spec.gpu_index is not None:
            gpus = query_gpu_vram_mib()
            if spec.gpu_index in gpus:
                used, total = gpus[spec.gpu_index]
                free = total - used
                if estimate > free:
                    raise LlamaServerError(
                        f"VRAM arbitration: gpu {spec.gpu_index} has {free} MiB free "
                        f"but '{spec.model_path.name}' needs an estimated {estimate} MiB; "
                        "free VRAM or lower the footprint (quantize, smaller ctx) first"
                    )

        log_path = self.state_dir / f"{spec.name}.log"
        env = os.environ.copy()
        if spec.gpu_index is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(spec.gpu_index)

        with log_path.open("wb") as log:
            process = subprocess.Popen(
                spec.command(self.server_bin),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
        deadline = time.monotonic() + timeout_s
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise LlamaServerError(
                        f"llama-server '{spec.name}' exited during startup "
                        f"(code {process.returncode}); see {log_path}"
                    )
                if self._health_ok(f"http://{spec.host}:{spec.port}/health"):
                    handle = ServerHandle(
                        spec=spec, pid=process.pid, started_at=time.time(), log_path=log_path
                    )
                    self._save_handle(handle)
                    return handle
                time.sleep(HEALTH_POLL_SECONDS)
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)
            raise
        process.kill()
        process.wait(timeout=30)
        raise LlamaServerError(
            f"llama-server '{spec.name}' did not become healthy within {timeout_s:.0f}s; "
            f"see {log_path}"
        )

    # ---------------------------------------------------------------- stop

    def stop(self, name: str) -> dict:
        """Stop a managed server by name. Returns a small status dict."""
        handle = self._load_handle(name)
        if handle is None:
            return {"name": name, "stopped": False, "reason": "no managed instance recorded"}
        if not self._pid_alive(handle.pid):
            self._state_path(name).unlink(missing_ok=True)
            return {
                "name": name,
                "stopped": True,
                "reason": f"stale record removed (pid {handle.pid} already dead)",
            }
        subprocess.run(["taskkill", "/PID", str(handle.pid), "/T", "/F"], capture_output=True, check=False)
        deadline = time.monotonic() + 30.0
        while self._pid_alive(handle.pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        still_alive = self._pid_alive(handle.pid)
        if still_alive:
            raise LlamaServerError(f"managed server '{name}' (pid {handle.pid}) survived taskkill /F")
        self._state_path(name).unlink(missing_ok=True)
        return {"name": name, "stopped": True, "reason": f"pid {handle.pid} terminated"}

    # -------------------------------------------------------------- status

    def status(self) -> list[dict]:
        """List every managed instance with its liveness."""
        rows = []
        for path in sorted(self.state_dir.glob("*.json")):
            handle = self._load_handle(path.stem)
            if handle is None:
                continue
            rows.append(
                {
                    "name": handle.spec.name,
                    "pid": handle.pid,
                    "port": handle.spec.port,
                    "model": handle.spec.model_path.name,
                    "alive": self._pid_alive(handle.pid),
                    "healthy": self._health_ok(f"http://{handle.spec.host}:{handle.spec.port}/health"),
                    "log": str(handle.log_path),
                }
            )
        return rows
