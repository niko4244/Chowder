from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hardware import HardwareSnapshot, detect_hardware


UNSLOTH_ENV_SCHEMA_VERSION = 1
DEFAULT_UNSLOTH_VERSION = "2026.9.2"
DEFAULT_UNSLOTH_PYTHON = "3.13"
DEFAULT_PROBE_TIMEOUT_SECONDS = 180


class UnslothEnvironmentError(RuntimeError):
    """Raised when the isolated Unsloth runtime cannot be prepared safely."""


@dataclass(frozen=True)
class CapabilityCheck:
    name: str
    ok: bool
    detail: str
    required: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CapabilityCheck":
        return cls(
            name=str(raw.get("name", "unknown")),
            ok=bool(raw.get("ok", False)),
            detail=str(raw.get("detail", "")),
            required=bool(raw.get("required", True)),
        )


@dataclass(frozen=True)
class UnslothDoctorReport:
    env_dir: Path
    python_executable: Path | None
    python_version: str | None
    platform: str
    versions: Mapping[str, str]
    cuda_version: str | None
    accelerators: tuple[Mapping[str, Any], ...]
    checks: tuple[CapabilityCheck, ...]
    process_returncode: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks if check.required)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "env_dir": str(self.env_dir),
            "python_executable": (
                str(self.python_executable) if self.python_executable is not None else None
            ),
            "python_version": self.python_version,
            "platform": self.platform,
            "versions": dict(self.versions),
            "cuda_version": self.cuda_version,
            "accelerators": [dict(row) for row in self.accelerators],
            "checks": [asdict(check) for check in self.checks],
            "process_returncode": self.process_returncode,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }


@dataclass(frozen=True)
class UnslothSetupResult:
    env_dir: Path
    manifest_path: Path
    uv_version: str
    requested_python: str
    requested_unsloth_version: str
    frozen_packages: tuple[str, ...]
    doctor: UnslothDoctorReport

    @property
    def ok(self) -> bool:
        return self.doctor.ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "env_dir": str(self.env_dir),
            "manifest_path": str(self.manifest_path),
            "uv_version": self.uv_version,
            "requested_python": self.requested_python,
            "requested_unsloth_version": self.requested_unsloth_version,
            "frozen_packages": list(self.frozen_packages),
            "doctor": self.doctor.to_dict(),
        }


def unsloth_env_dir(root: str | Path = ".") -> Path:
    return Path(root).expanduser().resolve() / ".chowder" / "envs" / "unsloth"


def unsloth_python(env_dir: str | Path, *, system_name: str | None = None) -> Path:
    env = Path(env_dir)
    system = (system_name or platform.system()).lower()
    if system == "windows":
        return env / "Scripts" / "python.exe"
    return env / "bin" / "python"


def _tail(value: str, *, limit: int = 4000) -> str:
    return value[-limit:] if len(value) > limit else value


def _run(
    command: Sequence[str],
    *,
    timeout: int = 300,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(command),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode != 0:
        rendered = " ".join(str(part) for part in command)
        raise UnslothEnvironmentError(
            f"command failed ({completed.returncode}): {rendered}\n"
            f"stdout:\n{_tail(completed.stdout)}\n"
            f"stderr:\n{_tail(completed.stderr)}"
        )
    return completed


def _requested_major_minor(value: str) -> tuple[int, int] | None:
    parts = value.strip().split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _environment_major_minor(python_executable: Path) -> tuple[int, int]:
    completed = _run(
        [
            str(python_executable),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        timeout=30,
    )
    major, minor = completed.stdout.strip().split(".", 1)
    return int(major), int(minor)


def _require_nvidia_gpu(snapshot: HardwareSnapshot) -> None:
    if snapshot.accelerators:
        return
    raise UnslothEnvironmentError(
        "No NVIDIA GPU was detected through nvidia-smi. Refusing to report an "
        "Unsloth CUDA setup as successful because uv --torch-backend=auto may "
        "legitimately select a CPU PyTorch build when no compatible GPU is visible."
    )


def _require_uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise UnslothEnvironmentError(
            "uv is required to create the isolated Unsloth runtime. Install the "
            "current Astral uv release, then rerun `chowder setup unsloth`."
        )
    return uv


_PROBE_SCRIPT = r'''
from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import sys
import traceback
from pathlib import Path

output = Path(sys.argv[1])
checks = []
versions = {}
accelerators = []
cuda_version = None


def add(name, ok, detail, required=True):
    checks.append({"name": name, "ok": bool(ok), "detail": str(detail), "required": bool(required)})


def dist_version(dist):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


for dist in (
    "unsloth",
    "unsloth-zoo",
    "torch",
    "transformers",
    "peft",
    "trl",
    "bitsandbytes",
    "xformers",
    "triton",
    "triton-windows",
):
    versions[dist] = dist_version(dist)

py_ok = (3, 9) <= sys.version_info[:2] < (3, 15)
add("Python", py_ok, sys.version.split()[0])

modules = {}
try:
    # Unsloth intentionally patches parts of the ML stack at import time and
    # upstream asks users to import it before Transformers. This happens only
    # inside the isolated child interpreter, never in Chowder's controller.
    modules["unsloth"] = importlib.import_module("unsloth")
    add("Unsloth", True, versions["unsloth"])
except Exception as exc:
    add("Unsloth", False, f"{type(exc).__name__}: {exc}")

for label, module_name, dist_name in (
    ("unsloth_zoo", "unsloth_zoo", "unsloth-zoo"),
    ("TRL", "trl", "trl"),
    ("PEFT", "peft", "peft"),
    ("bitsandbytes", "bitsandbytes", "bitsandbytes"),
):
    try:
        modules[module_name] = importlib.import_module(module_name)
        add(label, True, versions[dist_name])
    except Exception as exc:
        add(label, False, f"{type(exc).__name__}: {exc}")

try:
    torch = importlib.import_module("torch")
    modules["torch"] = torch
    add("Torch", True, versions["torch"])
except Exception as exc:
    torch = None
    add("Torch", False, f"{type(exc).__name__}: {exc}")

if torch is not None:
    try:
        cuda_version = str(torch.version.cuda) if torch.version.cuda is not None else None
        cuda_ok = bool(torch.cuda.is_available())
        add("CUDA", cuda_ok, f"torch CUDA={cuda_version!r}; available={cuda_ok}")
        if cuda_ok:
            for index in range(int(torch.cuda.device_count())):
                props = torch.cuda.get_device_properties(index)
                accelerators.append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "capability": list(torch.cuda.get_device_capability(index)),
                        "total_memory_gb": float(props.total_memory / (1024 ** 3)),
                    }
                )
        add(
            "NVIDIA GPU",
            bool(accelerators),
            ", ".join(row["name"] for row in accelerators) if accelerators else "none visible",
        )
    except Exception as exc:
        add("CUDA", False, f"{type(exc).__name__}: {exc}")
        add("NVIDIA GPU", False, "CUDA device enumeration failed")
else:
    add("CUDA", False, "torch import failed")
    add("NVIDIA GPU", False, "torch import failed")

try:
    importlib.import_module("triton")
    triton_dist = "triton-windows" if versions["triton-windows"] != "not-installed" else "triton"
    add("Triton", True, f"{triton_dist}={versions[triton_dist]}")
except Exception as exc:
    add("Triton", False, f"{type(exc).__name__}: {exc}")

if versions["xformers"] != "not-installed":
    try:
        importlib.import_module("xformers")
        add("xFormers", True, versions["xformers"], required=False)
    except Exception as exc:
        add("xFormers", False, f"installed but import failed: {type(exc).__name__}: {exc}", required=False)
else:
    add("xFormers", True, "not installed in this resolved stack", required=False)

if torch is not None and bool(getattr(torch.cuda, "is_available", lambda: False)()):
    try:
        bnb = modules.get("bitsandbytes") or importlib.import_module("bitsandbytes")
        device = torch.device("cuda:0")
        layer = bnb.nn.Linear4bit(
            16,
            16,
            bias=False,
            compute_dtype=torch.float16,
            compress_statistics=True,
            quant_type="nf4",
        ).to(device)
        sample = torch.randn(1, 2, 16, device=device, dtype=torch.float16)
        with torch.no_grad():
            result = layer(sample)
        torch.cuda.synchronize(device)
        finite = bool(torch.isfinite(result).all().item())
        add("4-bit CUDA support", finite, "bitsandbytes NF4 Linear4bit forward executed on cuda:0")
    except Exception as exc:
        add("4-bit CUDA support", False, f"{type(exc).__name__}: {exc}")
else:
    add("4-bit CUDA support", False, "CUDA is unavailable")

payload = {
    "schema_version": 1,
    "python_version": sys.version.split()[0],
    "platform": platform.platform(),
    "versions": versions,
    "cuda_version": cuda_version,
    "accelerators": accelerators,
    "checks": checks,
}
try:
    output.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
except Exception:
    traceback.print_exc()
    raise

raise SystemExit(0 if all(row["ok"] for row in checks if row["required"]) else 1)
'''


def _failed_report(
    env_dir: Path,
    python_executable: Path | None,
    *,
    name: str,
    detail: str,
    process_returncode: int | None = None,
    stdout: str = "",
    stderr: str = "",
) -> UnslothDoctorReport:
    return UnslothDoctorReport(
        env_dir=env_dir,
        python_executable=python_executable,
        python_version=None,
        platform=platform.platform(),
        versions={},
        cuda_version=None,
        accelerators=(),
        checks=(CapabilityCheck(name=name, ok=False, detail=detail),),
        process_returncode=process_returncode,
        stdout_tail=_tail(stdout),
        stderr_tail=_tail(stderr),
    )


def doctor_unsloth_environment(
    root: str | Path = ".",
    *,
    probe_timeout_seconds: int = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> UnslothDoctorReport:
    env_dir = unsloth_env_dir(root)
    python_executable = unsloth_python(env_dir)
    if not python_executable.is_file():
        return _failed_report(
            env_dir,
            None,
            name="Environment",
            detail=f"isolated Python not found at {python_executable}",
        )

    result_path = env_dir / f".chowder-doctor-{uuid.uuid4().hex}.json"
    try:
        completed = _run(
            [str(python_executable), "-c", _PROBE_SCRIPT, str(result_path)],
            timeout=probe_timeout_seconds,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return _failed_report(
            env_dir,
            python_executable,
            name="Probe",
            detail=f"{type(exc).__name__}: {exc}",
        )

    try:
        if not result_path.is_file():
            return _failed_report(
                env_dir,
                python_executable,
                name="Probe",
                detail="isolated capability probe did not produce its JSON result",
                process_returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        raw = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
            return _failed_report(
                env_dir,
                python_executable,
                name="Probe",
                detail="isolated capability probe returned an unsupported result schema",
                process_returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        checks_raw = raw.get("checks")
        if not isinstance(checks_raw, list) or not checks_raw:
            return _failed_report(
                env_dir,
                python_executable,
                name="Probe",
                detail="isolated capability probe returned no checks",
                process_returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        checks = tuple(
            CapabilityCheck.from_mapping(row)
            for row in checks_raw
            if isinstance(row, Mapping)
        )
        versions_raw = raw.get("versions", {})
        versions = (
            {str(key): str(value) for key, value in versions_raw.items()}
            if isinstance(versions_raw, Mapping)
            else {}
        )
        accelerators_raw = raw.get("accelerators", [])
        accelerators = tuple(
            dict(row) for row in accelerators_raw if isinstance(row, Mapping)
        )
        return UnslothDoctorReport(
            env_dir=env_dir,
            python_executable=python_executable,
            python_version=(
                str(raw["python_version"]) if raw.get("python_version") is not None else None
            ),
            platform=str(raw.get("platform", platform.platform())),
            versions=versions,
            cuda_version=(
                str(raw["cuda_version"]) if raw.get("cuda_version") is not None else None
            ),
            accelerators=accelerators,
            checks=checks,
            process_returncode=completed.returncode,
            stdout_tail=_tail(completed.stdout),
            stderr_tail=_tail(completed.stderr),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return _failed_report(
            env_dir,
            python_executable,
            name="Probe",
            detail=f"could not parse isolated capability result: {type(exc).__name__}: {exc}",
            process_returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    finally:
        try:
            result_path.unlink(missing_ok=True)
        except OSError:
            pass


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def setup_unsloth_environment(
    root: str | Path = ".",
    *,
    python_request: str = DEFAULT_UNSLOTH_PYTHON,
    unsloth_version: str = DEFAULT_UNSLOTH_VERSION,
) -> UnslothSetupResult:
    root_path = Path(root).expanduser().resolve()
    snapshot = detect_hardware(root_path)
    _require_nvidia_gpu(snapshot)
    uv = _require_uv()
    uv_version_result = _run([uv, "--version"], timeout=30)
    uv_version = uv_version_result.stdout.strip() or uv_version_result.stderr.strip()

    env_dir = unsloth_env_dir(root_path)
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    python_executable = unsloth_python(env_dir)
    if python_executable.is_file():
        requested = _requested_major_minor(python_request)
        if requested is not None:
            actual = _environment_major_minor(python_executable)
            if actual != requested:
                raise UnslothEnvironmentError(
                    f"existing Unsloth environment uses Python {actual[0]}.{actual[1]}, "
                    f"but Python {python_request} was requested; remove {env_dir} or "
                    "rerun with the environment's existing Python version"
                )
    else:
        _run([uv, "venv", str(env_dir), "--python", python_request], timeout=600)
        if not python_executable.is_file():
            raise UnslothEnvironmentError(
                f"uv reported success but isolated Python was not created at {python_executable}"
            )

    _run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python_executable),
            f"unsloth=={unsloth_version}",
            "--torch-backend=auto",
        ],
        timeout=1800,
    )
    pip_check = _run(
        [uv, "pip", "check", "--python", str(python_executable)],
        timeout=120,
        check=False,
    )
    freeze = _run(
        [uv, "pip", "freeze", "--python", str(python_executable)],
        timeout=120,
    )
    frozen_packages = tuple(
        line.strip() for line in freeze.stdout.splitlines() if line.strip()
    )

    doctor = doctor_unsloth_environment(root_path)
    if pip_check.returncode != 0:
        doctor = UnslothDoctorReport(
            env_dir=doctor.env_dir,
            python_executable=doctor.python_executable,
            python_version=doctor.python_version,
            platform=doctor.platform,
            versions=doctor.versions,
            cuda_version=doctor.cuda_version,
            accelerators=doctor.accelerators,
            checks=(
                CapabilityCheck(
                    name="Dependency graph",
                    ok=False,
                    detail=(
                        _tail(pip_check.stdout + "\n" + pip_check.stderr).strip()
                        or f"uv pip check exited {pip_check.returncode}"
                    ),
                ),
                *doctor.checks,
            ),
            process_returncode=doctor.process_returncode,
            stdout_tail=doctor.stdout_tail,
            stderr_tail=doctor.stderr_tail,
        )

    manifest_path = env_dir / "chowder-unsloth-manifest.json"
    manifest = {
        "schema_version": UNSLOTH_ENV_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": "unsloth",
        "env_dir": str(env_dir),
        "controller": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "requested": {
            "python": python_request,
            "unsloth_version": unsloth_version,
            "torch_backend": "auto",
        },
        "uv_version": uv_version,
        "hardware_preflight": snapshot.to_dict(),
        "frozen_packages": list(frozen_packages),
        "doctor": doctor.to_dict(),
    }
    _write_manifest(manifest_path, manifest)
    return UnslothSetupResult(
        env_dir=env_dir,
        manifest_path=manifest_path,
        uv_version=uv_version,
        requested_python=python_request,
        requested_unsloth_version=unsloth_version,
        frozen_packages=frozen_packages,
        doctor=doctor,
    )


def format_unsloth_doctor(report: UnslothDoctorReport) -> str:
    rows = []
    width = max((len(check.name) for check in report.checks), default=10)
    for check in report.checks:
        marker = "OK" if check.ok else "FAIL"
        optional = " (optional)" if not check.required else ""
        rows.append(f"{check.name:<{width}}  [{marker}] {check.detail}{optional}")
    return "\n".join(rows)
