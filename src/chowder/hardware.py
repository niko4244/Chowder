from __future__ import annotations

import math
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


_GIB = 1024 ** 3


@dataclass(frozen=True)
class AcceleratorProfile:
    vendor: str
    name: str
    memory_gb: float
    bus_id: str | None = None
    index: int | None = None
    uuid: str | None = None
    compute_capability: str | None = None

    def __post_init__(self) -> None:
        if not self.vendor.strip() or not self.name.strip():
            raise ValueError("accelerator vendor and name are required")
        memory = float(self.memory_gb)
        if not math.isfinite(memory) or memory < 0:
            raise ValueError("accelerator memory_gb must be finite and non-negative")
        if self.index is not None and self.index < 0:
            raise ValueError("accelerator index cannot be negative")

    @property
    def sm_marker(self) -> str | None:
        if self.compute_capability is None:
            return None
        raw = self.compute_capability.strip().lower().removeprefix("sm_")
        parts = raw.split(".", 1)
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            return f"sm_{parts[0]}{parts[1]}"
        if raw.isdigit():
            return f"sm_{raw}"
        return None


@dataclass(frozen=True)
class AcceleratorLink:
    source_index: int
    target_index: int
    kind: str = "unknown"
    measured_bandwidth_gbps: float | None = None

    def __post_init__(self) -> None:
        if self.source_index < 0 or self.target_index < 0:
            raise ValueError("accelerator link indexes cannot be negative")
        if self.source_index == self.target_index:
            raise ValueError("accelerator link endpoints must be different")
        if self.measured_bandwidth_gbps is not None:
            value = float(self.measured_bandwidth_gbps)
            if not math.isfinite(value) or value < 0:
                raise ValueError("accelerator link bandwidth must be finite and non-negative")


@dataclass(frozen=True)
class HardwareTopology:
    accelerators: tuple[AcceleratorProfile, ...]
    links: tuple[AcceleratorLink, ...] = ()

    def __post_init__(self) -> None:
        count = len(self.accelerators)
        for link in self.links:
            if link.source_index >= count or link.target_index >= count:
                raise ValueError("accelerator link references an unknown accelerator")

    @property
    def memory_pools_gb(self) -> tuple[float, ...]:
        return tuple(float(accelerator.memory_gb) for accelerator in self.accelerators)

    @property
    def total_accelerator_memory_gb(self) -> float:
        """Aggregate physical capacity; never implies one contiguous allocation."""
        return sum(self.memory_pools_gb)

    @property
    def max_contiguous_accelerator_memory_gb(self) -> float:
        return max(self.memory_pools_gb, default=0.0)

    def can_fit_single_device(self, required_gb: float) -> bool:
        required = float(required_gb)
        if not math.isfinite(required) or required < 0:
            raise ValueError("required_gb must be finite and non-negative")
        return any(pool >= required for pool in self.memory_pools_gb)


@dataclass(frozen=True)
class HardwareSnapshot:
    platform: str
    cpu_count: int
    ram_gb: float
    storage_total_gb: float
    storage_free_gb: float
    accelerators: tuple[AcceleratorProfile, ...]

    def __post_init__(self) -> None:
        if self.cpu_count <= 0:
            raise ValueError("cpu_count must be positive")
        for label, value in (
            ("ram_gb", self.ram_gb),
            ("storage_total_gb", self.storage_total_gb),
            ("storage_free_gb", self.storage_free_gb),
        ):
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        if self.storage_free_gb > self.storage_total_gb + 1e-12:
            raise ValueError("storage_free_gb cannot exceed storage_total_gb")

    @property
    def topology(self) -> HardwareTopology:
        return HardwareTopology(self.accelerators)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _system_ram_bytes() -> int:
    if os.name == "nt":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(status.ullTotalPhys)

    page_size = os.sysconf("SC_PAGE_SIZE")
    pages = os.sysconf("SC_PHYS_PAGES")
    return int(page_size * pages)


def _parse_nvidia_smi(text: str) -> tuple[AcceleratorProfile, ...]:
    profiles: list[AcceleratorProfile] = []
    for ordinal, raw in enumerate(text.splitlines()):
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) < 2:
            continue
        name, memory_mib = parts[:2]
        bus_id = parts[2] if len(parts) == 3 and parts[2] else None
        try:
            memory_gb = float(memory_mib) / 1024.0
        except ValueError:
            continue
        profiles.append(
            AcceleratorProfile(
                "nvidia",
                name,
                memory_gb,
                bus_id,
                index=ordinal,
            )
        )
    return tuple(profiles)


def _detect_nvidia() -> tuple[AcceleratorProfile, ...]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,pci.bus_id",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ()
    return _parse_nvidia_smi(result.stdout)


def detect_hardware(path: str | Path = ".") -> HardwareSnapshot:
    """Collect dependency-free inventory without inventing performance data."""
    storage = shutil.disk_usage(Path(path).resolve())
    try:
        ram_bytes = _system_ram_bytes()
    except (OSError, ValueError, AttributeError):
        ram_bytes = 0

    return HardwareSnapshot(
        platform=f"{platform.system()} {platform.release()}".strip(),
        cpu_count=os.cpu_count() or 1,
        ram_gb=ram_bytes / _GIB,
        storage_total_gb=storage.total / _GIB,
        storage_free_gb=storage.free / _GIB,
        accelerators=_detect_nvidia(),
    )
