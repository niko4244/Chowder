from __future__ import annotations

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


@dataclass(frozen=True)
class HardwareSnapshot:
    platform: str
    cpu_count: int
    ram_gb: float
    storage_total_gb: float
    storage_free_gb: float
    accelerators: tuple[AcceleratorProfile, ...]

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
    for raw in text.splitlines():
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
        profiles.append(AcceleratorProfile("nvidia", name, memory_gb, bus_id))
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
    """Collect a dependency-free hardware snapshot.

    This is inventory, not performance calibration. Bandwidth and allocator peak
    measurements belong to the calibration layer because guessed bus bandwidth is
    not strong enough evidence for memory-placement decisions.
    """
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
