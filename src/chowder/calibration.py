from __future__ import annotations

import os
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_GIB = 1024 ** 3
_MIB = 1024 ** 2


@dataclass(frozen=True)
class StorageCalibration:
    read_gbps: float
    write_gbps: float
    sample_mib: int
    passes: int


@dataclass(frozen=True)
class HostMemoryCalibration:
    copy_gbps: float
    sample_mib: int
    passes: int


@dataclass(frozen=True)
class CudaTransferCalibration:
    device_index: int
    host_to_device_gbps: float
    device_to_host_gbps: float
    sample_mib: int
    passes: int
    total_vram_gb: float
    free_vram_gb: float


@dataclass(frozen=True)
class HardwareCalibration:
    storage: StorageCalibration | None = None
    host_memory: HostMemoryCalibration | None = None
    cuda: CudaTransferCalibration | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _median_gbps(byte_count: int, durations: list[float]) -> float:
    valid = [duration for duration in durations if duration > 0]
    if not valid:
        return 0.0
    seconds = statistics.median(valid)
    return (byte_count / _GIB) / seconds


def calibrate_storage(
    path: str | Path = ".",
    *,
    sample_mib: int = 64,
    passes: int = 3,
) -> StorageCalibration:
    """Measure sequential file throughput using a temporary file.

    This reports end-to-end filesystem throughput, not raw device marketing
    bandwidth. The temporary file is removed even when calibration fails.
    """
    if sample_mib <= 0 or passes <= 0:
        raise ValueError("sample_mib and passes must be positive")

    directory = Path(path).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    size = sample_mib * _MIB
    block = b"\0" * min(size, 8 * _MIB)
    write_times: list[float] = []
    read_times: list[float] = []

    fd, filename = tempfile.mkstemp(prefix=".chowder-cal-", dir=directory)
    os.close(fd)
    temp_path = Path(filename)
    try:
        for _ in range(passes):
            start = time.perf_counter()
            with temp_path.open("wb", buffering=0) as handle:
                remaining = size
                while remaining:
                    chunk = block if remaining >= len(block) else block[:remaining]
                    handle.write(chunk)
                    remaining -= len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            write_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            read_bytes = 0
            with temp_path.open("rb", buffering=0) as handle:
                while True:
                    chunk = handle.read(len(block))
                    if not chunk:
                        break
                    read_bytes += len(chunk)
            if read_bytes != size:
                raise OSError(f"calibration read {read_bytes} bytes, expected {size}")
            read_times.append(time.perf_counter() - start)
    finally:
        temp_path.unlink(missing_ok=True)

    return StorageCalibration(
        read_gbps=_median_gbps(size, read_times),
        write_gbps=_median_gbps(size, write_times),
        sample_mib=sample_mib,
        passes=passes,
    )


def calibrate_host_memory(*, sample_mib: int = 128, passes: int = 5) -> HostMemoryCalibration:
    """Measure effective host buffer-copy throughput.

    The result is a conservative application-visible copy measurement, not a
    claim about theoretical DRAM channel bandwidth.
    """
    if sample_mib <= 0 or passes <= 0:
        raise ValueError("sample_mib and passes must be positive")

    size = sample_mib * _MIB
    source = bytearray(size)
    destination = bytearray(size)
    src = memoryview(source)
    dst = memoryview(destination)
    durations: list[float] = []

    dst[:] = src
    for _ in range(passes):
        start = time.perf_counter()
        dst[:] = src
        durations.append(time.perf_counter() - start)

    return HostMemoryCalibration(
        copy_gbps=_median_gbps(size, durations),
        sample_mib=sample_mib,
        passes=passes,
    )


def calibrate_cuda_transfer(
    *,
    device_index: int = 0,
    sample_mib: int = 64,
    passes: int = 5,
) -> CudaTransferCalibration | None:
    """Measure pinned-host CUDA transfer throughput when PyTorch/CUDA is available."""
    if sample_mib <= 0 or passes <= 0:
        raise ValueError("sample_mib and passes must be positive")

    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
        return None

    size = sample_mib * _MIB
    device = torch.device(f"cuda:{device_index}")
    host_source = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    host_destination = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    device_buffer = torch.empty(size, dtype=torch.uint8, device=device)

    def timed_copy(destination, source) -> float:
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        destination.copy_(source, non_blocking=True)
        torch.cuda.synchronize(device)
        return time.perf_counter() - start

    timed_copy(device_buffer, host_source)
    timed_copy(host_destination, device_buffer)

    h2d = [timed_copy(device_buffer, host_source) for _ in range(passes)]
    d2h = [timed_copy(host_destination, device_buffer) for _ in range(passes)]
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)

    return CudaTransferCalibration(
        device_index=device_index,
        host_to_device_gbps=_median_gbps(size, h2d),
        device_to_host_gbps=_median_gbps(size, d2h),
        sample_mib=sample_mib,
        passes=passes,
        total_vram_gb=total_bytes / _GIB,
        free_vram_gb=free_bytes / _GIB,
    )


def calibrate_hardware(
    path: str | Path = ".",
    *,
    sample_mib: int = 64,
    passes: int = 3,
    include_cuda: bool = True,
) -> HardwareCalibration:
    notes: list[str] = []
    storage = calibrate_storage(path, sample_mib=sample_mib, passes=passes)
    host_memory = calibrate_host_memory(sample_mib=max(sample_mib, 64), passes=max(passes, 3))
    cuda = calibrate_cuda_transfer(sample_mib=sample_mib, passes=max(passes, 3)) if include_cuda else None
    if include_cuda and cuda is None:
        notes.append("CUDA transfer calibration unavailable; PyTorch/CUDA was not detected")
    return HardwareCalibration(storage=storage, host_memory=host_memory, cuda=cuda, notes=tuple(notes))
