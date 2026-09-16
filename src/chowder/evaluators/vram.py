"""How much VRAM did *this process* use? Reported so a run can answer it.

The training workers have always reported `peak_vram_gb`. The evaluation workers
did not, and that gap cost a verdict.

`PRUNED_9B_REAL_TRAINING_PREREG.md` pre-registered "peak VRAM under 15.93 GiB" and
listed oversubscription as a FAIL "judged by headroom and step-time blowup". For the
evaluation leg neither quantity existed in the run's artifacts, so the only available
proxy was `nvidia-smi` -- which reports the whole **machine**: every browser, every
service, every other process. It read 559 MiB free during the candidate eval and the
run was recorded as FAIL on oversubscription.

Controlled re-measurement afterwards put the evaluation process's own footprint at
~6.6 GiB, flat across token budgets from 1 to 768 and across 12 sequential problems,
with ~8.8 GiB of card free throughout. Five candidate mechanisms were eliminated
(adapter weights, token count, accumulation, cache class, instrument disagreement),
and the remaining ~9 GiB was held by something outside the experiment that could not
be identified. A run has to be able to answer "how much did *I* use" from its own
evidence; otherwise a busy desktop can fail an experiment.

Both numbers are reported because today's investigation turned on the difference:

* `peak_vram_gb` (allocated) -- what the model actually needed.
* `peak_vram_reserved_gb` -- what torch took from the driver, which is the figure
  that decides whether the run fits *alongside anything else*.

Never raises. Telemetry must not be able to kill a run -- the same rule
`progress_write.py` exists to enforce, learned the same way.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence

#: Explicit "not measured" rather than 0.0, so a CPU run or a torch-less
#: environment cannot be read as "used no memory". Unknown is not zero -- the same
#: distinction `target_coverage` makes with `status: "not_reported"`.
_UNMEASURED: dict[str, Any] = {"peak_vram_gb": None, "peak_vram_reserved_gb": None}


def peak_vram(device_name: str) -> dict[str, Any]:
    """Peak VRAM this process allocated and reserved on `device_name`, in GiB."""
    if not device_name.startswith("cuda"):
        return dict(_UNMEASURED)
    try:
        import torch

        index = int(device_name.split(":", 1)[1]) if ":" in device_name else 0
        return {
            "peak_vram_gb": float(torch.cuda.max_memory_allocated(index) / (1024**3)),
            "peak_vram_reserved_gb": float(torch.cuda.max_memory_reserved(index) / (1024**3)),
        }
    except Exception:  # pragma: no cover - defensive: telemetry is never fatal
        return dict(_UNMEASURED)


#: Field names the sampler summaries use, and which each measures.
_SAMPLE_FIELDS = (
    "free_device_bytes",
    "used_device_bytes",
    "host_rss_bytes",
    "host_commit_bytes",
)

#: Where a device reading's underlying sample key lives, per reported field.
_SAMPLE_KEYS: dict[str, str] = {
    "free_device_bytes": "free_bytes",
    "used_device_bytes": "used_bytes",
    "host_rss_bytes": "rss_bytes",
    "host_commit_bytes": "commit_bytes",
}


def host_memory() -> dict[str, Any]:
    """This process's resident/committed bytes and the machine's total.

    Separate from the device figures on purpose: a host-pressure refusal and a
    VRAM refusal are different failures with different remedies, and summing
    them (or reporting host RSS as "memory") would hide which one happened.
    Never raises; an unavailable field is ``None``, never ``0``.
    """
    report: dict[str, Any] = {
        "rss_bytes": None,
        "commit_bytes": None,
        "total_system_bytes": None,
        "source": None,
    }
    try:
        import psutil  # type: ignore

        process = psutil.Process(os.getpid())
        info = process.memory_info()
        report["rss_bytes"] = int(info.rss)
        # Windows calls it private/committed; psutil exposes vms portably, and
        # pagefile (commit) only on Windows. Use what this platform really has.
        commit = getattr(info, "private", None) or getattr(info, "vms", None)
        report["commit_bytes"] = None if commit is None else int(commit)
        report["total_system_bytes"] = int(psutil.virtual_memory().total)
        report["source"] = "psutil"
        return report
    except Exception:
        pass

    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes

            class _ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
                ctypes.windll.kernel32.GetCurrentProcess(),  # type: ignore[attr-defined]
                ctypes.byref(counters),
                counters.cb,
            )
            if ok:
                report["rss_bytes"] = int(counters.WorkingSetSize)
                report["commit_bytes"] = int(counters.PagefileUsage)
                report["source"] = "win32-psapi"

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                report["total_system_bytes"] = int(status.ullTotalPhys)
                report["source"] = report["source"] or "win32-globalmemorystatus"
            return report
        except Exception:  # pragma: no cover - defensive: telemetry is never fatal
            return report

    try:
        with open("/proc/self/statm", "r", encoding="ascii") as handle:
            fields = handle.read().split()
        page = os.sysconf("SC_PAGE_SIZE")
        report["rss_bytes"] = int(fields[1]) * int(page)
        report["source"] = "/proc/self/statm"
        with open("/proc/meminfo", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    report["total_system_bytes"] = int(line.split()[1]) * 1024
                    break
    except Exception:  # pragma: no cover - defensive: telemetry is never fatal
        pass
    return report


def device_memory(device_name: str) -> dict[str, Any]:
    """Free/total/used device memory *right now*, in bytes.

    ``used_bytes`` is derived from the same single reading as free/total so the
    three cannot disagree. A CPU device or a failed query reports ``None`` --
    an unavailable reading is not an empty device.
    """
    unknown = {"free_bytes": None, "total_bytes": None, "used_bytes": None}
    if not str(device_name).startswith("cuda"):
        return unknown
    try:
        import torch

        if not torch.cuda.is_available():
            return unknown
        index = int(device_name.split(":", 1)[1]) if ":" in device_name else 0
        free, total = torch.cuda.mem_get_info(index)
        return {
            "free_bytes": int(free),
            "total_bytes": int(total),
            "used_bytes": int(total) - int(free),
        }
    except Exception:  # pragma: no cover - defensive: telemetry is never fatal
        return unknown


def summarize_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    cadence_seconds: float,
    span_seconds: float,
) -> dict[str, Any]:
    """Extremes across sampled headroom, with the sampling caveat attached.

    The minimum free figure is the minimum *observed*. It is not the minimum
    that occurred: a sample is a point reading, and this summary says so in
    ``note`` rather than letting a sampled number pass as a continuous record.
    """
    observed: dict[str, list[int]] = {field: [] for field in _SAMPLE_FIELDS}
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        for field, key in _SAMPLE_KEYS.items():
            value = sample.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            observed[field].append(value)

    def minimum(field: str) -> int | None:
        return min(observed[field]) if observed[field] else None

    def maximum(field: str) -> int | None:
        return max(observed[field]) if observed[field] else None

    unavailable = [field for field in _SAMPLE_FIELDS if not observed[field]]
    return {
        "samples": len(samples),
        "cadence_seconds": float(cadence_seconds),
        "span_seconds": float(span_seconds),
        "min_free_device_bytes": minimum("free_device_bytes"),
        "max_used_device_bytes": maximum("used_device_bytes"),
        "min_host_rss_bytes": minimum("host_rss_bytes"),
        "max_host_commit_bytes": maximum("host_commit_bytes"),
        "unavailable_fields": unavailable,
        "note": (
            "sampled at the recorded cadence for the recorded span; these are the "
            "extremes OBSERVED, not a continuous record -- an instantaneous minimum "
            "between samples is not excluded"
        ),
    }


class MemorySampler:
    """Sample this process's own headroom while a phase runs.

    Deliberately self-measuring: the machine-wide figure (``nvidia-smi``) is what
    produced a spurious oversubscription FAIL, because a busy desktop is not the
    experiment. Probes are injectable so the sampling logic is testable without a
    GPU, and a probe that raises degrades that field to unknown instead of taking
    down an otherwise-successful run.
    """

    def __init__(
        self,
        device_name: str,
        *,
        interval_seconds: float = 0.5,
        device_probe: Callable[[str], Mapping[str, Any] | None] | None = None,
        host_probe: Callable[[], Mapping[str, Any] | None] | None = None,
    ) -> None:
        interval = float(interval_seconds)
        if interval <= 0:
            raise ValueError("interval_seconds must be positive")
        self.device_name = str(device_name)
        self.interval_seconds = interval
        self._device_probe = device_probe or device_memory
        self._host_probe = host_probe or host_memory
        self._samples: list[dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self.report: dict[str, Any] | None = None

    # -- sampling ---------------------------------------------------------

    def sample_now(self) -> dict[str, Any]:
        """Take one reading. Never raises."""
        sample: dict[str, Any] = {}
        for probe, label in ((self._device_probe, "device"), (self._host_probe, "host")):
            try:
                reading = probe(self.device_name) if label == "device" else probe()
            except Exception:
                continue
            if isinstance(reading, Mapping):
                sample.update({key: value for key, value in reading.items() if key in _SAMPLE_KEYS.values()})
        self._samples.append(sample)
        return sample

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self.sample_now()
            self._stop_event.wait(self.interval_seconds)

    def start(self) -> "MemorySampler":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._started_at = time.perf_counter()
        self._stop_event.clear()
        self.sample_now()
        self._thread = threading.Thread(
            target=self._loop, name="chowder-memory-sampler", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(1.0, self.interval_seconds * 4))
        self._stopped_at = time.perf_counter()
        span = (
            0.0
            if self._started_at is None
            else max(0.0, self._stopped_at - self._started_at)
        )
        report = summarize_samples(
            self._samples, cadence_seconds=self.interval_seconds, span_seconds=span
        )
        report["device"] = self.device_name
        self.report = report
        return report

    def __enter__(self) -> "MemorySampler":
        return self.start()

    def __exit__(self, *exc_info: Any) -> bool:
        self.stop()
        return False

