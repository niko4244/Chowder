"""Source-bound private Windows Job birth/exit event journal.

An *owned* Win32 Job completion port captures NEW_PROCESS/EXIT_PROCESS
notifications even if a native helper lives for less than the supervisor's
three-second polling interval.  This does not allow a PID-only event to
authorize executable provenance.  A matching, independently inspected live
PID/create-time/parent/cmdline/executable/member record is required; otherwise
an original candidate run is REFUSED, never promoted to a Gen-0 PASS.

No foreign process control, GPU access, process creation or model inference.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time
from typing import Any

import psutil

SCHEMA = "chowder.reap-v2-phase51-owned-private-job-lifetime-events.v1"
NEW_PROCESS = 6
EXIT_PROCESS = 7
ABNORMAL_EXIT_PROCESS = 8
ACTIVE_PROCESS_ZERO = 4
TIMEOUT = 258
MAX_EVENTS = 8192


class EventWitnessRefused(ValueError):
    pass


class _ASSOCIATE_COMPLETION_PORT(ctypes.Structure):
    _fields_ = [("CompletionKey", ctypes.c_void_p),
                ("CompletionPort", wintypes.HANDLE)]


class _FILETIME(ctypes.Structure):
    _fields_ = [("lo", wintypes.DWORD), ("hi", wintypes.DWORD)]


@dataclass(frozen=True)
class BirthEvidence:
    pid: int
    birth: float
    parent: int
    argv: tuple[str, ...]
    image_path: str
    inside_private_job: bool

    def record(self) -> dict:
        return {
            "pid": self.pid, "birth": self.birth, "parent": self.parent,
            "argv": list(self.argv), "image_path": self.image_path,
            "job_member": self.inside_private_job,
        }


class PrivateJobEvents:
    """Only the original private Job HANDLE can produce this journal.

    The watcher begins before the root ResumeThread, reads a separately
    associated kernel completion port, and attempts source-grade process
    identity capture immediately on every NEW_PROCESS message. It cannot
    validate frozen commands itself (see reconcile_births).
    """

    def __init__(self, *, kernel: Any, job: int, member_check, event_wait_ms: int = 50):
        if (not job or not callable(member_check)
                or type(event_wait_ms) is not int or not 1 <= event_wait_ms <= 200):
            raise EventWitnessRefused("PRIVATE_JOB_EVENT_OWNER_MISSING")
        self._kernel = kernel
        self._job = int(job)
        self._member_check = member_check
        self._wait_ms = event_wait_ms
        self._port = None
        self._events: list[dict] = []
        self._births: dict[int, BirthEvidence] = {}
        self._process_handles: dict[int, int] = {}
        self._faults: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._quiet_polls = 0

    def _native(self, name: str, args, types, restype=wintypes.BOOL):
        f = getattr(self._kernel, name)
        f.argtypes = types
        f.restype = restype
        return f(*args)

    def attach_before_resume(self):
        if self._port is not None:
            raise EventWitnessRefused("PRIVATE_JOB_EVENT_PORT_ALREADY_ATTACHED")
        port = self._native(
            "CreateIoCompletionPort",
            (ctypes.c_void_p(-1), None, ctypes.c_size_t(0), wintypes.DWORD(1)),
            (wintypes.HANDLE, wintypes.HANDLE, ctypes.c_size_t, wintypes.DWORD),
            wintypes.HANDLE,
        )
        if not port:
            raise EventWitnessRefused("PRIVATE_JOB_EVENT_PORT_CREATE_FAILED")
        self._port = port
        associated = _ASSOCIATE_COMPLETION_PORT(
            CompletionKey=ctypes.c_void_p(self._job), CompletionPort=port,
        )
        if not self._native(
            "SetInformationJobObject",
            (self._job, 7, ctypes.byref(associated), ctypes.sizeof(associated)),
            (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD),
        ):
            self.close()
            raise EventWitnessRefused("PRIVATE_JOB_EVENT_PORT_ASSOCIATION_FAILED")
        return self

    def start(self):
        if not self._port or self._thread is not None:
            raise EventWitnessRefused("UNBOUND_OR_DUPLICATE_JOB_EVENT_CONSUMER")
        self._thread = threading.Thread(
            target=self._poll, name="chowder-owned-job-events", daemon=True,
        )
        self._thread.start()
        return self

    def _capture_birth(self, pid: int) -> BirthEvidence | None:
        process_handle = None
        try:
            # Keep the original kernel process HANDLE until terminal cleanup.
            # A recycled PID with an identical command string cannot replace
            # a previously observed process's real Win32 creation time.
            process_handle = self._native(
                "OpenProcess", (0x101000, False, pid),
                (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD),
                wintypes.HANDLE,
            )
            if not process_handle:
                return None
            belongs = wintypes.BOOL()
            if not self._native(
                "IsProcessInJob",
                (process_handle, self._job, ctypes.byref(belongs)),
                (wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)),
            ) or not belongs.value:
                return None
            create, exit_time, kernel_time, user_time = (
                _FILETIME() for _ in range(4)
            )
            if not self._native(
                "GetProcessTimes",
                (process_handle, ctypes.byref(create), ctypes.byref(exit_time),
                 ctypes.byref(kernel_time), ctypes.byref(user_time)),
                (wintypes.HANDLE, ctypes.POINTER(_FILETIME),
                 ctypes.POINTER(_FILETIME), ctypes.POINTER(_FILETIME),
                 ctypes.POINTER(_FILETIME)),
            ):
                return None
            handle_birth = (
                ((int(create.hi) << 32) + int(create.lo))
                / 10**7 - 11644473600
            )
            image_buffer = ctypes.create_unicode_buffer(32768)
            image_length = wintypes.DWORD(len(image_buffer))
            if not self._native(
                "QueryFullProcessImageNameW",
                (process_handle, 0, image_buffer, ctypes.byref(image_length)),
                (wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                 ctypes.POINTER(wintypes.DWORD)),
            ):
                return None
            native_image = image_buffer.value
            proc = psutil.Process(pid)
            with proc.oneshot():
                birth = proc.create_time()
                parent = proc.ppid()
                argv = proc.cmdline()
                image = proc.exe()
            if (type(birth) not in (int, float) or birth <= 0
                    or not math.isfinite(birth)
                    or abs(handle_birth - birth) > 0.05
                    or type(parent) is not int or parent < 0
                    or not isinstance(argv, list) or not argv
                    or not image or not self._member_check(pid)
                    or Path(image).resolve(strict=False)
                    != Path(native_image).resolve(strict=False)):
                return None
            with self._lock:
                if pid in self._process_handles:
                    return None
                self._process_handles[pid] = int(process_handle)
            process_handle = None
            return BirthEvidence(pid, float(birth), parent, tuple(argv),
                                 native_image, True)
        except (psutil.Error, OSError, ValueError, TypeError):
            return None
        finally:
            if process_handle:
                self._kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
                self._kernel.CloseHandle.restype = wintypes.BOOL
                self._kernel.CloseHandle(process_handle)

    def _poll(self):
        try:
            self._poll_owned_packets()
        except BaseException:
            # A daemon's unexpected exit must never make an incomplete event
            # journal look like a successfully monitored native process tree.
            with self._lock:
                self._faults.append("PRIVATE_JOB_EVENT_READER_UNEXPECTED_EXCEPTION")
        finally:
            if not self._stop.is_set():
                with self._lock:
                    self._faults.append("PRIVATE_JOB_EVENT_READER_STOPPED_UNEXPECTEDLY")

    def _poll_owned_packets(self):
        while not self._stop.is_set():
            message = wintypes.DWORD()
            key = ctypes.c_size_t()
            overlapped = ctypes.c_void_p()
            ok = self._native(
                "GetQueuedCompletionStatus",
                (self._port, ctypes.byref(message), ctypes.byref(key),
                 ctypes.byref(overlapped), self._wait_ms),
                (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD),
                 ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_void_p),
                 wintypes.DWORD),
            )
            if not ok:
                # Timeout is benign; any other native error invalidates the
                # entire lifetime witness without touching any Job member.
                if ctypes.get_last_error() == TIMEOUT:
                    with self._lock:
                        self._quiet_polls += 1
                    continue
                with self._lock:
                    self._faults.append("PRIVATE_JOB_EVENT_QUEUE_NATIVE_ERROR")
                return
            pid = int(overlapped.value or 0)
            kind = int(message.value)
            if key.value != self._job:
                with self._lock:
                    self._faults.append("FOREIGN_COMPLETION_KEY")
                return
            captured = self._capture_birth(pid) if kind == NEW_PROCESS and pid else None
            with self._lock:
                self._quiet_polls = 0
                if len(self._events) >= MAX_EVENTS:
                    self._faults.append("PRIVATE_JOB_EVENT_QUEUE_TOO_LARGE")
                    return
                self._events.append({
                    "message": kind, "pid": pid,
                    "birth_identity_captured": captured is not None,
                })
                if kind == NEW_PROCESS:
                    if pid <= 0 or pid in self._births:
                        self._faults.append("DUPLICATE_OR_INVALID_JOB_BIRTH_EVENT")
                    if captured is None:
                        self._faults.append("JOB_NEW_PROCESS_IDENTITY_UNAVAILABLE")
                    else:
                        self._births[pid] = captured
                elif kind in (EXIT_PROCESS, ABNORMAL_EXIT_PROCESS, ACTIVE_PROCESS_ZERO):
                    if kind == ABNORMAL_EXIT_PROCESS:
                        self._faults.append("PRIVATE_JOB_ABNORMAL_PROCESS_EXIT")
                    if (kind == ACTIVE_PROCESS_ZERO and pid != 0
                            or kind in (EXIT_PROCESS, ABNORMAL_EXIT_PROCESS) and pid <= 0):
                        self._faults.append("INVALID_JOB_EXIT_OR_ZERO_PACKET_PID")
                else:
                    # Memory limit, process limit, or other abnormal Job
                    # notification is diagnostic; a complete qualifying
                    # lifetime cannot ignore or self-approve it.
                    self._faults.append("UNREGISTERED_JOB_NOTIFICATION")

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "schema": SCHEMA,
                "status": "OBSERVED_ONLY_NOT_APPROVED",
                "events": list(self._events),
                "births": {pid: witness.record() for pid, witness in self._births.items()},
                "faults": sorted(set(self._faults)),
                "read_only": True,
                "gen0": {"status": "BLOCKED", "chosen": None, "release_approved": False},
            }

    def stop(self) -> dict:
        # Called after the root and all Job members have naturally exited.
        # Allow already-queued native packets to drain through two consecutive
        # empty polls, rather than truncating the final event stream at once.
        # A lost Windows packet is still fatal via cumulative Job accounting.
        if self._thread is not None and self._thread.is_alive() and not self._stop.is_set():
            deadline = time.monotonic() + 0.35
            while time.monotonic() < deadline:
                with self._lock:
                    if self._quiet_polls >= 2 or self._faults:
                        break
                time.sleep(0.01)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                with self._lock:
                    self._faults.append("PRIVATE_JOB_EVENT_THREAD_STILL_LIVE")
        return self.snapshot()

    def close(self):
        self.stop()
        with self._lock:
            handles = list(self._process_handles.items())
        errors = []
        for pid, handle in handles:
            self._kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            self._kernel.CloseHandle.restype = wintypes.BOOL
            if not self._kernel.CloseHandle(handle):
                errors.append("ORIGINAL_JOB_PROCESS_HANDLE_CLOSE_FAILED")
            else:
                with self._lock:
                    self._process_handles.pop(pid, None)
        if self._port:
            api = self._kernel.CloseHandle
            api.argtypes = (wintypes.HANDLE,)
            api.restype = wintypes.BOOL
            if not api(self._port):
                errors.append("PRIVATE_JOB_COMPLETION_PORT_CLOSE_FAILED")
            else:
                self._port = None
        if errors:
            raise EventWitnessRefused(";".join(sorted(set(errors))))
