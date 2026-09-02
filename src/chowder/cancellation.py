from __future__ import annotations

import threading
from typing import Protocol


class OperationCancelled(RuntimeError):
    """Raised when a CancellationToken was set while the operation it
    guards was in flight -- a clean, requested stop, not a crash."""


class _Cancellable(Protocol):
    def cancel(self, run_id: str) -> None: ...


class CancellationToken:
    """Thread-safe cooperative cancellation, shared across one run_project()
    call.

    request() is safe to call from any thread -- a TUI's main thread while
    training runs on a worker thread, for instance -- and takes effect two
    ways: it stops the next candidate (or the next bounded-repair hop) from
    starting at all, and, if a training or evaluation subprocess is already
    in flight, it terminates that subprocess immediately via the executor's
    own cancel(run_id) rather than waiting for it to finish naturally.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._active: tuple[_Cancellable, str] | None = None

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def raise_if_requested(self) -> None:
        if self._event.is_set():
            raise OperationCancelled("cancellation was requested")

    def request(self) -> None:
        self._event.set()
        with self._lock:
            active = self._active
        if active is not None:
            executor, run_id = active
            executor.cancel(run_id)

    def _register_active(self, executor: _Cancellable, run_id: str) -> None:
        with self._lock:
            self._active = (executor, run_id)
        # A request() that raced ahead of registration must still take
        # effect rather than being lost.
        if self._event.is_set():
            executor.cancel(run_id)

    def _clear_active(self) -> None:
        with self._lock:
            self._active = None
