from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

_T = TypeVar("_T")

logger = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BASE_DELAY_SECONDS = 1.0
DEFAULT_MAX_DELAY_SECONDS = 30.0


def _permanent_hub_error_types() -> tuple[type[BaseException], ...]:
    """Errors retrying can never fix: a bad model name, no access to a
    gated repo, a bad revision, a malformed repo id, or -- with offline
    mode enabled -- a cache miss (LocalEntryNotFoundError, raised when
    local_files_only=True and the file simply isn't cached; retrying can't
    materialize a file offline mode has already refused to go fetch).
    These must raise immediately, not be retried into a long, confusing
    wait that ends in the exact same error.
    """
    try:
        from huggingface_hub import errors as hub_errors
    except ImportError:
        return ()
    return (
        hub_errors.RepositoryNotFoundError,
        hub_errors.RevisionNotFoundError,
        hub_errors.GatedRepoError,
        hub_errors.DisabledRepoError,
        hub_errors.BadRequestError,
        hub_errors.HFValidationError,
        hub_errors.LocalEntryNotFoundError,
    )


def _transient_error_types() -> tuple[type[BaseException], ...]:
    """Network/server-side failures worth retrying: any Hub HTTP error that
    isn't one of the permanent types above (typically a 5xx or a rate
    limit), plus the underlying transport-level connection/timeout errors.

    huggingface_hub's HTTP transport is httpx-based as of its 1.x line
    (verified directly against the installed version rather than assumed --
    httpx.TransportError, not any requests.exceptions type, is what a real
    dropped connection or timeout actually raises at that layer). requests
    is still checked too, defensively, in case a different huggingface_hub
    version or an unrelated network path in a dependency uses it instead.
    """
    types: tuple[type[BaseException], ...] = ()
    try:
        from huggingface_hub import errors as hub_errors

        types += (hub_errors.HfHubHTTPError,)
    except ImportError:
        pass
    try:
        import httpx

        types += (httpx.TransportError,)
    except ImportError:
        pass
    try:
        import requests.exceptions as request_errors

        types += (
            request_errors.ConnectionError,
            request_errors.Timeout,
            request_errors.ChunkedEncodingError,
        )
    except ImportError:
        pass
    return types


def _exception_chain(exc: BaseException, *, max_depth: int = 8) -> list[BaseException]:
    """exc plus every __cause__/__context__ ancestor, most-recent first.

    Load libraries -- transformers in particular -- routinely catch a
    specific, typed huggingface_hub/httpx error and re-raise a generic
    OSError with a friendlier message (`raise OSError(...) from e`). The
    generic wrapper is what call sites actually see; the real, classifiable
    error only survives in the exception chain, not as the caught object's
    own type.
    """
    chain = [exc]
    current = exc
    for _ in range(max_depth):
        nxt = current.__cause__ or current.__context__
        if nxt is None or nxt in chain:
            break
        chain.append(nxt)
        current = nxt
    return chain


def is_retriable_hub_error(exc: BaseException) -> bool:
    chain = _exception_chain(exc)
    permanent = _permanent_hub_error_types()
    transient = _transient_error_types()
    # Permanent takes priority across the whole chain: a not-found error's
    # own cause can be a generic-looking transport error (an HTTP 404 is
    # still delivered over a real connection), and that must not make an
    # unfixable error look retriable.
    if any(isinstance(candidate, permanent) for candidate in chain):
        return False
    return any(isinstance(candidate, transient) for candidate in chain)


def with_hub_retries(
    func: Callable[[], _T],
    *,
    label: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay_seconds: float = DEFAULT_BASE_DELAY_SECONDS,
    max_delay_seconds: float = DEFAULT_MAX_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> _T:
    """Call func(), retrying with exponential backoff plus jitter on
    transient Hugging Face Hub / network errors only.

    Non-transient errors (bad model name, no access to a gated repo, a bad
    revision, a malformed repo id) propagate on the first attempt --
    retrying those can never succeed and would only hide a real
    configuration problem behind a long wait. Any exception type unrelated
    to Hub/network access (a bug in application code, for instance) also
    propagates immediately, for the same reason.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    attempt = 0
    while True:
        attempt += 1
        try:
            return func()
        except Exception as exc:
            if attempt >= max_attempts or not is_retriable_hub_error(exc):
                raise
            delay = min(max_delay_seconds, base_delay_seconds * (2 ** (attempt - 1)))
            delay *= 1 + random.uniform(0, 0.25)
            logger.warning(
                "%s failed (attempt %d/%d): %s: %s -- retrying in %.1fs",
                label,
                attempt,
                max_attempts,
                type(exc).__name__,
                exc,
                delay,
            )
            sleep(delay)
