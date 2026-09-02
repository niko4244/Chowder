import pytest

from chowder.hf_resilience import with_hub_retries


# --- with_hub_retries core loop mechanics ------------------------------------
#
# These tests are deliberately decoupled from real huggingface_hub/httpx
# exception types (monkeypatching is_retriable_hub_error directly instead) so
# they run in every environment, including the regular CI test jobs that
# only install chowder's [dev] extra, not [train] -- huggingface_hub/httpx
# aren't guaranteed importable there. Classification correctness against the
# real library types is covered separately below, gated on those imports.


def test_succeeds_on_first_attempt_without_sleeping(monkeypatch):
    monkeypatch.setattr("chowder.hf_resilience.is_retriable_hub_error", lambda exc: True)
    calls = []
    assert with_hub_retries(lambda: "ok", label="t", sleep=calls.append) == "ok"
    assert calls == []


def test_retries_a_retriable_error_and_then_succeeds(monkeypatch):
    monkeypatch.setattr("chowder.hf_resilience.is_retriable_hub_error", lambda exc: True)
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    sleeps = []
    result = with_hub_retries(flaky, label="t", sleep=sleeps.append, base_delay_seconds=0.01)
    assert result == "ok"
    assert attempts["n"] == 3
    assert len(sleeps) == 2


def test_raises_immediately_when_not_retriable(monkeypatch):
    monkeypatch.setattr("chowder.hf_resilience.is_retriable_hub_error", lambda exc: False)
    attempts = {"n": 0}

    def always_fails():
        attempts["n"] += 1
        raise ValueError("permanent")

    sleeps = []
    with pytest.raises(ValueError, match="permanent"):
        with_hub_retries(always_fails, label="t", sleep=sleeps.append)
    assert attempts["n"] == 1
    assert sleeps == []


def test_exhausts_max_attempts_and_raises_the_last_error(monkeypatch):
    monkeypatch.setattr("chowder.hf_resilience.is_retriable_hub_error", lambda exc: True)
    attempts = {"n": 0}

    def always_fails():
        attempts["n"] += 1
        raise RuntimeError(f"fail {attempts['n']}")

    sleeps = []
    with pytest.raises(RuntimeError, match="fail 3"):
        with_hub_retries(
            always_fails, label="t", max_attempts=3, sleep=sleeps.append, base_delay_seconds=0.01
        )
    assert attempts["n"] == 3
    assert len(sleeps) == 2  # slept between attempts 1->2 and 2->3, not after the last


def test_backoff_delay_grows_exponentially_and_caps(monkeypatch):
    monkeypatch.setattr("chowder.hf_resilience.is_retriable_hub_error", lambda exc: True)
    monkeypatch.setattr("chowder.hf_resilience.random.uniform", lambda a, b: 0.0)  # no jitter
    attempts = {"n": 0}

    def always_fails():
        attempts["n"] += 1
        raise RuntimeError("fail")

    sleeps = []
    with pytest.raises(RuntimeError):
        with_hub_retries(
            always_fails,
            label="t",
            max_attempts=5,
            sleep=sleeps.append,
            base_delay_seconds=1.0,
            max_delay_seconds=3.5,
        )
    # 1 * 2**0, 1 * 2**1, then capped at 3.5 instead of 1 * 2**2 = 4 and 1 * 2**3 = 8
    assert sleeps == [1.0, 2.0, 3.5, 3.5]


def test_max_attempts_below_one_is_rejected():
    with pytest.raises(ValueError, match="max_attempts must be at least 1"):
        with_hub_retries(lambda: "ok", label="t", max_attempts=0)


# --- real classification against actual library exception types -------------


def _fake_hub_response(status_code: int):
    """A real httpx.Response, not a hand-rolled stand-in -- HfHubHTTPError's
    own constructor reads several attributes off it (.request, .headers)
    that a minimal fake is easy to get subtly wrong."""
    import httpx

    request = httpx.Request("GET", "https://huggingface.co/fake")
    return httpx.Response(status_code, request=request)


def test_permanent_hub_errors_are_never_retried():
    hub_errors = pytest.importorskip("huggingface_hub.errors")
    pytest.importorskip("httpx")
    from chowder.hf_resilience import is_retriable_hub_error

    exc = hub_errors.RepositoryNotFoundError("nope", response=_fake_hub_response(404))
    assert is_retriable_hub_error(exc) is False


def test_generic_hub_http_error_is_retried():
    hub_errors = pytest.importorskip("huggingface_hub.errors")
    pytest.importorskip("httpx")
    from chowder.hf_resilience import is_retriable_hub_error

    exc = hub_errors.HfHubHTTPError("server error", response=_fake_hub_response(503))
    assert is_retriable_hub_error(exc) is True


def test_httpx_transport_errors_are_retried():
    httpx = pytest.importorskip("httpx")
    from chowder.hf_resilience import is_retriable_hub_error

    assert is_retriable_hub_error(httpx.ConnectTimeout("timed out")) is True
    assert is_retriable_hub_error(httpx.ConnectError("refused")) is True


def test_unrelated_exceptions_are_never_retried():
    from chowder.hf_resilience import is_retriable_hub_error

    assert is_retriable_hub_error(ValueError("not a hub error")) is False
    assert is_retriable_hub_error(KeyError("also not")) is False


def test_wrapped_permanent_error_is_found_through_the_cause_chain():
    """Mirrors what transformers actually does: catch a specific
    huggingface_hub error and re-raise a generic OSError with `from e`. The
    wrapper itself carries no useful type information -- only its __cause__
    does."""
    hub_errors = pytest.importorskip("huggingface_hub.errors")
    pytest.importorskip("httpx")
    from chowder.hf_resilience import is_retriable_hub_error

    try:
        try:
            raise hub_errors.RepositoryNotFoundError("nope", response=_fake_hub_response(404))
        except hub_errors.RepositoryNotFoundError as inner:
            raise OSError("wrapped like transformers.utils.hub.cached_file does") from inner
    except OSError as wrapped:
        assert type(wrapped) is OSError  # confirms the type itself is uninformative
        assert is_retriable_hub_error(wrapped) is False


def test_wrapped_transient_error_is_found_through_the_cause_chain():
    httpx = pytest.importorskip("httpx")
    from chowder.hf_resilience import is_retriable_hub_error

    try:
        try:
            raise httpx.ConnectTimeout("simulated timeout")
        except httpx.ConnectTimeout as inner:
            raise OSError("wrapped like transformers.utils.hub.cached_file does") from inner
    except OSError as wrapped:
        assert is_retriable_hub_error(wrapped) is True


def test_permanent_wins_over_a_transient_looking_root_cause():
    """A not-found error's own root cause can be a generic HTTP status
    error -- that must not make an unfixable error look retriable."""
    httpx = pytest.importorskip("httpx")
    hub_errors = pytest.importorskip("huggingface_hub.errors")
    from chowder.hf_resilience import is_retriable_hub_error

    try:
        try:
            request = httpx.Request("GET", "https://huggingface.co/nope")
            raise httpx.HTTPStatusError(
                "404", request=request, response=httpx.Response(404, request=request)
            )
        except httpx.HTTPStatusError as inner:
            raise hub_errors.RepositoryNotFoundError(
                "nope", response=_fake_hub_response(404)
            ) from inner
    except hub_errors.RepositoryNotFoundError as chained:
        assert is_retriable_hub_error(chained) is False


# --- cache_status -------------------------------------------------------------


def test_cache_status_hit_when_a_cached_path_is_found(monkeypatch):
    pytest.importorskip("huggingface_hub")
    from chowder.hf_resilience import cache_status

    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda **kwargs: "/cache/models--org--model/snapshots/abc/config.json",
    )
    assert cache_status("org/model", "main") == "hit"


def test_cache_status_miss_when_nothing_is_cached(monkeypatch):
    pytest.importorskip("huggingface_hub")
    from chowder.hf_resilience import cache_status

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda **kwargs: None)
    assert cache_status("org/model", None) == "miss"


def test_cache_status_miss_for_the_cached_no_exist_sentinel(monkeypatch):
    """try_to_load_from_cache can return a special non-string sentinel
    meaning "we already know this file doesn't exist at this revision" --
    that is not a usable cached file, so it must count as a miss too."""
    pytest.importorskip("huggingface_hub")
    from chowder.hf_resilience import cache_status

    sentinel = object()
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda **kwargs: sentinel)
    assert cache_status("org/model", None) == "miss"


def test_cache_status_passes_repo_id_revision_and_filename_through(monkeypatch):
    pytest.importorskip("huggingface_hub")
    from chowder.hf_resilience import cache_status

    seen = {}

    def fake_try_to_load_from_cache(*, repo_id, filename, revision):
        seen.update(repo_id=repo_id, filename=filename, revision=revision)
        return None

    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache", fake_try_to_load_from_cache
    )
    cache_status("org/model", "abc123")
    assert seen == {"repo_id": "org/model", "filename": "config.json", "revision": "abc123"}
