import pytest

from chowder.cancellation import CancellationToken, OperationCancelled


class _FakeExecutor:
    def __init__(self):
        self.cancelled_run_ids = []

    def cancel(self, run_id):
        self.cancelled_run_ids.append(run_id)


def test_not_requested_initially():
    token = CancellationToken()
    assert token.requested is False


def test_raise_if_requested_is_a_noop_before_request():
    token = CancellationToken()
    token.raise_if_requested()  # must not raise


def test_raise_if_requested_raises_after_request():
    token = CancellationToken()
    token.request()
    with pytest.raises(OperationCancelled):
        token.raise_if_requested()


def test_request_is_a_noop_with_nothing_registered():
    token = CancellationToken()
    token.request()  # must not raise
    assert token.requested is True


def test_request_cancels_the_registered_active_executor():
    token = CancellationToken()
    executor = _FakeExecutor()
    token._register_active(executor, "run-1")
    token.request()
    assert executor.cancelled_run_ids == ["run-1"]


def test_a_request_that_races_ahead_of_registration_still_cancels():
    """If request() fires between two threads before _register_active() has
    run, the cancellation must not be silently lost -- registering after
    the fact must still trigger cancel()."""
    token = CancellationToken()
    token.request()
    executor = _FakeExecutor()
    token._register_active(executor, "run-1")
    assert executor.cancelled_run_ids == ["run-1"]


def test_clear_active_prevents_a_later_request_from_reaching_a_finished_run():
    token = CancellationToken()
    executor = _FakeExecutor()
    token._register_active(executor, "run-1")
    token._clear_active()
    token.request()
    assert executor.cancelled_run_ids == []


def test_registering_a_new_active_run_replaces_the_previous_one():
    token = CancellationToken()
    first = _FakeExecutor()
    second = _FakeExecutor()
    token._register_active(first, "run-1")
    token._clear_active()
    token._register_active(second, "run-2")
    token.request()
    assert first.cancelled_run_ids == []
    assert second.cancelled_run_ids == ["run-2"]
