"""The literature watchdog's decisions, pinned without the network.

``watchdog.py`` is verified live against GitHub as well -- a healthy run, and the
alert path in dry-run -- but the states that matter most (a drop whose required
checks never attached, a drop that conflicts with ``main``) cannot be produced on
demand without leaving junk on the repository. They are pinned here by stubbing
the one function that talks to the API, so each branch an operator can see is
decided by an assertion instead of by whatever the repository happens to look
like on the day the suite runs.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

WATCHDOG = Path(__file__).resolve().parent.parent / "docs" / "literature" / "watchdog.py"
_spec = importlib.util.spec_from_file_location("literature_watchdog", WATCHDOG)
assert _spec is not None and _spec.loader is not None
watchdog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(watchdog)

BRANCH = "automation/literature-watch"


def _stamp(days_ago: float) -> str:
    moment = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days_ago)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


class _API:
    """A stand-in for the GitHub REST surface the watchdog reads."""

    def __init__(
        self,
        *,
        runs: tuple[dict, ...] = (),
        drops: tuple[dict, ...] = (),
        check_runs: int = 6,
        head_age_days: float = 0.0,
        issues: tuple[dict, ...] = (),
        state_sequence: tuple[str, ...] = (),
    ) -> None:
        self.runs = list(runs)
        self.drops = list(drops)
        self.check_runs = check_runs
        self.head_age_days = head_age_days
        self.issues = list(issues)
        self.state_sequence = list(state_sequence)
        self.state_reads = 0
        self.mutations: list[tuple[str, str, dict | None]] = []

    def __call__(self, method: str, path: str, token: str, payload=None):
        assert token == "test-token"
        if method != "GET":
            self.mutations.append((method, path, payload))
            return {}
        if "/actions/workflows/" in path:
            return {"workflow_runs": self.runs}
        if "/commits/" in path and path.endswith("/check-runs?per_page=100"):
            return {"total_count": self.check_runs, "check_runs": []}
        if "/commits/" in path:
            return {"commit": {"committer": {"date": _stamp(self.head_age_days)}}}
        if path.endswith("/pulls?state=open&per_page=100"):
            return self.drops
        if "/pulls/" in path:
            number = int(path.rsplit("/", 1)[1])
            pull = dict(next(drop for drop in self.drops if drop["number"] == number))
            if self.state_sequence:
                # GitHub computes ``mergeable_state`` lazily; a sequence models a
                # value that is unknown on the first read and settled later.
                pull["mergeable_state"] = self.state_sequence[
                    min(self.state_reads, len(self.state_sequence) - 1)
                ]
                self.state_reads += 1
            return pull
        if path.endswith("/issues?state=open&per_page=100"):
            return self.issues
        raise AssertionError(f"unexpected call: {method} {path}")


def _drop(
    *,
    number: int = 214,
    created_days_ago: float = 0.1,
    mergeable_state: str = "clean",
    mergeable: bool | None = True,
    sha: str = "a" * 40,
) -> dict:
    return {
        "number": number,
        "html_url": f"https://github.com/niko4244/Chowder/pull/{number}",
        "created_at": _stamp(created_days_ago),
        "mergeable": mergeable,
        "mergeable_state": mergeable_state,
        "head": {"ref": BRANCH, "sha": sha},
    }


def _run(*, conclusion: str = "success", days_ago: float = 0.2, run_id: int = 2) -> dict:
    return {
        "id": run_id,
        "status": "completed",
        "conclusion": conclusion,
        "created_at": _stamp(days_ago),
        "html_url": f"https://github.com/niko4244/Chowder/actions/runs/{run_id}",
    }


def _guard(monkeypatch, api: _API) -> None:
    monkeypatch.setattr(watchdog, "_request", api)


def _argv(*extra: str) -> list[str]:
    return ["--repo", "niko4244/Chowder", "--token", "test-token", *extra]


def test_a_live_schedule_and_a_landable_drop_are_silent(monkeypatch) -> None:
    """The common case: nothing to say, nothing opened, exit 0."""
    api = _API(runs=(_run(),), drops=(_drop(),))
    _guard(monkeypatch, api)
    assert watchdog.main(_argv()) == 0
    assert api.mutations == []


def test_no_drop_is_not_an_alert(monkeypatch) -> None:
    """A quiet week is legitimate: the watch writes nothing when nothing is new."""
    api = _API(runs=(_run(),))
    _guard(monkeypatch, api)
    assert watchdog.main(_argv()) == 0
    assert api.mutations == []


def test_a_schedule_that_stopped_successfully_running_is_loud(monkeypatch) -> None:
    api = _API(runs=(_run(days_ago=30.0),), drops=(_drop(),))
    _guard(monkeypatch, api)
    assert watchdog.main(_argv()) == 1
    method, path, payload = api.mutations[0]
    assert (method, path) == ("POST", "/repos/niko4244/Chowder/issues")
    assert payload is not None and payload["title"] == watchdog.ISSUE_TITLE


def test_a_failing_latest_run_still_alerts_when_no_success_is_recent(monkeypatch) -> None:
    api = _API(runs=(_run(conclusion="failure", days_ago=1.0),), drops=())
    _guard(monkeypatch, api)
    assert watchdog.main(_argv()) == 1
    assert api.mutations[0][0] == "POST"


def test_a_blocked_drop_with_no_check_runs_is_the_silent_failure(monkeypatch) -> None:
    """Branch protection cannot settle on an empty rollup; nobody can merge it."""
    api = _API(runs=(_run(),), drops=(_drop(mergeable_state="blocked"),), check_runs=0, head_age_days=3.0)
    _guard(monkeypatch, api)
    assert watchdog.main(_argv()) == 1
    assert "required contexts never attached" in api.mutations[0][2]["body"]


def test_a_freshly_pushed_drop_gets_its_grace_period(monkeypatch) -> None:
    """Minutes after a push the checks are legitimately still attaching."""
    api = _API(
        runs=(_run(),),
        drops=(_drop(mergeable_state="blocked"),),
        check_runs=0,
        head_age_days=0.01,
    )
    _guard(monkeypatch, api)
    assert watchdog.main(_argv()) == 0
    assert api.mutations == []


def test_a_drop_that_conflicts_with_main_is_loud(monkeypatch) -> None:
    api = _API(
        runs=(_run(),),
        drops=(_drop(mergeable_state="dirty", mergeable=False),),
        head_age_days=0.2,
    )
    _guard(monkeypatch, api)
    assert watchdog.main(_argv()) == 1
    assert "conflicts with the base branch" in api.mutations[0][2]["body"]


def test_a_drop_waiting_past_the_window_is_a_nudge_not_a_verdict(monkeypatch) -> None:
    api = _API(runs=(_run(),), drops=(_drop(created_days_ago=9.0),))
    _guard(monkeypatch, api)
    assert watchdog.main(_argv()) == 1
    body = api.mutations[0][2]["body"]
    assert "open for 9.0 days" in body
    assert "prunes drops at 30 days" in body


def test_a_healthy_check_closes_the_tracking_issue(monkeypatch) -> None:
    """The issue cannot become furniture: the next healthy run closes it."""
    issue = {"number": 99, "title": watchdog.ISSUE_TITLE}
    api = _API(runs=(_run(),), drops=(_drop(),), issues=(issue,))
    _guard(monkeypatch, api)
    assert watchdog.main(_argv()) == 0
    assert ("POST", "/repos/niko4244/Chowder/issues/99/comments") == api.mutations[0][:2]
    assert api.mutations[1][:2] == ("PATCH", "/repos/niko4244/Chowder/issues/99")
    assert api.mutations[1][2] == {"state": "closed"}


def test_an_alert_updates_the_existing_issue_instead_of_opening_a_second(monkeypatch) -> None:
    issue = {"number": 99, "title": watchdog.ISSUE_TITLE}
    api = _API(runs=(_run(days_ago=12.0),), drops=(), issues=(issue,))
    _guard(monkeypatch, api)
    assert watchdog.main(_argv()) == 1
    assert api.mutations[0][:2] == ("PATCH", "/repos/niko4244/Chowder/issues/99")


def test_a_lazy_merge_state_is_waited_for_instead_of_skipped(monkeypatch) -> None:
    """Measured on the first live run: the state came back ``unknown`` for a drop
    whose state was ``clean`` a minute earlier. Skipping the read would skip the
    blocked-with-no-checks alarm for a whole week, so the read is retried."""
    monkeypatch.setattr(watchdog, "MERGE_STATE_PAUSE_SECONDS", 0.0)
    api = _API(
        runs=(_run(),),
        drops=(_drop(),),
        state_sequence=("unknown", "blocked"),
        check_runs=0,
        head_age_days=3.0,
    )
    _guard(monkeypatch, api)
    assert watchdog.main(_argv()) == 1
    assert "required contexts never attached" in api.mutations[0][2]["body"]
    assert api.state_reads == 2


def test_a_state_that_stays_unknown_is_reported_and_never_alerts(monkeypatch) -> None:
    """An uncomputable state must not fail a run -- and must not read as clean."""
    monkeypatch.setattr(watchdog, "MERGE_STATE_PAUSE_SECONDS", 0.0)
    api = _API(runs=(_run(),), drops=(_drop(),), state_sequence=("unknown",))
    _guard(monkeypatch, api)
    assert watchdog.main(_argv()) == 0
    assert api.mutations == []
    assert api.state_reads == watchdog.MERGE_STATE_ATTEMPTS
