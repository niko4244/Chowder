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

import base64
import datetime as dt
import importlib.util
from pathlib import Path
import sys

import pytest

WATCHDOG = Path(__file__).resolve().parent.parent / "docs" / "literature" / "watchdog.py"
_spec = importlib.util.spec_from_file_location("literature_watchdog", WATCHDOG)
assert _spec is not None and _spec.loader is not None
watchdog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(watchdog)

BRANCH = "automation/literature-watch"

#: A minimal but complete drop: the shape ``watch.py`` writes, held still so a
#: test can mutate exactly one thing and watch the checker notice it.
DROP_LOG = """# Watch log

## Automated pool drop -- 2026-10-08

> **Unvetted, machine-appended -- nothing here is registered.** These are
> arXiv search hits that named a surface mechanism in a watched primary
> category, listed for triage.

- window: last 14 day(s) | surfaced: 1 | new: 1 | already in the log: 0
- source: `.github/workflows/literature-watch.yml` (`watch.py --append-log`)

### compression
- **YANchor-4B: O(1) Expert Routing at 2026 Scale** -- `arXiv:2610.12345v1` -- 2026-10-05 -- cs.LG -- matched: quantization
  - https://arxiv.org/abs/2610.12345
"""


def _stamp(days_ago: float) -> str:
    moment = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days_ago)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _workflow_stamp(days_ago: float) -> str:
    """The workflow object's own clock format, which is not the runs' format.

    Read live on the first dry run of this script: ``/actions/workflows/<name>``
    answers ``created_at`` as ``2026-10-08T11:11:42.000-05:00``, while runs and
    commits answer ``...Z``. Both are handed to the same parser, so both shapes are
    pinned here rather than only the one the stubs happened to use.
    """
    moment = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days_ago)
    return moment.astimezone(dt.timezone(dt.timedelta(hours=-5))).strftime(
        "%Y-%m-%dT%H:%M:%S.000-05:00"
    )


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
        log_text: str = DROP_LOG,
        workflow_runs: dict[str, tuple[dict, ...]] | None = None,
        workflow_age_days: float = 400.0,
        workflow_updated_days: float = 400.0,
        contents_error: bool = False,
    ) -> None:
        self.runs = list(runs)
        self.log_text = log_text
        self.workflow_runs = workflow_runs
        self.workflow_age_days = workflow_age_days
        self.workflow_updated_days = workflow_updated_days
        self.contents_error = contents_error
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
        if "/actions/workflows/" in path and "/runs" in path:
            name = path.split("/actions/workflows/", 1)[1].split("/", 1)[0]
            runs = (self.workflow_runs or {}).get(name, self.runs)
            return {"workflow_runs": list(runs)}
        if "/actions/workflows/" in path:
            # The workflow object itself: read for its registration date (which
            # separates "added this morning" from "never fired") and for its own
            # update time, which is where the drift check's history begins.
            return {
                "created_at": _workflow_stamp(self.workflow_age_days),
                "updated_at": _workflow_stamp(self.workflow_updated_days),
            }
        if "/contents/" in path:
            if self.contents_error:
                raise watchdog.GitHubError("GET contents -> 404: Not Found")
            return {
                "encoding": "base64",
                "content": base64.b64encode(self.log_text.encode("utf-8")).decode("ascii"),
            }
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


# ---------------- every scheduled workflow, not only the watch ----------------


def _load_watch():
    """The watch's own module, loaded for its declarations and its renderer.

    Loading it issues no request: the module is definitions, constants and a
    guarded ``main()``. ``sys.modules`` has to see it because one of its classes
    is a dataclass, which resolves annotations through the module registry.
    """
    path = WATCHDOG.parent / "watch.py"
    spec = importlib.util.spec_from_file_location("literature_watch_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["literature_watch_probe"] = module
    spec.loader.exec_module(module)
    return module


def test_age_days_reads_both_clock_formats_github_answers_with() -> None:
    """A single-format parser passed the stubs and crashed on the first live run."""
    zulu = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2.5)
    offset = zulu.astimezone(dt.timezone(dt.timedelta(hours=-5)))

    for stamp in (
        zulu.strftime("%Y-%m-%dT%H:%M:%SZ"),
        offset.strftime("%Y-%m-%dT%H:%M:%S.000-05:00"),
    ):
        assert watchdog._age_days(stamp) == pytest.approx(2.5, abs=0.01)
    assert watchdog._age_days(None) is None
    assert watchdog._age_days("") is None
    with pytest.raises(watchdog.GitHubError):
        watchdog._age_days("last tuesday")


def test_a_cron_cadence_is_measured_from_the_expression() -> None:
    """The liveness window comes from the schedule, so moving one moves the other."""
    assert watchdog.cron_cadence_days("0 6 * * 1")[0] == pytest.approx(7.0, rel=1e-6)
    assert watchdog.cron_cadence_days("0 6 * * *")[0] == pytest.approx(1.0, rel=1e-6)
    assert watchdog.cron_cadence_days("*/15 * * * *")[0] == pytest.approx(0.25 / 24, rel=1e-3)
    with pytest.raises(watchdog.ScheduleError):
        watchdog.cron_cadence_days("0 6 * *")


def test_scheduled_workflows_reads_only_files_that_declare_a_schedule(tmp_path) -> None:
    (tmp_path / "scheduled.yml").write_text(
        "on:\n  schedule:\n    - cron: '0 6 * * 1'\n  workflow_dispatch:\n", encoding="utf-8"
    )
    (tmp_path / "push_only.yml").write_text(
        "on:\n  push:\n    branches: [main]\n", encoding="utf-8"
    )
    (tmp_path / "unreadable.yml").write_text(
        "on:\n  schedule:\n    - cron: 'every other tuesday'\n", encoding="utf-8"
    )

    found = {entry["file"]: entry for entry in watchdog.scheduled_workflows(tmp_path)}

    assert set(found) == {"scheduled.yml", "unreadable.yml"}
    assert found["scheduled.yml"]["crons"] == ["0 6 * * 1"]
    assert found["scheduled.yml"]["cadence_days"] == pytest.approx(7.0, rel=1e-6)
    assert found["scheduled.yml"]["errors"] == []
    assert found["unreadable.yml"]["cadence_days"] is None
    assert found["unreadable.yml"]["errors"]


def test_a_schedule_whose_cron_cannot_be_read_is_loud(monkeypatch, tmp_path) -> None:
    (tmp_path / "odd.yml").write_text(
        "on:\n  schedule:\n    - cron: 'weekly'\n", encoding="utf-8"
    )
    api = _API(runs=(_run(),), drops=())
    _guard(monkeypatch, api)

    assert watchdog.main(_argv("--workflows-dir", str(tmp_path))) == 1

    assert "readable cadence" in api.mutations[0][2]["body"]


def test_a_schedule_that_stopped_firing_is_loud_even_when_the_watch_is_fine(
    monkeypatch,
) -> None:
    """The check the repo asked for: a dead schedule anywhere, not only this one."""
    api = _API(
        runs=(_run(),),
        drops=(),
        workflow_runs={"literature-watchdog.yml": (_run(days_ago=30.0, run_id=9),)},
    )
    _guard(monkeypatch, api)

    assert watchdog.main(_argv()) == 1

    body = api.mutations[0][2]["body"]
    assert "scheduled workflow literature-watchdog.yml still fires" in body
    assert "30.0 days old" in body


def test_a_newly_registered_schedule_is_not_yet_overdue(monkeypatch) -> None:
    """A workflow added hours ago has not had its first Monday yet."""
    api = _API(
        runs=(_run(),),
        drops=(),
        workflow_runs={"literature-watchdog.yml": ()},
        workflow_age_days=0.4,
    )
    _guard(monkeypatch, api)

    assert watchdog.main(_argv()) == 0
    assert api.mutations == []


def test_a_schedule_registered_long_ago_that_never_fired_is_loud(monkeypatch) -> None:
    api = _API(
        runs=(_run(),),
        drops=(),
        workflow_runs={"literature-watchdog.yml": ()},
        workflow_age_days=60.0,
    )
    _guard(monkeypatch, api)

    assert watchdog.main(_argv()) == 1
    assert "no scheduled run" in api.mutations[0][2]["body"]


# ---------------- the drop's own rules ----------------


def test_a_drop_that_follows_the_rules_is_silent(monkeypatch) -> None:
    api = _API(runs=(_run(),), drops=(_drop(),))
    _guard(monkeypatch, api)

    assert watchdog.main(_argv()) == 0
    assert api.mutations == []


def test_a_drop_carrying_a_result_shaped_number_is_loud(monkeypatch) -> None:
    api = _API(
        runs=(_run(),),
        drops=(_drop(),),
        log_text=DROP_LOG.replace(
            "- **YANchor-4B: O(1) Expert Routing at 2026 Scale**",
            "- **A 2.4x faster adapter merge**",
        ),
    )
    _guard(monkeypatch, api)

    assert watchdog.main(_argv()) == 1
    body = api.mutations[0][2]["body"]
    assert "result-shaped number" in body


def test_a_number_in_the_matched_terms_is_loud(monkeypatch) -> None:
    api = _API(
        runs=(_run(),),
        drops=(_drop(),),
        log_text=DROP_LOG.replace("matched: quantization", "matched: quantization 3.2"),
    )
    _guard(monkeypatch, api)

    assert watchdog.main(_argv()) == 1
    assert "matched-terms cell carries a number" in api.mutations[0][2]["body"]


def test_a_drop_that_pastes_abstract_text_is_loud(monkeypatch) -> None:
    api = _API(
        runs=(_run(),),
        drops=(_drop(),),
        log_text=DROP_LOG.replace(
            "### compression",
            "### compression\nThis paper shows merged adapters retain 97.5% of the gains.",
        ),
    )
    _guard(monkeypatch, api)

    assert watchdog.main(_argv()) == 1
    assert "none of the shapes a drop may carry" in api.mutations[0][2]["body"]


def test_a_drop_without_the_unvetted_disclosure_is_loud(monkeypatch) -> None:
    api = _API(
        runs=(_run(),),
        drops=(_drop(),),
        log_text=DROP_LOG.replace(
            "**Unvetted, machine-appended -- nothing here is registered.**",
            "Collected by the weekly watch.",
        ),
    )
    _guard(monkeypatch, api)

    assert watchdog.main(_argv()) == 1
    assert "no unvetted disclosure" in api.mutations[0][2]["body"]


def test_a_drop_whose_url_belongs_to_another_paper_is_loud(monkeypatch) -> None:
    api = _API(
        runs=(_run(),),
        drops=(_drop(),),
        log_text=DROP_LOG.replace("abs/2610.12345", "abs/2610.99999"),
    )
    _guard(monkeypatch, api)

    assert watchdog.main(_argv()) == 1
    assert "not followed by its own url line" in api.mutations[0][2]["body"]


def test_an_unlabelled_drop_section_is_loud(monkeypatch) -> None:
    api = _API(
        runs=(_run(),),
        drops=(_drop(),),
        log_text=DROP_LOG.replace("## Automated pool drop -- 2026-10-08", "## Automated pool drop"),
    )
    _guard(monkeypatch, api)

    assert watchdog.main(_argv()) == 1
    assert "not the dated label" in api.mutations[0][2]["body"]


def test_a_drop_branch_with_no_drop_at_all_is_loud(monkeypatch) -> None:
    api = _API(runs=(_run(),), drops=(_drop(),), log_text="# Watch log\n")
    _guard(monkeypatch, api)

    assert watchdog.main(_argv()) == 1
    assert "carries no drop section at all" in api.mutations[0][2]["body"]


def test_a_drop_that_cannot_be_read_is_loud_not_silent(monkeypatch) -> None:
    """A drop nobody could read is not a drop that satisfies the rules."""
    api = _API(runs=(_run(),), drops=(_drop(),), contents_error=True)
    _guard(monkeypatch, api)

    assert watchdog.main(_argv()) == 1
    assert "could not be read" in api.mutations[0][2]["body"]


def test_a_legacy_titled_issue_is_still_the_tracking_issue(monkeypatch) -> None:
    """Renaming the alert must not orphan the open alert it was carrying."""
    issue = {"number": 42, "title": watchdog.LEGACY_ISSUE_TITLES[0]}
    api = _API(runs=(_run(days_ago=12.0),), drops=(), issues=(issue,))
    _guard(monkeypatch, api)

    assert watchdog.main(_argv()) == 1
    assert api.mutations[0][:2] == ("PATCH", "/repos/niko4244/Chowder/issues/42")


def test_the_watch_writers_own_drop_satisfies_the_checker() -> None:
    """The writer/checker pin, in one test instead of in a convention.

    ``watch.py`` renders a drop in exactly one place; this runs that renderer over
    a synthetic hit -- with the source's own name-numbers in the title, which is
    the case that would false-alarm if the two ever drifted -- and requires the
    checker to accept it.
    """
    watch = _load_watch()
    section, stats = watch.build_append_section(
        [
            {
                "profile": watch.PROFILES[0]["name"],
                "entries": [
                    {
                        "arxiv_id": "2610.12345v1",
                        "title": "YANchor-4B: O(1) Expert Routing at 2026 Scale",
                        "summary": "quantization and continued pretraining",
                        "published": "2026-10-05T00:00:00Z",
                        "primary_category": "cs.LG",
                        "url": "https://arxiv.org/abs/2610.12345",
                    }
                ],
            }
        ],
        log_text="# Watch log\n",
        days=14,
        limit=40,
    )
    assert stats["appended"] == 1

    result = watchdog.check_drop_invariants(
        "# Watch log\n\n" + section, source="the writer's own output", require_drop=True
    )

    assert result["healthy"], result["detail"]


# ---------------- does the schedule fire *this* often? ----------------
#: One workflow file of our own, so the declared cadence under test is the only
#: one in the run: the repository's own two workflows are weekly, and a test that
#: wants a daily history against a weekly declaration has to say which is which.


def _one_workflow(tmp_path, cron: str = "0 6 * * 1") -> str:
    (tmp_path / "scheduled.yml").write_text(
        f"on:\n  schedule:\n    - cron: '{cron}'\n", encoding="utf-8"
    )
    return str(tmp_path)


def _spaced(step_days: float, count: int, *, first_id: int = 100) -> tuple[dict, ...]:
    """``count`` runs, ``step_days`` apart, newest first by id."""
    return tuple(
        _run(days_ago=step_days * index, run_id=first_id - index) for index in range(count)
    )


def test_a_history_that_does_not_match_the_declared_cadence_is_loud(monkeypatch, tmp_path) -> None:
    """The drift the liveness checks cannot see: alive, fresh, and not this schedule.

    A workflow firing every day while its file declares a weekly cron passes every
    other check in this script -- a successful run inside the window, a schedule
    that fires on time. The declaration is the only place that says otherwise, so
    the run it drives is not the run the file describes.
    """
    api = _API(
        runs=(_run(),),
        drops=(),
        workflow_runs={"scheduled.yml": _spaced(1.0, 6)},
    )
    _guard(monkeypatch, api)

    assert watchdog.main(_argv("--workflows-dir", _one_workflow(tmp_path))) == 1

    body = api.mutations[0][2]["body"]
    assert "describe different schedules" in body
    assert "1.00 days apart" in body
    # Only the drift row is unhappy: the same runs keep the liveness check green,
    # which is exactly why the drift check has to exist.
    assert body.count("| ATTENTION |") == 1


def test_a_history_that_matches_the_declaration_is_silent(monkeypatch, tmp_path, capsys) -> None:
    """The control, and the row a healthy run still prints."""
    api = _API(
        runs=(_run(),),
        drops=(),
        workflow_runs={"scheduled.yml": _spaced(7.0, 6, first_id=200)},
    )
    _guard(monkeypatch, api)

    assert watchdog.main(_argv("--workflows-dir", _one_workflow(tmp_path))) == 0
    assert api.mutations == []

    printed = capsys.readouterr().out
    assert "scheduled workflow scheduled.yml fires on the cadence it declares" in printed
    assert "implies 7.00 days between fires" in printed
    assert "days apart (median" in printed


def test_a_declaration_change_resets_the_drift_baseline(monkeypatch, tmp_path) -> None:
    """The old history belongs to the old cron, so the change is not an alarm.

    The runs below are three weekly fires since the declaration changed and three
    monthly ones from before it. Without the truncation the widest gap is 30 days
    against a declared 7, and a schedule that was deliberately changed would alarm
    until the old runs aged out of the window.
    """
    runs = (
        _run(days_ago=0.1, run_id=6),
        _run(days_ago=7.1, run_id=5),
        _run(days_ago=14.1, run_id=4),
        _run(days_ago=44.0, run_id=3),
        _run(days_ago=74.0, run_id=2),
        _run(days_ago=104.0, run_id=1),
    )
    api = _API(
        runs=(_run(),),
        drops=(),
        workflow_runs={"scheduled.yml": runs},
        workflow_updated_days=20.0,
    )
    _guard(monkeypatch, api)

    assert watchdog.main(_argv("--workflows-dir", _one_workflow(tmp_path))) == 0
    assert api.mutations == []


def test_too_little_history_is_not_a_drift_finding(monkeypatch, tmp_path) -> None:
    """Two fires cannot measure a cadence, so they are not asked to."""
    api = _API(
        runs=(_run(),),
        drops=(),
        workflow_runs={"scheduled.yml": _spaced(1.0, 2)},
    )
    _guard(monkeypatch, api)

    assert watchdog.main(_argv("--workflows-dir", _one_workflow(tmp_path))) == 0
    assert api.mutations == []


def test_a_history_that_spans_much_longer_than_declared_is_loud(monkeypatch, tmp_path) -> None:
    """The other direction: a weekly declaration whose runs are a month apart."""
    api = _API(
        runs=(_run(),),
        drops=(),
        workflow_runs={"scheduled.yml": _spaced(30.0, 5, first_id=300)},
    )
    _guard(monkeypatch, api)

    assert watchdog.main(_argv("--workflows-dir", _one_workflow(tmp_path))) == 1

    body = api.mutations[0][2]["body"]
    assert "describe different schedules" in body
    assert "30.00 days apart" in body
