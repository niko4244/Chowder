#!/usr/bin/env python3
"""The literature watch's own watchdog: is anything about it still alive?

`watch.py` fetches and screens; the schedule drops what is new and opens a
reviewable pull request. Any of that can stop without anything turning red -- a
schedule that never fires (this one or any other workflow's), a drop whose
required checks never report, a drop nobody ever merges, a drop whose content
breaks the rules it was written under -- and a silently broken schedule looks
exactly like a quiet week. This script is the part that tells the two apart, and
it is deliberately the only part that can shout.

What it checks, in the order an operator would ask:

1. **Did the watch run?** A successful run of the watch workflow within
   ``--max-run-age-days``. A schedule that has not completed successfully is the
   alarm the rest of this file exists to raise, because *no drop* is a legitimate
   outcome (the watch writes nothing when nothing new matches, README rule 5) and
   therefore cannot be an alarm by itself.
2. **Did every scheduled workflow in this repository fire?** Not only the watch:
   every workflow file that declares ``on.schedule`` is read, its cron is
   simulated forward to measure the cadence it implies, and the newest run of
   that workflow with ``event=schedule`` is compared against that cadence plus
   ``--schedule-grace-days``. A schedule that stops firing writes nothing
   anywhere, so this check is the only place it can be noticed at all.
3. **Can the drop land, and how long has it waited?** An open drop whose required
   checks never reported is stuck no matter who looks at it: branch protection
   cannot settle on an empty check rollup. That is the failure mode
   `literature-watch.yml` approves its own parked run to avoid, and this is the
   check that notices if it comes back. A drop that conflicts with `main` cannot
   be merged until the next run rebuilds the machine-owned branch, and a drop
   older than ``--max-drop-age-days`` is a nudge, not a verdict.
4. **Does the drop follow the log's own rules?** The merged log and each open
   drop's own copy -- the text a merge would land -- are checked against the
   invariants the drop is written under: the dated label, the unvetted
   disclosure, no quoted number, and no text the drop's shapes do not describe.
   That last one is what "no abstract text" means mechanically: a drop is a
   triage list of ids, dates and matched terms, never a paragraph.

Alerts are loud twice over: the run exits non-zero, and one tracking issue is
opened or updated (and closed again the moment a check is healthy, so it cannot
become furniture). Nothing here touches the log, the branch or the drop: this
script reads, reports, and opens one issue. It never re-runs the screen and never
judges whether a hit matters -- those are the watch's job and the reader's.

Usage:
    GITHUB_TOKEN=... python docs/literature/watchdog.py \
        --repo owner/name --watch-workflow literature-watch.yml \
        --branch automation/literature-watch --max-run-age-days 8 \
        --max-drop-age-days 7 --max-blocked-hours 24
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Mapping

API = "https://api.github.com"
ISSUE_TITLE = "[watchdog] a scheduled workflow or the literature drop needs attention"
#: Titles this script used before the check set widened. An issue under one of
#: them is still *the* tracking issue: renaming the alert must not orphan the
#: open alert it was carrying.
LEGACY_ISSUE_TITLES = ("[watchdog] the literature watch needs attention",)
DRY_RUN = False

#: How long to wait between reads of a merge state GitHub has not computed yet,
#: and how many times to read it. Kept as module constants so a test can set the
#: pause to zero without touching the code under test.
MERGE_STATE_PAUSE_SECONDS = 2.0
MERGE_STATE_ATTEMPTS = 5

#: Where the schedule declarations and the log live, resolved from this file so
#: the watchdog always reads the repository it was checked out with.
DEFAULT_WORKFLOWS_DIR = Path(__file__).resolve().parents[2] / ".github" / "workflows"
DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "WATCH_LOG.md"
LOG_REPO_PATH = "docs/literature/WATCH_LOG.md"

#: How long past its own cadence a schedule may run late before it counts as
#: stale. One day is the grace the weekly watch already gets from
#: ``--max-run-age-days`` (7 + 1), generalised to whatever cadence a workflow's
#: own cron implies instead of being fixed for one workflow.
SCHEDULE_GRACE_DAYS = 1.0

#: How far ahead a cron expression is simulated to measure its own cadence. Two
#: months of minutes keeps weekly, daily, monthly and quarter-hourly expressions
#: inside a single horizon; an expression that fires fewer than twice inside it
#: is reported with the horizon as its cadence rather than guessed at.
CRON_HORIZON_DAYS = 62

_MONTH_NAMES = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_WEEKDAY_NAMES = ("sun", "mon", "tue", "wed", "thu", "fri", "sat")

# ---------------------------------------------------------------------------
# the drop's own rules, verified instead of assumed
# ---------------------------------------------------------------------------

#: The shapes ``watch.py`` writes into a drop, as `docs/literature/README.md`
#: states them. The writer is pinned against this checker by a test that runs the
#: writer's own renderer through it, so the two cannot drift apart silently.
DROP_HEADING_PREFIX = "## Automated pool drop"
DROP_HEADING_RE = re.compile(r"^## Automated pool drop -- (?P<date>\d{4}-\d{2}-\d{2})$")
DROP_UNVETTED_MARK = "**Unvetted, machine-appended -- nothing here is registered.**"
DROP_ENTRY_RE = re.compile(
    r"^- \*\*(?P<title>.+?)\*\* -- `(?P<id>arXiv:(?P<number>\d{4}\.\d{4,5})(?:v\d+)?)` -- "
    r"(?P<date>\d{4}-\d{2}-\d{2}) -- (?P<category>[A-Za-z][A-Za-z.\-]*) -- "
    r"matched: (?P<terms>.+)$"
)
DROP_URL_RE = re.compile(r"^  - https://arxiv\.org/abs/(?P<number>\d{4}\.\d{4,5})(?:v\d+)?$")
DROP_MENTION_RE = re.compile(
    r"^- (?:window: .+|source: .+|\d+ further new hit\(s\) exceeded the drop limit; .+)$"
)
DROP_SUBSECTION_RE = re.compile(r"^### [a-z][a-z0-9-]{0,39}$")
#: A number shaped like a *result* rather than a name: ``2.4x``, ``40%``,
#: ``3 times``, ``2x faster``. ``YANchor-4B`` and ``NTCIR-19`` are names and do
#: not match. A title is quoted verbatim from the source, so it may carry the
#: source's own name-numbers -- and must not carry a claim, which is why this
#: regex runs over the whole line rather than over the machine-authored fields.
DROP_MEASUREMENT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:[x\u00d7%]|times\b|faster\b|slower\b|speedup\b)", re.IGNORECASE
)
_DIGIT_RE = re.compile(r"\d")
DROP_TITLE_MAX_CHARS = 400
DROP_TERMS_MAX_CHARS = 200
DROP_VIOLATIONS_NAMED = 4


class GitHubError(RuntimeError):
    """A GitHub API call failed; the message carries the status and body."""


class ScheduleError(RuntimeError):
    """A cron expression the watchdog will not guess the meaning of."""


def _request(method: str, path: str, token: str, payload: dict[str, Any] | None = None) -> Any:
    url = path if path.startswith("http") else f"{API}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    if DRY_RUN and method != "GET":
        print(f"[dry-run] {method} {url}")
        if payload is not None:
            print(json.dumps(payload, indent=2)[:4000])
        return None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:  # pragma: no cover - network path
        detail = error.read().decode("utf-8", "replace")
        raise GitHubError(f"{method} {url} -> {error.code}: {detail[:400]}") from error
    return json.loads(body) if body else None


def _age_days(timestamp: str | None) -> float | None:
    """How long ago a GitHub timestamp was, in days.

    GitHub is not one clock format: run and commit timestamps end in ``Z``, while
    the workflow object's ``created_at`` comes back with a UTC offset
    (``2026-10-08T11:11:42.000-05:00`` -- measured on the first live dry run of
    this script, where a single-format parse raised instead of reading the age it
    had already been handed). Both are ISO-8601, so both are parsed as such, and a
    stamp that is neither raises naming itself rather than being reported as an
    age of zero or as "never fired".
    """
    if not timestamp:
        return None
    try:
        parsed = dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise GitHubError(
            f"the timestamp {timestamp!r} is in no format this script can read"
        ) from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return (dt.datetime.now(dt.timezone.utc) - parsed).total_seconds() / 86400.0


def _table_row(name: str, state: str, detail: str) -> str:
    return f"| {name} | {state} | {detail} |"


def _clip(text: str, limit: int = 110) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


# ---------------------------------------------------------------------------
# cron: the cadence a schedule declares, measured from the schedule itself
# ---------------------------------------------------------------------------


def _cron_value(token: str, names: tuple[str, ...], base: int) -> int:
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    if token in names:
        return names.index(token) + base
    raise ScheduleError(f"cron token {token!r} is neither a number nor a declared name")


def _expand_cron_field(
    field: str, low: int, high: int, names: tuple[str, ...] = (), base: int = 0
) -> frozenset[int]:
    values: set[int] = set()
    for part in field.split(","):
        part = part.strip().lower()
        if not part:
            raise ScheduleError(f"empty part in cron field {field!r}")
        step = 1
        if "/" in part:
            part, _, raw_step = part.partition("/")
            if not raw_step.isdigit() or int(raw_step) <= 0:
                raise ScheduleError(f"cron step {raw_step!r} is not a positive integer")
            step = int(raw_step)
        if part in {"*", ""}:
            start, end = low, high
        elif "-" in part:
            raw_start, _, raw_end = part.partition("-")
            start = _cron_value(raw_start, names, base)
            end = _cron_value(raw_end, names, base)
        else:
            start = end = _cron_value(part, names, base)
        if start < low or end > high or end < start:
            raise ScheduleError(f"cron range {part!r} is outside {low}..{high}")
        values.update(range(start, end + 1, step))
    return frozenset(values)


def cron_cadence_days(
    expression: str, *, horizon_days: int = CRON_HORIZON_DAYS, now: dt.datetime | None = None
) -> tuple[float, int]:
    """The widest gap between two fires of ``expression``, and how many it fires.

    The cadence is *measured from the expression* rather than declared beside it,
    because a threshold stated a second time is a second place to be wrong: a
    workflow that moves from weekly to daily carries its own liveness window with
    it. Fires are found by walking the horizon one minute at a time -- two months
    of minutes is a few hundred milliseconds, which this job can afford once a
    week, and no calendar rule is re-implemented from memory.

    Standard cron semantics: day-of-month and day-of-week are OR-ed when both are
    restricted, names are accepted for months and weekdays, and Sunday is 0 or 7.
    An expression that fires fewer than twice inside the horizon returns the
    horizon as its cadence, which is deliberately generous -- a schedule this
    sparse is not one to guess about.
    """
    fields = expression.split()
    if len(fields) != 5:
        raise ScheduleError(f"cron expression {expression!r} does not have five fields")
    minutes = _expand_cron_field(fields[0], 0, 59)
    hours = _expand_cron_field(fields[1], 0, 23)
    days_of_month = _expand_cron_field(fields[2], 1, 31)
    months = _expand_cron_field(fields[3], 1, 12, _MONTH_NAMES, base=1)
    days_of_week = _expand_cron_field(fields[4].replace("7", "0"), 0, 6, _WEEKDAY_NAMES, base=0)
    day_of_month_restricted = fields[2].strip() != "*"
    day_of_week_restricted = fields[4].strip() != "*"

    start = (now or dt.datetime.now(dt.timezone.utc)).replace(second=0, microsecond=0)
    fires: list[dt.datetime] = []
    for offset in range(horizon_days * 24 * 60):
        moment = start + dt.timedelta(minutes=offset)
        if moment.minute not in minutes or moment.hour not in hours:
            continue
        if moment.month not in months:
            continue
        day_matches = moment.day in days_of_month
        # cron numbers weekdays with Sunday at 0; datetime numbers Monday at 0.
        weekday_matches = (moment.weekday() + 1) % 7 in days_of_week
        if day_of_month_restricted and day_of_week_restricted:
            if not (day_matches or weekday_matches):
                continue
        elif not (day_matches and weekday_matches):
            continue
        fires.append(moment)
    if len(fires) < 2:
        return float(horizon_days), len(fires)
    gaps = [(later - earlier).total_seconds() / 86400.0 for earlier, later in zip(fires, fires[1:])]
    return max(gaps), len(fires)


def _schedule_crons(text: str) -> tuple[list[str], bool]:
    """The ``cron:`` entries under a top-level ``on.schedule`` key.

    A minimal scanner, not a YAML parser, because this script is stdlib-only on
    purpose: the key is located by indentation and the entries under it are taken
    verbatim. ``declared`` is True for a file that declares ``schedule:`` even
    when no cron could be read from it, so the caller can fail closed on a
    declaration it cannot interpret instead of skipping it as "not scheduled".
    """
    crons: list[str] = []
    declared = False
    schedule_indent: int | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if stripped == "schedule:":
            declared = True
            schedule_indent = indent
            continue
        if schedule_indent is None:
            continue
        if indent <= schedule_indent and not stripped.startswith("-"):
            schedule_indent = None
            continue
        if stripped.startswith("- cron:"):
            value = stripped[len("- cron:") :].strip().strip("'\"")
            if value:
                crons.append(value)
    return crons, declared


def scheduled_workflows(directory: Path) -> list[dict[str, Any]]:
    """Every workflow that declares a schedule, with the cadence its cron implies."""
    workflows: list[dict[str, Any]] = []
    for path in sorted(list(directory.glob("*.yml")) + list(directory.glob("*.yaml"))):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            workflows.append(
                {
                    "file": path.name,
                    "crons": [],
                    "cadence_days": None,
                    "errors": [f"{path} could not be read: {error}"],
                }
            )
            continue
        crons, declared = _schedule_crons(text)
        if not declared:
            continue
        errors: list[str] = []
        cadence: float | None = None
        for expression in crons:
            try:
                value, _fires = cron_cadence_days(expression)
            except ScheduleError as error:
                errors.append(f"cron {expression!r}: {error}")
                continue
            cadence = value if cadence is None else max(cadence, value)
        if not crons:
            errors.append("the file declares a schedule but no cron entry could be read from it")
        workflows.append(
            {"file": path.name, "crons": crons, "cadence_days": cadence, "errors": errors}
        )
    return workflows


def check_scheduled_workflows(
    repo: str,
    token: str,
    workflows: list[dict[str, Any]],
    *,
    grace_days: float = SCHEDULE_GRACE_DAYS,
) -> list[dict[str, Any]]:
    """Did every scheduled workflow fire, on the cadence its own cron implies?

    The check above asks whether one named workflow *succeeded* inside a fixed
    window. This one asks the wider question: every schedule in the repository
    still fires. Only runs with ``event=schedule`` count, so a manual dispatch
    cannot stand in for a schedule that died, and a workflow too young for its
    first scheduled run is told apart from one that has simply never fired by
    reading the workflow's own registration date.
    """
    if not workflows:
        return [
            {
                "name": "the repository declares at least one scheduled workflow",
                "healthy": True,
                "detail": "none found, so there is no schedule to keep alive",
            }
        ]
    results: list[dict[str, Any]] = []
    for workflow in workflows:
        name = workflow["file"]
        crons = ", ".join(workflow["crons"]) or "none readable"
        if workflow["errors"]:
            results.append(
                {
                    "name": f"scheduled workflow {name} declares a readable cadence",
                    "healthy": False,
                    "detail": "; ".join(workflow["errors"]),
                }
            )
            continue
        cadence = float(workflow["cadence_days"] or 0.0)
        window = cadence + grace_days
        runs = _request(
            "GET",
            f"/repos/{repo}/actions/workflows/{name}/runs?event=schedule&per_page=20",
            token,
        )["workflow_runs"]
        newest = max(runs, key=lambda run: run["id"], default=None)
        age = _age_days(newest["created_at"]) if newest else None
        if age is not None and age <= window:
            healthy, detail = True, (
                f"cron {crons} implies a {cadence:.2f}-day cadence; the newest "
                f"scheduled run is {age:.1f} days old ({newest['html_url']})"
            )
        elif age is not None:
            healthy, detail = False, (
                f"cron {crons} implies a {cadence:.2f}-day cadence, but the newest "
                f"scheduled run is {age:.1f} days old (window {window:.2f} days) -- a "
                "schedule that stopped firing writes nothing anywhere"
            )
        else:
            registration = _request("GET", f"/repos/{repo}/actions/workflows/{name}", token)
            registered_age = _age_days(registration.get("created_at"))
            if registered_age is not None and registered_age <= window:
                healthy, detail = True, (
                    f"registered {registered_age:.1f} days ago with cron {crons}; no "
                    "scheduled run yet, and the first is not overdue"
                )
            else:
                healthy, detail = False, (
                    f"no scheduled run of {name} is on record (cron {crons}); either the "
                    "schedule has never fired or the workflow was added less than one "
                    "cadence ago"
                )
        results.append(
            {"name": f"scheduled workflow {name} still fires", "healthy": healthy, "detail": detail}
        )
    return results


def check_watch_runs(repo: str, token: str, workflow: str, max_run_age_days: float) -> dict[str, Any]:
    """Is the schedule alive? A successful run inside the window is the bar."""
    runs = _request(
        "GET",
        f"/repos/{repo}/actions/workflows/{workflow}/runs?per_page=30",
        token,
    )["workflow_runs"]
    successful = [run for run in runs if run.get("conclusion") == "success"]
    newest_success = max(successful, key=lambda run: run["id"], default=None)
    age = _age_days(newest_success["created_at"]) if newest_success else None
    latest = runs[0] if runs else None
    healthy = age is not None and age <= max_run_age_days
    if newest_success is None:
        detail = (
            "no successful run of "
            f"`{workflow}` was found"
            + (f"; the latest run is {latest['status']}/{latest['conclusion']} ({latest['html_url']})" if latest else "")
        )
    else:
        detail = (
            f"last success {newest_success['created_at']} ({age:.1f} days ago, "
            f"{newest_success['html_url']})"
        )
        if not healthy:
            detail += f"; the window is {max_run_age_days:g} days"
        if latest is not None and latest["id"] != newest_success["id"]:
            detail += (
                f"; the latest run is {latest['status']}/{latest['conclusion']} "
                f"({latest['html_url']})"
            )
    return {
        "name": "the weekly watch completed successfully",
        "healthy": healthy,
        "detail": detail,
    }


def _open_drops(repo: str, token: str, branch: str) -> list[dict[str, Any]]:
    """Every open pull request on the machine-owned branch.

    The list endpoint names the head branch but carries no merge state, so the
    merge state is read per pull request in :func:`_merge_state`.
    """
    pulls = _request("GET", f"/repos/{repo}/pulls?state=open&per_page=100", token)
    return [pull for pull in pulls if (pull.get("head") or {}).get("ref") == branch]


def _merge_state(repo: str, token: str, number: int) -> tuple[str, bool | None]:
    """One pull request's merge state, waited for when GitHub is still computing.

    ``mergeable_state`` comes back ``unknown`` while GitHub recomputes it --
    measured on the first live run of this workflow, which read ``unknown`` for a
    drop whose state was ``clean`` a minute earlier. An alarm that skips the week
    it is needed because a field was lazy is the failure mode this whole file
    exists to prevent, so the read is retried a bounded number of times.

    If it stays unknown that is reported as unknown and nothing is raised on it:
    an uncomputable state must never fail a run, and must never be read as clean
    either -- the detail says which it was.
    """
    state, mergeable = "unknown", None
    for attempt in range(MERGE_STATE_ATTEMPTS):
        pull = _request("GET", f"/repos/{repo}/pulls/{number}", token)
        state = pull.get("mergeable_state") or "unknown"
        mergeable = pull.get("mergeable")
        if state != "unknown":
            return state, mergeable
        if attempt + 1 < MERGE_STATE_ATTEMPTS:
            time.sleep(MERGE_STATE_PAUSE_SECONDS)
    return state, mergeable


def _head_commit_age(repo: str, token: str, sha: str) -> float | None:
    commit = _request("GET", f"/repos/{repo}/commits/{sha}", token)
    return _age_days(((commit.get("commit") or {}).get("committer") or {}).get("date"))


def check_drops(
    repo: str,
    token: str,
    branch: str,
    *,
    max_drop_age_days: float,
    max_blocked_hours: float,
    drops: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Can each open drop land, and how long has it been waiting?"""
    if drops is None:
        drops = _open_drops(repo, token, branch)
    if not drops:
        return [
            {
                "name": "an open drop pull request exists",
                "healthy": True,
                "detail": (
                    "none is open, which is not an alert: the watch writes nothing "
                    "when nothing new matches, and check 1 covers a schedule that "
                    "stopped running"
                ),
            }
        ]

    results: list[dict[str, Any]] = []
    for pull in drops:
        head_sha = (pull.get("head") or {}).get("sha") or ""
        checks = _request(
            "GET", f"/repos/{repo}/commits/{head_sha}/check-runs?per_page=100", token
        )
        reported = checks.get("total_count", 0)
        state, mergeable = _merge_state(repo, token, pull["number"])
        blocked = state == "blocked"
        head_age = _head_commit_age(repo, token, head_sha) if head_sha else None
        grace_passed = head_age is None or head_age * 24.0 > max_blocked_hours

        # 2. Stuck with no required context attached: unmergeable for anyone.
        stuck = blocked and reported == 0 and grace_passed
        # 3. Conflicted with main: mergeable only after the next run rebuilds.
        conflicted = state == "dirty" or mergeable is False
        # 4. Waiting on a human decision, past the window the design expects.
        waiting_age = _age_days(pull.get("created_at"))
        waiting = waiting_age is not None and waiting_age > max_drop_age_days

        if stuck:
            detail = (
                f"{state} with no check runs on {head_sha[:7]} "
                f"({reported} reported) and the head is {(head_age or 0):.1f} days old; "
                f"the required contexts never attached -- {pull['html_url']}"
            )
        elif conflicted:
            detail = (
                f"conflicts with the base branch ({state}); the "
                f"next run rebuilds the branch from main -- {pull['html_url']}"
            )
        elif waiting:
            detail = (
                f"open for {waiting_age:.1f} days (window {max_drop_age_days:g}); merge it, "
                "promote a hit into a curated section, or delete the drop -- rule 5 "
                "prunes drops at 30 days, so waiting is not wrong, only unread "
                f"-- {pull['html_url']} ({reported} check(s) reported)"
            )
        else:
            detail = (
                f"#{pull['number']} open {(waiting_age or 0):.1f} days, "
                f"{reported} check(s) reported, merge state {state}"
                + (
                    " (GitHub had not computed it; nothing is raised on an unknown "
                    "state, and the next run reads it again)"
                    if state == "unknown"
                    else ""
                )
            )
        results.append(
            {
                "name": f"drop pull request #{pull['number']} is landable",
                "healthy": not (stuck or conflicted or waiting),
                "detail": detail,
            }
        )
    return results


def _drop_sections(log_text: str) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """The ``## Automated pool drop`` sections of a log, split at the headings.

    Returns ``(sections, unlabelled)``: each labelled section with its body lines,
    and the headings that say "Automated pool drop" without the dated label the
    writer always writes -- which is itself a violation, not something to skip.
    """
    sections: list[tuple[str, list[str]]] = []
    unlabelled: list[str] = []
    heading: str | None = None
    body: list[str] = []

    def flush() -> None:
        if heading is None or not heading.startswith(DROP_HEADING_PREFIX):
            return
        if DROP_HEADING_RE.match(heading):
            sections.append((heading, list(body)))
        else:
            unlabelled.append(heading)

    for line in log_text.splitlines():
        if line.startswith("## "):
            flush()
            heading, body = line, []
            continue
        if heading is not None:
            body.append(line)
    flush()
    return sections, unlabelled


def check_drop_section(heading: str, body: list[str]) -> list[str]:
    """Every way one drop section can break the rules it was written under.

    The rules are the log's own: the drop is labelled unvetted (`README.md`, rule
    7's "a machine-appended drop registers nothing"), it carries no quoted number
    (rule 5: a paper's measurement never transfers without a first-party run), and
    it carries no abstract text -- a triage line, not a finding. The abstract rule
    is enforced structurally: every line must be one of the shapes the writer
    emits, so a pasted sentence has nowhere to stand, and an oversized title or
    matched-terms cell is refused because a title is one line and an abstract is
    neither one line nor short.
    """
    violations: list[str] = []
    if DROP_UNVETTED_MARK not in "\n".join(body):
        violations.append(
            "no unvetted disclosure: the section does not carry the sentence "
            f"{DROP_UNVETTED_MARK!r} (README rules 5 and 7)"
        )
    consumed: set[int] = set()
    for index, line in enumerate(body):
        if index in consumed:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            continue
        if DROP_SUBSECTION_RE.match(stripped):
            continue
        if DROP_MENTION_RE.match(stripped):
            continue
        entry = DROP_ENTRY_RE.match(stripped)
        if entry:
            title, terms = entry.group("title"), entry.group("terms")
            if len(title) > DROP_TITLE_MAX_CHARS:
                violations.append(
                    f"a title is {len(title)} characters (limit {DROP_TITLE_MAX_CHARS}); "
                    f"a drop carries a title, not a paragraph: {_clip(title)}"
                )
            if len(terms) > DROP_TERMS_MAX_CHARS:
                violations.append(
                    f"a matched-terms cell is {len(terms)} characters "
                    f"(limit {DROP_TERMS_MAX_CHARS}): {_clip(terms)}"
                )
            if _DIGIT_RE.search(terms):
                violations.append(
                    "a matched-terms cell carries a number; matched terms are the "
                    f"screen's own words: {_clip(terms)}"
                )
            following = DROP_URL_RE.match(body[index + 1]) if index + 1 < len(body) else None
            if following is None or following.group("number") != entry.group("number"):
                violations.append(
                    f"the entry for arXiv:{entry.group('number')} is not followed by its "
                    "own url line"
                )
            else:
                consumed.add(index + 1)
            continue
        if DROP_URL_RE.match(line):
            violations.append(f"a url line stands without its entry row: {_clip(stripped)}")
            continue
        violations.append(
            "a line is none of the shapes a drop may carry (ids, dates and matched "
            f"terms; no abstract text, no number): {_clip(stripped)}"
        )
    for line in body:
        match = DROP_MEASUREMENT_RE.search(line)
        if match:
            violations.append(
                f"a result-shaped number appears in the drop: {_clip(line)} "
                "(README rule 5: quote a paper's gain as theirs or not at all)"
            )
    return violations


def check_drop_invariants(
    log_text: str, *, source: str, require_drop: bool
) -> dict[str, Any]:
    """Does this copy of the log carry only drops that follow the rules?"""
    sections, unlabelled = _drop_sections(log_text)
    problems: list[str] = []
    for heading in unlabelled:
        problems.append(
            f"a drop heading is not the dated label the writer writes: {_clip(heading)}"
        )
    if require_drop and not sections and not unlabelled:
        problems.append("the machine-owned branch carries no drop section at all")
    for heading, body in sections:
        for violation in check_drop_section(heading, body):
            problems.append(f"{heading}: {violation}")
    if not problems and not sections:
        detail = (
            f"no drop section to check in {source}: the watch writes none when "
            "nothing new matches, so there is nothing for a rule to be broken by"
        )
    elif not problems:
        detail = (
            f"{len(sections)} drop section(s) checked in {source}; every one carries the "
            "dated label, the unvetted disclosure, no quoted number and no text the "
            "drop's shapes do not describe"
        )
    else:
        named = problems[:DROP_VIOLATIONS_NAMED]
        detail = f"{len(problems)} violation(s) in {source}: " + " | ".join(named)
        if len(problems) > len(named):
            detail += f" | ... and {len(problems) - len(named)} more"
    return {
        "name": f"the drop content in {source} follows the log's own rules",
        "healthy": not problems,
        "detail": detail,
    }


def _drop_log_text(repo: str, token: str, sha: str) -> str:
    """The log as a drop's head commit holds it, decoded from the contents API.

    Fail-closed: a read that does not return decodable base64 text raises, and the
    caller reports it as an alert rather than as a pass. A drop nobody could read
    is not a drop that satisfies the rules.
    """
    payload = _request("GET", f"/repos/{repo}/contents/{LOG_REPO_PATH}?ref={sha}", token)
    if not isinstance(payload, Mapping) or payload.get("encoding") != "base64":
        raise GitHubError(
            f"the contents API returned no base64 payload for {LOG_REPO_PATH} at {sha[:7]}"
        )
    content = payload.get("content")
    if not isinstance(content, str) or not content.strip():
        raise GitHubError(f"the contents API returned no content for {LOG_REPO_PATH} at {sha[:7]}")
    return base64.b64decode(content).decode("utf-8")


def check_drop_contents(
    repo: str,
    token: str,
    drops: list[dict[str, Any]],
    *,
    log_path: Path = DEFAULT_LOG_PATH,
) -> list[dict[str, Any]]:
    """The drop's rules, verified on every copy of the log this run can read.

    The merged log (the working tree) and each open drop's own copy (its head
    commit, through the contents API) are checked. The open drop's copy is the one
    that matters most -- it is the text a merge would land -- and it is the copy
    the watch wrote seconds or days earlier, so the writer and this checker are
    pinned against each other by construction instead of by convention.
    """
    results: list[dict[str, Any]] = []
    try:
        merged = log_path.read_text(encoding="utf-8")
    except OSError as error:
        results.append(
            {
                "name": "the merged watch log is readable",
                "healthy": False,
                "detail": f"{log_path} could not be read: {error}",
            }
        )
    else:
        results.append(
            check_drop_invariants(merged, source="the merged log", require_drop=False)
        )
    for pull in drops:
        sha = (pull.get("head") or {}).get("sha") or ""
        source = f"drop pull request #{pull['number']}"
        try:
            text = _drop_log_text(repo, token, sha)
        except (GitHubError, ValueError, UnicodeDecodeError) as error:
            results.append(
                {
                    "name": f"the drop content in {source} is readable",
                    "healthy": False,
                    "detail": f"the log at {sha[:7]} could not be read: {error}",
                }
            )
            continue
        results.append(check_drop_invariants(text, source=source, require_drop=True))
    return results


def _tracking_issue(repo: str, token: str) -> dict[str, Any] | None:
    titles = (ISSUE_TITLE, *LEGACY_ISSUE_TITLES)
    issues = _request("GET", f"/repos/{repo}/issues?state=open&per_page=100", token)
    for issue in issues:
        if issue.get("title") in titles:
            return issue
    return None


def _render(results: list[dict[str, Any]], run_url: str | None) -> str:
    lines = [
        "<!-- opened by .github/workflows/literature-watchdog.yml -->",
        "The literature watchdog found something that needs a person.",
        "This issue is updated by every check and **closes itself** as soon as the",
        "next check is healthy, so it cannot become furniture.",
        "",
        "| check | state | detail |",
        "| --- | --- | --- |",
    ]
    for result in results:
        lines.append(_table_row(result["name"], "ok" if result["healthy"] else "ATTENTION", result["detail"]))
    lines += [
        "",
        "No drop is not a failure: the watch writes nothing when nothing new",
        "matches (`docs/literature/README.md`, rule 5). What must never be silent is",
        "a schedule anywhere in the repository, the required checks reporting, a drop",
        "that can no longer land, and a drop that breaks the rules it was written",
        "under.",
    ]
    if run_url:
        lines += ["", f"Run: {run_url}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--watch-workflow", default="literature-watch.yml")
    parser.add_argument("--branch", default="automation/literature-watch")
    parser.add_argument("--max-run-age-days", type=float, default=8.0)
    parser.add_argument("--max-drop-age-days", type=float, default=7.0)
    parser.add_argument("--max-blocked-hours", type=float, default=24.0)
    parser.add_argument(
        "--schedule-grace-days",
        type=float,
        default=SCHEDULE_GRACE_DAYS,
        help="Days past its own cron cadence a schedule may run late before it is stale",
    )
    parser.add_argument(
        "--workflows-dir",
        default=str(DEFAULT_WORKFLOWS_DIR),
        help="Directory of workflow files whose schedules are checked",
    )
    parser.add_argument(
        "--log-path",
        default=str(DEFAULT_LOG_PATH),
        help="The merged watch log whose drops are verified",
    )
    parser.add_argument("--dry-run", action="store_true", help="report without opening or closing the issue")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"))
    parser.add_argument("--run-url", default=None)
    args = parser.parse_args(argv)

    global DRY_RUN
    DRY_RUN = args.dry_run
    if not args.token:
        print("::error::no token: set GITHUB_TOKEN (or pass --token)")
        return 2

    workflows = scheduled_workflows(Path(args.workflows_dir))
    drops = _open_drops(args.repo, args.token, args.branch)
    results = [
        check_watch_runs(args.repo, args.token, args.watch_workflow, args.max_run_age_days),
        *check_scheduled_workflows(
            args.repo, args.token, workflows, grace_days=args.schedule_grace_days
        ),
        *check_drops(
            args.repo,
            args.token,
            args.branch,
            max_drop_age_days=args.max_drop_age_days,
            max_blocked_hours=args.max_blocked_hours,
            drops=drops,
        ),
        *check_drop_contents(args.repo, args.token, drops, log_path=Path(args.log_path)),
    ]
    alerts = [result for result in results if not result["healthy"]]
    body = _render(results, args.run_url)

    print(f"literature watch watchdog: {len(results) - len(alerts)}/{len(results)} checks ok")
    for result in results:
        print(f"  {'ok  ' if result['healthy'] else 'ALERT'} {result['name']}: {result['detail']}")

    issue = _tracking_issue(args.repo, args.token)
    if alerts:
        if issue is None:
            _request("POST", f"/repos/{args.repo}/issues", args.token, {"title": ISSUE_TITLE, "body": body})
            print(f"opened the tracking issue: {ISSUE_TITLE}")
        else:
            _request("PATCH", f"/repos/{args.repo}/issues/{issue['number']}", args.token, {"body": body})
            print(f"updated the tracking issue #{issue['number']}")
    elif issue is not None:
        _request(
            "POST",
            f"/repos/{args.repo}/issues/{issue['number']}/comments",
            args.token,
            {"body": "Every check is healthy again; closing.\n\n" + body},
        )
        _request("PATCH", f"/repos/{args.repo}/issues/{issue['number']}", args.token, {"state": "closed"})
        print(f"closed the tracking issue #{issue['number']}: all checks healthy")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write("## Literature watch watchdog\n\n" + body + "\n")

    if alerts:
        for result in alerts:
            print(f"::error::{result['name']}: {result['detail']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
