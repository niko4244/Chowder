#!/usr/bin/env python3
"""The literature watch's own watchdog: is the weekly loop still alive?

`watch.py` fetches and screens; the schedule drops what is new and opens a
reviewable pull request. Both of those can stop without anything turning red --
a schedule that never fires, a drop whose required checks never report, a drop
nobody ever merges -- and a silently broken schedule looks exactly like a quiet
week. This script is the part that tells the two apart, and it is deliberately
the only part that can shout.

What it checks, in the order an operator would ask:

1. **Did the watch run?** A successful run of the watch workflow within
   ``--max-run-age-days``. A schedule that has not completed successfully is the
   alarm the rest of this file exists to raise, because *no drop* is a legitimate
   outcome (the watch writes nothing when nothing new matches, README rule 5) and
   therefore cannot be an alarm by itself.
2. **Can the drop land?** An open drop whose required checks never reported is
   stuck no matter who looks at it: branch protection cannot settle on an empty
   check rollup. That is the failure mode `literature-watch.yml` approves its own
   parked run to avoid, and this is the check that notices if it comes back.
3. **Is the drop conflicted?** A drop that conflicts with `main` cannot be merged
   until the next run rebuilds the machine-owned branch, so it is reported rather
   than left invisible.
4. **Has the drop been waiting too long?** An open drop older than
   ``--max-drop-age-days`` is waiting on a human decision. The design tolerates
   that for the 30-day prune horizon (README rule 5), so this is a nudge that
   says how long and what the two ways out are -- not a claim that something is
   broken.

Alerts are loud twice over: the run exits non-zero, and one tracking issue is
opened or updated (and closed again the moment a check is healthy, so it cannot
become furniture). Nothing here touches the log, the branch or the drop: this
script reads, reports, and opens one issue.

Usage:
    GITHUB_TOKEN=... python docs/literature/watchdog.py \
        --repo owner/name --watch-workflow literature-watch.yml \
        --branch automation/literature-watch --max-run-age-days 8 \
        --max-drop-age-days 7 --max-blocked-hours 24
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

API = "https://api.github.com"
ISSUE_TITLE = "[watchdog] the literature watch needs attention"
DRY_RUN = False

#: How long to wait between reads of a merge state GitHub has not computed yet,
#: and how many times to read it. Kept as module constants so a test can set the
#: pause to zero without touching the code under test.
MERGE_STATE_PAUSE_SECONDS = 2.0
MERGE_STATE_ATTEMPTS = 5


class GitHubError(RuntimeError):
    """A GitHub API call failed; the message carries the status and body."""


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
    if not timestamp:
        return None
    parsed = dt.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=dt.timezone.utc
    )
    return (dt.datetime.now(dt.timezone.utc) - parsed).total_seconds() / 86400.0


def _table_row(name: str, state: str, detail: str) -> str:
    return f"| {name} | {state} | {detail} |"


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
) -> list[dict[str, Any]]:
    """Can each open drop land, and how long has it been waiting?"""
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


def _tracking_issue(repo: str, token: str) -> dict[str, Any] | None:
    issues = _request("GET", f"/repos/{repo}/issues?state=open&per_page=100", token)
    for issue in issues:
        if issue.get("title") == ISSUE_TITLE:
            return issue
    return None


def _render(results: list[dict[str, Any]], run_url: str | None) -> str:
    lines = [
        "<!-- opened by .github/workflows/literature-watchdog.yml -->",
        "The literature watch's own watchdog found something that needs a person.",
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
        "matches (`docs/literature/README.md`, rule 5). What must never be silent",
        "is the schedule, the required checks reporting, and a drop that can no",
        "longer land.",
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
    parser.add_argument("--dry-run", action="store_true", help="report without opening or closing the issue")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"))
    parser.add_argument("--run-url", default=None)
    args = parser.parse_args(argv)

    global DRY_RUN
    DRY_RUN = args.dry_run
    if not args.token:
        print("::error::no token: set GITHUB_TOKEN (or pass --token)")
        return 2

    results = [
        check_watch_runs(args.repo, args.token, args.watch_workflow, args.max_run_age_days),
        *check_drops(
            args.repo,
            args.token,
            args.branch,
            max_drop_age_days=args.max_drop_age_days,
            max_blocked_hours=args.max_blocked_hours,
        ),
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
