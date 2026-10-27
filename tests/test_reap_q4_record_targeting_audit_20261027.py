"""Tests for the record-targeting audit: no inference, exports only."""

from __future__ import annotations

import json

from chowder_batch.reap_q4_record_targeting_audit_20261027 import (
    NEVER,
    REQUESTED,
    classify_task,
    request_corpus,
    summarize,
)


def task(*calls, assertions):
    """Build a minimal evaluator task with given api_fetch calls and assertions."""
    messages = []
    for url, body in calls:
        arguments = json.dumps({"method": "POST", "url": url, "params": "", "body": body})
        messages.append({"role": "assistant",
                         "tool_calls": [json.dumps({"name": "api_fetch",
                                                    "arguments": arguments})]})
    return {"name": "sales.example", "messages": messages,
            "assertion_results": assertions}


def assertion(kind, params, passed=False, excluded=False):
    return {"type": kind, "params": params, "passed": passed, "excluded": excluded}


def test_corpus_includes_url_and_body_text():
    corpus = request_corpus(task(("sales/notes/001X", '{"text":"hi"}'), assertions=[]))
    assert "001X" in corpus
    assert "hi" in corpus


def test_a_failure_on_a_record_that_was_never_addressed():
    t = task(("sales/notes/001OTHER", '{"text":"hi"}'),
             assertions=[assertion("salesforce_note_exists", {"parent_id": "001TARGET"})])
    row = classify_task(t)[0]
    assert row["class"] == NEVER
    assert row["missing_targets"] == ["parent_id=001TARGET"]


def test_a_failure_on_a_record_that_was_addressed():
    t = task(("sales/notes/001TARGET", '{"text":"hi"}'),
             assertions=[assertion("salesforce_note_body_contains",
                                   {"parent_id": "001TARGET", "body_contains": ["x"]})])
    row = classify_task(t)[0]
    assert row["class"] == REQUESTED
    assert row["missing_targets"] == []


def test_an_id_inside_a_damaged_payload_still_counts_as_addressed():
    # The model did try to target the record; the JSON broke on the way out.
    messages = [{"role": "assistant", "tool_calls": [json.dumps({
        "name": "api_fetch",
        "arguments": '{"method":"POST","url":"sales/notes/001TARGET","body":"{"text":'})]}]
    t = {"name": "sales.example", "messages": messages,
         "assertion_results": [assertion("salesforce_note_exists", {"parent_id": "001TARGET"})]}
    assert classify_task(t)[0]["class"] == REQUESTED


def test_excluded_assertions_are_not_counted():
    t = task(("sales/notes/001OTHER", "{}"),
             assertions=[assertion("salesforce_note_exists", {"parent_id": "001TARGET"},
                                   excluded=True)])
    assert classify_task(t) == []


def test_assertions_without_a_target_are_skipped():
    t = task(("sales/notes/001OTHER", "{}"),
             assertions=[assertion("gmail_message_is_read", {"label": "INBOX"})])
    assert classify_task(t) == []


def test_summary_counts_both_classes_and_ignores_passes():
    rows = [{**r, "passed": passed} for r, passed in (
        (classify_task(task(("x", "{}"), assertions=[
            assertion("salesforce_note_exists", {"parent_id": "001TARGET"})]))[0], False),
        (classify_task(task(("001TARGET", "{}"), assertions=[
            assertion("salesforce_note_exists", {"parent_id": "001TARGET"})]))[0], False),
        (classify_task(task(("001TARGET", "{}"), assertions=[
            assertion("salesforce_note_exists", {"parent_id": "001TARGET"})]))[0], True),
    )]
    summary = summarize(rows)
    assert summary["failed_with_target"] == 2
    assert summary["failed_target_never_requested"] == 1
    assert summary["failed_target_requested"] == 1
