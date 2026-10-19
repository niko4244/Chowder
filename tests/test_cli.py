from __future__ import annotations

import json
from types import SimpleNamespace

from chowder import cli


def _outcome(*, succeeded: bool, candidates=(), terminal_state=None, promoted=None):
    return SimpleNamespace(
        project=SimpleNamespace(name="test-project"),
        succeeded=succeeded,
        promoted_experiment_id=promoted,
        generation=SimpleNamespace(
            candidates=tuple(candidates),
            goal_terminal_state=terminal_state,
            goal_assessment=None,
        ),
    )


def test_train_handles_successful_parent_without_candidates(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "run_project",
        lambda *args, **kwargs: _outcome(
            succeeded=True,
            terminal_state="STOP_GOALS_MET",
        ),
    )

    exit_code = cli._train(SimpleNamespace(project="project.json"))

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["succeeded"] is True
    assert payload["terminal_state"] == "STOP_GOALS_MET"
    assert payload["experiment_id"] is None


def test_train_uses_project_outcome_not_candidate_success(monkeypatch, capsys):
    candidate = SimpleNamespace(
        experiment_id="candidate-1",
        succeeded=True,
        artifact=None,
        result=None,
        error=None,
    )
    monkeypatch.setattr(
        cli,
        "run_project",
        lambda *args, **kwargs: _outcome(
            succeeded=False,
            candidates=(candidate,),
            promoted="candidate-1",
        ),
    )

    exit_code = cli._train(SimpleNamespace(project="project.json"))

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["succeeded"] is False
    assert payload["promoted_experiment_id"] == "candidate-1"


def test_train_reports_lifecycle_refusal_as_non_success(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "run_project",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("objective refused")),
    )

    exit_code = cli._train(SimpleNamespace(project="project.json"))

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["succeeded"] is False
    assert "objective refused" in payload["error"]
