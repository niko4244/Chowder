"""The evals package and CLI surface.

Adapters normalize external harness output honestly (UNKNOWN stays UNKNOWN,
unsupported modality is N/A not zero); the scoreboard renders raw-vs-harness
and contamination marks; the CLI handlers stay read-mostly with one mutating
data path that registers as QUARANTINE.
"""

from __future__ import annotations

import json

import pytest

from chowder.evals.adapters import ADAPTERS, AdapterUnavailable, unavailable_run
from chowder.evals.result import (
    NOT_APPLICABLE_MODALITY,
    SUPPORTED,
    UNSUPPORTED_HARNESS,
    EvalReport,
)
from chowder.evals.runner import UNKNOWN_MARK, Scoreboard, category_aggregates
from chowder.growth.catalog import default_registry


LIVECODE = "livecodebench@2025-04"


def _run(benchmark: str, score: float | None, support: str, **extra):
    from chowder.evals.result import BenchmarkRun

    base = dict(
        benchmark_qualified_id=benchmark,
        adapter="inspect",
        generation_version="v1.0",
        score=score,
        support=support,
    )
    base.update(extra)
    return BenchmarkRun(**base)


def _report(runs) -> EvalReport:
    return EvalReport(generation_version="v1.0", runs=tuple(runs), date="2026-09-15")


# ---------------- adapters ----------------


def test_adapter_registry_covers_the_four_harness_families():
    names = set(ADAPTERS)
    assert {"inspect", "lm_eval", "native_agent", "chowder_custom"} <= names


def test_missing_harness_raises_adapter_unavailable_not_a_fake_score():
    adapter = ADAPTERS["inspect"]()
    with pytest.raises(AdapterUnavailable):
        adapter.run(
            "livecodebench@2025-04",
            "v1.0",
            model="chowder-9b",
            task="livecodebench",
        )


def test_unavailable_harness_normalizes_to_honest_non_measurement():
    run = unavailable_run(
        "livecodebench@2025-04", "v1.0", "inspect", reason="inspect not installed"
    )
    assert run.support == UNSUPPORTED_HARNESS
    assert run.score is None


# ---------------- scoreboard ----------------


def test_scoreboard_marks_na_and_unknown_distinctly():
    runs = [
        _run(LIVECODE, 0.42, SUPPORTED, measurement_kind="raw_model", n_samples=100),
        _run("osworld@2.0", None, NOT_APPLICABLE_MODALITY),
        _run("frontiermath@2025-04", None, UNSUPPORTED_HARNESS),
    ]
    scoreboard = Scoreboard(default_registry())
    markdown = scoreboard.render(_report(runs))
    assert "N/A" in markdown or "not-applicable" in markdown.lower()
    assert UNKNOWN_MARK in markdown
    assert "0.42" in markdown


def test_tainted_benchmark_is_marked_not_hidden():
    runs = [_run(LIVECODE, 0.42, SUPPORTED, n_samples=100)]
    scoreboard = Scoreboard(default_registry())
    markdown = scoreboard.render(
        _report(runs),
        contamination={"benchmarks": {LIVECODE: {"status": "KNOWN_CONTAMINATION"}}},
    )
    assert "KNOWN_CONTAMINATION" in markdown.upper()


def test_category_aggregates_exclude_na_and_unknown():
    runs = [
        _run(LIVECODE, 0.42, SUPPORTED),
        _run("osworld@2.0", None, NOT_APPLICABLE_MODALITY),
        _run("frontiermath@2025-04", None, UNSUPPORTED_HARNESS),
    ]
    aggregates = category_aggregates(
        _report(runs), {"livecodebench@2025-04": "coding"}
    )
    # Only the supported run enters the aggregate; N/A and unknown are
    # excluded, not zeroed.
    assert aggregates == {"coding": 0.42}


def test_eval_report_roundtrips_through_disk(tmp_path):
    report = _report(
        [
            _run(LIVECODE, 0.42, SUPPORTED, n_samples=100, per_sample_scores=(1.0, 0.0)),
            _run("osworld@2.0", None, NOT_APPLICABLE_MODALITY),
        ]
    )
    path = tmp_path / "eval-report.json"
    report.save(path)
    loaded = EvalReport.load(path)
    assert loaded.generation_version == report.generation_version
    assert len(loaded.runs) == 2
    supported = loaded.supported_runs()
    assert len(supported) == 1
    assert supported[0].score == 0.42


# ---------------- CLI ----------------


def _main(cwd, *argv):
    from chowder.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(list(argv))
    return args.func(args)


def test_cli_eval_catalog_lists_pinned_benchmarks(capsys):
    exit_code = _main(None, "eval", "catalog")
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] >= 40
    assert payload["runnable"] > 0
    row = next(
        r for r in payload["benchmarks"] if r["benchmark"].startswith("livecodebench@")
    )
    assert row["status"] == "RUNNABLE_PUBLIC"
    assert row["split_policy"] in {"protected", "dev", "none"}


def test_cli_data_audit_reports_seed_registry(tmp_path, capsys):
    exit_code = _main(None, "data", "audit", "--root", str(tmp_path))
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_sources"] > 0
    # Seed sources are not trainable until explicitly admitted.
    assert payload["trainable_sources"] == []


def test_cli_data_register_enters_quarantine(tmp_path, capsys):
    exit_code = _main(
        None,
        "data",
        "register",
        "--root",
        str(tmp_path),
        "--source-id",
        "cli-sample",
        "--name",
        "CLI Sample",
        "--origin",
        "huggingface",
        "--url",
        "https://example/ds",
        "--revision",
        "v1.0",
        "--license",
        "MIT",
        "--domain",
        "reasoning",
        "--language",
        "en",
        "--source-type",
        "synthetic",
        "--examples",
        "100",
        "--tokens",
        "50000",
        "--quality",
        "0.8",
        "--reason",
        "smoke test",
        "--pii-reviewed",
        "--secrets-reviewed",
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["trust_class"] == "QUARANTINE"
    assert payload["trainable"] is False

    audit = json.loads(
        capsys.readouterr().out or "{}"
    )  # no output yet; re-run audit below
    exit_code = _main(None, "data", "audit", "--root", str(tmp_path))
    audit_payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert "cli-sample" in [s["source_id"] for s in audit_payload["sources"]]
    assert audit_payload["trainable_sources"] == []


def test_cli_data_register_refuses_latest_revision(tmp_path, capsys):
    exit_code = _main(
        None,
        "data",
        "register",
        "--root",
        str(tmp_path),
        "--source-id",
        "bad-pin",
        "--name",
        "Bad Pin",
        "--origin",
        "huggingface",
        "--url",
        "https://example/ds",
        "--revision",
        "latest",
        "--license",
        "MIT",
        "--domain",
        "reasoning",
        "--language",
        "en",
        "--source-type",
        "synthetic",
        "--examples",
        "100",
        "--tokens",
        "50000",
        "--quality",
        "0.8",
        "--reason",
        "should fail",
        "--pii-reviewed",
        "--secrets-reviewed",
    )
    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert "latest" in payload["error"]


def test_cli_data_contamination_text_probe(capsys):
    exit_code = _main(None, "data", "contamination", "--text", "an ordinary sentence")
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["clean"] is True
    assert payload["verdict"] == "CLEAN"


def test_cli_growth_curriculum_reports_unknown_skills_cleanly(tmp_path, capsys):
    profile = {
        "model_version": "v0.1",
        "raw_scores": {},
        "skills": [
            {"skill": "not.a.skill", "estimate": 0.5, "confidence": 0.9, "evidence": []}
        ],
        "unsupported": [],
        "tainted": [],
        "notes": {},
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    exit_code = _main(None, "growth", "curriculum", str(path))
    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["unknown_skills"] == ["not.a.skill"]


def test_cli_growth_curriculum_builds_plan_from_profile(tmp_path, capsys):
    profile = {
        "model_version": "v0.1",
        "raw_scores": {"gpqa_diamond@2025-05-30": 0.31},
        "skills": [
            {
                "skill": "coding.generation",
                "estimate": 0.28,
                "confidence": 0.9,
                "evidence": ["livecodebench@2025-04"],
            }
        ],
        "unsupported": [],
        "tainted": [],
        "notes": {},
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    exit_code = _main(
        None, "growth", "curriculum", str(path), "--protected", "gpqa_diamond@2025-05-30"
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["items"]
    item = payload["items"][0]
    assert item["skill"] == "coding.generation"
    assert item["protected_regression_set"] == ["gpqa_diamond@2025-05-30"]
    assert item["decision_trace"]["components"]


def test_cli_growth_status_summarizes_ledger(tmp_path, capsys):
    outcome = {
        "cycle_id": "cycle-001",
        "parent_version": "v0.1",
        "candidate_version": "v0.2",
        "verdict": "PROMOTED",
        "phases": [{"phase": "promotion", "detail": {}}],
        "promotion": {
            "reasons": ["all predeclared promotion checks passed"],
            "checks": {"target_improvement": "met", "protected_regression": "ok"},
        },
    }
    path = tmp_path / "outcome.json"
    path.write_text(json.dumps(outcome), encoding="utf-8")
    exit_code = _main(None, "growth", "status", str(path))
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "PROMOTED"
    assert payload["checks"]["target_improvement"] == "met"
