"""The evals package and CLI surface.

Adapters normalize external harness output honestly (UNKNOWN stays UNKNOWN,
unsupported modality is N/A not zero); the scoreboard renders raw-vs-harness
and contamination marks; the CLI handlers stay read-mostly with one mutating
data path that registers as QUARANTINE.
"""

from __future__ import annotations

import json
import sys

import pytest

from chowder.evals.adapters import ADAPTERS, AdapterUnavailable, LMEvalAdapter, unavailable_run
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


# ---------------- scoreboard polarity (reported-metric audit R1) ----------------


def _lower_is_better_entry():
    """One benchmark whose declared metric improves as its raw value falls.

    ``default_registry()`` refuses a row that redeclares a metric name's
    polarity, so a hand-built entry plus a hand-built registry is the only way to
    put the registry's own ``lower_is_better`` vocabulary in front of the
    renderer -- which is exactly what the vocabulary declares and no shipped row
    exercises yet.
    """
    from chowder.growth.benchmark_registry import (
        BenchmarkEntry,
        BenchmarkRegistry,
        Normalization,
    )

    entry = BenchmarkEntry(
        benchmark_id="latency_probe",
        version="2026-01",
        name="Latency probe",
        category="reasoning",
        subcategory="serving latency",
        status="RUNNABLE_PUBLIC",
        lifecycle="ACTIVE_DIAGNOSTIC",
        tier=2,
        scorer="exact_match",
        primary_metric="token_latency_ms",
        direction="lower_is_better",
        normalization=Normalization(kind="identity"),
        skills=("reasoning.abstract",),
        dataset_source="internal",
        implementation_source="internal",
        source="internal",
        license="internal",
        release_date="2026-01-01",
        adapter="chowder_custom",
    )
    return BenchmarkRegistry((entry,))


LATENCY = "latency_probe@2026-01"


def _delta_section(markdown: str) -> str:
    assert "## vs previous generation" in markdown, markdown
    return markdown.split("## vs previous generation", 1)[1]


def test_scoreboard_arrows_follow_the_declared_lower_is_better_polarity():
    # Raw value rises (5.0 -> 9.0): worse for a lower-is-better metric, so the
    # arrow must point down even though ``compare`` calls a rise "improved".
    parent = _report(
        [_run(LATENCY, 5.0, SUPPORTED, per_sample_scores=(5.0, 5.0, 5.0, 5.0))]
    )
    candidate = _report(
        [_run(LATENCY, 9.0, SUPPORTED, per_sample_scores=(9.0, 9.0, 9.0, 9.0))]
    )

    section = _delta_section(Scoreboard(_lower_is_better_entry()).render(candidate, parent_report=parent))

    row = next(line for line in section.splitlines() if line.startswith(f"| {LATENCY}"))
    assert "| +4.000 |" in row, row
    assert "↓" in row and "↑" not in row, row
    assert "(lower is better)" in row, row


def test_scoreboard_arrows_are_unchanged_for_the_shipped_higher_is_better_rows():
    parent = _report(
        [_run(LIVECODE, 0.40, SUPPORTED, per_sample_scores=(0.4, 0.4, 0.4, 0.4))]
    )
    candidate = _report(
        [_run(LIVECODE, 0.46, SUPPORTED, per_sample_scores=(0.46, 0.46, 0.46, 0.46))]
    )

    section = _delta_section(Scoreboard(default_registry()).render(candidate, parent_report=parent))

    row = next(line for line in section.splitlines() if line.startswith(f"| {LIVECODE}"))
    assert "| +0.060 |" in row, row
    assert "↑" in row and "↓" not in row, row
    assert "lower is better" not in row, row


# --------------------------------------------------------------------------
# the reported-metric audit's decisions (R4-R6)
# --------------------------------------------------------------------------


class _FakeLMEval:
    """The one harness call the lm-eval adapter makes, with a chosen table."""

    def __init__(self, task_results: dict) -> None:
        self._task_results = task_results

    def simple_validate(self, **kwargs):
        return {"results": {kwargs["tasks"][0]: self._task_results}}


def _lm_eval_row(monkeypatch, task_results: dict):
    monkeypatch.setitem(sys.modules, "lm_eval", _FakeLMEval(task_results))
    return LMEvalAdapter().run("gsm8k@2024-06", "gen2", task="gsm8k", model="hf")


def test_an_lm_eval_row_uses_the_declared_alias_and_keeps_the_harness_key(monkeypatch):
    """R4: the two metric vocabularies meet in one declared table, never a guess.

    The registry names the quantity ("accuracy") and the harness names its own
    number ("acc"); before this table nothing related them, so a correctly
    measured lm-eval row could not bind at all.
    """
    row = _lm_eval_row(monkeypatch, {"acc,none": 0.5, "alias": "gsm8k"})

    assert row.metric == "accuracy"
    assert row.metadata["harness_metric"] == "acc", "the harness's own key is kept"
    assert row.score == 0.5


def test_an_undeclared_harness_key_is_not_guessed(monkeypatch):
    """Only declared keys are relabelled; the binder stays the loud refusal."""
    row = _lm_eval_row(monkeypatch, {"bleu,none": 0.25})

    assert row.metric == "bleu"
    assert row.metadata["harness_metric"] == "bleu"


def test_a_harness_row_with_no_numeric_metric_names_none(monkeypatch):
    """No numeric key means no measurement, so the row claims no metric either."""
    row = _lm_eval_row(monkeypatch, {"alias": "gsm8k"})

    assert row.metric == ""
    assert row.score is None


def test_a_report_row_that_declares_no_metric_reads_back_undeclared(tmp_path):
    """R5: the load default turned an absent name into ``"accuracy"``."""
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "generation_version": "v1",
                "runs": [
                    {
                        "benchmark_qualified_id": LIVECODE,
                        "adapter": "inspect",
                        "generation_version": "v1",
                        "score": 0.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert EvalReport.load(path).runs[0].metric == ""
    assert _run(LIVECODE, 0.5, SUPPORTED).metric == "", "the constructor default too"


def test_the_settle_payload_names_the_settled_cost(tmp_path):
    """R6: the key said "actual" for a device figure that may be unmeasured.

    ``device_measured`` travels inside the settled object, so the disclosure was
    there; the key's wording was not. It now names the operation the command
    performed, which is the wording amendment 18 applied to the judge's T13 row.
    """
    import argparse
    import io
    from contextlib import redirect_stdout

    import test_growth_campaign_runner as campaign_fixture
    from chowder.growth.cli import _growth_campaign_settle

    campaign_fixture._campaign(tmp_path)
    accounting = tmp_path / "cycle_compute_accounting.json"
    accounting.write_text(
        json.dumps(
            {
                "totals": {
                    "incremental": {
                        "device_gpu_hours": 0.4,
                        "wall_gpu_hours": 1.1,
                        "device_measured": False,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        _growth_campaign_settle(
            argparse.Namespace(
                manifest=str(tmp_path / "inputs" / "campaign.json"),
                accounting=str(accounting),
            )
        )

    payload = json.loads(buffer.getvalue())
    assert "settled" in payload, payload
    assert "actual" not in payload, payload
    assert payload["settled"]["device_measured"] is False
