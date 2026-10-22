"""The frozen REAP-MoE judge, tested before the run it will judge.

A judge that always refuses is as useless as one that always certifies, so
every gating row is pinned in both directions with synthetic run roots: a clean
improving candidate promotes, and each defect that must refuse (a target
regression, a score its own items do not produce, a missing raw file, a
termination drop, a settlement overrun) gets its own root.

The epistemics rule is inherited from the gen2 judge: an artifact that is
missing or unreadable is UNKNOWN, never a silent pass, and any UNKNOWN makes
the verdict INCONCLUSIVE rather than PROMOTED. The bar a candidate must clear
is the *campaign's* ``min_delta`` — one test moves only that declaration and
shows the verdict follow it, because a hard-coded threshold would be an
invented standard.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from chowder.evals.result import (
    MEASURED_PARENT,
    MEASURED_THIS_GENERATION,
    BenchmarkRun,
    EvalReport,
)
from chowder.growth.campaign import (
    CampaignManifest,
    TargetThreshold,
    settle_campaign,
)
from chowder.growth.compute_cost import ComputeCost, CycleCostLedger, ledger_digest
from chowder.growth.training_binding import directory_digest

JUDGE_PATH = (
    Path(__file__).resolve().parent.parent / "docs" / "reap_moe" / "judge_reap_moe.py"
)
_spec = importlib.util.spec_from_file_location("judge_reap_moe", JUDGE_PATH)
judge = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(judge)

TARGET = "tool-use@reap-moe-v1"
MATH = "math500@2024-04"
MGSM = "mgsm@2022-11"
N_ITEMS = 20
DECODING = {"temperature": 0.0, "do_sample": False, "max_new_tokens": 32}

#: What a rendered verdict row's status looks like: the status field is padded
#: to eight characters, so it is always followed by at least two spaces.
_STATUS = re.compile(r"\s+(PASS|FAIL|UNKNOWN|INFO)\s{2,}")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        },
    }
]


# --------------------------------------------------------------------------
# evidence fixtures
# --------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tool_items(*, correct: int, eos_misses: int = 0) -> list[dict[str, Any]]:
    """``correct`` of N items make the right call; the rest make a wrong one."""
    items = []
    for index in range(N_ITEMS):
        expected = {"name": "lookup", "arguments": {"q": f"q{index}"}}
        call = expected if index < correct else {"name": "lookup", "arguments": {"q": "wrong"}}
        items.append(
            {
                "uuid": f"tool-{index}",
                "prompt": f"prompt {index}",
                "tools": TOOLS,
                "expected": expected,
                "prediction": f"<tool_call>{json.dumps(call)}</tool_call>",
                "score": 1.0 if index < correct else 0.0,
                "generated_tokens": 8,
                "eos_terminated": index >= eos_misses,
            }
        )
    return items


def _math_items(*, correct: int, prefix: str) -> list[dict[str, Any]]:
    items = []
    for index in range(N_ITEMS):
        answer = str(index)
        items.append(
            {
                "uuid": f"{prefix}-{index}",
                "prompt": f"{prefix} prompt {index}",
                "expected": answer,
                "prediction": f"<think>work</think>\n{answer if index < correct else 'wrong'}",
                "score": 1.0 if index < correct else 0.0,
                "generated_tokens": 8,
                "eos_terminated": True,
            }
        )
    return items


def _write_jsonl(path: Path, items: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in items) + "\n",
        encoding="utf-8",
    )


def _run(
    benchmark: str,
    version: str,
    origin: str,
    items: Sequence[Mapping[str, Any]],
    reference: str,
    digest: str,
    *,
    score_override: float | None = None,
) -> BenchmarkRun:
    samples = tuple(float(item["score"]) for item in items)
    return BenchmarkRun(
        benchmark_qualified_id=benchmark,
        adapter="transformers_text",
        generation_version=version,
        score=sum(samples) / len(samples) if score_override is None else score_override,
        n_samples=len(samples),
        per_sample_scores=samples,
        metric="accuracy",
        measurement_origin=origin,
        raw_artifact_ref=reference,
        metadata={"artifact_sha256": digest, "sample_indices": list(range(len(samples)))},
    )


# --------------------------------------------------------------------------
# run-root builder
# --------------------------------------------------------------------------


def _campaign_document(
    root: Path, *, min_delta: float, ceilings: Mapping[str, float]
) -> dict[str, Any]:
    dataset = root / "material" / "items.jsonl"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text("", encoding="utf-8")
    material = root / "material" / "evaluation.json"
    material.write_text(
        json.dumps(
            {
                "suites": [
                    {
                        "benchmark_qualified_id": TARGET,
                        "name": "tool-use",
                        "dataset": str(dataset),
                        "scoring": "tool_call_exact",
                    },
                    {
                        "benchmark_qualified_id": MATH,
                        "name": "math500",
                        "dataset": str(dataset),
                        "scoring": "final_number_match",
                    },
                    {
                        "benchmark_qualified_id": MGSM,
                        "name": "mgsm",
                        "dataset": str(dataset),
                        "scoring": "final_number_match",
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "cycle_id": "gen1-reap-moe-tool-use",
        "parent_version": "gen0",
        "base_model_path": str(root / "base"),
        "base_model_digest": "b" * 64,
        "state_root": str(root / "state"),
        "target_benchmarks": [TARGET],
        "protected_benchmarks": [MATH, MGSM],
        "broad_benchmarks": [MATH, MGSM],
        "calibration_benchmarks": [],
        "reliability_benchmarks": [],
        "budget": {
            "device_gpu_hours_ceiling_per_recipe": ceilings["per_recipe"],
            "wall_gpu_hours_ceiling_per_recipe": ceilings["per_recipe"],
            "device_gpu_hours_ceiling_campaign": ceilings["campaign"],
            "wall_gpu_hours_ceiling_campaign": ceilings["campaign"],
        },
        "recipes": ["recipe-00", "recipe-01"],
        "candidate_selection_policy": "first_successful",
        "promotion_policy_version": "promotion-policy-v2-provenance-settlement",
        "protection": {
            "trusted_ancestor_version": "gen0",
            "slice_regression_max": 0.0625,
            "n_samples": N_ITEMS,
            "seed": 1234,
            "shuffle": False,
            "decoding": dict(DECODING),
            "prompt_policy": "chat_template",
        },
        "retention_profile": {
            "profile_id": "moe-gen1-protection",
            "constraints": [
                {"dimension": "math500", "kind": "max-regression", "value": -0.0625, "benchmark": MATH},
                {"dimension": "mgsm", "kind": "max-regression", "value": -0.0625, "benchmark": MGSM},
            ],
        },
        "target_threshold": {"benchmark": TARGET, "min_delta": min_delta},
        "evaluation_material_path": str(material),
    }


def _run_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidate_tool_correct: int = 14,
    parent_tool_correct: int = 4,
    candidate_tool_score: float | None = None,
    candidate_eos_misses: int = 0,
    parent_eos_misses: int = 0,
    candidate_math_correct: int = 16,
    parent_math_correct: int = 16,
    min_delta: float = 0.05,
    wall_gpu_hours: float = 1.0,
    record_verdict: str = "PROMOTED",
    record_reasons: Sequence[str] = (),
    with_campaign: bool = True,
    drop_raw: str | None = None,
    descriptive: Mapping[str, Any] | None = None,
) -> Path:
    root = tmp_path / "run"
    root.mkdir(parents=True, exist_ok=True)

    artifact = root / "candidate-adapter"
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    digest, _entries = directory_digest(artifact)
    (root / "chosen_candidate.json").write_text(
        json.dumps({"artifact_ref": str(artifact), "artifact_sha256": digest}),
        encoding="utf-8",
    )

    limits = {"per_recipe": 10.0, "campaign": 10.0}
    document = _campaign_document(root, min_delta=min_delta, ceilings=limits)
    campaign_path = root / "reap_moe_campaign.json"
    campaign_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    monkeypatch.setattr(
        judge, "CAMPAIGN_MANIFEST", campaign_path if with_campaign else root / "absent.json"
    )

    def arm(
        role: str,
        version: str,
        origin: str,
        tool_correct: int,
        math_correct: int,
        *,
        eos_misses: int = 0,
        identity: Mapping[str, str] | None = None,
        tool_score: float | None = None,
    ) -> tuple[EvalReport, dict[str, str]]:
        runs: list[BenchmarkRun] = []
        digests: dict[str, str] = {}
        for benchmark, items in (
            (TARGET, _tool_items(correct=tool_correct, eos_misses=eos_misses)),
            (MATH, _math_items(correct=math_correct, prefix="math")),
            (MGSM, _math_items(correct=math_correct, prefix="mgsm")),
        ):
            reference = f"raw/{role}-{benchmark.replace('@', '-')}.jsonl"
            if drop_raw == f"{role}-{benchmark.split('@')[0]}":
                continue
            path = root / reference
            _write_jsonl(path, items)
            sha = _sha256(path)
            digests[benchmark] = sha
            runs.append(
                _run(
                    benchmark,
                    version,
                    origin,
                    items,
                    reference,
                    sha,
                    score_override=tool_score if benchmark == TARGET else None,
                )
            )
        report = EvalReport(
            generation_version=version,
            runs=tuple(runs),
            model_identity=dict(identity or {}),
        )
        return report, digests

    candidate_report, _ = arm(
        "candidate",
        "gen1",
        MEASURED_THIS_GENERATION,
        candidate_tool_correct,
        candidate_math_correct,
        eos_misses=candidate_eos_misses,
        identity={"adapter_digest": digest},
        tool_score=candidate_tool_score,
    )
    candidate_report.save(root / "candidate_evaluation.json")
    parent_report, _ = arm(
        "parent",
        "gen0",
        MEASURED_PARENT,
        parent_tool_correct,
        parent_math_correct,
        eos_misses=parent_eos_misses,
        identity={"adapter_digest": "p" * 64},
    )
    parent_report.save(root / "parent_evaluation.json")

    ledger = CycleCostLedger(cycle_id=document["cycle_id"])
    for index in range(2):
        ledger.add(
            f"train-{index}", "training",
            ComputeCost(device_gpu_hours=0.25, wall_gpu_hours=wall_gpu_hours / 2, source="test"),
            recipe_id=f"recipe-0{index}",
        )
    accounting = ledger.render()
    totals = accounting["totals"]["incremental"]
    (root / "cycle_compute_accounting.json").write_text(
        json.dumps(accounting, indent=2), encoding="utf-8"
    )
    manifest = CampaignManifest.from_mapping(document, source="<judge-test>")
    settlement = settle_campaign(manifest, total=ComputeCost.from_dict(totals))
    record = {
        "cycle_id": document["cycle_id"],
        "parent_version": "gen0",
        "candidate_version": "gen1",
        "verdict": record_verdict,
        "cost": {"accounting_digest": ledger_digest(accounting)},
        "settlement": settlement.to_dict(),
        "promotion": {
            "decision": {"verdict": record_verdict, "reasons": list(record_reasons)}
        },
    }
    (root / "campaign-run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    if descriptive is not None:
        reports = root / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        (reports / "verdict.json").write_text(json.dumps(descriptive), encoding="utf-8")
    return root


def _rows(output: str) -> dict[str, list[str]]:
    """threshold id -> the verdict statuses rendered for it."""
    found: dict[str, list[str]] = {}
    for line in output.splitlines():
        parts = line.split()
        if not parts or parts[0] not in {
            "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10",
        }:
            continue
        status = _STATUS.search(line)
        if status is not None:
            found.setdefault(parts[0], []).append(status.group(1))
    return found


def _judge(root: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = judge.main(["judge", str(root)])
    return code, capsys.readouterr().out


# --------------------------------------------------------------------------
# the clean path
# --------------------------------------------------------------------------


def test_clean_improving_candidate_promotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _run_root(tmp_path, monkeypatch)
    code, output = _judge(root, capsys)
    rows = _rows(output)
    assert code == 0, output
    assert "VERDICT: PROMOTED" in output
    assert set(rows) == {f"R{index}" for index in range(1, 11)}
    assert all(status == "PASS" for statuses in rows.values() for status in statuses), output


def test_target_bar_comes_from_the_declaration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same evidence, two declared bars: the verdict follows the campaign.

    The clean root's paired delta is +0.50; declaring 0.60 means the candidate
    did not clear the campaign's own bar, and no threshold of the judge's
    choosing may override that.
    """
    root = _run_root(tmp_path, monkeypatch, min_delta=0.60)
    code, output = _judge(root, capsys)
    assert code == 1, output
    assert _rows(output)["R7"] == ["FAIL"], output


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


def test_tool_use_regression_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _run_root(
        tmp_path,
        monkeypatch,
        candidate_tool_correct=2,
        parent_tool_correct=12,
        candidate_eos_misses=0,
    )
    code, output = _judge(root, capsys)
    assert code == 1, output
    assert _rows(output)["R7"] == ["FAIL"], output
    assert "VERDICT: REJECTED" in output


def test_reported_score_the_items_do_not_produce_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _run_root(tmp_path, monkeypatch, candidate_tool_score=0.95)
    code, output = _judge(root, capsys)
    assert code == 1, output
    assert _rows(output)["R6"] == ["FAIL"], output
    assert "reported" in output and "recomputed" in output


def test_missing_raw_file_is_unknown_not_a_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _run_root(tmp_path, monkeypatch, drop_raw="candidate-tool-use")
    code, output = _judge(root, capsys)
    assert code == 2, output
    assert "R6" in _rows(output) and _rows(output)["R6"] == ["UNKNOWN"], output
    assert "VERDICT: INCONCLUSIVE" in output


def test_gate0_termination_drop_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _run_root(tmp_path, monkeypatch, candidate_eos_misses=4, parent_eos_misses=0)
    code, output = _judge(root, capsys)
    assert code == 1, output
    assert _rows(output)["R9"] == ["FAIL"], output
    assert "candidate 0.800 vs parent 1.000" in output


def test_gate0_degeneracy_rise_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _run_root(tmp_path, monkeypatch)
    # A repeated line in most candidate completions: production's loop detector
    # fires, so the item is degenerate even though it terminates and scores 1.0.
    path = root / "raw" / "candidate-tool-use-reap-moe-v1.jsonl"
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for item in items[:8]:
        item["prediction"] = "line\nline\nline\n"
        item["score"] = 0.0
    _write_jsonl(path, items)
    report_path = root / "candidate_evaluation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for run in report["runs"]:
        if run["benchmark_qualified_id"] == TARGET:
            run["metadata"]["artifact_sha256"] = _sha256(path)
            run["per_sample_scores"] = [item["score"] for item in items]
            run["score"] = sum(item["score"] for item in items) / len(items)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    code, output = _judge(root, capsys)
    assert code == 1, output
    assert _rows(output)["R10"] == ["FAIL"], output


def test_settlement_over_ceiling_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _run_root(tmp_path, monkeypatch, wall_gpu_hours=25.0)
    code, output = _judge(root, capsys)
    assert code == 1, output
    assert _rows(output)["R2"] == ["FAIL"], output
    assert "WALL_GPU_HOURS_EXCEEDED" in output


def test_tampered_accounting_artifact_fails_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _run_root(tmp_path, monkeypatch)
    path = root / "cycle_compute_accounting.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["entries"].append({"kind": "evaluation", "recipe_id": "recipe-00"})
    path.write_text(json.dumps(document), encoding="utf-8")
    code, output = _judge(root, capsys)
    assert code == 1, output
    assert _rows(output)["R3"] == ["FAIL"], output


@pytest.mark.parametrize("tamper", ["total", "measurement", "paid_flag", "entry_cost", "entries", "cycle", "flag_type"])
def test_reap_settlement_rejects_self_hashed_inconsistent_ledgers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], tamper: str,
) -> None:
    root = _run_root(tmp_path, monkeypatch)
    path = root / "cycle_compute_accounting.json"
    accounting = json.loads(path.read_text(encoding="utf-8"))
    if tamper == "total":
        accounting["totals"]["incremental"]["wall_gpu_hours"] = 0.0
    elif tamper == "measurement":
        accounting["totals"]["incremental"]["device_measured"] = True
    elif tamper == "paid_flag":
        accounting["entries"][0]["cost"]["incremental"] = False
    elif tamper == "entry_cost":
        accounting["entries"][0]["cost"]["wall_gpu_hours"] = 0.0
    elif tamper == "entries":
        accounting["entries"] = []
    elif tamper == "cycle":
        accounting["cycle_id"] = "different-campaign"
    else:
        accounting["entries"][0]["cost"]["device_measured"] = "false"
    # Rehash both identities: the accounting check must not trust a digest
    # that the attacker could recompute alongside the asserted totals.
    accounting["digest_sha256"] = ledger_digest(accounting)
    path.write_text(json.dumps(accounting), encoding="utf-8")
    record_path = root / "campaign-run.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["cost"]["accounting_digest"] = accounting["digest_sha256"]
    record_path.write_text(json.dumps(record), encoding="utf-8")

    code, output = _judge(root, capsys)
    assert code == 1, output
    assert _rows(output)["R2"] == ["FAIL"], output
    assert _rows(output)["R3"] == ["FAIL"], output


def test_reap_legacy_totals_only_accounting_remains_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _run_root(tmp_path, monkeypatch)
    path = root / "cycle_compute_accounting.json"
    accounting = json.loads(path.read_text(encoding="utf-8"))
    accounting.pop("entries")
    accounting.pop("digest_sha256")
    path.write_text(json.dumps(accounting), encoding="utf-8")
    record_path = root / "campaign-run.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["cost"]["accounting_digest"] = ledger_digest(accounting)
    record_path.write_text(json.dumps(record), encoding="utf-8")
    code, output = _judge(root, capsys)
    assert code == 2, output
    assert _rows(output)["R2"] == ["UNKNOWN"], output
    assert _rows(output)["R3"] == ["UNKNOWN"], output


def test_reap_recorded_settlement_flag_must_be_boolean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _run_root(tmp_path, monkeypatch)
    path = root / "campaign-run.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["settlement"]["budget_compliant"] = "true"
    path.write_text(json.dumps(record), encoding="utf-8")
    code, output = _judge(root, capsys)
    assert code == 1, output
    assert _rows(output)["R3"] == ["FAIL"], output


def test_retention_regression_fails_and_the_run_coupled_refusal_agrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _run_root(
        tmp_path,
        monkeypatch,
        candidate_math_correct=1,
        parent_math_correct=16,
        record_verdict="REJECTED",
        record_reasons=["RETENTION_REGRESSION: candidate 0.05 vs parent 0.80 on 'math500'"],
    )
    code, output = _judge(root, capsys)
    rows = _rows(output)
    assert code == 1, output
    assert rows["R4"] == ["FAIL"], output
    # The run refused itself for the same reason: the two authorities agree on
    # the breach, but this judge still refuses to certify what the branch refused.
    assert rows["R5"] == ["FAIL"], output


def test_run_refusal_is_not_overturned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _run_root(
        tmp_path,
        monkeypatch,
        record_verdict="REJECTED",
        record_reasons=["TARGET_REGRESSION: paired delta -0.100 below the declared bar"],
    )
    code, output = _judge(root, capsys)
    assert code == 1, output
    assert _rows(output)["R5"] == ["FAIL"], output


def test_retention_recomputation_disagreement_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A record that claims a breach the arms do not show is a refusal too."""
    root = _run_root(
        tmp_path,
        monkeypatch,
        record_verdict="REJECTED",
        record_reasons=["RETENTION_REGRESSION: a breach nothing measured"],
    )
    code, output = _judge(root, capsys)
    assert code == 1, output
    assert "RETENTION_RECOMPUTATION_DISAGREES" in output


# --------------------------------------------------------------------------
# descriptive evidence never gates
# --------------------------------------------------------------------------


def test_descriptive_reports_never_change_the_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _run_root(
        tmp_path,
        monkeypatch,
        descriptive={"verdict": "REJECTED", "narrative": "the judge should fail"},
    )
    code, output = _judge(root, capsys)
    assert code == 0, output
    assert "VERDICT: PROMOTED" in output
    assert "descriptive reports (non-gating" in output
    assert "verdict.json" in output


# --------------------------------------------------------------------------
# verdict -> exit code
# --------------------------------------------------------------------------


def test_missing_declaration_is_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _run_root(tmp_path, monkeypatch, with_campaign=False)
    code, output = _judge(root, capsys)
    assert code == 2, output
    assert "VERDICT: INCONCLUSIVE" in output
    assert _rows(output)["R7"] == ["UNKNOWN"], output


def test_usage_errors_exit_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert judge.main(["judge"]) == 2
    assert judge.main(["judge", str(tmp_path / "nope")]) == 2


# --------------------------------------------------------------------------
# the declared bar itself
# --------------------------------------------------------------------------


def test_target_threshold_is_validated_at_load(tmp_path: Path) -> None:
    document = _campaign_document(
        tmp_path, min_delta=0.05, ceilings={"per_recipe": 1, "campaign": 1}
    )
    assert CampaignManifest.from_mapping(document, source="t").target_threshold == TargetThreshold(
        benchmark=TARGET, min_delta=0.05
    )
    for bad in (
        {"benchmark": TARGET, "min_delta": 0.0},
        {"benchmark": "undeclared@2024-01", "min_delta": 0.05},
        {"benchmark": TARGET, "min_delta": float("nan")},
        {"benchmark": TARGET, "min_delta": 0.05, "extra": 1},
    ):
        document["target_threshold"] = bad
        with pytest.raises(ValueError):
            CampaignManifest.from_mapping(document, source="t")
