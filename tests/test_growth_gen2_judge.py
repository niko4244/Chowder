"""The frozen gen2 judge, tested before the run it will judge.

A judge that always refuses is as useless as one that always certifies, so
both directions are pinned with synthetic fixtures, plus the epistemics rule
inherited from the re-adjudication: undeclared provenance is UNKNOWN, never a
silent pass.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

GEN2 = Path(__file__).resolve().parent.parent / "docs" / "gen2"
_spec = importlib.util.spec_from_file_location("judge_gen2", GEN2 / "judge_gen2.py")
judge_gen2 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(judge_gen2)


def _completion(*, duplicate: bool = False, echo: bool = False, answer: str = "ping") -> str:
    """A completion whose reasoning block either echoes the answer (the
    measured gen1 defect) or does not, and which optionally opens by echoing
    the prompt template."""
    head = " with exactly: ping\nassistant\n" if echo else ""
    # A CLEAN reasoning block must not contain the answer string at all;
    # the duplicated variant reproduces the gen1 defect verbatim.
    think_body = f"I should answer directly.\n{answer}\n" if duplicate else "I should answer directly.\n"
    return f"{head}<think>\n{think_body}</think>\n\n{answer}"


def _instrument(*, dup: int, echo: int, n: int = 16) -> dict:
    prompts = list(judge_gen2.CONSTRAINED_PROMPTS) + ["filler"] * (n - len(judge_gen2.CONSTRAINED_PROMPTS))
    per_prompt = []
    for i in range(n):
        per_prompt.append(
            {
                "prompt": prompts[i],
                "expected": "ping",
                "completion": _completion(duplicate=i < dup, echo=i < echo, answer="ping"),
                "eos_terminated": True,
                "cap_hit": False,
            }
        )
    return {
        "measurement_origin": {"diagnostics": "MEASURED_THIS_GENERATION"},
        "diagnostics": {
            "n_prompts": n,
            "per_prompt": per_prompt,
            "eos_termination_rate": 1.0,
            "max_token_cap_rate": 0.0,
            "unclosed_think_rate": 0.0,
            "obvious_loop_count": 0,
            "distinct_trigram_ratio_mean": 0.97,
        },
    }


def _run_root(tmp_path: Path, instrument: dict | None, *, slices: list | None = None) -> Path:
    root = tmp_path / "run"
    root.mkdir()
    if instrument is not None:
        (root / "candidate_evaluation.json").write_text(json.dumps(instrument), encoding="utf-8")
    (root / "cycle_compute_accounting.json").write_text(
        json.dumps(
            {
                "totals": {"incremental": {"device_gpu_hours": 0.4, "wall_gpu_hours": 1.1}},
                "entries": [
                    {"kind": "training", "recipe_id": "gen2-recipe-a"},
                    {"kind": "training", "recipe_id": "gen2-recipe-b"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "gen2_contamination_manifest.json").write_text(
        json.dumps({"benchmarks": {"math500@2024-04": {"status": "CLEAN"}}}), encoding="utf-8"
    )
    (root / "chosen_candidate.json").write_text(
        json.dumps({"artifact_ref": "artifacts/a", "artifact_sha256": "a" * 64}), encoding="utf-8"
    )
    if slices:
        instrument_doc = json.loads((root / "candidate_evaluation.json").read_text(encoding="utf-8"))
        instrument_doc["protected_slices"] = slices
        (root / "candidate_evaluation.json").write_text(json.dumps(instrument_doc), encoding="utf-8")
    return root


def test_judge_certifies_a_clean_run(tmp_path: Path) -> None:
    slices = [
        {
            "benchmark_qualified_id": "math500@2024-04",
            "score": 0.0,
            "parent_score": 0.0,
            "measurement_origin": "MEASURED_THIS_GENERATION",
        },
        {
            "benchmark_qualified_id": "mgsm@2022-11",
            "score": 0.0625,
            "parent_score": 0.0,
            "measurement_origin": "MEASURED_THIS_GENERATION",
        },
    ]
    root = _run_root(tmp_path, _instrument(dup=0, echo=0), slices=slices)
    assert judge_gen2.judge(root) == 0


def test_judge_refuses_gen1_shaped_defects(tmp_path: Path) -> None:
    """The measured gen1 defect rates (dup 0.688, echo 0.438) must refuse."""
    root = _run_root(
        tmp_path,
        _instrument(dup=11, echo=7),
        slices=[
            {
                "benchmark_qualified_id": "math500@2024-04",
                "score": 0.0,
                "parent_score": 0.0,
                "measurement_origin": "MEASURED_THIS_GENERATION",
            },
            {
                "benchmark_qualified_id": "mgsm@2022-11",
                "score": 0.0,
                "parent_score": 0.0,
                "measurement_origin": "MEASURED_THIS_GENERATION",
            },
        ],
    )
    assert judge_gen2.judge(root) == 1


def test_judge_refuses_carried_slice_evidence(tmp_path: Path) -> None:
    root = _run_root(
        tmp_path,
        _instrument(dup=0, echo=0),
        slices=[
            {
                "benchmark_qualified_id": "math500@2024-04",
                "score": 0.0,
                "parent_score": 0.0,
                "measurement_origin": "CARRIED_REFERENCE",
            },
            {
                "benchmark_qualified_id": "mgsm@2022-11",
                "score": 0.0,
                "parent_score": 0.0,
                "measurement_origin": "MEASURED_THIS_GENERATION",
            },
        ],
    )
    assert judge_gen2.judge(root) == 1


def test_judge_refuses_when_provenance_is_undeclared(tmp_path: Path) -> None:
    instrument = _instrument(dup=0, echo=0)
    del instrument["measurement_origin"]
    root = _run_root(tmp_path, instrument)
    assert judge_gen2.judge(root) == 1  # UNKNOWN refuses to certify


def test_judge_refuses_missing_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    assert judge_gen2.judge(root) == 1
