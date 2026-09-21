"""Tests for the `kaggle/*.py` notebook-facing scripts.

`build_equivalence_report.py` has no Kaggle/GPU dependency and is
exercised fully, end-to-end, with synthetic fixture files. The other
scripts (`acquire_parent.py`, `bootstrap_environment.py`,
`run_parent_evaluation.py`, `upload_protected_suite.py`) genuinely need a
Kaggle notebook, a GPU, or the `kaggle` CLI to do real work -- for those,
this file only checks that they parse, import cleanly, and expose a
sane `--help`/argument-parsing surface, without performing any network,
GPU, or subprocess side effect.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_DIR = REPO_ROOT / "kaggle"


def _subprocess_env() -> dict:
    # A plain subprocess does not inherit pytest's in-process sys.path
    # (which prepends this worktree's src/ ahead of whatever chowder-ai
    # install -- possibly editable, possibly pointed at a different
    # worktree entirely -- is on the real environment's path). Force this
    # worktree's src/ to the front so the scripts under test import the
    # code actually being reviewed, not a stale sibling checkout.
    env = dict(os.environ)
    src = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not existing else f"{src}{os.pathsep}{existing}"
    return env


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, KAGGLE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "script_name",
    ["acquire_parent", "bootstrap_environment", "run_parent_evaluation", "upload_protected_suite", "build_equivalence_report"],
)
def test_script_help_does_not_crash(script_name):
    result = subprocess.run(
        [sys.executable, str(KAGGLE_DIR / f"{script_name}.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def test_acquire_parent_requires_kaggle_secrets_outside_kaggle(tmp_path):
    module = _load_module("acquire_parent")
    with pytest.raises(RuntimeError, match="kaggle_secrets"):
        module.main(["--parent", "C", "--destination", str(tmp_path / "parent-c")])


def test_upload_protected_suite_refuses_public_without_acknowledgement(tmp_path):
    module = _load_module("upload_protected_suite")
    suite_root = tmp_path / "suite-v1"
    suite_root.mkdir()
    rc = module.main(
        [
            "--suite-root", str(suite_root),
            "--kaggle-dataset-slug", "someuser/qwen38-protected-suite-v1",
            "--publish-publicly",
        ]
    )
    assert rc == 1


def test_upload_protected_suite_refuses_missing_root(tmp_path):
    module = _load_module("upload_protected_suite")
    rc = module.main(
        [
            "--suite-root", str(tmp_path / "does-not-exist"),
            "--kaggle-dataset-slug", "someuser/qwen38-protected-suite-v1",
        ]
    )
    assert rc == 1


# ---- build_equivalence_report.py: fully exercised end to end ----------------


def _fingerprint_dict(**overrides) -> dict:
    base = {
        "python_version": "3.11.9",
        "torch_version": "2.11.0+cu128",
        "transformers_version": "5.16.1",
        "bitsandbytes_version": "0.50.2",
        "accelerate_version": "1.6.0",
        "cuda_runtime_version": "12.8",
        "gpu_models": ["NVIDIA GeForce RTX 5060 Ti"],
        "gpu_count": 1,
        "device_map_summary": '{"": 0}',
        "quantization": "4bit",
        "dtype": "bfloat16",
        "tokenizer_identity_sha256": "t" * 64,
        "chowder_commit_sha": "a" * 40,
    }
    base.update(overrides)
    return base


def _build_report_and_evidence(tmp_path, *, suffix: str):
    from chowder.parent_eval import PARENT_DIMENSIONS, ParentEvalSpec, ParentSuiteSpec, aggregate_parent_result

    suites = [
        ParentSuiteSpec(name=f"suite-{dim}-v1", dimension=dim, dataset=f"synthetic://{dim}", max_new_tokens=256)
        for dim in PARENT_DIMENSIONS
    ]
    spec = ParentEvalSpec(suites=tuple(suites))
    metrics = {suite.name: 1.0 for suite in spec.suites}
    report = aggregate_parent_result(
        spec=spec, base_model="parent-a-qwen38-27b-official", revision="a" * 40, metrics=metrics, evidence={}
    )
    suite_evidence = {suite.name: {"holdout_fingerprints_sha256": f"digest-{suite.name}"} for suite in spec.suites}

    predictions_dir = tmp_path / f"predictions-{suffix}"
    predictions_dir.mkdir()
    for suite in spec.suites:
        path = predictions_dir / f"predictions-{suite.name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for i in range(6):
                row = {
                    "prompt": f"prompt {suite.name} {i}",
                    "expected": "42",
                    "prediction": "reasoning\n</think>\n\n42",
                    "score": 1.0,
                }
                handle.write(json.dumps(row) + "\n")
    return report.to_dict(), suite_evidence, predictions_dir


def test_build_equivalence_report_end_to_end_qualified(tmp_path):
    local_report, local_suite_evidence, local_dir = _build_report_and_evidence(tmp_path, suffix="local")
    kaggle_report, kaggle_suite_evidence, kaggle_dir = _build_report_and_evidence(tmp_path, suffix="kaggle")

    (tmp_path / "local-report.json").write_text(json.dumps(local_report))
    (tmp_path / "kaggle-report.json").write_text(json.dumps(kaggle_report))
    (tmp_path / "local-suite-evidence.json").write_text(json.dumps(local_suite_evidence))
    (tmp_path / "kaggle-suite-evidence.json").write_text(json.dumps(kaggle_suite_evidence))
    (tmp_path / "local-tokenizer.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2Tokenizer", "vocab_size": 151936, "identity_sha256": "z" * 64})
    )
    (tmp_path / "kaggle-tokenizer.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2Tokenizer", "vocab_size": 151936, "identity_sha256": "z" * 64})
    )
    (tmp_path / "local-environment.json").write_text(json.dumps(_fingerprint_dict()))
    (tmp_path / "kaggle-environment.json").write_text(
        json.dumps(_fingerprint_dict(gpu_models=["Tesla T4"], dtype="float16"))
    )

    module = _load_module("build_equivalence_report")
    report_output = tmp_path / "equivalence-report.json"
    record_output = tmp_path / "qualification-record.json"
    rc = module.main(
        [
            "--parent-label", "parent-a-qwen38-27b-official",
            "--local-report", str(tmp_path / "local-report.json"),
            "--kaggle-report", str(tmp_path / "kaggle-report.json"),
            "--local-suite-evidence", str(tmp_path / "local-suite-evidence.json"),
            "--kaggle-suite-evidence", str(tmp_path / "kaggle-suite-evidence.json"),
            "--local-predictions-dir", str(local_dir),
            "--kaggle-predictions-dir", str(kaggle_dir),
            "--local-tokenizer", str(tmp_path / "local-tokenizer.json"),
            "--kaggle-tokenizer", str(tmp_path / "kaggle-tokenizer.json"),
            "--local-environment", str(tmp_path / "local-environment.json"),
            "--kaggle-environment", str(tmp_path / "kaggle-environment.json"),
            "--qualification-id", "local-rtx5060ti_vs_kaggle-t4x2",
            "--report-output", str(report_output),
            "--record-output", str(record_output),
        ]
    )
    assert rc == 0
    record = json.loads(record_output.read_text())
    assert record["status"] == "qualified"
    assert record["item_total_count"] == 54
    report = json.loads(report_output.read_text())
    assert report["item_score_agreement_count"] == 54


def test_build_equivalence_report_exit_code_2_when_not_qualified(tmp_path):
    local_report, local_suite_evidence, local_dir = _build_report_and_evidence(tmp_path, suffix="local")
    kaggle_report, kaggle_suite_evidence, kaggle_dir = _build_report_and_evidence(tmp_path, suffix="kaggle")

    (tmp_path / "local-report.json").write_text(json.dumps(local_report))
    (tmp_path / "kaggle-report.json").write_text(json.dumps(kaggle_report))
    # Suite digests disagree -> not_qualified.
    (tmp_path / "local-suite-evidence.json").write_text(json.dumps(local_suite_evidence))
    (tmp_path / "kaggle-suite-evidence.json").write_text(
        json.dumps({name: {"holdout_fingerprints_sha256": "DIFFERENT"} for name in local_suite_evidence})
    )
    (tmp_path / "local-tokenizer.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2Tokenizer", "vocab_size": 151936, "identity_sha256": "z" * 64})
    )
    (tmp_path / "kaggle-tokenizer.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2Tokenizer", "vocab_size": 151936, "identity_sha256": "z" * 64})
    )
    (tmp_path / "local-environment.json").write_text(json.dumps(_fingerprint_dict()))
    (tmp_path / "kaggle-environment.json").write_text(json.dumps(_fingerprint_dict(gpu_models=["Tesla T4"])))

    module = _load_module("build_equivalence_report")
    rc = module.main(
        [
            "--parent-label", "parent-a-qwen38-27b-official",
            "--local-report", str(tmp_path / "local-report.json"),
            "--kaggle-report", str(tmp_path / "kaggle-report.json"),
            "--local-suite-evidence", str(tmp_path / "local-suite-evidence.json"),
            "--kaggle-suite-evidence", str(tmp_path / "kaggle-suite-evidence.json"),
            "--local-predictions-dir", str(local_dir),
            "--kaggle-predictions-dir", str(kaggle_dir),
            "--local-tokenizer", str(tmp_path / "local-tokenizer.json"),
            "--kaggle-tokenizer", str(tmp_path / "kaggle-tokenizer.json"),
            "--local-environment", str(tmp_path / "local-environment.json"),
            "--kaggle-environment", str(tmp_path / "kaggle-environment.json"),
            "--qualification-id", "local-rtx5060ti_vs_kaggle-t4x2",
            "--report-output", str(tmp_path / "equivalence-report.json"),
            "--record-output", str(tmp_path / "qualification-record.json"),
        ]
    )
    assert rc == 2
