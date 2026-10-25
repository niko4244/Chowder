"""Isolated baseline copy preserves exact frozen rows and source evidence."""

from __future__ import annotations

import hashlib
import json

import pytest

from chowder_batch.reap_eval_workspace import create


def _fixture(tmp_path):
    old = tmp_path / "historical"
    old.mkdir()
    for arm, suites in {
        "dense": {"gsm8k": 1319, "humaneval": 164},
        "random128": {"gsm8k": 1319},
        "dense_q4": {"gsm8k": 200},
        "reap128": {"gsm8k": 1319, "humaneval": 164},
        "reap160": {"gsm8k": 1319, "humaneval": 164},
    }.items():
        directory = old / arm
        directory.mkdir()
        summaries = {}
        for suite, n in suites.items():
            rows = "".join(json.dumps({
                "id": str(i), "correct": True, "terminated": True,
                "trigram_ratio": 0.99, "predicted_per_second": 10.0,
            }) + "\n" for i in range(n))
            path = directory / f"{suite}.jsonl"
            path.write_text(rows, encoding="utf-8", newline="\n")
            summaries[suite] = {
                "n": n, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        (directory / "summary.json").write_text(json.dumps(summaries))
    return old


def test_copied_baselines_exact_and_isolated(tmp_path):
    old = _fixture(tmp_path)
    new = tmp_path / "clean-v2-evaluation"
    receipt = create(historical=old, intended=new)
    assert new.is_dir()
    assert receipt["v2_evaluation_missing"] is True
    assert receipt["release_approved"] is False
    assert len(receipt["source_sha256"]) == 13
    assert not (new / "reap128_v2").exists()
    for relative, digest in receipt["source_sha256"].items():
        assert hashlib.sha256((new / relative).read_bytes()).hexdigest() == digest
        assert (old / relative).read_bytes() == (new / relative).read_bytes()
    with pytest.raises(ValueError, match="disjoint"):
        create(historical=old, intended=new)


def test_missing_attested_baseline_fails_without_creating_workspace(tmp_path):
    old = _fixture(tmp_path)
    (old / "random128" / "gsm8k.jsonl").unlink()
    output = tmp_path / "must-not-create"
    with pytest.raises(ValueError, match="missing certified"):
        create(historical=old, intended=output)
    assert not output.exists()
