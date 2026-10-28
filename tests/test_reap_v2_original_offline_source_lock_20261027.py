"""Original canonical source identity regression; fixtures never model scores."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from chowder_batch import reap_eval
from chowder_batch import reap_v2_original_offline_source_lock_20261027 as source


def test_real_offline_gsm8k_and_humaneval_full_population(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    result = source.verify_current_offline()
    assert result["status"] == "COMPLETE_ORIGINAL_FROZEN_OFFLINE_SOURCES_MATCHED"
    assert result["cohorts"]["gsm8k"]["full_rendered_source_sha256"] == source.GSM[1]
    assert result["cohorts"]["humaneval"]["full_rendered_source_sha256"] == source.HUMAN[1]
    assert result["cohorts"]["gsm8k"]["n"] == 1319
    assert result["cohorts"]["humaneval"]["n"] == 164
    assert result["network_or_model_called"] is False
    assert result["gen0"]["chosen"] is None


@pytest.mark.parametrize("suite", ["gsm8k", "humaneval"])
def test_original_complete_fixture_is_source_bound(suite):
    items = reap_eval._load(suite)
    actual = source.prove_items(suite, items)
    assert actual["n"] in (1319, 164)


@pytest.mark.parametrize("suite,mutator,reason", [
    ("gsm8k", lambda items: items.pop(), "population"),
    ("gsm8k", lambda items: items[0].update(id="gsm8k-1"), "canonical"),
    ("gsm8k", lambda items: items[0].update(expected="tampered-answer"), "SHA drift"),
    ("humaneval", lambda items: items[0].update(test="pass"), "SHA drift"),
    ("humaneval", lambda items: items[0].update(entry_point=""), "test, header"),
])
def test_mutated_original_source_fail_closed(suite, mutator, reason):
    items = deepcopy(reap_eval._load(suite))
    mutator(items)
    with pytest.raises(source.FrozenSourceRefused, match=reason):
        source.prove_items(suite, items)


def test_offline_flags_required_before_loading_dataset(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    with pytest.raises(source.FrozenSourceRefused, match="both HF offline"):
        source.verify_current_offline()


def test_real_subprocess_offline_cli_writes_exclusive_original_receipt(tmp_path: Path):
    artifact = tmp_path / "real-original-offline.json"
    first = subprocess.run([
        sys.executable, "-m", "chowder_batch.reap_v2_original_offline_source_lock_20261027",
        "--out", str(artifact)], capture_output=True, text=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
        timeout=100, check=False)
    assert first.returncode == 0, first.stdout + first.stderr
    result = json.loads(artifact.read_text(encoding="utf-8"))
    assert result["cohorts"]["gsm8k"]["n"] == 1319
    assert result["cohorts"]["humaneval"]["n"] == 164
    assert result["official_quality_scored"] is False
    second = subprocess.run([
        sys.executable, "-m", "chowder_batch.reap_v2_original_offline_source_lock_20261027",
        "--out", str(artifact)], capture_output=True, text=True,
        timeout=15, check=False)
    assert second.returncode == 2
    assert json.loads(artifact.read_text(encoding="utf-8")) == result
