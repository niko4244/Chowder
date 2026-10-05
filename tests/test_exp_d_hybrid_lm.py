"""CPU runner and offline registry tests for Experiment D."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="experiment D's runner imports torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # chowder_batch is a script dir, not a package

from chowder_batch.exp_d_hybrid_lm import collate_registry, smoke, train  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "examples" / "experiment_d" / "configs"


def test_smoke_writes_cpu_random_init_evidence_exclusively(tmp_path):
    output = tmp_path / "smoke.json"
    artifact = smoke(
        config_dir=CONFIG_DIR,
        output_path=output,
        config_ids=["A_hybrid_dense.json"],
        sequence_length=4,
    )
    assert artifact["kind"] == "random_initialized_cpu_smoke"
    run = artifact["runs"][0]
    assert run["execution"]["device"] == "cpu"
    assert run["execution"]["quality"] is None
    assert run["execution"]["full_sequence_vs_incremental_max_abs_logit_error"] < 1e-5
    assert run["parameters"]["stored_parameters"] > 0
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        smoke(
            config_dir=CONFIG_DIR,
            output_path=output,
            config_ids=["A_hybrid_dense.json"],
            sequence_length=4,
        )


def test_cpu_train_requires_explicit_tokens_and_writes_labeled_smoke_result(tmp_path):
    tokens = tmp_path / "tokens.json"
    tokens.write_text(
        json.dumps({"sequences": [{"token_ids": [1, 3, 5, 7, 9]}, {"token_ids": [2, 4, 6, 8, 10]}]}),
        encoding="utf-8",
    )
    output = tmp_path / "train.json"
    result = train(
        config_path=CONFIG_DIR / "A_hybrid_dense.json",
        token_data_path=tokens,
        output_path=output,
        seed=3,
        steps=2,
        sequence_length=4,
        batch_size=1,
    )
    assert result["tokens_seen"] == 8
    assert result["device"] == "cpu"
    assert result["quality_metrics"] is None
    assert "smoke training only" in result["status"]
    assert result["step_records"][-1]["step"] == 2


def test_offline_registry_snapshot_binds_config_and_artifact_hashes(tmp_path):
    smoke_path = tmp_path / "smoke.json"
    smoke(config_dir=CONFIG_DIR, output_path=smoke_path, config_ids=["A_hybrid_dense.json"], sequence_length=3)
    output = tmp_path / "registry.json"
    registry = collate_registry(
        config_dir=CONFIG_DIR,
        artifact_paths=[smoke_path],
        output_path=output,
    )
    assert registry["experiment"] == "D"
    assert len(registry["configurations"]) == 5
    assert registry["completed_experiments"][0]["sha256"]
    assert registry["measurements"] == []
    assert registry["source_revision"] is None
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        collate_registry(config_dir=CONFIG_DIR, output_path=output)


def test_trainer_rejects_malformed_or_out_of_vocab_tokens_before_training(tmp_path):
    tokens = tmp_path / "bad.json"
    tokens.write_text(json.dumps({"sequences": [[0, 9999]]}), encoding="utf-8")
    with pytest.raises(ValueError, match="outside model vocabulary"):
        train(
            config_path=CONFIG_DIR / "A_hybrid_dense.json",
            token_data_path=tokens,
            output_path=tmp_path / "result.json",
            steps=1,
            sequence_length=2,
        )
