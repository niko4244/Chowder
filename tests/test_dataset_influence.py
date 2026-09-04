from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from chowder.checkpoint_bisect import CheckpointVerdict
from chowder.dataset_influence import TrainingExampleInfluence, rank_training_examples_by_loss_delta
from chowder.models import ExperimentResult, GateDecision

_REAL_ML_SMOKE = pytest.mark.skipif(
    os.environ.get("CHOWDER_REAL_ML_SMOKE") != "1",
    reason="real ML smoke requires CHOWDER_REAL_ML_SMOKE=1 and train dependencies",
)
_TINY_MODEL = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"


def _verdict(checkpoint_dir, step, sha="a" * 64):
    result = ExperimentResult(
        "e1", {"quality": 0.0}, 0.0, artifact_ref=str(checkpoint_dir),
        evidence={"checkpoint_step": step, "checkpoint_sha256": sha},
    )
    decision = GateDecision(True, 0.0, {}, (), (), True, "n/a")
    return CheckpointVerdict(str(checkpoint_dir), step, result, decision)


def _write_dataset(tmp_path, rows):
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps({"text": row}) + "\n" for row in rows), encoding="utf-8")
    return path


def _patch_subprocess(monkeypatch, losses_by_label):
    """Fakes subprocess.run: writes the result JSON the real worker's CLI
    would produce, keyed by which checkpoint (good/bad) the call is for,
    inferred from the --spec file's own adapter_dir content."""

    def _fake_run(command, **kwargs):
        spec_path = Path(command[command.index("--spec") + 1])
        result_path = Path(command[command.index("--result") + 1])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        label = "good" if "good" in spec["adapter_dir"] else "bad"
        result_path.write_text(
            json.dumps(
                {"losses": losses_by_label[label], "model_cache_status": "hit", "device": "cpu"}
            ),
            encoding="utf-8",
        )

        class _Completed:
            returncode = 0
            stderr = ""

        return _Completed()

    monkeypatch.setattr("chowder.dataset_influence.subprocess.run", _fake_run)


def test_ranks_descending_by_influence_score(tmp_path, monkeypatch):
    dataset = _write_dataset(tmp_path, ["row-a", "row-b", "row-c"])
    _patch_subprocess(monkeypatch, {"good": [1.0, 1.0, 1.0], "bad": [1.0, 3.0, 0.5]})
    good = _verdict(tmp_path / "good-checkpoint", 5)
    bad = _verdict(tmp_path / "bad-checkpoint", 10)

    records = rank_training_examples_by_loss_delta(
        good_checkpoint=good, bad_checkpoint=bad,
        base_model=_TINY_MODEL, dataset_path=str(dataset), work_dir=tmp_path,
    )

    assert [r.row_index for r in records] == [1, 0, 2]  # deltas: 2.0, 0.0, -0.5
    assert records[0].influence_score == pytest.approx(2.0)
    assert records[-1].influence_score == pytest.approx(-0.5)
    assert all(a.influence_score >= b.influence_score for a, b in zip(records, records[1:]))


def test_records_carry_prompt_excerpts_and_provenance(tmp_path, monkeypatch):
    dataset = _write_dataset(tmp_path, ["hello world", "goodbye world"])
    _patch_subprocess(monkeypatch, {"good": [1.0, 1.0], "bad": [1.5, 1.0]})
    good = _verdict(tmp_path / "good-checkpoint", 5, sha="g" * 64)
    bad = _verdict(tmp_path / "bad-checkpoint", 10, sha="b" * 64)

    records = rank_training_examples_by_loss_delta(
        good_checkpoint=good, bad_checkpoint=bad,
        base_model=_TINY_MODEL, dataset_path=str(dataset), work_dir=tmp_path,
    )

    top = records[0]
    assert top.row_index == 0
    assert top.prompt_excerpt == "hello world"
    assert top.checkpoint_interval == (str(tmp_path / "good-checkpoint"), str(tmp_path / "bad-checkpoint"))
    assert top.provenance["base_model"] == _TINY_MODEL
    assert top.provenance["good_checkpoint_step"] == 5
    assert top.provenance["bad_checkpoint_step"] == 10
    assert top.provenance["good_checkpoint_sha256"] == "g" * 64
    assert top.provenance["bad_checkpoint_sha256"] == "b" * 64


def test_confidence_reflects_how_much_a_delta_stands_out(tmp_path, monkeypatch):
    # One dramatic outlier among many near-zero deltas -> the outlier
    # should get high confidence, the rest low.
    dataset = _write_dataset(tmp_path, [f"row-{i}" for i in range(10)])
    good_losses = [1.0] * 10
    bad_losses = [1.0] * 9 + [10.0]  # row 9 is a real outlier
    _patch_subprocess(monkeypatch, {"good": good_losses, "bad": bad_losses})
    good = _verdict(tmp_path / "good-checkpoint", 5)
    bad = _verdict(tmp_path / "bad-checkpoint", 10)

    records = rank_training_examples_by_loss_delta(
        good_checkpoint=good, bad_checkpoint=bad,
        base_model=_TINY_MODEL, dataset_path=str(dataset), work_dir=tmp_path,
    )

    assert records[0].row_index == 9
    assert records[0].confidence == "high"
    assert records[-1].confidence == "low"


def test_raises_on_mismatched_row_counts_between_checkpoints(tmp_path, monkeypatch):
    dataset = _write_dataset(tmp_path, ["row-a", "row-b"])
    _patch_subprocess(monkeypatch, {"good": [1.0, 1.0], "bad": [1.0]})
    good = _verdict(tmp_path / "good-checkpoint", 5)
    bad = _verdict(tmp_path / "bad-checkpoint", 10)

    with pytest.raises(RuntimeError, match="disagree on row count"):
        rank_training_examples_by_loss_delta(
            good_checkpoint=good, bad_checkpoint=bad,
            base_model=_TINY_MODEL, dataset_path=str(dataset), work_dir=tmp_path,
        )


def test_raises_with_worker_stderr_when_subprocess_fails(tmp_path, monkeypatch):
    dataset = _write_dataset(tmp_path, ["row-a"])

    def _fake_run(command, **kwargs):
        class _Completed:
            returncode = 1
            stderr = "boom: real worker failure"

        return _Completed()

    monkeypatch.setattr("chowder.dataset_influence.subprocess.run", _fake_run)
    good = _verdict(tmp_path / "good-checkpoint", 5)
    bad = _verdict(tmp_path / "bad-checkpoint", 10)

    with pytest.raises(RuntimeError, match="boom: real worker failure"):
        rank_training_examples_by_loss_delta(
            good_checkpoint=good, bad_checkpoint=bad,
            base_model=_TINY_MODEL, dataset_path=str(dataset), work_dir=tmp_path,
        )


@_REAL_ML_SMOKE
def test_real_dataset_influence_ranks_harder_rows_above_easy_repetitive_ones(tmp_path):
    """End-to-end with a REAL tiny model: train to 2 real checkpoints on a
    mix of highly-repetitive "easy" rows and a few distinct "odd" rows,
    then confirm the real measured per-example loss delta ranks the odd
    rows (the ones the easy-pattern-fitting training helps least) higher
    than the well-learned repetitive ones -- a real, sensible signal, not
    just "runs without crashing"."""
    import json as json_module

    from chowder.backends.transformers_peft import TransformersPeftExecutor
    from chowder.executors import ExecutionContext
    from chowder.memory import HardwareProfile
    from chowder.models import Experiment, Hypothesis

    def _hardware():
        return HardwareProfile(16, 64, 500, 12, 40, 3)

    train_path = tmp_path / "train.jsonl"
    easy_rows = [{"text": f"Question: what is {i}? Answer: {i * 2}"} for i in range(15)] * 3
    odd_rows = [{"text": f"The quick brown fox jumps over the lazy dog number {i}."} for i in range(5)]
    rows = easy_rows + odd_rows
    train_path.write_text(
        "".join(json_module.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    resolved_config = {
        "backend": {
            "type": "transformers-peft",
            "base_model": _TINY_MODEL,
            "dataset": str(train_path),
            "max_length": 64,
            "quantization": "none",
            "precision": "fp32",
            "lora": {"r": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]},
            "training": {
                "max_steps": 12,
                "save_strategy": "steps",
                "save_steps": 6,
                "learning_rate": 0.01,
                "batch_size": 4,
                "gradient_accumulation_steps": 1,
                "logging_steps": 1,
                "gradient_checkpointing": False,
            },
            "runtime": {"timeout_seconds": 180.0},
        },
    }
    context = ExecutionContext(
        hardware=_hardware(), work_dir=str(tmp_path), seed=1, resolved_config=resolved_config
    )
    experiment = Experiment("influence-probe", None, Hypothesis("o", "c", "i"), {}, 1.0)

    trainer = TransformersPeftExecutor()
    artifact = trainer.run(experiment, context)
    checkpoints = sorted(
        (Path(artifact.artifact_ref) / "trainer").glob("checkpoint-*"),
        key=lambda p: int(p.name.rsplit("-", 1)[1]),
    )
    assert len(checkpoints) >= 2

    good = _verdict(checkpoints[0], 6)
    bad = _verdict(checkpoints[-1], 12)

    records = rank_training_examples_by_loss_delta(
        good_checkpoint=good, bad_checkpoint=bad,
        base_model=_TINY_MODEL, dataset_path=str(train_path), work_dir=tmp_path, max_length=64,
    )

    assert len(records) == len(rows)
    assert all(isinstance(r, TrainingExampleInfluence) for r in records)
    odd_row_indices = set(range(len(easy_rows), len(rows)))
    top_10_indices = {r.row_index for r in records[:10]}
    assert top_10_indices & odd_row_indices, (
        "expected at least one of the harder/odd rows among the highest-ranked (most-worsened) examples"
    )
