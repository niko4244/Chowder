"""Which training examples plausibly contributed to a regression -- a
cheap, honest approximation, not a claim of exact causality.

`checkpoint_bisect.py` already answers "when during a rejected run did a
regression first appear" (a `CheckpointVerdict` pair: the last real
gate-accepted checkpoint and the first real gate-rejected one). This module
answers the next question: which training examples' own loss got
measurably worse between those two real checkpoints -- a real, cheap
(forward-only, no re-training) per-example loss delta computed via
`backends/dataset_influence_worker.py`'s `compute_per_example_losses`,
isolated in its own subprocess the same way every other real-hardware
experiment in this codebase is (`activation_offload.py`, `optimizer_tiering.py`,
`frozen_layer_streaming.py`), so loading two different checkpoints'
models sequentially never shares one CUDA context.

Not causal: a training example whose own loss got measurably worse between
two real checkpoints is a real, measured correlation with the direction
training moved in during that interval -- not proof that example caused a
regression on some other, unrelated evaluation prompt. `confidence` is
reported as its own explicit field precisely so a caller cannot mistake
ranking position for a causal claim.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .backends.dataset_influence_worker import _load_rows
from .checkpoint_bisect import CheckpointVerdict

_DEFAULT_TIMEOUT_SECONDS = 300.0
_PROMPT_EXCERPT_CHARS = 200


@dataclass(frozen=True)
class TrainingExampleInfluence:
    """One training example's real, measured association with a regression
    between two real checkpoints. Ranked by `influence_score` (descending):
    a positive score means this example's own loss got *worse* moving from
    the good checkpoint to the bad one -- the direction a regression cause
    would plausibly show, not proof that it is one.
    """

    row_index: int
    prompt_excerpt: str
    good_checkpoint_loss: float
    bad_checkpoint_loss: float
    influence_score: float
    confidence: str  # "low" | "medium" | "high" -- how much this example's delta stands out from the rest
    supporting_evidence: Mapping[str, Any]
    checkpoint_interval: tuple[str, str]  # (good checkpoint dir, bad checkpoint dir)
    provenance: Mapping[str, Any]


def _confidence(z_score: float) -> str:
    magnitude = abs(z_score)
    if magnitude >= 2.0:
        return "high"
    if magnitude >= 1.0:
        return "medium"
    return "low"


def _run_loss_worker(
    *,
    base_model: str,
    adapter_dir: str,
    dataset: str,
    text_field: str,
    max_length: int,
    device: str,
    precision: str,
    quantization: str,
    revision: str | None,
    offline: bool,
    work_dir: str | Path,
    label: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    scratch_dir = Path(work_dir) / ".chowder" / "_dataset_influence_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    spec_path = scratch_dir / f"{label}-spec.json"
    result_path = scratch_dir / f"{label}-result.json"
    spec_path.write_text(
        json.dumps(
            {
                "base_model": base_model,
                "adapter_dir": adapter_dir,
                "dataset": dataset,
                "text_field": text_field,
                "max_length": max_length,
                "device": device,
                "precision": precision,
                "quantization": quantization,
                "revision": revision,
                "offline": offline,
            }
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable, "-m", "chowder.backends.dataset_influence_worker",
        "--spec", str(spec_path), "--result", str(result_path),
    ]
    process = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds)
    if process.returncode != 0:
        raise RuntimeError(
            f"dataset influence loss worker ({label}) failed with exit code "
            f"{process.returncode}:\n{process.stderr[-4000:]}"
        )
    if not result_path.is_file():
        raise RuntimeError(f"dataset influence loss worker ({label}) produced no result")
    return json.loads(result_path.read_text(encoding="utf-8"))


def rank_training_examples_by_loss_delta(
    *,
    good_checkpoint: CheckpointVerdict,
    bad_checkpoint: CheckpointVerdict,
    base_model: str,
    dataset_path: str,
    work_dir: str | Path,
    text_field: str = "text",
    max_length: int = 512,
    device: str = "auto",
    precision: str = "fp32",
    quantization: str = "none",
    revision: str | None = None,
    offline: bool = False,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> tuple[TrainingExampleInfluence, ...]:
    """Rank every row in `dataset_path` by how much its own real, measured
    loss got worse between `good_checkpoint` and `bad_checkpoint` -- two
    real checkpoints from the *same* rejected run, as already identified by
    `checkpoint_bisect.evaluate_all_checkpoints()` (pass its
    `verdicts[i-1]`/`first_regressing` pair, or any other real, gate-scored
    pair from that same outcome).

    Two real, isolated subprocess loss computations (one per checkpoint),
    never sharing a CUDA context. Confidence is a per-example z-score
    against the *population* of deltas measured here, not a claim about
    any individual example in isolation.
    """
    good_result = _run_loss_worker(
        base_model=base_model, adapter_dir=good_checkpoint.checkpoint_dir, dataset=dataset_path,
        text_field=text_field, max_length=max_length, device=device, precision=precision,
        quantization=quantization, revision=revision, offline=offline, work_dir=work_dir,
        label="good", timeout_seconds=timeout_seconds,
    )
    bad_result = _run_loss_worker(
        base_model=base_model, adapter_dir=bad_checkpoint.checkpoint_dir, dataset=dataset_path,
        text_field=text_field, max_length=max_length, device=device, precision=precision,
        quantization=quantization, revision=revision, offline=offline, work_dir=work_dir,
        label="bad", timeout_seconds=timeout_seconds,
    )
    good_losses = good_result["losses"]
    bad_losses = bad_result["losses"]
    if len(good_losses) != len(bad_losses):
        raise RuntimeError(
            "good/bad checkpoint loss computations disagree on row count "
            f"({len(good_losses)} vs {len(bad_losses)}) -- dataset_path must be identical "
            "for both checkpoints"
        )

    rows = _load_rows(dataset_path, text_field)
    deltas = [bad - good for good, bad in zip(good_losses, bad_losses)]
    mean_delta = statistics.mean(deltas)
    stdev_delta = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
    checkpoint_interval = (good_checkpoint.checkpoint_dir, bad_checkpoint.checkpoint_dir)
    provenance = {
        "base_model": base_model,
        "dataset_path": str(dataset_path),
        "good_checkpoint_step": good_checkpoint.step,
        "bad_checkpoint_step": bad_checkpoint.step,
        "good_checkpoint_sha256": good_checkpoint.result.evidence.get("checkpoint_sha256"),
        "bad_checkpoint_sha256": bad_checkpoint.result.evidence.get("checkpoint_sha256"),
        "good_model_cache_status": good_result.get("model_cache_status"),
        "bad_model_cache_status": bad_result.get("model_cache_status"),
    }

    records: list[TrainingExampleInfluence] = []
    for index, (good_loss, bad_loss, delta) in enumerate(zip(good_losses, bad_losses, deltas)):
        z_score = (delta - mean_delta) / stdev_delta if stdev_delta > 0 else 0.0
        records.append(
            TrainingExampleInfluence(
                row_index=index,
                prompt_excerpt=rows[index][:_PROMPT_EXCERPT_CHARS],
                good_checkpoint_loss=good_loss,
                bad_checkpoint_loss=bad_loss,
                influence_score=delta,
                confidence=_confidence(z_score),
                supporting_evidence={
                    "z_score": z_score,
                    "population_mean_delta": mean_delta,
                    "population_stdev_delta": stdev_delta,
                },
                checkpoint_interval=checkpoint_interval,
                provenance=provenance,
            )
        )
    return tuple(sorted(records, key=lambda record: record.influence_score, reverse=True))
