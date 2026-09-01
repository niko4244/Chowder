from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .contamination import ContaminationAudit, RepairExample, write_verified_repair_dataset
from .failures import FailureRecord


def write_verified_failure_repair_dataset(
    failures: Iterable[FailureRecord],
    holdout_fingerprint_files: Iterable[str | Path],
    output_path: str | Path,
) -> tuple[str, ContaminationAudit]:
    """Convert repair-source failures to SFT only after holdout overlap auditing.

    This is the autonomous repair path. Unlike the lower-level legacy direct
    writer, it requires a full holdout fingerprint index and refuses any prompt
    overlap with the promotion benchmark.
    """

    rows = tuple(failures)
    if not rows:
        raise ValueError("no repair-source failures supplied")
    blocked = tuple(f.failure_id for f in rows if not f.training_eligible)
    if blocked:
        raise ValueError("autonomous repair data contains non-training-eligible failures")

    examples = tuple(
        RepairExample(prompt=failure.prompt, expected=failure.expected, source_id=failure.failure_id)
        for failure in rows
    )
    return write_verified_repair_dataset(examples, holdout_fingerprint_files, output_path)
