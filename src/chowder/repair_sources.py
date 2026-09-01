from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .contamination import (
    ContaminationAudit,
    RepairExample,
    example_fingerprints,
    write_verified_repair_dataset,
)


@dataclass(frozen=True)
class RepairSource:
    source_id: str
    ref: str
    content_sha256: str
    kind: str = "corpus"

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("repair source_id is required")
        if not self.ref.strip():
            raise ValueError("repair source ref is required")
        if len(self.content_sha256) != 64:
            raise ValueError("repair source content_sha256 must be a SHA-256 digest")
        if not self.kind.strip():
            raise ValueError("repair source kind is required")


@dataclass(frozen=True)
class SourcedRepairExample:
    example_id: str
    source_id: str
    prompt: str
    expected: str

    def __post_init__(self) -> None:
        if not self.example_id.strip():
            raise ValueError("repair example_id is required")
        if not self.source_id.strip():
            raise ValueError("repair example source_id is required")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validated_sources(sources: Iterable[RepairSource]) -> tuple[RepairSource, ...]:
    rows = tuple(sources)
    if not rows:
        raise ValueError("at least one independent repair source is required")
    ids = [source.source_id for source in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("repair source IDs must be unique")
    return tuple(sorted(rows, key=lambda source: source.source_id))


def _validated_examples(
    examples: Iterable[SourcedRepairExample],
    sources: tuple[RepairSource, ...],
) -> tuple[SourcedRepairExample, ...]:
    rows = tuple(examples)
    if not rows:
        raise ValueError("at least one sourced repair example is required")
    example_ids = [example.example_id for example in rows]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("repair example IDs must be unique")
    source_ids = {source.source_id for source in sources}
    missing = sorted({example.source_id for example in rows if example.source_id not in source_ids})
    if missing:
        raise ValueError(f"repair examples reference undeclared sources: {missing}")
    return tuple(sorted(rows, key=lambda example: example.example_id))


def write_provenanced_repair_dataset(
    *,
    examples: Iterable[SourcedRepairExample],
    sources: Iterable[RepairSource],
    holdout_fingerprint_files: Iterable[str | Path],
    dataset_path: str | Path,
    manifest_path: str | Path,
) -> tuple[str, ContaminationAudit, str]:
    """Write contamination-checked repair data plus an independent-source manifest."""

    source_rows = _validated_sources(sources)
    example_rows = _validated_examples(examples, source_rows)
    repair_examples = tuple(
        RepairExample(
            prompt=example.prompt,
            expected=example.expected,
            source_id=example.example_id,
        )
        for example in example_rows
    )
    dataset_sha, audit = write_verified_repair_dataset(
        repair_examples,
        holdout_fingerprint_files,
        dataset_path,
    )

    example_manifest = []
    for example in example_rows:
        prompt_sha, pair_sha = example_fingerprints(example.prompt, example.expected)
        example_manifest.append(
            {
                "example_id": example.example_id,
                "source_id": example.source_id,
                "prompt_sha256": prompt_sha,
                "pair_sha256": pair_sha,
            }
        )

    payload = {
        "schema_version": 1,
        "repair_dataset_sha256": dataset_sha,
        "repair_index_sha256": audit.repair_index_sha256,
        "holdout_index_sha256": list(audit.holdout_index_sha256),
        "sources": [asdict(source) for source in source_rows],
        "examples": example_manifest,
    }
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    manifest_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    verify_source_manifest(
        path,
        dataset_sha256=dataset_sha,
        contamination_audit=audit,
    )
    return dataset_sha, audit, manifest_sha


def verify_source_manifest(
    manifest_path: str | Path,
    *,
    dataset_sha256: str,
    contamination_audit: ContaminationAudit,
) -> str:
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"repair source manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("repair source manifest is not an object")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported repair source manifest schema")
    if payload.get("repair_dataset_sha256") != dataset_sha256:
        raise ValueError("repair source manifest dataset SHA does not match repair dataset")
    if payload.get("repair_index_sha256") != contamination_audit.repair_index_sha256:
        raise ValueError("repair source manifest repair-index SHA does not match audit")
    declared_holdouts = payload.get("holdout_index_sha256")
    if not isinstance(declared_holdouts, list) or tuple(sorted(declared_holdouts)) != tuple(
        sorted(contamination_audit.holdout_index_sha256)
    ):
        raise ValueError("repair source manifest holdout indexes do not match audit")

    raw_sources = payload.get("sources")
    raw_examples = payload.get("examples")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("repair source manifest contains no sources")
    if not isinstance(raw_examples, list) or not raw_examples:
        raise ValueError("repair source manifest contains no examples")

    source_ids: set[str] = set()
    for source in raw_sources:
        if not isinstance(source, Mapping):
            raise ValueError("repair source manifest source entry is invalid")
        source_id = source.get("source_id")
        ref = source.get("ref")
        content_sha = source.get("content_sha256")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("repair source manifest source_id is invalid")
        if source_id in source_ids:
            raise ValueError("repair source manifest contains duplicate source IDs")
        source_ids.add(source_id)
        if not isinstance(ref, str) or not ref.strip():
            raise ValueError("repair source manifest source ref is invalid")
        if not isinstance(content_sha, str) or len(content_sha) != 64:
            raise ValueError("repair source manifest content hash is invalid")

    example_ids: set[str] = set()
    repair_identity_rows: list[dict[str, str]] = []
    for example in raw_examples:
        if not isinstance(example, Mapping):
            raise ValueError("repair source manifest example entry is invalid")
        example_id = example.get("example_id")
        source_id = example.get("source_id")
        prompt_sha = example.get("prompt_sha256")
        pair_sha = example.get("pair_sha256")
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError("repair source manifest example_id is invalid")
        if example_id in example_ids:
            raise ValueError("repair source manifest contains duplicate example IDs")
        example_ids.add(example_id)
        if source_id not in source_ids:
            raise ValueError("repair source manifest example references undeclared source")
        if not isinstance(prompt_sha, str) or len(prompt_sha) != 64:
            raise ValueError("repair source manifest prompt hash is invalid")
        if not isinstance(pair_sha, str) or len(pair_sha) != 64:
            raise ValueError("repair source manifest pair hash is invalid")
        repair_identity_rows.append(
            {
                "source_id": example_id,
                "prompt_sha256": prompt_sha,
                "pair_sha256": pair_sha,
            }
        )

    repair_identity_rows.sort(
        key=lambda row: (row["source_id"], row["prompt_sha256"], row["pair_sha256"])
    )
    if _digest(repair_identity_rows) != contamination_audit.repair_index_sha256:
        raise ValueError("repair source manifest examples do not match audited repair examples")
    return hashlib.sha256(path.read_bytes()).hexdigest()
