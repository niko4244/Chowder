from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


def normalize_example_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def example_fingerprints(prompt: str, expected: str) -> tuple[str, str]:
    normalized_prompt = normalize_example_text(prompt)
    normalized_expected = normalize_example_text(expected)
    prompt_sha = hashlib.sha256(normalized_prompt.encode("utf-8")).hexdigest()
    pair_sha = hashlib.sha256(
        (normalized_prompt + "\x00" + normalized_expected).encode("utf-8")
    ).hexdigest()
    return prompt_sha, pair_sha


@dataclass(frozen=True)
class RepairExample:
    prompt: str
    expected: str
    source_id: str


@dataclass(frozen=True)
class ContaminationAudit:
    clean: bool
    checked_examples: int
    prompt_overlap_sha256: tuple[str, ...]
    pair_overlap_sha256: tuple[str, ...]
    holdout_index_sha256: tuple[str, ...]
    repair_index_sha256: str

    @property
    def overlap_count(self) -> int:
        return len(set(self.prompt_overlap_sha256) | set(self.pair_overlap_sha256))


def _canonical_digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _repair_index_rows(examples: Iterable[RepairExample]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_source_ids: set[str] = set()
    for example in examples:
        source_id = example.source_id.strip()
        if not source_id:
            raise ValueError("repair example source_id is required")
        if source_id in seen_source_ids:
            raise ValueError(f"duplicate repair example source_id: {source_id}")
        seen_source_ids.add(source_id)
        prompt_sha, pair_sha = example_fingerprints(example.prompt, example.expected)
        rows.append(
            {
                "source_id": source_id,
                "prompt_sha256": prompt_sha,
                "pair_sha256": pair_sha,
            }
        )
    rows.sort(key=lambda row: (row["source_id"], row["prompt_sha256"], row["pair_sha256"]))
    return rows


def write_holdout_fingerprint_index(
    examples: Iterable[tuple[str, str]], output_path: str | Path
) -> str:
    """Write hash-only holdout fingerprints; no raw prompt/answer text is stored."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for prompt, expected in examples:
        prompt_sha, pair_sha = example_fingerprints(prompt, expected)
        key = (prompt_sha, pair_sha)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"prompt_sha256": prompt_sha, "pair_sha256": pair_sha})
    rows.sort(key=lambda row: (row["prompt_sha256"], row["pair_sha256"]))
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_index(path: Path) -> tuple[set[str], set[str], str]:
    if not path.is_file():
        raise FileNotFoundError(f"holdout fingerprint index not found: {path}")
    prompt_hashes: set[str] = set()
    pair_hashes: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{line_number} fingerprint row is not an object")
            prompt_sha = row.get("prompt_sha256")
            pair_sha = row.get("pair_sha256")
            if not isinstance(prompt_sha, str) or len(prompt_sha) != 64:
                raise ValueError(f"{path}:{line_number} invalid prompt fingerprint")
            if not isinstance(pair_sha, str) or len(pair_sha) != 64:
                raise ValueError(f"{path}:{line_number} invalid pair fingerprint")
            prompt_hashes.add(prompt_sha)
            pair_hashes.add(pair_sha)
    return prompt_hashes, pair_hashes, hashlib.sha256(path.read_bytes()).hexdigest()


def repair_dataset_index_digest(path: str | Path) -> str:
    """Recompute the audited repair-example identity from a written Chowder dataset."""

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"repair dataset not found: {dataset_path}")
    examples: list[RepairExample] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{dataset_path}:{line_number} repair row is not an object")
            source_id = row.get("source_id")
            prompt = row.get("prompt")
            expected = row.get("expected")
            declared_prompt_sha = row.get("prompt_sha256")
            declared_pair_sha = row.get("pair_sha256")
            if not all(isinstance(value, str) for value in (source_id, prompt, expected, declared_prompt_sha, declared_pair_sha)):
                raise ValueError(f"{dataset_path}:{line_number} repair row lacks verification fields")
            prompt_sha, pair_sha = example_fingerprints(prompt, expected)
            if prompt_sha != declared_prompt_sha or pair_sha != declared_pair_sha:
                raise ValueError(f"{dataset_path}:{line_number} repair row fingerprint mismatch")
            examples.append(RepairExample(prompt=prompt, expected=expected, source_id=source_id))
    if not examples:
        raise ValueError("repair dataset contains no examples")
    return _canonical_digest(_repair_index_rows(examples))


def audit_repair_examples(
    examples: Iterable[RepairExample],
    holdout_fingerprint_files: Iterable[str | Path],
) -> ContaminationAudit:
    rows = tuple(examples)
    if not rows:
        raise ValueError("no repair examples supplied")
    repair_index_rows = _repair_index_rows(rows)

    holdout_prompts: set[str] = set()
    holdout_pairs: set[str] = set()
    index_digests: list[str] = []
    files = tuple(Path(path) for path in holdout_fingerprint_files)
    if not files:
        raise ValueError("at least one holdout fingerprint index is required")
    for path in files:
        prompts, pairs, digest = _load_index(path)
        holdout_prompts.update(prompts)
        holdout_pairs.update(pairs)
        index_digests.append(digest)

    prompt_overlaps: set[str] = set()
    pair_overlaps: set[str] = set()
    for row in repair_index_rows:
        prompt_sha = row["prompt_sha256"]
        pair_sha = row["pair_sha256"]
        if prompt_sha in holdout_prompts:
            prompt_overlaps.add(prompt_sha)
        if pair_sha in holdout_pairs:
            pair_overlaps.add(pair_sha)

    return ContaminationAudit(
        clean=not prompt_overlaps and not pair_overlaps,
        checked_examples=len(rows),
        prompt_overlap_sha256=tuple(sorted(prompt_overlaps)),
        pair_overlap_sha256=tuple(sorted(pair_overlaps)),
        holdout_index_sha256=tuple(sorted(index_digests)),
        repair_index_sha256=_canonical_digest(repair_index_rows),
    )


def write_verified_repair_dataset(
    examples: Iterable[RepairExample],
    holdout_fingerprint_files: Iterable[str | Path],
    output_path: str | Path,
) -> tuple[str, ContaminationAudit]:
    """Write repair SFT data only after a full holdout-overlap audit passes."""

    rows = tuple(examples)
    audit = audit_repair_examples(rows, holdout_fingerprint_files)
    if not audit.clean:
        raise ValueError(
            "repair dataset overlaps holdout benchmark prompts; refusing contaminated training data"
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in rows:
            prompt_sha, pair_sha = example_fingerprints(example.prompt, example.expected)
            handle.write(
                json.dumps(
                    {
                        "text": f"User: {example.prompt}\nAssistant: {example.expected}",
                        "prompt": example.prompt,
                        "expected": example.expected,
                        "source_id": example.source_id,
                        "prompt_sha256": prompt_sha,
                        "pair_sha256": pair_sha,
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )
    if repair_dataset_index_digest(path) != audit.repair_index_sha256:
        raise RuntimeError("written repair dataset does not match contamination audit")
    return hashlib.sha256(path.read_bytes()).hexdigest(), audit
