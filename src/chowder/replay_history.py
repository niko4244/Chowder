from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .provenance import sha256_file
from .repair_candidates import VerifiedReplayDataset


@dataclass(frozen=True)
class ReplayHistorySource:
    path: str
    sha256: str
    text_field: str = "text"
    role: str = "prior_training"

    def verify(self) -> Path:
        if len(self.sha256) != 64:
            raise ValueError("replay history source SHA-256 is invalid")
        if not self.text_field.strip():
            raise ValueError("replay history source text_field is required")
        if not self.role.strip():
            raise ValueError("replay history source role is required")
        path = Path(self.path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"replay history source not found: {path}")
        if sha256_file(path) != self.sha256:
            raise ValueError("replay history source content changed after training")
        return path


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _rows(path: Path) -> tuple[Mapping[str, object], ...]:
    raw = path.read_text(encoding="utf-8")
    stripped = raw.strip()
    if not stripped:
        raise ValueError(f"replay history source is empty: {path}")

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        rows = tuple(parsed)
    elif isinstance(parsed, Mapping):
        rows = (parsed,)
    else:
        decoded: list[object] = []
        for line_number, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                decoded.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL replay source {path}:{line_number}"
                ) from exc
        rows = tuple(decoded)

    if not rows:
        raise ValueError(f"replay history source contains no rows: {path}")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"replay history source contains a non-object row: {path}")
    return tuple(row for row in rows if isinstance(row, Mapping))


def _texts(source: ReplayHistorySource) -> tuple[str, ...]:
    path = source.verify()
    values: list[str] = []
    for row_index, row in enumerate(_rows(path)):
        value = row.get(source.text_field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"replay history source {path} row {row_index} is missing "
                f"text field {source.text_field!r}"
            )
        values.append(value)
    return tuple(values)


def _write_exact_bytes(path: Path, payload: str) -> None:
    """Write deterministic UTF-8 bytes without platform newline translation."""

    path.write_bytes(payload.encode("utf-8"))


def materialize_replay_history(
    *,
    sources: Iterable[ReplayHistorySource],
    work_dir: str | Path,
    ratio: float,
) -> VerifiedReplayDataset:
    """Build one canonical replay corpus from all prior training sources.

    Materialized replay bytes are identical on Linux and Windows. Text-mode
    writes are intentionally avoided because Python translates ``\n`` to
    ``\r\n`` on Windows, which would make a digest computed from canonical LF
    content disagree with the bytes actually persisted.
    """

    source_rows = tuple(sources)
    if not source_rows:
        raise ValueError("replay history requires at least one source")

    verified_sources: list[dict[str, object]] = []
    unique_texts: list[str] = []
    seen_texts: set[str] = set()
    total_rows = 0
    for source in source_rows:
        texts = _texts(source)
        total_rows += len(texts)
        for text in texts:
            if text in seen_texts:
                continue
            seen_texts.add(text)
            unique_texts.append(text)
        verified_sources.append(
            {
                "role": source.role,
                "source_sha256": source.sha256,
                "text_field": source.text_field,
                "row_count": len(texts),
            }
        )

    if not unique_texts:
        raise ValueError("replay history contains no usable unique text rows")

    identity_payload = {
        "version": 1,
        "sources": [
            {
                "role": row["role"],
                "source_sha256": row["source_sha256"],
                "text_field": row["text_field"],
            }
            for row in verified_sources
        ],
        "unique_text_sha256": [
            hashlib.sha256(text.encode("utf-8")).hexdigest()
            for text in unique_texts
        ],
    }
    history_id = hashlib.sha256(
        _canonical_json(identity_payload).encode("utf-8")
    ).hexdigest()

    root = Path(work_dir).resolve() / ".chowder" / "replay-history"
    root.mkdir(parents=True, exist_ok=True)
    dataset_path = root / f"replay-{history_id[:20]}.jsonl"
    manifest_path = root / f"replay-{history_id[:20]}.manifest.json"

    dataset_text = "".join(
        json.dumps({"text": text}, sort_keys=True, ensure_ascii=False) + "\n"
        for text in unique_texts
    )
    expected_dataset_sha = hashlib.sha256(dataset_text.encode("utf-8")).hexdigest()
    if dataset_path.exists():
        if sha256_file(dataset_path) != expected_dataset_sha:
            raise ValueError("existing replay history file does not match deterministic content")
    else:
        _write_exact_bytes(dataset_path, dataset_text)

    manifest = {
        "version": 1,
        "history_id": history_id,
        "replay_dataset_sha256": expected_dataset_sha,
        "source_row_count": total_rows,
        "unique_row_count": len(unique_texts),
        "sources": verified_sources,
    }
    manifest_text = json.dumps(
        manifest, sort_keys=True, ensure_ascii=False, indent=2
    ) + "\n"
    expected_manifest_sha = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    if manifest_path.exists():
        if sha256_file(manifest_path) != expected_manifest_sha:
            raise ValueError("existing replay history manifest does not match deterministic content")
    else:
        _write_exact_bytes(manifest_path, manifest_text)

    replay = VerifiedReplayDataset(
        path=str(dataset_path),
        sha256=expected_dataset_sha,
        ratio=float(ratio),
        manifest_path=str(manifest_path),
        manifest_sha256=expected_manifest_sha,
    )
    replay.verify()
    return replay
