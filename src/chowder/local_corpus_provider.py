from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .repair_requests import RepairRequest, RepairSourceProposal
from .repair_sources import RepairSource, SourcedRepairExample


@dataclass(frozen=True)
class LocalCorpusFile:
    path: str
    logical_name: str | None = None

    def resolved_path(self) -> Path:
        path = Path(self.path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"repair corpus not found: {path}")
        return path


@dataclass(frozen=True)
class _CorpusRow:
    source: RepairSource
    example: SourcedRepairExample
    suite: str
    strategies: tuple[str, ...]
    failure_kinds: tuple[str, ...]
    priority: float


class LocalCorpusRepairProvider:
    """Deterministic repair-source provider backed by local JSONL corpora.

    The provider receives only ``RepairRequest`` aggregate metadata. It never
    receives the failed benchmark rows. Corpus selection is therefore based on
    declared suite/failure class/repair strategy rather than prompt similarity.

    Each JSONL row requires ``prompt`` and ``expected`` and may define:

    - ``example_id``: stable row identifier (otherwise row number is used)
    - ``suite``: exact suite name or ``*`` (default ``*``)
    - ``strategy`` / ``strategies``: repair strategy names or ``*``
    - ``failure_kind`` / ``failure_kinds``: optional failure-class filter
    - ``priority``: finite numeric deterministic tie-break priority
    """

    name = "local-corpus"
    version = "1"

    def __init__(
        self,
        corpus_files: Iterable[str | Path | LocalCorpusFile],
        *,
        max_examples: int = 32,
        min_examples: int = 1,
        examples_per_failure: int = 2,
    ) -> None:
        files: list[LocalCorpusFile] = []
        for item in corpus_files:
            if isinstance(item, LocalCorpusFile):
                files.append(item)
            else:
                files.append(LocalCorpusFile(str(item)))
        if not files:
            raise ValueError("local corpus provider requires at least one corpus file")
        if max_examples <= 0:
            raise ValueError("max_examples must be positive")
        if min_examples <= 0 or min_examples > max_examples:
            raise ValueError("min_examples must be positive and <= max_examples")
        if examples_per_failure <= 0:
            raise ValueError("examples_per_failure must be positive")
        self._files = tuple(files)
        self._max_examples = int(max_examples)
        self._min_examples = int(min_examples)
        self._examples_per_failure = int(examples_per_failure)

    @staticmethod
    def _string_set(row: Mapping[str, object], singular: str, plural: str) -> tuple[str, ...]:
        raw = row.get(plural, row.get(singular, "*"))
        if isinstance(raw, str):
            values = (raw,)
        elif isinstance(raw, (list, tuple)):
            values = tuple(str(value) for value in raw)
        else:
            raise ValueError(f"corpus field {plural!r}/{singular!r} must be a string or list")
        normalized = tuple(sorted({value.strip() for value in values if value.strip()}))
        if not normalized:
            raise ValueError(f"corpus field {plural!r}/{singular!r} cannot be empty")
        return normalized

    def _load_file(self, spec: LocalCorpusFile) -> tuple[_CorpusRow, ...]:
        path = spec.resolved_path()
        raw_bytes = path.read_bytes()
        digest = hashlib.sha256(raw_bytes).hexdigest()
        logical_name = (spec.logical_name or path.name).strip()
        if not logical_name:
            raise ValueError("local corpus logical_name cannot be empty")
        source_identity = hashlib.sha256(
            (logical_name + "\x00" + digest).encode("utf-8")
        ).hexdigest()
        source_id = f"local-{source_identity[:24]}"
        source = RepairSource(
            source_id=source_id,
            ref=f"local-corpus://{logical_name}@sha256:{digest}",
            content_sha256=digest,
            kind="local_corpus",
        )

        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"repair corpus is not UTF-8: {path}") from exc

        rows: list[_CorpusRow] = []
        seen_row_ids: set[str] = set()
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            parsed = json.loads(line)
            if not isinstance(parsed, Mapping):
                raise ValueError(f"{path}:{line_number} corpus row is not an object")
            prompt = parsed.get("prompt")
            expected = parsed.get("expected")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{line_number} corpus prompt is required")
            if not isinstance(expected, str) or not expected.strip():
                raise ValueError(f"{path}:{line_number} corpus expected answer is required")
            row_id = str(parsed.get("example_id", f"row-{line_number}")).strip()
            if not row_id:
                raise ValueError(f"{path}:{line_number} example_id cannot be empty")
            if row_id in seen_row_ids:
                raise ValueError(f"{path} contains duplicate example_id {row_id!r}")
            seen_row_ids.add(row_id)
            suite = str(parsed.get("suite", "*")).strip() or "*"
            strategies = self._string_set(parsed, "strategy", "strategies")
            failure_kinds = self._string_set(parsed, "failure_kind", "failure_kinds")
            priority_raw = parsed.get("priority", 0.0)
            if isinstance(priority_raw, bool) or not isinstance(priority_raw, (int, float)):
                raise ValueError(f"{path}:{line_number} priority must be numeric")
            priority = float(priority_raw)
            if not math.isfinite(priority):
                raise ValueError(f"{path}:{line_number} priority must be finite")
            example_id = f"{source_id}:{row_id}"
            rows.append(
                _CorpusRow(
                    source=source,
                    example=SourcedRepairExample(
                        example_id=example_id,
                        source_id=source_id,
                        prompt=prompt,
                        expected=expected,
                    ),
                    suite=suite,
                    strategies=strategies,
                    failure_kinds=failure_kinds,
                    priority=priority,
                )
            )
        if not rows:
            raise ValueError(f"repair corpus is empty: {path}")
        return tuple(rows)

    @staticmethod
    def _eligible(row: _CorpusRow, request: RepairRequest) -> bool:
        suite_ok = row.suite in {"*", request.suite}
        strategy_ok = "*" in row.strategies or request.strategy.value in row.strategies
        failure_ok = "*" in row.failure_kinds or request.failure_kind in row.failure_kinds
        return suite_ok and strategy_ok and failure_ok

    @staticmethod
    def _rank(row: _CorpusRow, request: RepairRequest) -> tuple[float, int, str, str]:
        specificity = 0
        specificity += 4 if row.suite == request.suite else 0
        specificity += 2 if request.strategy.value in row.strategies else 0
        specificity += 1 if request.failure_kind in row.failure_kinds else 0
        return (-row.priority, -specificity, row.source.source_id, row.example.example_id)

    def propose(self, request: RepairRequest) -> RepairSourceProposal:
        if not isinstance(request, RepairRequest):
            raise TypeError("LocalCorpusRepairProvider requires a RepairRequest")

        rows: list[_CorpusRow] = []
        for corpus_file in self._files:
            rows.extend(self._load_file(corpus_file))
        eligible = [row for row in rows if self._eligible(row, request)]
        eligible.sort(key=lambda row: self._rank(row, request))

        target = min(
            self._max_examples,
            max(self._min_examples, request.failure_count * self._examples_per_failure),
        )
        selected = eligible[:target]
        if len(selected) < self._min_examples:
            raise ValueError(
                "local repair corpus does not contain enough eligible examples "
                f"for suite={request.suite!r} strategy={request.strategy.value!r}"
            )

        sources_by_id = {row.source.source_id: row.source for row in selected}
        examples = tuple(row.example for row in selected)
        return RepairSourceProposal(
            request_id=request.request_id,
            provider_name=self.name,
            provider_version=self.version,
            sources=tuple(sources_by_id[key] for key in sorted(sources_by_id)),
            examples=examples,
        )
