"""A training-data provider over a frozen, digest-pinned Ornith admitted JSONL.

Ornith (``frontier-lowram-autoresearch/hermes``) proposes curriculum tasks and
admits them through its V x D x N gates, appending each admitted row to
``admitted.jsonl``.  This provider turns such a file into training lines for
``materialise`` -- and only that file, at exactly the bytes pinned by sha256 when
the provider was built.  It is deliberately *not* in ``DEFAULT_PROVIDERS``: a
prepare that wants it passes it via ``materialise(providers=...)``, so no
existing campaign declaration changes.

Fail-closed rules:

* the file's sha256 must equal the pinned digest, at construction and again on
  every ``produce`` (a file edited in between refuses);
* any row whose provenance (every field except the task text itself) mentions
  Fable refuses the whole file -- Fable-derived text is never training material;
* only admitted, valid, SFT-lane rows are served; ``executable_tests`` also
  requires the row's ``verification_code``;
* a row is served at most once, so two items never train on the same task.

Protected-eval and contamination guards are not re-implemented here: they live
in ``materialise`` and apply to this provider's lines exactly as to the
built-ins.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from chowder.growth.data_providers import (
    PROVIDER_SCHEMA,
    TrainingDataRefusal,
    TrainingDataRequest,
    _chat_line,
)

ORNITH_SCHEMA = f"{PROVIDER_SCHEMA}/ORNITH"

#: Verifications an Ornith admitted row can honestly carry: its admission ran
#: the row's verification code, or it was cross-checked by independent models.
SUPPORTED_VERIFICATIONS = frozenset({"executable_tests", "multi_judge"})

#: Ornith rows are prompt/answer pairs: SFT material, not preference pairs.
SUPPORTED_TRAINING_TYPES = frozenset({"sft", "targeted_repair"})

#: The task text itself; every other field is provenance and is scanned.
_TASK_TEXT_FIELDS = frozenset({"prompt", "expected_answer", "verification_code"})


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mentions_fable(value: Any) -> bool:
    if isinstance(value, str):
        return "fable" in value.lower()
    if isinstance(value, Mapping):
        return any(_mentions_fable(k) or _mentions_fable(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_mentions_fable(v) for v in value)
    return False


def _field(row: Mapping[str, Any], name: str) -> Any:
    """A proposal field, top-level or nested under ``task`` (TaskProposal)."""
    if name in row:
        return row[name]
    task = row.get("task")
    return task.get(name) if isinstance(task, Mapping) else None


class OrnithCurriculumProvider:
    """Serves one declared skill from a pinned Ornith ``admitted.jsonl``."""

    provider_id = "ornith-curriculum"

    def __init__(
        self,
        admitted_path: str | Path,
        expected_sha256: str,
        *,
        skill: str,
        verification: str,
    ) -> None:
        if verification not in SUPPORTED_VERIFICATIONS:
            raise TrainingDataRefusal(
                f"{ORNITH_SCHEMA}: verification {verification!r} is not one an "
                f"Ornith admitted row carries ({sorted(SUPPORTED_VERIFICATIONS)})"
            )
        self._path = Path(admitted_path)
        self._sha = str(expected_sha256).strip().lower()
        self._check_digest()
        self.skill = str(skill)
        self.verification = verification
        self.source_id = f"ornith-admitted:{self._sha[:16]}"
        self._rows = self._load()
        self._cursor = 0
        #: line digest -> per-example provenance, filled as lines are produced.
        self.provenance: dict[str, dict[str, Any]] = {}

    @property
    def served_task_ids(self) -> tuple[str, ...]:
        return tuple(str(_field(row, "task_id")) for row in self._rows)

    def _check_digest(self) -> None:
        actual = _sha256_file(self._path)
        if actual != self._sha:
            raise TrainingDataRefusal(
                f"{ORNITH_SCHEMA}: {self._path} has sha256 digest {actual}, not the "
                f"pinned {self._sha}; only the frozen admitted file may be served"
            )

    def _load(self) -> list[dict[str, Any]]:
        served: list[dict[str, Any]] = []
        lines = self._path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingDataRefusal(
                    f"{ORNITH_SCHEMA}: {self._path}:{number} is not JSON ({exc})"
                ) from exc
            if not isinstance(row, dict):
                raise TrainingDataRefusal(
                    f"{ORNITH_SCHEMA}: {self._path}:{number} is not a JSON object"
                )
            provenance = {k: v for k, v in row.items() if k not in _TASK_TEXT_FIELDS}
            if _mentions_fable(provenance):
                raise TrainingDataRefusal(
                    f"{ORNITH_SCHEMA}: {self._path}:{number} (task "
                    f"{_field(row, 'task_id')!r}) has Fable in its provenance; "
                    "Fable-derived text is never training material"
                )
            if self._servable(row):
                served.append(row)
        return served

    def _servable(self, row: Mapping[str, Any]) -> bool:
        validity = row.get("validity")
        if row.get("admitted") is not True or row.get("lane") != "sft":
            return False
        if not (isinstance(validity, Mapping) and validity.get("valid") is True):
            return False
        prompt, answer = _field(row, "prompt"), _field(row, "expected_answer")
        if not (isinstance(prompt, str) and prompt.strip()):
            return False
        if not (isinstance(answer, str) and answer.strip()):
            return False
        if self.verification == "executable_tests":
            code = _field(row, "verification_code")
            return isinstance(code, str) and bool(code.strip())
        return True

    def supports(self, request: TrainingDataRequest) -> bool:
        return (
            request.skill == self.skill
            and request.verification_method == self.verification
            and request.training_type in SUPPORTED_TRAINING_TYPES
            and len(self._rows) - self._cursor >= request.example_count
        )

    def produce(self, request: TrainingDataRequest) -> Sequence[str]:
        self._check_digest()
        remaining = len(self._rows) - self._cursor
        if request.example_count > remaining:
            raise TrainingDataRefusal(
                f"{ORNITH_SCHEMA}: {request.item_id!r} asks for "
                f"{request.example_count} examples but only {remaining} unserved "
                "admitted rows remain; rows are never reused"
            )
        rows = self._rows[self._cursor : self._cursor + request.example_count]
        self._cursor += request.example_count
        lines: list[str] = []
        for row in rows:
            line = _chat_line(str(_field(row, "prompt")), str(_field(row, "expected_answer")))
            self.provenance[hashlib.sha256(line.encode("utf-8")).hexdigest()] = {
                "item_id": request.item_id,
                "task_id": _field(row, "task_id"),
                "family": _field(row, "family"),
                "proposal_sha256": _field(row, "proposal_sha256"),
                "lane": row.get("lane"),
                "admitted_at": row.get("_admitted_at"),
                "admitted_path": str(self._path),
                "admitted_sha256": self._sha,
                "verification": self.verification,
            }
            lines.append(line)
        return lines
