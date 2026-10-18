"""Agentic tool-use SFT providers: local Fable rows, and NVIDIA Nemotron-Agentic.

Two providers over frozen, digest-pinned agentic corpora, in the discipline
``ornith_provider.py`` established:

* ``FableAgenticProvider`` -- the owner-approved Fable agentic SFT sets
  (``glint_local``, ``lordx64``, ``swarm``).  Fable output is AGPL and its use
  is a licence decision the owner makes explicitly, so construction refuses
  unless ``allow_fable=True``; every row it serves is marked
  ``fable_derived: true`` in its provenance.
* ``NemotronAgenticProvider`` -- NVIDIA Nemotron-Agentic-v1 (CC BY 4.0).  Every
  row the evaluation split reserves is excluded, so no eval row can be trained
  on even by accident.

Digest discipline, serving discipline and provenance discipline are shared:
each source's sha256 is checked at construction and again on every ``produce``
(a file edited in between refuses); a row is served at most once; each served
example carries per-example provenance.  Protected-eval and contamination
guards are *not* re-implemented here -- they live in ``materialise`` and apply
to these lines exactly as to the built-ins.

The three normalisations that turn agentic rows into messages Qwen's template
accepts -- JSON-string tool arguments becoming dicts, lordx64's pre-rendered
``<tool_use>`` XML becoming native tool calls, and the local Fable ``target``
turn being appended -- are copied verbatim from
``chowder_batch/reap_calibration.py`` (copied, not imported: that module is a
read-only reference, and importing a batch script would drag its ``main``
along).

Line shape
----------
``OrnithCurriculumProvider`` emits ``{"text": ...}`` -- a flattened chat
rendering.  A tool-use turn cannot be flattened to text without losing the call
structure, so these providers emit the ``messages`` variant instead:
``{"messages": [...]}``.  The worker's ``dataset_format: "chat"`` path cannot
take a ``tools`` key, so none is emitted: the tools block is *also* written into
the leading system turn, exactly as the Qwen chat template renders it
(``chat_template.jinja`` lines 45-60), so the model still sees the schema
without a key the worker would never read.

``transformers_worker``'s ``dataset_format: "chat"`` path is *not* a fully
compatible consumer of these rows either: it reads only ``messages_field``
(``transformers_worker.py:651,659`` -- ``select_columns([field])``), and
``_validate_chat_messages`` accepts only ``system``/``user``/``assistant`` and
keeps only ``role``/``content`` (``training_data.py:67-95``).  A served row
carries ``tool`` turns and native ``tool_calls``, so the worker's chat path
rejects the former and drops the latter -- the rows are consumed as text or by
a tools-aware renderer, never by that path as-is.  This boundary is pinned by
``tests/test_agentic_providers.py::test_worker_chat_path_rejects_tool_turns``.

Neither provider is in ``DEFAULT_PROVIDERS``: a campaign passes them explicitly
per campaign, exactly as it passes Ornith's.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from chowder.growth.data_providers import (
    PROVIDER_SCHEMA,
    TrainingDataRefusal,
    TrainingDataRequest,
)

AGENTIC_SCHEMA = f"{PROVIDER_SCHEMA}/AGENTIC"
FABLE_SCHEMA = f"{AGENTIC_SCHEMA}/FABLE"
NEMOTRON_SCHEMA = f"{AGENTIC_SCHEMA}/NEMOTRON"

#: Deterministic serve order seed, the calibration set's own seed.
SEED = 20261017

#: Agentic rows are SFT material, not preference pairs.
SUPPORTED_TRAINING_TYPES = frozenset({"sft", "targeted_repair"})

#: External curation (LICENSE + dataset cards) rather than an executed verifier.
VERIFICATION = "curated_trusted"

#: Fable source kinds, and the dedupe order the calibration set used: the local
#: rows seed the seen-set, the two downloads are checked against it in turn.
FABLE_SOURCE_ORDER = ("glint_local", "lordx64", "swarm")

#: Nemotron-Agentic-v1's licence, recorded per served row.
NEMOTRON_LICENCE = "CC-BY-4.0"

#: A fingerprint is an assistant text long enough to identify a turn across
#: repackagings; only the first 2,000 characters are hashed.
_FINGERPRINT_MIN_CHARS = 80
_FINGERPRINT_CHARS = 2000


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_reserved_for_evaluation(uuid: str) -> bool:
    """Whether a Nemotron row's uuid belongs to the reserved evaluation split.

    The exact rule the evaluation instrument pins, and the same function/name a
    parallel task adds there; keep the formula character-for-character identical.
    """
    return int(hashlib.sha256(uuid.encode()).hexdigest()[:8], 16) % 10 == 0


# --------------------------------------------------------------------------- #
# Normalisations copied verbatim from chowder_batch/reap_calibration.py.
# --------------------------------------------------------------------------- #

_TOOL_USE = re.compile(
    r'<tool_use name="([^"]+)"(?: id="[^"]*")?>\s*(.*?)\s*</tool_use>', re.DOTALL
)
_TURN = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.DOTALL)


def _native_tool_calls(messages: list[dict]) -> list[dict]:
    """Tool-call arguments as dicts, the shape Qwen's template iterates."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for call in m["tool_calls"]:
                fn = dict(call.get("function") or {})
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except json.JSONDecodeError:
                        fn["arguments"] = {"input": fn["arguments"]}
                calls.append({**call, "function": fn})
            m["tool_calls"] = calls
        out.append(m)
    return out


def lordx64_messages(text: str) -> list[dict]:
    """A pre-rendered lordx64 row back into messages, its XML tool_use as native tool calls."""
    messages = []
    for role, body in _TURN.findall(text):
        if role != "assistant":
            messages.append({"role": role, "content": body.strip()})
            continue
        calls = []
        for name, raw in _TOOL_USE.findall(body):
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                args = {"input": raw}
            calls.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args if isinstance(args, dict) else {"input": args},
                    },
                }
            )
        turn = {"role": "assistant", "content": _TOOL_USE.sub("", body).strip()}
        if calls:
            turn["tool_calls"] = calls
        messages.append(turn)
    return messages


def _fable_messages(row: dict) -> list[dict]:
    """The logged context plus the Fable ``target`` turn as the final assistant turn."""
    target = row["target"]
    turn = {"role": "assistant", "content": target.get("content") or ""}
    if target.get("reasoning"):
        turn["reasoning_content"] = target["reasoning"]
    if target.get("tool_calls"):
        turn["tool_calls"] = target["tool_calls"]
    return [*row["messages"], turn]


def _fingerprints(messages: list[dict]) -> set[str]:
    """Normalised assistant texts long enough to identify a turn across repackagings."""
    prints = set()
    for m in messages:
        if m.get("role") != "assistant":
            continue
        # Tool-call arguments too: a repackaged agent turn is often a short preamble
        # plus a call, and the call is what identifies it.
        calls = [
            json.dumps((c.get("function") or {}).get("arguments"), sort_keys=True)
            for c in m.get("tool_calls") or []
        ]
        for text in (m.get("content"), m.get("reasoning_content"), m.get("reasoning"), *calls):
            norm = re.sub(r"\s+", " ", str(text or "")).strip().lower()
            if len(norm) >= _FINGERPRINT_MIN_CHARS:
                prints.add(hashlib.sha1(norm[:_FINGERPRINT_CHARS].encode("utf-8")).hexdigest())
    return prints


# --------------------------------------------------------------------------- #
# The Qwen tools block, as chat_template.jinja renders it (lines 45-60).
# --------------------------------------------------------------------------- #

_TOOLS_BLOCK_HEAD = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
_TOOLS_BLOCK_TAIL = (
    "\n</tools>"
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:"
    "\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
    "value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the "
    "second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>"
    "\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your "
    "function call in natural language BEFORE the function call, but NOT after\n- If there is "
    "no function call available, answer the question like normal with your current knowledge "
    "and do not tell the user about function calls\n</IMPORTANT>"
)


def _tools_system_block(tools: Sequence[Mapping[str, Any]]) -> str:
    """The tools list rendered the way Qwen's template renders it."""
    rendered = "\n".join(json.dumps(tool, ensure_ascii=False) for tool in tools)
    return _TOOLS_BLOCK_HEAD + "\n" + rendered + _TOOLS_BLOCK_TAIL


def _with_tools_system(
    messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] | None
) -> list[dict[str, Any]]:
    """Fold the tools block into the leading system turn, as the template would.

    ``transformers_worker``'s chat path never passes ``tools`` to
    ``apply_chat_template``, so without this the served rows would show the model
    no tool schemas at all.  Mirrors the template's own branch: an existing
    system turn keeps its content after the block, and a synthetic one is
    prepended when there is none.
    """
    if not tools:
        return [dict(m) for m in messages]
    block = _tools_system_block(tools)
    out = [dict(m) for m in messages]
    if out and out[0].get("role") == "system":
        content = str(out[0].get("content") or "")
        out[0]["content"] = block + ("\n\n" + content if content else "")
    else:
        out.insert(0, {"role": "system", "content": block})
    return out


def _agentic_line(
    messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] | None
) -> str:
    """One training line: the ``messages`` variant Ornith's ``text`` shape cannot carry.

    ``messages`` only: the worker's chat path cannot take a ``tools`` key, so the
    tools are folded into the leading system turn instead (see the module docstring).
    """
    return json.dumps(
        {"messages": _with_tools_system(messages, tools)}, sort_keys=True
    )


class _DigestPinnedSources:
    """Shared digest + serve-once machinery for the agentic providers."""

    #: Refusal-code prefix; each concrete provider names its own.
    _schema = AGENTIC_SCHEMA

    def __init__(self) -> None:
        self._paths: dict[str, Path] = {}
        self._digests: dict[str, str] = {}
        self._rows: list[dict[str, Any]] = []
        self._cursor = 0
        #: line digest -> per-example provenance, filled as lines are produced.
        self.provenance: dict[str, dict[str, Any]] = {}

    # -- digests ---------------------------------------------------------- #

    def _pin(self, name: str, path: Path, expected_sha256: str) -> None:
        self._paths[name] = path
        self._digests[name] = str(expected_sha256).strip().lower()
        self._check_digest(name)

    def _check_digest(self, name: str) -> None:
        path = self._paths[name]
        pinned = self._digests[name]
        actual = _sha256_file(path)
        if actual != pinned:
            raise TrainingDataRefusal(
                f"{self._schema}: {path} has sha256 digest {actual}, not the pinned "
                f"{pinned}; only the frozen {name} file may be served"
            )

    def _check_all_digests(self) -> None:
        for name in self._paths:
            self._check_digest(name)

    # -- serving ---------------------------------------------------------- #

    def supports(self, request: TrainingDataRequest) -> bool:
        return (
            request.skill == self.skill
            and request.verification_method == self.verification
            and request.training_type in SUPPORTED_TRAINING_TYPES
            and len(self._rows) - self._cursor >= request.example_count
        )

    def produce(self, request: TrainingDataRequest) -> Sequence[str]:
        self._check_all_digests()
        remaining = len(self._rows) - self._cursor
        if request.example_count > remaining:
            raise TrainingDataRefusal(
                f"{self._schema}: {request.item_id!r} asks for "
                f"{request.example_count} examples but only {remaining} unserved "
                "rows remain; rows are never reused"
            )
        rows = self._rows[self._cursor : self._cursor + request.example_count]
        self._cursor += request.example_count
        lines: list[str] = []
        for row in rows:
            line = _agentic_line(row["messages"], row["tools"])
            self.provenance[hashlib.sha256(line.encode("utf-8")).hexdigest()] = dict(
                row["provenance"], item_id=request.item_id
            )
            lines.append(line)
        return lines


class FableAgenticProvider(_DigestPinnedSources):
    """Serves one declared skill from pinned Fable agentic SFT sources."""

    provider_id = "fable-agentic"
    verification = VERIFICATION
    _schema = FABLE_SCHEMA

    def __init__(
        self,
        sources: Mapping[str, tuple[str, str]],
        *,
        skill: str,
        allow_fable: bool,
    ) -> None:
        if not allow_fable:
            raise TrainingDataRefusal(
                f"{FABLE_SCHEMA}: Fable-derived text is AGPL and its use is the "
                "owner's explicit licence decision (Anthropic usage terms); build "
                "this provider with allow_fable=True only after that decision"
            )
        super().__init__()
        self.skill = str(skill)
        self.allow_fable = True
        self.source_id = "fable-agentic:" + hashlib.sha256(
            "|".join(f"{n}:{s}" for n, (_, s) in sorted(sources.items())).encode("utf-8")
        ).hexdigest()[:16]
        ordered = [name for name in FABLE_SOURCE_ORDER if name in sources]
        unknown = sorted(set(sources) - set(FABLE_SOURCE_ORDER))
        if unknown:
            raise TrainingDataRefusal(
                f"{FABLE_SCHEMA}: unknown source kind(s) {unknown}; supported "
                f"kinds are {sorted(FABLE_SOURCE_ORDER)}"
            )
        if not ordered:
            raise TrainingDataRefusal(
                f"{FABLE_SCHEMA}: no Fable sources supplied; supported kinds are "
                f"{sorted(FABLE_SOURCE_ORDER)}"
            )
        #: name -> duplicate rows dropped against earlier sources.
        self.dropped_duplicates: dict[str, int] = {}
        for name in ordered:
            path, sha = sources[name]
            self._pin(name, Path(path), sha)
        self._rows = self._load_deduped(ordered)
        random.Random(SEED).shuffle(self._rows)
        self._cursor = 0

    def _load_deduped(self, ordered: Sequence[str]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for name in ordered:
            path = self._paths[name]
            sha = self._digests[name]
            loaded = self._load_source(name, path, sha)
            for row in loaded:
                prints = _fingerprints(row["messages"])
                # The local rows seed the seen-set; the downloads are checked
                # against it, and against each other, in order.
                if name != "glint_local" and prints and 2 * len(prints & seen) >= len(prints):
                    self.dropped_duplicates[name] = self.dropped_duplicates.get(name, 0) + 1
                    continue
                seen |= prints
                rows.append(row)
        for name in ordered:
            self.dropped_duplicates.setdefault(name, 0)
        return rows

    def _load_source(self, name: str, path: Path, sha: str) -> list[dict[str, Any]]:
        if name == "glint_local":
            return [
                self._row(index, sha, name, _native_tool_calls(_fable_messages(row)),
                          row.get("tools") or None)
                for index, row in self._jsonl_rows(path, name)
            ]
        if name == "swarm":
            return [
                self._row(index, sha, name, _native_tool_calls(row["messages"]),
                          row.get("tools") or None)
                for index, row in self._jsonl_rows(path, name)
            ]
        if name == "lordx64":
            return [
                self._row(index, sha, name, lordx64_messages(row["text"]), None)
                for index, row in self._parquet_rows(path, name)
            ]
        raise TrainingDataRefusal(f"{FABLE_SCHEMA}: unknown source kind {name!r}")

    @staticmethod
    def _row(
        index: int,
        sha: str,
        name: str,
        messages: list[dict],
        tools: list | None,
    ) -> dict[str, Any]:
        return {
            "messages": messages,
            "tools": tools,
            "provenance": {
                "fable_derived": True,
                "source": name,
                "row_index": index,
                "source_sha256": sha,
            },
        }

    def _jsonl_rows(self, path: Path, name: str) -> list[tuple[int, dict]]:
        rows: list[tuple[int, dict]] = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingDataRefusal(
                    f"{FABLE_SCHEMA}: {path}:{number} is not JSON ({exc})"
                ) from exc
            if not isinstance(row, dict):
                raise TrainingDataRefusal(
                    f"{FABLE_SCHEMA}: {path}:{number} is not a JSON object"
                )
            rows.append((number - 1, row))
        return rows

    def _parquet_rows(self, path: Path, name: str) -> list[tuple[int, dict]]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - pyarrow ships with the data
            raise TrainingDataRefusal(
                f"{FABLE_SCHEMA}: source {name!r} is a parquet file but pyarrow is "
                f"not importable ({exc})"
            ) from exc
        return list(enumerate(pq.read_table(path).to_pylist()))


class NemotronAgenticProvider(_DigestPinnedSources):
    """Serves one declared skill from a pinned Nemotron-Agentic JSONL, minus evals."""

    provider_id = "nemotron-agentic"
    verification = VERIFICATION
    _schema = NEMOTRON_SCHEMA

    def __init__(self, path: str, expected_sha256: str, *, skill: str) -> None:
        super().__init__()
        self.skill = str(skill)
        self._pin("nemotron", Path(path), expected_sha256)
        self.source_id = f"nemotron-agentic:{self._digests['nemotron'][:16]}"
        #: rows excluded because the evaluation split reserves them.
        self.reserved_excluded = 0
        self._rows = self._load(self._paths["nemotron"])
        random.Random(SEED).shuffle(self._rows)
        self._cursor = 0

    def _load(self, path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingDataRefusal(
                    f"{NEMOTRON_SCHEMA}: {path}:{number} is not JSON ({exc})"
                ) from exc
            if not isinstance(row, dict):
                raise TrainingDataRefusal(
                    f"{NEMOTRON_SCHEMA}: {path}:{number} is not a JSON object"
                )
            uuid = row.get("uuid")
            if not isinstance(uuid, str) or not uuid.strip():
                raise TrainingDataRefusal(
                    f"{NEMOTRON_SCHEMA}: {path}:{number} has no uuid, so its "
                    "evaluation reservation cannot be decided"
                )
            if is_reserved_for_evaluation(uuid):
                self.reserved_excluded += 1
                continue
            rows.append(
                {
                    "messages": _native_tool_calls(row["messages"]),
                    "tools": row.get("tools") or None,
                    "provenance": {
                        "uuid": uuid,
                        "licence": NEMOTRON_LICENCE,
                        "row_index": number - 1,
                        "source_sha256": self._digests["nemotron"],
                    },
                }
            )
        return rows
