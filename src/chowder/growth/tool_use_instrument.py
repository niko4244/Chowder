"""The ``tool-use@reap-moe-v1`` instrument, in production.

The campaign's declared target instrument is a tool-use benchmark: each item is
a frozen conversation context plus the tool call the *next* assistant turn must
make.  ``reap-moe-v1`` measures whether an expert-pruned MoE still selects the
right tool with the exact arguments, so the item carries the tool schemas the
arguments must satisfy -- a call with the right name and the wrong argument type
is not the same call.

Like the generation-diagnostics instrument, the items live in-repo rather than
in a cache the run would have to fetch: the items file is generated once by
``chowder_batch/build_tool_use_instrument.py`` and committed, and this module is
the production owner that a campaign reads them through.  It refuses to load
rather than run when the committed bytes do not match the manifest's digest --
a slice that silently changed is not the slice the run declared.

The held-out rule is the contamination boundary: the public
``is_reserved_for_evaluation`` predicate is applied to the *source* rows when
the instrument is built, so an item a protected evaluation measures can never be
lifted into it.  The rule is mirrored here, in the open, exactly as the builder
applies it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..evaluators.scoring import tool_schema_for

#: The qualified id a campaign declares to measure this instrument.
INSTRUMENT_ID = "tool_use@reap-moe-v1"

#: The committed instrument files, both under :data:`INSTRUMENTS_DIR`.
ITEMS_NAME = "tool_use_reap_moe_v1.jsonl"
MANIFEST_NAME = "tool_use_reap_moe_v1.manifest.json"

#: Where the committed items and manifest live.
INSTRUMENTS_DIR = Path(__file__).resolve().parent / "instruments"

#: The held-out rule, as source text. The builder drops every source row whose
#: uuid satisfies this, and :func:`is_reserved_for_evaluation` is the same
#: predicate so a reader can check a uuid without reading the builder.
HELD_OUT_RULE = (
    "A source row is reserved for evaluation -- and never enters the "
    "instrument -- when "
    "int(hashlib.sha256(row['uuid'].encode()).hexdigest()[:8], 16) % 10 == 0, "
    "i.e. the first eight hex digits of the uuid's sha256 are divisible by ten. "
    "One row in ten is held out on a hashed uuid, so the split is reproducible "
    "and independent of dataset order."
)


class ToolUseInstrumentRefusal(RuntimeError):
    """The instrument cannot be read as declared.

    Raised with a stable machine identifier so preparation and CI act on the
    refusal rather than parsing prose; never converted into a placeholder slice.
    """


def is_reserved_for_evaluation(uuid: str) -> bool:
    """Whether a source row's uuid is held out of the instrument (see
    :data:`HELD_OUT_RULE`).  The one definition the builder and any reader share.
    """
    digest = hashlib.sha256(str(uuid).encode()).hexdigest()
    return int(digest[:8], 16) % 10 == 0


def items_path() -> Path:
    return INSTRUMENTS_DIR / ITEMS_NAME


def manifest_path() -> Path:
    return INSTRUMENTS_DIR / MANIFEST_NAME


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest() -> Mapping[str, Any]:
    """The committed manifest, or a refusal when it is absent or not an object."""
    path = manifest_path()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ToolUseInstrumentRefusal(
            f"TOOL_USE_INSTRUMENT_UNAVAILABLE: no manifest at {path}"
        ) from error
    except ValueError as error:
        raise ToolUseInstrumentRefusal(
            f"TOOL_USE_INSTRUMENT_INVALID: manifest at {path} is not JSON: {error}"
        ) from error
    if not isinstance(document, Mapping):
        raise ToolUseInstrumentRefusal(
            f"TOOL_USE_INSTRUMENT_INVALID: manifest at {path} is not an object"
        )
    return document


def load_items() -> tuple[Mapping[str, Any], ...]:
    """The committed instrument items, verified against the manifest digest.

    The digest is checked before a single item is parsed: a slice whose bytes do
    not match what the manifest pinned is a different slice, and measuring it
    under this instrument's name would bind evidence to the wrong bytes.
    """
    manifest = load_manifest()
    path = items_path()
    try:
        raw = path.read_bytes()
    except FileNotFoundError as error:
        raise ToolUseInstrumentRefusal(
            f"TOOL_USE_INSTRUMENT_UNAVAILABLE: no items file at {path}"
        ) from error
    expected_digest = manifest.get("items_sha256")
    actual_digest = _sha256_bytes(raw)
    if expected_digest != actual_digest:
        raise ToolUseInstrumentRefusal(
            f"TOOL_USE_INSTRUMENT_DIGEST_MISMATCH: {path} hashes to "
            f"{actual_digest} but the manifest pins {expected_digest}"
        )
    items: list[Mapping[str, Any]] = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError as error:
            raise ToolUseInstrumentRefusal(
                f"TOOL_USE_INSTRUMENT_INVALID: {path}:{number} is not JSON: {error}"
            ) from error
        if not isinstance(item, Mapping):
            raise ToolUseInstrumentRefusal(
                f"TOOL_USE_INSTRUMENT_INVALID: {path}:{number} is not an object"
            )
        items.append(item)
    return tuple(items)


def _slice_item(item: Mapping[str, Any]) -> Mapping[str, str]:
    """One instrument item as the ``(prompt, expected)`` the runtime consumes.

    ``prompt`` is the frozen context -- the offered tool schemas and the
    conversation before the assistant turn -- as one JSON string.  ``expected``
    is JSON because the two-string :func:`chowder.evaluators.scoring.score`
    signature has nowhere else to put the tool's ``parameters`` schema; the
    committed item keeps ``expected`` as the bare ``{name, arguments}`` call and
    the schema is added here, from the same item's ``tools``.
    """
    expected = item.get("expected", {})
    expected = expected if isinstance(expected, Mapping) else {}
    name = str(expected.get("name", ""))
    schema = tool_schema_for(item.get("tools", ()), name)
    return {
        "prompt": json.dumps(
            {"tools": item.get("tools", []), "messages": item.get("messages", [])},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "expected": json.dumps(
            {
                "name": name,
                "arguments": expected.get("arguments", {}),
                "parameters": schema if schema is not None else {},
            },
            sort_keys=True,
        ),
    }


def load_tool_use_slice() -> tuple[Mapping[str, str], ...]:
    """The instrument's items in the slice shape :func:`load_pinned_slices`
    returns: ``(prompt, expected)`` mappings, in the committed file's order.
    """
    return tuple(_slice_item(item) for item in load_items())
