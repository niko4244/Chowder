"""The scoring implementation's identity, bindable into a protocol fingerprint.

The 2026-09-12 audit found that the evaluation protocol fingerprint pins the
scoring *name* (`"scoring": "final_number_match"`) and dependency versions but
not the scoring *implementation*. The historical lenient rule and the current
strict rule could share a nominal protocol identity if every other payload
field happened to match, because the rule itself changed inside the same
scorer name — exactly the identity gap P4 exists to close.

Design: the content hash IS the version. No invented version numbers, no
changelog to drift out of date — a byte of change in `scoring.py` changes the
digest, and two different contents can never share an identity. The functions
exposed by the module are recorded so a reader can see *what* was hashed
without opening the file.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

#: The module whose implementation is the scoring rule. Hashed as source bytes.
SCORER_MODULE = "chowder.evaluators.scoring"

_FUNC_RE = re.compile(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def identity_from_source(source: bytes, *, module: str = SCORER_MODULE, functions: list[str] | None = None) -> dict[str, Any]:
    """Build a scorer identity from raw module source bytes.

    `functions` overrides function discovery (used by tests to keep the
    identity well-defined for synthetic sources).
    """
    if functions is None:
        functions = sorted(set(_FUNC_RE.findall(source.decode("utf-8", errors="replace"))))
    return {
        "module": module,
        "content_sha256": hashlib.sha256(source).hexdigest(),
        "functions": functions,
    }


def scorer_identity() -> dict[str, Any]:
    """Identity of the scoring module THIS process imported.

    The digest is over the source file that backs the imported module, so it
    reflects the code a worker will actually execute (workers import chowder
    from the parent's checkout via `worker_env()`).
    """
    import chowder.evaluators.scoring as scoring

    source_path = Path(scoring.__file__).resolve()
    source = source_path.read_bytes()
    identity = identity_from_source(source)
    # Bind the imported module object's own public surface as a secondary
    # witness: if the file on disk and the imported module ever diverge
    # (stale bytecode caches, import hooks), the function list from the
    # module itself makes that visible instead of silently hashing text the
    # running code never loaded.
    identity["functions"] = sorted(n for n in vars(scoring) if not n.startswith("_") and callable(getattr(scoring, n, None)) and getattr(scoring, n).__module__ == scoring.__name__)
    return identity


def scorer_identity_json() -> str:
    """Canonical JSON form for embedding in a protocol payload."""
    return json.dumps(scorer_identity(), sort_keys=True, separators=(",", ":"))


def scorer_identity_sha256() -> str:
    """The digest a protocol fingerprint actually binds."""
    return hashlib.sha256(scorer_identity_json().encode("utf-8")).hexdigest()
