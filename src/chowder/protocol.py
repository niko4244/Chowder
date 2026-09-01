from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def canonical_protocol_json(protocol: Mapping[str, Any]) -> str:
    """Serialize an evaluation protocol deterministically for comparison."""
    return json.dumps(dict(protocol), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def protocol_fingerprint(protocol: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_protocol_json(protocol).encode("utf-8")).hexdigest()


def result_protocol_fingerprint(evidence: Mapping[str, Any]) -> str | None:
    """Extract a protocol fingerprint from an ExperimentResult evidence payload."""
    direct = evidence.get("evaluation_protocol_sha256")
    if isinstance(direct, str) and len(direct) == 64:
        return direct
    evaluation = evidence.get("evaluation")
    if isinstance(evaluation, Mapping):
        nested = evaluation.get("protocol_sha256")
        if isinstance(nested, str) and len(nested) == 64:
            return nested
    return None
