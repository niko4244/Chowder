"""P4a: the scoring implementation's identity must be part of the protocol.

The 2026-09-12 audit found that the protocol fingerprint pins the scoring
*name* (`"scoring": "final_number_match"`) and dependency versions but not
the scoring *implementation*: the historical lenient rule and the current
strict rule could share a nominal protocol identity if every other payload
field matched, because the rule itself changed inside the same scorer name.

This file pins the fix at three levels:
1. `scorer_identity()` exposes a content digest of `evaluators/scoring.py`
   (the content hash IS the version -- no invented version numbers).
2. Both text evaluators embed that identity in the protocol they fingerprint,
   so a scorer change changes the fingerprint and an old result never
   claims the new scorer's identity.
3. A worker that applies a per-suite chat template reports the template's
   digest, and the controller binds it into the protocol's suite entry --
   the chat template decides rendered bytes, so it is protocol, not trivia.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import chowder
from chowder.evaluators.scorer_identity import (
    SCORER_MODULE,
    identity_from_source,
    scorer_identity,
)


def test_scorer_identity_is_the_hash_of_the_real_scoring_module():
    scoring_path = Path(chowder.__file__).parent / "evaluators" / "scoring.py"
    identity = scorer_identity()
    assert identity["module"] == SCORER_MODULE == "chowder.evaluators.scoring"
    assert identity["content_sha256"] == hashlib.sha256(scoring_path.read_bytes()).hexdigest()
    assert "score" in identity["functions"]
    assert "final_number" in identity["functions"]


def test_identity_changes_when_the_scoring_implementation_changes():
    """The whole point: the digest is a real version, so two scorer contents
    cannot share an identity even under the same name."""
    old = identity_from_source(b"def score():\n    return 1\n", functions=["score"])
    new = identity_from_source(b"def score():\n    return 2\n", functions=["score"])
    assert old["content_sha256"] != new["content_sha256"]
    assert old["module"] == new["module"]


def test_identity_is_stable_across_calls():
    assert scorer_identity() == scorer_identity()


def test_scorer_identity_is_json_canonical():
    payload = json.dumps(scorer_identity(), sort_keys=True, separators=(",", ":"))
    assert "content_sha256" in payload
