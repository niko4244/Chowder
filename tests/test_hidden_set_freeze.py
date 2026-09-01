"""Guards docs/HIDDEN_SET_FREEZE.md's freeze, not the fixtures themselves.

If any of the three rule-bearing files this freeze protects changes after
the freeze date, this must fail loudly at collection time -- a hidden-set
result computed against rules that moved after the freeze would not be
evidence of generalization, only of the target having moved. This is
deliberately the only test Task 7 adds; scoring a run against the hidden
fixtures is Task 8's job (docs/EXECUTOR_INVESTIGATOR_PLAN.md), not this
file's.
"""
import hashlib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_FROZEN_HASHES = {
    "src/chowder/incident.py": "d43dd2b16f05cf98c6ddd61d913dd489f6386bb1284b1884fd35cdef37eb0cea",
    "src/chowder/probes.py": "b10e8b4685d4399f4cc2906f9d21f5e59b0499092298b9feecc7440fc82a84ec",
    "src/chowder/hypothesis_generation.py": (
        "21b35ae5f30635d37681d6f72f7e9b29bfad20950d841337b08a5117f9698cfa"
    ),
}


def _sha256_of(relative_path: str) -> str:
    return hashlib.sha256((_REPO_ROOT / relative_path).read_bytes()).hexdigest()


def test_freeze_intact():
    mismatched = {
        path: (expected, _sha256_of(path))
        for path, expected in _FROZEN_HASHES.items()
        if _sha256_of(path) != expected
    }
    assert not mismatched, (
        "one or more files frozen in docs/HIDDEN_SET_FREEZE.md have changed since "
        f"the freeze -- {mismatched!r}. See that file for what to do: re-run the "
        "dev set, decide explicitly whether the hidden set stays comparable, and "
        "re-freeze with a new hash and a dated note, rather than silently updating "
        "this test's expected hashes."
    )
