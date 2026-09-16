"""P4c: parent and child must run — and record — the same Chowder source.

`worker_env()` already forces the child to import the parent's checkout by
pinning PYTHONPATH, and its docstring records the measured failure that
motivated it: a worker silently importing another checkout records results
against the wrong code (the wire-compatible case fails loudly only by luck).

What was missing is the evidence half of that guarantee. This file pins:

1. `chowder_source_identity()` — a content digest of the `chowder` package
   this process actually imported (path + hash over all non-pycache files),
   stable across calls and sensitive to any byte of package source.
2. `verify_source_identity()` — the child-side check: refuse to do any work
   when the identity the parent declared does not match the code this
   interpreter actually imported. Called first in every worker `main()`,
   this is the plan's "fail before training when they disagree".
3. Every worker main() declares `--chowder-identity`, verifies it before
   anything else, and every controller that spawns a worker passes it —
   pinned by source checks because spawning real workers in unit tests is
   the expensive case the identity check exists to protect.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import chowder
from chowder.worker_env import (
    WorkerIdentityError,
    chowder_source_identity,
    identity_from_tree,
    verify_source_identity,
)


def test_source_identity_points_at_the_imported_package():
    identity = chowder_source_identity()
    pkg = Path(chowder.__file__).resolve().parent
    assert Path(identity["source_root"]) == pkg
    assert len(identity["source_sha256"]) == 64


def test_source_identity_is_stable_across_calls():
    assert chowder_source_identity() == chowder_source_identity()


def test_identity_digest_is_sensitive_to_package_content(tmp_path):
    """The digest is over content, not names: one changed byte in one module
    must change it (the wrong-checkout case the guarantee exists for)."""
    pkg = tmp_path / "chowder"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text("X = 1\n", encoding="utf-8")
    (pkg / "sub" / "mod.py").write_text("Y = 2\n", encoding="utf-8")
    (pkg / "sub" / "__pycache__").mkdir()
    (pkg / "sub" / "__pycache__" / "mod.cpython-311.pyc").write_bytes(b"\x00")

    one = identity_from_tree(pkg)
    (pkg / "sub" / "mod.py").write_text("Y = 3\n", encoding="utf-8")
    two = identity_from_tree(pkg)
    assert one["source_sha256"] != two["source_sha256"]

    three = identity_from_tree(pkg)
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    import shutil

    shutil.copytree(pkg, moved / "chowder")
    four = identity_from_tree(moved / "chowder")
    assert three["source_sha256"] == four["source_sha256"], (
        "same content under a different root is the same code identity; "
        "the root is recorded separately"
    )


def test_verify_passes_when_the_child_imports_the_declared_code():
    expected = chowder_source_identity()
    actual = verify_source_identity(expected)
    assert actual == expected


def test_verify_refuses_tampered_expectations():
    expected = dict(chowder_source_identity())
    expected["source_sha256"] = "0" * 64
    with pytest.raises(WorkerIdentityError, match="source identity mismatch"):
        verify_source_identity(expected)

    expected = dict(chowder_source_identity())
    expected["source_root"] = "C:/not/the/right/checkout"
    with pytest.raises(WorkerIdentityError, match="source identity mismatch"):
        verify_source_identity(expected)


def test_verify_refuses_malformed_expectations():
    with pytest.raises(WorkerIdentityError, match="missing|invalid"):
        verify_source_identity({"source_root": "x"})


# ---- the wiring is pinned: every worker gets it, every controller passes it ----

_BACKENDS = Path(chowder.__file__).parent / "backends"
_EVALUATORS = Path(chowder.__file__).parent / "evaluators"


@pytest.mark.parametrize(
    "worker_path",
    [
        _BACKENDS / "transformers_worker.py",
        _EVALUATORS / "transformers_text_worker.py",
        _EVALUATORS / "base_text_worker.py",
    ],
)
def test_every_worker_verifies_identity_before_anything_else(worker_path):
    text = worker_path.read_text(encoding="utf-8")
    assert "--chowder-identity" in text, f"{worker_path.name} does not accept the pin"
    assert "verify_source_identity" in text, f"{worker_path.name} never verifies it"
    # within main(), the verification must run before the worker touches its
    # spec -- nothing may be loaded, run, or written before the pin checks out
    main_body = text[text.index("def main") :]
    assert main_body.index("verify_source_identity") < main_body.index("args.spec"), (
        f"{worker_path.name} reads its spec before verifying its source identity"
    )


@pytest.mark.parametrize(
    "controller_path",
    [
        _BACKENDS / "transformers_peft.py",
        _EVALUATORS / "transformers_text.py",
        _EVALUATORS / "base_text.py",
    ],
)
def test_every_controller_passes_the_identity_to_its_worker(controller_path):
    text = controller_path.read_text(encoding="utf-8")
    assert "chowder_source_identity" in text, (
        f"{controller_path.name} never records the parent's source identity"
    )
    assert "--chowder-identity" in text, (
        f"{controller_path.name} does not hand the identity to its worker"
    )


def test_identity_survives_a_json_round_trip():
    """Controllers pass the identity as a JSON file; the worker re-reads it."""
    identity = chowder_source_identity()
    restored = json.loads(json.dumps(identity))
    assert verify_source_identity(restored) == identity
