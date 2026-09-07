"""Protected nine-dimension suite content (program Phase 4).

Tests enforce the properties that make this content usable as the
tournament's protected holdout: nine-dimension coverage, deterministic
materialization, hash-only fingerprint indexes, a byte-identical
root-free manifest, and a working Contamination Guard round trip.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chowder import parent_suite_content as psc

DIMENSION_ITEMS = [
    ("knowledge", psc.KNOWLEDGE_ITEMS),
    ("reasoning", psc.REASONING_ITEMS),
    ("coding", psc.CODING_ITEMS),
    ("instruction_following", psc.INSTRUCTION_FOLLOWING_ITEMS),
    ("self_correction", psc.SELF_CORRECTION_ITEMS),
    ("agentic", psc.AGENTIC_ITEMS),
    ("thinking_efficiency", psc.THINKING_EFFICIENCY_ITEMS),
    ("behavior", psc.BEHAVIOR_ITEMS),
    ("calibration", psc.CALIBRATION_ITEMS),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_nine_dimensions_six_items_each() -> None:
    assert set(psc.PROTECTED_SUITES) == {d for d, _ in DIMENSION_ITEMS}
    for _dimension, items in DIMENSION_ITEMS:
        assert len(items) == 6
        for prompt, expected in items:
            assert prompt.strip() and expected.strip()
    # Prompt/expected pairs are unique across the whole suite: a duplicated
    # pair would silently narrow the holdout the contamination guard protects.
    all_pairs = [pair for _, items in DIMENSION_ITEMS for pair in items]
    assert len(set(all_pairs)) == len(all_pairs)


def test_materialize_writes_datasets_indexes_manifest(tmp_path: Path) -> None:
    manifest = psc.materialize_protected_suites(tmp_path)
    assert manifest["suite_content_version"] == "v1"
    assert set(manifest["suites"]) == {
        f"suite-{d.replace('_', '-')}-v1" for d, _ in DIMENSION_ITEMS
    }
    for dimension, _items in DIMENSION_ITEMS:
        suite_name = f"suite-{dimension.replace('_', '-')}-v1"
        dataset = tmp_path / "datasets" / f"{suite_name}.jsonl"
        index = tmp_path / "fingerprints" / f"{suite_name}.fingerprints.jsonl"
        assert dataset.is_file() and index.is_file()
        entry = manifest["suites"][suite_name]
        assert entry["dimension"] == dimension
        assert entry["items"] == 6
        assert entry["dataset_sha256"] == _sha256(dataset)
        assert entry["index_sha256"] == _sha256(index)


def test_fingerprint_indexes_are_hash_only(tmp_path: Path) -> None:
    psc.materialize_protected_suites(tmp_path)
    for dimension, items in DIMENSION_ITEMS:
        suite_name = f"suite-{dimension.replace('_', '-')}-v1"
        raw = (tmp_path / "fingerprints" / f"{suite_name}.fingerprints.jsonl").read_text(
            encoding="utf-8"
        )
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        assert len(rows) == 6
        for row in rows:
            assert set(row) == {"prompt_sha256", "pair_sha256"}
            assert len(row["prompt_sha256"]) == 64 and len(row["pair_sha256"]) == 64
        # Raw content must not survive into the index. Every prompt and every
        # expected value here is long enough that a real leak shows up; short
        # expected values ("7", "42") are excluded because any 64-hex-char
        # hash alphabet contains those substrings by chance.
        for prompt, expected in items:
            if len(prompt) >= 20:
                assert prompt[:20] not in raw, f"{suite_name}: prompt text leaked"
            if len(expected) >= 6:
                assert expected not in raw, f"{suite_name}: expected text leaked"
        # Distinct prompts must produce distinct prompt hashes: the index is
        # useless as a contamination guard if it collapses items.
        assert len({row["prompt_sha256"] for row in rows}) == 6


def test_manifest_is_deterministic_and_root_free(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "nested" / "deeper" / "b"
    manifest_a = psc.materialize_protected_suites(root_a)
    manifest_b = psc.materialize_protected_suites(root_b)
    # Content identity must not depend on where the root sits: identical
    # content on any machine yields the identical manifest. The protocol
    # fingerprint is derived at run time (it legitimately embeds deployment
    # paths) and is therefore deliberately absent here.
    assert manifest_a == manifest_b
    assert "protected_root" not in manifest_a
    assert "evaluation_protocol_sha256" not in manifest_a
    manifest_text = (root_a / "manifest.json").read_text(encoding="utf-8")
    assert str(root_a) not in manifest_text


def test_tournament_spec_covers_nine_dimensions(tmp_path: Path) -> None:
    psc.materialize_protected_suites(tmp_path)
    spec = psc.build_tournament_spec(tmp_path)
    dimensions = {suite.dimension for suite in spec.suites}
    assert dimensions == {d for d, _ in DIMENSION_ITEMS}
    # The derived protocol digest is the identity the tournament records;
    # it must be stable across re-derivation at the same root.
    assert spec.digest() == psc.build_tournament_spec(tmp_path).digest()


def test_unmaterialized_root_is_a_hard_stop(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not materialized"):
        psc.build_tournament_spec(tmp_path)


def test_short_suite_is_rejected(tmp_path: Path) -> None:
    original = psc.PROTECTED_SUITES["knowledge"]["suite-knowledge-v1"]
    psc.PROTECTED_SUITES["knowledge"]["suite-knowledge-v1"] = original[:-1]
    try:
        with pytest.raises(ValueError, match="broken suite"):
            psc.materialize_protected_suites(tmp_path)
    finally:
        psc.PROTECTED_SUITES["knowledge"]["suite-knowledge-v1"] = original


def test_contamination_guard_round_trip(tmp_path: Path) -> None:
    from chowder.parent_eval import audit_training_examples_against_tournament

    psc.materialize_protected_suites(tmp_path)
    index_paths = sorted(
        str(p) for p in (tmp_path / "fingerprints").glob("*.fingerprints.jsonl")
    )

    # A verbatim protected prompt, run through the guard, must be caught —
    # this is the mechanical form of the Phase 13 ban.
    guilty = [("What is the chemical symbol for the element tungsten? "
               "Respond with the symbol only.", "w", "drill-0")]
    audit = audit_training_examples_against_tournament(
        [(p, e) for p, e, _ in guilty], index_paths
    )
    assert not audit.clean

    # Unrelated examples pass cleanly.
    clean = [
        ("Completely unrelated question about tartiflette recipes?", "unrelated",),
        ("Another unrelated prompt about tidal energy conversions?", "unrelated",),
    ]
    audit_clean = audit_training_examples_against_tournament(
        [(p, e) for p, e in clean], index_paths
    )
    assert audit_clean.clean
