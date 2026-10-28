"""Adversarial fixtures and authentic-source checks for Freebuff stage 0/1 audit.

Synthetic fixtures in this module are NOT original model benchmark evidence.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from chowder_batch.freebuff_stage01_source_audit_20261027 import (
    DOMAINS,
    EvidenceError,
    audit_stage0,
    audit_stage1,
    group_scores,
    verify_curves,
    verify_sources,
)


def fixture_rows() -> list[dict]:
    return [
        {
            "layer": 1,
            "expert": 2,
            "domain": domain,
            "codec": codec,
            "item": item,
            "score": float(item == 0 and codec == "bf16"),
        }
        for domain in DOMAINS
        for codec in ("bf16", "nf4")
        for item in range(48)
    ]


def fixture_curve() -> dict:
    signed = -1 / 48
    return {
        "row_count": 288,
        "baseline_codec": "bf16",
        "domains": list(DOMAINS),
        "codecs": {"nf4": 4.0},
        "curves": [
            {
                "layer": 1,
                "expert": 2,
                "options": [
                    {
                        "codec": "nf4",
                        "bits_per_param": 4.0,
                        "domain_damage": dict.fromkeys(DOMAINS, -signed),
                        "mean_damage": -signed,
                    }
                ],
            }
        ],
        "signed_deltas": [
            {"layer": 1, "expert": 2, "domain": domain, "codec": "nf4", "signed_delta": signed}
            for domain in DOMAINS
        ],
    }


def test_fixture_valid_population_and_curve() -> None:
    grouped = group_scores(fixture_rows())
    assert len(grouped) == 6
    verify_curves(grouped, fixture_curve(), 288)


def test_fixture_duplicate_item_fails_closed() -> None:
    rows = fixture_rows()
    with pytest.raises(EvidenceError, match="duplicate"):
        group_scores([*rows, rows[0]])


def test_fixture_missing_holdout_item_fails_closed() -> None:
    with pytest.raises(EvidenceError, match="non-complete"):
        group_scores(fixture_rows()[:-1])


@pytest.mark.parametrize("bad_score", [2.0, -1.0, "1", None, True, 0.5])
def test_fixture_invalid_score_fails_closed(bad_score: object) -> None:
    rows = fixture_rows()
    rows[0] = {**rows[0], "score": bad_score}
    with pytest.raises(EvidenceError, match="score type"):
        group_scores(rows)


def test_fixture_curve_tampering_fails_closed() -> None:
    grouped = group_scores(fixture_rows())
    changed = deepcopy(fixture_curve())
    changed["curves"][0]["options"][0]["domain_damage"][DOMAINS[1]] = 0.0
    with pytest.raises(EvidenceError, match="damage mismatch"):
        verify_curves(grouped, changed, 288)


def test_fixture_omitted_entire_codec_curve_fails_closed() -> None:
    """An empty set of self-consistent curves must not certify measured arms."""
    grouped = group_scores(fixture_rows())
    changed = deepcopy(fixture_curve())
    changed["curves"] = []
    changed["signed_deltas"] = []
    with pytest.raises(EvidenceError, match="curve coverage"):
        verify_curves(grouped, changed, 288)


def test_fixture_signed_delta_tampering_fails_closed() -> None:
    grouped = group_scores(fixture_rows())
    changed = deepcopy(fixture_curve())
    changed["signed_deltas"][0]["signed_delta"] *= -1
    with pytest.raises(EvidenceError, match="signed delta mismatch"):
        verify_curves(grouped, changed, 288)


def test_digest_tampering_fails_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "rows.jsonl"
    artifact.write_text(json.dumps({"score": 1}) + "\n", encoding="utf-8")
    original = hashlib.sha256(artifact.read_bytes()).hexdigest()
    verify_sources(tmp_path, {"rows.jsonl": original})
    artifact.write_text(json.dumps({"score": 0}) + "\n", encoding="utf-8")
    with pytest.raises(EvidenceError, match="SHA256 mismatch"):
        verify_sources(tmp_path, {"rows.jsonl": original})


def test_missing_source_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(EvidenceError, match="missing source"):
        audit_stage0(tmp_path)


def test_authentic_oct14_archives_when_available() -> None:
    stage0 = Path("G:/freebuff-stage0/run")
    stage1 = Path("G:/freebuff-stage1/run")
    if not stage0.is_dir() or not stage1.is_dir():
        pytest.skip("External archived local Freebuff experiment roots unavailable on this machine")

    a = audit_stage0(stage0)
    b = audit_stage1(stage1)
    assert a["raw_rows"] == 864
    assert a["unique_logical_model_domain_codec_items"] == 432
    assert b["raw_rows"] == 2736
    assert b["fully_paired_two_codec_experts"] == [54]
    assert b["completed_altered_arms"] == ["pruned-l8e54", "crit2bit-l8e54", "pruned-l8e49"]
    first = {(r["domain"], r["codec"]): r["correct"] for r in a["summary_scores"]}
    assert first[("gsm8k-numeric", "bf16")] == 19
    assert first[("gsm8k-numeric", "int8")] == 20
    assert first[("gsm8k-numeric", "nf4")] == 18
    second = {(r["expert"], r["domain"], r["codec"]): r["correct"] for r in b["summary_scores_measured_blocks"]}
    assert second[(54, "math500-numeric", "bf16")] == 4
    assert second[(54, "math500-numeric", "pruned")] == 5
    assert second[(54, "math500-numeric", "crit2bit")] == 2
