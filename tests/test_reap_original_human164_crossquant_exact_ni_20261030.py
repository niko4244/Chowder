"""Adversarial CPU-only research checks; no original execution journals opened."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import pytest

from chowder_batch import reap_original_human164_crossquant_exact_ni_20261030 as h


def _cert() -> dict:
    return json.loads(h.PAIR_PATH.read_bytes())


def _raw(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")


def test_authentic_original_164_is_independently_recounted_without_journals():
    if not h.PAIR_PATH.is_file():
        pytest.skip("original F: certificate unavailable on this host")
    result = h.verify_authentic_pair()
    assert result["paired_cells"] == {
        "both_correct": 147, "q8_only": 5, "q4_only": 4, "neither_correct": 8,
    }
    assert result["discordant"] == 9
    assert result["observed_q8_minus_q4"] == "1/164"
    assert result["paired_original_certificate_sha256"] == h.PAIR_SHA256
    assert result["original_journals_reopened"] is False
    assert result["research_only"] is True
    assert result["status"] == "BLOCKED"
    assert result["chosen"] is None and result["release_approved"] is False
    assert result["not_frozen_q8_vs_dense_gate2"] is True
    assert result["frozen_amendment4_method_adjudicated"] is False
    assert result["fixed_population_iid_not_established"] is True


def test_exact_conditional_tail_matches_independent_binomial_sum():
    result = h.exact_research(h.EXPECTED_CELLS)
    p = Fraction(result["null_conditional_p_upper"])
    independently = sum((Fraction(math.comb(9, j)) * p**j * (1 - p)**(9 - j)
                         for j in range(5, 10)), Fraction(0))
    digest = hashlib.sha256(
        f"{independently.numerator}/{independently.denominator}".encode("ascii")
    ).hexdigest()
    assert result["exact_conditional_p_rational_sha256"] == digest
    assert abs(float(independently) - 0.26990312265048455) < 1e-14
    assert result["reject_minus_two_percent_null_under_iid"] is False
    assert independently > Fraction(1, 40)
    assert result["iid_composite_null_type_i_bound"] == "1/20"


@pytest.mark.parametrize("bad", [
    (147, 5, 4),
    (147, 5, 4, 7),
    (147, 5, 4, -1),
    (147, True, 4, 12),
    [147, 5, 4, 8],
    (147.0, 5, 4, 8),
    "147,5,4,8",
])
def test_invalid_pure_research_cells_refused(bad):
    with pytest.raises(h.Human164NIRefused):
        h.exact_research(bad)


@pytest.mark.parametrize("change", [
    lambda j: j.update(status="FULL_FROZEN_GATE2_APPROVED"),
    lambda j: j.update(gen0_status="PASS"),
    lambda j: j.update(release_approved=True),
    lambda j: j.update(chosen="reap128_v2"),
    lambda j: j.update(is_original_frozen_Q8_vs_dense_gate2=True),
    lambda j: j.update(is_amendment4_paired_mcnemar_ci=True),
    lambda j: j.update(is_confirmatory_quantization_comparison=True),
    lambda j: j.update(paired_original_case_count=163),
    lambda j: j.update(remaining_unscored_q8_original_cases=1),
    lambda j: j.update(original_q4_model_sha256="0" * 64),
    lambda j: j.update(original_q8_witnessed_journal_sha256="0" * 64),
    lambda j: j.update(original_q8_serving_session=j["original_q4_serving_session"]),
    lambda j: j.update(q8_only_correct=6),
    lambda j: j.update(original_q8_correct_in_paired_items=153),
    lambda j: j["per_item_original_source_bound_pairs"].pop(),
    lambda j: j["per_item_original_source_bound_pairs"][0].update(index=2),
    lambda j: j["per_item_original_source_bound_pairs"][0].update(id="HumanEval/1"),
    lambda j: j["per_item_original_source_bound_pairs"][0].update(q8_independent_correct=False),
    lambda j: j["per_item_original_source_bound_pairs"][0].update(q4_independent_correct=1),
    lambda j: j["per_item_original_source_bound_pairs"][0].update(q4_original_program_sha256="bad"),
    lambda j: j["per_item_original_source_bound_pairs"][0].update(unexpected_field="PASS"),
    lambda j: j["per_item_original_source_bound_pairs"][0].update(q8_original_program_sha256=None),
])
def test_adversarial_modified_certificate_rejected(change):
    if not h.PAIR_PATH.is_file():
        pytest.skip("original F: certificate unavailable on this host")
    cert = _cert()
    change(cert)
    with pytest.raises(h.Human164NIRefused):
        h._read_pair(_raw(cert))


def test_duplicate_json_key_and_nonfinite_json_refused():
    with pytest.raises(h.Human164NIRefused):
        h._read_pair(b'{"schema":"a","schema":"b"}')
    with pytest.raises(h.Human164NIRefused):
        h._read_pair(b'{"status":NaN}')


def test_pair_certificate_source_sha_pinned_and_no_journal_reopen(monkeypatch, tmp_path):
    if not h.PAIR_PATH.is_file():
        pytest.skip("original F: certificate unavailable on this host")
    raw = h.PAIR_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == h.PAIR_SHA256
    altered = tmp_path / "resealed-pair.json"
    altered.write_bytes(raw + b"\n")
    monkeypatch.setattr(h, "PAIR_PATH", altered)
    with pytest.raises(h.Human164NIRefused, match="SHA changed"):
        h.verify_authentic_pair()


def test_source_drift_refuses_before_research(monkeypatch):
    h._source_lock()
    monkeypatch.setattr(h, "NI_SOURCE_SHA256", "0" * 64)
    with pytest.raises(h.Human164NIRefused, match="source drift"):
        h.verify_authentic_pair()


def test_pair_producer_drift_refuses(monkeypatch):
    monkeypatch.setattr(h, "PAIR_PRODUCER_SHA256", "0" * 64)
    with pytest.raises(h.Human164NIRefused, match="source drift"):
        h.verify_authentic_pair()


def test_original_journal_paths_are_never_read(monkeypatch):
    original_read = Path.read_bytes

    def guarded_read(path):
        if "human164-v38-private" in str(path).lower():
            raise AssertionError("original completed execution journal was reopened")
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    assert h.verify_authentic_pair()["original_journals_reopened"] is False
