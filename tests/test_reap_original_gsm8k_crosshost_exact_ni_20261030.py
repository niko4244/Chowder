"""Adversarial, CPU-only proof checks for prospective exact paired NI research."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from chowder_batch import reap_original_gsm8k_crosshost_exact_ni_20261030 as ni


def test_integer_lower_tail_matches_independent_combinatorial_sum():
    for n in range(1, 13):
        for num, den in ((0, 1), (1, 5), (1, 3), (1, 2), (4, 5), (1, 1)):
            for m in range(n + 1):
                numerator, denominator = ni.binomial_lower_tail(n, m, num, den)
                independent = sum(
                    math.comb(n, k) * num**k * (den - num) ** (n - k)
                    for k in range(m + 1)
                )
                assert denominator == den**n
                assert numerator == independent


@pytest.mark.parametrize("n,m,num,den", [
    (0, 0, 1, 2), (1320, 0, 1, 2), (4, -1, 1, 2),
    (4, 5, 1, 2), (4, 1, -1, 2), (4, 1, 3, 2),
    (4, 1, 1, 0), (True, 0, 1, 2), (4, 1, False, 2),
    (4, 1, 1, 2.0),
])
def test_lower_tail_refuses_invalid_arguments(n, m, num, den):
    with pytest.raises(ni.ExactNIRefused):
        ni.binomial_lower_tail(n, m, num, den)


def test_cp_dyadic_upper_bound_is_conservative_at_exact_rational_boundary():
    for m in (0, 1, 25, 46):
        lo, hi = ni.discordance_q_upper(1319, m)
        assert 0 <= lo < hi <= 1
        assert hi - lo == Fraction(1, 1 << ni.BITS)
        left, ld = ni.binomial_lower_tail(1319, m, lo.numerator, lo.denominator)
        right, rd = ni.binomial_lower_tail(1319, m, hi.numerator, hi.denominator)
        assert 40 * left > ld
        assert 40 * right <= rd
    assert ni.discordance_q_upper(1319, 1319) == (Fraction(1), Fraction(1))


def test_cp_q_bound_monotone_in_discordance():
    q0 = ni.discordance_q_upper(1319, 0)[1]
    q1 = ni.discordance_q_upper(1319, 1)[1]
    q25 = ni.discordance_q_upper(1319, 25)[1]
    q46 = ni.discordance_q_upper(1319, 46)[1]
    assert 0 < q0 < q1 < q25 < q46 < 1


def test_cp_upper_coverage_exactly_enumerated_over_small_iid_binomials():
    """Exact probability, not Monte Carlo: CP upper-bound failure <= 1/40."""
    for n in range(1, 10):
        bounds = [ni.discordance_q_upper(n, m)[1] for m in range(n + 1)]
        for q in (Fraction(0), Fraction(1, 10), Fraction(1, 5),
                  Fraction(1, 2), Fraction(4, 5), Fraction(1)):
            failure = sum(
                (Fraction(math.comb(n, m) * q.numerator**m
                          * (q.denominator - q.numerator) ** (n - m),
                          q.denominator**n)
                 for m in range(n + 1) if q > bounds[m]),
                Fraction(0),
            )
            assert failure <= Fraction(1, 40)


def test_observed_original_counts_exact_tail_matches_independent_sum():
    report = ni.exact_research((1199, 25, 21, 74))
    assert report["n"] == 1319 and report["discordant"] == 46
    assert report["observed_candidate_minus_reference"] == "4/1319"
    assert report["reject_minus_two_percent_composite_null_under_iid"] is True
    assert report["exact_conditional_p_decimal"].startswith("0.000191533192786468")
    p = Fraction(report["null_conditional_p_upper"])
    independent = sum(
        Fraction(math.comb(46, k) * p.numerator**k
                 * (p.denominator - p.numerator) ** (46 - k), p.denominator**46)
        for k in range(25, 47)
    )
    assert independent < Fraction(1, 40)
    assert hashlib.sha256(
        (str(independent.numerator) + "/" + str(independent.denominator)).encode()
    ).hexdigest() == report["exact_conditional_p_rational_sha256"]
    assert report["frozen_amendment4_method_adjudicated"] is False
    assert report["same_host_frozen_comparison"] is False
    assert report["native_http_lineage_authenticated"] is False
    assert report["status"] == "BLOCKED"
    assert report["chosen"] is None and report["release_approved"] is False


def test_all_agree_and_all_disagree_do_not_fake_noninferiority():
    all_agree = ni.exact_research((1319, 0, 0, 0))
    assert all_agree["exact_conditional_p_decimal"] == "1"
    assert all_agree["reject_minus_two_percent_composite_null_under_iid"] is False
    all_reference = ni.exact_research((0, 0, 1319, 0))
    assert all_reference["reject_minus_two_percent_composite_null_under_iid"] is False
    all_candidate = ni.exact_research((0, 1319, 0, 0))
    assert all_candidate["reject_minus_two_percent_composite_null_under_iid"] is True


@pytest.mark.parametrize("cells,n", [
    ((1199, 25, 21, 73), 1319), ((1199, 25, 21, 75), 1319),
    ((1199, -1, 21, 100), 1319), ((1199, True, 21, 98), 1319),
    ([1199, 25, 21, 74], 1319), ((1199, 25, 21), 1319),
    ((1199, 25, 21, 74), 164), ((1199, 25, 21, 74), True),
])
def test_forged_or_incompatible_pair_counts_refused(cells, n):
    with pytest.raises(ni.ExactNIRefused):
        ni.exact_research(cells, n=n)


def test_source_pin_and_module_origin_refused_when_changed(monkeypatch):
    ni._source_lock()
    monkeypatch.setitem(ni.PINNED_SOURCES,
                        "reap_original_gsm8k_windows_q4_linux_q8_pair_20261030.py",
                        "0" * 64)
    with pytest.raises(ni.ExactNIRefused, match="source verifier drift"):
        ni._source_lock()


def test_foreign_import_origin_refused(monkeypatch):
    monkeypatch.setattr(ni.original, "__file__", "C:/outside/replaced_pair.py")
    with pytest.raises(ni.ExactNIRefused, match="outside authorized worktree"):
        ni._source_lock()


def test_forged_research_pair_source_is_not_promoted(monkeypatch):
    authentic = ni.original.verify_original_research()
    forged = {**authentic, "paired_counts": {
        "both_correct": 1199, "windows_q4_only": 26,
        "linux_q8_only": 20, "neither_correct": 74,
    }}
    monkeypatch.setattr(ni.original, "verify_original_research", lambda: forged)
    with pytest.raises(ni.ExactNIRefused, match="paired research cells changed"):
        ni.verify_authentic_crosshost_research()


def test_genuine_original_frozen_source_bytes_reproduced():
    report = ni.verify_authentic_crosshost_research()
    assert report["candidate"] == "original_windows_q4"
    assert report["reference"] == "original_linux_q8"
    assert report["paired_cells"] == {
        "both_correct": 1199, "candidate_only": 25,
        "reference_only": 21, "both_incorrect": 74,
    }
    assert report["original_q4_rows_sha256"] == ni.original.Q4_ROWS_SHA256
    assert report["original_q8_rows_sha256"] == ni.original.Q8_ROWS_SHA256
    assert report["original_paired_per_item_sha256"] == (
        "af5b1f2c7aa6c6759ed3bfb54f450d1fc27ddb7ba0a06227c3409599713ead77"
    )
    assert report["status"] == "BLOCKED"


def test_optimized_python_rejects_source_approval():
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-O", "-c",
         "from chowder_batch import reap_original_gsm8k_crosshost_exact_ni_20261030 as n;"
         "n._source_lock()"],
        cwd=root, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode != 0
    assert "optimized Python is not source-approved" in result.stderr


def test_no_false_release_authority_in_serialized_certificate():
    report = ni.verify_authentic_crosshost_research()
    roundtrip = json.loads(json.dumps(report, allow_nan=False))
    assert roundtrip["research_only"] is True
    assert roundtrip["fixed_benchmark_not_iid_certificate"] is True
    assert roundtrip["frozen_amendment4_method_adjudicated"] is False
    assert roundtrip["same_host_frozen_comparison"] is False
    assert roundtrip["native_http_lineage_authenticated"] is False
    assert roundtrip["chosen"] is None
    assert roundtrip["release_approved"] is False
