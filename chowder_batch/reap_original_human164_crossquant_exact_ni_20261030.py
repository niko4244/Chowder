"""Source-pinned, CPU-only exact NI research on completed HumanEval Q8/Q4.

Reopens ONLY the immutable, previously independently certified 164-case PAIR
certificate. Never reopens completed execution journals or reruns generated
programs. This is not the frozen Q8-vs-dense Gate2 or Amendment4 adjudication.
The conditional exact method assumes iid paired observations, an assumption
NOT established for the fixed HumanEval population.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import sys

from chowder_batch import reap_original_gsm8k_crosshost_exact_ni_20261030 as ni


SCHEMA = "chowder.reap-original-human164-crossquant-exact-ni-research.v1"
PAIR_PATH = Path("F:/reap-v2-original-human164-q4q8-COMPLETE-paired-research-20261027-1525.json")
PAIR_SHA256 = "5a2ff5e6ab9f371171de96d368672d52854dd47705c6abb6047de432520893c7"
NI_SOURCE_SHA256 = "0b20f89fc4fa1ce1eac7d4a9a20ca8ca31d34c9190634b182913d69f3c0cdef4"
PAIR_PRODUCER_SHA256 = "23ddb5004d2342516b25dd140f345597cc7639c5678f9824648870a8606d3f0a"
ORIGINAL_MODEL_SHA256 = {
    "q4": "2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7",
    "q8": "89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6",
}
ORIGINAL_JOURNAL_SHA256 = {
    "q4": "d77ff1475bbea9dce352cb1318d15cdc2e76bae6c9d7869b17cf24cf0ec8a902",
    "q8": "03d7931bebcaaf70235f9e2dd6d30d0b23d5a476c487a83f8ef25852297376e8",
}
EXPECTED_CELLS = (147, 5, 4, 8)  # both, Q8 only, Q4 only, neither
N = 164
MARGIN = Fraction(1, 50)
ALPHA_PART = Fraction(1, 40)
PAIR_ITEM_FIELDS = frozenset({
    "id", "index", "q4_independent_correct", "q8_independent_correct",
    "q4_original_program_sha256", "q8_original_program_sha256",
})


class Human164NIRefused(ValueError):
    """Original evidence or prospective exact arithmetic not admissible."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise Human164NIRefused(reason)


def _source_lock() -> None:
    _require(sys.flags.optimize == 0 and __debug__, "optimized Python is not source-approved")
    directory = Path(__file__).resolve().parent
    pins = {
        "reap_original_gsm8k_crosshost_exact_ni_20261030.py": NI_SOURCE_SHA256,
        "reap_humaneval_v38_q4q8_paired_original_20261027.py": PAIR_PRODUCER_SHA256,
    }
    for filename, digest in pins.items():
        path = directory / filename
        _require(path.is_file() and not path.is_symlink()
                 and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                 "original research method/source drift: " + filename)
    _require(Path(ni.__file__).resolve() == directory /
             "reap_original_gsm8k_crosshost_exact_ni_20261030.py",
             "imported exact method outside approved worktree")
    ni._source_lock()


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        _require(key not in result, "duplicate original certificate JSON key")
        result[key] = value
    return result


def _read_pair(raw: bytes) -> tuple[int, int, int, int]:
    """Recompute every cell from the previously certified 164 individual rows."""
    _require(type(raw) is bytes and 0 < len(raw) < 256 * 1024,
             "original paired certificate absent or oversized")
    try:
        obj = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
    except (UnicodeError, ValueError) as exc:
        raise Human164NIRefused("original paired certificate malformed") from exc
    _require(type(obj) is dict
             and obj.get("schema") == "chowder.original-human164-q4-q8-paired-independent-research.v1"
             and obj.get("status") == "FULL_164_INDEPENDENT_RESEARCH_PAIRED_Q8_Q4_ONLY"
             and type(obj.get("population")) is int and obj["population"] == N
             and type(obj.get("paired_original_case_count")) is int
             and obj["paired_original_case_count"] == N
             and type(obj.get("remaining_unscored_q8_original_cases")) is int
             and obj["remaining_unscored_q8_original_cases"] == 0
             and obj.get("gen0_status") == "BLOCKED"
             and obj.get("chosen", 1) is None and obj.get("release_approved") is False
             and obj.get("is_original_frozen_Q8_vs_dense_gate2") is False
             and obj.get("is_amendment4_paired_mcnemar_ci") is False
             and obj.get("is_confirmatory_quantization_comparison") is False,
             "original research-only 164-case identity changed")
    for arm in ("q4", "q8"):
        _require(obj.get(f"original_{arm}_model_sha256") == ORIGINAL_MODEL_SHA256[arm]
                 and obj.get(f"original_{arm}_witnessed_journal_sha256")
                 == ORIGINAL_JOURNAL_SHA256[arm]
                 and type(obj.get(f"original_{arm}_serving_session")) is str
                 and re.fullmatch(r"[0-9a-f]{32}", obj[f"original_{arm}_serving_session"])
                 is not None,
                 "original model/session/journal provenance differs")
    _require(obj["original_q4_serving_session"] != obj["original_q8_serving_session"],
             "distinct quantizations cannot share serving session")
    rows = obj.get("per_item_original_source_bound_pairs")
    _require(type(rows) is list and len(rows) == N,
             "original 164 canonical paired cases incomplete")
    both = q8_only = q4_only = neither = 0
    for index, row in enumerate(rows):
        _require(type(row) is dict and set(row) == PAIR_ITEM_FIELDS
                 and type(row["index"]) is int and row["index"] == index
                 and row["id"] == f"HumanEval/{index}"
                 and type(row["q4_independent_correct"]) is bool
                 and type(row["q8_independent_correct"]) is bool,
                 "original paired case identity or correctness invalid")
        for arm in ("q4", "q8"):
            program = row[f"{arm}_original_program_sha256"]
            _require((program is None and not row[f"{arm}_independent_correct"])
                     or (type(program) is str and
                         re.fullmatch(r"[0-9a-f]{64}", program) is not None),
                     "original generated program source identity invalid")
        c4, c8 = row["q4_independent_correct"], row["q8_independent_correct"]
        both += int(c4 and c8)
        q8_only += int(c8 and not c4)
        q4_only += int(c4 and not c8)
        neither += int(not c4 and not c8)
    cells = (both, q8_only, q4_only, neither)
    _require(cells == EXPECTED_CELLS
             and all(type(obj.get(name)) is int and obj[name] == value
                     for name, value in zip(("both_correct", "q8_only_correct",
                                             "q4_only_correct", "both_incorrect"), cells))
             and obj.get("original_q4_correct_in_paired_items") == both + q4_only
             and obj.get("original_q8_correct_in_paired_items") == both + q8_only,
             "per-item paired outcomes differ from certified totals")
    return cells


def exact_research(cells: tuple[int, int, int, int]) -> dict:
    """Prospective exact test, Q8 candidate vs Q4 reference, at -2% margin."""
    _require(type(cells) is tuple and len(cells) == 4
             and all(type(value) is int and value >= 0 for value in cells)
             and sum(cells) == N, "paired HumanEval164 cells invalid")
    both, candidate_only, reference_only, neither = cells
    m = candidate_only + reference_only
    q_lower, q_upper = ni.discordance_q_upper(N, m)
    p_upper = max(Fraction(0), (q_upper - MARGIN) / (2 * q_upper))
    numerator, denominator = ni.exact.binomial_upper_tail_rational(
        m, candidate_only, p_upper.numerator, p_upper.denominator)
    tail = Fraction(numerator, denominator)
    with localcontext() as ctx:
        ctx.prec = 48
        decimal_p = str(Decimal(tail.numerator) / Decimal(tail.denominator))
    return {
        "schema": SCHEMA, "population": N,
        "paired_cells": {"both_correct": both, "q8_only": candidate_only,
                         "q4_only": reference_only, "neither_correct": neither},
        "discordant": m,
        "observed_q8_minus_q4": str(Fraction(candidate_only - reference_only, N)),
        "discordance_q_upper_dyadic": str(q_upper),
        "discordance_q_lower_dyadic": str(q_lower),
        "null_conditional_p_upper": str(p_upper),
        "exact_conditional_p_decimal": decimal_p,
        "exact_conditional_p_rational_sha256": hashlib.sha256(
            f"{tail.numerator}/{tail.denominator}".encode("ascii")).hexdigest(),
        "alpha_for_discordance_bound": "1/40",
        "alpha_for_conditional_tail": "1/40",
        "reject_minus_two_percent_null_under_iid": tail < ALPHA_PART,
        "iid_composite_null_type_i_bound": "1/20",
        "fixed_population_iid_not_established": True,
        "not_frozen_q8_vs_dense_gate2": True,
        "frozen_amendment4_method_adjudicated": False,
        "original_journals_reopened": False,
        "research_only": True,
        "status": "BLOCKED", "chosen": None, "release_approved": False,
    }


def verify_authentic_pair() -> dict:
    """Verify only the existing SHA-pinned, already certified paired artifact."""
    _source_lock()
    _require(PAIR_PATH.is_file() and not PAIR_PATH.is_symlink(),
             "previously certified paired artifact absent or linked")
    raw = PAIR_PATH.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == PAIR_SHA256,
             "original independently certified paired artifact SHA changed")
    cells = _read_pair(raw)
    result = exact_research(cells)
    result["paired_original_certificate_sha256"] = PAIR_SHA256
    result["q4_original_journal_sha256"] = ORIGINAL_JOURNAL_SHA256["q4"]
    result["q8_original_journal_sha256"] = ORIGINAL_JOURNAL_SHA256["q8"]
    return result


if __name__ == "__main__":
    print(json.dumps(verify_authentic_pair(), sort_keys=True, indent=2))
