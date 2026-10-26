"""Synthetic-only tests for the frozen text-only HLE research protocol.

No HLE question text, reference answer or model output is present here: every
row is invented so the tests cannot depend on or leak benchmark content.
"""

import hashlib
import json

import pytest

from chowder_batch import reap_hle_protocol as P


def row(item_id="a", question="What is 2+2?", answer="4", answer_type="exactMatch", image=None):
    return {
        "id": item_id,
        "question": question,
        "answer": answer,
        "answer_type": answer_type,
        "image": image,
    }


def test_manifest_is_frozen_and_self_hashing():
    first, second = P.frozen_manifest(), P.frozen_manifest()
    assert first == second
    assert first["protocol_sha256"] == hashlib.sha256(
        json.dumps(first["protocol"], sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert first["protocol"]["official_hle_score_claimed"] is False
    assert first["protocol"]["original_scientific_gate_eligible"] is False
    assert first["protocol"]["serving_flags"] == ["--fit", "on", "-np", "2", "-c", "16384", "--jinja"]
    assert first["protocol"]["wall_budget_seconds"] < 12 * 3600


def test_only_genuinely_empty_image_field_is_text_only():
    assert P.is_text_only(row()) is True
    assert P.is_text_only(row(image="")) is True
    assert P.is_text_only(row(image=None)) is True
    assert P.is_text_only(row(image="data:image/jpeg;base64,AAAA")) is False
    assert P.is_text_only(row(image=b"\x89PNG")) is False
    with pytest.raises(ValueError):
        P.is_text_only(row(image=["not", "a", "blob"]))


def test_population_validation_fails_closed():
    assert len(P.text_only_rows([row("a"), row("b", image="x")])) == 1
    with pytest.raises(ValueError, match="duplicate"):
        P.text_only_rows([row("a"), row("a")])
    with pytest.raises(ValueError, match="missing HLE id"):
        P.text_only_rows([row("")])
    with pytest.raises(ValueError, match="question"):
        P.text_only_rows([row(question="   ")])
    with pytest.raises(ValueError, match="answer_type"):
        P.text_only_rows([row(answer_type="freeform")])


def test_canonical_order_is_deterministic_and_selection_free():
    rows = [row(str(n)) for n in range(60)]
    order = [r["id"] for r in P.canonical_order(rows)]
    assert order == [r["id"] for r in P.canonical_order(list(reversed(rows)))]
    expected = sorted((r["id"] for r in rows), key=lambda i: hashlib.sha256(i.encode()).hexdigest())
    assert order == expected
    assert order != sorted(order)


def test_messages_use_the_official_answer_format():
    messages = P.build_messages("Question?")
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == P.SYSTEM_PROMPT
    assert "Answer: {your chosen answer}" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "Question?"}
    with pytest.raises(ValueError):
        P.build_messages("  ")


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Articles are intentionally preserved: dropping them would create
        # false matches that only the official judge is entitled to resolve.
        ("  The Answer.  ", "the answer"),
        ("$1,000", "1000"),
        ("\\text{Yes}", "yes"),
        ("2/7.", "2/7"),
        ("50 percent", "50%"),
        ("**Yes**", "yes"),
        ('"Katie"', "katie"),
    ],
)
def test_normalize_answer_is_conservative(raw, expected):
    assert P.normalize_answer(raw) == expected


def test_normalize_answer_does_not_claim_mathematical_equivalence():
    assert P.normalize_answer("\\frac{2}{7}") != P.normalize_answer("2/7")
    with pytest.raises(TypeError):
        P.normalize_answer(None)


@pytest.mark.parametrize(
    "response,expected",
    [
        ("Explanation: because\nAnswer: A\nConfidence: 90%", "A"),
        ("Explanation: x\nAnswer: (C)\nConfidence: 12%", "C"),
        ("Explanation: x\nanswer: 18", "18"),
        ("Explanation: x\nAnswer: d. Egalitarian\nConfidence: 5%", "d"),
        ("no marker at all\nfinal line 42", "final line 42"),
    ],
)
def test_extract_final_answer(response, expected):
    assert P.extract_final_answer(response) == expected


def test_extract_final_answer_prefers_the_last_answer_line():
    response = "Answer: A\nExplanation: revised\nAnswer: B\nConfidence: 70%"
    assert P.extract_final_answer(response) == "B"
    assert P.extract_final_answer("") == ""


def test_grade_exact_match_and_multiple_choice():
    assert P.grade("Explanation: x\nAnswer: 4", "4", "exactMatch")["correct"] is True
    assert P.grade("Explanation: x\nAnswer: $5", "5", "exactMatch")["correct"] is True
    assert P.grade("Explanation: x\nAnswer: 6", "4", "exactMatch")["correct"] is False
    assert P.grade("Explanation: x\nAnswer: B", "B", "multipleChoice")["correct"] is True
    assert P.grade("Explanation: x\nAnswer: A", "B", "multipleChoice")["correct"] is False
    with pytest.raises(ValueError):
        P.grade("Answer: A", "A and B", "multipleChoice")
    with pytest.raises(ValueError):
        P.grade("Answer: A", "A", "freeform")


def test_summarize_reports_descriptive_only():
    scored = [
        {"answer_type": "exactMatch", "correct": True},
        {"answer_type": "exactMatch", "correct": False},
        {"answer_type": "multipleChoice", "correct": True},
    ]
    summary = P.summarize(scored, population=10)
    assert summary["completed_n"] == 3
    assert summary["correct"] == 2
    assert summary["accuracy"] == pytest.approx(2 / 3)
    assert summary["completion_fraction_of_population"] == pytest.approx(0.3)
    assert summary["by_answer_type"]["exactMatch"] == {"n": 2, "correct": 1}
    assert summary["official_judge_score"] == "UNAVAILABLE"
    assert summary["metric"] == "deterministic_non_official_normalized_match"
    assert P.summarize([], population=0)["accuracy"] is None


def test_protocol_module_has_no_execution_side_effects():
    source = open(P.__file__, encoding="utf-8").read()
    for forbidden in ("subprocess", "requests", "socket", "urllib", "datasets.load_dataset"):
        assert forbidden not in source
