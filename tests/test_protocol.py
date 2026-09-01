from chowder.gate import evaluate_candidate
from chowder.models import ExperimentResult, Goal, MetricTarget
from chowder.protocol import protocol_fingerprint, result_protocol_fingerprint


def _result(name: str, score: float, protocol_sha: str | None):
    evidence = {}
    if protocol_sha is not None:
        evidence["evaluation_protocol_sha256"] = protocol_sha
    return ExperimentResult(name, {"quality": score}, 0.1, evidence=evidence)


def test_protocol_fingerprint_is_order_independent_and_sensitive_to_protocol_changes():
    a = {"task": "x", "settings": {"fewshot": 0, "seed": 7}}
    b = {"settings": {"seed": 7, "fewshot": 0}, "task": "x"}
    c = {"task": "x", "settings": {"fewshot": 5, "seed": 7}}
    assert protocol_fingerprint(a) == protocol_fingerprint(b)
    assert protocol_fingerprint(a) != protocol_fingerprint(c)


def test_strict_gate_accepts_improvement_only_when_protocol_matches():
    protocol = "a" * 64
    goal = Goal(
        (MetricTarget("quality"),),
        gpu_hour_budget=1,
        require_protocol_match=True,
    )
    baseline = _result("base", 0.70, protocol)
    candidate = _result("candidate", 0.80, protocol)
    decision = evaluate_candidate(goal=goal, baseline=baseline, candidate=candidate)
    assert decision.accepted is True


def test_strict_gate_rejects_mismatched_protocol_even_when_score_improves():
    goal = Goal(
        (MetricTarget("quality"),),
        gpu_hour_budget=1,
        require_protocol_match=True,
    )
    decision = evaluate_candidate(
        goal=goal,
        baseline=_result("base", 0.70, "a" * 64),
        candidate=_result("candidate", 0.95, "b" * 64),
    )
    assert decision.accepted is False
    assert "protocol" in decision.reason


def test_strict_gate_rejects_missing_protocol_evidence():
    goal = Goal(
        (MetricTarget("quality"),),
        gpu_hour_budget=1,
        require_protocol_match=True,
    )
    decision = evaluate_candidate(
        goal=goal,
        baseline=_result("base", 0.70, "a" * 64),
        candidate=_result("candidate", 0.80, None),
    )
    assert decision.accepted is False
    assert "evaluation_protocol" in decision.missing_metrics


def test_protocol_extraction_supports_nested_evaluator_evidence():
    evidence = {"evaluation": {"protocol_sha256": "c" * 64}}
    assert result_protocol_fingerprint(evidence) == "c" * 64
