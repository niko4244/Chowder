import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / "experiments" / "teacher_free_distill" / "build_openr1_pilot.py"
_spec = importlib.util.spec_from_file_location("build_openr1_pilot", _path)
b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)


def test_complete_trace_passes() -> None:
    assert b.audit_target("<think>\nwork it out\n</think>\n\nSo \\boxed{64}.") is None


def test_condition_a_shapes_are_rejected() -> None:
    # Mid-trace chunk (no close), empty block then reopened reasoning, no final answer.
    assert b.audit_target("<think> Okay, so the degree drops by 1") == "think_markers_not_single_pair"
    assert b.audit_target("<think>\n\n</think>\n\n<think> Okay, wait") == "think_markers_not_single_pair"
    assert b.audit_target("<think>\n\n</think>\n\nThe answer is \\boxed{1}") == "empty_reasoning"
    assert b.audit_target("<think>\nreasoning\n</think>\n\nLet me check again.") == "no_boxed_answer_after_think"
    assert b.audit_target("Wait, the problem says...") == "no_opening_think"


def test_decontamination_catches_reworded_copies() -> None:
    q = ("Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes "
         "muffins for her friends every day with four.")
    norm = {" ".join(b._words(q))}
    grams = b.ngrams(q)
    assert b.contaminated(q.upper(), norm, grams)
    assert b.contaminated("Problem 3. " + q + " How much does she make?", set(), grams)
    assert not b.contaminated("A ship travels 24 km upstream and 28 km downstream.", norm, grams)
