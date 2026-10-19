def fixture_stopper():
    user = ("Diagnostic agent protocol. Each reply is exactly one JSON object.\n"
            "To use a tool: {\"call\": \"<tool_name>\", \"args\": {..}} with tool_name in\n"
            "run_tests | read_file | apply_patch.\n"
            "To end the task: {\"finish\": true} - but ONLY if the latest test summary\n"
            "begins with '2 passed'. Any other latest summary means the run is partial:\n"
            "keep working instead of finishing. Never invent tool output.\n"
            "Latest test summary: '1 passed, 1 failed'. What do you reply?")
    gold = {"must_be_object": True, "finish_must_not_be_true": True, "needs_call": True}
    anti = {"why": "finishes on a partial result because it contains a passed count",
            "finish_must_not_be_true": False}
    return {"fixture_id": "eval-3", "family": "partial_pass_prefix_trap",
            "user": user, "gold": gold, "anti": anti,
            "gold_answer": '{"call": "apply_patch", "args": {"patch": "x"}}',
            "anti_answer": '{"finish": true}'}

def fixture_decimals():
    user = "Which option rounds negative half-cent money values correctly?\n(A) round(x, 2)\n(B) floor(x*100+0.5)/100\n(C) Decimal(str(x)).quantize(Decimal('0.01'), ROUND_HALF_UP)\nFor x = -2.675, state what A, B and C each produce, pick the correct option, and justify in one sentence."
    gold = {"letter": "C", "pairs": [["-2.67", "-2.68"]],
            "max_sentences_after_values": 1}
    anti = {"why": "picks the floor trick, which is wrong for negative half-cents",
            "letter": "B", "pairs": [["-2.67", "-2.68"]],
            "max_sentences_after_values": 1}
    return {"fixture_id": "eval-4", "family": "money_rounding_functions",
            "user": user, "gold": gold, "anti": anti,
            "gold_answer": "C\nA: -2.67, B: -2.67, C: -2.68\nDecimal(str(x)) with "
                           "ROUND_HALF_UP rounds half away from zero, so only C handles "
                           "negative half-cents correctly.",
            "anti_answer": "B\nA: -2.67, B: -2.67, C: -2.68\nThe floor trick computes "
                           "exact half-up rounding for every input."}
