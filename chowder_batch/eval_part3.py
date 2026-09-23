def fixture_urljoin():
    user = ("Fill in each output exactly as the standard library would print it:\n"
            "1. urllib.parse.urljoin('https://h/a', 'b')\n"
            "2. urllib.parse.urljoin('https://h/a', 'b?')\n"
            "Then name the rule you relied on (one sentence).")
    gold = {"pairs": [["https://h/b", "https://h/b?"]], "rule_max_sentences": 1}
    anti = {"why": "claims the trailing '?' is dropped", "pairs": [["https://h/b", "https://h/b"]]}
    return {"fixture_id": "eval-5", "family": "stdlib_behavior_quirks",
            "user": user, "gold": gold, "anti": anti,
            "gold_answer": "1. https://h/b\n2. https://h/b?\nRule: a relative reference "
                           "replaces the last path segment, keeping any query marker.",
            "anti_answer": "1. https://h/b\n2. https://h/b\nRule: trailing question marks "
                           "are insignificant and get stripped by urljoin."}

def fixture_decimal_mul():
    user = ("Decimal arithmetic. Write what each expression evaluates to, exactly as a "
            "str() of the result would show it:\n"
            "1. Decimal('0.00') * Decimal('0')\n"
            "2. Decimal('1.11') + Decimal('0.00')\n"
            "Then name the rule (one sentence).")
    gold = {"pairs": [["0.0000", "1.11"]], "rule_max_sentences": 1}
    anti = {"why": "normalizes the product to '0'", "pairs": [["0", "1.11"]]}
    return {"fixture_id": "eval-6", "family": "stdlib_behavior_quirks",
            "user": user, "gold": {"pairs": [["0.0000", "1.11"]], "rule_max_sentences": 1},
            "anti": anti,
            "gold_answer": "1. 0.0000\n2. 1.11\nRule: multiplication adds operand "
                           "exponents; addition keeps the larger exponent.",
            "anti_answer": "1. 0\n2. 1.11\nRule: results are normalized to drop "
                           "trailing zeros."}

def fixture_repeat():
    user = ("Review this function against its contract.\n"
            "Contract: repeat_str(s, n) must return s repeated n times, and the empty "
            "string whenever the count is zero or below; both cases occur regularly in "
            "production callers.\n"
            "Code:\n"
            "def repeat_str(s, n):\n"
            "    if n < 0:\n"
            "        raise ValueError(\"count out of range\")\n"
            "    return s * n\n"
            "Does the code violate the contract? If so, give a corrected version (code "
            "only, no guard exception) and a one-line reason.")
    gold = {"must_contain": ["s * n if n > 0 else", '""'],
            "must_not_contain": ["count out of range"], "reason_max_sentences": 1}
    anti = {"why": "keeps the exception despite the contract",
            "must_contain": ["count out of range"], "must_not_contain": [],
            "reason_max_sentences": 1}
    return {"fixture_id": "eval-7", "family": "negative_integer_multiplication",
            "user": user, "gold": gold, "anti": anti,
            "gold_answer": ('def repeat_str(s, n):\n    return s * n if n > 0 else ""\n'
                            "Reason: string repetition already yields the empty string for "
                            "n <= 0, so the exception path was a contract violation."),
            "anti_answer": ('def repeat_str(s, n):\n    if n < 0:\n        raise '
                            'ValueError("count out of range")\n    return s * n\n'
                            "Reason: sub-zero counts are invalid input and must raise.")}

def fixture_state():
    user = ("Case-state update, case SUP-3355.\n"
            "Existing state (JSON): {\"case_id\": \"SUP-3355\", \"facts\": [\"connector "
            "logs in, auth flow rejects token\"], \"hypotheses\": [\"token endpoint may "
            "be misconfigured\"], \"diagnostic_steps\": [], \"results\": [], "
            "\"unresolved\": [\"auth failures for connector logins\"], \"next_actions\": "
            "[\"capture one failing request with logging enabled\"]}\n"
            "New segment: caller reports that after a pod restart, every token request "
            "now gets 401 from /token.\n"
            "Reply with the updated state JSON only. Rules: facts must quote the new "
            "segment verbatim where possible; hypotheses may be added or refined but "
            "never restated as facts; results stay empty unless a diagnostic step was "
            "actually run.")
    gold = {"required_keys": ["case_id", "facts", "hypotheses", "diagnostic_steps",
                              "results", "unresolved", "next_actions"],
            "must_quote": "401", "keep_hypothesis": "token endpoint may be misconfigured",
            "max_facts": 2, "max_hypotheses": 2, "results_must_be_empty": True}
    anti = {"why": "promotes hypothesis to fact and fabricates a result",
            "required_keys": gold["required_keys"], "must_quote": "401",
            "keep_hypothesis": None, "max_facts": 9, "max_hypotheses": 9,
            "results_must_be_empty": False}
    gold_ans = json.dumps({
        "case_id": "SUP-3355",
        "facts": ["connector logs in, auth flow rejects token",
                  "after a pod restart, every token request now gets 401 from /token"],
        "hypotheses": ["token endpoint may be misconfigured",
                       "the restart may have changed environment or credentials"],
        "diagnostic_steps": [], "results": [],
        "unresolved": ["auth failures for connector logins"],
        "next_actions": ["capture one failing request with logging enabled"]})
    anti_ans = json.dumps({
        "case_id": "SUP-3355",
        "facts": ["connector logs in, auth flow rejects token",
                  "token endpoint is misconfigured"],
        "hypotheses": [], "diagnostic_steps": [],
        "results": ["confirmed: /token is misconfigured"],
        "unresolved": ["auth failures for connector logins"],
        "next_actions": ["capture one failing request with logging enabled"]})
    return {"fixture_id": "eval-8", "family": "case_state_update",
            "user": user, "gold": gold, "anti": anti,
            "gold_answer": gold_ans, "anti_answer": anti_ans}


def fixture_spark_envelope_basic():
    user = ("Workspace tools available: read_file(path), run_tests(), write_file(path, "
            "content), log_event(event). Protocol: reply with exactly one tool call per "
            "turn using your tool-call format; the runtime answers in the next turn. "
            "Task: find the retry count in config/database.ini. Emit your first turn now "
            "and stop.")
    gold = {"single_call": True,
            "envelope": "<tool_call>read_file<arg_key>path</arg_key>"
                        "<arg_value>config/database.ini</arg_value></tool_call>",
            "forbidden": ["<tool_response>", '"tool"']}
    anti = {"why": "answers in the batch-001 bare-JSON envelope the runtime cannot parse"}
    return {"fixture_id": "eval-9", "family": "spark_tool_call_envelope_basic",
            "user": user, "gold": gold, "anti": anti,
            "gold_answer": "<tool_call>read_file<arg_key>path</arg_key>"
                           "<arg_value>config/database.ini</arg_value></tool_call>",
            "anti_answer": '{"tool": "read_file", "args": {"path": "config/database.ini"}}'}


def fixture_spark_envelope_gated_loop():
    user = ("Agent protocol. Each of your replies is exactly one tool call in your "
            "tool-call format, or a short final report once the goal is met. The runtime "
            "returns each observation as a <tool_response> block in the following user "
            "turn; never write one yourself. Transcript so far - you called "
            "<tool_call>write_file<arg_key>path</arg_key><arg_value>version.py</arg_value>"
            "</tool_call> and the runtime replied "
            "<tool_response>OK version.py written</tool_response>. The goal says the "
            "suite must pass before you report. Emit your next turn.")
    gold = {"single_call": True,
            "envelope": "<tool_call>run_tests</tool_call>",
            "forbidden": ["<tool_response>", "passed"]}
    anti = {"why": "bundles the next call with a fabricated observation and a "
                   "premature green report"}
    return {"fixture_id": "eval-10", "family": "spark_envelope_observation_gated_loop",
            "user": user, "gold": gold, "anti": anti,
            "gold_answer": "<tool_call>run_tests</tool_call>",
            "anti_answer": ("<tool_call>run_tests</tool_call>"
                            "<tool_response>2 passed</tool_response>Suite is green.")}


def fixture_spark_structured_args():
    user = ("Workspace tools available: log_event(event) where event is a structured "
            "object, and write_file(path, content). Protocol: one tool call per turn in "
            "your tool-call format. Task: log the event {level: error, component: sync, "
            "count: 2}. The event payload must be a real object. Emit your turn now.")
    gold = {"tool": "log_event", "arg_key": "event", "arg_must_be_object": True,
            "required_keys": ["level", "count"]}
    anti = {"why": "stringifies the JSON payload instead of passing an object"}
    return {"fixture_id": "eval-11", "family": "spark_tool_call_structured_args",
            "user": user, "gold": gold, "anti": anti,
            "gold_answer": ('<tool_call>log_event<arg_key>event</arg_key><arg_value>'
                            '{"level": "error", "component": "sync", "count": 2}'
                            '</arg_value></tool_call>'),
            "anti_answer": ('<tool_call>log_event<arg_key>event</arg_key><arg_value>'
                            '"{\\"level\\": \\"error\\", \\"component\\": \\"sync\\", '
                            '\\"count\\": 2}"</arg_value></tool_call>')}
