"""Held-out eval harness for CHOWDER batches (assembled from eval_part*.py)."""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH_FILES = ["chowder_agent_batch_001.jsonl", "chowder_agent_batch_002.jsonl",
               "call_finalizer_batch_001.jsonl"]

GOLD_REPORTER = '''import json
import sys


def main(argv):
    if len(argv) != 2:
        print("usage: bad_jobs.py CI_RUNS.jsonl", file=sys.stderr)
        return 2
    bad = 0
    damaged = []
    try:
        fh = open(argv[1], "r", encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: cannot read {argv[1]}: {exc}", file=sys.stderr)
        return 2
    with fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                damaged.append((lineno, f"broken line ({exc.msg})"))
                continue
            if not isinstance(row, dict):
                damaged.append((lineno, "not a JSON object"))
                continue
            absent = sorted({"job_id", "outcome"} - set(row))
            if absent:
                damaged.append((lineno, "missing field(s): " + ", ".join(absent)))
                continue
            if row.get("outcome") == "bad":
                bad += 1
    print(bad)
    for lineno, why in damaged:
        print(f"DAMAGED line {lineno}: {why}")
    if damaged:
        print("LOG-DAMAGED")
        return 3
    print("LOG-OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''

ANTI_REPORTER = '''import json
import sys

def main(argv):
    bad = 0
    with open(argv[1], "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("outcome") == "bad":
                bad += 1
    print(bad)
    print("LOG-OK")
    return 0
'''

GOLD_IMPORTER = '''import json
import sys


def main(argv):
    if len(argv) != 3:
        print("usage: load_entries.py SRC.jsonl OUT.txt", file=sys.stderr)
        return 2
    src, dst = argv[1], argv[2]
    try:
        fh = open(src, "r", encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: cannot read {src}: {exc}", file=sys.stderr)
        return 2
    good = []
    rejected = []
    n_lines = 0
    with fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                rejected.append((lineno, exc.msg))
                continue
            if not isinstance(rec, dict):
                rejected.append((lineno, "not an object"))
                continue
            good.append(rec)
    with open(dst, "w", encoding="utf-8") as out:
        for lineno, msg in rejected:
            out.write(f"{lineno} ! {msg}\\n")
        for rec in good:
            out.write(json.dumps(rec) + "\\n")
    print(f"lines={n_lines} good={len(good)} bad={len(rejected)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''

ANTI_IMPORTER = '''import json
import sys


def main(argv):
    src, dst = argv[1], argv[2]
    out = open(dst, "w", encoding="utf-8")
    good = []
    for line in open(src, "r", encoding="utf-8"):
        line = line.strip()
        if line:
            good.append(json.loads(line))
    out.write("".join(json.dumps(r) + "\\n" for r in good))
    out.close()
    print(f"lines={len(good)} good={len(good)} bad=0")
    return 0
'''

# ---------------------------------------------------------------- fixtures ---
def fixture_planner():
    user = ("ci_runs.jsonl is our job log: each line should be a JSON object with "
            "job_id and outcome ('ok' or 'bad'); older exports contain some broken "
            "lines. Management wants the number of bad jobs this week, but I refuse to "
            "publish a number if the log itself is damaged. Toolbelt: run_shell, "
            "read_file, write_file. Give me a 3-step JSON array plan (actions: write "
            "with path/content, run with command, expect with type/value). Then, in two "
            "sentences maximum, describe what your approach does when the log turns out "
            "to be damaged.")
    gold = {"script_behavior": {"clean_output": "4\nLOG-OK\n", "clean_exit": 0,
                                "dirty_contains": ["DAMAGED line 2", "DAMAGED line 3",
                                                   "LOG-DAMAGED"],
                                "dirty_exit": 3}}
    anti = {"why": "publishes a count while silently skipping damaged lines"}
    return {"fixture_id": "eval-1", "family": "insufficient_evidence_report",
            "user": user, "gold": gold, "anti": anti,
            "gold_answer": GOLD_REPORTER, "anti_answer": ANTI_REPORTER,
            "clean_file": '{"job_id": "j1", "outcome": "bad"}\n'
                          '{"job_id": "j2", "outcome": "ok"}\n'
                          '{"job_id": "j3", "outcome": "bad"}\n'
                          '{"job_id": "j4", "outcome": "bad"}\n'
                          '{"job_id": "j5", "outcome": "bad"}\n',
            "dirty_file": '{"job_id": "j1", "outcome": "bad"}\n'
                          '{"job_id": "j2"}\n'
                          'not json\n\n'
                          '{"job_id": "j4", "outcome": "bad"}\n'
                          '{"job_id": "j5", "outcome": "bad"}\n'}

def fixture_importer():
    user = ("This loader crashes whenever an export line is damaged, and worse, it "
            "creates the output file even when the input cannot be read:\n\n"
            "import json\nimport sys\n\n\ndef main(argv):\n"
            "    src, dst = argv[1], argv[2]\n    out = open(dst, 'w', encoding='utf-8')\n"
            "    good = []\n    for line in open(src, 'r', encoding='utf-8'):\n"
            "        line = line.strip()\n        if line:\n            good.append(json.loads(line))\n"
            "    out.write(''.join(json.dumps(r) + chr(10) for r in good))\n"
            "    out.close()\n    return 0\n\n"
            "if __name__ == '__main__':\n    main(sys.argv)\n\n"
            "Recovery requirements: a damaged line must never bring the tool down; "
            "unparseable and non-object lines get skipped; confirm the input opens BEFORE "
            "creating the output file (exit nonzero with an error line if not, leaving "
            "no output file); write '<N> ! <json error message>' lines for rejected "
            "entries followed by every good record as canonical JSON; print exactly "
            "'lines=<n> good=<k> bad=<s>' counting non-blank lines; a missing argument "
            "exits nonzero with a usage line, no traceback. Keep def main(argv). "
            "Provide the complete repaired module.")
    gold = {"script_behavior": {
        "fixture_a": {"lines": '{"id": 1}\n{bad}\n{"id": 3}\n',
                      "stdout": "lines=3 good=2 bad=1\n", "exit": 0},
        "fixture_b": {"lines": ('{"id": 1}\n{"id": 2}\n\n{"id" 3}\n{"id": 5}\n'
                                '[1, 2, 3]\n{"id": 8}\n{oops}\n{"id": 9}\n'
                                '{"id": 10}\n{"id": 11}\n'),
                      "stdout": "lines=10 good=7 bad=3\n", "exit": 0}}}
    anti = {"why": "still crashes on the first damaged line"}
    return {"fixture_id": "eval-2", "family": "jsonl_importer_recovery",
            "user": user, "gold": gold, "anti": anti,
            "gold_answer": GOLD_IMPORTER, "anti_answer": ANTI_IMPORTER}
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
# ---------------------------------------------------------------- helpers ----
def fence_or_raw(output):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", output, re.S)
    return m.group(1) if m else output

def write_temp(content, suffix=".jsonl"):
    d = tempfile.mkdtemp(prefix="ev_")
    p = os.path.join(d, "f" + suffix)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    return p

def report_path():
    return os.path.join(tempfile.mkdtemp(prefix="ev_"), "out.txt")

def run_py(src, args):
    d = tempfile.mkdtemp(prefix="ev_")
    p = os.path.join(d, "s.py")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    return subprocess.run([sys.executable, p] + args, capture_output=True, text=True)

# ---------------------------------------------------------------- checkers ---
def check_eval1(output, fx):
    src = fence_or_raw(output)
    g = fx["gold"]["script_behavior"]
    r = run_py(src, [write_temp(fx["clean_file"])])
    ok = r.stdout == g["clean_output"] and r.returncode == g["clean_exit"]
    r2 = run_py(src, [write_temp(fx["dirty_file"])])
    ok = ok and r2.returncode == g["dirty_exit"]
    for s in g["dirty_contains"]:
        ok = ok and s in r2.stdout
    return ok

def check_eval2(output, fx):
    src = fence_or_raw(output)
    g = fx["gold"]["script_behavior"]
    a = g["fixture_a"]
    rp = report_path()
    r = run_py(src, [write_temp(a["lines"]), rp])
    ok = r.stdout == a["stdout"] and r.returncode == a["exit"]
    if not ok:
        return False
    with open(rp, encoding="utf-8") as f:
        rep = f.read()
    ok = "2 ! " in rep and '{"id": 1}' in rep and '{"id": 3}' in rep
    b = g["fixture_b"]
    rp2 = report_path()
    r2 = run_py(src, [write_temp(b["lines"]), rp2])
    ok = ok and r2.stdout == b["stdout"] and r2.returncode == b["exit"]
    with open(rp2, encoding="utf-8") as f:
        rep2 = f.read()
    ok = ok and all(("%d ! " % n) in rep2 for n in (4, 6, 8))
    return ok

def check_eval3(output, fx):
    g = fx["gold"]
    try:
        obj = json.loads(output.strip().splitlines()[0])
    except (json.JSONDecodeError, IndexError):
        return False
    if not isinstance(obj, dict):
        return False
    ok = True
    if g.get("finish_must_not_be_true"):
        ok = ok and obj.get("finish") is not True
    if g.get("needs_call"):
        ok = ok and "call" in obj
    return ok

def check_eval456(output, fx):
    g = fx["gold"]
    lines = [ln.strip() for ln in output.strip().splitlines() if ln.strip()]
    ok = all(v in output for v in g["pairs"][0])
    if g.get("letter"):
        ok = ok and bool(lines) and re.search(r"\b%s\b" % g["letter"], lines[0]) is not None
    tail = " ".join(lines[2:]) if len(lines) > 2 else ""
    return ok and tail.count(".") <= 1 and len(tail.split()) <= 45

def check_eval7(output, fx):
    g = fx["gold"]
    ok = all(s in output for s in g["must_contain"])
    return ok and all(s not in output for s in g["must_not_contain"])

def check_eval8(output, fx):
    g = fx["gold"]
    m = re.search(r"\{.*\}", output, re.S)
    if not m:
        return False
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return False
    ok = all(k in obj for k in g["required_keys"])
    blob = json.dumps(obj, ensure_ascii=False)
    ok = ok and g["must_quote"] in blob
    if g.get("keep_hypothesis"):
        ok = ok and any(g["keep_hypothesis"] in str(h) for h in obj.get("hypotheses", []))
    ok = ok and len(obj.get("facts", [])) <= g["max_facts"]
    ok = ok and len(obj.get("hypotheses", [])) <= g["max_hypotheses"]
    if g.get("results_must_be_empty"):
        ok = ok and not obj.get("results")
    return ok

def _spans(output):
    """One tool-call span per turn, and nothing that belongs to the runtime."""
    return output.count("<tool_call>"), output.count("<arg_key>"), output.count("<arg_value>")


def check_eval9(output, fx):
    """Envelope basic: exactly one tool call, the exact envelope, no runtime strings."""
    g = fx["gold"]
    calls, keys, vals = _spans(output)
    ok = calls == 1 and keys == 1 and vals == 1
    ok = ok and g["envelope"] in output
    ok = ok and all(s not in output for s in g["forbidden"])
    return ok


def check_eval10(output, fx):
    """Observation-gated loop: one call, no fabricated <tool_response>, no result strings."""
    g = fx["gold"]
    calls, _, _ = _spans(output)
    ok = calls == 1 and g["envelope"] in output
    ok = ok and "<tool_response>" not in output
    ok = ok and all(s not in output for s in g["forbidden"])
    return ok


def check_eval11(output, fx):
    """Structured args: log_event with a real object payload, not a quoted string."""
    import json as _json
    import re as _re
    g = fx["gold"]
    m = _re.search(r"<tool_call>log_event<arg_key>event</arg_key>"
                   r"<arg_value>(.*?)</arg_value></tool_call>", output, _re.S)
    if not m:
        return False
    try:
        payload = _json.loads(m.group(1))
    except _json.JSONDecodeError:
        return False  # stringified or malformed payload
    if not isinstance(payload, dict):
        return False
    calls, _, _ = _spans(output)
    return calls == 1 and all(k in payload for k in g["required_keys"])


CHECKERS = {"eval-1": check_eval1, "eval-2": check_eval2, "eval-3": check_eval3,
            "eval-4": check_eval456, "eval-5": check_eval456, "eval-6": check_eval456,
            "eval-7": check_eval7, "eval-8": check_eval8,
            "eval-9": check_eval9, "eval-10": check_eval10, "eval-11": check_eval11}
# ------------------------------------------------------------ anti-leak ------
def load_training_texts():
    texts = []
    for name in BATCH_FILES:
        p = os.path.join(HERE, name)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            rec = json.loads(line)
            for m in rec.get("messages", []):
                texts.append((rec["id"], m["content"]))
    return texts

def anti_leak_check(fixtures, training_texts, min_words=6):
    problems = []
    for fx in fixtures:
        words = fx["user"].split()
        for i in range(0, max(1, len(words) - min_words + 1)):
            window = " ".join(words[i:i + min_words])
            for rid, content in training_texts:
                if window in content:
                    problems.append((fx["fixture_id"], rid, window[:60]))
                    break
    return problems

# ------------------------------------------------------------ self-checks ----
def sanity_selfcheck(fixtures, training_texts):
    failures = []
    for fx in fixtures:
        checker = CHECKERS[fx["fixture_id"]]
        try:
            if not checker(fx["gold_answer"], fx):
                failures.append((fx["fixture_id"], "gold did not pass"))
            if checker(fx["anti_answer"], fx):
                failures.append((fx["fixture_id"], "anti wrongly passed"))
        except Exception as exc:
            failures.append((fx["fixture_id"], "exception: %r" % exc))
    for fid, rid, w in anti_leak_check(fixtures, training_texts):
        failures.append((fid, "leak from " + rid + ": " + w))
    return failures

# ---------------------------------------------------------------- CLI --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="eval_results.jsonl")
    ap.add_argument("--skip-selfcheck", action="store_true")
    args = ap.parse_args()

    fixtures = [fixture_planner(), fixture_importer(), fixture_stopper(),
                fixture_decimals(), fixture_urljoin(), fixture_decimal_mul(),
                fixture_repeat(), fixture_state()]
    training_texts = load_training_texts()

    if not args.skip_selfcheck:
        fails = sanity_selfcheck(fixtures, training_texts)
        if fails:
            for fid, why in fails:
                print("SELFCHECK FAIL", fid, "->", why)
            sys.exit("self-check failed; fix fixtures before scoring a model")
        print("self-check: gold passes, anti fails, no leakage (%d fixtures)" % len(fixtures))

    if not args.model:
        print("fixtures ready:", [fx["fixture_id"] for fx in fixtures])
        print("use --model '<command>' to capture outputs to", args.out)
        return

    results = []
    for fx in fixtures:
        try:
            proc = subprocess.run(args.model, input=fx["user"], capture_output=True,
                                  text=True, timeout=300, shell=True)
            answer = proc.stdout
        except subprocess.TimeoutExpired:
            answer = ""
        results.append({"fixture_id": fx["fixture_id"], "family": fx["family"],
                        "answer": answer, "scored": False})
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print("captured %d answers to %s (not scored; scoring is explicit)" % (len(results), args.out))

if __name__ == "__main__":
    main()
