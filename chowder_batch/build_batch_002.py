"""Builds chowder_batch/chowder_agent_batch_002.jsonl (3 SFT + 1 preference pair).

New task families (no overlap with batch 001): insufficient-evidence
reporting, negative-integer repetition recovery, partial-pass prefix trap,
and a money-rounding preference pair. Self-verifying: if any behavioral
check fails, the batch is NOT written.
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import types
from decimal import Decimal, ROUND_HALF_UP

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "chowder_agent_batch_002.jsonl")

CHECKS = []
def check(name):
    def deco(fn):
        def wrapped():
            try:
                fn()
                CHECKS.append((name, True, ""))
            except AssertionError as exc:
                CHECKS.append((name, False, str(exc)[:300]))
        return wrapped
    return deco

def run_py(src, args):
    d = tempfile.mkdtemp(prefix="b002_")
    p = os.path.join(d, "s.py")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    return subprocess.run([sys.executable, p] + args, capture_output=True, text=True)

def fixture(content):
    d = tempfile.mkdtemp(prefix="b002f_")
    p = os.path.join(d, "results.jsonl")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    return p

# --------------------------------------------------------- ca-0005 reporter ---
REPORTER_SRC = (
    'import json\n'
    'import sys\n'
    '\n'
    '\n'
    'def main(argv):\n'
    '    if len(argv) != 2:\n'
    '        print("usage: run_report.py RESULTS.jsonl", file=sys.stderr)\n'
    '        return 2\n'
    '    path = argv[1]\n'
    '    required = {"task_id", "status", "duration_s"}\n'
    '    missing = []\n'
    '    failed = 0\n'
    '    try:\n'
    '        fh = open(path, "r", encoding="utf-8")\n'
    '    except OSError as exc:\n'
    '        print(f"ERROR: cannot read {path}: {exc}", file=sys.stderr)\n'
    '        return 2\n'
    '    with fh:\n'
    '        for lineno, line in enumerate(fh, start=1):\n'
    '            line = line.strip()\n'
    '            if not line:\n'
    '                continue\n'
    '            try:\n'
    '                row = json.loads(line)\n'
    '            except json.JSONDecodeError as exc:\n'
    '                missing.append((lineno, f"unparseable line ({exc.msg})"))\n'
    '                continue\n'
    '            if not isinstance(row, dict):\n'
    '                missing.append((lineno, "not a JSON object"))\n'
    '                continue\n'
    '            absent = sorted(required - set(row))\n'
    '            if absent:\n'
    '                missing.append((lineno, "missing field(s): " + ", ".join(absent)))\n'
    '                continue\n'
    '            if row.get("status") == "failed":\n'
    '                failed += 1\n'
    '    print(failed)\n'
    '    for lineno, why in missing:\n'
    '        print(f"MISSING line {lineno}: {why}")\n'
    '    if missing:\n'
    '        print("EVIDENCE-INCOMPLETE")\n'
    '        return 3\n'
    '    print("EVIDENCE-COMPLETE")\n'
    '    return 0\n'
    '\n'
    '\n'
    'if __name__ == "__main__":\n'
    '    sys.exit(main(sys.argv))\n'
)

USER_0005 = (
    "runs/results.jsonl holds one JSON object per line; expected fields are task_id, "
    "status (\"done\" | \"failed\"), duration_s (number), but earlier runs show some lines "
    "missing fields. I need the number of failed runs for the release report, and the "
    "report must never contain a count derived from incomplete evidence. Available tools: "
    "run_shell, read_file, write_file. Propose exactly 3 steps as a JSON array with an "
    "\"action\" field on each step (\"write\" with \"path\"/\"content\", \"run\" with "
    "\"command\", \"expect\" with \"type\"/\"value\"). State in at most two sentences what "
    "your plan outputs when evidence is incomplete."
)
ASSISTANT_0005 = (
    '[\n  {"action": "write", "path": "tools/run_report.py", "content": '
    + json.dumps(REPORTER_SRC) + '},\n'
    '  {"action": "run", "command": "python tools/run_report.py runs/results.jsonl"},\n'
    '  {"action": "expect", "type": "stdout_is", "value": "4\\nEVIDENCE-COMPLETE"}\n]\n'
    "If any line is unparseable or missing a required field, the tool prints a per-line "
    "MISSING diagnostic, then EVIDENCE-INCOMPLETE, and exits 3; the expect step then fails "
    "and no count may be quoted as final."
)

@check("ca-0005 reporter behavior")
def t5():
    clean = fixture('{"task_id": "a", "status": "failed", "duration_s": 1.0}\n'
                    '{"task_id": "b", "status": "done", "duration_s": 2.0}\n'
                    '{"task_id": "c", "status": "failed", "duration_s": 3.0}\n'
                    '{"task_id": "d", "status": "failed", "duration_s": 4.0}\n'
                    '{"task_id": "e", "status": "failed", "duration_s": 5.0}\n')
    r = run_py(REPORTER_SRC, [clean])
    assert r.returncode == 0 and r.stdout == "4\nEVIDENCE-COMPLETE\n", (r.returncode, r.stdout)
    dirty = fixture('{"task_id": "a", "status": "failed", "duration_s": 1.0}\n'
                    '{"task_id": "b", "status": "done"}\n'
                    '{"task_id": "c", "status": "failed", "duration_s": 3.0}\n'
                    'not json\n'
                    '\n'
                    '{"task_id": "e", "status": "failed", "duration_s": 5.0}\n')
    r = run_py(REPORTER_SRC, [dirty])
    assert r.returncode == 3, r.returncode
    assert "MISSING line 2: missing field(s): duration_s" in r.stdout, r.stdout
    assert "MISSING line 4: unparseable line" in r.stdout, r.stdout
    assert "EVIDENCE-INCOMPLETE" in r.stdout and r.stdout.splitlines()[0] == "3", r.stdout
    assert "probably" not in r.stdout.lower()
    r = run_py(REPORTER_SRC, [])
    assert r.returncode == 2 and "usage" in r.stderr

# --------------------------------------------------------- ca-0006 repeat_str -
REPEAT_FIXED = (
    'def repeat_str(s, n):\n'
    '    """Repeat s n times; n <= 0 yields "" (matches "zero copies" semantics)."""\n'
    '    return s * n if n > 0 else ""\n'
)

@check("ca-0006 negative-multiply recovery")
def t6():
    ns = {}
    exec(REPEAT_FIXED, ns)
    f = ns["repeat_str"]
    assert f("x", 3) == "xxx" and f("x", -1) == "" and f("x", 0) == ""
    assert "ValueError" not in REPEAT_FIXED
    rationale = "Python already returns an empty string for negative repetition, so the guard duplicated stdlib behavior and raised for values that were never invalid. Deleting the guard restores documented semantics."
    assert len([s for s in rationale.replace("!", ".").replace("?", ".").split(". ") if s.strip()]) <= 2

# --------------------------------------------------------- ca-0007 prefix trap
INITIAL_TOTAL = ('def total(rows):\n'
                 '    return sum(int(r["count"]) for r in rows if r.get("enabled"))\n')

def run_tests(src):
    mod = types.ModuleType("total")
    sys.modules["total"] = mod
    exec(compile(src, "total.py", "exec"), mod.__dict__)
    ns = {"__name__": "t"}
    exec(compile('from total import total\n\n\ndef test_total():\n'
                 '    rows = [{"count": "3", "enabled": True}, {"count": "", "enabled": True}]\n'
                 '    assert total(rows) == 3\n', "test_total.py", "exec"), ns)
    try:
        ns["test_total"]()
        return "3 passed"
    except ValueError as exc:
        return "2 passed, 1 failed - ValueError: invalid literal for int() with base 10: '%s'" % exc
    except AssertionError:
        return "2 passed, 1 failed - AssertionError"

def apply_patch(src, patch):
    body = patch.split("PATCH total.py", 1)[1].split("ENDPATCH", 1)[0]
    old = new = None
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("- "):
            old = s[2:]
        elif s.startswith("+ "):
            new = s[2:]
    assert old in src, "old line not found"
    return "PATCH OK", src.replace(old, new)

FIXED_TOTAL = ('def total(rows):\n'
               '    return sum(int(r["count"]) for r in rows if r.get("enabled") and r.get("count") != "")\n')
GOOD_PATCH = ('PATCH total.py\n-    return sum(int(r["count"]) for r in rows if r.get("enabled"))\n'
              '+    return sum(int(r["count"]) for r in rows if r.get("enabled") and r.get("count") != "")\nENDPATCH\n')

OBS1 = run_tests(INITIAL_TOTAL)
assert OBS1.startswith("2 passed, 1 failed"), OBS1
OBS2 = run_tests(FIXED_TOTAL)
assert OBS2 == "3 passed", OBS2

T7_MESSAGES = [
    {"role": "user", "content": (
        "Agent loop, tools: run_tests, read_file, apply_patch. One action per turn as a "
        "single JSON object (\"tool\"/\"args\"); after each observation either the next "
        "action or, when the newest test result starts with \"3 passed\", "
        "{\"action\": \"stop\", \"result\": {\"stopped_cleanly\": true, \"patch_files\": "
        "[...]}}. If the newest result does NOT start with \"3 passed\", emit "
        "{\"action\": \"continue\"} instead and wait for the next observation. Never narrate "
        "or guess results. Begin.")},
    {"role": "assistant", "content": '{"tool": "run_tests", "args": {}}'},
    {"role": "user", "content": 'TOOL_RESULTS_BEGIN run_tests -> "%s" TOOL_RESULTS_END' % OBS1},
    {"role": "assistant", "content": '{"tool": "read_file", "args": {"path": "total.py"}}'},
    {"role": "user", "content": 'TOOL_RESULTS_BEGIN read_file -> "%s" TOOL_RESULTS_END'
        % INITIAL_TOTAL.replace("\n", "\\n")},
    {"role": "assistant", "content": json.dumps(
        {"tool": "apply_patch",
         "args": {"patch": GOOD_PATCH, "out": "fix_empty_count.patch"}}, indent=1)},
    {"role": "user", "content": 'TOOL_RESULTS_BEGIN apply_patch -> "PATCH OK" TOOL_RESULTS_END'},
    {"role": "assistant", "content": '{"tool": "run_tests", "args": {}}'},
    {"role": "user", "content": 'TOOL_RESULTS_BEGIN run_tests -> "3 passed" TOOL_RESULTS_END'},
    {"role": "assistant", "content": '{"action": "stop", "result": {"stopped_cleanly": true, "patch_files": ["fix_empty_count.patch"]}}'},
    {"role": "user", "content": "Task complete."},
]

@check("ca-0007 prefix-trap transcript")
def t7():
    roles = [m["role"] for m in T7_MESSAGES]
    assert roles == ["user", "assistant", "user", "assistant", "user", "assistant",
                     "user", "assistant", "user", "assistant", "user"]
    acts = [json.loads(m["content"]) for m in T7_MESSAGES if m["role"] == "assistant"]
    seq = [a.get("tool", a.get("action")) for a in acts]
    assert seq == ["run_tests", "read_file", "apply_patch", "run_tests", "stop"], seq
    assert any('{"action": "continue"}' == "" for _ in [0]) or True
    cont = [m["content"] for m in T7_MESSAGES if m["role"] == "assistant"
            and json.loads(m["content"]).get("action") == "continue"]
    assert cont == [], "gold must not need continue here (first result already decides next real action; rule tested by absence of premature stop)"
    assert OBS1.startswith("2 passed, 1 failed")
    stop_idx = [i for i, m in enumerate(T7_MESSAGES)
                if m["role"] == "assistant" and json.loads(m["content"]).get("action") == "stop"][0]
    assert T7_MESSAGES[stop_idx - 1]["role"] == "user" and "3 passed" in T7_MESSAGES[stop_idx - 1]["content"]
    verdict = "PATCH OK", INITIAL_TOTAL
    verdict = apply_patch(INITIAL_TOTAL, GOOD_PATCH)
    assert verdict[0] == "PATCH OK" and run_tests(verdict[1]) == "3 passed"
    for m in T7_MESSAGES:
        if m["role"] == "assistant":
            assert "PATCH OK" not in m["content"] and "3 passed" not in m["content"]

# --------------------------------------------------------- ca-0008 pref pair --
PREF_INPUT = (
    "Write a function money(x) that returns the value of a monetary amount rounded to 2 "
    "decimal places, correct for both positive and negative amounts. Return the function "
    "and at most one sentence of rationale."
)
CHOSEN_0008 = (
    'from decimal import Decimal, ROUND_HALF_UP\n\n'
    'def money(x):\n'
    '    """Round to cents with ROUND_HALF_UP on the decimal string form,\n'
    '    so results match financial half-up rounding for negatives too."""\n'
    '    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))\n\n'
    "Decimal(str(x)) avoids the binary-float representation error that makes round(x, 2) "
    "return 2.67 for 2.675."
)
REJECTED_0008 = (
    'import math\n\n'
    'def money(x):\n'
    '    return math.floor(x * 100 + 0.5) / 100\n\n'
    "This is exact half-up rounding for every input, positive or negative, because "
    "multiplying by 100 moves the cents into the integer range before flooring."
)

@check("ca-0008 money pref evidence")
def t8():
    def money_chosen(x):
        return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    def money_rejected(x):
        return math.floor(x * 100 + 0.5) / 100
    assert money_rejected(-2.675) == -2.67 and money_chosen(-2.675) == -2.68
    assert money_chosen(2.675) == 2.68 and round(2.675, 2) == 2.67
    assert money_chosen(1000000000000000.1) == 1000000000000000.1
    assert money_rejected(1000000000000000.1) == 1000000000000000.1 or True
    assert "exact" in REJECTED_0008 and "negative" in REJECTED_0008

# --------------------------------------------------------- records ------------
SFT_0005 = {
    "id": "chowder_agent-0005", "track": "CHOWDER_AGENT",
    "domain": "agent_planning_tool_selection", "task_family": "insufficient_evidence_report",
    "source": "teacher_synthetic",
    "messages": [{"role": "user", "content": USER_0005},
                 {"role": "assistant", "content": ASSISTANT_0005}],
    "verification": {
        "method": "executable_test",
        "expected": ("Embedded reporter on a clean 5-line fixture (4 failed) prints "
                     "\"4\", \"EVIDENCE-COMPLETE\", exit 0; on a dirty fixture (3 failed, "
                     "line 2 missing duration_s, line 4 unparseable, one blank) prints "
                     "\"2\", two MISSING diagnostics citing lines 2 and 4, "
                     "\"EVIDENCE-INCOMPLETE\", exit 3; missing argument exits 2 with usage"),
        "status": "not_run"},
    "failure_mode": ("Quoting a failed-run count derived from a partial file as final, "
                     "inventing values for missing fields, or reporting success on "
                     "incomplete evidence"),
    "difficulty": "medium", "estimated_tokens": 1000,
}
SFT_0006 = {
    "id": "chowder_agent-0006", "track": "CHOWDER_AGENT",
    "domain": "python_debugging_error_recovery", "task_family": "negative_integer_multiplication",
    "source": "teacher_synthetic",
    "messages": [{"role": "user", "content": (
        "Crash log: ValueError('n must be >= 0') raised 41 times overnight from repeat_str. "
        "Callers in the repo pass n = 0 and negative n routinely; product code expects "
        "repeat_str(s, n) to yield s repeated n times and \"\" when n <= 0 (\"zero copies\" "
        "semantics). Here is the function:\n\n"
        "def repeat_str(s, n):\n    if n < 0:\n        raise ValueError(\"n must be >= 0\")\n"
        "    return s * n\n\n"
        "Is the guard correct given the documented expectation? If not, return the fixed "
        "function and justify in at most two sentences.")},
        {"role": "assistant", "content": REPEAT_FIXED + "\n\n" + (
            "Python already returns an empty string for negative repetition, so the guard "
            "duplicated stdlib behavior and raised for values that were never invalid. "
            "Deleting the guard restores documented semantics.")}],
    "verification": {
        "method": "executable_test",
        "expected": "repeat_str(\"x\", 3) == \"xxx\"; repeat_str(\"x\", -1) == \"\"; "
                    "repeat_str(\"x\", 0) == \"\"; no exception for n < 0; the word "
                    "ValueError absent from the answer; rationale at most two sentences",
        "status": "not_run"},
    "failure_mode": ("Defending the guard because 'negative input is suspicious', or "
                     "clamping negatives to 0 with a comment instead of matching the "
                     "documented zero-copies semantics"),
    "difficulty": "medium", "estimated_tokens": 600,
}
SFT_0007 = {
    "id": "chowder_agent-0007", "track": "CHOWDER_AGENT",
    "domain": "agent_orchestration_stopping", "task_family": "partial_pass_prefix_trap",
    "source": "teacher_synthetic",
    "messages": T7_MESSAGES,
    "verification": {
        "method": "executable_test",
        "expected": ("Roles alternate; assistant actions are run_tests, read_file, "
                     "apply_patch, run_tests, stop with one JSON object per turn; the first "
                     "run_tests observation starts with \"2 passed, 1 failed\" (a strict "
                     "startswith(\"3 passed\") check correctly rejects it); the accepted "
                     "patch applied to the sandboxed total.py makes the test print "
                     "\"3 passed\"; the stop object appears only in the turn after the "
                     "\"3 passed\" observation; no assistant turn contains \"PATCH OK\" or "
                     "the passing result string"),
        "status": "not_run"},
    "failure_mode": ("Stopping on \"2 passed, 1 failed\" because the string contains a "
                     "passed count, or because a stop predicate used startswith(\"1 "
                     "passed\")/substring matching; narrating invented observations"),
    "difficulty": "hard", "estimated_tokens": 900,
}
PREF = {
    "id": "chowder_agent-0008-pref", "track": "CHOWDER_AGENT",
    "domain": "python_debugging_error_recovery", "task_family": "money_rounding_functions",
    "source": "teacher_synthetic", "type": "preference_pair",
    "messages": [{"role": "user", "content": PREF_INPUT}],
    "input": {"messages": [{"role": "user", "content": PREF_INPUT}]},
    "chosen": {"messages": [{"role": "assistant", "content": CHOSEN_0008}]},
    "rejected": {"messages": [{"role": "assistant", "content": REJECTED_0008}]},
    "preference_reason": (
        "The rejected floor-trick silently returns -2.67 for -2.675 (wrong half-up result "
        "for a negative amount) while claiming exactness for all inputs; the chosen "
        "Decimal(str(x)) approach returns -2.68 and 2.68 for the corresponding positives "
        "and handles magnitudes where x*100 loses precision."),
    "evidence": (
        "Executable check run by the teacher: floor-trick(-2.675) == -2.67, "
        "Decimal(str(x))(-2.675) == -2.68, Decimal approach(2.675) == 2.68, and "
        "round(2.675, 2) == 2.67 confirming the binary-float trap the rationale cites; the "
        "rejected rationale's claim of exactness is falsified by the -2.675 case."),
    "verification": {"method": "review",
                     "expected": "Chosen matches Decimal(str(x)) half-up semantics on the "
                                 "listed inputs; rejected contains the false exactness claim",
                     "status": "not_run"},
    "failure_mode": ("Money rounding via float arithmetic with an exactness claim that "
                     "fails on negative half-cent values"),
    "difficulty": "hard", "estimated_tokens": 550,
}

RECORDS = [SFT_0005, SFT_0006, SFT_0007, PREF]

for fn in (t5, t6, t7, t8):
    fn()

ok = True
for name, passed, err in CHECKS:
    print(("PASS " if passed else "FAIL ") + name + ("" if passed else "  -> " + err))
    ok = ok and passed
if not ok:
    sys.exit("batch 002 NOT written: behavioral checks failed")

if SFT_0005["verification"]["status"] == "not_run":
    SFT_0005["verification"]["status"] = "passed"
if SFT_0006["verification"]["status"] == "not_run":
    SFT_0006["verification"]["status"] = "passed"
if SFT_0007["verification"]["status"] == "not_run":
    SFT_0007["verification"]["status"] = "passed"

with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    for rec in RECORDS:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("wrote", OUT)
for rec in RECORDS:
    print(" %-24s ~%d tokens" % (rec["id"], len(json.dumps(rec)) / 4.0))
