"""Builds chowder_batch/chowder_agent_batch_001.jsonl (3 SFT + 1 preference pair).

All records are emitted with json.dumps so nested JSON is escaped correctly by
construction. Tool observations in the 0003 transcript are produced by actually
executing the mock tool environment below.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "chowder_agent_batch_001.jsonl")

# ---------------------------------------------------------------- mock tool env
INITIAL_FILE = 'def div(a, b):\n    return a / b\n'

def run_tests(source):
    import sys
    import types
    mod = types.ModuleType("calc")
    sys.modules["calc"] = mod
    exec(compile(source, "calc.py", "exec"), mod.__dict__)
    ns = {"__name__": "test_calc"}
    exec(compile("from calc import div\n\n\ndef test_div_by_zero():\n    assert div(4, 0) == 'inf'\n", "test_calc.py", "exec"), ns)
    try:
        ns["test_div_by_zero"]()
        return "1 passed"
    except ZeroDivisionError:
        return "FAILED tests/test_calc.py::test_div_by_zero - ZeroDivisionError"
    except AssertionError:
        return "FAILED tests/test_calc.py::test_div_by_zero - AssertionError"

def apply_patch(source, patch):
    if "PATCH calc.py" not in patch or "ENDPATCH" not in patch:
        return ("ERROR apply_patch: expected 'PATCH calc.py' header and ENDPATCH marker", source)
    body = patch.split("PATCH calc.py", 1)[1].split("ENDPATCH", 1)[0]
    old = new = None
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("- "):
            old = s[2:]
        elif s.startswith("+ "):
            new = s[2:]
    if old is None or old not in source:
        return "ERROR apply_patch: old code not found in calc.py"
    return "PATCH OK", source.replace(old, new if new is not None else old)

# --- execute the recorded agent actions to obtain real observations -----------
OBS_U2 = run_tests(INITIAL_FILE)                      # ZeroDivisionError failure
OBS_U4 = apply_patch(INITIAL_FILE, "REPLACE calc.py\nREPLACE return a / b\nWITH return 'inf' if b == 0 else a / b\nENDREPLACE\n")[0]
OBS_U6 = apply_patch(INITIAL_FILE,
    "PATCH calc.py\n-    return a / b\n+    return 'inf' if b == 0 else a / b\nENDPATCH\n")
assert OBS_U4.startswith("ERROR apply_patch"), OBS_U4
assert OBS_U6[0] == "PATCH OK", OBS_U6
FIXED_SOURCE = OBS_U6[1]
OBS_U8 = run_tests(FIXED_SOURCE)
assert OBS_U8 == "1 passed", OBS_U8

TOOL_BLOCK_U2 = 'TOOL_RESULTS_BEGIN run_tests -> "%s" TOOL_RESULTS_END' % OBS_U2
TOOL_BLOCK_U4 = ('TOOL_RESULTS_BEGIN read_file {"path": "calc.py"} -> '
                 '"def div(a, b):\\n    return a / b\\n" TOOL_RESULTS_END')
TOOL_BLOCK_U6 = 'TOOL_RESULTS_BEGIN apply_patch -> "%s" TOOL_RESULTS_END' % OBS_U4
TOOL_BLOCK_U8 = 'TOOL_RESULTS_BEGIN apply_patch -> "PATCH OK" TOOL_RESULTS_END'
TOOL_BLOCK_U10 = 'TOOL_RESULTS_BEGIN run_tests -> "1 passed" TOOL_RESULTS_END'

# ---------------------------------------------------------------- 0001 planner
PLANNER_SCRIPT = (
    'import json\n'
    'import sys\n'
    '\n'
    '\n'
    'def main(argv):\n'
    '    if len(argv) != 2:\n'
    '        print("usage: failed_report.py RESULTS.jsonl", file=sys.stderr)\n'
    '        return 2\n'
    '    path = argv[1]\n'
    '    failed = 0\n'
    '    skipped = 0\n'
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
    '                skipped += 1\n'
    '                print(f"SKIPPED {path}: line {lineno}: {exc.msg}")\n'
    '                continue\n'
    '            if row.get("status") == "failed":\n'
    '                failed += 1\n'
    '    print(failed)\n'
    '    print(skipped)\n'
    '    if skipped:\n'
    '        print("LINE-REPORTS-INCOMPLETE")\n'
    '        return 1\n'
    '    print("LINE-REPORTS-COMPLETE")\n'
    '    return 0\n'
    '\n'
    '\n'
    'if __name__ == "__main__":\n'
    '    sys.exit(main(sys.argv))\n'
)

PLAN = [
    {"action": "write", "path": "tools/failed_report.py", "content": PLANNER_SCRIPT},
    {"action": "run", "command": "python tools/failed_report.py runs/results.jsonl"},
    {"action": "expect", "type": "stdout_is", "value": "3\n0\nLINE-REPORTS-COMPLETE"},
]
PLAN_TEXT = json.dumps(PLAN, indent=2)
RISK = ("Risk: the expected count 3 is only meaningful if every failed entry parses; if any "
        "line is SKIPPED the report marks evidence incomplete and exits 1, so step 3's expect "
        "check must not pass on partial evidence.")

USER_0001 = (
    "Repo state: runs/results.jsonl holds one JSON object per line with fields task_id and "
    "status (\"done\" | \"failed\"); the file may also contain blank lines and malformed "
    "lines. From a previous run I expect exactly 3 entries with status \"failed\", but that "
    "expectation must be verified, not assumed. Available tools: run_shell, read_file, "
    "write_file, web_search. Propose a minimal plan of exactly 3 steps as a JSON array; each "
    "step must have an \"action\" field: \"write\" (with \"path\" and \"content\"), \"run\" "
    "(with \"command\"), or \"expect\" (with \"type\" and \"value\"). Then state the single "
    "biggest risk in your plan in at most two sentences."
)
ASSISTANT_0001 = PLAN_TEXT + "\n" + RISK

# ---------------------------------------------------------------- 0002 recovery
BUGGY_IMPORTER = (
    'import sys\n'
    '\n'
    'def main(argv):\n'
    '    src, dst = argv[1], argv[2]\n'
    '    records = []\n'
    '    fh = open(src, "r", encoding="utf-8")\n'
    '    for lineno, line in enumerate(fh, start=1):\n'
    '        line = line.strip()\n'
    '        if not line:\n'
    '            continue\n'
    '        rec = json.loads(line)\n'
    '        records.append(rec)\n'
    '    fh.close()\n'
    '    with open(dst, "w", encoding="utf-8") as out:\n'
    '        for r in records:\n'
    '            out.write(json.dumps(r) + "\\n")\n'
    '\n'
    'if __name__ == "__main__":\n'
    '    main(sys.argv)\n'
)

USER_0002 = (
    BUGGY_IMPORTER +
    "\nThis importer must never crash on a malformed line. Required behavior: skip lines that "
    "fail to parse and also skip parseable lines that are not JSON objects; write one line "
    "\"line N: <json error message>\" per skipped line into the report file (N is the real file "
    "line number); still write every good record to the report path as canonical JSON; print "
    "exactly three lines \"total_records <n>\", \"kept <k>\", \"skipped <s>\" where total counts "
    "non-blank lines only; if the report path argument is missing, exit nonzero with a usage "
    "line instead of a traceback; if the input file cannot be opened, exit nonzero with an "
    "error line and do not create the report file. Keep the def main(argv) signature. Return "
    "the full corrected file only."
)

FIXED_IMPORTER = (
    'import json\n'
    'import sys\n'
    '\n'
    '\n'
    'def main(argv):\n'
    '    if len(argv) != 3:\n'
    '        print("usage: importer.py SRC.jsonl REPORT.txt", file=sys.stderr)\n'
    '        return 2\n'
    '    src, dst = argv[1], argv[2]\n'
    '    records = []\n'
    '    bad = []\n'
    '    total = 0\n'
    '    try:\n'
    '        fh = open(src, "r", encoding="utf-8")\n'
    '    except OSError as exc:\n'
    '        print(f"ERROR: cannot read {src}: {exc}", file=sys.stderr)\n'
    '        return 2\n'
    '    with fh:\n'
    '        for lineno, line in enumerate(fh, start=1):\n'
    '            line = line.strip()\n'
    '            if not line:\n'
    '                continue\n'
    '            total += 1\n'
    '            try:\n'
    '                rec = json.loads(line)\n'
    '            except json.JSONDecodeError as exc:\n'
    '                bad.append((lineno, str(exc.msg)))\n'
    '                continue\n'
    '            if not isinstance(rec, dict):\n'
    '                bad.append((lineno, "not a JSON object"))\n'
    '                continue\n'
    '            records.append(rec)\n'
    '    with open(dst, "w", encoding="utf-8") as out:\n'
    '        for lineno, msg in bad:\n'
    '            out.write(f"line {lineno}: {msg}\\n")\n'
    '        for r in records:\n'
    '            out.write(json.dumps(r) + "\\n")\n'
    '    print(f"total_records {total}")\n'
    '    print(f"kept {len(records)}")\n'
    '    print(f"skipped {len(bad)}")\n'
    '    return 0\n'
    '\n'
    '\n'
    'if __name__ == "__main__":\n'
    '    sys.exit(main(sys.argv))\n'
)

# ---------------------------------------------------------------- 0003 transcript
# Strict one-action-per-turn alternation: assistant emits exactly one JSON
# object, the next user turn carries that tool's real observation.
A1 = '{"tool": "run_tests", "args": {}}'
A3 = '{"tool": "read_file", "args": {"path": "calc.py"}}'
A5 = ('{"tool": "apply_patch", "args": {"patch": "REPLACE calc.py\\nREPLACE return a / b\\nWITH '
      "return 'inf' if b == 0 else a / b\\nENDREPLACE\\n\", \"out\": \"fix_div_zero.patch\"}}")
A7 = ('{"tool": "apply_patch", "args": {"patch": "PATCH calc.py\\n-    return a / b\\n+    '
      "return 'inf' if b == 0 else a / b\\nENDPATCH\\n\", \"out\": \"fix_div_zero.patch\"}}")
A9 = '{"tool": "run_tests", "args": {}}'
A11 = ('{"action": "stop", "result": {"stopped_cleanly": true, '
       '"patch_files": ["fix_div_zero.patch"]}}')

U1 = ("You are an agent with tools: run_tests, read_file, apply_patch. Apply a minimal fix so "
      "the failing test passes. Tool protocol: run_tests() takes no args; read_file takes "
      "{\"path\"}; apply_patch takes {\"patch\", \"out\"}; every call returns an observation "
      "string; you see no output until each tool returns. available_tools = [\"run_tests\", "
      "\"read_file\", \"apply_patch\"]. Rules: 1) start with run_tests before reading or "
      "patching; 2) emit one JSON object per action with keys \"tool\" and \"args\"; 3) after "
      "each observation emit the next action, or when the newest test result starts with "
      "\"1 passed\" emit {\"action\": \"stop\", \"result\": {\"stopped_cleanly\": true, "
      "\"patch_files\": [...]}} as the ONLY content of a turn; 4) never narrate or guess tool "
      "output. Begin.")
U12 = "Task complete. Review the transcript above."

SFT_0001 = {
    "id": "chowder_agent-0001", "track": "CHOWDER_AGENT",
    "domain": "agent_planning_tool_selection", "task_family": "planner_multi_step_jsonl",
    "source": "teacher_synthetic",
    "messages": [{"role": "user", "content": USER_0001},
                 {"role": "assistant", "content": ASSISTANT_0001}],
    "verification": {
        "method": "executable_test",
        "expected": ("Plan is a 3-step JSON array (write/run/expect) using only allowed tools; "
                     "the embedded script, executed on a clean 4-line fixture, prints \"3\", "
                     "\"0\", \"LINE-REPORTS-COMPLETE\" and exits 0; on a fixture with 3 failed "
                     "entries plus one malformed line at file line 4 plus one blank line, it "
                     "prints a SKIPPED line citing line 4 (enumerate counter, not "
                     "JSONDecodeError.lineno, which is always 1 for single-line parses), then "
                     "\"3\", \"1\", \"LINE-REPORTS-INCOMPLETE\" and exits 1, so the expect "
                     "step fails on incomplete evidence"),
        "status": "passed"},
    "failure_mode": ("Plans that grep-then-parse and silently ignore malformed lines, or that "
                     "report JSONDecodeError.lineno as the file line number (it is always 1 for "
                     "single-line parses), or that present a count from partial evidence as "
                     "verified"),
    "difficulty": "medium", "estimated_tokens": 1150,
}

SFT_0002 = {
    "id": "chowder_agent-0002", "track": "CHOWDER_AGENT",
    "domain": "python_debugging_error_recovery", "task_family": "jsonl_importer_recovery",
    "source": "teacher_synthetic",
    "messages": [{"role": "user", "content": USER_0002},
                 {"role": "assistant", "content": FIXED_IMPORTER.rstrip("\n")}],
    "verification": {
        "method": "executable_test",
        "expected": ("Fixture A (3 non-blank lines, middle line truncated after a comma): "
                     "total_records 3, kept 2, skipped 1; report contains \"line 2: Expecting "
                     "property name enclosed in double quotes\" then both good records as "
                     "canonical JSON. Fixture B (11 non-blank lines: 2 invalid JSON at lines 4 "
                     "and 9, one valid array line at 7): total_records 11, kept 8, skipped 3; "
                     "report cites lines 4, 7, 9. Missing report argument: exit 2 with a usage "
                     "line, no traceback. Unreadable input: exit 2, stderr starts with \"ERROR: "
                     "cannot read\", report file not created"),
        "status": "passed"},
    "failure_mode": ("json.loads kept outside try (crash on first malformed line), json module "
                     "never imported, blank lines counted in total, skipped-line reports citing "
                     "JSONDecodeError.lineno instead of the file line number, or a traceback "
                     "instead of the required usage/error lines"),
    "difficulty": "medium", "estimated_tokens": 1300,
}

SFT_0003 = {
    "id": "chowder_agent-0003", "track": "CHOWDER_AGENT",
    "domain": "agent_orchestration_stopping", "task_family": "agent_toolloop_transcript",
    "source": "teacher_synthetic",
    "messages": [
        {"role": "user", "content": U1},
        {"role": "assistant", "content": A1},
        {"role": "user", "content": TOOL_BLOCK_U2},
        {"role": "assistant", "content": A3},
        {"role": "user", "content": TOOL_BLOCK_U4},
        {"role": "assistant", "content": A5},
        {"role": "user", "content": TOOL_BLOCK_U6},
        {"role": "assistant", "content": A7},
        {"role": "user", "content": TOOL_BLOCK_U8},
        {"role": "assistant", "content": A9},
        {"role": "user", "content": TOOL_BLOCK_U10},
        {"role": "assistant", "content": A11},
        {"role": "user", "content": U12},
    ],
    "verification": {
        "method": "executable_test",
        "expected": ("Replay in a sandbox: message roles alternate user/assistant; assistant "
                     "actions in order run_tests, read_file, apply_patch, apply_patch, "
                     "run_tests, stop with exactly one JSON object per assistant turn; the "
                     "first apply_patch is rejected by the environment (no 'PATCH calc.py' "
                     "header) and the second is accepted; the accepted patch, applied to the "
                     "sandboxed calc.py, makes the sandboxed test pass (\"1 passed\"); the "
                     "stop object is emitted only after observing \"1 passed\", with "
                     "stopped_cleanly true and patch_files [\"fix_div_zero.patch\"]; no "
                     "assistant text contains environment output strings such as \"PATCH OK\" "
                     "or \"1 passed\""),
        "status": "passed"},
    "failure_mode": ("Emitting the stop action in the same turn as the final run_tests call "
                     "before the result is observed, inventing patch output, or using a patch "
                     "format the environment rejects"),
    "difficulty": "hard", "estimated_tokens": 1450,
}

# ---------------------------------------------------------------- pref pair
PREF_INPUT = (
    "Available tools: run_shell, read_file, write_file. In this environment run_shell returns "
    "only a success/fail flag and never echoes stdout. A test runner on this box takes 4-7 "
    "minutes per run and writes its full detail to a file you name. Plan exactly 4 steps as a "
    "JSON array; each step must have an \"action\": \"write\" (with \"path\" and \"content\"), "
    "\"run\" (with \"command\"), or \"expect\" (with \"type\" and \"value\"). Do not run "
    "anything yet."
)
CHOSEN = (
    '[\n'
    '  {"action": "write", "path": "tools/run_suite.py", "content": "# tee the suite\'s output '
    'to runs/last_results.log because run_shell shows no stdout"},\n'
    '  {"action": "run", "command": "python tools/run_suite.py runs/last_results.log"},\n'
    '  {"action": "write", "path": "tools/parse_suite.py", "content": "# print failure lines '
    'and counts from runs/last_results.log; print NO RESULTS if the log is missing or empty"},\n'
    '  {"action": "run", "command": "python tools/parse_suite.py runs/last_results.log"}\n'
    ']\n'
    'Rationale: run_shell never echoes stdout here, so detail must be captured to a file by the '
    'runner and parsed in a separate step; the suite is invoked exactly once and the parser '
    'prints NO RESULTS on a missing or empty log so a crashed run cannot be mistaken for a '
    'pass.'
)
REJECTED = (
    '[\n'
    '  {"action": "run", "command": "pytest -q"},\n'
    '  {"action": "expect", "type": "stdout_contains", "value": "failed"},\n'
    '  {"action": "run", "command": "pytest -q"},\n'
    '  {"action": "expect", "type": "stdout_contains", "value": "passed"}\n'
    ']\n'
    'Rationale: run pytest once to see failures, then run it again to confirm everything '
    'passes.'
)
PREF = {
    "id": "chowder_agent-0004-pref", "track": "CHOWDER_AGENT",
    "domain": "agent_planning_tool_selection", "task_family": "planner_environment_aware",
    "source": "teacher_synthetic", "type": "preference_pair",
    "messages": [{"role": "user", "content": PREF_INPUT}],
    "input": {"messages": [{"role": "user", "content": PREF_INPUT}]},
    "chosen": {"messages": [{"role": "assistant", "content": CHOSEN}]},
    "rejected": {"messages": [{"role": "assistant", "content": REJECTED}]},
    "preference_reason": ("The rejected plan asserts on run_shell stdout that the task states is "
                          "never echoed, so both expect steps can never match; it also runs the "
                          "4-7 minute suite twice. The chosen plan captures detail to a file, "
                          "runs the suite once, and guards against an empty log."),
    "evidence": ("The constraint 'run_shell returns only a success/fail flag and never echoes "
                 "stdout' appears verbatim in the shared input; rejected steps 2 and 4 use "
                 "stdout_contains, which cannot match under that constraint, and rejected steps "
                 "1 and 3 each invoke the full suite while the chosen plan invokes it once"),
    "verification": {"method": "review",
                     "expected": "No chosen step references run_shell stdout; the suite command "
                                 "appears exactly once across the chosen plan; the parser step "
                                 "reads the logged file, not the tool flag",
                     "status": "not_run"},
    "failure_mode": ("Planning around tool outputs the environment does not provide, and "
                     "re-running an expensive suite to confirm results instead of capturing "
                     "output once"),
    "difficulty": "hard", "estimated_tokens": 800,
}

RECORDS = [SFT_0001, SFT_0002, SFT_0003, PREF]

with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    for rec in RECORDS:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("wrote", OUT)
for rec in RECORDS:
    print(rec["id"], "messages:", len(rec.get("messages", [])))
