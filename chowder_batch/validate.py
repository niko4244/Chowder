"""Validates chowder_batch/chowder_agent_batch_001.jsonl.

Gates: structure/schema, duplication, answer leakage, and behavioral
execution of the embedded scripts in records 0001 and 0002, plus
replay checks on record 0003 and the preference-pair invariants.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH = os.path.join(HERE, "chowder_agent_batch_001.jsonl")

REQUIRED_SFT = {"id", "track", "domain", "task_family", "source", "messages",
                "verification", "failure_mode", "difficulty", "estimated_tokens"}
REQUIRED_PREF = REQUIRED_SFT | {"type", "input", "chosen", "rejected",
                                "preference_reason", "evidence"}
DIFFICULTIES = {"medium", "hard"}
LEAK_STRINGS = ("1 passed", "PATCH OK")  # observed-output strings; plan-time expect values are hypotheses, not leakage

records = []
with open(BATCH, encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        assert line, "blank line %d in batch" % i
        records.append(json.loads(line))
assert len(records) == 4, len(records)

sft = [r for r in records if r.get("type") != "preference_pair"]
pref = [r for r in records if r.get("type") == "preference_pair"]
assert len(sft) == 3 and len(pref) == 1

# ---- structural gate --------------------------------------------------------
ids = set()
for r in records:
    assert r["id"] not in ids, "duplicate id " + r["id"]
    ids.add(r["id"])
    required = REQUIRED_PREF if r.get("type") == "preference_pair" else REQUIRED_SFT
    missing = required - set(r)
    assert not missing, (r["id"], missing)
    assert r["track"] == "CHOWDER_AGENT"
    assert r["source"] in ("teacher_synthetic", "user_failure")
    assert r["difficulty"] in DIFFICULTIES
    assert isinstance(r["estimated_tokens"], int) and r["estimated_tokens"] > 0
    v = r["verification"]
    assert v["method"] in ("executable_test", "review")
    assert v["status"] in ("not_run", "passed", "failed")
    assert isinstance(v["expected"], str) and v["expected"]
    assert isinstance(r["failure_mode"], str) and r["failure_mode"]
    if r.get("type") != "preference_pair":
        for m in r["messages"]:
            assert set(m) == {"role", "content"}
            assert m["role"] in ("user", "assistant")
            assert isinstance(m["content"], str) and m["content"].strip()
        roles = [m["role"] for m in r["messages"]]
        assert roles[0] == "user" and roles[-1] in ("assistant", "user")
        for a, b in zip(roles, roles[1:]):
            assert a != b, "role repetition in " + r["id"]

families = [r["task_family"] for r in sft]
assert len(set(families)) == 3, families

# ---- leakage gate (no fabricated observations inside assistant targets) ------
for r in sft:
    for m in r["messages"]:
        if m["role"] == "assistant":
            for s in LEAK_STRINGS:
                assert s not in m["content"], (r["id"], s)

# ---- token budget gate -------------------------------------------------------
for r in records:
    size = len(json.dumps(r)) / 4.0
    assert size <= 2048, (r["id"], size)
    print("  token estimate %-24s ~%d" % (r["id"], size))

def run_script(source, args, cwd=None):
    d = tempfile.mkdtemp(prefix="chowder_val_")
    path = os.path.join(d, "script.py")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(source)
    return subprocess.run([sys.executable, path] + args,
                          capture_output=True, text=True, cwd=cwd or d)

def write_fixture(name, content):
    d = tempfile.mkdtemp(prefix="chowder_fix_")
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    return p

# ---- behavioral gate: 0001 embedded planner ----------------------------------
plan = json.loads(sft[0]["messages"][1]["content"].split("\nRisk:")[0])
assert [s.get("action") for s in plan] == ["write", "run", "expect"], plan
planner_src = plan[0]["content"]

clean = write_fixture("results.jsonl",
                      '{"task_id": "T1", "status": "done"}\n'
                      '{"task_id": "T2", "status": "failed"}\n'
                      '{"task_id": "T3", "status": "failed"}\n'
                      '{"task_id": "T4", "status": "failed"}\n')
r = run_script(planner_src, [clean])
assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
assert r.stdout == "3\n0\nLINE-REPORTS-COMPLETE\n", repr(r.stdout)

dirty = write_fixture("results.jsonl",
                      '{"task_id": "T1", "status": "done"}\n'
                      '{"task_id": "T2", "status": "failed"}\n'
                      '{"task_id": "T3", "status": "failed"}\n'
                      '{task_id: 3, status: "failed"}\n'
                      '\n'
                      '{"task_id": "T5", "status": "failed"}\n')
r = run_script(planner_src, [dirty])
assert r.returncode == 1, (r.returncode, r.stdout)
lines = r.stdout.splitlines()
assert {"3", "1", "LINE-REPORTS-INCOMPLETE"} <= set(lines), lines
skipped_lines = [ln for ln in lines if ln.startswith("SKIPPED")]
assert len(skipped_lines) == 1 and ": line 4: " in skipped_lines[0], skipped_lines
assert ": line 1: " not in r.stdout, "lineno trap regressed"

# ---- behavioral gate: 0002 embedded importer ---------------------------------
importer_src = sft[1]["messages"][1]["content"] + "\n"

fa = write_fixture("a.jsonl", '{"id": 1}\n{"id": 2,\n{"id": 3}\n')
rep = os.path.join(tempfile.mkdtemp(prefix="chowder_rep_"), "rep.txt")
r = run_script(importer_src, [fa, rep])
assert r.returncode == 0 and "Traceback" not in r.stderr, (r.returncode, r.stderr)
assert r.stdout == "total_records 3\nkept 2\nskipped 1\n", repr(r.stdout)
with open(rep, encoding="utf-8") as f:
    rep_lines = f.read().splitlines()
assert rep_lines[0] == "line 2: Expecting property name enclosed in double quotes", rep_lines
assert json.loads(rep_lines[1]) == {"id": 1} and json.loads(rep_lines[2]) == {"id": 3}

fb = write_fixture("b.jsonl",
                   '{"id": 1}\n{"id": 2}\n\n{"id" 3}\n{"id": 5}\n{"id": 6}\n'
                   '[1, 2, 3]\n{"id": 8}\n{oops}\n{"id": 9}\n{"id": 10}\n{"id": 11}\n')
rep2 = os.path.join(tempfile.mkdtemp(prefix="chowder_rep_"), "rep.txt")
r = run_script(importer_src, [fb, rep2])
assert r.returncode == 0, r.stderr
assert r.stdout == "total_records 11\nkept 8\nskipped 3\n", repr(r.stdout)
with open(rep2, encoding="utf-8") as f:
    bad_lines = [ln for ln in f.read().splitlines() if ln.startswith("line ")]
assert [ln.split(":")[0] for ln in bad_lines] == ["line 4", "line 7", "line 9"], bad_lines

r = run_script(importer_src, [fa])             # missing report arg
assert r.returncode == 2 and "usage" in r.stderr and "Traceback" not in r.stderr

missing = os.path.join(tempfile.mkdtemp(prefix="chowder_no_"), "gone.jsonl")
outdir = tempfile.mkdtemp(prefix="chowder_out_")
rep3 = os.path.join(outdir, "rep.txt")
r = run_script(importer_src, [missing, rep3])
assert r.returncode == 2 and r.stderr.startswith("ERROR: cannot read")
assert not os.path.exists(rep3), "report created despite unreadable input"

# ---- replay gate: 0003 transcript --------------------------------------------
t = sft[2]["messages"]
assert [m["role"] for m in t] == ["user", "assistant", "user", "assistant", "user",
                                  "assistant", "user", "assistant", "user",
                                  "assistant", "user", "assistant", "user"]
assert all(isinstance(json.loads(m["content"]), dict)
           and ("tool" in json.loads(m["content"]) or "action" in json.loads(m["content"]))
           for m in t if m["role"] == "assistant"), "each assistant turn is one action object"
def tool_of(content):
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return None
    return obj.get("tool") or ("stop" if obj.get("action") == "stop" else "???")

actions = [tool_of(m["content"]) for m in t if m["role"] == "assistant"]
assert actions == ["run_tests", "read_file", "apply_patch", "apply_patch",
                   "run_tests", "stop"], actions
assert t[6]["content"].startswith('TOOL_RESULTS_BEGIN apply_patch -> "ERROR')
assert t[8]["content"] == 'TOOL_RESULTS_BEGIN apply_patch -> "PATCH OK" TOOL_RESULTS_END'
stop = json.loads(t[11]["content"])
assert stop == {"action": "stop", "result": {"stopped_cleanly": True,
                                             "patch_files": ["fix_div_zero.patch"]}}
for m in t:
    if m["role"] == "assistant":
        for s in ("1 passed", "PATCH OK", "ERROR"):
            assert s not in m["content"], (s, m["content"][:60])

# ---- preference pair gate ----------------------------------------------------
p = pref[0]
inp = json.dumps(p["input"], sort_keys=True)
assert p["input"]["messages"] == p["chosen"]["messages"][:0] + p["input"]["messages"]
assert json.dumps(p["chosen"]["messages"][0]["content"]) != \
       json.dumps(p["rejected"]["messages"][0]["content"])
ch = p["chosen"]["messages"][0]["content"]
rj = p["rejected"]["messages"][0]["content"]
assert "stdout_contains" not in ch
assert rj.count('"command": "pytest -q"') == 2 and rj.count("stdout_contains") == 2
assert ch.count("tools/run_suite.py") >= 1 and "pytest" not in ch
assert p["preference_reason"] and p["evidence"]
assert p["verification"]["method"] == "review" and p["verification"]["status"] == "not_run"

print("validate: ALL GATES PASS")
