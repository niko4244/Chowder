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
