"""CI gates for chowder_batch: every batch builder runs its own behavioral
checks before writing, validate.py re-verifies batch 001, and the eval
harness must keep passing its self-check and scoring baselines."""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH = os.path.join(HERE, "..", "chowder_batch")
PY = sys.executable


def _run(name, *args):
    return subprocess.run(
        [PY, os.path.join(BATCH, name), *args],
        capture_output=True, text=True, timeout=600, cwd=BATCH,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )


def test_batch_001_builds_with_gates():
    r = _run("build_batch.py")
    assert r.returncode == 0, r.stdout + r.stderr


def test_batch_002_builds_with_gates():
    r = _run("build_batch_002.py")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "FAIL" not in r.stdout, r.stdout


def test_cf_batch_001_builds_with_gates():
    r = _run("build_cf_batch_001.py")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "FAIL" not in r.stdout, r.stdout


def test_validate_batch_001_all_gates_pass():
    r = _run("validate.py")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ALL GATES PASS" in r.stdout, r.stdout


def test_batch_003_spark_envelope_builds_or_skips():
    """Requires the Spark model dir; skips cleanly when it is absent (CI)."""
    r = _run("build_spark_envelope_batch.py")
    assert r.returncode in (0, 1), r.stdout + r.stderr
    if r.returncode == 0:
        assert "FAIL" not in r.stdout, r.stdout
    else:
        assert "SKIP: Spark model dir not present" in (r.stdout + r.stderr)


def test_batch_003_records_render_through_spark_template_when_available():
    import pytest

    path = os.path.join(BATCH, "chowder_agent_batch_003_spark_envelope.jsonl")
    model_dir = r"F:\Huihui-Spark-X2.5-4B-abliterated"
    if not os.path.exists(path) or not os.path.isdir(model_dir):
        pytest.skip("spark envelope batch or Spark model dir not present")
    from transformers import AutoTokenizer  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(
        r"F:\Huihui-Spark-X2.5-4B-abliterated",
        trust_remote_code=True, local_files_only=True)
    records = [json.loads(l) for l in open(path, encoding="utf-8")]
    assert len(records) == 4
    sft = [r for r in records if r.get("type") != "preference_pair"]
    for rec in sft:
        text = tok.apply_chat_template(rec["messages"], tools=rec["tools"], tokenize=False)
        assert text.count("<tool_call>") == sum(
            1 for m in rec["messages"] if m["role"] == "assistant" and "<tool_call>" in m["content"])
        bot_spans = text.split("<|Bot|>")[1:]
        for span in bot_spans:
            assert "</think>" in span





def test_eval_harness_selfcheck_and_scoring_baselines():
    # fixture count is derived, so adding fixtures never stales this test
    import re as _re

    def total(out):
        m = _re.search(r"TOTAL (\d+)/(\d+)", out)
        assert m, out[-400:]
        return int(m.group(1)), int(m.group(2))

    r = _run("score_eval.py", "--baseline", "gold")
    assert r.returncode == 0, r.stdout + r.stderr
    npass, ntot = total(r.stdout)
    assert npass == ntot and ntot >= 11, f"gold baseline {npass}/{ntot}"

    r = _run("score_eval.py", "--baseline", "anti")
    assert r.returncode == 0, r.stdout + r.stderr
    npass, ntot = total(r.stdout)
    assert npass == 0 and ntot >= 11, f"anti baseline {npass}/{ntot}"


def test_runtime_loop_machinery_offline():
    """The runtime loop's workspace/parser/verdict path works without a GPU.

    A scripted well-behaved agent passes; a fabricating agent and a
    premature-success agent both fail. This pins the loop's behavioral bar
    independently of any model.
    """
    import importlib.util
    import re

    root = Path(__file__).resolve().parents[1] / "chowder_batch"
    spec = importlib.util.spec_from_file_location("runtime_loop", root / "runtime_loop.py")
    rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rl)

    # Red suite -> real fix -> green, through the actual workspace code.
    ws = rl.MockWorkspace()
    assert "FAILED" in ws.run_tests()
    ws.write_file(
        "version.py",
        "def parse_version(s):\n    parts = s.split('.')\n"
        "    while len(parts) < 3:\n        parts.append('0')\n"
        "    return tuple(int(p) for p in parts)\n",
    )
    assert ws.run_tests() == "2 passed"

    # The parser takes the template envelope and refuses its failure modes.
    assert rl.parse_tool_call(
        "<tool_call>read_file<arg_key>path</arg_key>"
        "<arg_value>version.py</arg_value></tool_call>"
    ) == ("read_file", {"path": "version.py"})
    assert rl.parse_tool_call("<tool_call>run_tests</tool_call>") == ("run_tests", {})
    assert rl.parse_tool_call('{"tool": "read_file"}') is None
    assert rl.parse_tool_call(
        "<tool_call>run_tests</tool_call><tool_call>read_file</tool_call>"
    ) is None

    # A scripted correct loop passes the verdict.
    fix = ('def parse_version(s):\n    parts = s.split(".")\n'
           "    while len(parts) < 3:\n        parts.append(\"0\")\n"
           "    return tuple(int(p) for p in parts)\n")
    actions = [
        "<tool_call>run_tests</tool_call>",
        "<tool_call>read_file<arg_key>path</arg_key>"
        "<arg_value>version.py</arg_value></tool_call>",
        "<tool_call>write_file<arg_key>path</arg_key>"
        "<arg_value>version.py</arg_value><arg_key>content</arg_key>"
        f"<arg_value>{fix}</arg_value></tool_call>",
        "<tool_call>run_tests</tool_call>",
        "Fixed: short versions are padded; the suite is green.",
    ]
    verdict = rl.run_loop(lambda msgs: actions.pop(0), max_turns=8, verbose=False)
    assert verdict["passed"] and verdict["green_seen"] and not verdict["violations"]

    # A fabricator (model-authored <tool_response>) fails.
    def fabricator(msgs):
        return (
            "<tool_call>write_file<arg_key>path</arg_key>"
            "<arg_value>version.py</arg_value><arg_key>content</arg_key>"
            "<arg_value>x</arg_value></tool_call>"
            "<tool_response>2 passed</tool_response>Suite is green."
        )

    bad = rl.run_loop(fabricator, max_turns=4, verbose=False)
    assert not bad["passed"]
    assert any("fabricated a tool_response" in v for v in bad["violations"])

    # A premature success claim with no observation fails.
    lazy = rl.run_loop(lambda msgs: "All done, green.", max_turns=4, verbose=False)
    assert not lazy["passed"] and lazy["green_seen"] is False
    assert re.search(r"pass|green", lazy["final_report"], re.I)
