"""Actual isolated script-entry import regression for Q4 speed sealing.

The registered Windows wrapper is launched by absolute *file* path, not
``python -m``. Its evaluator child has PYTHONPATH, but the wrapper process
does not. A completed 200-row run must never crash with ModuleNotFoundError
while checking the original source-bound same-session receipt.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.name != "nt", reason="registered speed certification is Windows-only")
def test_absolute_script_entry_resolves_own_original_eval_without_pythonpath(tmp_path):
    wrapper = Path(__file__).resolve().parents[1] / "chowder_batch" / "reap_serve.py"
    source = """
import runpy
import sys
from pathlib import Path
mod = runpy.run_path(sys.argv[1], run_name='__script_entry_import_probe__')
try:
    mod['_certify_q4_speed_session'](
        Path(sys.argv[2]),
        eval_args=['--suites', 'gsm8k', '--limit', '200'],
        receipt={'measured_experts_offloaded': 0},
        session_id='f' * 32, port=8790, server_pid=42,
    )
except ValueError as exc:
    assert str(exc).startswith('REAP speed session lacks a regular'), str(exc)
else:
    raise AssertionError('missing native rows unexpectedly certified')
"""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, "-I", "-B", "-c", source, str(wrapper), str(tmp_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
