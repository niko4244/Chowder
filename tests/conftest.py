"""Shared test configuration.

The teacher-free pilot's pipeline lives as flat scripts in
``experiments/teacher_free_distill`` (``prepare.py``, ``ot3_subset.py``,
``replay_smith.py`` …) and they import each other by bare name, exactly as they
do when run from that directory as CLIs. Tests load them by file location, so
the directory is appended to ``sys.path`` — appended, never prepended, so a
module in there can never shadow a stdlib or package name used elsewhere.
"""
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments" / "teacher_free_distill"

if str(EXPERIMENTS) not in sys.path:
    sys.path.append(str(EXPERIMENTS))
