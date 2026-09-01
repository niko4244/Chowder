"""Chowder: autonomous evidence-gated post-training orchestration."""

from .cycle import ExperimentCycleRunner
from .engine import EvolutionEngine
from .models import Goal, MetricTarget, Experiment, ExperimentResult

__all__ = [
    "ExperimentCycleRunner",
    "EvolutionEngine",
    "Goal",
    "MetricTarget",
    "Experiment",
    "ExperimentResult",
]

__version__ = "0.2.0"
