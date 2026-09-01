"""Chowder: autonomous evidence-gated post-training orchestration."""

from .cycle import ExperimentCycleRunner
from .engine import EvolutionEngine
from .models import Experiment, ExperimentResult, Goal, MetricTarget

__all__ = [
    "ExperimentCycleRunner",
    "EvolutionEngine",
    "Goal",
    "MetricTarget",
    "Experiment",
    "ExperimentResult",
]

__version__ = "0.3.0"
